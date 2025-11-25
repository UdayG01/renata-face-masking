import os
import glob
import time
import argparse
from pathlib import Path
import subprocess
import sys

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = ['videos', 'results', 'models']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Directory '{directory}' ready")

def get_video_files(videos_dir):
    """Get all video files from the videos directory."""
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv']
    video_files = []
    
    for extension in video_extensions:
        pattern = os.path.join(videos_dir, extension)
        video_files.extend(glob.glob(pattern))
    
    # Sort files naturally (input1.mp4, input2.mp4, etc.)
    video_files.sort()
    return video_files

def process_single_video(input_path, output_path, model_path, confidence, mask_type):
    """Process a single video file."""
    print(f"\n🎬 Processing: {os.path.basename(input_path)}")
    print(f"📤 Output: {os.path.basename(output_path)}")
    
    # Build command
    cmd = [
        sys.executable, 'src/face_masking_v2.py',
        '--input', input_path,
        '--output', output_path,
        '--model', model_path,
        '--confidence', str(confidence),
        '--mask-type', mask_type
    ]
    
    # Record start time
    start_time = time.time()
    
    try:
        # Run the head detection script
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Check if output file was created
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            print(f"✅ Success! Time: {processing_time:.1f}s, Size: {file_size:.1f}MB")
            return True, processing_time
        else:
            print(f"❌ Error: Output file not created")
            return False, processing_time
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error processing video:")
        print(f"   Command: {' '.join(cmd)}")
        print(f"   Error: {e.stderr}")
        return False, 0
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False, 0

def generate_output_filename(input_path, results_dir, naming_style='output'):
    """Generate output filename based on input filename."""
    input_name = Path(input_path).stem
    
    if naming_style == 'output':
        # input1.mp4 -> output1.mp4
        if input_name.startswith('input'):
            number = input_name.replace('input', '')
            output_name = f"output{number}.mp4"
        else:
            output_name = f"output_{input_name}.mp4"
    elif naming_style == 'masked':
        # input1.mp4 -> masked_input1.mp4
        output_name = f"masked_{input_name}.mp4"
    elif naming_style == 'same':
        # input1.mp4 -> input1.mp4 (in results directory)
        output_name = f"{input_name}.mp4"
    else:
        output_name = f"processed_{input_name}.mp4"
    
    return os.path.join(results_dir, output_name)

def print_summary(total_videos, successful, failed, total_time):
    """Print processing summary."""
    print("\n" + "="*50)
    print("📊 BATCH PROCESSING SUMMARY")
    print("="*50)
    print(f"📁 Total videos processed: {total_videos}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    if successful > 0:
        print(f"⚡ Average time per video: {total_time/successful:.1f} seconds")
    print(f"🎯 Success rate: {(successful/total_videos*100):.1f}%")
    print("="*50)

def main():
    parser = argparse.ArgumentParser(description='Batch Process Videos for Head Detection')
    parser.add_argument('--videos-dir', '-v', type=str, default='videos',
                       help='Directory containing input videos')
    parser.add_argument('--results-dir', '-r', type=str, default='results',
                       help='Directory to save processed videos')
    parser.add_argument('--model', '-m', type=str, default='models/medium.pt',
                       help='Path to YOLO head detection model')
    parser.add_argument('--confidence', '-c', type=float, default=0.3,
                       help='Detection confidence threshold')
    parser.add_argument('--mask-type', '-t', type=str, default='pixelate',
                       choices=['pixelate', 'blur', 'solid', 'black'],
                       help='Type of mask to apply')
    parser.add_argument('--naming', '-n', type=str, default='output',
                       choices=['output', 'masked', 'same', 'processed'],
                       help='Output filename style')
    parser.add_argument('--skip-existing', '-s', action='store_true',
                       help='Skip processing if output file already exists')
    
    args = parser.parse_args()
    
    print("🚀 BATCH VIDEO HEAD DETECTION PROCESSOR")
    print("="*50)
    
    # Create directories
    create_directories()
    Path(args.results_dir).mkdir(exist_ok=True)
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Error: Model file not found at {args.model}")
        return
    
    # Check if main script exists
    if not os.path.exists('src/face_masking_v2.py'):
        print("❌ Error: Head detection script not found at src/face_masking_v2.py")
        return
    
    # Get list of video files
    video_files = get_video_files(args.videos_dir)
    
    if not video_files:
        print(f"❌ No video files found in {args.videos_dir}")
        print("   Supported formats: mp4, avi, mov, mkv, flv, wmv")
        return
    
    print(f"📁 Found {len(video_files)} video files in {args.videos_dir}")
    print(f"💾 Results will be saved to {args.results_dir}")
    print(f"🎯 Model: {args.model}")
    print(f"🎨 Mask type: {args.mask_type}")
    print(f"📊 Confidence: {args.confidence}")
    
    # Process each video
    successful = 0
    failed = 0
    total_start_time = time.time()
    
    for i, input_path in enumerate(video_files, 1):
        print(f"\n📹 [{i}/{len(video_files)}] Processing video:")
        
        # Generate output filename
        output_path = generate_output_filename(input_path, args.results_dir, args.naming)
        
        # Skip if output already exists and skip_existing is enabled
        if args.skip_existing and os.path.exists(output_path):
            print(f"⏭️  Skipping (output exists): {os.path.basename(output_path)}")
            continue
        
        # Process the video
        success, processing_time = process_single_video(
            input_path, output_path, args.model, args.confidence, args.mask_type
        )
        
        if success:
            successful += 1
        else:
            failed += 1
    
    # Calculate total time
    total_time = time.time() - total_start_time
    
    # Print summary
    print_summary(len(video_files), successful, failed, total_time)
    
    # List output files
    if successful > 0:
        print(f"\n📂 Output files in {args.results_dir}:")
        output_files = glob.glob(os.path.join(args.results_dir, "*.mp4"))
        for output_file in sorted(output_files):
            file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
            print(f"   📄 {os.path.basename(output_file)} ({file_size:.1f} MB)")

if __name__ == "__main__":
    main()