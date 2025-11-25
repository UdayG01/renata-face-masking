import cv2
import numpy as np
import argparse
import time
import torch
from ultralytics import YOLO
from typing import List, Tuple

class YOLOHeadDetector:
    """
    Simple and effective head detection using YOLO models.
    Much simpler than MediaPipe approaches with better accuracy.
    """
    
    def __init__(self, 
                 model_path: str = 'models/nano.pt',
                 confidence_threshold: float = 0.3,
                 mask_type: str = 'pixelate',
                 mask_intensity: float = 0.95):
        """
        Initialize YOLO head detector for head-specific models.
        
        Args:
            model_path: Path to YOLO head detection model
            confidence_threshold: Minimum confidence for detections
            mask_type: Type of mask ('pixelate', 'blur', 'solid', 'black')
            mask_intensity: Intensity of masking effect
        """
        self.confidence_threshold = confidence_threshold
        self.mask_type = mask_type
        self.mask_intensity = mask_intensity
        
        # Load YOLO head detection model
        print(f"Loading YOLO head detection model: {model_path}")
        try:
            self.model = YOLO(model_path)
            
            # Force model to use GPU if available
            if torch.cuda.is_available():
                self.model.to('cuda')
                print(f"✅ YOLO model loaded successfully on GPU: {torch.cuda.get_device_name(0)}")
                print(f"🎯 Model device: {self.model.device}")
            else:
                print("⚠️ CUDA not available, using CPU")
                print(f"🎯 Model device: {self.model.device}")
                
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            print("Make sure the model file exists at: {model_path}")
            raise
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        print(f"🎯 Detection confidence threshold: {confidence_threshold}")
        print(f"🎨 Mask type: {mask_type}")
    
    def detect_model_type(self) -> str:
        """Not needed - we know it's a head-specific model."""
        return 'head'
    
    def detect_heads(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect heads using the head-specific YOLO model.
        
        Returns:
            List of (x, y, width, height, confidence) tuples for detected heads
        """
        detections = []
        
        try:
            # Run YOLO inference with detect mode for head-specific model
            results = self.model.predict(frame, conf=self.confidence_threshold, verbose=False, mode='detect')
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates and confidence
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        
                        if confidence >= self.confidence_threshold:
                            # Head-specific model: use detection directly
                            head_x = int(x1)
                            head_y = int(y1)
                            head_width = int(x2 - x1)
                            head_height = int(y2 - y1)
                            
                            # Add padding for better coverage
                            padding_factor = 0.15  # 15% padding
                            h_padding = int(head_width * padding_factor)
                            v_padding = int(head_height * padding_factor)
                            
                            # Apply padding
                            head_x = max(0, head_x - h_padding)
                            head_y = max(0, head_y - v_padding)
                            head_width = min(frame.shape[1] - head_x, head_width + 2 * h_padding)
                            head_height = min(frame.shape[0] - head_y, head_height + 2 * v_padding)
                            
                            # Minimum size check
                            if head_width > 10 and head_height > 10:
                                detections.append((head_x, head_y, head_width, head_height, confidence))
                                
        except Exception as e:
            print(f"Head detection error: {e}")
        
        return detections
    
    def apply_pixelate_mask(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Apply pixelation mask to specified region."""
        if w <= 0 or h <= 0:
            return image
            
        # Extract region
        region = image[y:y+h, x:x+w]
        
        # Calculate pixelation size
        pixel_size = max(3, int(8 + (self.mask_intensity * 25)))
        
        # Resize down and up for pixelation effect
        small_w = max(1, w // pixel_size)
        small_h = max(1, h // pixel_size)
        
        small_region = cv2.resize(region, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        pixelated_region = cv2.resize(small_region, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Replace original region
        image[y:y+h, x:x+w] = pixelated_region
        return image
    
    def apply_blur_mask(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Apply blur mask to specified region."""
        if w <= 0 or h <= 0:
            return image
            
        # Extract region
        region = image[y:y+h, x:x+w]
        
        # Calculate blur intensity
        blur_intensity = int(15 + (self.mask_intensity * 40))
        if blur_intensity % 2 == 0:
            blur_intensity += 1
            
        # Apply Gaussian blur
        blurred_region = cv2.GaussianBlur(region, (blur_intensity, blur_intensity), 0)
        
        # Replace original region
        image[y:y+h, x:x+w] = blurred_region
        return image
    
    def apply_solid_mask(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Apply solid color mask to specified region."""
        color = (128, 128, 128)  # Gray color
        cv2.rectangle(image, (x, y), (x+w, y+h), color, -1)
        return image
    
    def apply_black_mask(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Apply black mask to specified region."""
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 0), -1)
        return image
    
    def apply_mask(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Apply the configured mask type to the specified region."""
        if self.mask_type == 'pixelate':
            return self.apply_pixelate_mask(image, x, y, w, h)
        elif self.mask_type == 'blur':
            return self.apply_blur_mask(image, x, y, w, h)
        elif self.mask_type == 'solid':
            return self.apply_solid_mask(image, x, y, w, h)
        elif self.mask_type == 'black':
            return self.apply_black_mask(image, x, y, w, h)
        return image
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame: detect heads and apply masks.
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with head regions masked
        """
        # Detect heads
        detections = self.detect_heads(frame)
        
        # Apply masks to detected heads
        for x, y, w, h, confidence in detections:
            frame = self.apply_mask(frame, x, y, w, h)
        
        return frame
    
    def update_fps(self) -> None:
        """Update FPS counter."""
        self.fps_counter += 1
        if time.time() - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = time.time()
    
    def add_info_overlay(self, frame: np.ndarray, detections: List[Tuple]) -> np.ndarray:
        """Add information overlay to frame."""
        # FPS counter
        cv2.putText(frame, f'FPS: {self.current_fps}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Detection info
        cv2.putText(frame, f'YOLO Head Detection', 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame, f'Detections: {len(detections)}', 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame, f'Confidence: {self.confidence_threshold:.2f}', 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame, f'Mask: {self.mask_type.upper()}', 
                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw bounding boxes around detections
        for x, y, w, h, confidence in detections:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f'{confidence:.2f}', 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame


def main():
    """Main function for YOLO head detection."""
    parser = argparse.ArgumentParser(description='Simple YOLO Head Detection System')
    parser.add_argument('--input', '-i', type=str, default='webcam',
                       help='Input source: "webcam" or path to video file')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output video file path (optional)')
    parser.add_argument('--model', '-m', type=str, default='models/nano.pt',
                       help='Path to YOLO head detection model')
    parser.add_argument('--confidence', '-c', type=float, default=0.3,
                       help='Detection confidence threshold (0.0-1.0)')
    parser.add_argument('--mask-type', '-t', type=str, default='pixelate',
                       choices=['pixelate', 'blur', 'solid', 'black'],
                       help='Type of mask to apply')
    parser.add_argument('--show-info', '-s', action='store_true',
                       help='Show detection info overlay')
    parser.add_argument('--show-boxes', '-b', action='store_true',
                       help='Show bounding boxes around detections')
    
    args = parser.parse_args()
    
    print("🚀 YOLO Head Detection System")
    print("=" * 40)
    
    # Initialize YOLO head detector with your specific model
    try:
        detector = YOLOHeadDetector(
            model_path=args.model,
            confidence_threshold=args.confidence,
            mask_type=args.mask_type
        )
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        return
    
    # Setup input source
    if args.input == 'webcam':
        cap = cv2.VideoCapture(0)
        print("📹 Using webcam input...")
    else:
        cap = cv2.VideoCapture(args.input)
        print(f"📹 Using video input: {args.input}")
    
    if not cap.isOpened():
        print("❌ Error: Could not open input source")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📊 Video properties: {width}x{height} @ {fps} FPS")
    
    # Setup output video writer if specified
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"💾 Saving output to: {args.output}")
    
    print("\n🎮 Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 'i' to toggle info overlay")
    print("  - Press 'b' to toggle bounding boxes")
    print("  - Press '+/-' to adjust confidence")
    print("  - Press '1/2/3/4' to change mask type")
    print("\n⚡ Starting detection...")
    
    show_info = args.show_info
    show_boxes = args.show_boxes
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if args.input == 'webcam':
                    print("❌ Error: Failed to read from webcam")
                    break
                else:
                    print("📁 End of video reached")
                    break
            
            frame_count += 1
            
            # Get detections for info display
            detections = detector.detect_heads(frame.copy()) if (show_info or show_boxes) else []
            
            # Process frame (apply masks)
            processed_frame = detector.process_frame(frame)
            
            # Add info overlay if enabled
            if show_info or show_boxes:
                processed_frame = detector.add_info_overlay(processed_frame, detections)
            
            # Update FPS counter
            detector.update_fps()
            
            # Write frame to output video if specified
            if out:
                out.write(processed_frame)
            
            # Display frame
            cv2.imshow('YOLO Head Detection - Simple & Effective', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('i'):
                show_info = not show_info
                print(f"ℹ️ Info overlay: {'ON' if show_info else 'OFF'}")
            elif key == ord('b'):
                show_boxes = not show_boxes
                print(f"📦 Bounding boxes: {'ON' if show_boxes else 'OFF'}")
            elif key == ord('+') or key == ord('='):
                detector.confidence_threshold = min(1.0, detector.confidence_threshold + 0.05)
                print(f"📈 Confidence: {detector.confidence_threshold:.2f}")
            elif key == ord('-'):
                detector.confidence_threshold = max(0.05, detector.confidence_threshold - 0.05)
                print(f"📉 Confidence: {detector.confidence_threshold:.2f}")
            elif key == ord('1'):
                detector.mask_type = 'pixelate'
                print("🔲 Switched to pixelate mask")
            elif key == ord('2'):
                detector.mask_type = 'blur'
                print("🟦 Switched to blur mask")
            elif key == ord('3'):
                detector.mask_type = 'solid'
                print("⬜ Switched to solid mask")
            elif key == ord('4'):
                detector.mask_type = 'black'
                print("⬛ Switched to black mask")
    
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        print(f"\n✅ Processed {frame_count} frames")
        print("🧹 Cleanup completed")


if __name__ == "__main__":
    main()