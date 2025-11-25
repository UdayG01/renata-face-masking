import cv2
import mediapipe as mp
import numpy as np
import argparse
import time
from typing import Optional, Tuple, List, Dict
from collections import deque

class StableFaceMasker:
    """
    Enhanced MediaPipe-based face masking with tracking and temporal smoothing
    to maintain masks during head movements and brief detection losses.
    """
    
    def __init__(self, 
                 detection_confidence: float = 0.2,  # Lower for better tracking
                 mask_type: str = 'pixelate',
                 mask_intensity: float = 0.8,
                 tracking_frames: int = 10,  # Number of frames to remember
                 prediction_frames: int = 5):  # Frames to predict during loss
        """
        Initialize the stable face masker.
        
        Args:
            detection_confidence: Minimum confidence for face detection
            mask_type: Type of mask ('blur', 'solid', 'pixelate', 'black')
            mask_intensity: Intensity of the mask effect
            tracking_frames: Number of previous frames to store for tracking
            prediction_frames: Number of frames to predict mask when detection is lost
        """
        self.detection_confidence = detection_confidence
        self.mask_type = mask_type
        self.mask_intensity = mask_intensity
        self.tracking_frames = tracking_frames
        self.prediction_frames = prediction_frames
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Use model_selection=1 for better longer range detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # Better for side profiles and longer range
            min_detection_confidence=detection_confidence
        )
        
        # Tracking data structures
        self.face_history = deque(maxlen=tracking_frames)  # Store recent face positions
        self.frames_since_detection = 0  # Count frames without detection
        self.last_valid_bbox = None  # Last known good bounding box
        self.movement_predictor = MovementPredictor()  # Predict face movement
        
        # Smoothing parameters
        self.bbox_smoothing_factor = 0.3  # Lower = more smoothing
        self.size_change_threshold = 0.3  # Maximum allowed size change between frames
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
    def smooth_bbox(self, new_bbox: Tuple[int, int, int, int], 
                   prev_bbox: Optional[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
        """
        Apply temporal smoothing to bounding box to reduce jitter.
        """
        if prev_bbox is None:
            return new_bbox
            
        x_new, y_new, w_new, h_new = new_bbox
        x_prev, y_prev, w_prev, h_prev = prev_bbox
        
        # Check for unrealistic size changes (likely detection error)
        size_change = abs(w_new * h_new - w_prev * h_prev) / (w_prev * h_prev)
        if size_change > self.size_change_threshold:
            # Use previous bbox with slight position update
            return (
                int(x_prev + self.bbox_smoothing_factor * (x_new - x_prev)),
                int(y_prev + self.bbox_smoothing_factor * (y_new - y_prev)),
                w_prev, h_prev
            )
        
        # Apply exponential smoothing
        x_smooth = int(x_prev + self.bbox_smoothing_factor * (x_new - x_prev))
        y_smooth = int(y_prev + self.bbox_smoothing_factor * (y_new - y_prev))
        w_smooth = int(w_prev + self.bbox_smoothing_factor * (w_new - w_prev))
        h_smooth = int(h_prev + self.bbox_smoothing_factor * (h_new - h_prev))
        
        return (x_smooth, y_smooth, w_smooth, h_smooth)
    
    def predict_face_position(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Predict face position based on movement history when detection is lost.
        """
        if len(self.face_history) < 2:
            return self.last_valid_bbox
            
        return self.movement_predictor.predict_next_position(
            list(self.face_history), self.frames_since_detection
        )
    
    def get_face_bbox_enhanced(self, detection, image_width: int, image_height: int) -> Tuple[int, int, int, int]:
        """
        Enhanced bounding box extraction with better padding for head movements.
        """
        bbox = detection.location_data.relative_bounding_box
        
        # Convert relative coordinates to absolute
        x = int(bbox.xmin * image_width)
        y = int(bbox.ymin * image_height)
        w = int(bbox.width * image_width)
        h = int(bbox.height * image_height)
        
        # Enhanced padding for head movements - LARGER MASK SIZE
        # More horizontal padding for left-right movement
        h_padding = int(0.4 * w)   # 40% horizontal padding (increased from 25%)
        v_padding = int(0.3 * h)   # 30% vertical padding (increased from 15%)
        
        x = max(0, x - h_padding)
        y = max(0, y - v_padding)
        w = min(image_width - x, w + 2 * h_padding)
        h = min(image_height - y, h + 2 * v_padding)
        
        return x, y, w, h
    
    def apply_mask_with_confidence(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                                 confidence: float = 1.0) -> np.ndarray:
        """
        Apply mask with variable intensity based on confidence.
        """
        if confidence < 0.3:  # Very low confidence, lighter mask
            intensity_factor = 0.5
        elif confidence < 0.7:  # Medium confidence
            intensity_factor = 0.8
        else:  # High confidence, full mask
            intensity_factor = 1.0
            
        # Temporarily adjust mask intensity
        original_intensity = self.mask_intensity
        self.mask_intensity *= intensity_factor
        
        # Apply the mask
        if self.mask_type == 'blur':
            frame = self.apply_blur_mask(frame, bbox)
        elif self.mask_type == 'solid':
            frame = self.apply_solid_mask(frame, bbox)
        elif self.mask_type == 'pixelate':
            frame = self.apply_pixelate_mask(frame, bbox)
        elif self.mask_type == 'black':
            frame = self.apply_black_mask(frame, bbox)
            
        # Restore original intensity
        self.mask_intensity = original_intensity
        return frame
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhanced frame processing with tracking and prediction.
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe
        results = self.face_detection.process(rgb_frame)
        
        current_detections = []
        
        # Process detections
        if results.detections:
            for detection in results.detections:
                confidence = detection.score[0]
                
                if confidence >= self.detection_confidence:
                    # Get bounding box
                    bbox = self.get_face_bbox_enhanced(detection, frame.shape[1], frame.shape[0])
                    
                    # Apply smoothing if we have previous detection
                    if self.last_valid_bbox is not None:
                        bbox = self.smooth_bbox(bbox, self.last_valid_bbox)
                    
                    current_detections.append((bbox, confidence))
        
        # Determine which bounding boxes to use for masking
        bboxes_to_mask = []
        
        if current_detections:
            # We have detections - reset counters and update tracking
            self.frames_since_detection = 0
            
            for bbox, confidence in current_detections:
                bboxes_to_mask.append((bbox, confidence))
                
                # Update tracking history
                self.face_history.append(bbox)
                self.last_valid_bbox = bbox
                
        else:
            # No detections - use prediction if within threshold
            self.frames_since_detection += 1
            
            if self.frames_since_detection <= self.prediction_frames:
                # Predict position based on movement history
                predicted_bbox = self.predict_face_position()
                
                if predicted_bbox is not None:
                    # Use predicted position with reduced confidence
                    prediction_confidence = max(0.3, 1.0 - (self.frames_since_detection * 0.2))
                    bboxes_to_mask.append((predicted_bbox, prediction_confidence))
        
        # Apply masks
        for bbox, confidence in bboxes_to_mask:
            frame = self.apply_mask_with_confidence(frame, bbox, confidence)
        
        return frame
    
    # Include all the original masking methods
    def apply_blur_mask(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Apply Gaussian blur mask to face region."""
        x, y, w, h = bbox
        
        # Ensure coordinates are within image bounds
        x, y = max(0, x), max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        
        if w <= 0 or h <= 0:
            return image
            
        face_region = image[y:y+h, x:x+w]
        blur_intensity = int(15 + (self.mask_intensity * 35))
        if blur_intensity % 2 == 0:
            blur_intensity += 1
            
        blurred_face = cv2.GaussianBlur(face_region, (blur_intensity, blur_intensity), 0)
        image[y:y+h, x:x+w] = blurred_face
        return image
    
    def apply_solid_mask(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Apply solid color mask to face region."""
        x, y, w, h = bbox
        
        colors = {
            'low': (128, 128, 128),
            'medium': (0, 0, 0),
            'high': (255, 255, 255)
        }
        
        if self.mask_intensity < 0.33:
            color = colors['low']
        elif self.mask_intensity < 0.66:
            color = colors['medium']
        else:
            color = colors['high']
            
        cv2.rectangle(image, (x, y), (x+w, y+h), color, -1)
        return image
    
    def apply_pixelate_mask(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Apply pixelation mask to face region."""
        x, y, w, h = bbox
        
        # Ensure coordinates are within bounds
        x, y = max(0, x), max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        
        if w <= 0 or h <= 0:
            return image
            
        face_region = image[y:y+h, x:x+w]
        pixel_size = max(2, int(5 + (self.mask_intensity * 20)))
        
        # Avoid division by zero
        small_w = max(1, w // pixel_size)
        small_h = max(1, h // pixel_size)
        
        small_face = cv2.resize(face_region, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        pixelated_face = cv2.resize(small_face, (w, h), interpolation=cv2.INTER_NEAREST)
        image[y:y+h, x:x+w] = pixelated_face
        return image
    
    def apply_black_mask(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Apply black mask to face region."""
        x, y, w, h = bbox
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 0), -1)
        return image
    
    def update_fps(self) -> None:
        """Update FPS counter."""
        self.fps_counter += 1
        if time.time() - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = time.time()
    
    def add_info_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Add enhanced information overlay."""
        cv2.putText(frame, f'FPS: {self.current_fps}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'Mask: {self.mask_type.upper()}', 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'Confidence: {self.detection_confidence:.2f}', 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'Tracking: {len(self.face_history)} frames', 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show prediction status
        if self.frames_since_detection > 0:
            status = f'PREDICTING ({self.frames_since_detection}/{self.prediction_frames})'
            color = (0, 255, 255)  # Yellow for prediction
        else:
            status = 'DETECTING'
            color = (0, 255, 0)  # Green for active detection
            
        cv2.putText(frame, status, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return frame


class MovementPredictor:
    """
    Predicts face movement based on historical positions.
    """
    
    def predict_next_position(self, position_history: List[Tuple[int, int, int, int]], 
                            frames_ahead: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Predict face position based on movement velocity.
        """
        if len(position_history) < 2:
            return position_history[-1] if position_history else None
            
        # Calculate movement velocity from last two positions
        current_pos = position_history[-1]
        prev_pos = position_history[-2]
        
        x_curr, y_curr, w_curr, h_curr = current_pos
        x_prev, y_prev, w_prev, h_prev = prev_pos
        
        # Calculate velocity (pixels per frame)
        dx = x_curr - x_prev
        dy = y_curr - y_prev
        
        # Predict position with damping (movement slows down over time)
        damping_factor = 0.8 ** frames_ahead  # Movement slows down
        
        predicted_x = int(x_curr + dx * frames_ahead * damping_factor)
        predicted_y = int(y_curr + dy * frames_ahead * damping_factor)
        
        # Keep size relatively stable with slight decay
        size_factor = 0.95 ** frames_ahead
        predicted_w = int(w_curr * size_factor)
        predicted_h = int(h_curr * size_factor)
        
        return (predicted_x, predicted_y, predicted_w, predicted_h)


def main():
    """Main function with enhanced face masking."""
    parser = argparse.ArgumentParser(description='Stable MediaPipe Face Masking System')
    parser.add_argument('--input', '-i', type=str, default='webcam',
                       help='Input source: "webcam", "video", or path to video file')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output video file path (optional)')
    parser.add_argument('--mask-type', '-m', type=str, default='pixelate',
                       choices=['blur', 'solid', 'pixelate', 'black'],
                       help='Type of face mask to apply')
    parser.add_argument('--confidence', '-c', type=float, default=0.2,
                       help='Detection confidence threshold (0.0-1.0)')
    parser.add_argument('--intensity', '-int', type=float, default=0.8,
                       help='Mask intensity (0.0-1.0)')
    parser.add_argument('--tracking-frames', '-t', type=int, default=10,
                       help='Number of frames to track for smoothing')
    parser.add_argument('--prediction-frames', '-p', type=int, default=5,
                       help='Number of frames to predict during detection loss')
    parser.add_argument('--show-info', '-s', action='store_true',
                       help='Show information overlay')
    
    args = parser.parse_args()
    
    # Initialize enhanced face masker
    masker = StableFaceMasker(
        detection_confidence=args.confidence,
        mask_type=args.mask_type,
        mask_intensity=args.intensity,
        tracking_frames=args.tracking_frames,
        prediction_frames=args.prediction_frames
    )
    
    # Setup input source
    if args.input == 'webcam':
        cap = cv2.VideoCapture(0)
        print("Using webcam input...")
    elif args.input == 'video':
        video_path = input("Enter video file path: ")
        cap = cv2.VideoCapture(video_path)
        print(f"Using video input: {video_path}")
    else:
        cap = cv2.VideoCapture(args.input)
        print(f"Using video input: {args.input}")
    
    if not cap.isOpened():
        print("Error: Could not open input source")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties: {width}x{height} @ {fps} FPS")
    print(f"Tracking: {args.tracking_frames} frames, Prediction: {args.prediction_frames} frames")
    
    # Setup output
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"Saving output to: {args.output}")
    
    print("\nEnhanced Controls:")
    print("- Press 'q' to quit")
    print("- Press 'b/s/p/k' to change mask type")
    print("- Press '+/-' to adjust confidence")
    print("- Press 'i' to toggle info overlay")
    print("- Press 't' to increase tracking frames")
    print("- Press 'r' to decrease tracking frames")
    
    show_info = args.show_info
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if args.input == 'webcam':
                    print("Error: Failed to read from webcam")
                    break
                else:
                    print("End of video reached")
                    break
            
            # Process frame with enhanced tracking
            processed_frame = masker.process_frame(frame)
            
            # Add info overlay
            if show_info:
                processed_frame = masker.add_info_overlay(processed_frame)
            
            # Update FPS counter
            masker.update_fps()
            
            # Write frame
            if out:
                out.write(processed_frame)
            
            # Display frame
            cv2.imshow('Stable Face Masking - Enhanced Tracking', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('b'):
                masker.mask_type = 'blur'
                print("Switched to blur mask")
            elif key == ord('s'):
                masker.mask_type = 'solid'
                print("Switched to solid mask")
            elif key == ord('p'):
                masker.mask_type = 'pixelate'
                print("Switched to pixelate mask")
            elif key == ord('k'):
                masker.mask_type = 'black'
                print("Switched to black mask")
            elif key == ord('+') or key == ord('='):
                masker.detection_confidence = min(1.0, masker.detection_confidence + 0.1)
                print(f"Detection confidence: {masker.detection_confidence:.2f}")
            elif key == ord('-'):
                masker.detection_confidence = max(0.1, masker.detection_confidence - 0.1)
                print(f"Detection confidence: {masker.detection_confidence:.2f}")
            elif key == ord('i'):
                show_info = not show_info
                print(f"Info overlay: {'ON' if show_info else 'OFF'}")
            elif key == ord('t'):
                masker.tracking_frames = min(20, masker.tracking_frames + 2)
                masker.face_history = deque(masker.face_history, maxlen=masker.tracking_frames)
                print(f"Tracking frames: {masker.tracking_frames}")
            elif key == ord('r'):
                masker.tracking_frames = max(3, masker.tracking_frames - 2)
                masker.face_history = deque(masker.face_history, maxlen=masker.tracking_frames)
                print(f"Tracking frames: {masker.tracking_frames}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        print("Cleanup completed")


if __name__ == "__main__":
    main()