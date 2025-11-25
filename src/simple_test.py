#!/usr/bin/env python3
"""
Simple test script for MediaPipe face masking.
This script provides an easy way to test the face masking functionality.
"""

import cv2
import mediapipe as mp
import numpy as np

def simple_face_masking_test():
    """
    Simple function to test face masking with webcam.
    No command line arguments needed - just run and test!
    """
    print("=== MediaPipe Face Masking Test ===")
    print("This will open your webcam and apply face masking in real-time.")
    print("\nControls:")
    print("- Press 'q' to quit")
    print("- Press 'b' for blur mask")
    print("- Press 's' for solid mask") 
    print("- Press 'p' for pixelate mask")
    print("- Press 'k' for black mask")
    print("\nPress any key to continue...")
    input()
    
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    )
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        print("Make sure your webcam is connected and not being used by another application")
        return
    
    print("Webcam opened successfully!")
    print("You should see a window with your video feed and face masking applied.")
    
    mask_type = 'blur'  # Default mask type
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read from webcam")
            break
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = face_detection.process(rgb_frame)
        
        # Apply face masking if faces detected
        if results.detections:
            for detection in results.detections:
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                
                # Convert to pixel coordinates
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Add padding
                padding = 10
                x = max(0, x - padding)
                y = max(0, y - padding)
                width = min(w - x, width + 2 * padding)
                height = min(h - y, height + 2 * padding)
                
                # Apply selected mask type
                if mask_type == 'blur':
                    # Gaussian blur
                    face_region = frame[y:y+height, x:x+width]
                    blurred_face = cv2.GaussianBlur(face_region, (51, 51), 0)
                    frame[y:y+height, x:x+width] = blurred_face
                    
                elif mask_type == 'solid':
                    # Solid gray rectangle
                    cv2.rectangle(frame, (x, y), (x+width, y+height), (128, 128, 128), -1)
                    
                elif mask_type == 'pixelate':
                    # Pixelation effect
                    face_region = frame[y:y+height, x:x+width]
                    small_face = cv2.resize(face_region, (width//15, height//15), interpolation=cv2.INTER_LINEAR)
                    pixelated_face = cv2.resize(small_face, (width, height), interpolation=cv2.INTER_NEAREST)
                    frame[y:y+height, x:x+width] = pixelated_face
                    
                elif mask_type == 'black':
                    # Black rectangle
                    cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 0, 0), -1)
        
        # Add current mask type to display
        cv2.putText(frame, f'Mask: {mask_type.upper()}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Face Masking Test', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            mask_type = 'blur'
            print("Switched to blur mask")
        elif key == ord('s'):
            mask_type = 'solid'
            print("Switched to solid mask")
        elif key == ord('p'):
            mask_type = 'pixelate'
            print("Switched to pixelate mask")
        elif key == ord('k'):
            mask_type = 'black'
            print("Switched to black mask")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Test completed!")


if __name__ == "__main__":
    simple_face_masking_test()