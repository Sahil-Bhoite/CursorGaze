#!/usr/bin/env python3
"""
Cursor Gaze Application

A real-time eye gaze redirection implementation using a single window.

Usage:
    python cursor_gaze.py                      # Use dlib backend
    python cursor_gaze.py --backend mediapipe  # Use mediapipe backend
    python cursor_gaze.py --camera 1           # Use camera device 1

Controls:
    - 'g': Toggle Cursor Gaze on/off
    - 'c': Toggle calibration mode
    - 'q': Quit
"""

import cv2
from displayers.dis_single_window import CursorGazeApp, DisplayConfig
from displayers.face_predictor import create_face_predictor


def detect_camera_resolution(camera_id: int) -> tuple[int, int]:
    """
    Detect the actual resolution of the specified camera.
    
    Args:
        camera_id: Camera device ID
        
    Returns:
        Tuple of (width, height) in pixels
    """
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Warning: Could not open camera {camera_id}, using default resolution")
        return (640, 480)
    
    # Get the actual resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    print(f"Detected camera resolution: {width}x{height}")
    return (width, height)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Cursor Gaze Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
  'g' - Toggle Cursor Gaze on/off
  'c' - Toggle calibration mode
  'q' - Quit the application

Examples:
  %(prog)s                         # Use default dlib backend
  %(prog)s --backend mediapipe     # Use MediaPipe for face detection
  %(prog)s --camera 1              # Use camera device 1
        """,
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="mediapipe",
        choices=["dlib", "mediapipe"],
        help="Face detection backend (default: mediapipe)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device ID (default: 0)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./model_managers/cursor_gaze_v1_01.yaml",
        help="Path to Cursor Gaze config file (default: ./model_managers/cursor_gaze_v1_01.yaml)",
    )
    parser.add_argument(
        "--disable-tflite",
        action="store_true",
        help="Disable TFLite inference and use slower TF1.x Graph",
    )
    parser.add_argument(
        "--disable-coreml",
        action="store_true",
        help="Disable CoreML inference (M1/M2) and fallback to TFLite/TF",
    )
    parser.add_argument(
        "--virtual-cam",
        action="store_true",
        help="Output to OBS Virtual Camera (requires OBS Studio installed)",
    )
    args = parser.parse_args()

    # Detect camera resolution
    video_size = detect_camera_resolution(args.camera)
    
    # Calculate appropriate face detection size (half resolution)
    face_detect_size = (video_size[0] // 2, video_size[1] // 2)
    
    # Create display config with detected resolution
    display_config = DisplayConfig(
        video_size=video_size,
        face_detect_size=face_detect_size,
    )
    
    print(f"Video size: {video_size}, Face detection size: {face_detect_size}")

    # Create face predictor based on selected backend
    predictor = create_face_predictor(args.backend)

    # Create and run the application
    app = CursorGazeApp(
        face_predictor=predictor,
        display_config=display_config,
        camera_id=args.camera,
        config_path=args.config,
        use_tflite=not args.disable_tflite,
        use_coreml=not args.disable_coreml,
        use_virtual_cam=args.virtual_cam,
    )
    app.run()


if __name__ == "__main__":
    main()
