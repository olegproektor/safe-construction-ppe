"""
Online Camera PPE Detection Module with IP Camera Support

This module provides real-time PPE (Personal Protective Equipment) detection
through webcam feed or IP camera streams using YOLOv8 model. It supports local cameras,
RTSP/HTTP IP cameras, video recording, and comprehensive error handling.

Features:
- Local camera support (webcam)
- IP camera support (RTSP/HTTP/HTTPS)
- Real-time PPE detection and violation highlighting
- Video recording capability
- Automatic reconnection for unstable streams
- Command-line interface with configurable parameters

Author: PPE Detection System
Date: 2025-08-26
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import cv2
import numpy as np

# Add app directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from inference import run_inference_with_ppe_analysis
except ImportError:
    print("Warning: inference.py module not found. Please ensure inference.py exists in app/ directory.")
    sys.exit(1)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OnlinePPEDetector:
    """
    Real-time PPE detection class for webcam and IP camera streaming.
    
    This class handles camera capture (local/IP), frame processing through YOLOv8 inference,
    real-time display of annotated results with PPE violation detection, and optional video recording.
    
    Features:
    - Support for local cameras and IP cameras (RTSP/HTTP)
    - Real-time PPE detection with violation highlighting
    - Automatic reconnection for unstable streams
    - Video recording capability
    - Configurable detection parameters
    """
    
    def __init__(
        self, 
        stream_url: Optional[str] = None,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.5,
        save_output: Optional[str] = None,
        window_name: str = "PPE Detection - Live Stream"
    ):
        """
        Initialize the online PPE detector.
        
        Args:
            stream_url (Optional[str]): Camera stream URL (RTSP/HTTP) or None for local camera
            conf_threshold (float): Confidence threshold for detections
            iou_threshold (float): IoU threshold for NMS
            save_output (Optional[str]): Output video file path for recording
            window_name (str): Name of the OpenCV display window
        """
        self.stream_url = stream_url
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.save_output = save_output
        self.window_name = window_name
        
        # Camera and video writer objects
        self.cap: Optional[cv2.VideoCapture] = None
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.is_running = False
        
        # Stream monitoring
        self.last_frame_time = time.time()
        self.reconnect_delay = 5.0  # seconds
        self.max_reconnect_attempts = 3
        
        # Display settings
        self.window_width = 1280
        self.window_height = 720
        
        # Frame properties (will be set after first successful frame)
        self.frame_width = None
        self.frame_height = None
        self.fps = 30.0
        
        # Violation tracking
        self.last_violation_save = 0
        self.violation_save_interval = 30.0  # Save at most every 30 seconds
        self.auto_record_violations = True  # Automatically record videos when violations occur
        self.violations_video_writer: Optional[cv2.VideoWriter] = None
        self.recording_violation = False
        self.violation_recording_start_time = 0
        self.violation_recording_duration = 10.0  # Record 10 seconds when violation detected
        
        # Enhanced violation session tracking
        self.current_violation_session = None
        self.violation_frame_count = 0
        self.min_violation_frames = 30  # Require at least 30 frames with violations before saving
        self.min_session_duration = 5.0  # Require at least 5 seconds of violations
        
    def initialize_camera(self) -> bool:
        """
        Initialize and configure the camera (local or IP stream).
        
        Returns:
            bool: True if camera initialization successful, False otherwise
        """
        try:
            # Determine camera source
            if self.stream_url:
                # IP camera (RTSP/HTTP)
                logger.info(f"Connecting to IP camera: {self.stream_url}")
                camera_source = self.stream_url
                
                # Set additional options for IP cameras
                self.cap = cv2.VideoCapture(camera_source)
                
                # Configure IP camera specific settings
                if self.stream_url.startswith('rtsp://'):
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for real-time
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    
            else:
                # Local camera
                logger.info("Connecting to local camera (index 0)")
                camera_source = 0
                self.cap = cv2.VideoCapture(camera_source)
                
                # Set local camera properties
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Check if camera opened successfully
            if not self.cap.isOpened():
                error_msg = f"Failed to open camera: {camera_source}"
                logger.error(error_msg)
                if self.stream_url:
                    print(f"Ошибка: невозможно подключиться к камере {self.stream_url}")
                return False
            
            # Test camera by reading one frame
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                error_msg = "Camera opened but cannot read frames"
                logger.error(error_msg)
                if self.stream_url:
                    print(f"Ошибка: невозможно подключиться к камере {self.stream_url}")
                return False
            
            # Store frame properties
            self.frame_height, self.frame_width = test_frame.shape[:2]
            
            # Get actual FPS if available
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                self.fps = fps
                
            logger.info(f"Camera initialized successfully - Resolution: {self.frame_width}x{self.frame_height}, FPS: {self.fps}")
            
            # Initialize video writer if output file specified
            if self.save_output:
                self._initialize_video_writer()
                
            return True
            
        except Exception as e:
            error_msg = f"Error initializing camera: {str(e)}"
            logger.error(error_msg)
            if self.stream_url:
                print(f"Ошибка: невозможно подключиться к камере {self.stream_url}")
            return False
    
    def _initialize_video_writer(self) -> bool:
        """
        Initialize video writer for recording output.
        
        Returns:
            bool: True if video writer initialized successfully
        """
        try:
            if not self.save_output or not self.frame_width or not self.frame_height:
                return False
                
            # Define codec and create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            self.video_writer = cv2.VideoWriter(
                self.save_output,
                fourcc,
                self.fps,
                (self.frame_width, self.frame_height)
            )
            
            if not self.video_writer.isOpened():
                logger.error(f"Failed to initialize video writer for {self.save_output}")
                return False
                
            logger.info(f"Video recording initialized: {self.save_output}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing video writer: {str(e)}")
            return False
    
    def reconnect_camera(self) -> bool:
        """
        Attempt to reconnect to the camera stream.
        
        Returns:
            bool: True if reconnection successful
        """
        logger.info("Attempting to reconnect to camera...")
        
        # Release current connection
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Wait before reconnection
        time.sleep(self.reconnect_delay)
        
        # Try to reconnect
        return self.initialize_camera()
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process a single frame through PPE detection inference.
        
        Args:
            frame (np.ndarray): Input frame from camera
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: Annotated frame and detection results
        """
        try:
            # Run inference with drawing enabled and configurable thresholds
            result = run_inference_with_ppe_analysis(
                frame, 
                draw=True,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold
            )
            
            # Extract annotated frame
            if 'annotated_frame' in result and result['annotated_frame'] is not None:
                annotated_frame = result['annotated_frame']
            else:
                # If no annotated frame available, use original
                annotated_frame = frame
                logger.warning("No annotated_frame in result")
            
            return annotated_frame, result
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            # Return original frame and empty result on error
            return frame, {"error": str(e), "violations": [], "detections": []}
    
    def display_frame_info(self, frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """
        Add additional information overlay to the frame with prominent violation count.
        
        Args:
            frame (np.ndarray): Input frame
            result (Dict[str, Any]): Detection results
            
        Returns:
            np.ndarray: Frame with added information overlay
        """
        try:
            # Create a copy to avoid modifying original
            display_frame = frame.copy()
            
            # Get frame dimensions
            height, width = display_frame.shape[:2]
            
            # Get violation and detection counts
            violations_count = len(result.get('violations', []))
            detections_count = len(result.get('detections', []))
            persons_count = result.get('total_persons', 0)
            
            # Prominent violations display in top-left corner
            violation_text = f"Violations: {violations_count}"
            violation_color = (0, 0, 255) if violations_count > 0 else (0, 255, 0)  # Red if violations, green if none
            
            # Large background for violation count
            cv2.rectangle(display_frame, (10, 10), (300, 60), (0, 0, 0), -1)  # Black background
            cv2.rectangle(display_frame, (10, 10), (300, 60), violation_color, 3)  # Colored border
            
            # Large violation text
            cv2.putText(display_frame, violation_text, 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, violation_color, 3)
            
            # Additional info below
            info_y_start = 80
            info_background_height = 80
            
            # Background for additional info
            cv2.rectangle(display_frame, (10, info_y_start), (400, info_y_start + info_background_height), (0, 0, 0), -1)
            cv2.rectangle(display_frame, (10, info_y_start), (400, info_y_start + info_background_height), (255, 255, 255), 2)
            
            # Stream source info
            source_text = "IP Camera" if self.stream_url else "Local Camera"
            cv2.putText(display_frame, f"Source: {source_text}", 
                       (20, info_y_start + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Detection statistics
            cv2.putText(display_frame, f"Persons: {persons_count} | Total Detections: {detections_count}", 
                       (20, info_y_start + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Thresholds info
            cv2.putText(display_frame, f"Conf: {self.conf_threshold:.2f} | IoU: {self.iou_threshold:.2f}", 
                       (20, info_y_start + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Instructions in bottom-right
            cv2.putText(display_frame, "Press 'q' to quit", 
                       (width - 200, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Recording indicator if saving video
            if self.save_output and self.video_writer:
                cv2.circle(display_frame, (width - 30, 30), 10, (0, 0, 255), -1)  # Red circle
                cv2.putText(display_frame, "REC", 
                           (width - 70, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            return display_frame
            
        except Exception as e:
            logger.error(f"Error adding frame info: {str(e)}")
            return frame
    
    def start_violation_recording(self) -> bool:
        """
        Start recording video when violation is detected.
        
        Returns:
            bool: True if recording started successfully
        """
        try:
            if self.recording_violation or not self.frame_width or not self.frame_height:
                return False
            
            # Create violations folder if it doesn't exist
            from pathlib import Path
            violations_folder = Path.cwd() / "violations"
            violations_folder.mkdir(exist_ok=True)
            
            # Generate filename with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            camera_type = "ip" if self.stream_url else "local"
            filename = f"violation_video_{camera_type}_{timestamp}.mp4"
            video_path = violations_folder / filename
            
            # Define codec and create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            self.violations_video_writer = cv2.VideoWriter(
                str(video_path),
                fourcc,
                self.fps,
                (self.frame_width, self.frame_height)
            )
            
            if not self.violations_video_writer.isOpened():
                logger.error(f"Failed to initialize violation video writer: {video_path}")
                return False
            
            self.recording_violation = True
            self.violation_recording_start_time = time.time()
            logger.info(f"Started violation recording: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting violation recording: {str(e)}")
            return False
    
    def stop_violation_recording(self) -> None:
        """
        Stop violation video recording.
        """
        try:
            if self.violations_video_writer is not None:
                self.violations_video_writer.release()
                logger.info("Violation recording stopped and saved")
                self.violations_video_writer = None
            
            self.recording_violation = False
            self.violation_recording_start_time = 0
            
        except Exception as e:
            logger.error(f"Error stopping violation recording: {str(e)}")
    
    def save_violation_record(self, result: Dict[str, Any], frame_count: int) -> None:
        """
        Save violation record to violations folder with smart session management.
        
        Args:
            result (Dict[str, Any]): Detection results from inference
            frame_count (int): Current frame number
        """
        try:
            current_time = time.time()
            
            # Check if there are any violations
            violations_summary = result.get('violations_summary', {})
            violations_count = violations_summary.get('violations_count', {})
            total_violations = sum(violations_count.values()) if violations_count else 0
            
            if total_violations > 0:
                # Count violation frames
                self.violation_frame_count += 1
                
                # Start session if not already started
                if self.current_violation_session is None:
                    self.current_violation_session = {
                        'start_time': current_time,
                        'start_frame': frame_count
                    }
                    logger.info(f"Started violation session at frame {frame_count}")
            else:
                # No violations in this frame
                if self.current_violation_session is not None:
                    # Check if we should save the completed session
                    session_duration = current_time - self.current_violation_session['start_time']
                    
                    if (self.violation_frame_count >= self.min_violation_frames and 
                        session_duration >= self.min_session_duration and
                        current_time - self.last_violation_save >= self.violation_save_interval):
                        
                        # Save the violation session
                        self._save_violation_session(result, frame_count, session_duration)
                    
                    # Reset session
                    self.current_violation_session = None
                    self.violation_frame_count = 0
                return
            
            # Check if current session should be saved (for ongoing violations)
            if (self.current_violation_session is not None and
                self.violation_frame_count >= self.min_violation_frames):
                
                session_duration = current_time - self.current_violation_session['start_time']
                
                # Save if enough time passed and session is long enough
                if (session_duration >= self.min_session_duration and
                    current_time - self.last_violation_save >= self.violation_save_interval):
                    
                    self._save_violation_session(result, frame_count, session_duration)
                    
        except Exception as e:
            logger.error(f"Error in violation tracking: {str(e)}")
    
    def _save_violation_session(self, result: Dict[str, Any], frame_count: int, session_duration: float) -> None:
        """
        Save a complete violation session to file.
        
        Args:
            result (Dict[str, Any]): Detection results from inference
            frame_count (int): Current frame number
            session_duration (float): Duration of violation session in seconds
        """
        try:
            # Create violations folder if it doesn't exist
            from pathlib import Path
            violations_folder = Path.cwd() / "violations"
            violations_folder.mkdir(exist_ok=True)
            
            # Generate filename with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
            filename = f"camera_violation_session_{timestamp}.json"
            file_path = violations_folder / filename
            
            # Create violation data
            violation_data = {
                "timestamp": datetime.now().isoformat(),
                "source": "camera_application",
                "camera_type": "ip_camera" if self.stream_url else "local_camera",
                "stream_url": self.stream_url or "local_camera",
                "session_info": {
                    "start_frame": self.current_violation_session['start_frame'],
                    "end_frame": frame_count,
                    "duration_seconds": round(session_duration, 2),
                    "violation_frames_count": self.violation_frame_count
                },
                "violations_summary": result.get('violations_summary', {}),
                "persons": result.get('persons', []),
                "detections": result.get('detections', []),
                "processing_parameters": {
                    "conf_threshold": self.conf_threshold,
                    "iou_threshold": self.iou_threshold
                },
                "video_recording": self.save_output is not None,
                "video_file": self.save_output
            }
            
            # Save to file
            import json
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(violation_data, f, indent=2, ensure_ascii=False)
            
            total_violations = sum(result.get('violations_summary', {}).get('violations_count', {}).values())
            logger.info(f"Violation session saved: {filename}")
            logger.info(f"  Duration: {session_duration:.1f}s, Frames: {self.violation_frame_count}, Violations: {total_violations}")
            
            # Update tracking
            self.last_violation_save = time.time()
            self.violation_frame_count = 0
            
        except Exception as e:
            logger.error(f"Error saving violation session: {str(e)}")

    def print_detection_results(self, result: Dict[str, Any], frame_count: int) -> None:
        """
        Print detection results to console in JSON format.
        
        Args:
            result (Dict[str, Any]): Detection results from inference
            frame_count (int): Current frame number
        """
        try:
            # Create output structure
            output = {
                "frame": frame_count,
                "timestamp": cv2.getTickCount() / cv2.getTickFrequency(),
                "detections": result.get('detections', []),
                "violations": result.get('violations', []),
                "total_persons": len([d for d in result.get('detections', []) if d.get('class') == 'person']),
                "total_violations": len(result.get('violations', []))
            }
            
            # Print JSON to console
            print(json.dumps(output, indent=2))
            
        except Exception as e:
            logger.error(f"Error printing results: {str(e)}")
            print(f'{{"frame": {frame_count}, "error": "{str(e)}"}}')
    
    def run(self) -> None:
        """
        Main execution loop for real-time PPE detection.
        
        Captures frames from camera (local/IP), processes them through inference,
        displays annotated results, prints detection data to console, and optionally
        records video output with automatic reconnection for IP cameras.
        """
        # Initialize camera
        if not self.initialize_camera():
            logger.error("Failed to initialize camera. Exiting.")
            return
        
        # Create display window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_width, self.window_height)
        
        camera_type = "IP camera" if self.stream_url else "local camera"
        logger.info(f"Starting real-time PPE detection with {camera_type}. Press 'q' to quit.")
        
        if self.save_output:
            logger.info(f"Recording video to: {self.save_output}")
        
        self.is_running = True
        frame_count = 0
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        try:
            while self.is_running:
                # Capture frame from camera
                ret, frame = self.cap.read()
                
                # Handle frame reading failures
                if not ret or frame is None:
                    consecutive_failures += 1
                    logger.warning(f"Failed to read frame from camera (attempt {consecutive_failures})")
                    
                    # For IP cameras, attempt reconnection after several failures
                    if self.stream_url and consecutive_failures >= max_consecutive_failures:
                        logger.warning("Too many consecutive failures. Attempting reconnection...")
                        
                        for attempt in range(self.max_reconnect_attempts):
                            if self.reconnect_camera():
                                logger.info("Reconnection successful")
                                consecutive_failures = 0
                                break
                            else:
                                logger.warning(f"Reconnection attempt {attempt + 1} failed")
                                if attempt < self.max_reconnect_attempts - 1:
                                    time.sleep(self.reconnect_delay)
                        
                        if consecutive_failures >= max_consecutive_failures:
                            logger.error("Failed to reconnect after multiple attempts. Exiting.")
                            break
                    
                    # Skip this iteration and try again
                    time.sleep(0.1)
                    continue
                
                # Reset failure counter on successful frame read
                consecutive_failures = 0
                frame_count += 1
                self.last_frame_time = time.time()
                
                # Process frame through PPE detection
                annotated_frame, result = self.process_frame(frame)
                
                # Check for violations and manage recording
                violations_summary = result.get('violations_summary', {})
                violations_count = violations_summary.get('violations_count', {})
                total_violations = sum(violations_count.values()) if violations_count else 0
                
                # Start violation recording if violations detected
                if total_violations > 0 and self.auto_record_violations and not self.recording_violation:
                    self.start_violation_recording()
                
                # Stop violation recording if duration exceeded or no violations
                if self.recording_violation:
                    current_recording_time = time.time() - self.violation_recording_start_time
                    if current_recording_time >= self.violation_recording_duration:
                        self.stop_violation_recording()
                
                # Add additional information overlay
                display_frame = self.display_frame_info(annotated_frame, result)
                
                # Save frame to video file if recording
                if self.video_writer and self.video_writer.isOpened():
                    try:
                        self.video_writer.write(display_frame)
                    except Exception as e:
                        logger.error(f"Error writing video frame: {str(e)}")
                
                # Save frame to violation video if recording violation
                if self.violations_video_writer and self.violations_video_writer.isOpened():
                    try:
                        self.violations_video_writer.write(display_frame)
                    except Exception as e:
                        logger.error(f"Error writing violation video frame: {str(e)}")
                
                # Display the frame
                cv2.imshow(self.window_name, display_frame)
                
                # Print detection results to console
                self.print_detection_results(result, frame_count)
                
                # Save violation record if violations detected
                self.save_violation_record(result, frame_count)
                
                # Check for quit command
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC key
                    logger.info("Quit command received. Stopping detection.")
                    break
                
                # Check if window was closed
                try:
                    if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                        logger.info("Window closed. Stopping detection.")
                        break
                except cv2.error:
                    # Window was closed
                    logger.info("Window closed. Stopping detection.")
                    break
                    
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Stopping detection.")
            
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {str(e)}")
            
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """
        Clean up resources, close video writer, and close windows.
        """
        self.is_running = False
        
        # Release video writer
        if self.video_writer is not None:
            try:
                self.video_writer.release()
                logger.info(f"Video saved to: {self.save_output}")
            except Exception as e:
                logger.error(f"Error releasing video writer: {str(e)}")
            finally:
                self.video_writer = None
        
        # Release violation video writer
        if self.violations_video_writer is not None:
            try:
                self.violations_video_writer.release()
                logger.info("Violation video recording stopped and saved")
            except Exception as e:
                logger.error(f"Error releasing violation video writer: {str(e)}")
            finally:
                self.violations_video_writer = None
        
        # Release camera
        if self.cap is not None:
            try:
                self.cap.release()
                logger.info("Camera released")
            except Exception as e:
                logger.error(f"Error releasing camera: {str(e)}")
            finally:
                self.cap = None
        
        # Close OpenCV windows
        try:
            cv2.destroyAllWindows()
            logger.info("OpenCV windows closed")
        except Exception as e:
            logger.error(f"Error closing windows: {str(e)}")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the PPE detection system.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Real-time PPE Detection System with IP Camera Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local camera with default settings
  python app/online_camera.py
  
  # IP camera (RTSP) with custom confidence
  python app/online_camera.py --stream_url rtsp://192.168.1.15:554/stream1 --conf 0.4
  
  # HTTP camera with video recording
  python app/online_camera.py --stream_url http://192.168.1.20:8080/video --save_output output.mp4
  
  # Full configuration
  python app/online_camera.py \
      --stream_url rtsp://192.168.1.15:554/stream1 \
      --conf 0.35 \
      --iou 0.5 \
      --save_output output.mp4
        """
    )
    
    parser.add_argument(
        '--stream_url',
        type=str,
        default=None,
        help='Camera stream URL (RTSP/HTTP/HTTPS). If not provided, uses local camera (index 0)'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=0.35,
        help='Confidence threshold for detections (0.0-1.0, default: 0.35)'
    )
    
    parser.add_argument(
        '--iou',
        type=float,
        default=0.5,
        help='IoU threshold for Non-Maximum Suppression (0.0-1.0, default: 0.5)'
    )
    
    parser.add_argument(
        '--save_output',
        type=str,
        default=None,
        help='Output video file path for recording (e.g., output.mp4)'
    )
    
    parser.add_argument(
        '--window_name',
        type=str,
        default="PPE Detection - Live Stream",
        help='Name of the display window'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.conf < 0.0 or args.conf > 1.0:
        parser.error("Confidence threshold must be between 0.0 and 1.0")
    
    if args.iou < 0.0 or args.iou > 1.0:
        parser.error("IoU threshold must be between 0.0 and 1.0")
    
    if args.stream_url:
        # Basic URL validation
        valid_schemes = ['rtsp://', 'http://', 'https://']
        if not any(args.stream_url.startswith(scheme) for scheme in valid_schemes):
            parser.error("Stream URL must start with rtsp://, http://, or https://")
    
    return args


def main() -> None:
    """
    Main function for direct script execution with command-line interface.
    
    Parses command-line arguments and initializes the PPE detector with
    specified configuration for local or IP camera streaming.
    """
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        # Display configuration
        print("\n" + "="*60)
        print("PPE Detection System - Camera Stream")
        print("="*60)
        
        if args.stream_url:
            print(f"Stream URL: {args.stream_url}")
        else:
            print("Camera: Local camera (index 0)")
            
        print(f"Confidence threshold: {args.conf}")
        print(f"IoU threshold: {args.iou}")
        
        if args.save_output:
            print(f"Recording to: {args.save_output}")
        else:
            print("Recording: Disabled")
            
        print("\nPress 'q' to quit")
        print("="*60 + "\n")
        
        # Create detector instance with parsed arguments
        detector = OnlinePPEDetector(
            stream_url=args.stream_url,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            save_output=args.save_output,
            window_name=args.window_name
        )
        
        # Run detection
        detector.run()
        
    except KeyboardInterrupt:
        logger.info("\nApplication interrupted by user")
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)
        
    finally:
        print("\nPPE Detection System stopped.")


if __name__ == "__main__":
    main()