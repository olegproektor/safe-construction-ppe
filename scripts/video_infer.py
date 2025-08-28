#!/usr/bin/env python3
"""
Video Inference Script for PPE Detection

This script processes video files or streams to detect PPE violations
and generate annotated output videos with detection results.

Features:
- Video file processing with batch inference
- Real-time video stream processing
- Progress tracking and statistics
- Multiple output formats
- Comprehensive error handling
- Performance optimization

Author: PPE Detection System
Date: 2025-08-26
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# Add app directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "app"))

try:
    from inference import run_inference_with_ppe_analysis, load_model
    from schemas import Detection, PersonStatus
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure you're running from the project root directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Video processing class for PPE detection.
    
    Handles video file input, frame-by-frame processing, and output generation
    with comprehensive error handling and progress tracking.
    """
    
    def __init__(
        self,
        input_path: str,
        output_path: str,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.5,
        save_detections: bool = False,
        batch_size: int = 1,
        skip_frames: int = 0
    ):
        """
        Initialize video processor.
        
        Args:
            input_path: Input video file path or stream URL
            output_path: Output video file path
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            save_detections: Whether to save detection data to JSON
            batch_size: Number of frames to process in batch
            skip_frames: Number of frames to skip between processing
        """
        self.input_path = input_path
        self.output_path = output_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.save_detections = save_detections
        self.batch_size = batch_size
        self.skip_frames = skip_frames
        
        # Video capture and writer objects
        self.cap: Optional[cv2.VideoCapture] = None
        self.writer: Optional[cv2.VideoWriter] = None
        
        # Video properties
        self.total_frames = 0
        self.fps = 30.0
        self.width = 0
        self.height = 0
        
        # Statistics
        self.processed_frames = 0
        self.total_detections = 0
        self.total_violations = 0
        self.processing_times = []
        
        # Detection data storage
        self.detections_data = []
    
    def initialize_video_capture(self) -> bool:
        """
        Initialize video capture from file or stream.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            self.cap = cv2.VideoCapture(self.input_path)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open video: {self.input_path}")
                return False
            
            # Get video properties
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"Video loaded: {self.width}x{self.height} @ {self.fps:.1f}fps, {self.total_frames} frames")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing video capture: {str(e)}")
            return False
    
    def initialize_video_writer(self) -> bool:
        """
        Initialize video writer for output.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            # Create output directory if it doesn't exist
            output_dir = Path(self.output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Define codec and create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            self.writer = cv2.VideoWriter(
                self.output_path,
                fourcc,
                self.fps,
                (self.width, self.height)
            )
            
            if not self.writer.isOpened():
                logger.error(f"Failed to initialize video writer: {self.output_path}")
                return False
            
            logger.info(f"Output video writer initialized: {self.output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing video writer: {str(e)}")
            return False
    
    def process_frame(self, frame: np.ndarray, frame_number: int) -> Tuple[np.ndarray, Dict]:
        """
        Process a single frame for PPE detection.
        
        Args:
            frame: Input frame
            frame_number: Frame number for tracking
            
        Returns:
            Tuple of (annotated_frame, detection_results)
        """
        try:
            start_time = time.time()
            
            # Run PPE detection
            result = run_inference_with_ppe_analysis(
                frame,
                draw=True,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold
            )
            
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Extract annotated frame
            if 'annotated_frame' in result and result['annotated_frame'] is not None:
                annotated_frame = result['annotated_frame']
            else:
                annotated_frame = frame
            
            # Add frame info overlay
            annotated_frame = self._add_frame_info(
                annotated_frame, result, frame_number, processing_time
            )
            
            # Update statistics
            self.total_detections += len(result.get('detections', []))
            self.total_violations += len(result.get('violations', []))
            
            # Store detection data if requested
            if self.save_detections:
                frame_data = {
                    "frame_number": frame_number,
                    "timestamp": frame_number / self.fps,
                    "processing_time_ms": processing_time * 1000,
                    "detections": result.get('detections', []),
                    "violations": result.get('violations', []),
                    "total_persons": result.get('total_persons', 0),
                    "total_violations": result.get('total_violations', 0)
                }
                self.detections_data.append(frame_data)
            
            return annotated_frame, result
            
        except Exception as e:
            logger.error(f"Error processing frame {frame_number}: {str(e)}")
            return frame, {"error": str(e), "violations": [], "detections": []}
    
    def _add_frame_info(self, frame: np.ndarray, result: Dict, frame_number: int, processing_time: float) -> np.ndarray:
        """
        Add frame information overlay to the processed frame.
        
        Args:
            frame: Input frame
            result: Detection results
            frame_number: Current frame number
            processing_time: Processing time for this frame
            
        Returns:
            Frame with information overlay
        """
        try:
            # Get frame dimensions
            height, width = frame.shape[:2]
            
            # Frame info
            progress = (frame_number / self.total_frames) * 100 if self.total_frames > 0 else 0
            violations_count = len(result.get('violations', []))
            persons_count = result.get('total_persons', 0)
            
            # Background for info
            cv2.rectangle(frame, (10, 10), (500, 120), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (500, 120), (255, 255, 255), 2)
            
            # Frame information
            cv2.putText(frame, f"Frame: {frame_number}/{self.total_frames} ({progress:.1f}%)",
                       (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.putText(frame, f"Persons: {persons_count} | Violations: {violations_count}",
                       (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.putText(frame, f"Processing: {processing_time*1000:.1f}ms",
                       (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.putText(frame, f"Conf: {self.conf_threshold} | IoU: {self.iou_threshold}",
                       (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Violation indicator
            if violations_count > 0:
                cv2.rectangle(frame, (width - 150, 10), (width - 10, 60), (0, 0, 255), -1)
                cv2.putText(frame, "VIOLATIONS", (width - 140, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error adding frame info: {str(e)}")
            return frame
    
    def process_video(self) -> bool:
        """
        Process the entire video file.
        
        Returns:
            bool: True if processing successful
        """
        try:
            # Initialize video capture and writer
            if not self.initialize_video_capture():
                return False
            
            if not self.initialize_video_writer():
                return False
            
            logger.info("Starting video processing...")
            start_time = time.time()
            
            # Progress bar
            progress_bar = tqdm(total=self.total_frames, desc="Processing frames", unit="frame")
            
            frame_number = 0
            
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    break
                
                frame_number += 1
                
                # Skip frames if specified
                if self.skip_frames > 0 and (frame_number - 1) % (self.skip_frames + 1) != 0:
                    progress_bar.update(1)
                    continue
                
                # Process frame
                annotated_frame, result = self.process_frame(frame, frame_number)
                
                # Write frame to output video
                if self.writer:
                    self.writer.write(annotated_frame)
                
                self.processed_frames += 1
                progress_bar.update(1)
                
                # Print progress every 100 frames
                if frame_number % 100 == 0:
                    avg_time = np.mean(self.processing_times[-100:]) if self.processing_times else 0
                    logger.info(f"Processed {frame_number}/{self.total_frames} frames (avg: {avg_time*1000:.1f}ms/frame)")
            
            progress_bar.close()
            
            # Calculate final statistics
            total_time = time.time() - start_time
            avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
            
            logger.info(f"Video processing completed in {total_time:.2f}s")
            logger.info(f"Processed {self.processed_frames} frames")
            logger.info(f"Average processing time: {avg_processing_time*1000:.1f}ms per frame")
            logger.info(f"Total detections: {self.total_detections}")
            logger.info(f"Total violations: {self.total_violations}")
            
            # Save detection data if requested
            if self.save_detections:
                self._save_detection_data()
            
            return True
            
        except Exception as e:
            logger.error(f"Error during video processing: {str(e)}")
            return False
            
        finally:
            self.cleanup()
    
    def _save_detection_data(self) -> None:
        """
        Save detection data to JSON file.
        """
        try:
            output_json = self.output_path.replace('.mp4', '_detections.json')
            
            summary = {
                "video_info": {
                    "input_path": self.input_path,
                    "output_path": self.output_path,
                    "total_frames": self.total_frames,
                    "processed_frames": self.processed_frames,
                    "fps": self.fps,
                    "resolution": [self.width, self.height]
                },
                "processing_info": {
                    "conf_threshold": self.conf_threshold,
                    "iou_threshold": self.iou_threshold,
                    "average_processing_time_ms": np.mean(self.processing_times) * 1000 if self.processing_times else 0,
                    "total_detections": self.total_detections,
                    "total_violations": self.total_violations
                },
                "frame_data": self.detections_data
            }
            
            with open(output_json, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"Detection data saved to: {output_json}")
            
        except Exception as e:
            logger.error(f"Error saving detection data: {str(e)}")
    
    def cleanup(self) -> None:
        """
        Clean up resources.
        """
        if self.cap:
            self.cap.release()
        
        if self.writer:
            self.writer.release()
        
        cv2.destroyAllWindows()


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Video PPE Detection Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process video file
  python scripts/video_infer.py --input video.mp4 --output output.mp4
  
  # Process with custom thresholds
  python scripts/video_infer.py --input video.mp4 --output output.mp4 --conf 0.4 --iou 0.6
  
  # Process and save detection data
  python scripts/video_infer.py --input video.mp4 --output output.mp4 --save_detections
  
  # Process RTSP stream
  python scripts/video_infer.py --input rtsp://192.168.1.15:554/stream1 --output stream_output.mp4
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input video file path or stream URL'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output video file path'
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
        help='IoU threshold for NMS (0.0-1.0, default: 0.5)'
    )
    
    parser.add_argument(
        '--save_detections',
        action='store_true',
        help='Save detection data to JSON file'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size for processing (default: 1)'
    )
    
    parser.add_argument(
        '--skip_frames',
        type=int,
        default=0,
        help='Number of frames to skip between processing (default: 0)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not (0.0 <= args.conf <= 1.0):
        parser.error("Confidence threshold must be between 0.0 and 1.0")
    
    if not (0.0 <= args.iou <= 1.0):
        parser.error("IoU threshold must be between 0.0 and 1.0")
    
    if args.batch_size < 1:
        parser.error("Batch size must be >= 1")
    
    if args.skip_frames < 0:
        parser.error("Skip frames must be >= 0")
    
    return args


def main() -> None:
    """
    Main function for video processing.
    """
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Display configuration
        print("\n" + "="*60)
        print("PPE Detection System - Video Processing")
        print("="*60)
        print(f"Input: {args.input}")
        print(f"Output: {args.output}")
        print(f"Confidence threshold: {args.conf}")
        print(f"IoU threshold: {args.iou}")
        print(f"Save detections: {args.save_detections}")
        print(f"Batch size: {args.batch_size}")
        print(f"Skip frames: {args.skip_frames}")
        print("="*60 + "\n")
        
        # Create processor
        processor = VideoProcessor(
            input_path=args.input,
            output_path=args.output,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            save_detections=args.save_detections,
            batch_size=args.batch_size,
            skip_frames=args.skip_frames
        )
        
        # Process video
        success = processor.process_video()
        
        if success:
            print(f"\n✅ Video processing completed successfully!")
            print(f"Output saved to: {args.output}")
            if args.save_detections:
                print(f"Detection data saved to: {args.output.replace('.mp4', '_detections.json')}")
        else:
            print("\n❌ Video processing failed. Check logs for details.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()