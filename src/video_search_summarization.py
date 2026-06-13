#!/usr/bin/env python3
"""
Video Search and Summarization Module using CLIP and GStreamer (GPU-only)

Provides capabilities for:
1. Extracting keyframes from video streams using GStreamer (hardware accelerated)
2. GPU-accelerated CLIP encoding (CUDA)
3. Searching video content by text queries
4. Summarizing video content based on visual analysis
5. Temporal tracking of content across video

NOTE: GPU-only, GStreamer-only implementation. Requires NVIDIA GPU and CUDA support.

Usage:
    python3 video_search_summarization.py --video input.mp4 --extract-frames --output frames/
    python3 video_search_summarization.py --video input.mp4 --search "person walking" --top-k 5
    python3 video_search_summarization.py --video input.mp4 --summarize
"""

import os
import sys
import argparse
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import threading
import queue
from dataclasses import dataclass
import io

# GStreamer imports (REQUIRED)
try:
    import gi
    gi.require_version('Gst', '1.0')
    gi.require_version('GstVideo', '1.0')
    from gi.repository import Gst, GObject, GstVideo, GLib
    HAS_GSTREAMER = True
except Exception as e:
    print(f"FATAL: GStreamer required but not available: {e}", file=sys.stderr)
    print("Install with: apt-get install python3-gi python3-gst-1.0 libgstreamer1.0-0", file=sys.stderr)
    sys.exit(1)

# CLIP imports (REQUIRED)
try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image
    HAS_CLIP = True
except ImportError as e:
    print(f"FATAL: CLIP required but not available: {e}", file=sys.stderr)
    print("Install with: pip install transformers torch pillow", file=sys.stderr)
    sys.exit(1)

# Verify GPU availability
if not torch.cuda.is_available():
    print("WARNING: CUDA not detected. GPU acceleration disabled.", file=sys.stderr)
    DEVICE = "cpu"
else:
    DEVICE = "cuda"
    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}", file=sys.stderr)


class CLIPVideoAnalyzer:
    """GPU-only CLIP analyzer for video frames and text"""
    
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cuda"):
        if not HAS_CLIP:
            raise RuntimeError("CLIP required. Install with: pip install transformers torch pillow")
        
        # Force GPU
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("GPU (CUDA) required but not available. Install CUDA and torch with GPU support.")
        
        self.device = "cuda"  # Force GPU-only
        
        print(f"Loading CLIP model ({model_name}) on {self.device}...")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        
        # Memory optimization
        torch.cuda.empty_cache()
    
    def extract_frame_embeddings(self, image_paths: List[str], batch_size: int = 8) -> np.ndarray:
        """Extract CLIP embeddings from image files (GPU-batched)"""
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i+batch_size]
                images = []
                
                for img_path in batch_paths:
                    try:
                        image = Image.open(img_path).convert('RGB')
                        images.append(image)
                    except Exception as e:
                        print(f"Warning: Could not process {img_path}: {e}", file=sys.stderr)
                
                if images:
                    # Batch process on GPU
                    inputs = self.processor(images=images, return_tensors="pt").to(self.device)
                    image_features = self.model.get_image_features(**inputs)
                    # Normalize
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    embeddings.extend(image_features.cpu().numpy().astype(np.float32))
        
        torch.cuda.empty_cache()
        return np.array(embeddings) if embeddings else np.array([])
    
    def extract_text_embedding(self, text: str) -> np.ndarray:
        """Extract CLIP text embedding (GPU)"""
        with torch.no_grad():
            inputs = self.processor(text=text, return_tensors="pt").to(self.device)
            text_features = self.model.get_text_features(**inputs)
            # Normalize
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features[0].cpu().numpy().astype(np.float32)


@dataclass
class FrameData:
    """Frame data structure"""
    frame_id: int
    timestamp: float
    data: np.ndarray  # RGB frame
    width: int
    height: int


class GStreamerFrameExtractor:
    """GStreamer-based frame extraction with hardware acceleration"""
    
    def __init__(self, video_path: str, output_dir: str, fps: int = 1, max_frames: Optional[int] = None):
        """
        Initialize frame extractor using GStreamer
        
        Args:
            video_path: Path to video file
            output_dir: Output directory for frames
            fps: Frame extraction rate
            max_frames: Maximum frames to extract
        """
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.max_frames = max_frames
        self.frames = []
        self.frame_queue = queue.Queue(maxsize=30)
        self.stop_flag = False
    
    def extract_gstreamer(self) -> List[Tuple[int, str, float]]:
        """Extract frames using GStreamer hardware acceleration"""
        if not HAS_GSTREAMER:
            raise RuntimeError("GStreamer required. Install: apt-get install python3-gst-1.0")
        
        Gst.init(None)
        
        # GStreamer pipeline with hardware video decoder
        # filesrc → qtdemux → h264parse → nvv4l2decoder → nvvideoconvert → appsink
        pipeline_str = (
            f'filesrc location="{self.video_path}" ! '
            'qtdemux ! h264parse ! nvv4l2decoder ! '
            'nvvideoconvert ! video/x-raw,format=RGB ! '
            f'videorate ! video/x-raw,framerate={self.fps}/1 ! '
            'videoconvert ! video/x-raw,format=RGB ! '
            'appsink name=sink emit-signals=True'
        )
        
        print(f"GStreamer pipeline (hardware-accelerated):")
        print(f"  filesrc → qtdemux → h264parse → nvv4l2decoder → nvvideoconvert → appsink")
        print(f"  Frame rate: {self.fps} fps")
        
        try:
            pipeline = Gst.parse_launch(pipeline_str)
            sink = pipeline.get_by_name('sink')
            
            frame_count = [0]
            
            def on_new_sample(sink):
                """GStreamer appsink callback"""
                sample = sink.emit("pull-sample")
                if sample is None:
                    return Gst.FlowReturn.ERROR
                
                # Get frame data
                buffer = sample.get_buffer()
                caps = sample.get_caps()
                structure = caps.get_structure(0)
                
                width = structure.get_int("width")[1]
                height = structure.get_int("height")[1]
                
                # Copy buffer to numpy
                result, mapinfo = buffer.map(Gst.MapFlags.READ)
                if result:
                    frame_data = np.ndarray(
                        shape=(height, width, 3),
                        dtype=np.uint8,
                        buffer=mapinfo.data
                    ).copy()
                    buffer.unmap(mapinfo)
                    
                    frame_id = frame_count[0]
                    
                    # Save frame
                    frame_path = self.output_dir / f"frame_{frame_id:06d}.png"
                    
                    # Convert numpy to PIL and save
                    from PIL import Image as PILImage
                    img = PILImage.fromarray(frame_data, 'RGB')
                    img.save(str(frame_path))
                    
                    timestamp = frame_id / self.fps
                    self.frames.append((frame_id, str(frame_path), timestamp))
                    
                    if frame_id % 10 == 0:
                        print(f"  Extracted {frame_id} frames...")
                    
                    frame_count[0] += 1
                    
                    if self.max_frames and frame_count[0] >= self.max_frames:
                        return Gst.FlowReturn.EOS
                
                return Gst.FlowReturn.OK
            
            sink.connect("new-sample", on_new_sample)
            
            # Run pipeline
            bus = pipeline.get_bus()
            pipeline.set_state(Gst.State.PLAYING)
            
            print(f"Processing video with GStreamer...")
            
            # Wait for end of stream
            while True:
                message = bus.timed_pop_filtered(Gst.CLOCK_TIME_NONE, 
                                                  Gst.MessageType.ERROR | 
                                                  Gst.MessageType.EOS)
                
                if message:
                    if message.type == Gst.MessageType.ERROR:
                        err, debug = message.parse_error()
                        print(f"GStreamer error: {err.message}", file=sys.stderr)
                        break
                    elif message.type == Gst.MessageType.EOS:
                        print("GStreamer pipeline finished")
                        break
            
            pipeline.set_state(Gst.State.NULL)
            
        except Exception as e:
            print(f"GStreamer pipeline error: {e}", file=sys.stderr)
            raise
        
        print(f"✓ Extracted {len(self.frames)} frames from video")
        return self.frames


class VideoSearchEngine:
    """Search video content by text queries"""
    
    def __init__(self, frame_embeddings: np.ndarray, frame_metadata: List[Dict]):
        """
        Initialize search engine
        
        Args:
            frame_embeddings: Array of shape (N, D) with frame embeddings
            frame_metadata: List of dicts with frame info {frame_id, path, timestamp}
        """
        self.embeddings = frame_embeddings
        self.metadata = frame_metadata
    
    def search_by_text(self, text_query: str, text_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Search video for content matching text query"""
        if len(self.embeddings) == 0:
            return []
        
        # Compute cosine similarity
        similarities = np.dot(self.embeddings, text_embedding)
        
        # Get top-k results
        top_indices = np.argsort(-similarities)[:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'rank': len(results) + 1,
                'frame_id': self.metadata[idx]['frame_id'],
                'timestamp': self.metadata[idx]['timestamp'],
                'path': self.metadata[idx]['path'],
                'similarity': float(similarities[idx])
            })
        
        return results


class VideoSummarizer:
    """Summarize video content based on visual analysis"""
    
    def __init__(self, frame_embeddings: np.ndarray, frame_metadata: List[Dict]):
        """
        Initialize summarizer
        
        Args:
            frame_embeddings: Array of shape (N, D) with frame embeddings
            frame_metadata: List of dicts with frame info
        """
        self.embeddings = frame_embeddings
        self.metadata = frame_metadata
    
    def select_keyframes(self, num_keyframes: int = 5) -> List[Dict]:
        """Select diverse keyframes that represent the video"""
        if len(self.embeddings) == 0:
            return []
        
        # Use clustering to find diverse frames
        # Simple approach: select frames that are most different from each other
        selected_indices = [0]  # Always include first frame
        
        for i in range(1, num_keyframes):
            # Find frame most different from selected ones
            max_min_dist = -1
            best_idx = i
            
            for candidate_idx in range(len(self.embeddings)):
                if candidate_idx in selected_indices:
                    continue
                
                # Minimum distance to any selected frame
                min_dist = float('inf')
                for selected_idx in selected_indices:
                    dist = np.linalg.norm(
                        self.embeddings[candidate_idx] - self.embeddings[selected_idx]
                    )
                    min_dist = min(min_dist, dist)
                
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = candidate_idx
            
            if best_idx not in selected_indices:
                selected_indices.append(best_idx)
        
        # Build results
        results = []
        for i, idx in enumerate(sorted(selected_indices)[:num_keyframes]):
            results.append({
                'keyframe_id': i,
                'frame_id': self.metadata[idx]['frame_id'],
                'timestamp': self.metadata[idx]['timestamp'],
                'path': self.metadata[idx]['path']
            })
        
        return results
    
    def analyze_content_distribution(self) -> Dict:
        """Analyze temporal distribution of content"""
        if len(self.embeddings) < 2:
            return {}
        
        # Compute pairwise differences to detect scene changes
        differences = []
        for i in range(1, len(self.embeddings)):
            diff = np.linalg.norm(self.embeddings[i] - self.embeddings[i-1])
            differences.append(diff)
        
        differences = np.array(differences)
        
        return {
            'total_frames': len(self.embeddings),
            'avg_frame_difference': float(np.mean(differences)),
            'max_frame_difference': float(np.max(differences)),
            'min_frame_difference': float(np.min(differences)),
            'std_frame_difference': float(np.std(differences))
        }


def main():
    parser = argparse.ArgumentParser(
        description="Video Search and Summarization using CLIP and GStreamer"
    )
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--extract-frames", action="store_true", help="Extract frames from video")
    parser.add_argument("--output", default="./frames", help="Output directory for frames")
    parser.add_argument("--fps", type=int, default=1, help="Frame extraction rate")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to extract")
    parser.add_argument("--search", help="Search video by text query")
    parser.add_argument("--summarize", action="store_true", help="Summarize video content")
    parser.add_argument("--keyframes", type=int, default=5, help="Number of keyframes for summary")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32", help="CLIP model name")
    parser.add_argument("--embeddings-output", help="Save frame embeddings to file (NPY format)")
    parser.add_argument("--metadata-output", help="Save frame metadata to file (JSON format)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results for search")
    
    args = parser.parse_args()
    
    # Validate input
    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}", file=sys.stderr)
        sys.exit(1)
    
    # Initialize CLIP analyzer (GPU-only)
    print(f"\n{'='*70}")
    print(f"AIVisionStreamAnalytics - Video Search & Summarization (GPU-Only)")
    print(f"{'='*70}")
    
    try:
        analyzer = CLIPVideoAnalyzer(args.clip_model)
    except RuntimeError as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Extract frames if requested
    if args.extract_frames:
        print(f"\n[1/3] Extracting frames from {args.video}...")
        extractor = GStreamerFrameExtractor(args.video, args.output, args.fps, args.max_frames)
        frame_info = extractor.extract_gstreamer()
        
        if frame_info:
            print(f"\n[2/3] Extracting CLIP embeddings ({len(frame_info)} frames)...")
            frame_paths = [info[1] for info in frame_info]
            embeddings = analyzer.extract_frame_embeddings(frame_paths)
            
            # Prepare metadata
            metadata = [
                {'frame_id': info[0], 'path': info[1], 'timestamp': info[2]}
                for info in frame_info
            ]
            
            # Save embeddings and metadata if requested
            if args.embeddings_output:
                np.save(args.embeddings_output, embeddings)
                print(f"✓ Saved embeddings to {args.embeddings_output} ({embeddings.nbytes/1e6:.1f}MB)")
            
            if args.metadata_output:
                with open(args.metadata_output, 'w') as f:
                    json.dump(metadata, f, indent=2)
                print(f"✓ Saved metadata to {args.metadata_output}")
    
    # Search video
    if args.search:
        print(f"\n[1/2] Processing video for search...")
        extractor = GStreamerFrameExtractor(args.video, args.output, 1)
        frame_info = extractor.extract_gstreamer()
        
        if frame_info:
            print(f"\n[2/2] Searching for '{args.search}'...")
            frame_paths = [info[1] for info in frame_info]
            embeddings = analyzer.extract_frame_embeddings(frame_paths)
            metadata = [
                {'frame_id': info[0], 'path': info[1], 'timestamp': info[2]}
                for info in frame_info
            ]
            
            # Search
            search_engine = VideoSearchEngine(embeddings, metadata)
            text_embedding = analyzer.extract_text_embedding(args.search)
            results = search_engine.search_by_text(args.search, text_embedding, args.top_k)
            
            # Print results
            print(f"\n{'Rank':<6}{'Frame ID':<12}{'Timestamp':<12}{'Similarity':<12}{'Path':<40}")
            print("-" * 90)
            for result in results:
                print(f"{result['rank']:<6}{result['frame_id']:<12}{result['timestamp']:<12.2f}"
                      f"{result['similarity']:<12.4f}{Path(result['path']).name:<40}")
    
    # Summarize video
    if args.summarize:
        print(f"\n[1/2] Analyzing video for summarization...")
        extractor = GStreamerFrameExtractor(args.video, args.output, 2)
        frame_info = extractor.extract_gstreamer()
        
        if frame_info:
            print(f"\n[2/2] Generating summary...")
            frame_paths = [info[1] for info in frame_info]
            embeddings = analyzer.extract_frame_embeddings(frame_paths)
            metadata = [
                {'frame_id': info[0], 'path': info[1], 'timestamp': info[2]}
                for info in frame_info
            ]
            
            summarizer = VideoSummarizer(embeddings, metadata)
            
            # Get keyframes
            keyframes = summarizer.select_keyframes(args.keyframes)
            print(f"\nSelected {len(keyframes)} keyframes:")
            print(f"{'ID':<6}{'Frame':<10}{'Timestamp':<12}{'Path':<40}")
            print("-" * 70)
            for kf in keyframes:
                print(f"{kf['keyframe_id']:<6}{kf['frame_id']:<10}{kf['timestamp']:<12.2f}"
                      f"{Path(kf['path']).name:<40}")
            
            # Analyze content
            analysis = summarizer.analyze_content_distribution()
            print(f"\nContent Distribution Analysis:")
            print(f"  Total frames: {analysis['total_frames']}")
            print(f"  Avg frame difference: {analysis['avg_frame_difference']:.4f}")
            print(f"  Max frame difference: {analysis['max_frame_difference']:.4f}")
            print(f"  Std frame difference: {analysis['std_frame_difference']:.4f}")
    
    print(f"\n{'='*70}")
    print(f"✓ Processing complete (GPU: {torch.cuda.get_device_name(0)})")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
