#!/usr/bin/env python3
"""
Example: Video Search and Summarization

Demonstrates how to use the video search and summarization capabilities
to analyze video content using CLIP embeddings.
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from video_search_summarization import (
    CLIPVideoAnalyzer,
    GStreamerFrameExtractor,
    VideoSearchEngine,
    VideoSummarizer
)
import numpy as np
import json


def example_basic_search():
    """Example 1: Basic video search"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Video Search")
    print("="*60)
    
    video_path = "sample_video.mp4"
    
    # Check if sample video exists
    if not Path(video_path).exists():
        print(f"Note: Sample video '{video_path}' not found.")
        print("Create a sample video first for this example.")
        return
    
    # Initialize CLIP analyzer
    print("\n1. Loading CLIP model...")
    analyzer = CLIPVideoAnalyzer(model_name="openai/clip-vit-base-patch32")
    
    # Extract frames
    print("\n2. Extracting frames from video...")
    extractor = GStreamerFrameExtractor(video_path, "./example_frames", fps=1, max_frames=50)
    frame_info = extractor.extract_opencv()
    print(f"   Extracted {len(frame_info)} frames")
    
    # Get embeddings
    print("\n3. Extracting CLIP embeddings...")
    frame_paths = [info[1] for info in frame_info]
    embeddings = analyzer.extract_frame_embeddings(frame_paths)
    metadata = [
        {'frame_id': info[0], 'path': info[1], 'timestamp': info[2]}
        for info in frame_info
    ]
    print(f"   Extracted {len(embeddings)} embeddings (shape: {embeddings.shape})")
    
    # Search
    print("\n4. Searching for 'person standing'...")
    search_engine = VideoSearchEngine(embeddings, metadata)
    text_embedding = analyzer.extract_text_embedding("person standing")
    results = search_engine.search_by_text("person standing", text_embedding, top_k=3)
    
    print("\n   Results:")
    for result in results:
        print(f"     #{result['rank']}: Frame {result['frame_id']} @ {result['timestamp']:.1f}s "
              f"(similarity: {result['similarity']:.4f})")


def example_summarization():
    """Example 2: Video summarization"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Video Summarization")
    print("="*60)
    
    video_path = "sample_video.mp4"
    
    if not Path(video_path).exists():
        print(f"Note: Sample video '{video_path}' not found.")
        return
    
    analyzer = CLIPVideoAnalyzer()
    
    print("\n1. Extracting frames for summarization...")
    extractor = GStreamerFrameExtractor(video_path, "./summary_frames", fps=1, max_frames=100)
    frame_info = extractor.extract_opencv()
    
    print("\n2. Extracting embeddings...")
    frame_paths = [info[1] for info in frame_info]
    embeddings = analyzer.extract_frame_embeddings(frame_paths)
    metadata = [
        {'frame_id': info[0], 'path': info[1], 'timestamp': info[2]}
        for info in frame_info
    ]
    
    print("\n3. Generating summary...")
    summarizer = VideoSummarizer(embeddings, metadata)
    keyframes = summarizer.select_keyframes(num_keyframes=5)
    
    print("\n   Selected keyframes:")
    for kf in keyframes:
        print(f"     Frame {kf['frame_id']} @ {kf['timestamp']:.1f}s")
    
    print("\n4. Analyzing content distribution...")
    stats = summarizer.analyze_content_distribution()
    print(f"   Total frames: {stats['total_frames']}")
    print(f"   Avg difference: {stats['avg_frame_difference']:.4f}")
    print(f"   Max difference: {stats['max_frame_difference']:.4f}")


def example_multi_query():
    """Example 3: Multiple queries on same video"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Multiple Queries")
    print("="*60)
    
    video_path = "sample_video.mp4"
    
    if not Path(video_path).exists():
        print(f"Note: Sample video '{video_path}' not found.")
        return
    
    analyzer = CLIPVideoAnalyzer()
    
    print("\n1. Processing video...")
    extractor = GStreamerFrameExtractor(video_path, "./multi_query_frames", fps=2, max_frames=100)
    frame_info = extractor.extract_opencv()
    
    frame_paths = [info[1] for info in frame_info]
    embeddings = analyzer.extract_frame_embeddings(frame_paths)
    metadata = [
        {'frame_id': info[0], 'path': info[1], 'timestamp': info[2]}
        for info in frame_info
    ]
    
    search_engine = VideoSearchEngine(embeddings, metadata)
    
    # Multiple queries
    queries = [
        "person walking",
        "outdoor scene",
        "people talking"
    ]
    
    print(f"\n2. Running {len(queries)} queries...\n")
    
    for query in queries:
        print(f"   Query: '{query}'")
        text_emb = analyzer.extract_text_embedding(query)
        results = search_engine.search_by_text(query, text_emb, top_k=2)
        
        for result in results:
            print(f"     → Frame {result['frame_id']} @ {result['timestamp']:.1f}s "
                  f"(score: {result['similarity']:.4f})")
        print()


def example_embeddings_export():
    """Example 4: Export and reuse embeddings"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Export and Reuse Embeddings")
    print("="*60)
    
    video_path = "sample_video.mp4"
    embeddings_file = "./example_embeddings.npy"
    metadata_file = "./example_metadata.json"
    
    if not Path(video_path).exists():
        print(f"Note: Sample video '{video_path}' not found.")
        return
    
    analyzer = CLIPVideoAnalyzer()
    
    # First run: extract and save
    print("\n1. Extracting frames and embeddings...")
    extractor = GStreamerFrameExtractor(video_path, "./cache_frames", fps=1, max_frames=50)
    frame_info = extractor.extract_opencv()
    
    frame_paths = [info[1] for info in frame_info]
    embeddings = analyzer.extract_frame_embeddings(frame_paths)
    metadata = [
        {'frame_id': info[0], 'path': info[1], 'timestamp': info[2]}
        for info in frame_info
    ]
    
    print(f"   Extracted {len(embeddings)} embeddings")
    
    # Save
    print(f"\n2. Saving embeddings to {embeddings_file}...")
    np.save(embeddings_file, embeddings)
    
    print(f"   Saving metadata to {metadata_file}...")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f)
    
    # Second run: load and use
    print(f"\n3. Loading saved embeddings...")
    loaded_embeddings = np.load(embeddings_file)
    with open(metadata_file) as f:
        loaded_metadata = json.load(f)
    
    print(f"   Loaded {len(loaded_embeddings)} embeddings")
    
    # Search with loaded embeddings
    print(f"\n4. Searching with loaded embeddings...")
    search_engine = VideoSearchEngine(loaded_embeddings, loaded_metadata)
    text_emb = analyzer.extract_text_embedding("motion")
    results = search_engine.search_by_text("motion", text_emb, top_k=2)
    
    for result in results:
        print(f"   Frame {result['frame_id']} @ {result['timestamp']:.1f}s")


def main():
    """Run all examples"""
    print("\nVideo Search and Summarization Examples")
    print("========================================")
    
    examples = [
        ("Basic Search", example_basic_search),
        ("Summarization", example_summarization),
        ("Multiple Queries", example_multi_query),
        ("Embeddings Export", example_embeddings_export),
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    # Run all by default
    print("\nRunning all examples...\n")
    
    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\nError in {name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("Examples complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
