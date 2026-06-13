#!/usr/bin/env python3
"""
Vector Similarity Search Utility

Provides similarity search capabilities for ReID embeddings using FAISS or file-based search.
Supports finding similar tracks and analyzing re-identification patterns.

Usage:
    python3 vector_search.py --top-k 5 --track-id 42
    python3 vector_search.py --query embedding.npy --top-k 10
    python3 vector_search.py --compare 42 100
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
import json

# Try to import FAISS
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("Warning: FAISS not available. Using file-based search only.", file=sys.stderr)

# Try to import CLIP for text encoding
try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False
    print("Warning: CLIP not available. Text queries disabled.", file=sys.stderr)
 



class CLIPTextEncoder:
    """CLIP text encoder for converting text queries to embeddings"""
    
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cuda" if torch.cuda.is_available() else "cpu"):
        if not HAS_CLIP:
            raise RuntimeError("CLIP not installed. Install with: pip install transformers torch")
        
        print(f"Loading CLIP model ({model_name}) on {device}...")
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
    
    def encode_text(self, texts):
        """Convert text(s) to embeddings"""
        if isinstance(texts, str):
            texts = [texts]
        
        with torch.no_grad():
            inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
            text_embeddings = self.model.get_text_features(**inputs)
            # Normalize embeddings
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        
        if len(texts) == 1:
            return text_embeddings[0].cpu().numpy().astype(np.float32)
        return text_embeddings.cpu().numpy().astype(np.float32)


class VectorDatabase:
    """File-based vector database reader"""
    
    def __init__(self, db_path, index_path=None, metadata_path=None):
        print(f"File path {db_path}")
        self.db_path = Path(db_path)
        self.index_path = Path(index_path) if index_path else None
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.vectors = []
        self.metadata = {}
        self.index = None
        self.load()
    
    def load(self):
        """Load vectors from database"""
        if self.db_path.exists():
            print(f"Loading vectors from {self.db_path}...")
            with open(self.db_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        parts = line.strip().split('|')
                        if len(parts) >= 6:
                            track_id = int(parts[0])
                            frame_id = int(parts[1])
                            class_id = int(parts[2])
                            confidence = float(parts[3])
                            timestamp = parts[4]
                            embedding = np.array([float(x) for x in parts[5].split(',')])
                            
                            self.vectors.append(embedding)
                            self.metadata[len(self.vectors) - 1] = {
                                'track_id': track_id,
                                'frame_id': frame_id,
                                'class_id': class_id,
                                'confidence': confidence,
                                'timestamp': timestamp
                            }
                    except Exception as e:
                        print(f"Warning: Error parsing line {line_num}: {e}", file=sys.stderr)
        
        if HAS_FAISS and self.index_path and self.index_path.exists():
            print(f"Loading FAISS index from {self.index_path}...")
            try:
                self.index = faiss.read_index(str(self.index_path))
            except Exception as e:
                print(f"Warning: Could not load FAISS index: {e}", file=sys.stderr)
        
        print(f"Loaded {len(self.vectors)} vectors")
    
    def get_vector_by_track_id(self, track_id):
        """Get all vectors for a track ID"""
        results = []
        for idx, meta in self.metadata.items():
            if meta['track_id'] == track_id:
                results.append((idx, self.vectors[idx], meta))
        return results
    
    def similarity_search_faiss(self, query_vector, k=5):
        """Search using FAISS (fast)"""
        if not self.index or len(self.vectors) == 0:
            return []
        
        query = np.array([query_vector], dtype=np.float32)
        distances, indices = self.index.search(query, min(k, len(self.vectors)))
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.vectors):
                results.append((dist, idx, self.vectors[idx], self.metadata[idx]))
        
        return results
    
    def similarity_search_linear(self, query_vector, k=5):
        """Search using linear scan (slower but always available)"""
        if len(self.vectors) == 0:
            return []
        
        query = np.array(query_vector)
        distances = []
        
        for idx, vec in enumerate(self.vectors):
            # Euclidean distance
            dist = np.sqrt(np.sum((query - vec) ** 2))
            distances.append((dist, idx))
        
        distances.sort(key=lambda x: x[0])
        
        results = []
        for dist, idx in distances[:k]:
            results.append((dist, idx, self.vectors[idx], self.metadata[idx]))
        
        return results
    
    def similarity_search(self, query_vector, k=5):
        """Search with best available method"""
        if HAS_FAISS and self.index:
            return self.similarity_search_faiss(query_vector, k)
        else:
            return self.similarity_search_linear(query_vector, k)


def print_results(results, method="similarity"):
    """Pretty print search results"""
    print(f"\n{'Rank':<6}{'Distance':<12}{'Track ID':<12}{'Frame':<8}{'Confidence':<12}{'Timestamp':<20}")
    print("-" * 80)
    
    for rank, (dist, idx, vec, meta) in enumerate(results, 1):
        print(f"{rank:<6}{dist:<12.4f}{meta['track_id']:<12}{meta['frame_id']:<8}"
              f"{meta['confidence']:<12.4f}{meta['timestamp']:<20}")


def main():
    parser = argparse.ArgumentParser(description="Vector similarity search utility with NLP text query support and video search")
    parser.add_argument("--db", default="../data/vectors_meta.txt", help="Vector metadata path (text lines)")
    parser.add_argument("--index", default="../data/vectors.faiss", help="FAISS index path")
    parser.add_argument("--metadata", default="../data/vectors_meta.txt", help="FAISS metadata path")
    parser.add_argument("--track-id", type=int, help="Find all vectors for a track ID")
    parser.add_argument("--query", help="Query vector file (NPY format)")
    parser.add_argument("--text-query", help="Natural language text query (uses CLIP encoder)")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32", help="CLIP model name")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("--compare", type=int, nargs=2, metavar=("TRACK1", "TRACK2"), 
                       help="Compare average vectors of two tracks")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    
    # Video search options
    parser.add_argument("--video-search", help="Search video by text query (uses CLIP vision model)")
    parser.add_argument("--video", help="Path to video file for search/summarization")
    parser.add_argument("--summarize", action="store_true", help="Summarize video content")
    parser.add_argument("--keyframes", type=int, default=5, help="Number of keyframes for summary")
    parser.add_argument("--fps", type=int, default=1, help="Frame extraction rate for video")
    parser.add_argument("--output-frames", help="Output directory for extracted frames")
    parser.add_argument("--frame-embeddings", help="Save frame embeddings to file")
    parser.add_argument("--frame-metadata", help="Save frame metadata to file")
    
    args = parser.parse_args()
    
    # Load database
    db = VectorDatabase(args.db, args.index, args.metadata)
    
    if args.stats:
        print(f"\nDatabase Statistics:")
        print(f"  Total vectors: {len(db.vectors)}")
        if len(db.vectors) > 0:
            print(f"  Vector dimension: {len(db.vectors[0])}")
            track_ids = set(meta['track_id'] for meta in db.metadata.values())
            print(f"  Unique tracks: {len(track_ids)}")
    
    elif args.text_query:
        print(f"\nText Query: '{args.text_query}'")
        try:
            if not HAS_CLIP:
                print("Error: CLIP (torch) not available. Install with: pip install transformers torch", file=sys.stderr)
                sys.exit(1)
            encoder = CLIPTextEncoder(args.clip_model)

            query_embedding = encoder.encode_text(args.text_query)
            print(f"Encoded text to embedding (dim={len(query_embedding)})")

            results = db.similarity_search(query_embedding, args.top_k)
            print_results(results)
        except Exception as e:
            print(f"Error processing text query: {e}", file=sys.stderr)
            sys.exit(1)
    
    elif args.track_id is not None:
        print(f"\nVectors for Track ID {args.track_id}:")
        results = db.get_vector_by_track_id(args.track_id)
        if results:
            print_results([(0, idx, vec, meta) for idx, vec, meta in results], "track")
            
            # Compute average embedding
            avg_embedding = np.mean([vec for _, vec, _ in results], axis=0)
            print(f"\nAverage embedding computed (dim={len(avg_embedding)})")
            
            # Find similar tracks
            print(f"\nFinding similar tracks...")
            similar = db.similarity_search(avg_embedding, args.top_k)
            print_results(similar)
        else:
            print(f"No vectors found for track {args.track_id}")
    
    elif args.query:
        print(f"\nLoading query vector from {args.query}")
        query_vec = np.load(args.query)
        print(f"Query vector shape: {query_vec.shape}")
        
        results = db.similarity_search(query_vec, args.top_k)
        print_results(results)
    
    elif args.compare:
        track1, track2 = args.compare
        print(f"\nComparing tracks {track1} and {track2}...")
        
        vecs1 = db.get_vector_by_track_id(track1)
        vecs2 = db.get_vector_by_track_id(track2)
        
        if vecs1 and vecs2:
            avg1 = np.mean([vec for _, vec, _ in vecs1], axis=0)
            avg2 = np.mean([vec for _, vec, _ in vecs2], axis=0)
            
            distance = np.sqrt(np.sum((avg1 - avg2) ** 2))
            similarity = 1.0 / (1.0 + distance)
            
            print(f"  Track {track1}: {len(vecs1)} vectors")
            print(f"  Track {track2}: {len(vecs2)} vectors")
            print(f"  Euclidean distance: {distance:.4f}")
            print(f"  Similarity score: {similarity:.4f}")
        else:
            print(f"Could not find vectors for one or both tracks")
    
    elif args.video and (args.video_search or args.summarize):
        # Video search and summarization using CLIP
        print(f"\n=== Video Search & Summarization ===")
        print(f"Video: {args.video}")
        
        if not HAS_CLIP:
            print("Error: CLIP not available. Install with: pip install transformers torch pillow", file=sys.stderr)
            sys.exit(1)
        
        if not Path(args.video).exists():
            print(f"Error: Video file not found: {args.video}", file=sys.stderr)
            sys.exit(1)
        
        # Import video search module
        sys.path.insert(0, '../src')
        try:
            from video_search_summarization import CLIPVideoAnalyzer, GStreamerFrameExtractor, VideoSearchEngine, VideoSummarizer
        except ImportError:
            print("Error: video_search_summarization module not found. Make sure it's in ../src/", file=sys.stderr)
            sys.exit(1)
        
        # Setup output directory
        output_dir = args.output_frames or "./video_frames"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Extract frames
        print(f"Extracting frames at {args.fps} fps...")
        extractor = GStreamerFrameExtractor(args.video, output_dir, args.fps)
        frame_info = extractor.extract_opencv()
        
        if not frame_info:
            print("Error: Could not extract frames from video", file=sys.stderr)
            sys.exit(1)
        
        print(f"Extracted {len(frame_info)} frames")
        
        # Extract CLIP embeddings
        print(f"Extracting CLIP embeddings...")
        analyzer = CLIPVideoAnalyzer(args.clip_model)
        frame_paths = [info[1] for info in frame_info]
        embeddings = analyzer.extract_frame_embeddings(frame_paths)
        
        if len(embeddings) == 0:
            print("Error: Could not extract embeddings", file=sys.stderr)
            sys.exit(1)
        
        print(f"Extracted {len(embeddings)} embeddings")
        
        # Prepare metadata
        metadata = [
            {'frame_id': info[0], 'path': info[1], 'timestamp': info[2]}
            for info in frame_info
        ]
        
        # Save embeddings if requested
        if args.frame_embeddings:
            np.save(args.frame_embeddings, embeddings)
            print(f"Saved embeddings to {args.frame_embeddings}")
        
        if args.frame_metadata:
            import json
            with open(args.frame_metadata, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Saved metadata to {args.frame_metadata}")
        
        # Video search
        if args.video_search:
            print(f"\nSearching for: '{args.video_search}'")
            search_engine = VideoSearchEngine(embeddings, metadata)
            text_embedding = analyzer.extract_text_embedding(args.video_search)
            results = search_engine.search_by_text(args.video_search, text_embedding, args.top_k)
            
            print(f"\n{'Rank':<6}{'Frame ID':<12}{'Time (s)':<12}{'Similarity':<12}")
            print("-" * 45)
            for result in results:
                print(f"{result['rank']:<6}{result['frame_id']:<12}{result['timestamp']:<12.2f}"
                      f"{result['similarity']:<12.4f}")
        
        # Video summarization
        if args.summarize:
            print(f"\nSummarizing video with {args.keyframes} keyframes...")
            summarizer = VideoSummarizer(embeddings, metadata)
            
            # Get keyframes
            keyframes = summarizer.select_keyframes(args.keyframes)
            print(f"\nKeyframes:")
            print(f"{'ID':<6}{'Frame':<10}{'Time (s)':<12}{'File':<40}")
            print("-" * 70)
            for kf in keyframes:
                frame_name = Path(kf['path']).name
                print(f"{kf['keyframe_id']:<6}{kf['frame_id']:<10}{kf['timestamp']:<12.2f}"
                      f"{frame_name:<40}")
            
            # Content analysis
            analysis = summarizer.analyze_content_distribution()
            print(f"\nContent Analysis:")
            print(f"  Total frames: {analysis['total_frames']}")
            print(f"  Avg frame difference: {analysis['avg_frame_difference']:.4f}")
            print(f"  Max frame difference: {analysis['max_frame_difference']:.4f}")
            print(f"  Std frame difference: {analysis['std_frame_difference']:.4f}")
    
    else:
        print(f"Database info:")
        print(f"  Path: {args.db}")
        print(f"  Vectors: {len(db.vectors)}")
        print(f"  FAISS available: {HAS_FAISS}")
        print(f"  CLIP available: {HAS_CLIP}")
        print(f"\nUse --help for available options")
        print(f"\nExample text query:")
        print(f"  python3 vector_search.py --text-query 'person wearing red shirt' --top-k 10")
        print(f"\nExample video search:")
        print(f"  python3 vector_search.py --video video.mp4 --video-search 'person running'")
        print(f"\nExample video summarization:")
        print(f"  python3 vector_search.py --video video.mp4 --summarize --keyframes 8")


if __name__ == "__main__":
    main()
