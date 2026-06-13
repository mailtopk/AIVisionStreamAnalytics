/**
 * @file video_search.h
 * @brief Video Search and Summarization capabilities for GStreamer pipeline
 * 
 * Provides frame-level CLIP embedding extraction for video search and summarization
 * Integrates with GStreamer pipeline to process frames in real-time
 */

#ifndef VIDEO_SEARCH_H
#define VIDEO_SEARCH_H

#include <vector>
#include <string>
#include <memory>
#include <queue>
#include <mutex>
#include <gst/gst.h>

/**
 * @struct FrameData
 * @brief Represents a video frame with metadata
 */
struct FrameData {
    uint32_t frame_id;              ///< Frame sequence number
    uint64_t timestamp_ms;          ///< Timestamp in milliseconds
    std::vector<uint8_t> frame_data; ///< Raw frame data (RGB)
    uint32_t width;                 ///< Frame width
    uint32_t height;                ///< Frame height
    
    FrameData() : frame_id(0), timestamp_ms(0), width(0), height(0) {}
};

/**
 * @struct FrameEmbedding
 * @brief CLIP embedding for a video frame
 */
struct FrameEmbedding {
    uint32_t frame_id;
    uint64_t timestamp_ms;
    std::vector<float> embedding;   ///< CLIP embedding vector (typically 512-dim)
    float confidence;               ///< Embedding confidence score
    
    FrameEmbedding() : frame_id(0), timestamp_ms(0), confidence(0.0f) {}
};

/**
 * @class FrameBuffer
 * @brief Thread-safe buffer for storing frames for processing
 */
class FrameBuffer {
private:
    std::queue<FrameData> m_queue;
    std::mutex m_mutex;
    size_t m_max_size;
    
public:
    FrameBuffer(size_t max_size = 30) : m_max_size(max_size) {}
    
    /**
     * Add frame to buffer
     * @param frame Frame data
     * @return true if added, false if buffer full
     */
    bool push(const FrameData& frame) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_queue.size() >= m_max_size) {
            return false;
        }
        m_queue.push(frame);
        return true;
    }
    
    /**
     * Get frame from buffer
     * @param frame Output frame data
     * @return true if frame available, false if empty
     */
    bool pop(FrameData& frame) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_queue.empty()) {
            return false;
        }
        frame = m_queue.front();
        m_queue.pop();
        return true;
    }
    
    /**
     * Get buffer size
     * @return Number of frames in buffer
     */
    size_t size() {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_queue.size();
    }
    
    /**
     * Clear buffer
     */
    void clear() {
        std::lock_guard<std::mutex> lock(m_mutex);
        while (!m_queue.empty()) {
            m_queue.pop();
        }
    }
};

/**
 * @class VideoFrameCapture
 * @brief Captures frames from GStreamer pipeline
 */
class VideoFrameCapture {
private:
    FrameBuffer& m_frame_buffer;
    uint32_t m_frame_count;
    uint32_t m_frame_stride;  ///< Process every Nth frame (1 = every frame)
    
public:
    VideoFrameCapture(FrameBuffer& buffer, uint32_t stride = 1)
        : m_frame_buffer(buffer), m_frame_count(0), m_frame_stride(stride) {}
    
    /**
     * Process frame from pipeline
     * @param buffer Pointer to frame buffer
     * @param width Frame width
     * @param height Frame height
     * @param timestamp_ms Timestamp in milliseconds
     */
    void processFrame(const uint8_t* buffer, uint32_t width, uint32_t height, uint64_t timestamp_ms) {
        if (m_frame_count % m_frame_stride != 0) {
            m_frame_count++;
            return;
        }
        
        FrameData frame;
        frame.frame_id = m_frame_count;
        frame.timestamp_ms = timestamp_ms;
        frame.width = width;
        frame.height = height;
        
        // Copy frame data
        size_t frame_size = width * height * 3; // Assuming RGB
        frame.frame_data.assign(buffer, buffer + frame_size);
        
        m_frame_buffer.push(frame);
        m_frame_count++;
    }
    
    /**
     * Get total frames processed
     */
    uint32_t getFrameCount() const {
        return m_frame_count;
    }
};

/**
 * @class VideoSearchConfig
 * @brief Configuration for video search capabilities
 */
class VideoSearchConfig {
public:
    bool enable_frame_capture{false};        ///< Enable frame capture for search
    uint32_t frame_capture_stride{30};       ///< Capture every Nth frame
    std::string frame_storage_dir{""};       ///< Directory to store keyframes
    bool enable_embeddings{false};           ///< Store frame embeddings
    std::string embeddings_output_path{""};  ///< Path to save embeddings
    
    bool validate() const {
        if (enable_frame_capture && frame_storage_dir.empty()) {
            return false;
        }
        return true;
    }
};

/**
 * @class VideoSearchEngine
 * @brief Engine for video search operations (Python-based)
 * 
 * Note: CLIP embedding extraction and search is implemented in Python
 * This class provides the interface for C++ pipeline integration
 */
class VideoSearchEngine {
private:
    std::string m_embedding_model;
    std::vector<FrameEmbedding> m_frame_embeddings;
    
public:
    VideoSearchEngine(const std::string& model_name = "openai/clip-vit-base-patch32")
        : m_embedding_model(model_name) {}
    
    /**
     * Extract CLIP embeddings from frames (delegates to Python)
     * @param frame_paths List of frame file paths
     * @return Number of embeddings extracted
     */
    size_t extractEmbeddings(const std::vector<std::string>& frame_paths);
    
    /**
     * Search for frames similar to query
     * @param query_text Text query
     * @param top_k Number of results
     * @return List of matching frame IDs with scores
     */
    std::vector<std::pair<uint32_t, float>> searchByText(
        const std::string& query_text, size_t top_k = 5
    );
    
    /**
     * Get frame embeddings
     */
    const std::vector<FrameEmbedding>& getEmbeddings() const {
        return m_frame_embeddings;
    }
    
    /**
     * Save embeddings to file
     * @param output_path Path to save embeddings
     */
    bool saveEmbeddings(const std::string& output_path);
    
    /**
     * Load embeddings from file
     * @param input_path Path to load embeddings
     */
    bool loadEmbeddings(const std::string& input_path);
};

/**
 * @class VideoSummarizer
 * @brief Summarizes video content based on frame analysis
 */
class VideoSummarizer {
private:
    const std::vector<FrameEmbedding>& m_embeddings;
    
public:
    VideoSummarizer(const std::vector<FrameEmbedding>& embeddings)
        : m_embeddings(embeddings) {}
    
    /**
     * Select keyframes representing the video
     * @param num_keyframes Number of keyframes to select
     * @return List of selected frame IDs
     */
    std::vector<uint32_t> selectKeyframes(size_t num_keyframes = 5);
    
    /**
     * Detect scene changes
     * @param threshold Similarity threshold for scene change detection
     * @return List of frame IDs where scene changes occur
     */
    std::vector<uint32_t> detectSceneChanges(float threshold = 0.5f);
    
    /**
     * Compute content statistics
     * @return Statistics about frame distribution
     */
    struct ContentStats {
        size_t total_frames;
        float avg_frame_similarity;
        float max_frame_similarity;
        float min_frame_similarity;
        size_t num_scenes;
    };
    
    ContentStats computeContentStats();
};

#endif // VIDEO_SEARCH_H
