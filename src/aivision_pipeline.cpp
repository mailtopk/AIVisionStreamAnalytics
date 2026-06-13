/*
$ sudo nvpmodel -m 0 --for MAX perf and power
$ sudo jetson_clocks

$ g++ -std=c++17 -o aivisionstreamer src/aivision_pipeline.cpp -I /opt/nvidia/deepstream/deepstream-7.1/sources/includes -I /usr/local/cuda/include $(pkg-config --cflags --libs gstreamer-1.0 glib-2.0) -L /opt/nvidia/deepstream/deepstream-7.1/lib -lnvdsgst_meta -lnvds_meta -lnvdsgst_helper -lnvds_infer

USAGE:
  ./aivisionstreamer                         Use I2C camera  (default)
  ./aivisionstreamer --input video.mp4       Analyze MP4 file (default: save to file)
  ./aivisionstreamer --display               Display output on screen
  
EXAMPLES:
  ./aivisionstreamer
  ./aivisionstreamer --input myvideo.mp4
  ./aivisionstreamer --input myvideo.mp4 --display
  ./aivisionstreamer --input myvideo.mp4 --headless

*/

#include "aivision_pipeline.h"
#include <csignal>
#include <sstream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <limits>

// ============================================================================
// VECTOR DATA IMPLEMENTATION
// ============================================================================

std::string VectorData::serialize() const {
    std::stringstream ss;
    ss << track_id << "|" << frame_id << "|" << class_id << "|" << confidence << "|" << timestamp;
    
    // Add embedding vector (comma-separated floats)
    ss << "|";
    for (size_t i = 0; i < embedding.size(); i++) {
        if (i > 0) ss << ",";
        ss << embedding[i];
    }
    
    return ss.str();
}

VectorData VectorData::deserialize(const std::string& data) {
    VectorData vec;
    std::stringstream ss(data);
    std::string token;
    int field = 0;
    
    while (std::getline(ss, token, '|')) {
        switch (field) {
            case 0: vec.track_id = std::stoul(token); break;
            case 1: vec.frame_id = std::stoul(token); break;
            case 2: vec.class_id = std::stoul(token); break;
            case 3: vec.confidence = std::stof(token); break;
            case 4: vec.timestamp = token; break;
            case 5: {
                // Parse embedding vector
                std::stringstream embed_ss(token);
                std::string val;
                while (std::getline(embed_ss, val, ',')) {
                    vec.embedding.push_back(std::stof(val));
                }
                break;
            }
        }
        field++;
    }
    
    return vec;
}

// ============================================================================
// FAISS VECTOR DATABASE IMPLEMENTATION (required)
// ============================================================================

FAISSVectorDB::FAISSVectorDB(const std::string& index_path, const std::string& metadata_path)
    : m_index_path(index_path), m_metadata_path(metadata_path) {
    initialize();
}

FAISSVectorDB::~FAISSVectorDB() {
    close();
}

bool FAISSVectorDB::initialize() {
    try {
        // Try to load existing index
        if (loadIndex()) {
            std::cout << "FAISS: Loaded existing index from " << m_index_path << "\n";
            m_initialized = true;
            return true;
        }
        
        // Index will be created when first vector is added
        m_initialized = true;
        std::cout << "FAISS: New index will be created\n";
        return true;
    } catch (const std::exception& e) {
        std::cerr << "FAISS initialization error: " << e.what() << "\n";
        return false;
    }
}

bool FAISSVectorDB::storeVector(const VectorData& vector) {
    if (!m_initialized) {
        return false;
    }
    
    try {
        // Set vector dimension on first vector
        if (m_metadata.empty() && !vector.embedding.empty()) {
            m_vector_dim = vector.embedding.size();
            
            // Create index if not already created
            if (!m_index) {
                m_index = new faiss::IndexFlatL2(m_vector_dim);
            }
        }
        
        // Add vector to index
        if (m_vector_dim > 0 && vector.embedding.size() == m_vector_dim) {
            auto* index = static_cast<faiss::IndexFlat*>(m_index);
            index->add(1, vector.embedding.data());
            m_metadata.push_back(vector);
            
            // Save periodically (every 100 vectors)
            if (m_metadata.size() % 100 == 0) {
                saveIndex();
            }
            
            return true;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error storing vector in FAISS: " << e.what() << "\n";
    }
    
    return false;
}

std::vector<VectorData> FAISSVectorDB::getVectorsByTrackId(guint track_id) {
    std::vector<VectorData> results;
    for (const auto& vec : m_metadata) {
        if (vec.track_id == track_id) {
            results.push_back(vec);
        }
    }
    return results;
}

std::vector<std::pair<float, VectorData>> FAISSVectorDB::similaritySearch(
    const std::vector<float>& query_vector, int k) {
    
    std::vector<std::pair<float, VectorData>> results;
    
    if (!m_index || query_vector.empty() || m_metadata.empty()) {
        return results;
    }
    
    try {
        if (query_vector.size() != m_vector_dim) {
            std::cerr << "Query vector dimension mismatch\n";
            return results;
        }
        
        auto* index = static_cast<faiss::IndexFlat*>(m_index);
        int limit = std::min(k, static_cast<int>(m_metadata.size()));
        
        std::vector<float> distances(limit);
        std::vector<long> labels(limit);
        
        index->search(1, query_vector.data(), limit, distances.data(), labels.data());
        
        for (int i = 0; i < limit; i++) {
            long lbl = labels[i];
            if (lbl >= 0 && lbl < static_cast<long>(m_metadata.size())) {
                results.push_back(std::make_pair(distances[i], m_metadata[static_cast<size_t>(lbl)]));
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in FAISS similarity search: " << e.what() << "\n";
    }
    
    return results;
}

std::vector<VectorData> FAISSVectorDB::getAllVectors() {
    return m_metadata;
}

bool FAISSVectorDB::loadIndex() {
    try {
        std::ifstream index_file(m_index_path, std::ios::binary);
        if (!index_file.good()) {
            return false;
        }
        
        m_index = faiss::read_index(m_index_path.c_str());
        
        if (!m_index) {
            return false;
        }
        
        m_vector_dim = static_cast<faiss::IndexFlat*>(m_index)->d;
        
        // Load metadata
        std::ifstream metadata_file(m_metadata_path);
        std::string line;
        while (std::getline(metadata_file, line)) {
            if (!line.empty()) {
                m_metadata.push_back(VectorData::deserialize(line));
            }
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading FAISS index: " << e.what() << "\n";
        return false;
    }
}

bool FAISSVectorDB::saveIndex() {
    try {
        if (!m_index) {
            return false;
        }
        
        // Save index
        faiss::write_index(static_cast<faiss::IndexFlat*>(m_index), m_index_path.c_str());
        
        // Save metadata
        std::ofstream metadata_file(m_metadata_path, std::ios::trunc);
        for (const auto& vec : m_metadata) {
            metadata_file << vec.serialize() << "\n";
        }
        metadata_file.flush();
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving FAISS index: " << e.what() << "\n";
        return false;
    }
}

void FAISSVectorDB::close() {
    if (m_index) {
        saveIndex();
        delete static_cast<faiss::IndexFlat*>(m_index);
        m_index = nullptr;
        m_initialized = false;
    }
}

// ============================================================================
// SGIE PROCESSOR IMPLEMENTATION
// ============================================================================

SGIEProcessor::SGIEProcessor(std::unique_ptr<IVectorDatabase> db) : m_vector_db(std::move(db)) {}

void SGIEProcessor::processMetadata(NvDsBatchMeta* batch_meta) {
    if (!batch_meta) return;
    
    for (NvDsFrameMetaList* l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta* frame_meta = static_cast<NvDsFrameMeta*>(l_frame->data);
        processFrameMetadata(frame_meta);
    }
}

void SGIEProcessor::processFrameMetadata(NvDsFrameMeta* frame_meta) {
    if (!frame_meta) return;
    
    for (NvDsObjectMetaList* l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
        NvDsObjectMeta* obj_meta = static_cast<NvDsObjectMeta*>(l_obj->data);
        processObjectMetadata(obj_meta, frame_meta->frame_num);
    }
}

void SGIEProcessor::processObjectMetadata(NvDsObjectMeta* obj_meta, guint frame_id) {
    if (!obj_meta) return;
    
    // Look for SGIE (Secondary GIE) inference metadata in object user metadata
    for (NvDsUserMetaList* l_user = obj_meta->obj_user_meta_list; l_user != NULL; 
         l_user = l_user->next) {
        NvDsUserMeta* user_meta = static_cast<NvDsUserMeta*>(l_user->data);
        
        // Check for infer tensor output metadata
        if (user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META) {
            NvDsInferTensorMeta* tensor_meta = static_cast<NvDsInferTensorMeta*>(user_meta->user_meta_data);
            
            // Extract embedding vector
            std::vector<float> embedding = extractTensorData(tensor_meta);
            
            if (!embedding.empty()) {
                VectorData vec;
                vec.track_id = obj_meta->object_id;
                vec.frame_id = frame_id;
                vec.class_id = obj_meta->class_id;
                vec.confidence = obj_meta->confidence;
                vec.embedding = embedding;
                
                // Generate timestamp
                auto now = std::chrono::system_clock::now();
                auto time = std::chrono::system_clock::to_time_t(now);
                std::stringstream ss;
                ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
                vec.timestamp = ss.str();
                
                // Store vector
                if (m_vector_db && m_vector_db->storeVector(vec)) {
                    m_vector_count++;
                    
                    // Log every 100 vectors
                    if (m_vector_count % 100 == 0) {
                        std::cout << "SGIE: Processed " << m_vector_count 
                                 << " vectors (Track ID: " << vec.track_id 
                                 << ", Size: " << embedding.size() << ")\n";
                    }
                }
            }
        }
    }
}

std::vector<float> SGIEProcessor::extractTensorData(NvDsInferTensorMeta* tensor_meta) {
    std::vector<float> embedding;
    
    if (!tensor_meta || tensor_meta->num_output_layers == 0) {
        return embedding;
    }
    
    try {
        // Get the first output buffer - prefer host buffers
        void* data_ptr = nullptr;
        
        if (tensor_meta->out_buf_ptrs_host) {
            data_ptr = tensor_meta->out_buf_ptrs_host[0];
        } else if (tensor_meta->out_buf_ptrs_dev) {
            // Device buffers require GPU->CPU transfer
            std::cerr << "Warning: Only device buffers available. Skipping tensor extraction.\n";
            return embedding;
        }
        
        if (data_ptr && tensor_meta->output_layers_info) {
            NvDsInferLayerInfo* layer = &tensor_meta->output_layers_info[0];
            
            // Get number of elements (suppress deprecation warning)
            #pragma GCC diagnostic push
            #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
            guint num_elements = layer->dims.numElements;
            #pragma GCC diagnostic pop
            
            if (num_elements > 0) {
                float* float_data = static_cast<float*>(data_ptr);
                embedding.assign(float_data, float_data + num_elements);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error extracting tensor data: " << e.what() << "\n";
    }
    
    return embedding;
}

// ============================================================================
// ORIGINAL IMPLEMENTATION CONTINUES BELOW
// ============================================================================

bool PipelineConfiguration::validate() const {
    if (source_type == SourceType::FILE && input_file.empty()) {
        std::cerr << "ERROR: File source selected but no input file specified\n";
        return false;
    }
    if (stream_width == 0 || stream_height == 0) {
        std::cerr << "ERROR: Invalid stream resolution\n";
        return false;
    }
    return true;
}

void PipelineConfiguration::print() const {
    std::cout << "\n╔════════════════════════════════════════════════╗\n"
              << "║        Pipeline Configuration Summary          ║\n"
              << "╠════════════════════════════════════════════════╣\n"
              << "║ Source: " << (source_type == SourceType::CSI_CAMERA ? "CSI Camera" : "File")
              << std::string(source_type == SourceType::CSI_CAMERA ? 29 : 30, ' ') << "║\n";
    
    if (source_type == SourceType::FILE) {
        std::cout << "║ Input File: " << input_file << "║\n"; 
    }
    
    std::cout << "║ Output: " << (output_type == OutputType::DISPLAY ? "Display" : "Headless")
              << std::string(output_type == OutputType::DISPLAY ? 29 : 30, ' ') << "║\n"
              << "║ Resolution: " << stream_width << "x" << stream_height
              << std::string(28 - std::to_string(stream_height).length(), ' ') << "║\n"
              << "║ FPS: " << fps << std::string(40, ' ') << "║\n"
              << "║ Model: YOLO + NvDCF Tracker                 ║\n"
              << "║ Tracker Resolution: " << tracker_width << "x" << tracker_height
              << std::string(19 - std::to_string(tracker_height).length(), ' ') << "║\n"
              << "║ SGIE (ReID): " << (enable_sgie ? "ENABLED" : "DISABLED")
              << std::string(32 - (enable_sgie ? 7 : 8), ' ') << "║\n"
              << "║ Vector DB: " << vector_db_path
              << std::string(32 - vector_db_path.length(), ' ') << "║\n"
              << "║────────────────────────────────────────────────║\n"
              << "║ Press Ctrl+C to exit gracefully              ║\n"
              << "╚════════════════════════════════════════════════╝\n\n";
}

// ============================================================================
// SOURCE FACTORY IMPLEMENTATION
// ============================================================================

GstElement* SourceFactory::createSource() const {
    switch (m_config.source_type) {
        case SourceType::CSI_CAMERA: //I2C camera
            return createCameraSource();
        case SourceType::FILE: //MP4 file input
            return createFileSource();
        default:
            throw GStreamerException("Unknown source type");
    }
}

GstElement* SourceFactory::createCameraSource() const {
    GstElement* source = gst_element_factory_make("nvarguscamerasrc", "csi-cam-source");
    if (source) {
        g_object_set(G_OBJECT(source), "sensor-id", 0, NULL);
        std::cout << "CSI Camera source created\n";
    } else {
        throw GStreamerException("Failed to create nvarguscamerasrc element");
    }
    return source;
}

GstElement* SourceFactory::createFileSource() const {
    GstElement* source = gst_element_factory_make("filesrc", "file-source");
    if (source) {
        g_object_set(G_OBJECT(source), "location", m_config.input_file.c_str(), NULL);
        std::cout << "File source created: " << m_config.input_file << "\n";
    } else {
        throw GStreamerException("Failed to create filesrc element");
    }
    return source;
}

GstElement* SinkFactory::createSink() const {
    switch (m_config.output_type) {
        case OutputType::DISPLAY:
            return createDisplaySink();
        case OutputType::HEADLESS:
            return createHeadlessSink();
        default:
            throw GStreamerException("Unknown output type");
    }
}

GstElement* SinkFactory::createDisplaySink() const {
    GstElement* sink = gst_element_factory_make("nveglglessink", "egl-sink");
    if (!sink) {
        std::cout << "nveglglessink not available, falling back to fakesink\n";
        sink = gst_element_factory_make("fakesink", "sink");
        if (sink) {
            g_object_set(G_OBJECT(sink), "sync", TRUE, NULL);
        }
    } else {
        std::cout << "Display sink (nveglglessink) created\n";
    }
    if (!sink) {
        throw GStreamerException("Failed to create display sink");
    }
    return sink;
}

GstElement* SinkFactory::createHeadlessSink() const {
    GstElement* sink = gst_element_factory_make("fakesink", "sink");
    if (sink) {
        g_object_set(G_OBJECT(sink), "sync", TRUE, NULL);
        std::cout << "Headless sink (fakesink) created\n";
    } else {
        throw GStreamerException("Failed to create fakesink element");
    }
    return sink;
}

void AnalyticsProcessor::processMetadata(NvDsBatchMeta* batch_meta) {
    if (!batch_meta) return;

    for (NvDsFrameMetaList* l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta* frame_meta = static_cast<NvDsFrameMeta*>(l_frame->data);
        processFrameMetadata(frame_meta);
    }
}

void AnalyticsProcessor::processFrameMetadata(NvDsFrameMeta* frame_meta) {
    if (!frame_meta) return;

    for (NvDsUserMetaList* l_user = frame_meta->frame_user_meta_list; l_user != NULL; l_user = l_user->next) {
        NvDsUserMeta* user_meta = static_cast<NvDsUserMeta*>(l_user->data);
        
        if (user_meta->base_meta.meta_type == NVDS_USER_FRAME_META_NVDSANALYTICS) {
            NvDsAnalyticsFrameMeta* meta = static_cast<NvDsAnalyticsFrameMeta*>(user_meta->user_meta_data);
            processAnalyticsFrame(meta);
        }
    }
}

void AnalyticsProcessor::processAnalyticsFrame(NvDsAnalyticsFrameMeta* meta) {
    if (!meta) return;

    std::cout << "\n" << std::string(50, '-') << "\n";
    
    // Process line crossing data
    if (!meta->objLCCumCnt.empty()) {
        std::cout << "│ [Line Crossing Events]\n";
        for (auto& lc : meta->objLCCumCnt) {
            std::cout << "│   " << lc.first << ": " << lc.second << " total\n";
        }
    }

    // Process crowding status
    if (!meta->ocStatus.empty()) {
        std::cout << "│ [Crowding Status]\n";
        for (auto& ocs : meta->ocStatus) {
            std::cout << "│   " << ocs.first << ": " << (ocs.second ? "CROWDED" : "NORMAL") << "\n";
        }
    }

    // Process ROI occupancy
    if (!meta->objInROIcnt.empty()) {
        std::cout << "│ [ROI Occupancy]\n";
        for (auto& roi : meta->objInROIcnt) {
            std::cout << "│   " << roi.first << ": " << roi.second << " objects\n";
        }
    }
    std::cout << std::string(50, '-') << "\n";
}

static GstPadProbeReturn analyticsPadProbe(GstPad* pad, GstPadProbeInfo* info, gpointer user_data) {
    GstBuffer* buf = static_cast<GstBuffer*>(info->data);
    NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    
    AnalyticsProcessor* processor = static_cast<AnalyticsProcessor*>(user_data);
    if (processor && batch_meta) {
        processor->processMetadata(batch_meta);
    }
    
    return GST_PAD_PROBE_OK;
}

static void onDemuxPadAdded(GstElement* element, GstPad* pad, gpointer user_data) {
    GstCaps* caps = gst_pad_get_current_caps(pad);
    const gchar* name = gst_structure_get_name(gst_caps_get_structure(caps, 0));
    GstElement* h265parser = static_cast<GstElement*>(user_data);
    
    std::cout << "→ qtdemux detected stream: " << name << "\n";
    
    if (g_str_has_prefix(name, "video/x-h265")) {
        GstPad* sinkpad = gst_element_get_static_pad(h265parser, "sink");
        
        if (!gst_pad_is_linked(sinkpad)) {
            if (gst_pad_link(pad, sinkpad) == GST_PAD_LINK_OK) {
                std::cout << "qtdemux → h265parse linked\n";
            } else {
                std::cerr << "✗ Failed to link qtdemux to h265parse\n";
            }
        }
        gst_object_unref(sinkpad);
    }
    
    gst_caps_unref(caps);
}


void BusMessageHandler::logError(GstMessage* msg) {
    gchar* debug = nullptr;
    GError* error = nullptr;
    gst_message_parse_error(msg, &error, &debug);
    
    std::cerr << "\n✗ GStreamer Error: " << error->message << "\n"
              << "Debug Info: " << (debug ? debug : "N/A") << "\n";
    
    g_free(debug);
    g_error_free(error);
}

void BusMessageHandler::logWarning(GstMessage* msg) {
    gchar* debug = nullptr;
    GError* error = nullptr;
    gst_message_parse_warning(msg, &error, &debug);
    
    std::cerr << "GStreamer Warning: " << error->message << "\n";
    
    g_free(debug);
    g_error_free(error);
}

gboolean DefaultBusMessageHandler::handleMessage(GstMessage* msg) {
    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_EOS:
            std::cout << "\nEnd of stream reached\n";
            g_main_loop_quit(m_main_loop);
            break;
            
        case GST_MESSAGE_ERROR:
            logError(msg);
            g_main_loop_quit(m_main_loop);
            break;
            
        case GST_MESSAGE_WARNING:
            logWarning(msg);
            break;
            
        default:
            break;
    }
    return TRUE;
}

static gboolean busCallback(GstBus* bus, GstMessage* msg, gpointer user_data) {
    BusMessageHandler* handler = static_cast<BusMessageHandler*>(user_data);
    return handler->handleMessage(msg);
}

PipelineBuilder::PipelineBuilder(const PipelineConfiguration& config) : m_config(config) {
    m_pipeline = gst_pipeline_new("ai-vision-tracker-pipeline");
    if (!m_pipeline) {
        throw GStreamerException("Failed to create GStreamer pipeline");
    }
}

GstElement* PipelineBuilder::createElement(const std::string& factory_name, const std::string& element_name) {
    GstElement* element = gst_element_factory_make(factory_name.c_str(), element_name.c_str());
    if (!element) {
        throw GStreamerException("Failed to create element: " + factory_name);
    }
    m_elements[element_name] = element;
    return element;
}

GstElement* PipelineBuilder::build() {
    try {
        std::cout << "\n╔════════════════════════════════════════╗\n"
                  << "║   Building GStreamer Pipeline...      ║\n"
                  << "╚════════════════════════════════════════╝\n\n";
        
        createElements();
        configureElements();
        linkElements();
        attachProbes();
        
        std::cout << "\nPipeline built successfully!\n\n";
        return m_pipeline;
    } catch (const GStreamerException& e) {
        std::cerr << "\n✗ Pipeline build failed: " << e.what() << "\n";
        throw;
    }
}

void PipelineBuilder::createElements() {
    std::cout << "Creating elements...\n";
    
    // Source element
    SourceFactory source_factory(m_config);
    GstElement* source = source_factory.createSource();
    m_elements["source"] = source;
    
    // File input specific elements
    if (m_config.source_type == SourceType::FILE) {
        createElement("qtdemux", "demux");
        createElement("h265parse", "h265parser");
        createElement("nvv4l2decoder", "decoder");
        createElement("nvvideoconvert", "nvvidconv_decoder");
    }
    
    // Common processing elements
    createElement("capsfilter", "capsfilter");

    createElement("nvstreammux", "streammux");
    createElement("queue", "queue_mux"); //decouples batching from inference

    createElement("nvinfer", "infrence"); // Primary GPU Inference engine
    createElement("queue", "queue_infer"); //lets inference run asynchronously

    // SGIE for ReID embedding extraction
    if (m_config.enable_sgie) {
        createElement("nvinfer", "sgie");
        createElement("queue", "queue_sgie"); // decouple SGIE from tracker
    }

    createElement("nvtracker", "tracker");
    createElement("queue", "queue_tracker"); //prevents tracker from stalling inference

    createElement("nvdsanalytics", "analytics");
    createElement("queue", "queue_analytics"); //isolates CPU-heavy analytics

    createElement("nvvideoconvert", "nvvidconv_osd");
    createElement("nvdsosd", "osd");

    createElement("queue", "queue_sink"); //avoids display blocking everything/Preventing a slow display
    
    // Sink element
    SinkFactory sink_factory(m_config);
    GstElement* sink = sink_factory.createSink();
    m_elements["sink"] = sink;
    
    // Add all elements to pipeline
    for (auto& e : m_elements) {
        gst_bin_add(GST_BIN(m_pipeline), e.second);
    }
}

void PipelineBuilder::configureElements() {
    std::cout << "Configuring elements...\n";
    
    configureSourceElement(m_elements["source"]);
    configureStreammux(m_elements["streammux"]);
    configureInference(m_elements["infrence"]);
    
    if (m_config.enable_sgie && m_elements.find("sgie") != m_elements.end()) {
        configureSGIE(m_elements["sgie"]);
    }
    
    configureTracker(m_elements["tracker"]);
    configureAnalytics(m_elements["analytics"]);
    configureQueues();
    
    // Configure capsfilter
    GstCaps* caps = nullptr;
    if (m_config.source_type == SourceType::CSI_CAMERA) {
        caps = gst_caps_from_string(
            "video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1");
    } else {
        caps = gst_caps_from_string("video/x-raw(memory:NVMM), format=NV12");
    }
    g_object_set(G_OBJECT(m_elements["capsfilter"]), "caps", caps, NULL);
    gst_caps_unref(caps);
}

void PipelineBuilder::configureSourceElement(GstElement* source) {
    // Additional source configuration if needed
    (void)source; // Suppress unused parameter warning
}

void PipelineBuilder::configureStreammux(GstElement* streammux) {
    gboolean is_live_source = (m_config.source_type == SourceType::CSI_CAMERA);
    g_object_set(G_OBJECT(streammux),
                 "width", m_config.stream_width,
                 "height", m_config.stream_height,
                 "batch-size", m_config.batch_size,
                 "live-source", is_live_source,
                 NULL);
    std::cout << "Streammux configured\n";
}

void PipelineBuilder::configureInference(GstElement* infrence) {
    g_object_set(G_OBJECT(infrence), 
                 "config-file-path", m_config.infer_config_path.c_str(),
                 "unique-id", 1,
                 NULL);
    std::cout << "Inference (PGIE) configured with unique-id=1\n";
}

void PipelineBuilder::configureTracker(GstElement* tracker) {
    g_object_set(G_OBJECT(tracker),
                 "ll-lib-file", m_config.tracker_lib_path.c_str(),
                 "ll-config-file", m_config.tracker_config_path.c_str(),
                 "tracker-width", m_config.tracker_width,
                 "tracker-height", m_config.tracker_height,
                 "compute-hw", m_config.compute_hw,
                 NULL);
    std::cout << "Tracker configured\n";
}

void PipelineBuilder::configureAnalytics(GstElement* analytics) {
    g_object_set(G_OBJECT(analytics), "config-file", m_config.analytics_config_path.c_str(), NULL);
    std::cout << "Analytics configured\n";
}

void PipelineBuilder::configureSGIE(GstElement* sgie) {
    g_object_set(G_OBJECT(sgie), 
                 "config-file-path", m_config.sgie_config_path.c_str(),
                 "unique-id", 2,
                 NULL);
    std::cout << "SGIE (Secondary GIE) configured with unique-id=2 for ReID embeddings\n";
}

void PipelineBuilder::configureQueues() {

    g_object_set(G_OBJECT(m_elements["queue_mux"]),
                 "max-size-buffers", m_config.queue_max_buffers,
                 "max-size-time", 0,
                 NULL);

    g_object_set(G_OBJECT(m_elements["queue_infer"]),
                 "max-size-buffers", m_config.queue_max_buffers,
                 "max-size-time", 0,
                 NULL);

    if (m_config.enable_sgie && m_elements.find("queue_sgie") != m_elements.end()) {
        g_object_set(G_OBJECT(m_elements["queue_sgie"]),
                     "max-size-buffers", m_config.queue_max_buffers,
                     "max-size-time", 0,
                     NULL);
    }

    g_object_set(G_OBJECT(m_elements["queue_tracker"]),
                 "max-size-buffers", m_config.queue_max_buffers,
                 "max-size-time", 0,
                 NULL);

    g_object_set(G_OBJECT(m_elements["queue_analytics"]),
                 "max-size-buffers", m_config.queue_max_buffers,
                 "max-size-time", 0,
                 NULL);
    g_object_set(G_OBJECT(m_elements["queue_sink"]),
                 "max-size-buffers", m_config.queue_max_buffers,
                 "max-size-time", 0,
                 NULL);
    std::cout << "Queues configured\n";
}

void PipelineBuilder::linkElements() {
    std::cout << "Linking elements...\n";
    linkSourceToProcessor();
    linkProcessingChain();
}

void PipelineBuilder::linkSourceToProcessor() {
    if (m_config.source_type == SourceType::FILE) {
        // filesrc → qtdemux
        if (!gst_element_link(m_elements["source"], m_elements["demux"])) {
            throw GStreamerException("Failed to link filesrc to qtdemux");
        }
        
        // qtdemux pad-added → h265parse (dynamic)
        g_signal_connect(m_elements["demux"], "pad-added", 
                        G_CALLBACK(onDemuxPadAdded), m_elements["h265parser"]);
        
        // h265parse → decoder → nvvidconv → capsfilter
        if (!gst_element_link_many(m_elements["h265parser"],
                                   m_elements["decoder"],
                                   m_elements["nvvidconv_decoder"],
                                   m_elements["capsfilter"],
                                   NULL)) {
            throw GStreamerException("Failed to link file processing chain");
        }
        std::cout << "File source chain linked\n";
    } else {
        // source → capsfilter
        if (!gst_element_link(m_elements["source"], m_elements["capsfilter"])) {
            throw GStreamerException("Failed to link camera source to capsfilter");
        }
        std::cout << "Camera source chain linked\n";
    }
}

void PipelineBuilder::linkProcessingChain() {
    // capsfilter → streammux
    GstPad* mux_sinkpad = gst_element_request_pad_simple(m_elements["streammux"], "sink_0");
    GstPad* capsfilter_srcpad = gst_element_get_static_pad(m_elements["capsfilter"], "src");
    
    if (gst_pad_link(capsfilter_srcpad, mux_sinkpad) != GST_PAD_LINK_OK) {
        throw GStreamerException("Failed to link capsfilter to streammux");
    }
    gst_object_unref(mux_sinkpad);
    gst_object_unref(capsfilter_srcpad);
    
    // Main processing chain with optional SGIE
    if (m_config.enable_sgie && m_elements.find("sgie") != m_elements.end()) {
        // Chain with SGIE: streammux → queue_mux → infrence → queue_infer → sgie → queue_sgie → tracker
        if (!gst_element_link_many(m_elements["streammux"],
                                   m_elements["queue_mux"],

                                   m_elements["infrence"],
                                   m_elements["queue_infer"],

                                   m_elements["sgie"],
                                   m_elements["queue_sgie"],

                                   m_elements["tracker"],
                                   m_elements["queue_tracker"],

                                   m_elements["analytics"],
                                   m_elements["queue_analytics"],

                                   m_elements["nvvidconv_osd"],
                                   m_elements["osd"],

                                   m_elements["queue_sink"],
                                   m_elements["sink"],
                                   NULL)) {
            throw GStreamerException("Failed to link processing chain with SGIE");
        }
    } else {
        // Chain without SGIE: streammux → queue_mux → infrence → queue_infer → tracker
        if (!gst_element_link_many(m_elements["streammux"],
                                   m_elements["queue_mux"],

                                   m_elements["infrence"],
                                   m_elements["queue_infer"],

                                   m_elements["tracker"],
                                   m_elements["queue_tracker"],

                                   m_elements["analytics"],
                                   m_elements["queue_analytics"],

                                   m_elements["nvvidconv_osd"],
                                   m_elements["osd"],

                                   m_elements["queue_sink"],
                                   m_elements["sink"],
                                   NULL)) {
            throw GStreamerException("Failed to link processing chain");
        }
    }
    std::cout << "Processing chain linked\n";
}

void PipelineBuilder::attachProbes() {
    GstPad* analytics_srcpad = gst_element_get_static_pad(m_elements["analytics"], "src");
    if (analytics_srcpad) {
        // Will be set up in PipelineManager with the analytics processor
        gst_object_unref(analytics_srcpad);
    }
}

GstElement* PipelineBuilder::getElement(const std::string& name) const {
    auto it = m_elements.find(name);
    return (it != m_elements.end()) ? it->second : nullptr;
}

PipelineManager* SignalHandler::s_manager = nullptr;

void SignalHandler::registerHandlers(PipelineManager* manager) {
    s_manager = manager;
    signal(SIGINT, handleSignal);
    signal(SIGTERM, handleSignal);
}

void SignalHandler::handleSignal(int sig) {
    if (s_manager) {
        std::cout << "\n\nSignal " << sig << " received. Shutting down gracefully...\n";
        s_manager->stop();
    }
}

PipelineManager::PipelineManager(const PipelineConfiguration& config) : m_config(config) {
    if (!m_config.validate()) {
        throw GStreamerException("Invalid pipeline configuration");
    }
}

PipelineManager::~PipelineManager() {
    cleanup();
}

void PipelineManager::initialize() {
    std::cout << "Initializing pipeline manager...\n\n";
    
    try {
        // Create main loop
        m_main_loop = g_main_loop_new(NULL, FALSE);
        if (!m_main_loop) {
            throw GStreamerException("Failed to create GMainLoop");
        }
        
        // Build pipeline
        PipelineBuilder builder(m_config);
        m_pipeline = builder.build();
        
        // Create and setup bus handler
        m_bus_handler = std::make_unique<DefaultBusMessageHandler>(m_main_loop);
        
        // Create analytics processor
        m_analytics_processor = std::make_unique<AnalyticsProcessor>();
        
        // Create SGIE processor with FAISS vector database (required)
        if (m_config.enable_sgie) {
            std::unique_ptr<IVectorDatabase> vector_db = std::make_unique<FAISSVectorDB>(
                m_config.faiss_index_path, m_config.faiss_metadata_path);
            std::cout << "Using FAISS vector database for fast similarity search\n";
            m_sgie_processor = std::make_unique<SGIEProcessor>(std::move(vector_db));
        }
        
        // Setup bus watch
        setupBusWatch();
        
        // Attach analytics probe
        GstElement* analytics = gst_bin_get_by_name(GST_BIN(m_pipeline), "analytics");
        if (analytics) {
            GstPad* analytics_srcpad = gst_element_get_static_pad(analytics, "src");
            if (analytics_srcpad) {
                gst_pad_add_probe(analytics_srcpad, GST_PAD_PROBE_TYPE_BUFFER,
                                 analyticsPadProbe, m_analytics_processor.get(), NULL);
                gst_object_unref(analytics_srcpad);
            }
            gst_object_unref(analytics);
        }
        
        // Attach SGIE probe for vector extraction
        if (m_config.enable_sgie && m_sgie_processor) {
            GstElement* sgie = gst_bin_get_by_name(GST_BIN(m_pipeline), "sgie");
            if (sgie) {
                GstPad* sgie_srcpad = gst_element_get_static_pad(sgie, "src");
                if (sgie_srcpad) {
                    gst_pad_add_probe(sgie_srcpad, GST_PAD_PROBE_TYPE_BUFFER,
                                     [](GstPad* pad, GstPadProbeInfo* info, gpointer user_data) -> GstPadProbeReturn {
                                         GstBuffer* buf = static_cast<GstBuffer*>(info->data);
                                         NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buf);
                                         SGIEProcessor* processor = static_cast<SGIEProcessor*>(user_data);
                                         if (processor && batch_meta) {
                                             processor->processMetadata(batch_meta);
                                         }
                                         return GST_PAD_PROBE_OK;
                                     },
                                     m_sgie_processor.get(), NULL);
                    gst_object_unref(sgie_srcpad);
                }
                gst_object_unref(sgie);
            }
        }
        
        // Setup signal handlers
        setupSignalHandlers();
        
        // Print configuration
        m_config.print();
        
        std::cout << "Pipeline manager initialized successfully\n\n";
        
    } catch (const GStreamerException& e) {
        std::cerr << "✗ Initialization failed: " << e.what() << "\n";
        cleanup();
        throw;
    }
}

void PipelineManager::run() {
    if (!m_pipeline || !m_main_loop) {
        throw GStreamerException("Pipeline not initialized");
    }
    
    std::cout << "Starting pipeline...\n";
    
    GstStateChangeReturn ret = gst_element_set_state(m_pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        throw GStreamerException("Failed to start pipeline");
    }
    
    if (ret == GST_STATE_CHANGE_ASYNC) {
        std::cout << "Pipeline transitioning to PLAYING state...\n";
    }
    
    std::cout << "Pipeline running\n"
              << "Press Ctrl+C to exit gracefully\n\n";
    
    m_is_running = true;
    g_main_loop_run(m_main_loop);
    m_is_running = false;
}

void PipelineManager::stop() {
    if (!m_pipeline) return;
    
    std::cout << "\nStopping pipeline...\n";
    gst_element_send_event(m_pipeline, gst_event_new_eos());
    
    if (m_main_loop && g_main_loop_is_running(m_main_loop)) {
        g_main_loop_quit(m_main_loop);
    }
}

void PipelineManager::setupBusWatch() {
    GstBus* bus = gst_pipeline_get_bus(GST_PIPELINE(m_pipeline));
    if (!bus) {
        throw GStreamerException("Failed to get pipeline bus");
    }
    
    m_bus_watch_id = gst_bus_add_watch(bus, busCallback, m_bus_handler.get());
    gst_object_unref(bus);
}

void PipelineManager::setupSignalHandlers() {
    SignalHandler::registerHandlers(this);
}

void PipelineManager::cleanup() {
   // std::cout << "\n\nCleaning up resources...\n";
    
    // Remove bus watch
    if (m_bus_watch_id > 0) {
        g_source_remove(m_bus_watch_id);
        //std::cout << "Bus watch removed\n";
    }
    
    // Stop pipeline
    if (m_pipeline) {
        gst_element_set_state(m_pipeline, GST_STATE_NULL);
        std::cout << "Pipeline stopped\n";
        
        gst_object_unref(GST_OBJECT(m_pipeline));
        m_pipeline = nullptr;
        //std::cout << "Pipeline unrefed\n";
    }
    
    // Cleanup main loop
    if (m_main_loop) {
        g_main_loop_unref(m_main_loop);
        m_main_loop = nullptr;
       // std::cout << "Main loop unrefed\n";
    }
    
    //std::cout << "All resources cleaned up successfully\n\n";
}

bool ArgumentParser::parse(int argc, char* argv[], PipelineConfiguration& config) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            exit(0);
        }
        else if (arg == "--input" && i + 1 < argc) {
            config.source_type = SourceType::FILE;
            config.input_file = argv[++i];
        }
        else if (arg == "--display") {
            config.output_type = OutputType::DISPLAY;
        }
        else {
            std::cerr << "Unknown argument: " << arg << "\n";
            printUsage(argv[0]);
            return false;
        }
    }
    return true;
}

void ArgumentParser::printUsage(const char* program_name) {
    std::cout << "\n╔═══════════════════════════════════════╗\n"
              << "║  AI Vision Stream Analytics Tracker   ║\n"
              << "╚═══════════════════════════════════════╝\n\n"
              << "USAGE: " << program_name << " [OPTIONS]\n\n"
              << "OPTIONS:\n"
              << "--input <path>     Analyze MP4 file (default: CSI camera)\n"
              << "--display          Display output on screen\n"
              << "--help, -h         Show this help message\n\n"
              << "EXAMPLES:\n"
              << "" << program_name << "\n"
              << "" << program_name << " --display\n"
              << "" << program_name << " --input video.mp4\n"
              << "" << program_name << " --input video.mp4 --display\n\n";
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main(int argc, char* argv[]) {
    try {
        // Initialize GStreamer
        gst_init(&argc, &argv);
        
        // Create and validate configuration
        PipelineConfiguration config;
        
        if (argc == 1) {
            ArgumentParser::printUsage(argv[0]);
            return 0;
        }
        
        if (!ArgumentParser::parse(argc, argv, config)) {
            return 1;
        }
        
        // Create and run pipeline manager
        PipelineManager manager(config);
        manager.initialize();
        manager.run();
        
        return 0;
        
    } catch (const GStreamerException& e) {
        std::cerr << "\n✗ Fatal Error: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "\n✗ Unexpected Error: " << e.what() << "\n";
        return 1;
    }
}