/*
$ sudo nvpmodel -m 0 --for MAX perf and power
$ sudo jetson_clocks

$ g++ -std=c++17 -o aivisionstreamer src/aivision_pipeline.cpp -I /opt/nvidia/deepstream/deepstream-7.1/sources/includes -I /usr/local/cuda/include $(pkg-config --cflags --libs gstreamer-1.0 glib-2.0) -L /opt/nvidia/deepstream/deepstream-7.1/lib -lnvdsgst_meta -lnvds_meta -lnvdsgst_helper -lnvds_infer

PIPELINE 
CSI Camera (Live) → Direct to capsfilter
MP4 File (H.265) → Demux → H.265 Parser → Hardware Decoder → Memory Convert

USAGE:
  ./aivisionstreamer                         Use CSI camera  (default)
  ./aivisionstreamer --input video.mp4       Analyze MP4 file (default: save to file)
  ./aivisionstreamer --display               Display output on screen
  
EXAMPLES:
  ./aivisionstreamer
  ./aivisionstreamer --input myvideo.mp4
  ./aivisionstreamer --input myvideo.mp4 --display
  ./aivisionstreamer --input myvideo.mp4 --headless


  nvarguscamerasrc → capsfilter → nvstreammux → queue → nvinfer → queue → nvtracker 
→ queue → nvdsanalytics → queue → nvvideoconvert → nvdsosd → queue → nveglglessink

*/

#include "aivision_pipeline.h"
#include <csignal>
#include <sstream>

// ============================================================================
// CONFIGURATION IMPLEMENTATION
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
              << "║────────────────────────────────────────────────║\n"
              << "║ Press Ctrl+C to exit gracefully              ║\n"
              << "╚════════════════════════════════════════════════╝\n\n";
}

// ============================================================================
// SOURCE FACTORY IMPLEMENTATION
// ============================================================================

GstElement* SourceFactory::createSource() const {
    switch (m_config.source_type) {
        case SourceType::CSI_CAMERA:
            return createCameraSource();
        case SourceType::FILE:
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

// ============================================================================
// SINK FACTORY IMPLEMENTATION
// ============================================================================

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

// ============================================================================
// ANALYTICS PROCESSOR IMPLEMENTATION
// ============================================================================

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

// ============================================================================
// PAD PROBE CALLBACKS
// ============================================================================

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

    createElement("nvinfer", "infrence");
    createElement("queue", "queue_infer"); //lets inference run asynchronously

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
    g_object_set(G_OBJECT(infrence), "config-file-path", m_config.infer_config_path.c_str(), NULL);
    std::cout << "Inference (infrence) configured\n";
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

void PipelineBuilder::configureQueues() {

    g_object_set(G_OBJECT(m_elements["queue_mux"]),
                 "max-size-buffers", m_config.queue_max_buffers,
                 "max-size-time", 0,
                 NULL);

    g_object_set(G_OBJECT(m_elements["queue_infer"]),
                 "max-size-buffers", m_config.queue_max_buffers,
                 "max-size-time", 0,
                 NULL);

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
    
    // Main processing chain
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




