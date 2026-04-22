#ifndef AIVISION_PIPELINE_H
#define AIVISION_PIPELINE_H

#include <gst/gst.h>
#include <glib.h>
#include <iostream>
#include <string>
#include <memory>
#include <unordered_map>
#include "gstnvdsmeta.h"
#include "nvds_analytics_meta.h"

enum class SourceType {
    CSI_CAMERA,
    FILE
};

enum class OutputType {
    DISPLAY,
    HEADLESS
};

/**
 * @class GStreamerException
 * @brief Custom exception class for GStreamer errors
 */
class GStreamerException : public std::exception {
private:
    std::string m_message;

public:
    explicit GStreamerException(const std::string& message) : m_message(message) {}
    const char* what() const noexcept override { return m_message.c_str(); }
};


/**
 * @class PipelineConfiguration
 * @brief Encapsulates all pipeline configuration parameters
 */

 // configuration class 
class PipelineConfiguration {
public:
    // Source configuration
    SourceType source_type{SourceType::CSI_CAMERA};
    std::string input_file;

    // Output configuration
    OutputType output_type{OutputType::HEADLESS};

    // Inference configuration
    std::string infer_config_path{"../config/config_infer_primary_yolo.txt"};
    std::string tracker_lib_path{"/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"};
    std::string tracker_config_path{"../config/config_tracker_NvDCF.yml"};
    std::string analytics_config_path{"../config/config_analytics.txt"};
    

    // Stream parameters
    guint stream_width{1280};
    guint stream_height{720};
    guint batch_size{1};
    guint fps{30};
    guint sensor_width{1920};
    guint sensor_height{1080};

    // Tracker parameters
    guint tracker_width{640};
    guint tracker_height{384};
    guint compute_hw{0};

    // Queue parameters
    guint queue_max_buffers{30};

    // Validation
    bool validate() const;
    void print() const;
};

/**
 * @interface IElementFactory
 * @brief Abstract interface for creating GStreamer elements
 */
class IElementFactory {
public:
    virtual ~IElementFactory() = default;
    virtual GstElement* create() = 0;
};

/**
 * @class SourceFactory
 * @brief Factory for creating source elements based on source type
 */
class SourceFactory {
private:
    const PipelineConfiguration& m_config;

public:
    explicit SourceFactory(const PipelineConfiguration& config) : m_config(config) {}
    GstElement* createSource() const;

private:
    GstElement* createCameraSource() const;
    GstElement* createFileSource() const;
};

/**
 * @class SinkFactory
 * @brief Factory for creating sink elements based on output type
 */
class SinkFactory {
private:
    const PipelineConfiguration& m_config;

public:
    explicit SinkFactory(const PipelineConfiguration& config) : m_config(config) {}
    GstElement* createSink() const;

private:
    GstElement* createDisplaySink() const;
    GstElement* createHeadlessSink() const;
};

// ============================================================================
// ANALYTICS PROCESSING
// ============================================================================

/**
 * @class AnalyticsData
 * @brief Container for processed analytics metadata
 */
struct AnalyticsData {
    std::unordered_map<std::string, int> line_crossing_counts;
    std::unordered_map<std::string, bool> crowded_status;
    std::unordered_map<std::string, int> roi_occupancy;
};

/**
 * @class AnalyticsProcessor
 * @brief Processes analytics metadata from DeepStream
 */
class AnalyticsProcessor {
public:
    AnalyticsProcessor() = default;
    ~AnalyticsProcessor() = default;

    /**
     * Process buffer metadata and extract analytics information
     * @param batch_meta Pointer to NvDsBatchMeta
     */
    void processMetadata(NvDsBatchMeta* batch_meta);

private:
    void processFrameMetadata(NvDsFrameMeta* frame_meta);
    void processAnalyticsFrame(NvDsAnalyticsFrameMeta* meta);
};

// ============================================================================
// PIPELINE BUILDING
// ============================================================================

/**
 * @class PipelineBuilder
 * @brief Builds GStreamer pipeline with proper element linkage
 */
class PipelineBuilder {
private:
    const PipelineConfiguration& m_config;
    GstElement* m_pipeline{nullptr};
    std::unordered_map<std::string, GstElement*> m_elements;

public:
    explicit PipelineBuilder(const PipelineConfiguration& config);
    ~PipelineBuilder() = default;

    /**
     * Build the complete pipeline
     * @return Pointer to the constructed pipeline
     * @throws GStreamerException if pipeline construction fails
     */
    GstElement* build();

    /**
     * Get a pipeline element by name
     * @param name Element name
     * @return Pointer to the element or nullptr
     */
    GstElement* getElement(const std::string& name) const;

private:
    void createElements();
    void configureElements();
    void linkElements();
    void linkSourceToProcessor();
    void linkProcessingChain();
    void attachProbes();

    GstElement* createElement(const std::string& factory_name, const std::string& element_name);
    void configureSourceElement(GstElement* source);
    void configureStreammux(GstElement* streammux);
    void configureInference(GstElement* pgie);
    void configureTracker(GstElement* tracker);
    void configureAnalytics(GstElement* analytics);
    void configureQueues();
};

// ============================================================================
// PIPELINE MANAGEMENT
// ============================================================================

/**
 * @class BusMessageHandler
 * @brief Handles GStreamer bus messages
 */
class BusMessageHandler {
public:
    BusMessageHandler() = default;
    virtual ~BusMessageHandler() = default;

    /**
     * Handle a GStreamer bus message
     * @param msg The GStreamer message
     * @return TRUE to continue processing, FALSE to stop
     */
    virtual gboolean handleMessage(GstMessage* msg) = 0;

protected:
    void logError(GstMessage* msg);
    void logWarning(GstMessage* msg);
};

/**
 * @class DefaultBusMessageHandler
 * @brief Default implementation of bus message handler
 */
class DefaultBusMessageHandler : public BusMessageHandler {
private:
    GMainLoop* m_main_loop{nullptr};

public:
    explicit DefaultBusMessageHandler(GMainLoop* loop) : m_main_loop(loop) {}

    gboolean handleMessage(GstMessage* msg) override;
};

/**
 * @class PipelineManager
 * @brief Manages the complete pipeline lifecycle
 */
class PipelineManager {
private:
    PipelineConfiguration m_config;
    GstElement* m_pipeline{nullptr};
    GMainLoop* m_main_loop{nullptr};
    guint m_bus_watch_id{0};
    std::unique_ptr<BusMessageHandler> m_bus_handler;
    std::unique_ptr<AnalyticsProcessor> m_analytics_processor;
    bool m_is_running{false};

public:
    /**
     * Constructor
     * @param config Pipeline configuration
     */
    explicit PipelineManager(const PipelineConfiguration& config);

    /**
     * Destructor - ensures proper cleanup
     */
    ~PipelineManager();

    /**
     * Initialize the pipeline
     * @throws GStreamerException if initialization fails
     */
    void initialize();

    /**
     * Start running the pipeline
     */
    void run();

    /**
     * Stop the pipeline gracefully
     */
    void stop();

    /**
     * Check if pipeline is running
     * @return true if running, false otherwise
     */
    bool isRunning() const { return m_is_running; }

    /**
     * Get the GStreamer pipeline element
     * @return Pointer to the pipeline
     */
    GstElement* getPipeline() const { return m_pipeline; }

    /**
     * Get the GLib main loop
     * @return Pointer to the main loop
     */
    GMainLoop* getMainLoop() const { return m_main_loop; }

private:
    void setupBusWatch();
    void setupSignalHandlers();
    void cleanup();
};

// ============================================================================
// SIGNAL HANDLING
// ============================================================================

/**
 * @class SignalHandler
 * @brief Manages process signal handling for graceful shutdown
 */
class SignalHandler {
private:
    static PipelineManager* s_manager;

public:
    /**
     * Register signal handlers
     * @param manager The pipeline manager to shutdown
     */
    static void registerHandlers(PipelineManager* manager);

private:
    static void handleSignal(int sig);
};

/**
 * @class ArgumentParser
 * @brief Parses command-line arguments and populates configuration
 */
class ArgumentParser {
public:
    /**
     * Parse command-line arguments
     * @param argc Argument count
     * @param argv Argument values
     * @param config Configuration to populate
     * @return true if parsing was successful
     */
    static bool parse(int argc, char* argv[], PipelineConfiguration& config);
    static void printUsage(const char* program_name);
};

#endif // AIVISION_PIPELINE_H
