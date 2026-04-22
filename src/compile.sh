#!/bin/bash
# Compilation script for DeepStream Object Tracker (Headless-Ready)

set -e

DEEPSTREAM_PATH="${DEEPSTREAM_PATH:-/opt/nvidia/deepstream/deepstream-7.1}"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"

# Check dependencies
echo "Checking GStreamer..."
pkg-config --exists gstreamer-1.0 || { echo "GStreamer not found. Install: sudo apt install libgstreamer1.0-dev"; exit 1; }

echo "GStreamer found"
echo "DeepStream path: $DEEPSTREAM_PATH"
echo "CUDA path: $CUDA_PATH"

# Compilation flags
CXXFLAGS="-std=c++17 -Wall -O2"
INCLUDES="-I$DEEPSTREAM_PATH/sources/includes -I$CUDA_PATH/include"
LIBS_FLAGS="-L$DEEPSTREAM_PATH/lib -lnvdsgst_meta -lnvds_meta -lnvdsgst_helper -lnvds_infer"
GSTREAMER_FLAGS="$(pkg-config --cflags --libs gstreamer-1.0 glib-2.0)"

echo ""
echo " Compiling DeepStream Object Tracker (Headless)..."
echo " Command: g++ aivision_pipeline.cpp -o  aivisionstreamer $INCLUDES $LIBS_FLAGS $GSTREAMER_FLAGS"
echo ""

g++ $CXXFLAGS \
    -o  aivisionstreamer \
    aivision_pipeline.cpp \
    $INCLUDES \
    $LIBS_FLAGS \
    $GSTREAMER_FLAGS

echo ""
echo "Compilation successful!"
echo ""
echo "Binary: ./ aivisionstreamer"
echo ""
echo "USAGE:"
echo "  ./ aivisionstreamer --help                  # Show help"
echo "  ./ aivisionstreamer --headless              # Analytics only (fakesink)"
echo "  ./ aivisionstreamer --file output.mp4       # Save to file"
echo "  ./ aivisionstreamer --display               # Display output"
echo ""
echo "EXAMPLES:"
echo "  # Run in Docker/SSH (headless, no display):"
echo "  ./ aivisionstreamer --headless --verbose"
echo ""
echo "  # Save detections to file:"
echo "  ./ aivisionstreamer --file detections.mp4"
echo ""
echo "  # Stream to another machine:"
echo "  ./ aivisionstreamer --udp 192.168.1.100:5000"
echo ""
