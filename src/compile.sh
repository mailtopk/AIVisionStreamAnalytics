#!/bin/bash
# Compilation script with optional FAISS support for similarity search

set -e

DEEPSTREAM_PATH="${DEEPSTREAM_PATH:-/opt/nvidia/deepstream/deepstream-7.1}"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
# FAISS is required: ensure libfaiss-dev / pkg-config entry is available

# Check dependencies
echo "Checking GStreamer..."
pkg-config --exists gstreamer-1.0 || { echo "GStreamer not found. Install: sudo apt install libgstreamer1.0-dev"; exit 1; }

echo "GStreamer found"
echo "DeepStream path: $DEEPSTREAM_PATH"
echo "CUDA path: $CUDA_PATH"
echo "FAISS support: required (will error if not found)"

# Compilation flags
CXXFLAGS="-std=c++17 -Wall -O2 -fopenmp"
INCLUDES="-I$DEEPSTREAM_PATH/sources/includes -I$CUDA_PATH/include"
LIBS_FLAGS="-L$DEEPSTREAM_PATH/lib -lnvdsgst_meta -lnvds_meta -lnvdsgst_helper -lnvds_infer"
GSTREAMER_FLAGS="$(pkg-config --cflags --libs gstreamer-1.0 glib-2.0)"

echo "Checking FAISS C++ library..."
if pkg-config --exists faiss; then
    echo "FAISS found via pkg-config"
    FAISS_FLAGS="$(pkg-config --cflags --libs faiss)"
    LIBS_FLAGS="$LIBS_FLAGS $FAISS_FLAGS"
else
    # Try system include check
    if [ -f "/usr/local/include/faiss/Index.h" ] || [ -f "/usr/include/faiss/Index.h" ]; then
        echo "FAISS headers found in standard system paths; linking libfaiss"
        LIBS_FLAGS="$LIBS_FLAGS -lfaiss"
    else
        echo "ERROR: FAISS C++ library not found. Install libfaiss-dev or build FAISS."
        echo "On Debian/Ubuntu: sudo apt install libfaiss-dev";
        exit 1
    fi
fi

# FAISS may depend on BLAS/LAPACK and OpenMP; link common numeric libraries explicitly
LIBS_FLAGS="$LIBS_FLAGS -lopenblas -llapack -lgomp"


echo ""
echo "Compiling AI Vision Stream Analytics with SGIE..."
echo ""

g++ $CXXFLAGS \
    -o aivisionstreamer \
    aivision_pipeline.cpp \
    $INCLUDES \
    $LIBS_FLAGS \
    $GSTREAMER_FLAGS

echo ""
echo "Compilation successful!"
echo ""
echo "Binary: ./aivisionstreamer"
echo ""
echo "USAGE:"
echo "  ./aivisionstreamer --help                  # Show help"
echo "  ./aivisionstreamer --input video.mp4       # Analyze MP4 (headless)"
echo "  ./aivisionstreamer --input video.mp4 --display  # With display"
echo ""
echo "VECTORS & SIMILARITY SEARCH:"
echo "  FAISS index: ../data/vectors.faiss"
echo "  FAISS metadata: ../data/vectors_meta.txt"
echo "  Search: python3 ../tools/vector_search.py --top-k 5"
echo ""

