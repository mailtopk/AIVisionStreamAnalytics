/* bash
$ g++ -o lib_custom_inf_impl.so -shared -fPIC custom_inf_impl.cpp \
-I /opt/nvidia/deepstream/deepstream-7.1/sources/includes/ \
-I /usr/local/cuda/include/ \
-L /opt/nvidia/deepstream/deepstream-7.1/lib/ \
-L /usr/local/cuda/lib64/ \
-L /usr/lib/aarch64-linux-gnu/ \
-lnvds_infer -lcudart
*/

#include <iostream>
#include <vector>
#include <string>
#include <algorithm> // Required for std::max/min
#include "nvdsinfer_custom_impl.h"

extern "C" bool NvDsInferParseCustomYolo(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList) 
{
    if (outputLayersInfo.empty()) {
    	return false;
    }

    const NvDsInferLayerInfo &layer = outputLayersInfo[0];
    float* output = (float*)layer.buffer;
    if (!output) return false;

    unsigned int num_rows = layer.inferDims.d[0];  // e.g., 84 (4 coords + 80 classes)
    unsigned int num_boxes = layer.inferDims.d[1]; // e.g., 8400
    unsigned int num_classes = detectionParams.numClassesConfigured;

    for (unsigned int i = 0; i < num_boxes; i++) {
        float max_score = 0;
        int max_class = -1;

        for (unsigned int c = 0; c < num_classes; c++) {
            unsigned int index = (4 + c) * num_boxes + i;
            float score = output[index];
            if (score > 0.25f) {
                if (score > max_score) {
                    max_score = score;
                    max_class = c;
                }
            }
        }

        if (max_class != -1) {
            float xc = output[0 * num_boxes + i];
            float yc = output[1 * num_boxes + i];
            float w  = output[2 * num_boxes + i];
            float h  = output[3 * num_boxes + i];

            NvDsInferObjectDetectionInfo obj;
            
            /* ADVANCED TWEAK: Coordinate Clipping */
            /* Trackers like NvDCF can crash if boxes fall outside the frame */
            obj.left   = std::max(0.0f, std::min((float)networkInfo.width - 1, (xc - w / 2.0f)));
            obj.top    = std::max(0.0f, std::min((float)networkInfo.height - 1, (yc - h / 2.0f)));
            obj.width  = std::min(w, (float)networkInfo.width - obj.left);
            obj.height = std::min(h, (float)networkInfo.height - obj.top);

            obj.detectionConfidence = max_score;
            obj.classId = max_class;

            /* Filter out tiny "noise" boxes that confuse the tracker */
            if (obj.width > 2 && obj.height > 2) {
                objectList.push_back(obj);
            }
        }
    }
    return true;
}
