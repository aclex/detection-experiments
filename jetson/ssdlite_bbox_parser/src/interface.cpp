#include <iostream>

#include <nvdsinfer_custom_impl.h>

#include "ssdlite_bbox_parser.h"

extern "C" bool NvDsInferParseCustomSSDLite(const std::vector<NvDsInferLayerInfo>& outputLayersInfo, const NvDsInferNetworkInfo& networkInfo, const NvDsInferParseDetectionParams& detectionParams, std::vector<NvDsInferObjectDetectionInfo>& objectList)
{
	float *cls_data, *loc_data;
	unsigned int num_locations { };
	const unsigned int num_classes { detectionParams.numClassesConfigured };

	const float threshold { num_classes ? detectionParams.perClassThreshold[0] : 0.5f };

	const float nms_threshold { 0.5 };

	for (const auto& layer_info : outputLayersInfo)
	{
		if (layer_info.layerName == "cls")
		{
			cls_data = static_cast<float*>(layer_info.buffer);

			if (!num_locations && layer_info.dims.numDims > 0)
				num_locations = layer_info.dims.d[0];

			if (layer_info.dims.numDims > 1 && num_classes != layer_info.dims.d[1])
			{
				std::cerr << "Number of classes in the configuration and returned from the model mismatch!" << std::endl;
			}
		}
		else if (layer_info.layerName = "loc")
		{
			loc_data = static_cast<float*>(layer_info.buffer);
		}
	}

	objectList = ssdlite::parse_bboxes(cls_data, loc_data, threshold, nms_threshold, num_locations, networkInfo.width, networkInfo.height);

	return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomSSDLite)
