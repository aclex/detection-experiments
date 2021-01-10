#include <iostream>
#include <cstring>

#include <nvdsinfer_custom_impl.h>

#include "bbox_parser.h"

extern "C" bool NvDsInferParseCustomBboxes(const std::vector<NvDsInferLayerInfo>& outputLayersInfo, const NvDsInferNetworkInfo& networkInfo, const NvDsInferParseDetectionParams& detectionParams, std::vector<NvDsInferObjectDetectionInfo>& objectList)
{
	const float *cls_data, *loc_data;
	unsigned int num_locations { };
	const unsigned int num_classes { detectionParams.numClassesConfigured };

	const float threshold { num_classes ? detectionParams.perClassThreshold[0] : 0.5f };

	const float nms_threshold { 0.5 };

	for (const auto& layer_info : outputLayersInfo)
	{
		if (!std::strcmp(layer_info.layerName, "cls"))
		{
			cls_data = static_cast<const float*>(layer_info.buffer);

#if DS_GENERATION == 4
			const auto& dims { layer_info.dims };
#else
			const auto& dims { layer_info.inferDims };
#endif

			if (!num_locations && dims.numDims > 0)
				num_locations = dims.d[0];

			if (dims.numDims > 1 && num_classes != dims.d[1])
			{
				std::cerr << "Number of classes in the configuration and returned from the model mismatch!" << std::endl;
			}
		}
		else if (!std::strcmp(layer_info.layerName, "reg"))
		{
			loc_data = static_cast<const float*>(layer_info.buffer);
		}
	}

	objectList = bbox_parser::parse_bboxes(cls_data, loc_data, threshold, nms_threshold, num_locations, num_classes, networkInfo.width, networkInfo.height);

	return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomBboxes)
