#include <iostream>

#include <nvdsinfer_custom_impl.h>

extern "C" bool NvDsInferParseCustomSSDLite(const std::vector<NvDsInferLayerInfo>& outputLayersInfo, const NvDsInferNetworkInfo& networkInfo, const NvDsInferParseDetectionParams& detectionParams, std::vector<NvDsInferObjectDetectionInfo>& objectList)
{
	for (const auto& layer_info : outputLayersInfo)
	{
		std::cout << "layer: " << layer_info.layerName << std::endl;
	}

	return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomSSDLite)
