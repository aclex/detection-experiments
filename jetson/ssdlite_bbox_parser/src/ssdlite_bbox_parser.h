#ifndef DETECTION_EXPERIMENTS_SSDLITE_BBOX_PARSER_H
#define DETECTION_EXPERIMENTS_SSDLITE_BBOX_PARSER_H

#include <vector>

struct NvDsInferObjectDetectionInfo;

namespace ssdlite
{
	std::vector<NvDsInferObjectDetectionInfo> parse_bboxes(float* const cls_data, float* const loc_data, const float threshold, const float nms_threshold, const unsigned int num_locations, const unsigned int num_classes);
}

#endif // DETECTION_EXPERIMENTS_SSDLITE_BBOX_PARSER_H
