#ifndef DETECTION_EXPERIMENTS_BBOX_PARSER_H
#define DETECTION_EXPERIMENTS_BBOX_PARSER_H

#include <vector>

#include <nvdsinfer.h>


namespace bbox_parser
{
	std::vector<NvDsInferObjectDetectionInfo> parse_bboxes(float* const cls_data, float* const loc_data, const float threshold, const float nms_threshold, const unsigned int num_locations, const unsigned int num_classes, const unsigned int width, const unsigned int height);
}

#endif // DETECTION_EXPERIMENTS_BBOX_PARSER_H
