#include "bbox_parser.h"

#include <algorithm>
#include <cmath>

#include <nvdsinfer.h>

using namespace std;

namespace
{
	unsigned int embed(const float value, const unsigned int limit) noexcept
	{
		return clamp(static_cast<unsigned int>(round(value)), 0u, limit);
	}

	NvDsInferObjectDetectionInfo parse_detection(float* const cls_data, float* const loc_data, const unsigned int num_classes, const unsigned int width, const unsigned int height)
	{
		NvDsInferObjectDetectionInfo result;

		const auto it { max_element(cls_data, cls_data + num_classes) };

		result.classId = static_cast<unsigned int>(it - cls_data);
		result.detectionConfidence = *it;

		result.left = embed(loc_data[0] * width, width);
		result.top = embed(loc_data[1] * height, height);
		result.width = embed((loc_data[2] - loc_data[0]) * width, width);
		result.height = embed((loc_data[3] - loc_data[1]) * height, height);

		return result;
	}

	const float iou(const NvDsInferParseObjectInfo& a, const NvDsInferParseObjectInfo& b) noexcept
	{
		const unsigned int overlap_left { max(a.left, b.left) };
		const unsigned int overlap_top { max(a.top, b.top) };
		const unsigned int overlap_right { min(a.left + a.width, b.left + b.width) };
		const unsigned int overlap_bottom { min(a.top + a.height, b.top + b.height) };

		if (overlap_right < overlap_left || overlap_bottom < overlap_top)
		{
			return 0.f;
		}

		const unsigned int overlap { (overlap_right - overlap_left) * (overlap_bottom - overlap_top) };
		const unsigned int comp { a.width * a.height + b.width * b.height - overlap };

		return float(overlap) / comp;
	}

	vector<NvDsInferParseObjectInfo> nms(vector<NvDsInferParseObjectInfo> detections, const float nms_threshold)
	{
		std::vector<NvDsInferParseObjectInfo> result;

		std::sort(detections.begin(), detections.end(),
			[](const auto& a, const auto& b)
			{
				return a.detectionConfidence > b.detectionConfidence;
			});

		for (auto d : detections)
		{
			bool keep { true };

			for (auto r : result)
			{
				if (keep)
				{
					keep = iou(d, r) <= nms_threshold;
				}
				else
					break;
			}

			if (keep)
				result.push_back(d);
		}

		return result;
	}
}

vector<NvDsInferObjectDetectionInfo> bbox_parser::parse_bboxes(float* const cls_data, float* const loc_data, const float threshold, const float nms_threshold, const unsigned int num_locations, const unsigned int num_classes, const unsigned int width, const unsigned int height)
{
	vector<NvDsInferObjectDetectionInfo> result;
	result.reserve(num_locations);

	vector<vector<NvDsInferObjectDetectionInfo>> bboxes_per_class(num_classes);

	for (unsigned int i = 0; i < num_locations; ++i)
	{
		const auto& detection { parse_detection(cls_data + i * num_classes, loc_data + i * 4, num_classes, width, height) };

		if (detection.classId == 0) // ignore background detections
			continue;

		if (detection.detectionConfidence >= threshold)
			bboxes_per_class[detection.classId].push_back(detection);
	}

	for (auto&& class_bboxes : bboxes_per_class)
	{
		const auto& kept_bboxes { nms(move(class_bboxes), nms_threshold) };
		result.insert(end(result), begin(kept_bboxes), end(kept_bboxes));
	}

	return result;
}
