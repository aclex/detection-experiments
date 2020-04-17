set(SOURCES
	interface.cpp
	ssdlite_bbox_parser.cpp
	)

add_library(ssdlite_bbox_parser SHARED ${SOURCES})
target_link_libraries(ssdlite_bbox_parser DeepStream::DeepStream)
