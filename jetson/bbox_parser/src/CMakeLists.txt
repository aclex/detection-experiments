set(SOURCES
	interface.cpp
	bbox_parser.cpp
	)

add_library(bbox_parser SHARED ${SOURCES})
target_compile_features(bbox_parser PRIVATE cxx_std_17)
target_link_libraries(bbox_parser DeepStream::DeepStream)
