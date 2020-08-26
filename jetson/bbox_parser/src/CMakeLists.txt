set(SOURCES
	interface.cpp
	bbox_parser.cpp
	)

add_library(bbox_parser SHARED ${SOURCES})
target_compile_features(bbox_parser PRIVATE cxx_std_17)
target_link_libraries(bbox_parser DeepStream::DeepStream)

if (${DeepStream_VERSION_STRING} VERSION_GREATER_EQUAL "5")
	set(DS_GENERATION "5")
elseif (${DeepStream_VERSION_STRING} VERSION_GREATER_EQUAL "4")
	set(DS_GENERATION "4")
endif()

target_compile_definitions(bbox_parser PRIVATE DS_GENERATION=${DS_GENERATION})
