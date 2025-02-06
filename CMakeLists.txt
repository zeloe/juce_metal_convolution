cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(MetalConv LANGUAGES CXX VERSION 0.0.1)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

# IDEs: Enable grouping of source files into folders in IDEs.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(LIB_JUCE_TAG "8.0.6")

include(FetchContent)

# Keep dependencies outside of the "Build" directory.
set(FETCHCONTENT_BASE_DIR "${PROJECT_SOURCE_DIR}/Libs" CACHE PATH "External dependencies path." FORCE)

# Fetch JUCE
FetchContent_Declare(juce
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG        ${LIB_JUCE_TAG}
    GIT_SHALLOW    TRUE
    GIT_CONFIG     advice.detachedHead=false
    SOURCE_DIR     "${FETCHCONTENT_BASE_DIR}/JUCE"
    SUBBUILD_DIR   "${FETCHCONTENT_BASE_DIR}/JUCE-Subbuild"
    BINARY_DIR     "${FETCHCONTENT_BASE_DIR}/JUCE-Build"
)
FetchContent_MakeAvailable(juce)

set(TARGET_NAME ${PROJECT_NAME})

# Add Metal-CMake subdirectory and define MetalCPP
add_subdirectory(metal-cmake)



# Add directories for project components
add_subdirectory(binarydata)
add_subdirectory(terminalapp)


