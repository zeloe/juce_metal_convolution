 

# Add all source files to the list of sources for the executable
file(GLOB_RECURSE JUCE_SOURCES CONFIGURE_DEPENDS source/*.cpp source/*.h)
# Add all sources to the target
juce_add_console_app(${TARGET_NAME}
    PRODUCT_NAME "MetalConv") 





target_sources(${TARGET_NAME} PRIVATE ${JUCE_SOURCES})
#target_sources(${TARGET_NAME} PRIVATE ../metal-cmake/convengine.cpp)
target_compile_definitions(${TARGET_NAME}
    PRIVATE
        # JUCE_WEB_BROWSER and JUCE_USE_CURL would be on by default, but you might not need them.
        JUCE_WEB_BROWSER=0  # If you remove this, add `NEEDS_WEB_BROWSER TRUE` to the `juce_add_console_app` call
        JUCE_USE_CURL=0)    # If you remove this, add `NEEDS_CURL TRUE` to the `juce_add_console_app` call


juce_generate_juce_header(${TARGET_NAME})

# Link the necessary JUCE modules to the target
target_link_libraries(${TARGET_NAME} 
PRIVATE
	BinaryData
    	juce::juce_core
    	juce::juce_audio_utils
    	juce::juce_dsp
    	juce::juce_audio_processors
	METAL
)
# Generate JUCE header