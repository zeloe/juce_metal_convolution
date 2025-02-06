
#include "../../metal-cmake/convengine.h"
#include "JuceHeader.h"

#include "audiocallback.h"
#include <chrono>
#include <ctime>
int main()
{
  //auto audioClass = std::make_unique<HandlerClass>();
    juce::ScopedJuceInitialiser_GUI juceInitialiser;
    const int bs = 1024;
    auto* IRStream = new juce::MemoryInputStream(BinaryData::imp_wav, BinaryData::imp_wavSize,false);
    auto* DRYStream= new juce::MemoryInputStream(BinaryData::dry_wav, BinaryData::dry_wavSize, false);
    
    WavAudioFormat wavFormat;
    std::unique_ptr<AudioFormatReader> impfile (wavFormat.createReaderFor (IRStream, false));
    
    
    const juce::int64 channels = impfile->numChannels;
    WavAudioFormat wavFormat2;
    std::unique_ptr<AudioFormatReader> dry (wavFormat2.createReaderFor (DRYStream, false));
    
    // Create the AudioDeviceManager instance
    juce::AudioDeviceManager audioDeviceManager;
    std::unique_ptr<MyAudioCallback> audiocallback ;
    size_t temp = impfile->lengthInSamples;
    size_t temp2 = dry->lengthInSamples;
    
    size_t bigger = temp2;
    if (temp > temp2) {
        bigger = temp;
    }
    
    
    
    
    
    
    juce::AudioBuffer<float> bufferimp;
    juce::AudioBuffer<float> bufferdry;
    bufferdry.setSize(channels, bigger);
    bufferdry.clear();
    bufferimp.setSize(channels, bigger);
    bufferimp.clear();
    
    
    impfile->read(&bufferimp, 0, temp, 0, true, true);
    
    dry->read(&bufferdry, 0, temp2, 0, true, true);
    
    juce::AudioBuffer<float> out;
    out.setSize(1, temp + temp2);
    out.clear();
    
    
    /*
    // Initialize the AudioDeviceManager with no input/output channels (default setup)
    audioDeviceManager.initialise(0, 2, nullptr, true);
    
    // Get the available device types
    auto& deviceTypes = audioDeviceManager.getAvailableDeviceTypes();
    
    // Iterate through the available device types
    for (auto& deviceType : deviceTypes) {
        std::cout << "Device Type: " << deviceType->getTypeName() << std::endl;
    }
    
    
    // Get the current audio device
    auto* currentDevice = audioDeviceManager.getCurrentAudioDevice();
    std::cout << "Current Device ";
    std::cout << currentDevice->getTypeName() << std::endl;
    std::cout << bs << " = Current Buffer Size" << std::endl;
    
    if (currentDevice == nullptr) {
        std::cerr << "No current audio device available!" << std::endl;
        return 1;
    }
    
    
    // Retrieve the current device setup
    juce::AudioDeviceManager::AudioDeviceSetup deviceSetup;
    audioDeviceManager.getAudioDeviceSetup(deviceSetup);
    
    // Set the desired buffer size (e.g., 128 samples)
    deviceSetup.bufferSize = bs;
    
    // Apply the updated setup
    juce::String error = audioDeviceManager.setAudioDeviceSetup(deviceSetup, true);
    
    // Verify the buffer size has been set
    currentDevice = audioDeviceManager.getCurrentAudioDevice();
    */
    int bufferSize = 512;
    auto engine = std::make_unique<ConvEngine>(bufferimp.getWritePointer(0),bufferSize,bufferimp.getNumSamples());
    
    juce::AudioBuffer<float> tempbuffer;
    tempbuffer.setSize(1, bufferSize);
    tempbuffer.clear();
    juce::AudioBuffer<float> result;
    result.setSize(1, bufferSize);
    result.clear();
    for(int i = 0; i < bigger - bufferSize; i += bufferSize) {
        int tempSize = bufferSize;
        int inc = i;
        float* tempwrite =tempbuffer.getWritePointer(0);
        const float* dry =bufferdry.getReadPointer(0);
        int counter = 0;
         
        std::memcpy(tempwrite, dry + i, bufferSize * sizeof(float));

        engine->render(tempwrite, result.getWritePointer(0));
        tempSize = bufferSize;
        inc = i;
        const float* resPrt = result.getReadPointer(0);
        float* outPtr = out.getWritePointer(0);
        std::memcpy(outPtr + i, resPrt, bufferSize * sizeof(float));

        
        
        
    }
    
    
    /*
    audiocallback = std::make_unique<MyAudioCallback>(engine.get(), deviceSetup.bufferSize,bufferdry.getWritePointer(0),bufferdry.getNumSamples());
    juce::Thread::sleep(1000); // Sleep for 1 second
    
    audioDeviceManager.addAudioCallback(audiocallback.get());
    std::cout << "STARTING" << std::endl;
    while (true) {
        
        // Print CPU usage
        std::cout << "CPU Usage: " << audioDeviceManager.getCpuUsage() * 100 << " %" << std::endl;
        
        
        
        
        //Wait for a short duration before printing CPU usage again
        std::this_thread::sleep_for(std::chrono::seconds(1)); // Adjust the duration as needed
    }
    audioDeviceManager.removeAudioCallback(audiocallback.get());
    audioDeviceManager.closeAudioDevice();
     */
    auto end = std::chrono::system_clock::now();
    
    
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    
   
    juce::File outfile(juce::String("/Users/zenoloesch/Documents/GitHub/juce_metal_convolution/renders/" + static_cast<juce::String>((std::ctime(&end_time))) + "conv.wav"));
    WavAudioFormat format;
    std::unique_ptr<AudioFormatWriter> writer;
    writer.reset (format.createWriterFor (new FileOutputStream (outfile),
                                          48000.0,
                                          out.getNumChannels(),
                                          24,
                                          {},
                                          0));
    writer->writeFromAudioSampleBuffer(out, 0, out.getNumSamples());
    
    return 0;
}
