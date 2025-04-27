
#include "../../metal-cmake/convengine.h"
#include "JuceHeader.h"

#include "audiocallback.h"
#include <chrono>
#include <ctime>
int main()
{

    juce::ScopedJuceInitialiser_GUI juceInitialiser;
    const int bs = 1024;
    const size_t channels = 1;
    // Load audiofiles and allocate memory
    auto* IRStream = new juce::MemoryInputStream(BinaryData::imp_wav, BinaryData::imp_wavSize,false);
    auto* DRYStream= new juce::MemoryInputStream(BinaryData::dry_wav, BinaryData::dry_wavSize, false);
    
    WavAudioFormat wavFormat;
    std::unique_ptr<AudioFormatReader> impfile (wavFormat.createReaderFor (IRStream, false));
    
    
    
    WavAudioFormat wavFormat2;
    std::unique_ptr<AudioFormatReader> dry (wavFormat2.createReaderFor (DRYStream, false));
    
    
    const int imp_file_length = int(impfile->lengthInSamples);
    const int dry_file_length = int(dry->lengthInSamples);
   
    juce::AudioBuffer<float> bufferimp;
    juce::AudioBuffer<float> bufferdry;
    bufferdry.setSize(channels, imp_file_length);
    bufferdry.clear();
    bufferimp.setSize(channels, imp_file_length);
    bufferimp.clear();
    
    
    impfile->read(&bufferimp, 0, imp_file_length, 0, true, true);
    
    dry->read(&bufferdry, 0, dry_file_length, 0, true, true);
    
    juce::AudioBuffer<float> out;
    out.setSize(1, imp_file_length + dry_file_length);
    out.clear();
    
    
    // Create the AudioDeviceManager instance
    juce::AudioDeviceManager audioDeviceManager;
    std::unique_ptr<MyAudioCallback> audiocallback ;
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
    
    // Set the desired buffer size (e.g., 1024 samples)
    deviceSetup.bufferSize = bs;
    
    // Apply the updated setup
    juce::String error = audioDeviceManager.setAudioDeviceSetup(deviceSetup, true);
    
    // Verify the buffer size has been set
    currentDevice = audioDeviceManager.getCurrentAudioDevice();
    
   
    auto engine = std::make_unique<ConvEngine>(bufferimp.getWritePointer(0),bs,bufferimp.getNumSamples());
    

    audiocallback = std::make_unique<MyAudioCallback>(engine.get(), deviceSetup.bufferSize,bufferdry.getWritePointer(0),bufferdry.getNumSamples());
    juce::Thread::sleep(1000); // Sleep for 1 second
    
    audioDeviceManager.addAudioCallback(audiocallback.get());
    std::cout << "STARTING" << std::endl;
    while (true) {
        
        // Print CPU usage
        std::cout << "CPU Usage: " << audioDeviceManager.getCpuUsage() * 100 << " %" << std::endl;
        
        
        
        
        //Wait for a short duration before printing CPU usage again
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    audioDeviceManager.removeAudioCallback(audiocallback.get());
    audioDeviceManager.closeAudioDevice();
   
    
    return 0;
}
