
/*
 This file contains a basic class for accessing audio drivers
*/



#ifndef _AUDIOCALLBACK_H_
#define _AUDIOCALLBACK_H_
#include "../../metal-cmake/convengine.h"
#include <JuceHeader.h>
class MyAudioCallback : public juce::AudioIODeviceCallback, public juce::Thread
    {
    public:
        MyAudioCallback(ConvEngine* _engine, int maxBufferSize, float* dryPtr, int drySize);
        ~MyAudioCallback() override;
        void audioDeviceIOCallbackWithContext(const float* const* inputChannelData,
            int	numInputChannels,
            float* const* outputChannelData,
            int	numOutputChannels,
            int	numSamples,
            const AudioIODeviceCallbackContext& context) override;
         
        bool hasFinished = false;
        
        virtual void 	audioDeviceAboutToStart(AudioIODevice* device) override;
           
        virtual void 	audioDeviceStopped() override;
            

        virtual void 	audioDeviceError(const String& errorMessage) override;
            

       


    private:
        void run () override;
        juce::AudioBuffer<float> resBuffer;
        ConvEngine* metal_engine = nullptr;
        int bs = -1;
        int drySize = -1;
        int counter = 0;
        juce::AudioBuffer<float>tempDry;
        float* dryPtr = nullptr;
        float* const* out = nullptr;
        float* in= nullptr;
        std::atomic<bool> isProcessing; 
         };

#endif //_AUDIOCALLBACK_H_
