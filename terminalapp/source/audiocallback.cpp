 

#include "audiocallback.h"

MyAudioCallback::MyAudioCallback(float* impulseResponseBufferData, int maxBufferSize, int impulseResponseSize, float* dryPtr, int drySize) :juce::Thread("GPUThread") {
    resBuffer.setSize(1,maxBufferSize);
    
    resBuffer.clear();
    tempDry.setSize(1, maxBufferSize);
    tempDry.clear();
    bs = maxBufferSize;
    this->dryPtr = dryPtr;
    this->drySize = drySize;
     

}

MyAudioCallback::MyAudioCallback(ConvEngine* _engine, int maxBufferSize, float* dryPtr, int drySize):  juce::Thread("GPUThread") {
    metal_engine = _engine;
    tempDry.setSize(1, maxBufferSize);
    tempDry.clear();
    resBuffer.setSize(1,maxBufferSize);
    resBuffer.clear();
    bs = maxBufferSize;
    this->dryPtr = dryPtr;
    this->drySize = drySize;
    processingInBackground.store(false, std::memory_order_release);
    startThread(Priority::highest);
}



MyAudioCallback::~MyAudioCallback()  
    {
        stopThread(2000);
        
    };
 
void MyAudioCallback::audioDeviceIOCallbackWithContext(const float* const* inputChannelData,
        int	numInputChannels,
        float* const* outputChannelData,
        int	numOutputChannels,
        int	numSamples,
        const AudioIODeviceCallbackContext& context)  
    {
        //copy Slice
        
        auto dry = tempDry.getWritePointer(0);



        for (int i = 0; i < numSamples; i++) {
            tempDry.setSample(0, i, dryPtr[counter]);
            counter++;
            if (counter >= drySize) {
                counter = 0;
            }
        }
    processingInBackground.store(true, std::memory_order_release);
        //
    while(processingInBackground.load(std::memory_order_acquire)) {};
    float* res = resBuffer.getWritePointer(0);
        float* ptr_L = outputChannelData[0];
        float* ptr_R = outputChannelData[1];
        for (int i = 0; i < numSamples; i++) {
            ptr_L[i] = res[i];
            ptr_R[i] = res[i];

        }
        

    } 
void MyAudioCallback::prepare(juce::AudioBuffer<float>& dry, juce::AudioBuffer<float>& imp, int bufferSize)
    {

       









    }
 
void MyAudioCallback::run() {
    while(!threadShouldExit()) {
        if (processingInBackground.load(std::memory_order_acquire)) {
            metal_engine->render(tempDry.getWritePointer(0),resBuffer.getWritePointer(0));
            processingInBackground.store(false, std::memory_order_release);
        }
    }
    
    
    
}


      void 	MyAudioCallback::audioDeviceAboutToStart(AudioIODevice* device)   {};

      void 	MyAudioCallback::audioDeviceStopped()   {};


      void 	MyAudioCallback::audioDeviceError(const String& errorMessage)  
    {
        std::cout << errorMessage << std::endl;
    };



     
