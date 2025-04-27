 

#include "audiocallback.h"



MyAudioCallback::MyAudioCallback(ConvEngine* _engine, int maxBufferSize, float* dryPtr, int drySize) : juce::Thread("GpuThread")
{
    metal_engine = _engine;
    tempDry.setSize(1, maxBufferSize);
    tempDry.clear();
    resBuffer.setSize(1,maxBufferSize);
    resBuffer.clear();
    bs = maxBufferSize;
    this->dryPtr = dryPtr;
    this->drySize = drySize;
    isProcessing.store(false);
    startThread(Priority::highest);
}



MyAudioCallback::~MyAudioCallback()  
{
        
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
        dry[i] =  dryPtr[counter];
        counter++;
        if (counter >= drySize) {
            counter = 0;
        }
    }
       
    isProcessing.store(true);
    while(isProcessing.load()) {};
    
    
    float* ptr_L = outputChannelData[0];
    float* ptr_R = outputChannelData[1];
    float* res = resBuffer.getWritePointer(0);
    for (int i = 0; i < numSamples; i++) {
        ptr_L[i] = res[i] * 0.25f;
        ptr_R[i] = res[i] * 0.25f;
    }
        

}

void MyAudioCallback::run() {
    while (!threadShouldExit()) {
        if(isProcessing.load()) {
            metal_engine->render(tempDry.getWritePointer(0),resBuffer.getWritePointer(0));
            isProcessing.store(false);
        }
    }
}


      void 	MyAudioCallback::audioDeviceAboutToStart(AudioIODevice* device)   {};

      void 	MyAudioCallback::audioDeviceStopped()   {};


      void 	MyAudioCallback::audioDeviceError(const String& errorMessage)  
    {
        std::cout << errorMessage << std::endl;
    };



     
