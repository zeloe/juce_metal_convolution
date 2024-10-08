#ifndef CONV_ENGINE_H
#define CONV_ENGINE_H

#include <iostream>
#include <cassert>
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
class ConvEngine
{
public:
    ConvEngine();
    ConvEngine(float* impulseResponse, int maxBufferSize, int impulseResponseSize);
    ~ConvEngine();
    
    void render(float* input);
    float* result = nullptr;
private:
    MTL::Buffer* _impulseResponse = nullptr ;
    MTL::Buffer* _sizes = nullptr ;
    MTL::Buffer* _dryBuffer = nullptr ;
    MTL::Buffer* _timeDomainBuffer = nullptr;
    MTL::Buffer* _resultBuffer = nullptr;
    MTL::Device* _pDevice = nullptr;
    MTL::CommandBuffer* _CommandBuffer = nullptr;
    MTL::CommandQueue* _mCommandQueue = nullptr;
    // Get the main bundle
    NS::Bundle* _bundle = NS::Bundle::mainBundle();
   
    MTL::Library* _library = nullptr;
    MTL::Function* _convolution = nullptr;
    MTL::Function* _shift_and_insert = nullptr;
    MTL::ComputePipelineState* _pipeLine = nullptr;
    
    
    int bs = 0;
    int sizeBs = 0;
    int convResSize = 0;
    int sizeConvResSize = 0;
    int paddedSize = 0;
    int sizePaddedSize = 0;
    int partitions = 0;
    float* convResBuffer = nullptr;
    float* overLapBuffer = nullptr;
};



#endif //CONV_ENGINE_H
