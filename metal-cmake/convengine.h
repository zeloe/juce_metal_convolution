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
    ConvEngine(int maxBufferSize);
    ~ConvEngine();
    
    void render(float* input, float* output);
    void init();
    
    float* result = nullptr;
private:
    void initDevice();
    
    void allocateOnDevice();
    
    void createDefaultLibrary();
    void createCommandQueue();
    void createComputePipeLine();
    void encodeComputeCommand(MTL::ComputeCommandEncoder* computeEncoder);
    void sendComputeCommandCommand();
    MTL::Buffer* _impulseResponse ;
    MTL::Buffer* _timeDomainBuffer2 ;
    MTL::Buffer* _sizes ;
    MTL::Buffer* _dryBuffer ;
    MTL::Buffer* _timeDomainBuffer;
    MTL::Buffer* _resultBuffer;
    MTL::Device* _pDevice;
    MTL::CommandBuffer* _CommandBuffer;
    MTL::CommandBuffer* _CommandBuffer2;
    MTL::CommandQueue* _mCommandQueue;
    
    MTL::CounterSampleBuffer* test;
    MTL::Library* metalDefaultLibrary;
    MTL::Library* _library;
    MTL::Function* _convolution;
    MTL::Function* _shift_and_insert;
    MTL::ComputePipelineState* _pipeLine;
    
    //
    MTL::ComputeCommandEncoder* encoder1;
    MTL::ComputeCommandEncoder* encoder2;
    //
    MTL::ComputePipelineState* _convolutionPipeline;
    MTL::ComputePipelineState* _shiftAndInsertPipeline;
    //
    int offset = 0;
    
    int bs = 0;
    int sizeBs = 0;
    int convResSize = 0;
    int sizeConvResSize = 0;
    int paddedSize = 0;
    int sizePaddedSize = 0;
    int partitions = 0;
    float* convResBuffer = nullptr;
    float* overLapBuffer = nullptr;
    
    //
    MTL::Size gridSize;
    MTL::Size threadGroupSize;
};



#endif //CONV_ENGINE_H
