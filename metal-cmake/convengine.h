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
    
    void render(const float* input, float* output);
    void init();
    
private:
    void initDevice();
    
    void allocateOnDevice();
    
    void createDefaultLibrary();
    void createCommandQueue();
    void createComputePipeLine();
    void encodeComputeCommand(MTL::ComputeCommandEncoder* computeEncoder);
    void sendComputeCommandCommand();
    void clear();
    void freeMemory();
    
    MTL::Device* _pDevice;
    
    MTL::Buffer* _impulseResponse ;
    MTL::Buffer* _sizes ;
    MTL::Buffer* _dryBuffer ;
    MTL::Buffer* _timeDomainBuffer;
    MTL::Buffer* _resultBuffer;
    
    MTL::CommandBuffer* _CommandBuffer;
    MTL::CommandQueue* _mCommandQueue;
    
    MTL::Library* metalDefaultLibrary;
    MTL::Library* _library;

    MTL::ComputePipelineState* _pipeLine;
    
    //
    MTL::ComputePipelineState* _convolutionPipeline;
    MTL::ComputePipelineState* _shiftAndInsertPipeline;
    //
    int offset = 0;
    int bs = 0;
    int bsFloat = 0;
    int convResSize = 0;
    int convResSizeFloat = 0;
    int paddedSize = 0;
    int paddedSizeFloat = 0;
    int partitions = 0;
    float* convResBuffer = nullptr;
    float* overLapBuffer = nullptr;
    uint totalSharedMemorySize = 0;
    //
    MTL::Size gridSize;
    MTL::Size numberOfThreads;
};



#endif //CONV_ENGINE_H
