#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include "convengine.h"

const char* metalLibPath = METAL_LIBRARY_PATH; // Access the path defined in CMake

ConvEngine::ConvEngine(){
    
    
}
ConvEngine::ConvEngine(float* impulseResponse, int maxBufferSize, int impulseResponseSize) {
    
    initDevice();
    // Get max buffer size
    bs = maxBufferSize;
    
    // Size of BlockSize (bytes)
    sizeBs = bs * sizeof(float);
    // Size of result (int)
    convResSize = bs * 2;
    // Size of result (bytes)
    sizeConvResSize = convResSize * sizeof(float);
    
    // Get number of partitions (round up)
    partitions = (impulseResponseSize / bs) + 1;
    // Get padded size
    paddedSize = partitions * bs;
    // Padded Size in bytes
    sizePaddedSize = paddedSize * sizeof(float);

    allocateOnDevice();
    _timeDomainBuffer2 = _pDevice->newBuffer(sizePaddedSize, MTL::ResourceStorageModeShared);
    

    // Initialize GPU buffers to zero (or use MTL::ResourceStorageModePrivate if not needed on the CPU)
    memset(_impulseResponse->contents(), 0, sizePaddedSize);
    memcpy(_impulseResponse->contents(), impulseResponse, impulseResponseSize * sizeof(float));

    // Initialize other buffers
    memset(_dryBuffer->contents(), 0, sizeBs);
    memset(_timeDomainBuffer->contents(), 0, sizePaddedSize);
    memset(_resultBuffer->contents(), 0, sizeConvResSize);

    // Allocate host memory
    result = (float*)calloc(bs, sizeof(float));
    convResBuffer = (float*)calloc(convResSize, sizeof(float));
    overLapBuffer = (float*)calloc(bs, sizeof(float));

    // Get the sizes
    uint sizes[3] = { static_cast<uint>(bs), static_cast<uint>(convResSize), static_cast<uint>(0) };
    
    // Allocate memory for _sizesbuffer
    _sizes = _pDevice->newBuffer(sizeof(uint) * 3, MTL::ResourceStorageModeShared);
    memcpy(_sizes->contents(), sizes, sizeof(uint) * 3);

    // Create a command queue
    createCommandQueue();
    
    // get Functions from .metallib
    createDefaultLibrary();
    
    //create Compute Pipelines
    createComputePipeLine();
    
  
  
}




ConvEngine::ConvEngine(int maxBufferSize) {
    
    initDevice();
    // Get max buffer size
    bs = maxBufferSize;
    
    // Size of BlockSize (bytes)
    sizeBs = bs * sizeof(float);
    // Size of result (int)
    convResSize = bs * 2;
    // Size of result (bytes)
    sizeConvResSize = convResSize * sizeof(float);
    
    paddedSize = ((48000 / bs) + 1) * bs;

    // Padded Size in bytes
    sizePaddedSize = paddedSize * sizeof(float);

    // Allocate memory on GPU
   
    _dryBuffer = _pDevice->newBuffer(sizeBs, MTL::ResourceStorageModeShared);
    _timeDomainBuffer = _pDevice->newBuffer(sizePaddedSize, MTL::ResourceStorageModeShared);
    _timeDomainBuffer2 = _pDevice->newBuffer(sizePaddedSize, MTL::ResourceStorageModeShared);
    
    _impulseResponse = _pDevice->newBuffer(sizeBs, MTL::ResourceStorageModeShared);
    _dryBuffer = _pDevice->newBuffer(sizeBs, MTL::ResourceStorageModeShared);
    _resultBuffer = _pDevice->newBuffer(sizeConvResSize, MTL::ResourceStorageModeShared);
    
    
    // Initialize other buffers
    memset(_dryBuffer->contents(), 0, sizeBs);
    memset(_timeDomainBuffer->contents(), 0, sizePaddedSize);
    memset(_timeDomainBuffer2->contents(), 0, sizePaddedSize);
    memset(_impulseResponse->contents(), 0, sizeBs);
    memset(_timeDomainBuffer2->contents(), 0, sizePaddedSize);
    memset(_resultBuffer->contents(), 0, sizeConvResSize);
   

  

    // Allocate host memory
    result = (float*)calloc(bs, sizeof(float));
    convResBuffer = (float*)calloc(convResSize, sizeof(float));
    overLapBuffer = (float*)calloc(bs, sizeof(float));

    // Get the sizes
    uint sizes[3] = { static_cast<uint>(bs), static_cast<uint>(convResSize), static_cast<uint>(0) };
    
    // Allocate memory for _sizesbuffer
    _sizes = _pDevice->newBuffer(sizeof(uint) * 3, MTL::ResourceStorageModeShared);
    memcpy(_sizes->contents(), sizes, sizeof(uint) * 3);

    // Create a command queue
    createCommandQueue();
    
    // get Functions from .metallib
    createDefaultLibrary();
    
    //create Compute Pipelines
    createComputePipeLine();
    
  
  
}



ConvEngine::~ConvEngine()
{
    free(result);
    free(overLapBuffer);
    free(convResBuffer);
    // Clean up Metal resources
    _pDevice->release();
    _impulseResponse->release();
    _dryBuffer->release();
    _timeDomainBuffer->release();
    _resultBuffer->release();
    _sizes->release();
    _mCommandQueue->release();
    //
    _convolution->release();
    _shift_and_insert->release();
    _library->release();
    
    _convolutionPipeline->release();
    _shiftAndInsertPipeline->release();
}
void ConvEngine::render(float* input, float* output)
{
    
   // int bsOffset = offset * sizeBs;
    
    // Copy input data to the dry buffer
    memcpy(static_cast<float*>(_dryBuffer->contents()), input, sizeBs);
    
    //compute
    @autoreleasepool {
        sendComputeCommandCommand();
   }
    //copy results to host
    memcpy(convResBuffer,static_cast<float*>( _resultBuffer->contents()), sizeConvResSize);
    
    
    
    for (int i = 0; i < bs; i++) {
        output[i] = (convResBuffer[i] + overLapBuffer[i]) * 0.15f;
        overLapBuffer[i] = convResBuffer[bs + i];
    }
    
    
    
    //reset result buffer
    memset(static_cast<float*>(_resultBuffer->contents()), 0, sizeConvResSize);
    
    
}


void ConvEngine::initDevice() {
    // Get the device (GPU)
    _pDevice = MTL::CreateSystemDefaultDevice();
}


void ConvEngine::allocateOnDevice() {
    
   
    _dryBuffer = _pDevice->newBuffer(sizeBs, MTL::ResourceStorageModeShared);
    _timeDomainBuffer = _pDevice->newBuffer(sizePaddedSize, MTL::ResourceStorageModeShared);
    
    _impulseResponse = _pDevice->newBuffer(sizePaddedSize, MTL::ResourceStorageModeShared);
   
    _resultBuffer = _pDevice->newBuffer(sizeConvResSize, MTL::ResourceStorageModeShared);
    
    // Initialize other buffers
    memset(_dryBuffer->contents(), 0, sizeBs);
    memset(_timeDomainBuffer->contents(), 0, sizePaddedSize);
    memset(_impulseResponse->contents(), 0, sizePaddedSize);
    memset(_resultBuffer->contents(), 0, sizeConvResSize);
    
}

void ConvEngine::createDefaultLibrary() {
    // Load the default library from the bundle
    NS::Error* pError = nullptr; // To capture any errors
    NS::String* filePath = NS::String::string(metalLibPath, NS::ASCIIStringEncoding);
    _library = _pDevice->newLibrary(filePath, &pError);
    if (!_library) {
        std::cerr << "Failed to load Metal library" ;
    }
}


void ConvEngine::createCommandQueue() {
    
    _mCommandQueue  = _pDevice->newCommandQueue();
    
}


void ConvEngine::createComputePipeLine() {
    
    /*
    NS::Array *names = _library->functionNames();
    
    for (NSUInteger i = 0; i < names->count(); ++i) {
        NS::String* nameObj = static_cast<NS::String*>(names->object(i));
        std::string func_name = nameObj->utf8String();
        std::cout << func_name << std::endl;
    }
    */
   
    MTL::Function *shift_and_insert = _library->newFunction(NS::String::string("shiftAndInsertKernel",NS::ASCIIStringEncoding));
    
    assert(shift_and_insert);
    
    MTL::Function* convolution = _library->newFunction(NS::String::string("shared_partitioned_convolution", NS::ASCIIStringEncoding));
    
    assert(convolution);
    
    
    NS::Error* shift_error;
    
   
    _shiftAndInsertPipeline = _pDevice->newComputePipelineState(shift_and_insert,&shift_error);
    
    assert(_shiftAndInsertPipeline);
    
    NS::Error* convolution_error;
    _convolutionPipeline = _pDevice->newComputePipelineState(convolution,&convolution_error);
    
    assert(_convolutionPipeline);
    
    
    shift_and_insert->release();
    convolution->release();
}


void ConvEngine::sendComputeCommandCommand() {
    
    
    _CommandBuffer = _mCommandQueue->commandBuffer();
    MTL::Size numThreadgroups = MTL::Size::Make(partitions,1,1);
    MTL::Size threadGroupSize = MTL::Size::Make(bs,1,1);
    // First compute operation: Shift and Insert
    {
        // Create and configure the first encoder
        auto encoder = _CommandBuffer->computeCommandEncoder();
        encoder->setComputePipelineState(_shiftAndInsertPipeline);
        encoder->setBuffer(_timeDomainBuffer, 0, 0);  // _timeDomainBuffer buffer
        encoder->setBuffer(_dryBuffer, 0, 1);          // _dryBuffer buffer
        encoder->setBuffer(_sizes, 0, 2);              // Size buffer

        // Dispatch the compute command
        encoder->dispatchThreads(numThreadgroups, threadGroupSize);
        
        // End encoding for this encoder
        encoder->endEncoding();
    }
  
   // _CommandBuffer2 = _mCommandQueue->commandBuffer();
    

    // Second compute operation: Convolution
    {
        auto encoder2 = _CommandBuffer->computeCommandEncoder();
        encoder2->setComputePipelineState(_convolutionPipeline);
        encoder2->setBuffer(_resultBuffer, 0, 0);         // Result input buffer
        encoder2->setBuffer(_timeDomainBuffer, 0, 1);     // Dry response buffer
        encoder2->setBuffer(_impulseResponse, 0, 2);      // Impulse Response buffer
        encoder2->setBuffer(_sizes, 0, 3);                 // Sizes buffer
      //  encoder2->setBuffer(_timeDomainBuffer2, 0, 4);
        //uint totalSharedMemorySize = bs * 4 * sizeof(float);  // For arr1 and arr2
       //encoder2->setThreadgroupMemoryLength(totalSharedMemorySize, 0);
        
        
        encoder2->dispatchThreads(numThreadgroups, threadGroupSize);

        // End encoding for this encoder
        encoder2->endEncoding();
    }
    _CommandBuffer->commit();
    _CommandBuffer->waitUntilCompleted();
    
    
}
