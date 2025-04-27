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
    bsFloat = bs * sizeof(float);
    // Size of result (int)
    convResSize = bs * 2;
    // Size of result (bytes)
    convResSizeFloat = convResSize * sizeof(float);
    
    // Get number of partitions (round up)
    partitions = (impulseResponseSize / bs) + 1;
    std::cout << partitions << std::endl;
    // Get padded size
    paddedSize = partitions * bs;
    // Padded Size in bytes
    paddedSizeFloat = paddedSize * sizeof(float);

    allocateOnDevice();

    

    // Copy contents to GPU
    memcpy(_impulseResponse->contents(), impulseResponse, impulseResponseSize * sizeof(float));

    // Allocate host memory
    convResBuffer = (float*)calloc(convResSize, sizeof(float));
    overLapBuffer = (float*)calloc(bs, sizeof(float));

    // Get the sizes
    uint sizes[2] = { static_cast<uint>(bs), static_cast<uint>(convResSize)};
    
    // Allocate memory for _sizesbuffer
    _sizes = _pDevice->newBuffer(sizeof(uint) * 2, MTL::ResourceStorageModeShared);
    memcpy(_sizes->contents(), sizes, sizeof(uint) * 2);

    // Create a command queue
    createCommandQueue();
    
    // get Functions from .metallib
    createDefaultLibrary();
    
    //create Compute Pipelines
    createComputePipeLine();
    
    //set gridsize and the number of threads and total size of shared memory
    gridSize = MTL::Size::Make(paddedSize,1,1);
    numberOfThreads = MTL::Size::Make(bs,1,1);
    totalSharedMemorySize = bs * 4 * sizeof(float);
}

void ConvEngine::freeMemory() {
    
    free(overLapBuffer);
    free(convResBuffer);
    
    _impulseResponse->release();
    _dryBuffer->release();
    _timeDomainBuffer->release();
    _resultBuffer->release();
    _CommandBuffer->release();
    _mCommandQueue->release();
    _convolutionPipeline->release();
    _shiftAndInsertPipeline->release();
    _sizes->release();
    _library->release();
    _pDevice->release();
}


ConvEngine::~ConvEngine()
{
    freeMemory();
}
void ConvEngine::render(const float* input, float* output)
{
    
    // Copy input data to the dry buffer
    memcpy(_dryBuffer->contents(), input, bsFloat);
    
    //compute
    @autoreleasepool {
        sendComputeCommandCommand();
   }
    //copy results to host
    memcpy(convResBuffer,_resultBuffer->contents(), convResSizeFloat);
    
    
    
    for (int i = 0; i < bs; i++) {
        output[i] = (convResBuffer[i] + overLapBuffer[i]) * 0.015f;
        overLapBuffer[i] = convResBuffer[bs + i];
    }
    
    //reset result buffer
    memset(_resultBuffer->contents(), 0, convResSizeFloat);
    
    
}


void ConvEngine::initDevice() {
    // Get the device (GPU)
    _pDevice = MTL::CreateSystemDefaultDevice();
}


void ConvEngine::allocateOnDevice() {
    
   
    _dryBuffer = _pDevice->newBuffer(bsFloat, MTL::ResourceStorageModeShared);
    _timeDomainBuffer = _pDevice->newBuffer(paddedSizeFloat, MTL::ResourceStorageModeShared);
    
    _impulseResponse = _pDevice->newBuffer(paddedSizeFloat, MTL::ResourceStorageModeShared);
   
    _resultBuffer = _pDevice->newBuffer(convResSizeFloat, MTL::ResourceStorageModeShared);
    
    clear();
    
    
}

void ConvEngine::clear() {
    memset(_dryBuffer->contents(), 0, bsFloat);
    memset(_timeDomainBuffer->contents(), 0, paddedSizeFloat);
    memset(_impulseResponse->contents(), 0, paddedSizeFloat);
    memset(_resultBuffer->contents(), 0, convResSizeFloat);
    
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
    
 
    // First compute operation: Shift and Insert
    {
        // Create and configure the first encoder
        auto shiftAndInsert = _CommandBuffer->computeCommandEncoder();
        shiftAndInsert->setComputePipelineState(_shiftAndInsertPipeline);
        shiftAndInsert->setBuffer(_timeDomainBuffer, 0, 0);  // _timeDomainBuffer buffer
        shiftAndInsert->setBuffer(_dryBuffer, 0, 1);          // _dryBuffer buffer
        shiftAndInsert->setBuffer(_sizes, 0, 2);              // Size buffer

        // Dispatch the compute command
        shiftAndInsert->dispatchThreads(gridSize, numberOfThreads);
        
        // End encoding for this encoder
        shiftAndInsert->endEncoding();
    }
  
   // _CommandBuffer2 = _mCommandQueue->commandBuffer();
    

    // Second compute operation: Convolution
    {
        auto sharedConvolution = _CommandBuffer->computeCommandEncoder();
        sharedConvolution->setComputePipelineState(_convolutionPipeline);
        sharedConvolution->setBuffer(_resultBuffer, 0, 0);         // Result input buffer
        sharedConvolution->setBuffer(_timeDomainBuffer, 0, 1);     // Dry response buffer
        sharedConvolution->setBuffer(_impulseResponse, 0, 2);      // Impulse Response buffer
        sharedConvolution->setBuffer(_sizes, 0, 3);                 // Sizes buffer
        sharedConvolution->setThreadgroupMemoryLength(totalSharedMemorySize, 0);
        
        
        sharedConvolution->dispatchThreads(gridSize, numberOfThreads);

        // End encoding for this encoder
        sharedConvolution->endEncoding();
    }
  
    _CommandBuffer->commit();
    _CommandBuffer->waitUntilCompleted();
    
}
