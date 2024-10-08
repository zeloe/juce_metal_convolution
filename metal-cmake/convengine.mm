#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include "convengine.h"

const char* metalLibPath = METAL_LIBRARY_PATH; // Access the path defined in CMake

ConvEngine::ConvEngine()
{
    
}
ConvEngine::ConvEngine(float* impulseResponse, int maxBufferSize, int impulseResponseSize) {
    // Get max buffer size
    bs = maxBufferSize;
    
    // Get sizes for copying
    sizeBs = bs * sizeof(float);
    convResSize = bs * 2 - 1;
    sizeConvResSize = convResSize * sizeof(float);
    
    // Get number of partitions
    partitions = (impulseResponseSize + bs - 1) / bs; // Ensure correct rounding up

    // Get padded size
    paddedSize = partitions * bs;
    sizePaddedSize = paddedSize * sizeof(float);

    // Get the device (GPU)
    _pDevice = MTL::CreateSystemDefaultDevice();
    
    // Allocate memory on GPU
    _impulseResponse = _pDevice->newBuffer(sizePaddedSize, MTL::ResourceStorageModeShared);
    _timeDomainBuffer = _pDevice->newBuffer(sizePaddedSize, MTL::ResourceStorageModeShared);
    _dryBuffer = _pDevice->newBuffer(sizeBs, MTL::ResourceStorageModeShared);
    _resultBuffer = _pDevice->newBuffer(sizeConvResSize, MTL::ResourceStorageModeShared);

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
    uint sizes[3] = { static_cast<uint>(bs), static_cast<uint>(partitions), static_cast<uint>(paddedSize) };
    
    // Allocate memory for _sizesbuffer
    _sizes = _pDevice->newBuffer(sizeof(sizes), MTL::ResourceStorageModeShared);
    memcpy(_sizes->contents(), sizes, sizeof(sizes));

    // Create a command queue
    _mCommandQueue = _pDevice->newCommandQueue();

    // Load the default library from the bundle
    NS::Error* pError = nullptr; // To capture any errors
    NS::String* filePath = NS::String::string(metalLibPath, NS::UTF8StringEncoding);
    _library = _pDevice->newLibrary(filePath, &pError);

    if (!_library) {
        std::cerr << "Failed to load default Metal library: " << pError->localizedDescription()->utf8String() << std::endl;
        throw std::runtime_error("Default library loading failed.");
    }

    // Create compute functions
    _convolution = _library->newFunction(NS::String::string("shared_partitioned_convolution4", NS::UTF8StringEncoding));
    _shift_and_insert = _library->newFunction(NS::String::string("shiftAndInsertKernel", NS::UTF8StringEncoding));

    // Create the pipeline state for the shift and insert function
    NS::Error* shiftandinsertError = nullptr;
    _shiftAndInsertPipeline = _pDevice->newComputePipelineState(_shift_and_insert, &shiftandinsertError);

    if (!_shiftAndInsertPipeline) {
        std::cerr << "Failed to create shift and insert pipeline state: " << shiftandinsertError->localizedDescription()->utf8String() << std::endl;
        throw std::runtime_error("Shift and insert pipeline creation failed.");
    }

    // Create the pipeline state for the convolution function
    NS::Error* convolutionError = nullptr;
    _convolutionPipeline = _pDevice->newComputePipelineState(_convolution, &convolutionError);

    if (!_convolutionPipeline) {
        std::cerr << "Failed to create convolution pipeline state: " << convolutionError->localizedDescription()->utf8String() << std::endl;
        throw std::runtime_error("Convolution pipeline creation failed.");
    }
    // Define thread group size (number of threads per threadgroup)
     threadGroupSize = MTL::Size::Make(bs, 1, 1);

    // Define grid size (total number of threads = numBerOFSubPartitions * bs)
     gridSize = MTL::Size::Make(partitions, 1, 1);
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
    _pipeLine->release();

    //
    
    //
    _convolutionPipeline->release();
    _shiftAndInsertPipeline->release();
}
void ConvEngine::render(float* input)
{
    // Copy input data to the dry buffer
    memcpy(_dryBuffer->contents(), input, sizeBs);
    
    // Create Command Buffer
    _CommandBuffer = _mCommandQueue->commandBuffer();

    // First compute operation: Shift and Insert
    {
        // Create and configure the first encoder
        auto encoder = _CommandBuffer->computeCommandEncoder();
        encoder->setComputePipelineState(_shiftAndInsertPipeline);
        encoder->setBuffer(_timeDomainBuffer, 0, 0);  // Input buffer
        encoder->setBuffer(_dryBuffer, 0, 1);          // Output buffer
        encoder->setBuffer(_sizes, 0, 2);              // Size buffer

        // Dispatch the compute command
        encoder->dispatchThreads(gridSize, threadGroupSize);
        
        // End encoding for this encoder
        encoder->endEncoding();
    }

    // Second compute operation: Convolution
    {
        auto encoder2 = _CommandBuffer->computeCommandEncoder();
        encoder2->setComputePipelineState(_convolutionPipeline);
        encoder2->setBuffer(_resultBuffer, 0, 0);         // Result input buffer
        encoder2->setBuffer(_timeDomainBuffer, 0, 1);     // Dry response buffer
        encoder2->setBuffer(_impulseResponse, 0, 2);      // Impulse Response buffer
        encoder2->setBuffer(_sizes, 0, 3);                 // Sizes buffer
        
        uint totalSharedMemorySize = bs * 2 * sizeof(float);  // For arr1 and arr2
        encoder2->setThreadgroupMemoryLength(totalSharedMemorySize, 0);
        
        // Dispatch the compute command
        encoder2->dispatchThreads(gridSize, threadGroupSize);

        // End encoding for this encoder
        encoder2->endEncoding();
    }

    // Commit the command buffer and wait for completion
    _CommandBuffer->commit();
    _CommandBuffer->waitUntilCompleted();
    
    // Copy results back to convResBuffer
    memcpy(convResBuffer, _resultBuffer->contents(), sizeConvResSize);
    
    // Process results
    for (int i = 0; i < bs; i++) {
        result[i] = (convResBuffer[i] + overLapBuffer[i]) * 0.015f;
        overLapBuffer[i] = convResBuffer[bs + i - 1];
    }
    
    // Reset the _resultBuffer contents to zero
    void* bufferContents = _resultBuffer->contents(); // Get the pointer to the buffer contents
    memset(bufferContents, 0, sizeConvResSize); // Use memset to fill the buffer with zeros
}
