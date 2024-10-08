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
    //get max buffer size
    bs = maxBufferSize;
    //get size for copying
    sizeBs = bs * sizeof(float);
   
    convResSize = bs * 2 - 1;
    //get size for copying
    sizeConvResSize =  convResSize * sizeof(float);
    //get num partitions
    partitions = (impulseResponseSize / bs) + 1;
    //get paddedSize
    paddedSize = partitions * bs;
    //get size for copying
    sizePaddedSize = paddedSize * sizeof(float);
    //get the device (GPU)
    _pDevice = MTL::CreateSystemDefaultDevice();
    //allocate memory on GPU for _impulseResponse
    _impulseResponse = _pDevice->newBuffer(sizePaddedSize, MTL::ResourceStorageModeManaged);
    //allocate memory on GPU for _timeDomainBuffer
    _timeDomainBuffer = _pDevice->newBuffer(sizePaddedSize, MTL::ResourceStorageModeManaged);
    //allocate memory on GPU for _dryBuffer
    _dryBuffer = _pDevice->newBuffer(sizeBs, MTL::ResourceStorageModeManaged );
    //allocate memory on GPU for _resultBuffer
    _resultBuffer = _pDevice->newBuffer(sizeConvResSize, MTL::ResourceStorageModeManaged );
    float zeroArray[paddedSize];
    for(int i = 0; i < paddedSize; i++) {
        zeroArray[i] = 0.f;
    }
    //set contents to 0
    memcpy( _impulseResponse->contents(), zeroArray, sizePaddedSize);
    //copy contents of impulse response
    memcpy( _impulseResponse->contents(), impulseResponse, sizePaddedSize);
    //set contents to 0
    memcpy( _dryBuffer->contents(), zeroArray, sizeBs);
    //set contents to 0
    memcpy( _timeDomainBuffer->contents(), zeroArray, sizePaddedSize);
    //set contents to 0
    memcpy(_resultBuffer,zeroArray,sizeBs);
    result = (float*)calloc(convResSize, sizeof(float));
    convResBuffer = (float*)calloc(convResSize, sizeof(float));
    overLapBuffer = (float*)calloc(bs, sizeof(float));
    //get the sizes
    uint sizes[3];
    sizes[0] = bs; //blocksize
    sizes[1] = partitions; //number of partitions
    sizes[2] = paddedSize; //paddedsize
    //allocate memory for _sizesbuffer
    _sizes = _pDevice->newBuffer(3 * sizeof(uint), MTL::ResourceStorageModeManaged );
    //copy content
    memcpy( _sizes->contents(), sizes, 3 * sizeof(uint));
    // Create a command queue
    _mCommandQueue = _pDevice->newCommandQueue();
    // Load the default library from the bundle
    // Create the bundle
    NS::Error* pError = nullptr; // To capture any errors
    // Create a string for the library path
    NS::String* filePath = NS::String::string(metalLibPath, NS::UTF8StringEncoding);
    _library = _pDevice->newLibrary(filePath, &pError);
            if (!_library) {
                std::cerr << "Failed to load default Metal library: " << pError->localizedDescription()->utf8String() << std::endl;
                throw std::runtime_error("Default library loading failed.");
            }
    
    _convolution = _library->newFunction(NS::String::string("shared_partitioned_convolution4", NS::UTF8StringEncoding));
    
    _shift_and_insert = _library->newFunction(NS::String::string("shiftAndInsertKernel", NS::UTF8StringEncoding));
    
    NS::Error* error = nullptr;
     _pipeLine = _pDevice->newComputePipelineState(_convolution, &error);
    if (error) {
        std::cerr << "Error creating pipeline state: " << error->localizedDescription()->utf8String() << std::endl;
            return;
    }
    
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
    _bundle->release();
}

void ConvEngine::render(float* input)
{
    memcpy( _dryBuffer->contents(), input, sizeBs);
    
    
    
    
    
    
    memcpy(convResBuffer,_resultBuffer->contents(), sizeConvResSize);
    
    for(int i = 0; i < bs; i++) {
        result[i] = (convResBuffer[i] + overLapBuffer[i]) * 0.015f;
        overLapBuffer[i] = convResBuffer[bs + i - 1];
    }
    
}
