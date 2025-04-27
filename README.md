# juce_metal_convolution
 Linear Convolution using Metal (GPU) on the audiothread.

## How to build
You need cmake and commandline tools for Xcode to build. 

```shell
  git clone https://github.com/zeloe/juce_metal_convolution.git
```
You need to download [metal-cpp](https://developer.apple.com/metal/cpp/) form apple. \
Extract and move metal-cpp folder inside metal-cmake. 
```shell
  juce_metal_convolution/metal-cmake/  <- copy metal-cpp here
```
Then use cmake to build with Xcode:
```shell
 cmake -B build -G Xcode
```
or 
```shell
 cmake -B build
 cd build
 make 
```

## How it works
This holds all values for convolution. \
Insert and shift kernel copies new buffer at beggining. \
All other content gets shifted by buffersize. \
Content at end of Time Domain Buffer gets discarded. \
Then partitioned convolution is done and you should hear a convolution reverb. \


