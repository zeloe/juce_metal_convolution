# juce_metal_convolution

Real-time convolution reverb without FFT, accelerated using Metal on the GPU.  \
Convolution processing runs on a dedicated background thread, producing a high-quality, natural reverb effect. \
Tested on a MacBook Air M4.

---

## Features

- Real-time, time-domain convolution (no FFT)
- GPU-accelerated with Metal
- Background-threaded processing
- CMake-based build system (easy setup)

---

## How to Build

You need:

- CMake installed
- Xcode with command-line tools installed

### 1. Clone the repository

```bash
git clone https://github.com/zeloe/juce_metal_convolution.git
```

### 2. Download `metal-cpp`

Download [metal-cpp](https://developer.apple.com/metal/cpp/) from Apple.  
Extract it, and copy the `metal-cpp` folder inside the `metal-cmake` directory:

```text
juce_metal_convolution/metal-cmake/  ← copy metal-cpp here
```

> Note: You must have the `metal-cpp` headers locally; they are not included in this repo.

### 3. Build using CMake

To generate an Xcode project:

```bash
cmake -B build -G Xcode
```

Or to build directly using Make:

```bash
cmake -B build
cd build
make
```

---

## Notes

- This project uses [JUCE](https://juce.com/) for audio setup and Metal interop.
- Metal is used directly via [metal-cpp](https://developer.apple.com/metal/cpp/), Apple's official C++ bindings for Metal.
- Currently tested only on Apple Silicon (MacBook Air M4).
- Check out [here](https://github.com/zeloe/RTConvolver) VST3 Plugin to use in DAW. 
---
