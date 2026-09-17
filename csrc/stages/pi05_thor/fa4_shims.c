// Runtime shims the CuTe-DSL AOT object expects: stable "_cuda*" aliases of
// the CUDA runtime/driver entry points it calls.
#include <cuda_runtime.h>
#include <cuda.h>

cudaError_t _cudaGetDevice(int * dev) {
    return cudaGetDevice(dev);
}

cudaError_t _cudaDeviceGetAttribute(int * value, enum cudaDeviceAttr attr, int device) {
    return cudaDeviceGetAttribute(value, attr, device);
}

cudaError_t _cudaFuncSetAttribute(const void * func, enum cudaFuncAttribute attr, int value) {
    return cudaFuncSetAttribute(func, attr, value);
}

cudaError_t _cudaKernelSetAttributeForDevice(cudaKernel_t kernel, enum cudaFuncAttribute attr,
                                             int value, int device) {
    return cudaKernelSetAttributeForDevice(kernel, attr, value, device);
}

cudaError_t _cudaLaunchKernelEx(const cudaLaunchConfig_t * config, const void * func, void ** args) {
    return cudaLaunchKernelExC(config, func, args);
}

cudaError_t _cudaLibraryGetKernel(cudaKernel_t * kernel, cudaLibrary_t library, const char * name) {
    return cudaLibraryGetKernel(kernel, library, name);
}

cudaError_t _cudaLibraryLoadData(cudaLibrary_t * library, const void * code,
                                 enum cudaJitOption * jitOptions, void ** jitOptionsValues,
                                 unsigned int numJitOptions,
                                 enum cudaLibraryOption * libraryOptions,
                                 void ** libraryOptionValues, unsigned int numLibraryOptions) {
    return cudaLibraryLoadData(library, code, jitOptions, jitOptionsValues, numJitOptions,
                               libraryOptions, libraryOptionValues, numLibraryOptions);
}

CUresult _cuKernelGetAttribute(int * pi, CUfunction_attribute attrib, CUkernel kernel, CUdevice dev) {
    return cuKernelGetAttribute(pi, attrib, kernel, dev);
}
