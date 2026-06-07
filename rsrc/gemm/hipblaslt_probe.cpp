#include "gemm/hipblaslt_probe.h"

#include <hipblaslt/hipblaslt.h>

namespace {

const char* hipblas_status_name(hipblasStatus_t status) {
  switch (status) {
    case HIPBLAS_STATUS_SUCCESS:
      return "HIPBLAS_STATUS_SUCCESS";
    case HIPBLAS_STATUS_NOT_INITIALIZED:
      return "HIPBLAS_STATUS_NOT_INITIALIZED";
    case HIPBLAS_STATUS_ALLOC_FAILED:
      return "HIPBLAS_STATUS_ALLOC_FAILED";
    case HIPBLAS_STATUS_INVALID_VALUE:
      return "HIPBLAS_STATUS_INVALID_VALUE";
    case HIPBLAS_STATUS_MAPPING_ERROR:
      return "HIPBLAS_STATUS_MAPPING_ERROR";
    case HIPBLAS_STATUS_EXECUTION_FAILED:
      return "HIPBLAS_STATUS_EXECUTION_FAILED";
    case HIPBLAS_STATUS_INTERNAL_ERROR:
      return "HIPBLAS_STATUS_INTERNAL_ERROR";
    case HIPBLAS_STATUS_NOT_SUPPORTED:
      return "HIPBLAS_STATUS_NOT_SUPPORTED";
    case HIPBLAS_STATUS_ARCH_MISMATCH:
      return "HIPBLAS_STATUS_ARCH_MISMATCH";
    case HIPBLAS_STATUS_HANDLE_IS_NULLPTR:
      return "HIPBLAS_STATUS_HANDLE_IS_NULLPTR";
    case HIPBLAS_STATUS_INVALID_ENUM:
      return "HIPBLAS_STATUS_INVALID_ENUM";
    case HIPBLAS_STATUS_UNKNOWN:
      return "HIPBLAS_STATUS_UNKNOWN";
    default:
      return "HIPBLAS_STATUS_UNRECOGNIZED";
  }
}

}  // namespace

HipblasLtProbeResult probe_hipblaslt() {
  HipblasLtProbeResult result;

  hipblasLtHandle_t handle = nullptr;
  hipblasStatus_t status = hipblasLtCreate(&handle);
  result.status_code = static_cast<int>(status);
  result.status_name = hipblas_status_name(status);
  result.available = (status == HIPBLAS_STATUS_SUCCESS);

  if (!result.available) {
    return result;
  }

  int version = 0;
  status = hipblasLtGetVersion(handle, &version);
  if (status == HIPBLAS_STATUS_SUCCESS) {
    result.version = version;
  }

  hipblasLtDestroy(handle);
  return result;
}
