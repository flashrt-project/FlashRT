#include "kernels/pi05/thor/pi05_pdl.cuh"
namespace flash_rt {
namespace fp4 {
bool& pdl_flag() { static bool flag = false; return flag; }
}  // namespace fp4
}  // namespace flash_rt
