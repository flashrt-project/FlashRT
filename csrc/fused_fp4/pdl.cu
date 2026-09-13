#include "fused_fp4/pdl.cuh"
namespace flash_rt {
namespace fp4 {
bool& pdl_flag() { static bool flag = false; return flag; }
}  // namespace fp4
}  // namespace flash_rt
