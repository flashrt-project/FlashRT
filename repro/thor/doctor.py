import json,sys
import torch
import flash_rt
from flash_rt import flash_rt_kernels,flash_rt_fp4
from flash_rt.hardware.thor import fa4_backend

assert torch.cuda.is_available(), 'CUDA torch required'
assert torch.cuda.get_device_capability()==(11,0), 'Run on Thor SM110'
available=fa4_backend.is_available()
report={'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'capability':torch.cuda.get_device_capability(),'source':flash_rt.__file__,'native_kernels':flash_rt_kernels.__file__,'fp4_kernels':flash_rt_fp4.__file__,'fa4_available':available,'fa4_status':fa4_backend.status()}
print(json.dumps(report,indent=2))
assert available, 'FA4 unavailable; do not silently publish fallback numbers'
