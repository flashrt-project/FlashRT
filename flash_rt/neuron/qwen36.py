"""Qwen3.6-35B-A3B text inference on Trainium 3 with NxDI/NKI."""

import hashlib
import json
import os
from pathlib import Path
import subprocess


def file_sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(16*1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


class Qwen36NeuronFrontend:
    """Compile/load a BF16, batch-one text model; vision is not part of this API.

    Select visible cores before importing Neuron. TP=4 with LNC=2 occupies
    eight physical NeuronCores (two Trainium 3 chips). Compilation artifacts
    are bound to the checkpoint and compile settings, not just a directory name.
    """

    def __init__(self, checkpoint, compiled_path, *, tp_degree=4, seq_len=128,
                 speculative=False, megakernel=False, layer_kernels=False,
                 faithful_numerics=True):
        if os.environ.get('NEURON_PLATFORM_TARGET_OVERRIDE', 'trn3') != 'trn3':
            raise ValueError('This backend requires the trn3 compilation target')
        os.environ['NEURON_PLATFORM_TARGET_OVERRIDE'] = 'trn3'
        os.environ['PJRT_DEVICE'] = 'NEURON'
        inventory = json.loads(subprocess.check_output(['/opt/aws/neuron/bin/neuron-ls', '-j']))
        if not inventory or any(not d['instance_type'].startswith('trn3') for d in inventory):
            raise RuntimeError('Trainium 3 hardware is required')
        if tp_degree < 1 or seq_len < 128 or seq_len % 128:
            raise ValueError('Positive TP and a sequence window divisible by 128 are required')
        if megakernel and not speculative:
            raise ValueError('The fused round megakernel requires speculative=True')
        if any(os.environ.get(name) == '1' for name in ('A3B_BYPASS_ATTN', 'A3B_BYPASS_MOE')):
            raise ValueError('Model bypass modes are not supported by FlashRT')
        if faithful_numerics and (speculative or megakernel or layer_kernels):
            raise ValueError('Faithful numerics currently supports ordinary greedy decoding')
        from flash_rt.neuron._vendor.qwen36 import inference_qwen36_a3b as driver
        from transformers import AutoTokenizer
        self.driver = driver
        checkpoint = Path(checkpoint).resolve()
        compiled_path = Path(compiled_path).resolve()
        raw_config = (checkpoint/'config.json').read_bytes()
        model_config = json.loads(raw_config)
        if model_config.get('text_config', model_config).get('model_type') not in ('qwen3_5_moe', 'qwen3_5_moe_text'):
            raise ValueError('Expected Qwen3.6-35B-A3B qwen3_5_moe text architecture')
        source = Path(driver.__file__).parent
        implementation = hashlib.sha256()
        for file in sorted(source.rglob('*.py')):
            implementation.update(str(file.relative_to(source)).encode())
            implementation.update(file.read_bytes())
        identity = {
            'checkpoint': str(checkpoint), 'config_sha256': hashlib.sha256(raw_config).hexdigest(),
            'weights': {p.name: file_sha256(p) for p in sorted(checkpoint.glob('*.safetensors'))},
            'implementation_sha256': implementation.hexdigest(), 'target': 'trn3',
            'tp_degree': tp_degree, 'logical_nc_config': 2, 'seq_len': seq_len,
            'speculative': speculative, 'megakernel': megakernel, 'layer_kernels': layer_kernels,
            'faithful_numerics': faithful_numerics,
            'compiler_environment': {name: os.environ.get(name) for name in (
                'USE_NKILIB_KERNEL', 'USE_NEWTON_RMSNORM', 'USE_PYTHON_RMSNORM',
                'USE_NKI_FUSED', 'USE_NKI', 'DELTANET_SEQUENTIAL', 'USE_PYTORCH_CHUNK',
                'USE_NKI_CHUNKED', 'NEURON_CC_FLAGS', 'A3B_SKIP_SHARD')},
        }
        if not identity['weights']:
            raise ValueError('No checkpoint weights found')
        from importlib.metadata import version
        identity['runtime'] = {name: version(name) for name in ('torch-neuronx', 'neuronx-distributed-inference', 'neuronx-cc', 'nki', 'numpy', 'islpy')}
        compiled_path.mkdir(parents=True, exist_ok=True)
        identity_path = compiled_path/'flashrt-identity.json'
        import fcntl
        with (compiled_path/'flashrt-compile.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if identity_path.exists():
                if json.loads(identity_path.read_text()) != identity:
                    raise ValueError('Compiled model identity differs; use a new artifact directory')
            elif (compiled_path/'model.pt').exists():
                raise ValueError('Existing executable has no FlashRT provenance manifest')
            else:
                identity_path.write_text(json.dumps(identity, indent=2)+'\n')
            config = driver.build_inference_config(str(checkpoint), tp_degree, seq_len,
                                                   spec_decode=speculative,
                                                   tkg_attention_kernel=layer_kernels,
                                                   moe_layer_kernel=layer_kernels,
                                                   speculation_megakernel=megakernel,
                                                   faithful_numerics=faithful_numerics)
            driver.maybe_compile(str(checkpoint), str(compiled_path), config)
        self.model = driver.load_model(str(compiled_path))
        self.tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), padding_side='right')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.identity = identity

    def generate(self, prompt, max_new_tokens=32):
        if max_new_tokens < 1:
            raise ValueError('max_new_tokens must be positive')
        if len(self.tokenizer.encode(prompt)) + max_new_tokens > self.model.neuron_config.seq_len:
            raise ValueError('Prompt plus requested generation exceeds the compiled window')
        tokens, text = self.driver.generate(self.model, self.tokenizer, prompt, max_new_tokens)
        return {'token_ids': tokens, 'text': text}

    def benchmark(self, prompt, max_new_tokens=32, num_runs=10):
        if num_runs < 2 or max_new_tokens < 2:
            raise ValueError('Benchmark requires at least two runs and two generated tokens')
        from flash_rt.neuron.benchmark import benchmark
        return benchmark(self, prompt, max_new_tokens, num_runs)
