"""Synchronous, batch-one Qwen timings with actual generated-token counts."""

import time

import numpy as np


def distribution(samples):
    values = np.asarray(samples, dtype=np.float64)
    return {
        'samples': values.tolist(), 'count': len(values),
        **{f'p{p}': float(np.percentile(values, p)) for p in (50, 90, 99)},
        'mean': float(values.mean()),
    }


def benchmark(frontend, prompt, max_new_tokens, num_runs):
    from transformers import GenerationConfig

    driver, model = frontend.driver, frontend.model
    if model.neuron_config.async_mode:
        raise ValueError('Synchronous module timing requires async_mode=False')
    inputs = frontend.tokenizer(prompt, return_tensors='pt')
    prompt_len = inputs.input_ids.shape[1]
    if prompt_len + max_new_tokens > model.neuron_config.seq_len:
        raise ValueError('Prompt plus requested generation exceeds the compiled window')
    adapter = driver.HuggingFaceGenerationAdapter(model)
    adapter.generation_config.transformers_version = driver.transformers.__version__
    config = GenerationConfig(do_sample=False, pad_token_id=frontend.tokenizer.pad_token_id,
                              eos_token_id=[])
    config.transformers_version = driver.transformers.__version__
    kwargs = dict(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                  generation_config=config, max_new_tokens=max_new_tokens,
                  do_sample=False, sampling_params=driver.prepare_sampling_params(
                      batch_size=1, top_k=[1], top_p=[1.0], temperature=[1.0]))
    speculative = model.neuron_config.enable_fused_speculation
    if speculative:
        kwargs['prompt_lookup_num_tokens'] = model.neuron_config.speculation_length

    def run():
        # HuggingFaceGenerationAdapter.generate resets recurrent and KV state.
        ids = adapter.generate(**kwargs)[0].tolist()
        if len(ids) - prompt_len != max_new_tokens:
            raise RuntimeError('Benchmark did not generate the requested token count')
        return len(ids) - prompt_len

    for _ in range(3):
        run()
    modules = {'prefill': model.context_encoding_model}
    modules['decode'] = model.fused_spec_model if speculative else model.token_generation_model
    samples = {key: [] for key in modules}
    starts, handles, first_tokens = {}, [], []
    run_start = 0

    def pre(key):
        def hook(*unused):
            starts[key] = time.perf_counter_ns()
        return hook

    def post(key):
        def hook(*unused):
            now = time.perf_counter_ns()
            samples[key].append((now - starts[key])/1e6)
            if key == 'prefill':
                first_tokens.append((now - run_start)/1e6)
        return hook

    elapsed, generated = [], []
    try:
        for key, module in modules.items():
            handles.extend([module.register_forward_pre_hook(pre(key)),
                            module.register_forward_hook(post(key))])
        for _ in range(num_runs):
            run_start = time.perf_counter_ns()
            generated.append(run())
            elapsed.append((time.perf_counter_ns() - run_start)/1e6)
    finally:
        for handle in handles:
            handle.remove()
    if len(samples['prefill']) != num_runs:
        raise RuntimeError('Expected exactly one prefill call per generation')
    if not speculative and len(samples['decode']) != num_runs*(max_new_tokens-1):
        raise RuntimeError('Decode call count differs from generated-token count')
    result = {
        'boundary': 'Pretokenized input through synchronous generation, including reset and host loop',
        'module_boundary': 'Host-observed synchronous compiled-module calls, including runtime transfers',
        'batch_size': 1, 'prompt_tokens': prompt_len, 'compiled_window': model.neuron_config.seq_len,
        'generated_tokens': generated, 'warmups': 3, 'runs': num_runs,
        'generation_latency_ms': distribution(elapsed),
        'first_token_module_completion_ms': distribution(first_tokens),
        'prefill_call_latency_ms': distribution(samples['prefill']),
        'decode_call_latency_ms': distribution(samples['decode']),
        'generated_tokens_per_second': sum(generated)/(sum(elapsed)/1000),
    }
    if not speculative:
        result['decode_tokens_per_second'] = sum(n-1 for n in generated)/(sum(samples['decode'])/1000)
    return result
