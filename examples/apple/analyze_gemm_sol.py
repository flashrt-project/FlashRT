"""Analytical BF16 GEMM roofline for the current Pi0.5 Apple traversal.

The compute reference is an explicit scenario, not an Apple-published peak.
Traffic assumes one cold read of A/B and one write of C, ideal reuse within
each GEMM, no launch cost, no padding, no bias or other epilogue traffic.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compute-tflops', type=float, default=32.0)
    p.add_argument('--bandwidth-gbps', type=float, default=307.0)
    p.add_argument('--prefix', type=int, default=522)
    p.add_argument('--decode', type=int, default=10)
    p.add_argument('--steps', type=int, default=10)
    p.add_argument('--views', type=int, default=2)
    p.add_argument('--output', type=Path, default=Path('.cache/apple-gemm-sol'))
    p.add_argument('--measurements', type=Path, help='Full-pipeline report with --profile-kernels')
    args = p.parse_args()
    if min(args.compute_tflops, args.bandwidth_gbps, args.prefix, args.decode, args.steps, args.views) <= 0:
        p.error('All numeric parameters must be positive')
    if args.prefix < args.views*256:
        p.error('Prefix must contain all image tokens')
    m, enc, dec, steps = args.views*256, args.prefix, args.decode, args.steps
    specs = [
        ('vision', 'patch_conv_equivalent', m, 1152, 588, 1),
        ('vision', 'qkv', m, 3456, 1152, 27),
        ('vision', 'out', m, 1152, 1152, 27),
        ('vision', 'up', m, 4304, 1152, 27),
        ('vision', 'down', m, 1152, 4304, 27),
        ('vision', 'projector', m, 2048, 1152, 1),
        ('prefix', 'qkv', enc, 2560, 2048, 18),
        ('prefix', 'out', enc, 2048, 2048, 17),
        ('prefix', 'gate_and_up_merged', enc, 32768, 2048, 17),
        ('prefix', 'down', enc, 2048, 16384, 17),
        ('decode', 'action_in', dec, 1024, 32, steps),
        ('decode', 'qkv', dec, 2560, 1024, 18*steps),
        ('decode', 'out', dec, 1024, 2048, 18*steps),
        ('decode', 'gate_and_up_merged', dec, 8192, 1024, 18*steps),
        ('decode', 'down', dec, 1024, 4096, 18*steps),
        ('decode', 'action_out', dec, 32, 1024, steps),
        # Precomputed once at model initialization, not each inference.
        ('init_only', 'time_mlp_each', 1, 1024, 1024, 2*steps),
        ('init_only', 'adarm_modulation_each', 1, 3072, 1024, 37*steps),
    ]
    measured = {}
    if args.measurements is not None:
        saved = json.loads(args.measurements.read_text())
        for group in saved['kernel_profile']['groups']:
            if group['kernel'].startswith('gemm'):
                measured[tuple(group['params'])] = group['total_gpu_us'] / group['count']
    rows = []
    for stage, name, M, N, K, count in specs:
        flop = 2*M*N*K
        traffic = 2*(M*K+K*N+M*N)
        intensity = flop/traffic
        compute_us = flop/(args.compute_tflops*1e6)
        memory_us = traffic/(args.bandwidth_gbps*1e3)
        lower_us = max(compute_us, memory_us)
        roof_tflops = min(args.compute_tflops, intensity*args.bandwidth_gbps/1e3)
        row = dict(stage=stage, operation=name, M=M, N=N, K=K, calls=count,
                   gflops_per_call=flop/1e9, ideal_mb_per_call=traffic/1e6,
                   arithmetic_intensity=intensity, compute_us=compute_us,
                   memory_us=memory_us, roofline_us=lower_us,
                   roof_tflops=roof_tflops,
                   compute_util_ceiling_percent=100*roof_tflops/args.compute_tflops,
                   bound='compute' if compute_us >= memory_us else 'memory',
                   measured_diagnostic_us=None, measured_tflops=None,
                   measured_compute_util_percent=None, measured_roofline_util_percent=None)
        metric = measured.get((M, N, K))
        if metric:
            t = metric
            row.update(measured_diagnostic_us=t, measured_tflops=flop/t/1e6,
                       measured_compute_util_percent=100*compute_us/t,
                       measured_roofline_util_percent=100*lower_us/t)
        rows.append(row)
    totals = {}
    for stage in ('vision', 'prefix', 'decode', 'init_only'):
        sr = [r for r in rows if r['stage'] == stage]
        totals[stage] = dict(calls=sum(r['calls'] for r in sr),
            gflops=sum(r['gflops_per_call']*r['calls'] for r in sr),
            ideal_gb=sum(r['ideal_mb_per_call']*r['calls']/1000 for r in sr),
            sum_roofline_ms=sum(r['roofline_us']*r['calls']/1000 for r in sr))
    # Attention matrix products are fused, not separate weight GEMMs. Report
    # their arithmetic without inventing global scores or duplicated GQA IO.
    attention = []
    for stage, q, kv, hd, batch, count in (
        ('vision', 256, 256, 72, args.views*16, 27),
        ('prefix', enc, enc, 256, 8, 17),
        ('decode', dec, enc+dec, 256, 8, 18*steps),
    ):
        for name, M, N, K in (('QK', q, kv, hd), ('PV', q, hd, kv)):
            attention.append(dict(stage=stage, operation=name, batch=batch,
                M=M, N=N, K=K, calls=count,
                gflops_per_batched_call=2*batch*M*N*K/1e9,
                total_gflops=2*batch*M*N*K*count/1e9))
    sensitivity = []
    for peak in (32, 40, 60, 80):
        sensitivity.append(dict(compute_tflops=peak, ridge_flop_per_byte=peak*1000/args.bandwidth_gbps,
            decode_qkv_ceiling_percent=min(100,100*rows[11]['arithmetic_intensity']*args.bandwidth_gbps/1000/peak)))
    report = dict(compute_reference_tflops=args.compute_tflops,
        compute_reference_status='Explicit working assumption, NOT published hardware peak',
        bandwidth_gbps=args.bandwidth_gbps, prefix_tokens=enc, decode_tokens=dec,
        denoising_steps=steps, views=args.views, rows=rows, totals=totals,
        attention_arithmetic=attention, sensitivity=sensitivity,
        caveats=[
            'FMA counts as 2 FLOPs. GB/TFLOP use decimal units.',
            'Roofline assumes cold operand traffic once per GEMM; cache residency may invalidate the DRAM assumption.',
            'Ignoring dispatch, extra tile reads, padding, epilogue and synchronization gives an optimistic bound, not a latency prediction.',
            'Prefix final layer computes QKV only; gate/up are merged GEMMs in the native plan.',
            'Diagnostic timings include fused epilogues and isolated submission overhead; they are not per-layer steady-state timings.',
            'Attention QK/PV FLOPs are separate; fused attention IO is not modeled as independent GEMMs.',
            'Initialization-only modulation GEMMs are excluded from inference totals.',
        ])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'sol.json').write_text(json.dumps(report, indent=2)+'\n')
    with (args.output/'sol.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    lines = [f'# Pi0.5 GEMM SOL — P={args.compute_tflops:g} TFLOPS scenario', '',
        '**Compute capacity is a working assumption, not an official hardware peak.**', '',
        f'Bandwidth: {args.bandwidth_gbps:g} GB/s. Vision M={m}, prefix M={enc}, decode M={dec}; {steps} denoising steps.', '',
        'F=2MNK; D=2(MK+KN+MN) bytes; AI=F/D; t_SOL=max(F/P,D/B).',
        'Compute utilization ceiling=min(1,AI*B/P). Measured SOL=t_SOL/t_measured.', '',
        '| Stage | Operation | M,N,K | Calls | GFLOPs/call | MB/call | FLOP/B | Compute us | Memory us | SOL us | Compute ceiling |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['stage']} | {r['operation']} | {r['M']},{r['N']},{r['K']} | {r['calls']} | {r['gflops_per_call']:.4f} | {r['ideal_mb_per_call']:.3f} | {r['arithmetic_intensity']:.2f} | {r['compute_us']:.2f} | {r['memory_us']:.2f} | {r['roofline_us']:.2f} | {r['compute_util_ceiling_percent']:.1f}% |")
    lines += ['', '## Current full-pipeline kernel diagnostics', '',
        '| GEMM+bias | GPU us | TFLOPS | Compute-reference utilization | Roofline utilization |',
        '|---|---:|---:|---:|---:|']
    for r in rows:
        if r['measured_diagnostic_us'] is not None:
            lines.append(f"| {r['operation']} | {r['measured_diagnostic_us']:.2f} | {r['measured_tflops']:.2f} | {r['measured_compute_util_percent']:.1f}% | {r['measured_roofline_util_percent']:.1f}% |")
    lines += ['', '## Traversal totals (linear GEMMs and patch-convolution equivalent)', '',
              '| Stage | GFLOPs | Ideal traffic GB | Sum of per-GEMM lower bounds ms |',
              '|---|---:|---:|---:|']
    for stage, r in totals.items():
        lines.append(f"| {stage} | {r['gflops']:.3f} | {r['ideal_gb']:.3f} | {r['sum_roofline_ms']:.3f} |")
    lines += ['', '## Fused attention arithmetic (not independent GEMM traffic)', '',
              '| Stage | Product | Batch × M,N,K | Calls | GFLOPs/call | Total GFLOPs |',
              '|---|---|---|---:|---:|---:|']
    for r in attention:
        lines.append(f"| {r['stage']} | {r['operation']} | {r['batch']} × {r['M']},{r['N']},{r['K']} | {r['calls']} | {r['gflops_per_batched_call']:.4f} | {r['total_gflops']:.3f} |")
    lines += ['', '## Compute-reference sensitivity', '',
              '| Assumed TFLOPS | Ridge FLOP/B | Decode QKV compute ceiling |', '|---:|---:|---:|']
    for r in sensitivity:
        lines.append(f"| {r['compute_tflops']} | {r['ridge_flop_per_byte']:.1f} | {r['decode_qkv_ceiling_percent']:.2f}% |")
    lines += ['', '## Limits', '']+[f'- {x}' for x in report['caveats']]
    lines += ['', '## Sources', '',
        '- Shapes and invocation counts: `flash_rt/models/pi05/torch_pipeline.py` and `flash_rt/hardware/torch_reference.py`.',
        '- [Apple M5 Pro bandwidth specification](https://www.apple.com/macbook-pro/specs/).',
        f'- Native diagnostics: `{args.measurements}`.' if args.measurements else '- No measured diagnostics supplied.',
        '']
    (args.output/'sol.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
