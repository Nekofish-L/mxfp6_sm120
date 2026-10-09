"""Isolated three-arm serving experiment: off, PDL M<=32, expanded PDL.

Each arm starts an otherwise identical server. Two opposite-order lifecycles
measure steady-state streaming requests with fixed seeds and token counts.
The supplied client is vllm-mach's streaming benchmark.py; run_phase is used
directly, so this is a custom workload, not its public six-point protocol.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--vllm-source', type=Path, required=True)
    parser.add_argument('--client-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', default='7')
    parser.add_argument('--port', type=int, default=18881)
    parser.add_argument('--max-rows', type=int, default=128)
    parser.add_argument('--points', nargs='+', type=int, default=[4, 16, 32, 64, 128])
    parser.add_argument('--input-tokens', type=int, default=1024)
    parser.add_argument('--output-tokens', type=int, default=512)
    parser.add_argument('--arms', nargs='+', default=['off', '32', 'expanded', 'expanded', '32', 'off'])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('pdl_streaming_client', args.client_source)
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {'workload': {'input_tokens': args.input_tokens, 'output_tokens': args.output_tokens,
                           'points': args.points, 'requests_per_point': '2 * concurrency',
                           'warmup_per_point': 'concurrency requests, 128 output tokens',
                           'sampling': 'greedy, ignore_eos, fixed per-request seeds',
                           'arms': args.arms, 'expanded_max_rows': args.max_rows}, 'runs': []}
    for index, arm in enumerate(args.arms):
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(('127.0.0.1', args.port))
        directory = args.output / f'{index:02d}-{arm}'
        directory.mkdir()
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(('VLLM_', 'MXFP6_', 'MXFP8_', 'MX8_'))}
        limit = args.max_rows if arm == 'expanded' else 32
        settings = dict(CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                        PYTHONPATH=f'{args.vllm_source}:{root / "python"}',
                        MXFP6_LIBRARY_PATH=str(root / 'build/mxfp8/mxfp6_torch.so'),
                        MXFP8_LIBRARY_PATH=str(root / 'build/mxfp8/mxfp8_torch.so'),
                        VLLM_MACH_MXFP8_BACKEND='native',
                        VLLM_MACH_MXFP8_PDL='0' if arm == 'off' else '1',
                        VLLM_MACH_MXFP8_PDL_MAX_ROWS=str(limit),
                        VLLM_MACH_MXFP8_NORM_QUANT='0', VLLM_MACH_GDN_TP1='0',
                        VLLM_MACH_MXFP8_FUSED_MLP='1', VLLM_MACH_FUSED_GEMMA_NORM='1',
                        VLLM_CACHE_ROOT=str(args.output / 'cache/vllm'),
                        TORCHINDUCTOR_CACHE_DIR=str(args.output / 'cache/inductor'))
        env.update(settings)
        command = [str(args.python), '-m', 'vllm_mach.mxfp6.serve', '--model', str(args.model),
                   '--tensor-parallel-size', '1', '--nvfp4-lm-head',
                   '--served-model-name', 'pdl-validation', '--host', '127.0.0.1',
                   '--port', str(args.port), '--kv-cache-dtype', 'fp8_e4m3',
                   '--max-num-seqs', str(max(args.points)), '--max-model-len', '4096',
                   '--max-num-batched-tokens', '2048', '--seed', '20261008',
                   '--gpu-memory-utilization', '0.80']
        (directory / 'command.json').write_text(json.dumps(
            {'command': command, 'environment': settings}, indent=2) + '\n')
        run = {'index': index, 'arm': arm, 'points': []}
        with (directory / 'server.log').open('w') as log:
            server = subprocess.Popen(command, env=env, cwd=args.vllm_source.parent,
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            (directory / 'server.pid').write_text(str(server.pid) + '\n')
            try:
                deadline = time.monotonic() + 1200
                while time.monotonic() < deadline:
                    if server.poll() is not None:
                        raise RuntimeError(f'{arm} server exited {server.returncode}: {directory}/server.log')
                    try:
                        with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/health', timeout=1):
                            break
                    except (OSError, TimeoutError):
                        time.sleep(1)
                else:
                    raise TimeoutError(f'{arm} startup timeout')
                print(f'run {index} {arm} ready', flush=True)
                points = args.points if index < len(args.arms) // 2 else args.points[::-1]
                for concurrency in points:
                    phase = argparse.Namespace(
                        base_url=f'http://127.0.0.1:{args.port}', model='pdl-validation',
                        max_concurrency=concurrency, num_prompts=2 * concurrency,
                        input_tokens=args.input_tokens, output_tokens=args.output_tokens,
                        contract_seed=20261008, request_seed_base=2026100800,
                        token_id_low=1000, token_id_high=240000, prompt_manifest=None,
                        request_rate=0, top_k=-1, top_p=1.0, warmup_requests=concurrency,
                        warmup_output_tokens=128, timeout_s=600)
                    result = asyncio.run(client.run_phase(phase))
                    aggregate = result['aggregate']
                    if (aggregate['completed'] != phase.num_prompts
                            or aggregate['completion_tokens'] != phase.num_prompts * phase.output_tokens):
                        raise AssertionError(f'incomplete serving run: {aggregate}')
                    (directory / f'c{concurrency}.json').write_text(json.dumps(result, indent=2) + '\n')
                    point = {'concurrency': concurrency, 'aggregate': aggregate,
                             'output_hashes': [hashlib.sha256(item['response_text'].encode()).hexdigest()
                                               for item in result['requests']]}
                    run['points'].append(point)
                    print(index, arm, concurrency, aggregate['output_throughput_tokens_per_s'],
                          'tok/s', flush=True)
                    (directory / 'summary.json').write_text(json.dumps(run, indent=2) + '\n')
            finally:
                if server.poll() is None:
                    os.killpg(server.pid, signal.SIGTERM)
                    try:
                        server.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(server.pid, signal.SIGKILL)
                        server.wait()
        summary['runs'].append(run)
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
