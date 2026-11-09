#!/usr/bin/env python3
"""Single-GPU pinned-memory PCIe H2D/D2H bandwidth test. No compute benchmark."""
import argparse
import ctypes as C
import datetime
import statistics
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description='Test PCIe H2D/D2H bandwidth on one GPU (no compute)')
    p.add_argument('--device', default='0f:00.0', help='GPU PCI BDF, e.g. 0f:00.0')
    p.add_argument('--output', required=True, help='Log file, e.g. gen2.log')
    p.add_argument('--mib', type=int, default=32, help='Transfer size in MiB (default: 32)')
    p.add_argument('--rounds', type=int, default=3, help='Measurements per direction (default: 3)')
    a = p.parse_args()
    if a.mib <= 0 or a.rounds <= 0:
        p.error('--mib and --rounds must be positive')
    bdf = a.device.lower().strip()
    if bdf.count(':') == 1:
        bdf = '0000:' + bdf
    sysdev = Path('/sys/bus/pci/devices') / bdf
    if not sysdev.exists():
        p.error(f'PCI device not found: {bdf}')

    speed = (sysdev / 'current_link_speed').read_text().strip()
    width = (sysdev / 'current_link_width').read_text().strip()
    size = a.mib * 1024 * 1024

    cuda = C.CDLL('libcuda.so.1')
    def api(name, argtypes):
        fn = getattr(cuda, name)
        fn.argtypes = argtypes
        fn.restype = C.c_int
        return fn
    I, P, S = C.c_int, C.c_void_p, C.c_size_t
    init = api('cuInit', [C.c_uint])
    get_device = api('cuDeviceGetByPCIBusId', [C.POINTER(I), C.c_char_p])
    ctx_create = api('cuCtxCreate_v2', [C.POINTER(P), C.c_uint, I])
    ctx_destroy = api('cuCtxDestroy_v2', [P])
    host_alloc = api('cuMemHostAlloc', [C.POINTER(P), S, C.c_uint])
    host_free = api('cuMemFreeHost', [P])
    dev_alloc = api('cuMemAlloc_v2', [C.POINTER(C.c_uint64), S])
    dev_free = api('cuMemFree_v2', [C.c_uint64])
    h2d = api('cuMemcpyHtoD_v2', [C.c_uint64, P, S])
    d2h = api('cuMemcpyDtoH_v2', [P, C.c_uint64, S])
    sync = api('cuCtxSynchronize', [])

    def check(fn, *args):
        err = fn(*args)
        if err:
            raise RuntimeError(f'{fn.__name__} returned CUDA error {err}')

    lines = [
        f'Timestamp: {datetime.datetime.now().astimezone().isoformat()}',
        f'GPU PCI BDF: {bdf}',
        f'Current PCIe link: {speed}, x{width}',
        f'Transfer: {a.mib} MiB, pinned host memory',
        f'Rounds per direction: {a.rounds}',
    ]
    check(init, 0)
    device, context, host, gpu = I(), P(), P(), C.c_uint64()
    check(get_device, C.byref(device), bdf.encode('ascii'))
    check(ctx_create, C.byref(context), 0, device)
    try:
        check(host_alloc, C.byref(host), size, 0)
        try:
            C.memset(host, 0x5a, size)
            check(dev_alloc, C.byref(gpu), size)
            try:
                check(h2d, gpu, host, size)
                check(sync)
                for label, fn, args in (
                    ('H2D (RAM -> GPU)', h2d, (gpu, host, size)),
                    ('D2H (GPU -> RAM)', d2h, (host, gpu, size)),
                ):
                    rates = []
                    for n in range(1, a.rounds + 1):
                        start = time.perf_counter()
                        check(fn, *args)
                        check(sync)
                        elapsed = time.perf_counter() - start
                        rate = size / elapsed / 1e9
                        rates.append(rate)
                        lines.append(f'{label} round {n}: {rate:.6f} GB/s ({elapsed*1000:.3f} ms)')
                    lines.append(f'{label} median: {statistics.median(rates):.6f} GB/s')
            finally:
                check(dev_free, gpu)
        finally:
            check(host_free, host)
    finally:
        check(ctx_destroy, context)
    result = '\n'.join(lines) + '\n'
    Path(a.output).write_text(result, encoding='utf-8')
    print(result, end='')
    print(f'Saved: {a.output}')


if __name__ == '__main__':
    try:
        main()
    except (OSError, RuntimeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
