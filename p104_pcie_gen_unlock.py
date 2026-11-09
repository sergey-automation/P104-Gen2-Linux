#!/usr/bin/env python3
"""P104-100 PCIe Gen1/Gen2 one-shot switch, standalone script.
Uses the tested BAR0 write sequence and PCIe link retraining procedure.
No background watchdog or recurring register writes.
"""
import argparse
import mmap
import os
import re
import time
from pathlib import Path

ROOT = Path('/sys/bus/pci/devices')
GPU_ID = 0x1B8710DE  # little-endian: NVIDIA 10de:1b87
LIVE = [
    (0x8872C, 0x0000000F, 0x00000006),
    (0x2241C, 0x00000081, 0x00000081),
    (0x8841C, 0x00006000, 0x00000000),
    (0x8C2C0, 0x00000004, 0x00000000),
    (0x8C300, 0x00000010, 0x00000000),
    (0x8C040, 0x000C0000, 0x00040000),
    (0x8C1C0, 0x000F0000, 0x00020000),
    (0x88610, 0x00001001, 0x00000001),
]
PAT = re.compile(r'^(?:[0-9a-f]{4}:)?[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$')


def read(fd, off, n):
    b = os.pread(fd, n, off)
    if len(b) != n or b == b'\xff' * n:
        raise RuntimeError(f'PCI config read failed at 0x{off:x}')
    return int.from_bytes(b, 'little')


def write(fd, off, n, val):
    if os.pwrite(fd, int(val).to_bytes(n, 'little'), off) != n:
        raise RuntimeError(f'PCI config write failed at 0x{off:x}')


def capability(fd):
    if not read(fd, 6, 2) & 0x10:
        raise RuntimeError('PCI capability list absent')
    p = read(fd, 0x34, 1) & 0xfc
    seen = set()
    while p and p not in seen and 0x40 <= p <= 0xfc:
        seen.add(p)
        if read(fd, p, 1) == 0x10:
            return p
        p = read(fd, p + 1, 1) & 0xfc
    raise RuntimeError('PCIe capability not found')


def link(fd, cap):
    capability_raw = read(fd, cap + 0x0c, 4)
    status_raw = read(fd, cap + 0x12, 2)
    return {
        'max_gen': capability_raw & 15,
        'max_width': (capability_raw >> 4) & 63,
        'gen': status_raw & 15,
        'width': (status_raw >> 4) & 63,
        'training': bool(status_raw & 0x800),
    }


def show(gpu, upstream):
    print('                  GPU             PCIe Port')
    print(f"  Max         Gen{gpu['max_gen']} x{gpu['max_width']:<2}       Gen{upstream['max_gen']} x{upstream['max_width']}")
    print(f"  Current     Gen{gpu['gen']} x{gpu['width']:<2}       Gen{upstream['gen']} x{upstream['width']}")


def apply_bar(gp, gf):
    """BAR0 masks and write order from the tested Linux worker."""
    fd = os.open(gp / 'resource0', os.O_RDWR)
    pages = {}
    try:
        for off, _, _ in LIVE:
            base = (off // mmap.PAGESIZE) * mmap.PAGESIZE
            if base not in pages:
                pages[base] = mmap.mmap(fd, mmap.PAGESIZE, flags=mmap.MAP_SHARED,
                                        prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=base)
        plan = []
        for off, mask, val in LIVE:
            pg = pages[(off // mmap.PAGESIZE) * mmap.PAGESIZE]
            idx = off % mmap.PAGESIZE
            old = int.from_bytes(pg[idx:idx + 4], 'little')
            if old == 0xffffffff:
                raise RuntimeError(f'BAR0 0x{off:x} unavailable')
            new = (old & ~mask) | (val & mask)
            if off == 0x8c040:
                legal = lambda v: bool(v & 0x80000000) != bool(v & 0x20000000)
                if not legal(old) or not legal(new):
                    raise RuntimeError(f'BAR0 0x8c040 unsafe state: {old:08x}->{new:08x}')
            plan.append((off, mask, old, new))
        for off, mask, old, new in plan:
            if old == new:
                continue
            if read(gf, 0, 4) != GPU_ID:
                raise RuntimeError('GPU disappeared during BAR0 update')
            pg = pages[(off // mmap.PAGESIZE) * mmap.PAGESIZE]
            idx = off % mmap.PAGESIZE
            actual = int.from_bytes(pg[idx:idx + 4], 'little')
            if actual != old:
                raise RuntimeError(f'BAR0 0x{off:x} changed before write')
            pg[idx:idx + 4] = new.to_bytes(4, 'little')
            time.sleep(0.020)  # one-time delay after register write, not a background loop
            if read(gf, 0, 4) != GPU_ID:
                raise RuntimeError('GPU disappeared after BAR0 write')
            back = int.from_bytes(pg[idx:idx + 4], 'little')
            if (back & mask) != (new & mask):
                raise RuntimeError(f'BAR0 verify failed at 0x{off:x}')
    finally:
        for pg in pages.values():
            pg.close()
        os.close(fd)


def retrain(gf, uf, gc, uc, gen):
    # Set Target Link Speed on both ends; retrain from the upstream port.
    for fd, c in ((gf, gc), (uf, uc)):
        old = read(fd, c + 0x30, 2)
        new = (old & ~0xf) | gen
        if new != old:
            write(fd, c + 0x30, 2, new)
        if read(fd, c + 0x30, 2) & 0xf != gen:
            raise RuntimeError('Target Link Speed write failed')
    for fd, c in ((gf, gc), (uf, uc)):
        if link(fd, c)['training']:
            raise RuntimeError('link already training')
    old = read(uf, uc + 0x10, 2)
    write(uf, uc + 0x10, 2, old | 0x20)
    # Wait up to 250 ms for link retraining, as in the tested worker.
    end = time.monotonic() + 0.250
    stable, last = 0, None
    while time.monotonic() <= end:
        g, u = link(gf, gc), link(uf, uc)
        if not g['training'] and not u['training'] and g['gen'] in (1, 2) and u['gen'] in (1, 2):
            state = (g['gen'], g['width'], u['gen'], u['width'])
            stable = stable + 1 if state == last else 1
            last = state
            if stable >= 3:
                return
        else:
            stable, last = 0, None
        time.sleep(0.005)


def run_one(bdf, gen):
    gp = ROOT / bdf
    up = gp.resolve().parent
    if not up.name or not PAT.fullmatch(up.name):
        raise RuntimeError('upstream PCI bridge not found')
    gf = os.open(gp / 'config', os.O_RDWR)
    try:
        uf = os.open(up / 'config', os.O_RDWR)
        try:
            if read(gf, 0, 4) != GPU_ID:
                raise RuntimeError('not NVIDIA P104-100 (10de:1b87)')
            gc, uc = capability(gf), capability(uf)
            print(f'GPU {bdf} | PCIe Port {up.name}', flush=True)
            print('Before:')
            show(link(gf, gc), link(uf, uc))
            print(f'\nRequested: Gen{gen}', flush=True)
            if gen == 2:
                apply_bar(gp, gf)
            retrain(gf, uf, gc, uc, gen)
            print('\nAfter:')
            g, u = link(gf, gc), link(uf, uc)
            show(g, u)
            print('Result: ' + ('Gen' + str(gen) + ' set' if
                                   g['gen'] == gen and u['gen'] == gen and not g['training'] and not u['training']
                                   else 'requested Gen not confirmed'), flush=True)
        finally:
            os.close(uf)
    finally:
        os.close(gf)


def main():
    p = argparse.ArgumentParser(description='P104-100: switch PCIe Gen1/Gen2 once')
    p.add_argument('--gen', type=int, choices=[1, 2], required=True)
    p.add_argument('--devices', required=True, help='Comma-separated PCI BDFs: 0f:00.0,10:00.0')
    a = p.parse_args()
    if os.geteuid() != 0:
        p.error('run as root (sudo)')
    bs = []
    for part in a.devices.split(','):
        bdf = part.strip().lower()
        if not PAT.fullmatch(bdf):
            p.error(f'invalid PCI BDF: {part}')
        if bdf.count(':') == 1:
            bdf = '0000:' + bdf
        bs.append(bdf)
    failed = False
    for bdf in bs:
        try:
            run_one(bdf, a.gen)
        except (OSError, RuntimeError, ValueError) as e:
            failed = True
            print(f'GPU {bdf}: ERROR: {e}', flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
