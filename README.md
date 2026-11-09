# NVIDIA P104-100 — PCIe Gen2 Unlock for Linux

**Linux adaptation** of the algorithm from the Windows project [skbskhud/p104-unlock-v3](https://github.com/skbskhud/p104-unlock-v3). Thanks to the original author for the method.

Unlike the Windows version, which repeatedly accesses GPU registers approximately every **20 ms** to maintain Gen2, this version switches the link **once after Linux boots**. A modification to the proprietary NVIDIA driver prevents the observed Gen2 → Gen1 fallback during CUDA initialization. No continuous watchdog is needed. The link must be switched again after a reboot.

**Tested configuration (2026-11-09):** Ubuntu 24.04, kernel `6.17.0-29-generic`, NVIDIA proprietary DKMS `535.309.01`, ASUS B250 Mining Expert, NVIDIA P104-100 8 GB (`10de:1b87`). **PCIe x1 risers** were used: the x1 link width in the results comes from the connection, not from a claim about the GPU's maximum link width. On the tested card, `LnkCap` reported Gen2 x4 for the GPU and Gen3 x1 for the upstream port.

## Result: PCIe Bandwidth

One P104-100, PCI address `0000:0f:00.0`; pinned host memory, 32 MiB, three measurements per direction. The table shows medians in GB/s (10⁹ bytes/s).

| Transfer | Gen1 x1 (2.5 GT/s) | Gen2 x1 (5.0 GT/s) | Speedup |
|---|---:|---:|---:|
| RAM → GPU (H2D) | 0.199652 | 0.396509 | 1.986× |
| GPU → RAM (D2H) | 0.208926 | 0.418162 | 2.002× |

Raw results: [`gen1.log`](benchmarks/single_gpu/gen1.log), [`gen2.log`](benchmarks/single_gpu/gen2.log). To reproduce the test: [`p104_pcie_transfer_bench.py`](benchmarks/single_gpu/p104_pcie_transfer_bench.py).

We also compared **Qwen3.6-35B-A3B Q8_0** on six P104-100 GPUs. At context lengths of 7,141 / 22,400 / 37,659 tokens, Decode increased by **18.19% / 17.18% / 15.53%**, respectively, and Prefill by **16.63% / 13.06% / 17.91%**. Detailed methodology, 16 test points, and raw logs: [`benchmarks/qwen/README.md`](benchmarks/qwen/README.md).

## 1. NVIDIA Driver Modification

During CUDA initialization, the tested driver reverted Gen2 to Gen1. In our copy of the proprietary `nv-kernel.o_binary` from version `535.309.01`, one instruction was modified:

```text
file: /usr/src/nvidia-535.309.01/nvidia/nv-kernel.o_binary
file offset: 0xCE8E5
before: 0D 00 00 08 00   (or eax, 0x80000)
after:  0D 00 00 04 00   (or eax, 0x40000)
```

**Applicable only to the tested NVIDIA 535.309.01 binary.** For other builds, the instruction location and contents must be investigated separately. Modifying a proprietary driver binary is experimental; make a backup, stop CUDA processes, and ensure you have a way to restore the system. If Secure Boot is enabled, the module may need to be signed.

The following is the same procedure used in the experiment, **without a separate `patch_nv_kernel.py` script**. Run it from a writable working directory:

```bash
python3 - <<'PY'
from pathlib import Path
import shutil

src = Path('/usr/src/nvidia-535.309.01/nvidia/nv-kernel.o_binary')
backup = Path('nv-kernel-535-original.o')
patched = Path('nv-kernel-535-gen2.o')
old = bytes.fromhex('0d 00 00 08 00 c7 45 0c 01 00 00 00')
new = bytes.fromhex('0d 00 00 04 00 c7 45 0c 01 00 00 00')
data = src.read_bytes()
assert data.count(old) == 1, f'Expected 1 match, found: {data.count(old)}'
assert data.find(old) == 0xCE8E5, f'Unexpected offset: 0x{data.find(old):x}'
shutil.copy2(src, backup)
patched.write_bytes(data.replace(old, new, 1))
print('Patched at:', hex(data.index(old)))
PY
```

Install the rebuilt DKMS driver:

```bash
sudo cp -a /usr/src/nvidia-535.309.01/nvidia/nv-kernel.o_binary   /usr/src/nvidia-535.309.01/nvidia/nv-kernel.o_binary.original
sudo cp nv-kernel-535-gen2.o /usr/src/nvidia-535.309.01/nvidia/nv-kernel.o_binary
sudo dkms build -m nvidia -v 535.309.01 -k "$(uname -r)" --force
sudo dkms install -m nvidia -v 535.309.01 -k "$(uname -r)" --force
sudo reboot
```

These installation steps assume that the NVIDIA driver and DKMS are already installed, as in the tested configuration. The modified NVIDIA binary is not included in the repository.

## 2. Switching the PCIe Generation

After rebooting, find the PCI addresses of the P104-100 GPUs:

```bash
lspci -Dnn | grep -i '10de:1b87'
```

From the repository directory, switch one GPU (replace `0f:00.0` with your PCI address):

```bash
sudo python3 p104_pcie_gen_unlock.py --gen 2 --devices 0f:00.0
sudo python3 p104_pcie_gen_unlock.py --gen 1 --devices 0f:00.0
```

Switch multiple GPUs with one command:

```bash
sudo python3 p104_pcie_gen_unlock.py --gen 2 --devices "0f:00.0,10:00.0,11:00.0"
sudo python3 p104_pcie_gen_unlock.py --gen 1 --devices "0f:00.0,10:00.0,11:00.0"
```

`--gen 1` / `--gen 2` **explicitly** selects the generation; `--devices` specifies one PCI address or a comma-separated list. The script reads `LnkCap` (maximum capabilities) and `LnkSta` (current link state) for the GPU and upstream PCIe port **before and after** switching, but does not select the generation automatically. Do not switch PCIe during active CUDA workloads. Changing the generation is an operation on the PCIe link; other topologies may experience errors or lose communication with the GPU.

Example of verified output:

```text
GPU 0000:0f:00.0 | PCIe Port 0000:00:1c.5
Before:
                  GPU             PCIe Port
  Max         Gen2 x4        Gen3 x1
  Current     Gen1 x1        Gen1 x1

Requested: Gen2

After:
                  GPU             PCIe Port
  Max         Gen2 x4        Gen3 x1
  Current     Gen2 x1        Gen2 x1
Result: Gen2 set
```

### Data Transfer Test (Optional)

```bash
python3 benchmarks/single_gpu/p104_pcie_transfer_bench.py --device 0f:00.0 --output gen2.log
```

Switch to Gen1 and repeat with `--output gen1.log`. The test uses `libcuda.so.1`, pinned host memory, and three transfers in each direction.

## 3. Restoring the Original Driver

If the `nv-kernel.o_binary.original` backup was saved:

```bash
sudo cp /usr/src/nvidia-535.309.01/nvidia/nv-kernel.o_binary.original   /usr/src/nvidia-535.309.01/nvidia/nv-kernel.o_binary
sudo dkms build -m nvidia -v 535.309.01 -k "$(uname -r)" --force
sudo dkms install -m nvidia -v 535.309.01 -k "$(uname -r)" --force
sudo reboot
```

**Scope of testing:** one-shot Gen1 ↔ Gen2 switching on one and three P104-100 GPUs, repeated switching, and CUDA and Qwen workloads on six P104-100 GPUs in Gen2. Long-term stability on arbitrary systems has not been established.
