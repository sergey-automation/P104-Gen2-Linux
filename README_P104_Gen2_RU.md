# NVIDIA P104-100 — PCIe Gen2 unlock для Linux

**Адаптация для Linux** алгоритма из Windows-проекта [skbskhud/p104-unlock-v3](https://github.com/skbskhud/p104-unlock-v3). Спасибо автору первоначального метода.

В отличие от Windows-варианта, где для удержания Gen2 используются повторные обращения к регистрам приблизительно каждые **20 мс**, здесь переключение выполняется **один раз после загрузки Linux**. Изменение проприетарного драйвера NVIDIA предотвращает наблюдавшийся возврат Gen2 → Gen1 при запуске CUDA. Постоянный watchdog не нужен. После перезагрузки переключение надо выполнить снова.

**Проверенная конфигурация (09.11.2026):** Ubuntu 24.04, ядро `6.17.0-29-generic`, NVIDIA proprietary DKMS `535.309.01`, ASUS B250 Mining Expert, NVIDIA P104-100 8 GB (`10de:1b87`). Использованы **райзеры PCIe x1** — ограничения ширины x1 в результатах вызваны подключением, а не заявлением о максимальной ширине GPU. На проверенной карте `LnkCap` GPU = Gen2 x4, порта = Gen3 x1.

## Результат: пропускная способность PCIe

Одна P104-100, адрес `0000:0f:00.0`; pinned RAM, 32 MiB, по 3 измерения для каждого направления. В таблице медианы, GB/s (10⁹ байт/с).

| Передача | Gen1 x1 (2.5 GT/s) | Gen2 x1 (5.0 GT/s) | Ускорение |
|---|---:|---:|---:|
| RAM → GPU (H2D) | 0.199652 | 0.396509 | 1.986× |
| GPU → RAM (D2H) | 0.208926 | 0.418162 | 2.002× |

Исходные результаты: [`gen1.log`](benchmarks/single_gpu/gen1.log), [`gen2.log`](benchmarks/single_gpu/gen2.log). Воспроизведение: [`p104_pcie_transfer_bench.py`](benchmarks/single_gpu/p104_pcie_transfer_bench.py).

Дополнительно сравнивали **Qwen3.6-35B-A3B Q8_0** на шести P104-100: при контексте 7 141 / 22 400 / 37 659 токенов Decode вырос соответственно на **18.19% / 17.18% / 15.53%**, Prefill на **16.63% / 13.06% / 17.91%**. Подробная методика, 16 точек и исходные логи: [`benchmarks/qwen/README.md`](benchmarks/qwen/README.md).

## 1. Модификация драйвера NVIDIA

При инициализации CUDA исследованный драйвер сбрасывал Gen2 до Gen1. В нашем экземпляре проприетарного `nv-kernel.o_binary` версии `535.309.01` изменена одна инструкция:

```text
файл: /usr/src/nvidia-535.309.01/nvidia/nv-kernel.o_binary
смещение в файле: 0xCE8E5
было:  0D 00 00 08 00   (or eax, 0x80000)
стало: 0D 00 00 04 00   (or eax, 0x40000)
```

**Применимо только к проверенному экземпляру NVIDIA 535.309.01.** Для других сборок расположение и содержимое инструкции необходимо исследовать отдельно. Изменение закрытой части драйвера экспериментально; сделайте резервную копию, остановите CUDA-процессы и обеспечьте возможность восстановления системы. Включённый Secure Boot может потребовать подпись модуля.

Ниже — та же последовательность действий, которая использовалась в эксперименте, **без отдельного `patch_nv_kernel.py`**. Выполнять из рабочего каталога с правом записи:

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
assert data.count(old) == 1, f'Ожидалось 1 совпадение, найдено: {data.count(old)}'
assert data.find(old) == 0xCE8E5, f'Неожиданное смещение: 0x{data.find(old):x}'
shutil.copy2(src, backup)
patched.write_bytes(data.replace(old, new, 1))
print('Patched at:', hex(data.index(old)))
```

Установка пересобранного драйвера DKMS:

```bash
sudo cp -a /usr/src/nvidia-535.309.01/nvidia/nv-kernel.o_binary   /usr/src/nvidia-535.309.01/nvidia/nv-kernel.o_binary.original
sudo cp nv-kernel-535-gen2.o /usr/src/nvidia-535.309.01/nvidia/nv-kernel.o_binary
sudo dkms build -m nvidia -v 535.309.01 -k "$(uname -r)" --force
sudo dkms install -m nvidia -v 535.309.01 -k "$(uname -r)" --force
sudo reboot
```

Установка рассчитана на систему с уже установленным драйвером NVIDIA и DKMS, как в проверенной конфигурации. Изменённый бинарный объект NVIDIA в репозиторий не включён.

## 2. Переключение PCIe

После перезагрузки найдите PCI-адреса P104-100:

```bash
lspci -Dnn | grep -i '10de:1b87'
```

В каталоге репозитория переключите одну карту (замените `0f:00.0` своим PCI-адресом):

```bash
sudo python3 p104_pcie_gen_unlock.py --gen 2 --devices 0f:00.0
sudo python3 p104_pcie_gen_unlock.py --gen 1 --devices 0f:00.0
```

Несколько карт одной командой:

```bash
sudo python3 p104_pcie_gen_unlock.py --gen 2 --devices "0f:00.0,10:00.0,11:00.0"
sudo python3 p104_pcie_gen_unlock.py --gen 1 --devices "0f:00.0,10:00.0,11:00.0"
```

`--gen 1` / `--gen 2` задаёт режим **явно**, `--devices` — адрес или список адресов через запятую. Скрипт читает `LnkCap` (максимальные возможности) и `LnkSta` (текущий режим) GPU и вышестоящего PCIe-порта **до и после**, но не выбирает поколение автоматически. Не переключайте PCIe во время активных CUDA-задач. Смена поколения — операция над PCIe-соединением; в других топологиях возможен сбой или потеря связи с GPU.

Пример проверенного вывода:

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

### Проверка передачи данных (по желанию)

```bash
python3 benchmarks/single_gpu/p104_pcie_transfer_bench.py --device 0f:00.0 --output gen2.log
```

Переключитесь в Gen1 и повторите с `--output gen1.log`. Тест использует `libcuda.so.1`, закреплённую оперативную память и три передачи в каждом направлении.

## 3. Возврат стандартного драйвера

При сохранённой резервной копии `nv-kernel.o_binary.original`:

```bash
sudo cp /usr/src/nvidia-535.309.01/nvidia/nv-kernel.o_binary.original   /usr/src/nvidia-535.309.01/nvidia/nv-kernel.o_binary
sudo dkms build -m nvidia -v 535.309.01 -k "$(uname -r)" --force
sudo dkms install -m nvidia -v 535.309.01 -k "$(uname -r)" --force
sudo reboot
```

**Проверенный объём:** однократное переключение Gen1 ↔ Gen2 на одной и на трёх P104-100, повторные переключения, CUDA и Qwen на шести P104-100 в Gen2. Длительная стабильность на произвольных системах не установлена.
