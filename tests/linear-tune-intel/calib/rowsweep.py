"""The W stream alone (rows-probe.comp): read rate over rows in flight, row stride, bytes per row per step,
workgroup size and buffer size.

  python rowsweep.py [--work DIR]   -> DIR/rows.out
"""
import argparse, os, subprocess
from run import EXE, ROOT

STRIDES = (1024, 2048, 4096, 8192, 16384, 19456, 19584, 26112, 32768, 32832, 65536)
ROWS = (8, 16, 32, 48, 64, 96, 128, 192, 256)


def specs():
    out = []
    for c in (1, 2, 4):                                  # 16, 32, 64 B a row per step
        for r in ROWS:
            if r * c > 512: continue                     # 8 KB of shared memory a workgroup at most: 8 fit a core
            out += [f'256,{r},{c},{s},{s}' for s in STRIDES]
    for wg in (128, 512):                                # other workgroup sizes, 16 B a step
        out += [f'{wg},{r},1,{s},{s}' for r in ROWS for s in STRIDES]
    for mb in (16, 64, 768):                             # buffer size, at the pi0.5-like stride
        out += [f'256,{r},1,19456,19456,{mb}' for r in ROWS]
    for s in (4096, 19456, 32768):                       # rows shorter than their stride: the same span, fewer bytes
        out += [f'256,{r},1,{s // 4},{s}' for r in ROWS]
    for _ in range(6):                                   # a fresh buffer each time: does the allocation move the rate?
        out += [f'256,{r},1,19456,19456,768' for r in (32, 128)]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--work')
    a = ap.parse_args(); a.work = a.work or os.path.join(ROOT, 'build', 'rowsweep'); os.makedirs(a.work, exist_ok=True)
    sp = specs(); out = open(os.path.join(a.work, 'rows.out'), 'a')
    for i in range(0, len(sp), 20):                     # a batch per process
        args = []
        for x in sp[i:i + 20]: args += ['-rows', x]
        r = subprocess.run([EXE, *args], capture_output=True, text=True)
        out.write(r.stdout); out.flush()
        print(i + len(sp[i:i + 20]), '/', len(sp), 'exit', r.returncode, flush=True)


if __name__ == '__main__':
    main()
