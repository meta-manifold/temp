"""Small-M, large-W probe: every f16 pool size at split 1 over M x N x K. K in pairs that differ only in
their power-of-two factor (a cache-set conflict moves with it; a page / row-count limit does not), and K
from 1024 up so the address span of the W rows in flight (rows x K x bytes) crosses any limit.

  python memprobe.py [--work DIR] [--path P]   -> DIR/memprobe.out
"""
import argparse, os
import pool
from run import ROOT

MS = (32, 64, 128, 256)
NS = (4096, 16384)
KS = (1024, 2048, 4096, 4160, 8192, 8256, 9728, 9792, 16384, 16448)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--work'); ap.add_argument('--path', default='f16i16')
    a = ap.parse_args(); a.work = a.work or os.path.join(ROOT, 'build', f'memprobe-{a.path}'); os.makedirs(a.work, exist_ok=True)
    reps = pool.load_reps(a.work)
    path = os.path.join(a.work, 'memprobe.out'); done = set(pool.parse(path)); out = open(path, 'a')
    cs = [pool.cfg(a.path, c, 0) for c in reps.values()]
    for M in MS:
        for N in NS:
            for K in KS:
                if (M, K, N) in done: continue
                print((M, K, N), len(cs), 'configs, exit', pool.harness(a.path, (M, K, N), cs, out, (), pool.PASSES), flush=True)


if __name__ == '__main__':
    main()
