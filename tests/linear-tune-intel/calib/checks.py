"""Three short checks after the pools (each a harness run per shape):

  python checks.py remeasure   -> DIR/pool-f16i16/remeasure-{1,2}.out: the collapsed I16 shapes, every config, twice
  python checks.py wpad        -> DIR/wpad-f32w32.out: f32w32 configs at W padding 0 and 1
  python checks.py autotune    -> DIR/autotune-<path>.out: S15's top 2 and their split neighbours, one quick pass each
Specs (shapes and configs) come from DIR/checks.json, written by the analysis.
"""
import json, os, subprocess, sys
import pool
from run import EXE, PATHS, ROOT, SPLITS, ok_split

B = os.path.join(ROOT, 'build')


def remeasure(spec):
    W = os.path.join(B, 'pool-f16i16'); reps = pool.load_reps(W)
    extra = [tuple(c) for c in json.load(open(os.path.join(W, 'extra.json')))]
    for i in (1, 2):
        out = open(os.path.join(W, f'remeasure-{i}.out'), 'a')
        for M, K, N in spec['remeasure']:
            cs = [pool.cfg('f16i16', c, s if s > 1 else 0) for c in reps.values() for s in SPLITS if ok_split(M, N, K, s)]
            cs += [pool.cfg('f16i16', c[:5], s if s > 1 else 0, c[5]) for c in extra for s in SPLITS if ok_split(M, N, K, s)]
            print('remeasure', i, (M, K, N), 'exit', pool.harness('f16i16', (M, K, N), cs, out), flush=True)


def wpad(spec):
    out = open(os.path.join(B, 'wpad-f32w32.out'), 'a')
    for (M, K, N), cfgs in spec['wpad']:
        cs = []
        for c, s in cfgs:
            base = pool.cfg('f32w32', tuple(c), s if s > 1 else 0).split(',')
            for pad in ('0', '1'):
                base[9] = pad; cs.append(','.join(base))
        print('wpad', (M, K, N), 'exit', pool.harness('f32w32', (M, K, N), cs, out), flush=True)


def autotune(spec):
    for path in ('f32w16', 'f32w32'):
        out = open(os.path.join(B, f'autotune-{path}.out'), 'a')
        for (M, K, N), cfgs in spec['autotune'][path]:
            cs = [pool.cfg(path, tuple(c), s if s > 1 else 0) for c, s in cfgs]
            # a load-time pick: no L2 flush, pass 0 warms up, pass 1 is the one timing
            r = subprocess.run([EXE, '-passes', '2', *PATHS[path][2], '-shape', f'{M},{K},{N}', *cs], capture_output=True, text=True)
            out.write(r.stdout); out.flush()
            print('autotune', path, (M, K, N), 'exit', r.returncode, flush=True)


if __name__ == '__main__':
    spec = json.load(open(os.path.join(B, 'checks.json')))
    for step in sys.argv[1:]:
        {'remeasure': remeasure, 'wpad': wpad, 'autotune': autotune}[step](spec)
