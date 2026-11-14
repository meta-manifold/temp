"""Cost-model calibration, step 2: measurements (5-cost-model-calibration.md).

  python run.py [--work DIR] [--limit N] [--path P] [--sets a,b]  -> DIR/<set>.out for every set of DIR/shapes.json (gen.py), in one session

Every timed run: linear-tile-11.comp (Linear's linear-tile-intel.comp with W padding as a constant) and split-reduce.comp,
GPU L2 flushed ahead of the run (-flush), 9 interleaved passes.
A config's opt (13th field) is the split, always through the reduce pass; opt 0 is split 1 without it.
"""
import argparse, json, os, re, subprocess
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
EXE = os.path.join(ROOT, 'build', 'bin', 'Release', 'test-linear-tune-intel.exe')
SETS = ['grid', 'pairs', 'gridsplit', 'split', 'full', 'wave', 'focus', 'val', 'val2', 'test']

# path: (acc, stage32, harness flags, accumulator bytes, register cache bytes). f16: fp16 accumulation;
# f32w16: fp32 accumulation, W fp16; f32w32: I and W fp32, staged and accumulated in fp32; i16: I fp16.
PATHS = {'f16': (1, 0, [], 2, 2), 'f32w16': (0, 0, [], 4, 2), 'f32w32': (0, 1, ['-w32'], 4, 4),
         'f16i16': (1, 0, ['-i16'], 2, 2), 'f32w16i16': (0, 0, ['-i16'], 4, 2)}   # I fp16
# (name, wgSize, tN, tM, SGN, order) per path: the candidates pool.py chose; order 0 = x walks M tiles,
# Linear's N-first (wgM * sizeof(I) > wgN * sizeof(W)).
TILES = {'f16': [('32x64', 128, 4, 4, 1, 1), ('48x96', 256, 6, 3, 1, 1), ('96x192', 512, 6, 6, 2, 1),
                 ('96x256', 512, 8, 6, 2, 1), ('128x192', 512, 6, 8, 2, 0), ('128x256', 512, 8, 8, 2, 1),
                 ('192x96', 512, 6, 6, 1, 0), ('192x128', 512, 8, 6, 1, 0), ('256x128', 512, 8, 8, 1, 0)],
         'f32w16': [('32x64', 128, 2, 8, 2, 1), ('64x32', 128, 2, 8, 1, 0)],
         'f32w32': [('32x64', 128, 2, 8, 2, 1), ('48x64', 128, 2, 12, 2, 1), ('64x128', 256, 2, 16, 4, 1)]}
# Per path: the reference tile (pairs, split, the reduce penalty's unit) and gridsplit's two tiles, so the
# split terms are measured at every core occupancy (f16: reference 4 workgroups a core, gridsplit 16 and 8).
REF = {'f16': '192x128', 'f32w16': '32x64', 'f32w32': '64x128'}
GRIDSPLIT = {'f16': ('32x64', '48x96'), 'f32w16': ('32x64', '64x32'), 'f32w32': ('32x64', '64x128')}
SPLITS = [1, 2, 4, 8, 16]
MAX_PARTIAL = 1 << 30   # M * N * s * 4 bytes of partials at most


def cfg(path, tile, s, reduce):
    _, wg, tn, tm, sgn, order = next(t for t in TILES[path] if t[0] == tile)
    acc, stage, _, _, _ = PATHS[path]   # W padding as Linear's: one vec4 slot with fp32 staging only
    return f'11,{wg},16,{tn},{tm},{order},{sgn},0,8,{stage},{acc},0,{s if reduce else 0},{stage}'


def ok_split(M, N, K, s):
    return (s == 1 or K // s >= 64) and M * N * s * 4 <= MAX_PARTIAL


def plan(name, shapes, path='f16'):
    """(shape, configs) per shape of a set."""
    tiles, ref = [t[0] for t in TILES[path]], REF[path]
    for M, K, N in shapes:
        if name == 'grid':          # tile constants: every tile, split 1, no reduce pass
            cs = [cfg(path, t, 1, False) for t in tiles]
        elif name == 'pairs':       # reduce cost at split 1: with and without the pass
            cs = [cfg(path, ref, 1, False), cfg(path, ref, 1, True)]
        elif name == 'gridsplit':   # split terms across tiles: 16 and 4 workgroups a core, at split 2 .. 16
            cs = [cfg(path, t, s, True) for t in GRIDSPLIT[path] for s in SPLITS[1:] if ok_split(M, N, K, s)]
        elif name == 'split':       # split terms across shapes: the reference tile at every split
            cs = [cfg(path, ref, 1, False)] + [cfg(path, ref, s, True) for s in SPLITS if ok_split(M, N, K, s)]
        else:                       # full / wave / focus / val / val2 / test: every tile and split, split 1 with and without
            cs = [cfg(path, t, 1, False) for t in tiles] + \
                 [cfg(path, t, s, True) for t in tiles for s in SPLITS if ok_split(M, N, K, s)]
        yield (M, K, N), cs


def done_shapes(path):
    """Shapes whose block in an output file reached the last pass."""
    if not os.path.exists(path): return set()
    txt = open(path, encoding='utf-8', errors='ignore').read(); out = set()
    for blk in re.split(r'\n(?=\S+  M \d+  K)', txt):
        m = re.search(r'M (\d+)  K (\d+)  N (\d+)', blk)
        if m and ' pass 8 ' in blk: out.add(tuple(map(int, m.groups())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', default=os.path.join(ROOT, 'build', 'calib')); ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--path', default='f16', choices=list(PATHS))
    ap.add_argument('--sets', default=','.join(SETS))   # the sets to measure, comma separated (e.g. all but focus)
    a = ap.parse_args(); flags = PATHS[a.path][2]
    sets = json.load(open(os.path.join(a.work, 'shapes.json')))
    log = open(os.path.join(a.work, 'run.log'), 'a')
    for name in [n for n in SETS if n in a.sets.split(',')]:   # one session, every set in a row, so the GPU state is shared
        shapes = [tuple(s) for s in sets[name]][:a.limit or None]
        groups = defaultdict(list)
        for shp, cs in plan(name, shapes, a.path): groups[tuple(cs)].append(shp)
        out = os.path.join(a.work, name + '.out'); done = done_shapes(out)
        with open(out, 'a') as f:
            for cs, shps in groups.items():
                todo = [x for x in shps if x not in done]
                for i in range(0, len(todo), 8):   # a long harness call died part way: 8 shapes per call
                    args = [EXE, '-passes', '9', '-flush', *flags]
                    for M, K, N in todo[i:i + 8]: args += ['-shape', f'{M},{K},{N}']
                    r = subprocess.run(args + list(cs), capture_output=True, text=True)
                    f.write(r.stdout); f.flush()
                    log.write(f'{name}: {len(todo[i:i + 8])} shapes, exit {r.returncode}\n'); log.flush()
                    if r.returncode:   # the harness holds every config's partials at once: a shape and a split per call
                        f.flush(); left = [x for x in todo[i:i + 8] if x not in done_shapes(out)]
                        for M, K, N in left:
                            for opt in sorted({c.split(',')[12] for c in cs}, key=int):
                                r = subprocess.run([EXE, '-passes', '9', '-flush', *flags, '-shape', f'{M},{K},{N}',
                                                    *[c for c in cs if c.split(',')[12] == opt]], capture_output=True, text=True)
                                f.write(r.stdout); f.flush()
                                log.write(f'{name}: {M},{K},{N} opt {opt}, exit {r.returncode}\n'); log.flush()
        missing = [x for x in shapes if x not in done_shapes(out)]
        print(f'{name:9s} done, {len(missing)} shapes missing' + (f' (run again to fill): {missing[:4]}' if missing else ''))


if __name__ == '__main__':
    main()
