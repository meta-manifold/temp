"""Tile candidates from a pool, by measurement only (no cost model).

  python pool.py rep     [--work DIR] [--path P] [--sg 16,8]   -> DIR/rep.out, DIR/rep.json: each pool size's config per SIMD width
  python pool.py measure [--work DIR] [--path P]               -> DIR/pool.out: every size's config at every split
  python pool.py select  [--work DIR] [--path P]               -> reduce to --keep, add back the last two dropped: DIR/chosen.json

Pool: wgM x wgN over {32, 48, 64, 96, 128, 192, 256}^2 per SIMD width sg, each size built from
(wgSize, sg, tM, tN, SGN) within the lane's register budget (a thread's 4 KB over sg lanes: 256 B at
SIMD16, 512 B at SIMD8); a config that spills (scratch > 0) or misses its SIMD width is dropped.
SIMD8 is for the fp32 paths: fp16's double rate needs SIMD16. Paths: f16 (fp16 accumulation),
f32w16 (fp32 accumulation, W fp16), f32w32 (I and W fp32).
A config's time is the mean of the faster half of passes 1 .. P-1 (the noise only slows a run down).
"""
import argparse, json, math, os, random, re, statistics as st, subprocess
from collections import defaultdict

import numpy as np

from gen import grid_like
from run import EXE, PATHS, ROOT, SPLITS, ok_split

SIZES = (32, 48, 64, 96, 128, 192, 256)
WGK = 8
PASSES = 6          # harness passes; pass 0 warms up
# Representative shapes for picking a size's config: the pool's distribution (log-uniform, M and N in
# steps of 16), so under- and over-filled cores, narrow N and long K all show up.
REP_SHAPES = grid_like(20, 21, 16)
VAL_SHAPES = 50     # of the measured pool shapes, held out to judge the chosen set
NAME = re.compile(r's11 wg(\d+)\s+sg(\d+)\s+t(\d+)x(\d+) it(\d+)x(\d+) tile(\d+)x(\d+) k\d+\s+sgN\d+\s+f\d+acc req\d+(?: opt(\d+))?')


def configs(path, sgs=(16,)):
    """{(wgM, wgN, sg): [(wgSize, sg, tM, tN, SGN)]} within the lane's register budget at each SIMD width."""
    _, _, _, accB, cacheB = PATHS[path]; out = defaultdict(list)
    for sg in sgs:
        budget = 4096 // sg
        for wg in (128, 256, 512):
            subgroups = wg // sg
            for sgn in (1, 2, 4, 8, 16):
                if subgroups % sgn: continue
                for tm in range(1, 33):
                    for tn in range(1, 17):
                        if tm * tn * accB + tn * WGK * cacheB > budget: continue
                        wgM, wgN = subgroups // sgn * tm, sgn * sg * tn
                        if wgM in SIZES and wgN in SIZES: out[(wgM, wgN, sg)].append((wg, sg, tm, tn, sgn))
    return out


def size_name(wgM, wgN, sg):
    return f'{wgM}x{wgN}-sg{sg}'


def load_reps(work):
    """rep.json as {name: (wgSize, sg, tM, tN, SGN)}; a config of four (before SIMD8) is SIMD16."""
    reps = json.load(open(os.path.join(work, 'rep.json')))
    return {(k if '-sg' in k else k + '-sg16'): tuple(v if len(v) == 5 else (v[0], 16, *v[1:])) for k, v in reps.items()}


def cfg(path, c, opt, wgK=WGK):
    """Harness config string; order as Linear's: x walks M tiles (0) when wgM * sizeof(I) > wgN * sizeof(W).
    W padding as Linear's: none with fp16 staging, one vec4 slot with fp32 staging."""
    wg, sg, tm, tn, sgn = c; acc, stage, flags, _, _ = PATHS[path]
    wgM, wgN = wg // sg // sgn * tm, sgn * sg * tn
    order = 0 if wgM * (2 if '-i16' in flags else 4) > wgN * (4 if path == 'f32w32' else 2) else 1
    return f'11,{wg},{sg},{tn},{tm},{order},{sgn},0,{wgK},{stage},{acc},0,{opt},{stage}'


def harness(path, shape, cs, out, extra=(), passes=PASSES):
    M, K, N = shape
    r = subprocess.run([EXE, '-passes', str(passes), '-flush', *PATHS[path][2], *extra, '-shape', f'{M},{K},{N}', *cs],
                       capture_output=True, text=True)
    out.write(r.stdout); out.flush()
    return r.returncode


def fast_half(v):
    """Mean of the faster half (at least one): interference and clock drops only lengthen a run."""
    v = sorted(v); return st.mean(v[:max(1, len(v) // 2)])


def parse(path):
    """{shape: {(wgSize, sg, tM, tN, SGN, opt): {'ms': fast_half of passes 1 .., 'rel': maxRel, 'scratch': B, 'sg': n}}}"""
    res = defaultdict(lambda: defaultdict(dict)); shape = None; runs = defaultdict(list)
    def key(m): return tuple(int(m.group(i)) for i in (1, 2, 3, 4, 5)) + (int(m.group(9) or 0),)
    if not os.path.exists(path): return res
    for l in open(path, encoding='utf-8', errors='ignore'):
        m = re.search(r'M (\d+)  K (\d+)  N (\d+)', l)
        if m and not l.startswith(' '): shape = tuple(map(int, m.groups())); continue
        n = NAME.search(l)
        if not n or shape is None: continue
        k = key(n)
        p = re.match(r'\s+pass (\d+)\s', l)
        if p:
            if int(p.group(1)) >= 1: runs[(shape, k)].append(float(re.search(r'([\d.]+) ms', l).group(1)))
            continue
        rel = re.search(r'maxRel ([\d.e+-]+)', l); sc = re.search(r'Scratch Memory Size=(\d+)', l)
        sg = re.search(r'\[compute sg(\d+)\]', l)
        res[shape][k].update(rel=float(rel.group(1)) if rel else 1.0, scratch=int(sc.group(1)) if sc else -1,
                             sg=int(sg.group(1)) if sg else 0)
    for (shape, k), v in runs.items(): res[shape][k]['ms'] = fast_half(v)
    return res


def usable(r):
    return 'ms' in r and r.get('rel', 1.0) <= 5e-2


def cmd_rep(a):
    by_size = configs(a.path, a.sg); out = open(os.path.join(a.work, 'rep.out'), 'a'); done = set(parse(os.path.join(a.work, 'rep.out')))
    cs = [cfg(a.path, c, 0) for size in sorted(by_size) for c in by_size[size]]
    print(f'{len(by_size)} sizes, {len(cs)} configs, {len(REP_SHAPES)} shapes, SIMD {a.sg}')
    for shp in REP_SHAPES:   # one shape per call, every config of the pool interleaved
        if shp not in done: print(shp, 'exit', harness(a.path, shp, cs, out, ['-stats'], a.passes))
    res = parse(os.path.join(a.work, 'rep.out')); reps = {}
    for size, cands in sorted(by_size.items()):
        score = {}
        for c in cands:
            rows = [res[s].get((*c, 0)) for s in REP_SHAPES]
            if any(r is None or not usable(r) or r['scratch'] != 0 or r['sg'] != c[1] for r in rows): continue
            # each shape's time over the size's best there: the config whose worst shape loses least (mean breaks ties)
            r = [r['ms'] / min(res[s][(*o, 0)]['ms'] for o in cands if usable(res[s].get((*o, 0), {})))
                 for s, r in zip(REP_SHAPES, rows)]
            score[c] = (max(r), st.mean(r))
        name = size_name(*size)
        if score:
            best = min(score, key=score.get); reps[name] = list(best)
            print(f'{name:14s} {len(cands)} configs, {len(score)} clean -> {best} '
                  f'(over the per-shape best: worst {100*(score[best][0]-1):.1f}%, mean {100*(score[best][1]-1):.1f}%)')
        else:
            print(f'{name:14s} {len(cands)} configs, none clean -> dropped')
    json.dump(reps, open(os.path.join(a.work, 'rep.json'), 'w'), indent=1)


def cmd_measure(a):
    reps = load_reps(a.work)
    # extra.json: [wgSize, sg, tM, tN, SGN, wgK] configs measured besides the sizes' (e.g. an earlier Linear's tiles)
    xf = os.path.join(a.work, 'extra.json'); extra = [tuple(c) for c in json.load(open(xf))] if os.path.exists(xf) else []
    shapes = grid_like(a.shapes, a.seed, a.step)
    if a.seed != 11:   # a later batch: its own files, none of the shapes already in an earlier batch
        taken = {tuple(x) for f in os.listdir(a.work) if f.startswith('pool-shapes') and f != f'pool-shapes-seed{a.seed}.json'
                 for x in json.load(open(os.path.join(a.work, f)))}
        shapes = [x for x in shapes if x not in taken]
    tag = '' if a.seed == 11 else f'-seed{a.seed}'
    path = os.path.join(a.work, f'pool{tag}.out')
    done = {s for s, d in parse(path).items() if any('ms' in r for r in d.values())}; out = open(path, 'a')
    print(f'{len(reps)} sizes + {len(extra)} extra configs x splits on {len(shapes)} shapes')
    for shp in shapes:   # every size of both SIMD widths in one call, so they share the GPU state
        if shp in done: continue
        M, K, N = shp
        cs = [cfg(a.path, c, s if s > 1 else 0) for c in reps.values() for s in SPLITS if ok_split(M, N, K, s)]
        cs += [cfg(a.path, c[:5], s if s > 1 else 0, c[5]) for c in extra for s in SPLITS if ok_split(M, N, K, s)]
        code = harness(a.path, shp, cs, out, (), a.passes)
        print(shp, len(cs), 'configs, exit', code, flush=True)
        if code:   # the harness holds every config's partials at once: one split per call instead
            for s in SPLITS:
                if ok_split(M, N, K, s):
                    print('  split', s, 'exit', harness(a.path, shp, [cfg(a.path, c, s if s > 1 else 0) for c in reps.values()]
                                                       + [cfg(a.path, c[:5], s if s > 1 else 0, c[5]) for c in extra], out, (), a.passes))
    json.dump(shapes, open(os.path.join(a.work, f'pool-shapes{tag}.json'), 'w'))


def losses(T, S):
    """Per shape: the best time within subset S over the best of the whole pool, minus 1."""
    return T[:, list(S)].min(axis=1) / T.min(axis=1) - 1


def loss(T, S):
    """What the subsets are chosen on: the mean loss plus the loss past 5 % (the tail), averaged."""
    l = losses(T, S); return float(np.mean(l) + np.mean(np.maximum(l - 0.05, 0)))


def tails(T, S):
    l = losses(T, S)
    return f'mean {100*l.mean():5.2f}%  max {100*l.max():5.1f}%  >5% {100*(l > .05).mean():4.1f}%'


def cmd_select(a):
    reps = load_reps(a.work); names = list(reps)
    res = parse(os.path.join(a.work, 'pool.out')); shapes = [tuple(s) for s in json.load(open(os.path.join(a.work, 'pool-shapes.json')))]
    rows, kept = [], []
    for shp in shapes:   # each size's best time over its splits
        d = res.get(shp, {}); best = []
        for nm in names:
            v = [r['ms'] for k, r in d.items() if k[:5] == reps[nm] and usable(r)]
            best.append(min(v) if v else math.inf)
        if all(math.isfinite(b) for b in best): rows.append(best); kept.append(shp)
    order = np.random.RandomState(31).permutation(len(rows))
    val = set(order[:VAL_SHAPES].tolist())
    T = np.array([r for i, r in enumerate(rows) if i not in val]); V = np.array([r for i, r in enumerate(rows) if i in val])
    n = len(names)
    print(f'{len(rows)} shapes ({T.shape[0]} to choose on, {V.shape[0]} held out) x {n} sizes')

    # Backward elimination from the whole pool down to --keep, each step polished by swaps; the dropped order kept.
    S = set(range(n)); dropped = []
    print(' count  choose-set loss                       held-out loss                         set')
    while len(S) > a.keep:
        out = min(S, key=lambda i: loss(T, S - {i})); S.remove(out); dropped.append(out)
        improved = True
        while improved:
            improved = False
            for i in list(S):
                for j in set(range(n)) - S:
                    if loss(T, S - {i} | {j}) < loss(T, S) - 1e-9: S = S - {i} | {j}; improved = True; break
                if improved: break
        if len(S) <= 16: print(f'  {len(S):3d}  {tails(T, S)}   {tails(V, S)}   ' + ', '.join(names[i] for i in sorted(S)))
    last = [i for i in reversed(dropped) if i not in S][:2]
    final = sorted(S | set(last))
    print(); print(f'kept {a.keep}: ' +', '.join(names[i] for i in sorted(S)) + ' | added back (last dropped): ' + ', '.join(names[i] for i in last))
    print(f'final {len(final)}: choose-set {tails(T, final)} | held-out {tails(V, final)}')
    chosen = {names[i]: reps[names[i]] for i in final}
    json.dump(chosen, open(os.path.join(a.work, 'chosen.json'), 'w'), indent=1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['rep', 'measure', 'select'])
    ap.add_argument('--work'); ap.add_argument('--shapes', type=int, default=80)
    ap.add_argument('--keep', type=int, default=7)    # select: reduce to this many, then add back the last two dropped
    ap.add_argument('--step', type=int, default=16)   # M, N rounding of the pool shapes
    ap.add_argument('--seed', type=int, default=11)   # measure: shape seed; another seed adds a batch (pool-seed<N>.out)
    ap.add_argument('--path', default='f16', choices=list(PATHS))
    ap.add_argument('--sg')                            # rep: SIMD widths, comma separated (default: 16 for f16, 16,8 for fp32)
    ap.add_argument('--passes', type=int, default=PASSES)
    a = ap.parse_args(); a.work = a.work or os.path.join(ROOT, 'build', f'calib-pool-{a.path}'); os.makedirs(a.work, exist_ok=True)
    a.sg = tuple(int(x) for x in a.sg.split(',')) if a.sg else ((16,) if a.path.startswith('f16') else (16, 8))
    {'rep': cmd_rep, 'measure': cmd_measure, 'select': cmd_select}[a.cmd](a)


if __name__ == '__main__':
    main()
