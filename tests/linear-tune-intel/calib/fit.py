"""Cost-model calibration, step 3: fit the constants and score them (5-cost-model-calibration.md).

  python fit.py [--work DIR] [--path P]

Model, per tile t and split s (path P's tiles and reference tile, run.py):
  per  = ceil(tiles * s / cores), cap = 128 / (wgSize / 16), full = per // cap, rem = per % cap, r = rem / cap
  load = full + (floor + (1 - floor) * r if rem else 0)                    a lone part-filled wave never costs under floor
  work = ceil(K/s) * (load + tau * [full >= 1] + eps * [full >= 2 and rem == 0])
         * (1 + beta * max(0, log2(M * ceil(K/s) * bytes(I) / L2)))         I re-read from memory past the GPU L2
  cost = w_t * work + pi * w_ref * M * N * (s + 1) / 4096 * [the reduce pass runs]
         pi = pi_in while the partials M * N * s * 4 fit the GPU L2, pi_out otherwise
Stage 1  w_t  grid (split 1, no reduce), relative least squares, Huber-weighted: the anchor of stage 2
Stage 2  every constant on the training selection loss (grid, gridsplit, split, full, wave, focus):
         a softmin over each shape's configs (temperature 0.01, per-shape loss capped at 30 %) plus a pull
         on the w_t toward stage 1 (0.1 x sum of squared log ratios), Nelder-Mead from three starts
Score    val / val2 / test selection loss (pick / best - 1) and pi0.5's Linear time per inference
"""
import argparse, json, math, os, re, statistics as st
from collections import defaultdict
import numpy as np

import run

ROOT = run.ROOT
CORES = 8
L2 = 4 << 20                       # Meteor Lake Xe-LPG GPU L2, bytes
T_SOFT, PULL, CAP = 0.01, 0.1, 0.3
TRAIN = ('grid', 'gridsplit', 'split', 'full', 'wave', 'focus')
TILES, NAMES, REF = {}, [], 0      # set per path in main: {name: (wgM, wgN, wgSize)}, the reference tile's index
IBYTES = 4                         # I is fp32 in every measured path


def use_path(path):
    global TILES, NAMES, REF
    TILES = {t[0]: (t[1] // 16 // t[4] * t[3], t[4] * 16 * t[2], t[1]) for t in run.TILES[path]}
    NAMES = list(TILES); REF = NAMES.index(run.REF[path])


def load(work, name):
    """{shape: {(tile, split, reduce): median ms over passes 1 .. 8}}"""
    path = os.path.join(work, name + '.out'); runs = defaultdict(lambda: defaultdict(list)); shape = None
    if not os.path.exists(path): return {}
    for l in open(path, encoding='utf-8', errors='ignore'):
        m = re.search(r'M (\d+)  K (\d+)  N (\d+)', l)
        if m: shape = tuple(map(int, m.groups())); continue
        m = re.match(r'\s+pass (\d+)\s+s11 .*tile(\d+)x(\d+) .*?(?: opt(\d+))?(?: stage32)?\s+([\d.]+) ms', l)
        if m and int(m.group(1)) >= 1:
            opt = int(m.group(4) or 0)   # 0: split 1 without the reduce pass, s: s splits through it
            runs[shape][(f'{m.group(2)}x{m.group(3)}', max(opt, 1), opt > 0)].append(float(m.group(5)))
    return {s: {c: st.median(v) for c, v in d.items()} for s, d in runs.items()}


def scenario(d, reduce_at_one):
    """The configs Linear can run: split 1 with the reduce pass only for a layer with an epilogue."""
    return {k: v for k, v in d.items() if k[1] > 1 or k[2] == reduce_at_one}


def rows(data, reduce_at_one):
    """Config rows grouped per (shape, scenario); both scenarios when reduce_at_one is None."""
    out = []; g = 0
    for shp, d in data.items():
        for r1 in ((False, True) if reduce_at_one is None else (reduce_at_one(shp),)):
            c = scenario(d, r1)
            if len(c) < 2: continue
            best = min(c.values())
            for (t, s, red), y in c.items(): out.append((g, NAMES.index(t), shp[0], shp[2], shp[1], s, red, y, best))
            g += 1
    a = np.array(out, float)
    return dict(g=a[:, 0].astype(int), t=a[:, 1].astype(int), M=a[:, 2], N=a[:, 3], K=a[:, 4], s=a[:, 5],
                red=a[:, 6], y=a[:, 7], best=a[:, 8], n=g)


def cost(A, c):
    """The model for config rows A and constants c (dict: w list, floor, tau, eps, beta, pi_in, pi_out)."""
    tw = np.array([TILES[n] for n in NAMES], float); w = np.asarray(c['w'])
    wm, wn, wg = tw[A['t'], 0], tw[A['t'], 1], tw[A['t'], 2]
    cap = 128 // (wg // 16)
    per = np.ceil(np.ceil(A['M'] / wm) * np.ceil(A['N'] / wn) * A['s'] / CORES)
    full = np.floor(per / cap); rem = per - full * cap
    load = full + np.where(rem > 0, c['floor'] + (1 - c['floor']) * rem / cap, 0.0)
    ks = np.ceil(A['K'] / A['s'])
    work = ks * (load + c['tau'] * (full >= 1) + c['eps'] * ((full >= 2) & (rem == 0)))
    work = work * (1 + c['beta'] * np.maximum(0, np.log2(A['M'] * ks * IBYTES / L2)))
    pi = np.where(A['M'] * A['N'] * A['s'] * 4 <= L2, c['pi_in'], c['pi_out'])
    return w[A['t']] * work + pi * w[REF] * A['M'] * A['N'] * (A['s'] + 1) / 4096 * A['red']


def loss(A, cc, cap=None):
    order = np.lexsort((cc, A['g'])); first = np.ones(len(order), bool); first[1:] = A['g'][order][1:] != A['g'][order][:-1]
    l = A['y'][order][first] / A['best'][order][first] - 1
    return float(np.mean(np.minimum(l, cap) if cap else l))


def soft_loss(A, cc):
    cmin = np.full(A['n'], np.inf); np.minimum.at(cmin, A['g'], cc)
    z = np.exp(-(cc / cmin[A['g']] - 1) / T_SOFT)
    zs = np.zeros(A['n']); np.add.at(zs, A['g'], z)
    l = np.zeros(A['n']); np.add.at(l, A['g'], z * np.minimum(A['y'] / A['best'] - 1, CAP))
    return float(np.mean(l / zs))


def huber_lsq(X, Y, c=0.1):
    """Least squares on relative rows, reweighted so points far off (shapes slow for reasons outside the
    model) pull less."""
    w = np.ones(len(Y))
    for _ in range(10):
        coef, *_ = np.linalg.lstsq(X * w[:, None], Y * w, rcond=None)
        r = np.abs(X @ coef - Y); w = np.sqrt(np.where(r < c, 1.0, c / np.maximum(r, 1e-9)))
    return coef, float(np.median(r))


def stage1(grid, floor=0.65):
    """w_t from split 1 without the reduce pass, the last wave at max(r, floor): the pull's anchor."""
    X, Y = [], []
    for (M, K, N), d in grid.items():
        for (t, s, red), y in d.items():
            if s != 1 or red: continue
            wm, wn, wg = TILES[t]; cap = 128 // (wg // 16)
            per = math.ceil(math.ceil(M / wm) * math.ceil(N / wn) / CORES); full, rem = divmod(per, cap)
            x = [0.0] * len(NAMES); x[NAMES.index(t)] = K * (full + (max(rem / cap, floor) if rem else 0)) / y
            X.append(x + [1 / y]); Y.append(1.0)                 # intercept: a fixed launch cost
    coef, err = huber_lsq(np.array(X), np.array(Y))
    return coef[:len(NAMES)], err


def nelder_mead(f, x0, step, iters=4000, tol=1e-7):
    n = len(x0); pts = [np.array(x0, float)] + [np.array(x0, float) + np.eye(n)[i] * step for i in range(n)]
    vals = [f(p) for p in pts]
    for _ in range(iters):
        o = np.argsort(vals); pts = [pts[i] for i in o]; vals = [vals[i] for i in o]
        if abs(vals[-1] - vals[0]) < tol: break
        c = np.mean(pts[:-1], axis=0); xr = c + (c - pts[-1]); fr = f(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - pts[-1]); fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]: pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c); fc = f(xc)
            if fc < vals[-1]: pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0]] + [pts[0] + 0.5 * (p - pts[0]) for p in pts[1:]]; vals = [vals[0]] + [f(p) for p in pts[1:]]
    return pts[int(np.argmin(vals))]


def stage2(A, w0):
    """Every constant on the training selection loss. x: log w offsets (all tiles but the reference),
    floor, tau, eps, beta, log pi_in, log pi_out."""
    others = [i for i in range(len(NAMES)) if i != REF]
    def unpack(x):
        lw = np.log(w0).copy(); lw[others] += x[:len(others)]; k = len(others)
        return dict(w=np.exp(lw), floor=float(np.clip(x[k], 0, 1)), tau=max(x[k + 1], 0.0), eps=max(x[k + 2], 0.0),
                    beta=max(x[k + 3], 0.0), pi_in=math.exp(x[k + 4]), pi_out=math.exp(x[k + 5]))
    def f(x): return soft_loss(A, cost(A, unpack(x))) + PULL * float(np.sum(x[:len(others)] ** 2))
    x0 = np.array([0.0] * len(others) + [0.48, 0.13, 0.4, 0.08, math.log(0.42), math.log(0.88)])
    best = None
    for jit in (0, 1, -1):
        start = x0 + jit * 0.05 * np.array([1 if i % 2 else -1 for i in range(len(x0))])
        x = nelder_mead(f, start, 0.05)
        if best is None or f(x) < best[0]: best = (f(x), x)
    return unpack(best[1])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--work', default=os.path.join(ROOT, 'build', 'calib'))
    ap.add_argument('--path', default='f16', choices=list(run.PATHS)); a = ap.parse_args(); use_path(a.path)
    S = {n: load(a.work, n) for n in run.SETS}
    shapes = json.load(open(os.path.join(a.work, 'shapes.json')))
    epi = {tuple(s): bool(e) for s, e in zip(shapes['test'], shapes['test_epilogue'])}
    print('loaded:', {k: len(v) for k, v in S.items()})

    w0, err = stage1(S['grid'])
    print(f'stage 1: median |rel err| {100*err:.1f}%; w (us) ' + ', '.join(f'{n} {1e3*w0[i]:.3f}' for i, n in enumerate(NAMES)))

    train = defaultdict(dict)
    for n in TRAIN:
        for shp, d in S[n].items(): train[shp].update(d)
    TR = rows(train, None)
    c = stage2(TR, w0)
    print(f'stage 2: train selection loss {100*loss(TR, cost(TR, c), CAP):.2f}% over {TR["n"]} (shape, scenario) groups')

    print('\nscore (selection loss vs the best measured config):')
    for name in ('val', 'val2'):
        if S[name]: A = rows(S[name], None); print(f'  {name:5s} both scenarios  {100*loss(A, cost(A, c)):5.2f}%  ({A["n"]} groups)')
    if S['test']:
        A = rows(S['test'], lambda s: epi.get(s, False)); print(f'  test  as the layer     {100*loss(A, cost(A, c)):5.2f}%  ({A["n"]} layers)')

    # pi0.5: one inference's Linear time, each layer's calls (layers x per-layer count x denoise steps).
    calls = {(768, 1152, 1152): 108, (768, 1152, 4304): 27, (768, 4304, 1152): 27, (768, 1152, 2048): 1,
             (922, 2048, 2048): 36, (922, 2048, 256): 36, (922, 2048, 32768): 18, (922, 16384, 2048): 18,
             (50, 32, 1024): 10, (50, 1024, 2048): 180, (50, 1024, 256): 360, (50, 2048, 1024): 180,
             (50, 1024, 8192): 180, (50, 4096, 1024): 180, (50, 1024, 32): 10}
    tb = tp = 0.0
    for shp, n in calls.items():
        if shp not in S['test']: continue
        A = rows({shp: S['test'][shp]}, lambda s: epi.get(s, False)); cc = cost(A, c); i = int(np.argmin(cc))
        tb += n * A['best'][i]; tp += n * A['y'][i]
    if tb: print(f'\npi0.5 Linear per inference (ms): best {tb:.1f}, picked {tp:.1f} ({100*(tp/tb-1):+.1f}%)')

    print('\nconstants for the ' + a.path + ' candidate table / intelTileParams (Linear):')
    print('  waveTime (us): ' + ', '.join(f'{n} {1e3*c["w"][i]:.3f}' for i, n in enumerate(NAMES)))
    print(f"  floor {c['floor']:.3f}, tail {c['tau']:.3f}, wholeWaves {c['eps']:.3f}, inputReread {c['beta']:.3f}, "
          f"reducePenalty in L2 {c['pi_in']:.3f} / past L2 {c['pi_out']:.3f}")


if __name__ == '__main__':
    main()
