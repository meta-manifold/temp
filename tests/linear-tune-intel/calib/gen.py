"""Cost-model calibration, step 1: the shapes of every set (5-cost-model-calibration.md).

  python gen.py [--work DIR] [--path P]  -> DIR/shapes.json, and each set's shape and config counts (for path P's tiles)

Seeded, so a run gives the same shapes every time (later sets skip shapes an earlier set took).
run.py measures them.
"""
import argparse, json, math, os, random

from run import MAX_PARTIAL, PATHS, ROOT, SETS, TILES, plan


def r128(x): return max(128, int(round(x / 128)) * 128)


def grid_like(n, seed, step=128):
    """Log-uniform M 32 .. 4096, N 128 .. 32768, K 256 .. 16384. M and N round to multiples of step
    (step 128: M under 128 in 32 steps); K to 128. A step under 128 lets every tile's edge padding vary."""
    rng = random.Random(seed); out = set()
    while len(out) < n:
        M = 2 ** rng.uniform(5, 12)
        if step >= 128: M = max(32, int(round(M / 32)) * 32) if M < 128 else r128(M)
        else: M = max(32, int(round(M / step)) * step)
        N = r128(2 ** rng.uniform(7, 15)) if step >= 128 else max(128, int(round(2 ** rng.uniform(7, 15) / step)) * step)
        K = r128(2 ** rng.uniform(8, 14))
        if M * N * K <= 1 << 36 and M * N * 16 * 4 <= MAX_PARTIAL: out.add((M, K, N))
    return sorted(out)


def split_set(n, seed):
    """128 x 128 tile count T, K and N / M ratio each spread evenly (Latin hypercube per factor);
    M and N multiples of 128, so no tile is partly padding."""
    Ts = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 56, 64, 72, 96, 120, 128, 136, 184, 192, 200, 256]
    Ks = [256, 512, 1024, 2048, 4096, 8192, 16384]
    Rs = [1 / 16, 1 / 8, 1, 8, 16]
    rng = random.Random(seed)
    def column(vals):
        c = [vals[i % len(vals)] for i in range(n)]; rng.shuffle(c); return c
    out = set()
    while len(out) < n:   # small T has few M x N factorings: draw more rounds until n distinct
        for T, K, r in zip(column(Ts), column(Ks), column(Rs)):
            a, b = min(((d, T // d) for d in range(1, T + 1) if T % d == 0),
                       key=lambda p: abs(math.log(p[1] / p[0]) - math.log(r)))
            out.add((128 * a, K, 128 * b))
            if len(out) == n: break
    return sorted(out)


def pairs_set(n):
    """The reduce pass's own cost at split 1: M x N log-spaced from 2^14 to 2^25 elements (in and past
    the 4 MB GPU L2), N / M cycling 1, 4, 1/4, K 1024; a shape taken already moves N up by 128."""
    out = []
    for i in range(n):
        mn = 2 ** (14 + i * 11 / (n - 1)); r = (1, 4, 1 / 4)[i % 3]
        M, N = r128(math.sqrt(mn / r)), r128(math.sqrt(mn * r))
        while (M, 1024, N) in out: N += 128
        out.append((M, 1024, N))
    return sorted(out)


def wave_set(path, seed, taken):
    """Each tile's workgroup count set to 0.75 .. 4 waves of the busiest core (8 cores, cap per core), at
    K 512, 2048, 8192: whole and part-filled last waves for every tile. M and N land off the tile grid."""
    rng = random.Random(seed); out = []; taken = set(taken)
    for t, wg, tn, tm, sgn, _ in TILES[path]:
        wm, wn = wg // 16 // sgn * tm, sgn * 16 * tn; slots = 8 * (128 // (wg // 16))
        for waves in (0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4):
            for K in (512, 2048, 8192):
                want = round(waves * slots)
                for _ in range(200):   # factor the workgroup count into M tiles x N tiles
                    tmc = rng.randint(1, max(1, int(math.sqrt(want)) * 2))
                    if want % tmc: continue
                    tnc = want // tmc
                    M = tmc * wm - rng.randrange(0, wm // 2, 16) if tmc * wm > 32 else tmc * wm
                    N = tnc * wn - rng.randrange(0, wn // 2, 16)
                    s = (M, K, N)
                    if M >= 32 and N >= 64 and s not in taken and M * N * 16 * 4 <= MAX_PARTIAL:
                        out.append(s); taken.add(s); break
    return sorted(out)


def focus_set(n, seed, centre):
    """Shapes around one real shape (M -13 .. +15 %, K and N +-25 %), to find what the model misses
    there; the centre itself is left out. M any integer, K a multiple of 128, N of 16."""
    M0, K0, N0 = centre; rng = random.Random(seed); out = set()
    while len(out) < n:
        s = (rng.randint(int(M0 * 0.868), int(M0 * 1.146)), rng.randrange(int(K0 * .75) // 128 * 128, int(K0 * 1.25) + 1, 128),
             rng.randrange(int(N0 * .75) // 16 * 16, int(N0 * 1.25) + 1, 16))
        if s != centre: out.add(s)
    return sorted(out)


def first_new(shapes, taken, n):
    return [s for s in shapes if s not in taken][:n]


# Real shapes (M, K, N, epilogue): pi0.5 (vision 768, language 922, action 50) and gemma3
# (vision 4096, text 268). epilogue = bias / residual / activation / scale, so the reduce pass
# runs at split 1 too.
REAL = [(768, 1152, 1152, 1), (768, 1152, 4304, 1), (768, 4304, 1152, 1), (768, 1152, 2048, 1),
        (922, 2048, 2048, 0), (922, 2048, 256, 0), (922, 2048, 32768, 0), (922, 16384, 2048, 1),
        (50, 32, 1024, 1), (50, 1024, 2048, 0), (50, 1024, 256, 0), (50, 2048, 1024, 0), (50, 1024, 8192, 0),
        (50, 4096, 1024, 0), (50, 1024, 32, 1),
        (4096, 1152, 1152, 1), (4096, 1152, 4304, 1), (4096, 4304, 1152, 1), (256, 1152, 2560, 1),
        (268, 2560, 2048, 0), (268, 2560, 1024, 0), (268, 2048, 2560, 1), (268, 2560, 20480, 0), (268, 10240, 2560, 1)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--work', default=os.path.join(ROOT, 'build', 'calib'))
    ap.add_argument('--path', default='f16', choices=list(PATHS))
    ap.add_argument('--step', type=int, default=16)   # grid / val / val2 / full M, N rounding: every tile's edge padding varies
    a = ap.parse_args(); os.makedirs(a.work, exist_ok=True)
    sets = {'grid': grid_like(100, 1, a.step), 'pairs': pairs_set(30), 'split': split_set(300, 2), 'val': grid_like(25, 3, a.step),
            'test': [s[:3] for s in REAL], 'test_epilogue': [s[3] for s in REAL]}
    sets['gridsplit'] = sets['grid']
    taken = {s for k in ('grid', 'split', 'val', 'test', 'pairs') for s in map(tuple, sets[k])}
    sets['val2'] = first_new(grid_like(130, 4, a.step), taken, 100); taken |= set(sets['val2'])
    sets['full'] = first_new(grid_like(200, 5, a.step), taken, 150); taken |= set(sets['full'])
    sets['wave'] = wave_set(a.path, 6, taken)
    sets['focus'] = focus_set(100, 7, (922, 2048, 2048))
    json.dump(sets, open(os.path.join(a.work, 'shapes.json'), 'w'))
    for k in SETS: print(f'{k:9s} {len(sets[k]):3d} shapes, {sum(len(c) for _, c in plan(k, sets[k], a.path)):5d} configs')


if __name__ == '__main__':
    main()
