"""Parallel evolution-strategy trainer for the residual MLP concession policy.

Serious neuroevolution attempt:
  * Residual MLP (67 weights); seed = ZEROS = the heuristic Boulware curve exactly,
    so training starts at the proven bar and is non-regressing by construction.
  * (mu/lambda) ES with weighted recombination of the top-mu and step-size decay.
  * DOMAIN RANDOMISATION: a fresh generated-scenario seed each generation, so the
    policy must generalise rather than overfit one scenario set / the eval noise.
  * Batched parallel workers (one process evaluates many candidates) to amortise
    the negmas import across all cores.
  * Periodic re-check of the running best on a fixed validation set; final winner
    re-validated out-of-sample by validate_policy (separate step).

    python eval/train_es.py [gens=80] [lambda=28] [mu=8] [n_gen=10] [workers=14]
"""
from __future__ import annotations
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
WORKER = str(ROOT / "eval" / "eval_policy_batch.py")
TMP = ROOT / "report" / "_es"
TMP.mkdir(exist_ok=True)
DIM = 67
import os
TRAIN_MODE = os.environ.get("TRAIN_MODE", "realistic")
TRAIN_NSTEPS = int(os.environ.get("TRAIN_NSTEPS", "100"))


def eval_pop(cands, n_gen, seed_base, workers):
    """cands: list of weight-lists (or None=heuristic). Returns list of [obj,...]."""
    chunks = [[] for _ in range(workers)]
    idxmap = [[] for _ in range(workers)]
    for k, c in enumerate(cands):
        w = k % workers
        chunks[w].append(c if c is None else [float(x) for x in c])
        idxmap[w].append(k)
    procs = []
    for w in range(workers):
        if not chunks[w]:
            continue
        spec = TMP / f"b{w}.json"
        spec.write_text(json.dumps({"n_gen": n_gen, "seed_base": seed_base,
                                    "cands": chunks[w], "mode": TRAIN_MODE,
                                    "nsteps": TRAIN_NSTEPS}))
        procs.append((w, subprocess.Popen([PY, WORKER, str(spec)],
                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)))
    results = [None] * len(cands)
    for w, pr in procs:
        out, _ = pr.communicate()
        try:
            res = json.loads(out.decode().strip().splitlines()[-1])
            for local, k in enumerate(idxmap[w]):
                results[k] = res[local]
        except Exception:
            pass
    return results


def main():
    gens = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    lam = int(sys.argv[2]) if len(sys.argv) > 2 else 28
    mu = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    n_gen = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 14
    np.random.seed(7)
    t0 = time.time()
    VAL_SEED = 90000  # fixed validation scenarios (separate from training)

    # Baseline + seed on the validation set.
    base, seed0 = eval_pop([None, [0.0] * DIM], n_gen, VAL_SEED, 2)
    print(f"HEURISTIC(val): obj={base[0]:.3f} mean={base[1]:.3f} q1={base[2]:.3f} deal={base[3]:.2f}", flush=True)
    print(f"SEED zeros(val): obj={seed0[0]:.3f} mean={seed0[1]:.3f} deal={seed0[3]:.2f}", flush=True)

    mean = np.zeros(DIM)            # distribution mean (current best policy)
    sigma = 0.6
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()       # recombination weights
    best_val = eval_pop([mean.tolist()], n_gen, VAL_SEED, 1)[0]
    best_w = mean.copy()

    for g in range(gens):
        seed_base = 10000 + g * 101          # domain randomisation per generation
        noise = np.random.randn(lam, DIM)
        cands = [(mean + sigma * noise[i]).tolist() for i in range(lam)]
        res = eval_pop(cands, n_gen, seed_base, workers)
        objs = np.array([r[0] if r else -9.9 for r in res])
        order = np.argsort(-objs)[:mu]
        mean = mean + sigma * (weights @ noise[order])   # weighted recombination
        sigma *= 0.985
        # Validate the running mean on the FIXED held-out set periodically.
        if g % 5 == 0 or g == gens - 1:
            val = eval_pop([mean.tolist()], n_gen, VAL_SEED, 1)[0]
            if val and val[0] > best_val[0]:
                best_val, best_w = val, mean.copy()
            print(f"gen {g}: trainBest={objs[order[0]]:.3f} | val mean={val[1]:.3f} "
                  f"deal={val[3]:.2f} | BESTval mean={best_val[1]:.3f} (+{best_val[1]-base[1]:.3f}) "
                  f"sigma={sigma:.3f} [{time.time()-t0:.0f}s]", flush=True)
            out = {"weights": [float(x) for x in best_w], "val_mean": best_val[1],
                   "val_obj": best_val[0], "baseline_mean": base[1], "gen": g,
                   "nfeat": 9, "nhid": 6}
            (ROOT / "advisers" / "policy_weights.json").write_text(json.dumps(out, indent=2))
        else:
            print(f"gen {g}: trainBest={objs[order[0]]:.3f} sigma={sigma:.3f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    print(f"\n=== RESULT === heuristic val mean {base[1]:.3f} | learned val mean {best_val[1]:.3f} "
          f"(delta {best_val[1]-base[1]:+.3f})", flush=True)


if __name__ == "__main__":
    main()
