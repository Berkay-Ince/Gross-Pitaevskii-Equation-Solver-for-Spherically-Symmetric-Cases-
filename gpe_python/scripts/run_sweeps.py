import os
import json
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # gpe_python/
sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
with open(SCRIPT_DIR / "input.json", "r") as f:
    cfg = json.load(f)

import gpe.runners
import gpe.common

methods = cfg["methods"]
g_list = cfg["g_list"]
R_list = cfg["R_list"]
dt_list = cfg["dt_list"]

num = cfg["numerics"]
N = num["N"]
tol = num["tol"]
max_iter = num["max_iter"]
renorm_every = num["renorm_every"]
report_every = num["report_every"]
E_aim = num["E_aim"]

out_dir = cfg["output"]["out_dir"]

def sweep_and_save(
    methods,
    g_list,
    R_list,
    dt_list,
    N,
    tol,
    max_iter,
    renorm_every,
    report_every,
    E_aim,
    out_dir="data",
):
    os.makedirs(out_dir, exist_ok=True)

    for method in methods:
        for g in g_list:
            for R in R_list:

                # grid only needed to get dr for dt expressions
                r_tmp, dr = gpe.common.full_grid(N,R)

                # evaluate dt list using YOUR function
                dt_vals = gpe.common.evaluate_dt_list(dt_list, dr)

                for dt, dt_val in zip(dt_list, dt_vals):

                    res = gpe.runners.run_method(
                        method=method,
                        g=g,
                        dt=dt_val,
                        N=N,
                        R=R,
                        tol=tol,
                        max_iter=max_iter,
                        renorm_every=renorm_every,
                        report_every=report_every,
                        E_aim=E_aim,
                    )

                    # ---- filename (simple & readable) ----
                    fname = (
                        f"{method}"
                        f"_g{str(g).replace('.', '_')}"
                        f"_R{str(R).replace('.', '_')}"
                        f"_{str(dt).replace('*', '_')}"
                        ".npz"
                    )

                    path = os.path.join(out_dir, fname)

                    # ---- save everything ----
                    np.savez(
                        path,
                        r=res["r"],
                        phi=res["phi"],
                        energy=res["energy"],
                        steps=res["steps"],
                        wall_time_s=res["wall_time_s"],
                        method=method,
                        g=g,
                        R=R,
                        dt_input=str(dt),
                        dt_value=dt_val,
                        N=N,
                    )

                    print(f"Saved: {path}")

sweep_and_save(
    methods,
    g_list,
    R_list,
    dt_list,
    N,
    tol,
    max_iter,
    renorm_every,
    report_every,
    E_aim,
    out_dir,
)