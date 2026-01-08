# gpe_python/scripts/plot_sweeps.py

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# Load .npz results

def load_all_npz_results(data_dir):
    paths = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not paths:
        raise FileNotFoundError(f"No .npz files found in: {data_dir}")

    results = []
    for p in paths:
        d = np.load(p, allow_pickle=False)

        # required arrays
        r = d["r"]
        phi = d["phi"]

        # scalars saved in your sweep
        method = str(d["method"])
        g = float(d["g"])
        R = float(d["R"])
        dt_value = float(d["dt_value"])
        dt_input = str(d["dt_input"])

        # optional (not necessary)
        energy = float(d["energy"]) if "energy" in d else None
        steps = int(d["steps"]) if "steps" in d else None

        results.append({
            "path": p,
            "method": method,
            "g": g,
            "R": R,
            "dt": dt_value,
            "dt_input": dt_input,
            "energy": energy,
            "steps": steps,
            "r": r,
            "phi": phi,
        })
    return results



# Density builder (your exact style)

def build_density_arrays(r, phi, k0=3):

    rr = r
    dens = (np.abs(phi) ** 2) / (rr ** 2)

    # --- regularize near r = 0 : |psi|^2 should be ~constant ---
    if dens.size > k0:
        plateau = dens[1:k0+1].mean()
        dens[0:k0+1] = plateau

    dens0 = dens[0]

    r_plot = np.empty(rr.size + 1, dtype=float)
    dens_plot = np.empty(dens.size + 1, dtype=float)

    r_plot[0] = 0.0
    r_plot[1:] = rr
    dens_plot[0] = dens0
    dens_plot[1:] = dens

    return r_plot, dens_plot


def fmt_float(x, nd=3):
    # short filename-friendly formatting
    return f"{x:.{nd}g}"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)



# 1) Compare methods for each (g,R,dt)

def plot_compare_methods(all_results, outdir, xlim=(0.05, 3.0), ylim=None):

    ensure_dir(outdir)

    # group by (g,R,dt)
    groups = {}
    for res in all_results:
        key = (res["g"], res["R"], res["dt_input"])
        groups.setdefault(key, []).append(res)

    for (g, R, dt_input), subset in groups.items():
        # sort by method for stable legend order
        subset.sort(key=lambda d: d["method"])

        fig, ax = plt.subplots(figsize=(4, 6))

        ymax = 0.0
        xmax = 0.0
        unique = {}
        for res in subset:
            unique[res["method"]] = res
        subset = [unique[m] for m in sorted(unique)]
        for res in subset:
            
            r_plot, dens_plot = build_density_arrays(res["r"], res["phi"])

            label = f"{res['method']}"
            (line,) = ax.plot(r_plot, dens_plot, linewidth=2, label=label)

            baseline = dens_plot[1:].min() - 0.01
            ax.fill_between(
                r_plot[1:],
                dens_plot[1:],
                baseline,
                alpha=0.15,
                color=line.get_color(),
            )

            ymax = max(ymax, float(dens_plot.max()))
            xmax = max(xmax, float(r_plot.max()))

        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.tick_params(axis="both", which="major", labelsize=14)

        ax.set_xlabel("r", fontsize=18)
        ax.set_ylabel(r"$|\psi(r)|^2$", fontsize=18)

        ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(0, 1.05 * ymax)

        ax.grid(True, linestyle="-", alpha=0.25)
        ax.legend(
            frameon=True, framealpha=0.85,
            facecolor="white", edgecolor="0.8",
            fancybox=True, borderpad=0.5,
        )

        plt.tight_layout()
        dt_tag = dt_input.replace(" ", "").replace("*", "_").replace("/", "div").replace(".", "p")
        tag = f"compare_methods__g{fmt_float(g)}__R{fmt_float(R)}__dt{dt_tag}"
        outpdf = os.path.join(outdir, f"{tag}.pdf")
        outpng = os.path.join(outdir, f"{tag}.png")
        plt.savefig(outpdf, dpi=200, bbox_inches="tight")
        plt.savefig(outpng, dpi=200, bbox_inches="tight")
        plt.close()



# 2) Fixed (R,dt): vary g, separately for each method  (YOU ASKED THIS)

def plot_fixed_R_dt_vary_g_per_method(all_results, outdir, xlim=(0.05, 3.0), ylim=None):

    ensure_dir(outdir)

    groups = {}
    for res in all_results:
        key = (res["method"], res["R"], res["dt_input"])
        groups.setdefault(key, []).append(res)

    for (method, R, dt_input), subset in groups.items():
        # sort by g for stable legend order
        subset.sort(key=lambda d: d["g"])

        fig, ax = plt.subplots(figsize=(4, 6))

        ymax = 0.0
        xmax = 0.0

        for res in subset:
            r_plot, dens_plot = build_density_arrays(res["r"], res["phi"])

            label = f"g={res['g']:.3g}"
            (line,) = ax.plot(r_plot, dens_plot, linewidth=2, label=label)

            baseline = dens_plot[1:].min() - 0.01
            ax.fill_between(
                r_plot[1:],
                dens_plot[1:],
                baseline,
                alpha=0.15,
                color=line.get_color(),
            )

            ymax = max(ymax, float(dens_plot.max()))
            xmax = max(xmax, float(r_plot.max()))

        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.tick_params(axis="both", which="major", labelsize=14)

        ax.set_xlabel("r", fontsize=18)
        ax.set_ylabel(r"$|\psi(r)|^2$", fontsize=18)

        ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(0, 1.05 * ymax)

        ax.grid(True, linestyle="-", alpha=0.25)
        ax.legend(
            frameon=True, framealpha=0.85,
            facecolor="white", edgecolor="0.8",
            fancybox=True, borderpad=0.5,
        )

        plt.tight_layout()
        dt_tag = dt_input.replace(" ", "").replace("*", "_").replace("/", "div").replace(".", "p")
        tag = f"vary_g__method{method}__R{fmt_float(R)}__dt{dt_tag}"
        outpdf = os.path.join(outdir, f"{tag}.pdf")
        outpng = os.path.join(outdir, f"{tag}.png")
        plt.savefig(outpdf, dpi=200, bbox_inches="tight")
        plt.savefig(outpng, dpi=200, bbox_inches="tight")
        plt.close()



# 3) Vary R, fixed (g,dt), separately for each method

def plot_fixed_g_dt_vary_R_per_method(all_results, outdir, xlim=(0.05, 3.0), ylim=None):

    ensure_dir(outdir)

    # group by (method, g, dt_input)  <-- IMPORTANT
    groups = {}
    for res in all_results:
        key = (res["method"], res["g"], res["dt_input"])
        groups.setdefault(key, []).append(res)

    for (method, g, dt_input), subset in groups.items():
        subset.sort(key=lambda d: d["R"])

        fig, ax = plt.subplots(figsize=(4, 6))
        ymax = 0.0

        for res in subset:
            r_plot, dens_plot = build_density_arrays(res["r"], res["phi"])
            label = f"R={res['R']:.3g}"
            (line,) = ax.plot(r_plot, dens_plot, linewidth=2, label=label)

            baseline = dens_plot[1:].min() - 0.01
            ax.fill_between(
                r_plot[1:], dens_plot[1:], baseline,
                alpha=0.15, color=line.get_color()
            )
            ymax = max(ymax, float(dens_plot.max()))

        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.tick_params(axis="both", which="major", labelsize=14)

        ax.set_xlabel("r", fontsize=18)
        ax.set_ylabel(r"$|\psi(r)|^2$", fontsize=18)

        ax.set_xlim(*xlim)
        ax.set_ylim(*(ylim if ylim is not None else (0, 1.05 * ymax)))

        ax.grid(True, linestyle="-", alpha=0.25)
        ax.legend(
            frameon=True, framealpha=0.85,
            facecolor="white", edgecolor="0.8",
            fancybox=True, borderpad=0.5
        )

        plt.tight_layout()

        # filename-safe dt tag
        dt_tag = dt_input.replace(" ", "").replace("*", "_").replace("/", "div").replace(".", "p")
        tag = f"vary_R__method{method}__g{fmt_float(g)}__dt{dt_tag}"

        plt.savefig(os.path.join(outdir, f"{tag}.pdf"), dpi=200, bbox_inches="tight")
        plt.savefig(os.path.join(outdir, f"{tag}.png"), dpi=200, bbox_inches="tight")
        plt.close()


def save_energies_csv(all_results, outpath):

    ensure_dir(os.path.dirname(outpath))

    with open(outpath, "w") as f:
        f.write("method,g,R,dt_input,dt_value,energy,steps,path\n")
        for res in all_results:
            f.write(
                f"{res['method']},"
                f"{res['g']},"
                f"{res['R']},"
                f"{res['dt_input']},"
                f"{res['dt']},"
                f"{res['energy']},"
                f"{res['steps']},"
                f"{res['path']}\n"
            )

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    DATA_DIR = "gpe_runs"          
    PLOT_DIR = "gpe_plots"         

    all_results = load_all_npz_results(DATA_DIR)

    plot_compare_methods(all_results, outdir=os.path.join(PLOT_DIR, "compare_methods"))

    plot_fixed_R_dt_vary_g_per_method(all_results, outdir=os.path.join(PLOT_DIR, "vary_g_fixed_R_dt"))

    plot_fixed_g_dt_vary_R_per_method(all_results, outdir=os.path.join(PLOT_DIR, "vary_R_fixed_g_dt"))

    save_energies_csv(all_results,outpath=os.path.join(PLOT_DIR, "energies", "energy_summary.csv")
)

