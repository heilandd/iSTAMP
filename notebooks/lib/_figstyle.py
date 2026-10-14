"""
Nature Communications-style matplotlib configuration for the MsClaude revision.

- white background, minimal spines, Helvetica-like sans-serif
- colorblind-safe Okabe-Ito palette
- vectorized PDF output (editable in Illustrator: fonts kept as text, no rasterization)
- consistent panel labels and sizing

Import and call apply_style() before plotting. Use save_fig() to emit PDF + PNG.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Okabe-Ito colorblind-safe palette
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
             "#56B4E9", "#F0E442", "#000000", "#999999"]

# subtype colors (consistent across all figures)
SUBTYPE_COLORS = {
    "GBM_MES_TYP": "#D55E00",   # vermillion
    "GBM_MES_ATYP": "#CC79A7",  # reddish purple
    "GBM_RTK1": "#0072B2",      # blue
    "GBM_RTK2": "#009E73",      # green
    "GBM_PNC": "#E69F00",       # orange
}
SUBTYPE_SHORT = {"GBM_MES_TYP": "MES", "GBM_MES_ATYP": "MES-A",
                 "GBM_RTK1": "RTK1", "GBM_RTK2": "RTK2", "GBM_PNC": "PNC"}
CENTER_COLORS = {"UKF": "#0072B2", "UKH": "#E69F00", "MUV": "#009E73", "UKK": "#D55E00"}


def apply_style():
    # prefer Helvetica/Arial if present, else DejaVu Sans (metrically similar sans)
    avail = {f.name for f in fm.fontManager.ttflist}
    family = next((f for f in ["Helvetica", "Arial", "Helvetica Neue", "Nimbus Sans"]
                   if f in avail), "DejaVu Sans")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": [family, "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.2,
        "legend.frameon": False,
        "pdf.fonttype": 42,   # TrueType -> editable text in Illustrator
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def panel_label(ax, label, x=-0.18, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="top", ha="right")


# Nature single column ~ 89 mm, double column ~ 183 mm
MM = 1 / 25.4
def figsize_mm(w_mm, h_mm):
    return (w_mm * MM, h_mm * MM)


def save_fig(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    pdf = os.path.join(out_dir, f"{name}.pdf")
    png = os.path.join(out_dir, f"{name}.png")
    fig.savefig(pdf, bbox_inches="tight", transparent=False)
    fig.savefig(png, bbox_inches="tight", dpi=300, transparent=False)
    plt.close(fig)
    return pdf, png
