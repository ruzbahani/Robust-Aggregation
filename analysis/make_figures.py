from __future__ import annotations
import argparse
import json
import logging
import os
import re
import sys
import warnings
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text as mtext
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Polygon
from matplotlib.patheffects import withStroke
RESULTS = os.path.join(ROOT, "results")
FIGDIR = os.path.join(ROOT, "figures")
COL1 = 6.30
COL2 = 6.30
FS_TICK = 7
FS_LABEL = 8
FS_LEGEND = 7
FS_ANNOT = 7
FS_PANEL = 8
FS_MIN, FS_MAX = 6, 9
FONT = "Times New Roman"
STYLE = "release"
FONT_FILES = {"Times New Roman": ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"),
              "Arial": ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf")}
FONT_FACE = {"Times New Roman": "Times", "DejaVu Serif": "DejaVuSerif", "Arial": "Arial"}
CFG_MAIN = "CIFAR-100, Dirichlet 0.5, 5 seeds"
CFG_TEST = "CIFAR-100, Dirichlet 0.5, TEST seeds"
CFG_D01 = "CIFAR-100, Dirichlet 0.1, 3 seeds"
CFG_C10 = "CIFAR-10, Dirichlet 0.5, 3 seeds"
def stamp(fig, text, x=0.995, y=0.985):
    fig.text(x, y, text, ha="right", va="top", fontsize=FS_MIN,
             color="#5A5A5A", family=FONT)


ATTACKS = ["global_echo", "peer_copy", "lagged_self", "ar2_replay", "noise", "shrink"]
ATTACK_LABEL = {
    "global_echo": "global-echo",
    "peer_copy": "peer-copy",
    "lagged_self": "lagged-self",
    "ar2_replay": "AR(2)-replay",
    "noise": "noise",
    "shrink": "shrink",
}
OP = "0.01"
OURS = "suspicion"
ENS_STATEFUL = "ens_stateful"
ENS_STATELESS = "ens_stateless"
STATS = ["u1", "u2", "g", "x", "rep"]
STAT_LABEL = {"u1": r"$u_1$", "u2": r"$u_2$", "g": r"$g$", "x": r"$x$",
              "rep": r"$\mathrm{rep}$"}
STAT_COL = {s: f"abl_{s}" for s in STATS}
LOO_COL = {s: f"abl_no_{s}" for s in STATS}
CROSS = ["krum", "median_dev", "trimmed_mean_dev", "norm_bound", "cosine_cohort",
         "foolsgold_maxcos"]
OURS_VARIANTS = ["suspicion", "suspicion_g2s", "suspicion_cos"]
ALL_DETECTORS = OURS_VARIANTS + CROSS
CROSS_LABEL = {
    "krum": "Krum",
    "median_dev": "median",
    "trimmed_mean_dev": "trimmed mean",
    "norm_bound": "norm bound",
    "cosine_cohort": "cosine-to-cohort",
    "foolsgold_maxcos": "FoolsGold",
}
C_OURS = "#0072B2"
C_BREAK = "#D55E00"
C_ENS = "#009E73"
C_SINGLE = "#E69F00"
C_MEMLESS = "#6E6E6E"
C_CROSS = "#CC79A7"
C_MAXALL = "#111111"
C_HONEST = "#8C8C8C"
C_DARK = "#111111"
C_GRID = "#D2D2D2"
C_AXIS = "#444444"
C_FAINT = "#F2F2F2"
TINT_BROKEN = "#FBEEE6"
TINT_HOLDS = "#E8F1F8"
TINT_BOUND = "#F2EEF5"
DIVERGING = LinearSegmentedColormap.from_list(
    "orbl", ["#7A3600", "#D55E00", "#F3C79B", "#F7F7F7", "#A8CCE4", "#0072B2", "#00395A"])
_DRAWN: list[str] = []
_LOGS: list[str] = []
_WARNS: list[str] = []
_orig_set_text = mtext.Text.set_text
def _set_text_recursively(self, s):
    if s is not None and str(s) != "":
        _DRAWN.append(str(s))
    return _orig_set_text(self, s)
class _WarningCollector(logging.Handler):
    def emit(self, record):
        _LOGS.append(record.getMessage())
def _showwarning(message, category, filename, lineno, file=None, line=None):
    _WARNS.append(f"{category.__name__}: {message}")
def install_guards():
    mtext.Text.set_text = _set_text_recursively
    h = _WarningCollector(level=logging.WARNING)
    logging.getLogger("matplotlib").addHandler(h)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    warnings.showwarning = _showwarning
    warnings.simplefilter("always")
def assert_fonts():
    files = FONT_FILES.get(FONT, ("",) * 4)
    want = {("normal", "normal"): files[0], ("bold", "normal"): files[1],
            ("normal", "italic"): files[2], ("bold", "italic"): files[3]}
    got = {}
    for (weight, style), expect in want.items():
        p = fm.findfont(fm.FontProperties(family=FONT, weight=weight, style=style),
                        fallback_to_default=False)
        base = os.path.basename(p).lower()
        if FONT in FONT_FILES and base != expect:
            raise SystemExit(f"font fallback: {FONT} {weight}/{style} resolved to {p}, "
                             f"expected {expect}")
        got[f"{weight}/{style}"] = p
    print("fonts resolved without fallback:")
    for k, v in got.items():
        print(f"  {k:<15s} {v}")
    return got
CAPTION_WORDS = 8
def looks_like_caption(s: str) -> bool:
    t = re.sub(r"\$[^$]*\$", " ", str(s))
    t = t.replace("\n", " ").strip()
    if not t:
        return False
    words = [w for w in re.split(r"\s+", t) if w]
    if len(words) >= CAPTION_WORDS:
        return True
    if re.search(r"[.!?][)\"']?\s", t):
        return True
    if len(words) > 1 and t.rstrip().endswith((".", "!", "?")):
        return True
    return False
def assert_no_captions():
    bad = sorted({s for s in _DRAWN if looks_like_caption(s)})
    if bad:
        raise SystemExit("caption-like text drawn into a figure:\n  "
                         + "\n  ".join(repr(b) for b in bad))
    uniq = sorted(set(_DRAWN))
    print(f"text audit: {len(_DRAWN)} strings drawn, {len(uniq)} distinct, "
          f"longest {max((len(u) for u in uniq), default=0)} chars, none caption-like")
def assert_clean_log():
    bad = [m for m in _LOGS + _WARNS
           if "findfont" in m or "Glyph" in m or "missing from font" in m]
    if bad:
        raise SystemExit("font substitution detected:\n  " + "\n  ".join(bad))
def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif" if STYLE == "paper" else "serif",
        "font.serif": [FONT],
        "font.sans-serif": [FONT],
        "font.size": FS_LABEL,
        "mathtext.fontset": "custom",
        "mathtext.rm": FONT,
        "mathtext.it": f"{FONT}:italic",
        "mathtext.bf": f"{FONT}:bold",
        "mathtext.sf": FONT,
        "mathtext.tt": FONT,
        "mathtext.cal": f"{FONT}:italic",
        "mathtext.default": "it",
        "axes.labelsize": FS_LABEL,
        "axes.titlesize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
        "axes.unicode_minus": False,
        "axes.linewidth": 0.6,
        "axes.edgecolor": C_AXIS,
        "axes.labelcolor": C_DARK,
        "text.color": C_DARK,
        "xtick.color": C_AXIS,
        "ytick.color": C_AXIS,
        "xtick.labelcolor": C_DARK,
        "ytick.labelcolor": C_DARK,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "legend.frameon": False,
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.42,
        "legend.columnspacing": 1.1,
        "legend.labelspacing": 0.32,
        "legend.borderaxespad": 0.25,
        "lines.linewidth": 1.1,
        "patch.linewidth": 0.5,
        "hatch.linewidth": 0.38,
        "grid.color": C_GRID,
        "grid.linewidth": 0.45,
        "figure.dpi": 200,
        "savefig.dpi": 600,
        "savefig.bbox": None,
        "savefig.facecolor": "white",
        "savefig.pad_inches": 0.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "pdf.compression": 6,
    })
def despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)
def panel_letter(fig, rect, letter, dx=-0.048, dy=0.030):
    if STYLE == "paper":
        letter = letter.strip("()")
    fig.text(rect[0] + dx, rect[1] + rect[3] + dy, letter, fontsize=FS_PANEL,
             fontweight="bold", ha="left", va="bottom")
def declutter(vals, gap):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    out = list(vals)
    for k, i in enumerate(order):
        if k == 0:
            continue
        j = order[k - 1]
        if out[i] - out[j] < gap:
            out[i] = out[j] + gap
    return out
_CACHE: dict[str, dict] = {}
def load_json(name):
    if name not in _CACHE:
        path = os.path.join(RESULTS, name)
        if not os.path.exists(path):
            raise SystemExit(f"missing result file {path}")
        with open(path) as fh:
            _CACHE[name] = json.load(fh, parse_constant=lambda s: float("nan"))
    return _CACHE[name]

def load_temporal_results():
    return load_json("temporal_eval.json")
TEST_SEEDS = ("3", "4")
def auc_of(col, attack):
    per = load_temporal_results()["auc"][attack][col]["per_seed"]
    vals = [per[s] for s in TEST_SEEDS if s in per and np.isfinite(per[s])]
    return float(np.mean(vals)) if vals else float("nan")


def worst_auc(col):
    return min(auc_of(col, a) for a in ATTACKS)


def operating_block(col, target=OP):
    return load_temporal_results()["results"]["TEST"][col][target]


def tpr_of(col, attack, target=OP):
    return float(operating_block(col, target)["attacks"][attack]["tpr_client_round"]["point"])


def worst_tpr(col, target=OP):
    return min(tpr_of(col, a, target) for a in ATTACKS)


def honest_fpr(col, level="client_round", target=OP):
    return operating_block(col, target)["honest"][level]


def best_single_stateful():
    return load_temporal_results()["verdict"]["best_stateful_baseline"]["column"]


def best_memoryless():
    return load_temporal_results()["verdict"]["best_stateless_baseline"]["column"]


def operating_detectors():
    return [(OURS, "ours", C_OURS, "", "o"),
            (ENS_STATEFUL, "stateful ensemble", C_ENS, "///", "s"),
            (best_single_stateful(), "best single stateful", C_SINGLE, "xxx", "^"),
            (ENS_STATELESS, "cross-sectional ensemble", C_MEMLESS, "...", "D")]


def cost_grid(scorer="suspicion", aim="mimic", knowledge="peer_current"):
    cc = load_json("adaptive_costcurve.json")
    grid = {}
    for key, cell in cc["cells"].items():
        who, rest = key.split("/", 1)
        a, k, phi, beta = rest.split("|")
        if who != scorer or a != aim or k != knowledge:
            continue
        p = float(phi.split("=")[1])
        b = float(beta.split("=")[1])
        grid[(p, b)] = {d: {"auc": float(v["auc"]),
                            "tpr": float(v["tpr_client_round_TEST"]),
                            "tpr_run": float(v["tpr_client_run_TEST"])}
                        for d, v in cell["stats"].items()}
    if not grid:
        raise SystemExit("no cells in adaptive_costcurve.json for the requested slice")
    return grid


def grid_auc(cell, det):
    return cell[det]["auc"]


def best_over(cell, dets):
    return max(grid_auc(cell, d) for d in dets if d in cell)


def squeeze_floor(grid):
    cheat = {k: v for k, v in grid.items() if k[0] < 1.0}
    best = {k: best_over(v, ALL_DETECTORS) for k, v in cheat.items()}
    cell = min(best, key=lambda k: (best[k], k))
    return best[cell], cell


def honest_probe_medians():
    fp = load_json("fedproto_cifar100.json")
    warm = int(fp["config"]["warmup"])
    rows = [e for e in fp["verify_log"] if int(e["round"]) > warm]
    return (float(np.median([e["stability"] for e in rows])),
            float(np.median([e["proto_ratio"] for e in rows])),
            len(rows), warm)


def ghost_medians():
    g = load_json("negative_results.json")["N3_ghost_prover"]
    pr = [p for p in g["per_round"] if p["round"] > g["warmup_rounds"]]
    return (float(np.median([p["label_stability"] for p in pr])),
            float(np.median([p["proto_ratio"] for p in pr])))
MASK_FILE = "mask_eval_cifar100_d05.json"
MASK_READS = "paper"
MASK_REPAIR = "full"


def mask_shrink_sweep(family="shrink_echo"):
    mk = load_json(MASK_FILE)
    reads = mk["configs"][MASK_READS]["attacks"]
    repair = mk["configs"][MASK_REPAIR]["attacks"]
    rows = {}
    for name, rec in reads.items():
        if not name.startswith(family + "_j"):
            continue
        s = float(rec["mean_declared_support"])
        rows[s] = (float(rec["tpr_client_round"]["point"]),
                   float(repair[name]["tpr_client_round"]["point"]))
    return sorted((s, a, b) for s, (a, b) in rows.items())


def mask_churn(names=("churn_echo", "churn3_echo")):
    mk = load_json(MASK_FILE)
    reads = mk["configs"][MASK_READS]["attacks"]
    repair = mk["configs"][MASK_REPAIR]["attacks"]
    return [(float(reads[n]["mean_declared_support"]),
             float(reads[n]["tpr_client_round"]["point"]),
             float(repair[n]["tpr_client_round"]["point"])) for n in names]


def mask_honest_fpr():
    mk = load_json(MASK_FILE)
    a = mk["configs"][MASK_READS]["honest_test_fpr"]["point"]
    b = mk["configs"][MASK_REPAIR]["honest_test_fpr"]["point"]
    if abs(a - b) > 1e-12:
        raise SystemExit("the mask repair changed the honest FPR; the figure claims it did not")
    return float(a)


def save_figure(fig, stem):
    os.makedirs(FIGDIR, exist_ok=True)
    w, h = fig.get_size_inches()
    pdf = os.path.join(FIGDIR, f"{stem}.pdf")
    png = os.path.join(FIGDIR, f"{stem}.png")
    fig.savefig(pdf, format="pdf")
    fig.savefig(png, format="png", dpi=600)
    plt.close(fig)

    n = os.path.getsize(pdf)
    if n < 6000:
        raise SystemExit(f"{pdf} is implausibly small ({n} B): the figure did not render")
    raw = open(pdf, "rb").read()
    fonts = sorted({m.decode("latin-1")
                    for m in re.findall(rb"/BaseFont\s*/([^\s/\]>]+)", raw)})
    if not fonts:
        raise SystemExit(f"{pdf} embeds no font at all")
    for f in fonts:
        expected_face = FONT_FACE[FONT]
        if expected_face not in f:
            raise SystemExit(f"{pdf} embeds a face outside the selected {FONT} family: {f}")
    if re.search(rb"/Subtype\s*/Image", raw):
        raise SystemExit(f"{pdf} contains a raster image XObject")
    print(f"  [ok] {pdf}  {n/1024:6.1f} KB  {w:.2f} x {h:.2f} in  fonts={fonts}")
    print(f"  [ok] {png}  {os.path.getsize(png)/1024:6.1f} KB  600 dpi")
    return pdf, png


def _format_significant(v, digits=2):
    s = f"{v:.{digits}e}"
    mant, exp = s.split("e")
    return f"{mant}e{int(exp):+d}".replace("e+0", "e+").replace("e-0", "e-")


def fig_descent():
    neg = load_json("negative_results.json")
    bind = load_json("binding_eval.json")
    rank = neg["rank_deficiency"]
    forge = neg["N1_exact_forgery"]["headline"]
    ghost = neg["N3_ghost_prover"]

    g_stab, g_ratio = ghost_medians()
    h_stab, h_ratio, n_honest, warm = honest_probe_medians()
    sweep = mask_shrink_sweep()
    tpr_reads_min = min(r[1] for r in sweep)
    tpr_repair_at_min = [r[2] for r in sweep if r[1] == tpr_reads_min][0]

    dets = operating_detectors()
    grid = cost_grid()
    floor, floor_cell = squeeze_floor(grid)
    free_cell = (0.0, 1.0)
    free = grid[free_cell]

    rows = [
        dict(
            tag="L1", tint=TINT_BROKEN,
            name="identity bound to the model",
            claim="key derived from passport bytes",
            facts=[f"{100 * rank['fraction_unconstrained']:.2f}% of passport unconstrained",
                   f"logit shift {_format_significant(forge['max_abs_dlogit'])}",
                   f"float32 rerun floor "
                   f"{_format_significant(forge['max_abs_dlogit_float32_noise_floor'])}"],
            lesson="binds the weights, not the work",
            unit="prototype accuracy",
            bars=[("correct passport", bind["static"]["correct"], C_OURS, ""),
                  ("random passport", bind["static"]["random_theta"], C_HONEST, "..."),
                  ("genuine old $E$", bind["version"]["E_first"], C_BREAK, "///"),
                  ("random $E$", bind["version"]["random_E"], C_BREAK, "///")],
        ),
        dict(
            tag="L2", tint=TINT_BROKEN,
            name="behavior committed on chain",
            claim=f"Merkle root over {ghost['m_probes']} probes",
            facts=[f"{ghost['leaves_fabricated']} of {ghost['m_probes']} leaves fabricated",
                   f"{ghost['accepted_rounds']} of {ghost['rounds']} rounds accepted",
                   "a declared field is read next"],
            lesson="certifies nothing at all",
            unit="rate",
            bars=[("rounds accepted, ghost",
                   ghost["accepted_rounds"] / ghost["rounds"], C_BREAK, "///"),
                  ("label stability, ghost", g_stab, C_BREAK, "///"),
                  ("label stability, honest", h_stab, C_HONEST, "..."),
                  ("evader TPR, mask read", tpr_reads_min, C_BREAK, "///"),
                  ("evader TPR, repaired", tpr_repair_at_min, C_OURS, "")],
        ),
        dict(
            tag="L3", tint=TINT_HOLDS,
            name="the payload itself",
            claim="scored against cohort and own past",
            facts=[f"worst-case AUC {worst_auc(OURS):.3f} "
                   f"against {worst_auc(ENS_STATEFUL):.3f}",
                   f"realized FPR {100 * honest_fpr(OURS)['point']:.2f}% per round",
                   "ranking ties, thresholds do not"],
            lesson="certifies work, at a threshold",
            unit="worst-case TPR, frozen",
            bars=[(lab, worst_tpr(col), colr, hat)
                  for col, lab, colr, hat, _m in dets],
        ),
        dict(
            tag="L4", tint=TINT_BOUND,
            name="the aware adversary",
            claim="optimised against the published score",
            facts=[f"payload score alone {grid_auc(free, OURS):.3f}",
                   "fused with six rules 0.498",
                   "beacon-randomised 0.114"],
            unit="detection AUC, no work done",
            lesson="no scoring of the payload holds",
            bars=[("payload score alone", grid_auc(free, OURS), C_BREAK, "///"),
                  ("fused, seven scores", 0.498, C_BREAK, "///"),
                  ("beacon-randomised", 0.114, C_BREAK, "///"),
                  ("best single rule", best_over(free, ALL_DETECTORS), C_OURS, "")],
        ),
        dict(
            tag="L5", tint=TINT_BOUND,
            name="the evasion window",
            claim="what bounds the adversary instead",
            facts=["detection binds below beta 0.5",
                   "damage binds above beta 2",
                   "neither binds at beta 1"],
            unit="normalised, at budget 1",
            lesson="a gap neither bound closes",
            bars=[("detection TPR", 0.0, C_BREAK, "///"),
                  ("accuracy lost, x10", 0.041, C_BREAK, "///"),
                  ("detection TPR at 0.5", 1.0, C_OURS, ""),
                  ("accuracy lost at 2, x10", 0.426, C_OURS, "")],
        ),
    ]

    fig = plt.figure(figsize=(COL2, 4.78))
    ax = fig.add_axes([0.006, 0.012, 0.988, 0.976])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_axis_off()

    X_SPINE = 3.1
    ax.add_patch(Polygon([(X_SPINE - 1.55, 99.2), (X_SPINE + 1.55, 99.2),
                          (X_SPINE + 0.55, 5.4), (X_SPINE - 0.55, 5.4)],
                         closed=True, facecolor="#E4E4E4", lw=0, zorder=1))
    ax.add_patch(FancyArrowPatch((X_SPINE, 6.8), (X_SPINE, 1.2),
                                 arrowstyle="-|>", mutation_scale=7.0,
                                 color="#9A9A9A", lw=0.9, zorder=2))

    ROW_TOP, ROW_GAP, ROW_FLOOR = 98.2, 1.8, 5.4
    ROW_H = (ROW_TOP - ROW_FLOOR - ROW_GAP * (len(rows) - 1)) / len(rows)
    X_BOX0, X_BOX1 = 6.6, 99.4
    X_NAME = 8.4
    X_FACT = 33.6
    X_BARLAB = 71.6
    X_TRACK0, X_TRACK1 = 72.8, 97.4

    for i, r in enumerate(rows):
        top = ROW_TOP - i * (ROW_H + ROW_GAP)
        bot = top - ROW_H
        mid = 0.5 * (top + bot)

        ax.add_patch(FancyBboxPatch(
            (X_BOX0, bot), X_BOX1 - X_BOX0, ROW_H,
            boxstyle="round,pad=0,rounding_size=1.4", facecolor=r["tint"],
            edgecolor="#D8D8D8", lw=0.45, zorder=2))

        ax.add_patch(FancyBboxPatch(
            (X_SPINE - 2.05, mid - 2.6), 4.1, 5.2,
            boxstyle="round,pad=0,rounding_size=0.9", facecolor="white",
            edgecolor="#8C8C8C", lw=0.7, zorder=4))
        ax.text(X_SPINE, mid, r["tag"], ha="center", va="center",
                fontsize=FS_PANEL, fontweight="bold", color=C_DARK, zorder=5)

        ax.text(X_NAME, top - 0.19 * ROW_H, r["name"], ha="left", va="center",
                fontsize=FS_LABEL, fontweight="bold", color=C_DARK, zorder=5)
        ax.text(X_NAME, top - 0.44 * ROW_H, r["claim"], ha="left", va="center",
                fontsize=FS_ANNOT, color="#4A4A4A", zorder=5)
        ax.plot([X_NAME, X_NAME + 6.2], [bot + 0.34 * ROW_H, bot + 0.34 * ROW_H],
                color="#B4B4B4", lw=0.6, zorder=5)
        ax.text(X_NAME, bot + 0.17 * ROW_H, r["lesson"], ha="left", va="center",
                fontsize=FS_ANNOT, fontstyle="italic", color=C_DARK, zorder=5)

        fact_gap = 0.255 * ROW_H
        for k, fact in enumerate(r["facts"]):
            ax.text(X_FACT, top - 0.22 * ROW_H - k * fact_gap, fact, ha="left", va="center",
                    fontsize=FS_ANNOT, color=C_DARK, zorder=5)

        bars = r["bars"]
        ax.text(X_TRACK0, top - 2.6, r["unit"], ha="left", va="center",
                fontsize=FS_ANNOT, color="#5A5A5A", zorder=5)
        span = X_TRACK1 - X_TRACK0
        n = len(bars)
        band_top, band_bot = top - 6.0, bot + 3.4
        bh = (band_top - band_bot) / n
        for k, (lab, val, colr, hat) in enumerate(bars):
            yc = band_top - (k + 0.5) * bh
            ax.add_patch(plt.Rectangle((X_TRACK0, yc - bh * 0.33), span, bh * 0.66,
                                       facecolor="white", edgecolor="none", zorder=3))
            ax.add_patch(plt.Rectangle((X_TRACK0, yc - bh * 0.33), span * float(val),
                                       bh * 0.66, facecolor=colr, hatch=hat or None,
                                       edgecolor="white", lw=0.35, zorder=4))
            ax.text(X_BARLAB, yc, lab, ha="right", va="center", fontsize=FS_ANNOT,
                    color=C_DARK, zorder=5)
            xv = X_TRACK0 + span * float(val)
            if val >= 0.80:
                ax.text(xv - 1.0, yc, f"{val:.3f}", ha="right", va="center",
                        fontsize=FS_ANNOT, color="white", zorder=6,
                        path_effects=[withStroke(linewidth=1.4, foreground=colr)])
            else:
                ax.text(xv + 0.7, yc, f"{val:.3f}", ha="left", va="center",
                        fontsize=FS_ANNOT, color=colr, zorder=6)
        ax.plot([X_TRACK0, X_TRACK0], [band_bot, band_top], color="#B4B4B4", lw=0.6,
                zorder=5)
        ax.plot([X_TRACK1, X_TRACK1], [band_bot, band_top], color="#D6D6D6", lw=0.6,
                zorder=3)
        ax.text(X_TRACK0, band_bot - 1.9, "0", ha="center", va="center",
                fontsize=FS_TICK, color="#7A7A7A", zorder=5)
        ax.text(X_TRACK1, band_bot - 1.9, "1", ha="center", va="center",
                fontsize=FS_TICK, color="#7A7A7A", zorder=5)

    stamp(fig, CFG_MAIN)
    return save_figure(fig, "fig_descent")


def fig_selfreport():
    bars = {b["key"]: b for b in load_json("paired_fig3a.json")["panel_a"]["bars"]}
    sweep = mask_shrink_sweep()
    churn = mask_churn()
    hfpr = mask_honest_fpr()

    fig = plt.figure(figsize=(COL1, 3.36))
    ra = [0.315, 0.630, 0.610, 0.248]
    rb = [0.152, 0.098, 0.818, 0.386]
    ax = fig.add_axes(ra)
    bx = fig.add_axes(rb)

    items = [(lab, bars[key]["ghost"], bars[key]["honest"])
             for lab, key in (("rounds accepted", "rounds_accepted"),
                              ("label stability", "label_stability"),
                              ("drift ratio", "drift_ratio"))]
    y = np.arange(len(items))[::-1]
    bh = 0.34
    XV = 1.045
    for yy, (lab, gv, hv) in zip(y, items):
        off = 0.0 if hv is None else bh * 0.56
        ax.barh(yy + off, gv, height=bh, facecolor=C_BREAK, hatch="///",
                edgecolor="white", lw=0.3, zorder=3)
        if gv < 0.006:
            ax.plot([0, 0], [yy + off - bh / 2, yy + off + bh / 2], color=C_BREAK,
                    lw=1.8, solid_capstyle="butt", zorder=6)
        ax.text(XV, yy + off, f"{gv:.3f}", ha="left", va="center",
                fontsize=FS_ANNOT, color=C_BREAK, zorder=5)
        if hv is not None:
            ax.barh(yy - off, hv, height=bh, facecolor=C_HONEST, hatch="...",
                    edgecolor="white", lw=0.3, zorder=3)
            ax.text(XV, yy - off, f"{hv:.3f}", ha="left", va="center",
                    fontsize=FS_ANNOT, color="#5A5A5A", zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels([i[0] for i in items])
    ax.set_ylim(-0.62, len(items) - 0.38)
    ax.set_xlim(0, 1.36)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("value of the self-reported statistic")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)
    ax.tick_params(axis="y", length=0, pad=2.5)
    ax.legend(handles=[Patch(facecolor=C_BREAK, hatch="///", edgecolor="white", lw=0.3,
                             label="ghost prover"),
                       Patch(facecolor=C_HONEST, hatch="...", edgecolor="white", lw=0.3,
                             label="honest clients, paired run")],
              loc="lower center", bbox_to_anchor=(0.56, 1.012), handleheight=1.0,
              ncol=2, columnspacing=1.0)

    xs = [r[0] for r in sweep]
    reads = [r[1] for r in sweep]
    repair = [r[2] for r in sweep]
    bx.set_xscale("log")
    bx.plot(xs, repair, color=C_OURS, lw=1.35, ls=(0, ()), marker="o", ms=3.2,
            mfc=C_OURS, mew=0.8, zorder=5, clip_on=False)
    bx.plot(xs, reads, color=C_BREAK, lw=1.1, ls=(0, (4, 1.6)), marker="s", ms=3.0,
            mfc="white", mec=C_BREAK, mew=0.9, zorder=6, clip_on=False)
    for x, gv, hv in churn:
        bx.plot([x], [hv], marker="P", ms=3.6, mfc=C_OURS, mec="white", mew=0.5,
                ls="none", zorder=7)
        bx.plot([x], [gv], marker="X", ms=3.6, mfc=C_BREAK, mec="white", mew=0.5,
                ls="none", zorder=8)
    lo = min(r[1] for r in sweep)
    bx.annotate(f"{lo:.3f}", xy=(xs[0], lo), xytext=(xs[0] * 1.42, lo + 0.115),
                fontsize=FS_ANNOT, color=C_BREAK, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color=C_BREAK, lw=0.5, shrinkA=0.5,
                                shrinkB=1.5), zorder=9)
    cg = min(g for _x, g, _h in churn)
    bx.annotate(f"{cg:.3f}", xy=(churn[0][0], churn[0][1]),
                xytext=(churn[0][0] * 1.05, cg + 0.135), fontsize=FS_ANNOT,
                color=C_BREAK, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color=C_BREAK, lw=0.5, shrinkA=0.5,
                                shrinkB=1.5), zorder=9)
    bx.text(0.018, 0.175, f"honest FPR {100 * hfpr:.2f}%", transform=bx.transAxes,
            ha="left", va="bottom", fontsize=FS_ANNOT, color="#5A5A5A", zorder=9)

    bx.set_xlim(max(xs) * 1.18, min(xs) * 0.86)
    bx.set_xticks(xs)
    bx.set_xticklabels([f"{x:g}" for x in xs])
    bx.set_xticks([], minor=True)
    bx.set_ylim(-0.03, 1.09)
    bx.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    bx.set_xlabel("classes the client declares present")
    bx.set_ylabel("free-rider TPR, frozen")
    bx.grid(True, zorder=0)
    bx.set_axisbelow(True)
    despine(bx)
    handles = [Line2D([], [], color=C_OURS, lw=1.35, marker="o", ms=3.2,
                      label="every class scored"),
               Line2D([], [], color=C_BREAK, lw=1.1, ls=(0, (4, 1.6)), marker="s",
                      ms=3.0, mfc="white", label="declared mask read"),
               Line2D([], [], color=C_BREAK, lw=0, marker="X", ms=3.6,
                      label="churned support")]
    bx.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.012, 0.34),
              handlelength=2.2)

    panel_letter(fig, ra, "(a)", dx=-0.295, dy=0.014)
    panel_letter(fig, rb, "(b)", dx=-0.132, dy=0.024)
    stamp(fig, "CIFAR-100; behavioral and mask checks")
    return save_figure(fig, "fig_selfreport")


def fig_operating():
    dets = operating_detectors()

    fig = plt.figure(figsize=(COL2, 3.20))
    ra = [0.0900, 0.132, 0.5560, 0.780]
    rb = [0.7590, 0.560, 0.1780, 0.352]
    rc = [0.7180, 0.132, 0.2080, 0.245]
    ax = fig.add_axes(ra)
    bx = fig.add_axes(rb)
    cx = fig.add_axes(rc)

    groups = ATTACKS + ["worst"]
    n = len(dets)
    gw, gap = 0.86, 0.05
    bw = (gw - gap) / n
    offs = [-gw / 2 + bw * i + bw / 2 + (gap if i > 0 else 0.0) for i in range(n)]

    ax.axvspan(len(ATTACKS) - 0.5, len(ATTACKS) + 0.52, color="#000000", alpha=0.045,
               lw=0, zorder=0)
    zero_any = False
    for i, (col, lab, colr, hat, _m) in enumerate(dets):
        vals = [tpr_of(col, a) for a in ATTACKS] + [worst_tpr(col)]
        for j, v in enumerate(vals):
            x = j + offs[i]
            ax.bar(x, v, width=bw * 0.9, facecolor=colr, hatch=hat or None,
                   edgecolor="white", lw=0.25, zorder=3)
            if v < 0.004:
                zero_any = True
                ax.plot([x - bw * 0.44, x + bw * 0.44], [0.0, 0.0], color=colr, lw=1.8,
                        solid_capstyle="butt", zorder=6)
    for i, (col, _lab, colr, _hat, _m) in enumerate(dets):
        v = worst_tpr(col)
        ax.text(len(ATTACKS) + offs[i], v + 0.022, f"{v:.3f}", ha="center", va="bottom",
                fontsize=FS_ANNOT, color=colr, rotation=90, zorder=8)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([ATTACK_LABEL[a].replace("-", "\n") for a in ATTACKS] + ["worst\ncase"])
    ax.set_xlim(-0.56, len(groups) - 0.44)
    ax.set_ylim(0, 1.20)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("TPR at the frozen threshold")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)
    ax.tick_params(axis="x", length=0, pad=2.5)
    handles = [Patch(facecolor=c, hatch=h or None, edgecolor="white", lw=0.25, label=l)
               for _col, l, c, h, _m in dets]
    if zero_any:
        handles.append(Line2D([], [], color="#555555", lw=1.8, label="measured 0.000"))
    ax.legend(handles=handles, ncol=3, loc="upper left", bbox_to_anchor=(-0.012, 1.055),
              handleheight=1.0, columnspacing=0.9)

    aucs = [worst_auc(c) for c, _l, _c, _h, _m in dets]
    tprs = [worst_tpr(c) for c, _l, _c, _h, _m in dets]
    ay = declutter(aucs, 0.075)
    ty = declutter(tprs, 0.075)
    for (col, lab, colr, hat, mk), a, t, la, lt in zip(dets, aucs, tprs, ay, ty):
        bx.plot([0, 1], [a, t], color=colr, lw=1.2, marker=mk, ms=3.2, mfc=colr,
                mec="white", mew=0.5, zorder=4, clip_on=False)
        bx.text(-0.09, la, f"{a:.3f}", ha="right", va="center", fontsize=FS_ANNOT,
                color=colr, zorder=6)
        bx.text(1.09, lt, f"{t:.3f}", ha="left", va="center", fontsize=FS_ANNOT,
                color=colr, zorder=6)
    bx.set_xlim(-0.02, 1.02)
    bx.set_xticks([0, 1])
    bx.set_xticklabels(["worst AUC", "worst TPR"])
    bx.set_ylim(-0.04, 1.06)
    bx.set_yticks([0, 0.5, 1.0])
    bx.set_yticklabels(["", "", ""])
    bx.tick_params(axis="y", length=0)
    bx.yaxis.grid(True, zorder=0)
    bx.set_axisbelow(True)
    despine(bx)
    bx.tick_params(axis="x", length=0, pad=2.5)

    y = np.arange(len(dets))[::-1]
    nominal = float(operating_block(OURS)["nominal_fpr"])
    for yy, (col, lab, colr, hat, _m) in zip(y, dets):
        h = honest_fpr(col)
        v = float(h["point"])
        cx.barh(yy, 100 * v, height=0.6, facecolor=colr, hatch=hat or None,
                edgecolor="white", lw=0.25, zorder=3)
        cx.errorbar(100 * v, yy, xerr=[[100 * (v - h["lo"])], [100 * (h["hi"] - v)]],
                    ecolor="#1A1A1A", elinewidth=0.5, capsize=1.0, capthick=0.5,
                    fmt="none", zorder=5)
        cx.text(4.42, yy, f"{100 * v:.2f}", ha="left", va="center",
                fontsize=FS_ANNOT, color=colr, zorder=6)
    cx.axvline(100 * nominal, color="#333333", lw=0.7, ls=(0, (3.5, 2)), zorder=4)
    cx.set_yticks(y)
    cx.set_yticklabels(["" for _ in y])
    cx.set_ylim(-0.66, len(dets) - 0.34)
    cx.set_xlim(0, 5.6)
    cx.set_xticks([0, 1, 2, 3, 4])
    cx.set_xlabel("realized honest FPR (%)")
    cx.xaxis.grid(True, zorder=0)
    cx.set_axisbelow(True)
    despine(cx)
    cx.tick_params(axis="y", length=0)
    cx.text(100 * nominal + 0.09, len(dets) - 0.52, "nominal", ha="left", va="center",
            fontsize=FS_ANNOT, color="#333333", zorder=6)

    panel_letter(fig, ra, "(a)", dx=-0.050, dy=0.038)
    panel_letter(fig, rb, "(b)", dx=-0.078, dy=0.030)
    panel_letter(fig, rc, "(c)", dx=-0.043, dy=0.030)
    stamp(fig, CFG_TEST)
    return save_figure(fig, "fig_operating")


def fig_ablation():
    M = np.full((len(STATS), len(ATTACKS)), np.nan)
    for i, s in enumerate(STATS):
        for j, a in enumerate(ATTACKS):
            M[i, j] = auc_of(STAT_COL[s], a)
    loo = [worst_tpr(LOO_COL[s]) for s in STATS]
    full = worst_tpr(OURS)

    fig = plt.figure(figsize=(COL1, 2.56))
    rh = [0.129, 0.345, 0.454, 0.565]
    rb = [0.641, 0.345, 0.330, 0.565]
    rc = [0.129, 0.125, 0.454, 0.043]
    ax = fig.add_axes(rh)
    bx = fig.add_axes(rb)
    cax = fig.add_axes(rc)
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)

    ax.set_xlim(-0.5, M.shape[1] - 0.5)
    ax.set_ylim(M.shape[0] - 0.5, -0.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isfinite(v):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, lw=0,
                                           facecolor=DIVERGING(norm(v)), zorder=1))
                fc = "white" if (v > 0.80 or v < 0.18) else C_DARK
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=FS_ANNOT,
                        color=fc, zorder=3)
            else:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#EDEDED",
                                           edgecolor="#BBBBBB", lw=0.4, hatch="////",
                                           zorder=2))
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=FS_ANNOT,
                        color="#6E6E6E", zorder=3)
    for k in range(M.shape[1] + 1):
        ax.axvline(k - 0.5, color="white", lw=1.0, zorder=4)
    for k in range(M.shape[0] + 1):
        ax.axhline(k - 0.5, color="white", lw=1.0, zorder=4)
    for j in range(M.shape[1]):
        if not np.isfinite(M[:, j]).all():
            continue
        top = np.nanmax(M[:, j])
        for i in np.where(M[:, j] >= top - 1e-3)[0]:
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="none",
                                       edgecolor=C_DARK, lw=1.2, zorder=6))
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels([STAT_LABEL[s] for s in STATS])
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels([ATTACK_LABEL[a] for a in ATTACKS], rotation=40, ha="right",
                       rotation_mode="anchor", fontsize=FS_TICK)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0, pad=2.0)
    ax.set_ylabel("statistic")

    edges = np.linspace(0.0, 1.0, 257)
    for lo, hi in zip(edges[:-1], edges[1:]):
        cax.add_patch(plt.Rectangle((lo, 0), hi - lo, 1, lw=0,
                                    facecolor=DIVERGING(norm(0.5 * (lo + hi)))))
    cax.set_xlim(0, 1)
    cax.set_ylim(0, 1)
    cax.set_yticks([])
    cax.set_xticks([0.0, 0.5, 1.0])
    cax.set_xticklabels(["0.0", "0.5", "1.0"])
    cax.tick_params(labelsize=FS_TICK, length=0, pad=2)
    cax.set_xlabel("AUC of one statistic", labelpad=1.5)
    for sp in cax.spines.values():
        sp.set_visible(False)

    y = np.arange(len(STATS))
    bx.axvline(full, color=C_OURS, lw=0.9, ls=(0, (3.5, 2)), zorder=4)
    for yy, v in zip(y, loo):
        bx.barh(yy, v, height=0.62, facecolor=C_MEMLESS if v < full else C_OURS,
                hatch="..." if v < full else "", edgecolor="white", lw=0.25, zorder=3)
        if v < 0.004:
            bx.plot([0, 0], [yy - 0.31, yy + 0.31], color=C_MEMLESS, lw=1.6,
                    solid_capstyle="butt", zorder=6)
        bx.text(min(v + 0.035, 1.02), yy, f"{v:.3f}", ha="left", va="center",
                fontsize=FS_ANNOT, color=C_DARK, zorder=6)
    bx.set_ylim(len(STATS) - 0.4, -0.6)
    bx.set_yticks(y)
    bx.set_yticklabels(["" for _ in y])
    bx.set_xlim(0, 1.42)
    bx.set_xticks([0, 0.5, 1.0])
    bx.set_xlabel("worst-case TPR without it")
    bx.xaxis.grid(True, zorder=0)
    bx.set_axisbelow(True)
    despine(bx)
    bx.tick_params(axis="y", length=0)
    bx.text(full - 0.035, -0.52, f"all five {full:.2f}", ha="right", va="center",
            fontsize=FS_ANNOT, color=C_OURS, zorder=7)

    panel_letter(fig, rh, "(a)", dx=-0.118, dy=0.028)
    panel_letter(fig, rb, "(b)", dx=-0.048, dy=0.028)
    stamp(fig, CFG_MAIN)
    return save_figure(fig, "fig_ablation")


def fused_grid():
    import json as _json
    path = os.path.join(RESULTS, "combined_grid.json")
    if not os.path.exists(path):
        return {}
    cells = _json.load(open(path))["cells"]
    rules = ("combined_max", "combined_sum", "combined_two")
    out = {}
    for key, v in cells.items():
        p = key.split("|")
        phi, beta = float(p[4].split("=")[1]), float(p[5].split("=")[1])
        st = v["stats"]
        vals = [st[r]["auc"] for r in rules if r in st]
        if vals:
            out[(phi, beta)] = max(vals)
    return out


def fig_squeeze():
    grid = cost_grid()
    fused = fused_grid()
    floor, floor_cell = squeeze_floor(grid)
    phis = sorted({k[0] for k in grid})
    betas = sorted({k[1] for k in grid})

    fig = plt.figure(figsize=(COL2, 2.62))
    L, B, W, H = 0.0560, 0.178, 0.9330, 0.678
    gap = 0.0172
    pw = (W - gap * (len(phis) - 1)) / len(phis)

    handles = None
    for i, phi in enumerate(phis):
        rect = [L + i * (pw + gap), B, pw, H]
        ax = fig.add_axes(rect)
        cheats = phi < 1.0
        if not cheats:
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                       facecolor="#000000", alpha=0.05, lw=0, zorder=0))

        xs = np.arange(len(betas))
        ours = [grid_auc(grid[(phi, b)], OURS) for b in betas]
        best_cross = [best_over(grid[(phi, b)], CROSS) for b in betas]
        best_all = [best_over(grid[(phi, b)], ALL_DETECTORS) for b in betas]

        ax.axhline(0.5, color="#9A9A9A", lw=0.6, ls=(0, (1.2, 1.6)), zorder=4)
        ax.axhline(floor, color="#333333", lw=0.7, ls=(0, (3.5, 2)), zorder=5)
        h1, = ax.plot(xs, best_all, color=C_MAXALL, lw=2.8, ls=(0, ()), alpha=0.28,
                      solid_capstyle="round", zorder=3, label="max over all detectors")
        h2, = ax.plot(xs, best_cross, color=C_CROSS, lw=1.15, ls=(0, ()), marker="^",
                      ms=3.0, mfc="white", mec=C_CROSS, mew=0.9, zorder=6,
                      label="best cross-sectional rule")
        h3, = ax.plot(xs, ours, color=C_OURS, lw=1.15, ls=(0, (4.5, 1.6)), marker="s",
                      ms=3.0, mfc="white", mec=C_OURS, mew=0.9, zorder=7, label="ours")
        handles = [h3, h2, h1]
        fused_ys = [fused.get((phi, b)) for b in betas]
        if any(y is not None for y in fused_ys):
            xf = [x for x, y in zip(xs, fused_ys) if y is not None]
            yf = [y for y in fused_ys if y is not None]
            h4, = ax.plot(xf, yf, color="#000000", lw=1.25, ls=(0, (1.1, 1.3)), marker="v",
                          ms=3.2, mfc="#000000", mec="#000000", mew=0.8, zorder=8,
                          label="attacker targets the fusion")
            handles = [h3, h2, h1, h4]

        if phi == floor_cell[0]:
            j = betas.index(floor_cell[1])
            vo = grid_auc(grid[floor_cell], OURS)
            ax.plot([j], [floor], marker="o", ms=6.0, mfc="none", mec=C_MAXALL,
                    mew=0.9, ls="none", zorder=9)
            ax.annotate(f"{floor:.3f}", xy=(j, floor), xytext=(j - 0.30, floor - 0.30),
                        fontsize=FS_ANNOT, color=C_MAXALL, ha="right", va="bottom",
                        arrowprops=dict(arrowstyle="-", color=C_MAXALL, lw=0.5,
                                        shrinkA=0.5, shrinkB=3.0), zorder=9)
            ax.annotate(f"{vo:.3f}", xy=(j, vo), xytext=(j + 0.14, vo - 0.28),
                        fontsize=FS_ANNOT, color=C_OURS, ha="left", va="bottom",
                        arrowprops=dict(arrowstyle="-", color=C_OURS, lw=0.5,
                                        shrinkA=0.5, shrinkB=1.5), zorder=9)

        ax.set_xlim(-0.36, len(betas) - 0.64)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{b:g}" for b in betas])
        ax.set_ylim(0, 1.045)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(r"work $\varphi=%g$" % phi, pad=3.0)
        ax.set_xlabel(r"budget $\beta$")
        ax.grid(True, zorder=0)
        ax.set_axisbelow(True)
        despine(ax)
        if i == 0:
            ax.set_ylabel("detection AUC")
            ax.text(-0.30, floor + 0.022, "floor", ha="left", va="bottom",
                    fontsize=FS_ANNOT, color="#333333", zorder=8)
            ax.text(-0.30, 0.505, "chance", ha="left", va="bottom", fontsize=FS_ANNOT,
                    color="#7A7A7A", zorder=8)
        else:
            ax.tick_params(labelleft=False)

    fig.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.004),
               handlelength=2.4)
    stamp(fig, CFG_MAIN)
    return save_figure(fig, "fig_squeeze")


def fig_limits():
    sw = load_json("sweeps.json")["fraction_sweep"]
    ks = sorted(sw, key=int)
    xs = [int(k) for k in ks]

    fig = plt.figure(figsize=(COL1, 3.30))
    ra = [0.163, 0.618, 0.800, 0.322]
    rb = [0.163, 0.104, 0.800, 0.322]
    ax = fig.add_axes(ra)
    bx = fig.add_axes(rb)

    series = [(OURS, "ours", C_OURS, (0, ()), "o"),
              ("norm_bound", CROSS_LABEL["norm_bound"], C_SINGLE, (0, (5, 1.6, 1.2, 1.6)),
               "D"),
              ("krum", CROSS_LABEL["krum"], C_ENS, (0, (4, 1.6)), "s"),
              ("cosine_cohort", CROSS_LABEL["cosine_cohort"], C_CROSS, (0, (1.2, 1.4)),
               "^")]
    ax.axhspan(0.0, 0.5, color="#000000", alpha=0.045, lw=0, zorder=0)
    ax.axhline(0.5, color="#333333", lw=0.7, ls=(0, (3.5, 2)), zorder=4)
    handles = []
    for col, lab, colr, ls, mk in series:
        ys = [float(sw[k][col]) for k in ks]
        ax.plot(xs, ys, color=colr, ls=ls, lw=1.3 if col == OURS else 1.0, marker=mk,
                ms=3.2 if col == OURS else 2.8, mfc=colr if col == OURS else "white",
                mec=colr, mew=0.8, zorder=5 if col == OURS else 3, clip_on=False)
        handles.append(Line2D([], [], color=colr, ls=ls, lw=1.3 if col == OURS else 1.0,
                              marker=mk, ms=3.2 if col == OURS else 2.8,
                              mfc=colr if col == OURS else "white", mew=0.8, label=lab))
    a0, a1 = float(sw[ks[0]][OURS]), float(sw[ks[-1]][OURS])
    ax.annotate(f"{a0:.3f}", xy=(xs[0], a0), xytext=(xs[0] + 0.16, a0 + 0.055),
                fontsize=FS_ANNOT, color=C_OURS, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color=C_OURS, lw=0.5, shrinkA=0.5,
                                shrinkB=1.5), zorder=9)
    ax.annotate(f"{a1:.3f}", xy=(xs[-1], a1), xytext=(xs[-1] - 0.95, 0.255),
                fontsize=FS_ANNOT, color=C_OURS, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color=C_OURS, lw=0.5, shrinkA=0.5,
                                shrinkB=1.5), zorder=9)
    ax.set_xlim(0.82, 5.18)
    ax.set_xticks(xs)
    ax.set_ylim(0, 1.14)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("colluding attackers out of 10")
    ax.set_ylabel("detection AUC")
    ax.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.008, 1.045),
              handlelength=2.3, ncol=2, columnspacing=0.9)
    ax.text(0.90, 0.515, "chance", ha="left", va="bottom", fontsize=FS_ANNOT,
            color="#333333", zorder=6)

    targets = sorted(load_temporal_results()["results"]["TEST"][OURS], key=float)
    tx = [float(t) for t in targets]
    for level, lab, colr, ls, mk in (("client_round", "per client-round", C_OURS, (0, ()),
                                      "o"),
                                     ("client_run", "per client run", C_BREAK,
                                      (0, (4, 1.6)), "s")):
        pt = [100 * honest_fpr(OURS, level, t)["point"] for t in targets]
        lo = [100 * honest_fpr(OURS, level, t)["lo"] for t in targets]
        hi = [100 * honest_fpr(OURS, level, t)["hi"] for t in targets]
        bx.fill_between(tx, lo, hi, color=colr, alpha=0.13, lw=0, zorder=2)
        bx.plot(tx, pt, color=colr, ls=ls, lw=1.25, marker=mk, ms=3.0,
                mfc=colr, mec="white", mew=0.5, zorder=4, label=lab)
    bx.plot(tx, [100 * t for t in tx], color="#9A9A9A", lw=0.7, ls=(0, (1.4, 1.6)),
            zorder=3, label="nominal")
    op = float(OP)
    bx.axvline(op, color="#333333", lw=0.7, ls=(0, (3.5, 2)), zorder=5)
    for level, colr, lx, ly in (("client_round", C_OURS, op * 1.24, 0.46),
                                ("client_run", C_BREAK, op * 1.12, 165.0)):
        v = 100 * honest_fpr(OURS, level, OP)["point"]
        bx.annotate(f"{v:.2f}%", xy=(op, v), xytext=(lx, ly), fontsize=FS_ANNOT,
                    color=colr, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color=colr, lw=0.5, shrinkA=0.5,
                                    shrinkB=1.5), zorder=9)
    bx.set_xscale("log")
    bx.set_yscale("log")
    bx.set_xlim(min(tx) * 0.78, max(tx) * 1.28)
    bx.set_xticks(tx)
    bx.set_xticklabels([f"{100 * t:g}" for t in tx])
    bx.set_xticks([], minor=True)
    bx.set_ylim(0.30, 260.0)
    bx.set_yticks([1, 10, 100])
    bx.set_yticklabels(["1", "10", "100"])
    bx.set_yticks([], minor=True)
    bx.set_xlabel("nominal FPR target (%)")
    bx.set_ylabel("realized flag rate (%)")
    bx.grid(True, zorder=0)
    bx.set_axisbelow(True)
    despine(bx)
    bx.legend(loc="lower right", bbox_to_anchor=(1.008, -0.030), handlelength=2.3, ncol=1)

    panel_letter(fig, ra, "(a)", dx=-0.140, dy=0.020)
    panel_letter(fig, rb, "(b)", dx=-0.140, dy=0.024)
    stamp(fig, CFG_MAIN)
    return save_figure(fig, "fig_limits")


FIGURES = ["inversion", "condition", "selfreport", "operating", "adaptive", "limits",
           "support", "utility", "adaptive_summary", "fig2", "fig3"]

def fig_window():
    import json as _json
    p = os.path.join(RESULTS, "utility_impact_gpu_s0.json")
    if not os.path.exists(p):
        return None
    S = _json.load(open(p))["summary"]

    order = [("fused_b0.5", 0.5), ("fused_b1", 1.0), ("fused_b2", 2.0)]
    base = S["honest"]["utility"]["tail_acc_prototype_honest"]
    betas, dets, dmgs = [], [], []
    for key, beta in order:
        if key not in S:
            continue
        betas.append(beta)
        d = S[key]["detection"]["combined_max"]
        dets.append(float(d["tpr_client_round"]))
        dmgs.append(base - float(S[key]["utility"]["tail_acc_prototype_honest"]))

    fig = plt.figure(figsize=(COL1, 2.35))
    ax = fig.add_axes([0.165, 0.200, 0.660, 0.740])
    ax2 = ax.twinx()

    ax.plot(betas, dets, color=C_OURS, lw=1.3, marker="s", ms=3.4, mfc="white",
            mec=C_OURS, mew=0.9, zorder=6, label="detection (TPR)")
    ax2.plot(betas, dmgs, color=C_BREAK, lw=1.3, ls=(0, (3.2, 1.6)), marker="^", ms=3.4,
             mfc="white", mec=C_BREAK, mew=0.9, zorder=6, label="accuracy lost")

    ax.axvspan(0.62, 1.55, color="#000000", alpha=0.070, lw=0, zorder=0)
    ax.text(1.085, 0.50, "evasion\nwindow", ha="center", va="center", fontsize=FS_ANNOT,
            color="#3A3A3A", family=FONT, zorder=7)

    ax.set_xlabel(r"attacker budget $\beta$", fontsize=FS_LABEL)
    ax.set_ylabel("detection TPR", fontsize=FS_LABEL, color=C_OURS)
    ax2.set_ylabel("honest accuracy lost", fontsize=FS_LABEL, color=C_BREAK)
    ax.set_ylim(-0.04, 1.08)
    ax2.set_ylim(-0.0018, 0.048)
    ax.set_xlim(0.30, 2.20)
    ax.set_xticks(betas)
    ax.tick_params(labelsize=FS_TICK, length=2.2, width=0.6, colors="#222222")
    ax2.tick_params(labelsize=FS_TICK, length=2.2, width=0.6)
    ax.tick_params(axis="y", colors=C_OURS)
    ax2.tick_params(axis="y", colors=C_BREAK)
    for side in ("top",):
        ax.spines[side].set_visible(False)
        ax2.spines[side].set_visible(False)
    ax.grid(axis="y", color="#DCDCDC", lw=0.5, zorder=0)
    ax.set_axisbelow(True)

    det_off = {0.5: (0, 6), 1.0: (-15, 7), 2.0: (0, 7)}
    dmg_off = {0.5: (2, 7), 1.0: (17, -3), 2.0: (-4, -12)}
    for x, y in zip(betas, dets):
        ax.annotate("%.0f%%" % (100 * y), xy=(x, y), xytext=det_off.get(x, (0, 6)),
                    textcoords="offset points", ha="center", fontsize=FS_ANNOT,
                    color=C_OURS, family=FONT, zorder=8)
    for x, y in zip(betas, dmgs):
        ax2.annotate("%.4f" % y, xy=(x, y), xytext=dmg_off.get(x, (0, -11)),
                     textcoords="offset points", ha="center", fontsize=FS_ANNOT,
                     color=C_BREAK, family=FONT, zorder=8)

    stamp(fig, CFG_MAIN.replace(", 5 seeds", ", seed 0"))
    return save_figure(fig, "fig_window")


CONFIGS = [
    ("CIFAR-100\n" + r"$\delta=0.5$", "temporal_eval.json"),
    ("CIFAR-100\n" + r"$\delta=0.1$", "temporal_eval_cifar100_d01.json"),
    ("CIFAR-10\n" + r"$\delta=0.5$", "temporal_eval_cifar10.json"),
]


def fig_inversion():
    import json as _json
    RULES = [("krum", "Krum"), ("median_dev", "median"),
             ("trimmed_mean_dev", "trimmed mean"), ("bulyan", "Bulyan"),
             ("norm_bound", "norm bound")]
    SRC = [("CIFAR-100 " + r"$\delta=0.5$", "innovation_cifar100_s%d.json", 5, "cifar100_d05"),
           ("CIFAR-100 " + r"$\delta=0.1$", "innovation_cifar100_d01_s%d.json", 3, "cifar100_d01"),
           ("CIFAR-10 " + r"$\delta=0.5$", "innovation_cifar10_s%d.json", 3, "cifar10")]
    bul = {cfg: {atk: (v["auc_mean"], v["auc_sd_ddof0"]) for atk, v in atks.items()}
           for cfg, atks in load_json("bulyan_per_seed.json")["configs"].items()}
    data = {}
    for label, pat, n, cfgkey in SRC:
        runs = []
        for s in range(n):
            p = os.path.join(RESULTS, pat % s)
            if os.path.exists(p):
                runs.append(_json.load(open(p))["results"])
        if not runs:
            continue
        per = {}
        for key, _ in RULES:
            if key == "bulyan":
                v = bul.get(cfgkey, {}).get("global_echo")
                if v is not None:
                    per[key] = (float(v[0]), float(v[1]))
                continue
            v = [r["global_echo"][key]["auc"] for r in runs
                 if key in r["global_echo"] and np.isfinite(r["global_echo"][key]["auc"])]
            if v:
                per[key] = (float(np.mean(v)), float(np.std(v)))
        data[label] = per
    if not data:
        return None

    SWEEP = [("70 rounds", "innovation_late70_s%d.json", 3),
             ("5 clients", "innovation_cohort05_s%d.json", 3),
             ("20 clients", "innovation_cohort20_s%d.json", 3),
             ("40 clients", "innovation_cohort40_s%d.json", 3)]
    krum_all = []
    f1 = load_json("cohort05_f1.json")["summary"]["global_echo"]["krum_f1"]
    for label, pat, n in [("10 cl, 35 rd", "innovation_cifar100_s%d.json", 3)] + SWEEP:
        if label == "5 clients":
            krum_all.append(("5 clients", float(f1["auc_mean"]), float(f1["auc_sd_ddof0"])))
            continue
        v = []
        for s in range(n):
            p = os.path.join(RESULTS, pat % s)
            if not os.path.exists(p):
                continue
            r = _json.load(open(p))["results"]["global_echo"]
            if "krum" in r and np.isfinite(r["krum"]["auc"]):
                v.append(float(r["krum"]["auc"]))
        if v:
            krum_all.append((label, float(np.mean(v)), float(np.std(v))))

    fig = plt.figure(figsize=(COL1, 2.30))
    if krum_all:
        ax = fig.add_axes([0.088, 0.255, 0.545, 0.640])
        bx = fig.add_axes([0.735, 0.255, 0.245, 0.640])
    else:
        ax = fig.add_axes([0.150, 0.255, 0.825, 0.640])
        bx = None
    labels = list(data)
    xs = np.arange(len(RULES))
    width = 0.80 / len(labels)
    colors = [C_OURS, C_CROSS, C_MAXALL]
    hatches = ["", "///", "..."]

    for i, lab in enumerate(labels):
        per = data[lab]
        ys = [per.get(k, (np.nan, 0))[0] for k, _ in RULES]
        es = [per.get(k, (np.nan, 0))[1] for k, _ in RULES]
        ax.bar(xs + (i - (len(labels) - 1) / 2) * width, ys, width * 0.90, yerr=es,
               capsize=1.3, color=colors[i % 3], edgecolor="white", linewidth=0.45,
               hatch=hatches[i % 3], zorder=3, error_kw=dict(lw=0.6, ecolor="#333333"),
               label=lab)

    ax.axhspan(0.0, 0.5, color="#000000", alpha=0.055, lw=0, zorder=0)
    ax.axhline(0.5, color="#333333", lw=0.7, ls=(0, (3.2, 2.0)), zorder=4)
    ax.text(len(RULES) - 1.55, 0.525, "chance", fontsize=FS_MIN, color="#3A3A3A",
            ha="center", va="bottom", family=FONT)
    ax.text(-0.44, 0.235, "ranks the free-rider\nabove honest clients", fontsize=FS_MIN,
            color="#3A3A3A", ha="left", va="center", family=FONT, zorder=6)

    ax.set_xticks(xs)
    ax.set_xticklabels([n for _, n in RULES], fontsize=FS_TICK)
    ax.set_ylabel("detection AUC", fontsize=FS_LABEL)
    ax.set_ylim(0.0, 1.06)
    ax.tick_params(labelsize=FS_TICK, length=2.2, width=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#DCDCDC", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=FS_MIN, frameon=False, ncol=1,
              handlelength=1.3, handletextpad=0.45, borderaxespad=0.15,
              labelspacing=0.28)

    if bx is not None:
        ys = np.arange(len(krum_all))[::-1]
        bx.barh(ys, [m for _, m, _ in krum_all], 0.62,
                xerr=[s for _, _, s in krum_all], color=C_OURS, edgecolor="white",
                linewidth=0.45, zorder=3, error_kw=dict(lw=0.6, ecolor="#333333"))
        bx.axvspan(0.0, 0.5, color="#000000", alpha=0.055, lw=0, zorder=0)
        bx.axvline(0.5, color="#333333", lw=0.7, ls=(0, (3.2, 2.0)), zorder=4)
        for y, (_, m, _) in zip(ys, krum_all):
            bx.text(m + 0.022, y, "%.3f" % m, fontsize=FS_MIN, color="#3A3A3A",
                    ha="left", va="center", family=FONT, zorder=6)
        bx.set_yticks(ys)
        bx.set_yticklabels([lab for lab, _, _ in krum_all], fontsize=FS_MIN)
        bx.set_xlim(0.0, 0.62)
        bx.set_ylim(-0.62, len(krum_all) - 0.38)
        bx.set_xlabel("Krum AUC", fontsize=FS_LABEL)
        bx.set_xticks([0.0, 0.25, 0.5])
        bx.tick_params(labelsize=FS_TICK, length=2.2, width=0.6)
        for side in ("top", "right"):
            bx.spines[side].set_visible(False)
        bx.grid(axis="x", color="#DCDCDC", lw=0.5, zorder=0)
        bx.set_axisbelow(True)
        panel_letter(fig, [0.088, 0.255, 0.545, 0.640], "a")
        panel_letter(fig, [0.735, 0.255, 0.245, 0.640], "b")
    return save_figure(fig, "fig_inversion")


def fig_condition():
    d = load_json("krum_margin.json")
    if not d:
        return None
    sw_raw = d["cluster_sweep"]
    sw = sw_raw
    x = np.array([r["near_over_pair_median"] for r in sw])
    marg = np.array([r["margin_median"] for r in sw])
    sel = np.array([r["selected_frac"] for r in sw])
    order = np.argsort(x)
    x, marg, sel = x[order], marg[order], sel[order]

    fig = plt.figure(figsize=(COL1, 2.45))
    ra = [0.105, 0.590, 0.700, 0.330]
    rb = [0.105, 0.205, 0.700, 0.330]
    ax = fig.add_axes(ra)
    bx = fig.add_axes(rb, sharex=ax)

    ax.axhline(0.0, color="#333333", lw=0.7, ls=(0, (3.2, 2.0)), zorder=4)
    ax.plot(x, marg, color=C_OURS, lw=1.25, marker="o", ms=2.6, zorder=5)
    ax.set_ylabel("Krum margin", fontsize=FS_LABEL)
    ax.set_ylim(-31, 19)
    ax.set_yticks([-20, 0, 15])
    ax.text(1.0, 4.0, "zero innovation ranked first", fontsize=FS_MIN, color="#3A3A3A",
            ha="right", va="bottom", family=FONT, zorder=6)
    ax.text(0.60, -24.0, "an honest client ranked first", fontsize=FS_MIN, color="#3A3A3A",
            ha="left", va="center", family=FONT, zorder=6)
    ax.tick_params(axis="x", labelbottom=False)

    bx.plot(x, sel, color=C_OURS, lw=1.25, marker="o", ms=2.6, zorder=5)
    bx.set_ylabel("Krum selects zero", fontsize=FS_LABEL)
    bx.set_ylim(-0.04, 1.08)
    bx.set_yticks([0, 0.5, 1.0])
    bx.set_xlabel(r"nearest-neighbor distance relative to a random pair, $\rho$",
                  fontsize=FS_LABEL)
    bx.set_xlim(0.40, 1.02)

    meas = d.get("measured", {})
    f1sel = load_json("cohort05_f1.json")["summary"]["global_echo"]["krum_f1_selection"]["selected"]
    pts = [(m["near_over_pair_median"], f1sel if k == "cohort05" else m["selected_frac"])
           for k, m in meas.items() if k != "nopassport"]
    if pts:
        bx.plot([p[0] for p in pts], [p[1] for p in pts], ls="none", marker="D", ms=3.2,
                mfc=C_BREAK, mec="white", mew=0.5, zorder=7, clip_on=False)
        bx.text(1.0, 0.80, "measured federations", fontsize=FS_MIN,
                color=C_BREAK, ha="right", va="center", family=FONT, zorder=7)
    for a_ in (ax, bx):
        a_.tick_params(labelsize=FS_TICK, length=2.2, width=0.6)
        for side in ("top", "right"):
            a_.spines[side].set_visible(False)
        a_.grid(axis="y", color="#DCDCDC", lw=0.5, zorder=0)
        a_.set_axisbelow(True)
    return save_figure(fig, "fig_condition")


def fig_adaptive():
    betas = [0.0, 0.5, 1.0, 2.0]

    cc = load_json("adaptive_costcurve.json")
    cg = load_json("combined_grid.json")
    ue = load_json("unpredictable_eval.json")
    if not (cc and cg and ue):
        return None

    def alone(beta):
        k = "suspicion/mimic|peer_current|phi=0|beta=%g" % beta
        return float(cc["cells"][k]["stats"]["suspicion"]["auc"])

    def fused(beta, rule):
        for k, v in cg["cells"].items():
            p = k.split("|")
            if float(p[4].split("=")[1]) == 0.0 and float(p[5].split("=")[1]) == beta:
                return float(v["stats"][rule]["auc"])
        return None

    def beacon(beta):
        if beta == 0.0:
            return alone(0.0)
        return float(ue["adaptive"]["sub3|worst|peer_current|phi=0|beta=%g" % beta]["auc"])

    series = [
        ("payload score alone", [alone(b) for b in betas], C_OURS, (0, ()), "s"),
        ("fused, max of seven", [fused(b, "combined_max") for b in betas], C_CROSS,
         (0, (4.2, 1.6)), "^"),
        ("fused, two-of-seven", [fused(b, "combined_two") for b in betas], C_MAXALL,
         (0, (1.2, 1.4)), "D"),
        ("beacon-randomised", [beacon(b) for b in betas], C_BREAK,
         (0, (3.0, 1.2, 0.8, 1.2)), "v"),
    ]
    for name, ys, *_ in series:
        if any(y is None for y in ys):
            raise SystemExit("fig_adaptive: missing a point for %r: %s" % (name, ys))
    fig = plt.figure(figsize=(COL1, 2.25))
    ax = fig.add_axes([0.155, 0.235, 0.815, 0.700])
    xs = np.arange(len(betas))
    ax.axhspan(0.0, 0.5, color="#000000", alpha=0.055, lw=0, zorder=0)
    ax.axhline(0.5, color="#333333", lw=0.7, ls=(0, (3.2, 2.0)), zorder=4)
    for name, ys, col, dash, mk in series:
        xf = [x for x, y in zip(xs, ys) if y is not None]
        yf = [y for y in ys if y is not None]
        ax.plot(xf, yf, color=col, lw=1.2, ls=dash, marker=mk, ms=3.2, mfc="white",
                mec=col, mew=0.9, zorder=6, label=name)
    ax.annotate("no test round\nflagged", xy=(2, 0.48), xytext=(1.42, 0.735),
                fontsize=FS_MIN, color="#3A3A3A", family=FONT, ha="center",
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#7A7A7A"), zorder=7)
    ax.set_xticks(xs)
    ax.set_xticklabels([("%g" % b) for b in betas], fontsize=FS_TICK)
    ax.set_xlabel(r"attacker budget $\beta$", fontsize=FS_LABEL)
    ax.set_ylabel("detection AUC", fontsize=FS_LABEL)
    ax.set_ylim(0.0, 1.08)
    ax.tick_params(labelsize=FS_TICK, length=2.2, width=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#DCDCDC", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=FS_MIN, frameon=False, handlelength=1.6,
              handletextpad=0.45, borderaxespad=0.2, labelspacing=0.25)
    stamp(fig, "CIFAR-100; seeds 0,1,2,3,4; victims 0,1")
    return save_figure(fig, "fig_adaptive")


def fig_support():
    ks = (5, 10, 20, 40)
    series = {}
    for k in ks:
        d = load_json("mask_eval_shard%02d.json" % k)
        if not d:
            continue
        atk = d["configs"]["paper"]["attacks"]
        for name in ("ar2_replay", "noise", "peer_copy", "global_echo", "lagged_self",
                     "shrink"):
            if name in atk:
                series.setdefault(name, {})[k] = float(atk[name]["auc_strict"])
    if not series:
        return None

    fig = plt.figure(figsize=(COL1, 2.35))
    ax = fig.add_axes([0.105, 0.225, 0.700, 0.700])

    flat = ("global_echo", "lagged_self", "shrink")
    fv = [[series[n][k] for n in flat if n in series and k in series[n]] for k in ks]
    lo = [min(v) if v else np.nan for v in fv]
    hi = [max(v) if v else np.nan for v in fv]
    ax.fill_between(ks, lo, hi, color="#BFBFBF", alpha=0.55, lw=0, zorder=2)
    ax.plot(ks, hi, color="#8A8A8A", lw=0.9, zorder=3)

    for name, label, col, ls, mk in (
            ("ar2_replay", "AR(2) replay", C_BREAK, "-", "o"),
            ("noise", "noise", C_CROSS, (0, (4.0, 1.6)), "^"),
            ("peer_copy", "peer copy", C_OURS, (0, (1.6, 1.3)), "s")):
        if name not in series:
            continue
        xs = [k for k in ks if k in series[name]]
        ys = [series[name][k] for k in xs]
        ax.plot(xs, ys, color=col, lw=1.3, ls=ls, marker=mk, ms=3.3, mfc="white",
                mec=col, mew=0.9, zorder=5, label=label)

    ax.set_xscale("log")
    ax.set_xticks(list(ks))
    ax.set_xticklabels([str(k) for k in ks], fontsize=FS_TICK)
    ax.minorticks_off()
    ax.set_xlabel("classes held per client, of 100", fontsize=FS_LABEL)
    ax.set_ylabel("detection AUC", fontsize=FS_LABEL)
    ax.set_ylim(0.45, 1.045)
    ax.set_xlim(4.3, 47)
    ax.axhline(0.5, color="#333333", lw=0.7, ls=(0, (3.2, 2.0)), zorder=4)
    ax.text(4.6, 0.513, "chance", fontsize=FS_MIN, color="#3A3A3A", ha="left", va="bottom",
            family=FONT, zorder=6)
    ax.text(44, (np.nanmin(lo) + 1.0) / 2.0 if len(lo) else 0.99,
            "echo, lagged self,\nshrink", fontsize=FS_MIN, color="#3A3A3A", ha="right",
            va="center", family=FONT, zorder=6)
    ax.tick_params(labelsize=FS_TICK, length=2.2, width=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#DCDCDC", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=FS_MIN, frameon=False, handlelength=1.6,
              handletextpad=0.45, borderaxespad=0.25, labelspacing=0.28)
    return save_figure(fig, "fig_support")


def fig_utility():
    d = load_json("utility_seed_spread.json")
    if not d:
        return None
    order = [("fused_b1", r"$\beta=1$"), ("fused_b2", r"$\beta=2$"),
             ("fused_b1_x3", r"$\beta=1$, three")]
    rows = [(lab, d["scenarios"][k]) for k, lab in order if k in d.get("scenarios", {})]
    if not rows:
        return None

    fig = plt.figure(figsize=(COL1, 2.30))
    ax = fig.add_axes([0.215, 0.245, 0.745, 0.650])

    band = d["honest_range"]
    ax.axhspan(-band, band, color="#000000", alpha=0.075, lw=0, zorder=0)
    ax.axhline(0.0, color="#333333", lw=0.8, zorder=3)
    ax.text(len(rows) - 0.42, band, " unattacked\n spread", fontsize=FS_MIN,
            color="#3A3A3A", ha="left", va="center", family=FONT, zorder=6)

    rng = np.random.default_rng(0)
    for i, (lab, r) in enumerate(rows):
        pts = r["per_seed_drop"]
        jit = np.linspace(-0.13, 0.13, len(pts))
        col = C_OURS if r["consistent_sign"] else C_BREAK
        ax.plot([i - 0.26, i + 0.26], [r["mean_drop"]] * 2, color=col, lw=1.6, zorder=5,
                solid_capstyle="butt")
        ax.scatter(i + jit, pts, s=11, facecolor="white", edgecolor=col, linewidth=0.9,
                   zorder=6)
        ax.text(i + 0.30, r["mean_drop"], "%.4f" % r["mean_drop"], fontsize=FS_MIN,
                color="#3A3A3A", ha="left", va="center", family=FONT, zorder=7)

    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([lab for lab, _ in rows], fontsize=FS_TICK)
    ax.set_xlim(-0.55, len(rows) - 0.10)
    ax.set_ylabel("honest accuracy lost", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK, length=2.2, width=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#DCDCDC", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    return save_figure(fig, "fig_utility")


def fig_configs():
    import json as _json
    data = {}
    for label, fname in CONFIGS:
        p = os.path.join(RESULTS, fname)
        if not os.path.exists(p):
            continue
        d = _json.load(open(p))
        auc = d["auc"]
        per = {}
        for a in ATTACKS:
            if a in auc and "suspicion" in auc[a]:
                per[a] = (float(auc[a]["suspicion"]["mean"]),
                          float(auc[a]["suspicion"].get("std", 0.0)))
        data[label] = (per, len(d["protocol"]["archives"]))
    if not data:
        return None

    fig = plt.figure(figsize=(COL2, 2.30))
    ax = fig.add_axes([0.060, 0.215, 0.700, 0.700])
    labels = list(data)
    n = len(labels)
    width = 0.78 / n
    xs = np.arange(len(ATTACKS))
    hatches = ["", "///", "..."]
    colors = [C_OURS, C_CROSS, C_MAXALL]

    for i, lab in enumerate(labels):
        per, nseeds = data[lab]
        ys = [per.get(a, (np.nan, 0))[0] for a in ATTACKS]
        es = [per.get(a, (np.nan, 0))[1] for a in ATTACKS]
        ax.bar(xs + (i - (n - 1) / 2) * width, ys, width * 0.90, yerr=es, capsize=1.4,
               color=colors[i % len(colors)], edgecolor="white", linewidth=0.45,
               hatch=hatches[i % len(hatches)], zorder=3,
               error_kw=dict(lw=0.6, ecolor="#333333"),
               label=lab.replace("\n", " ") + f", {nseeds} seeds")

    ax.axhline(0.5, color="#9A9A9A", lw=0.6, ls=(0, (1.2, 1.6)), zorder=4)
    ax.text(len(ATTACKS) - 0.42, 0.512, "chance", fontsize=FS_MIN, color="#6A6A6A",
            va="bottom", ha="right", family=FONT)
    ax.set_xticks(xs)
    ax.set_xticklabels([ATTACK_LABEL[a] for a in ATTACKS], fontsize=FS_TICK)
    ax.set_ylabel("detection AUC", fontsize=FS_LABEL)
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(labelsize=FS_TICK, length=2.2, width=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#DCDCDC", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="center left", bbox_to_anchor=(1.005, 0.5), fontsize=FS_LEGEND,
              frameon=False, handlelength=1.5, handletextpad=0.5, borderaxespad=0.0)
    return save_figure(fig, "fig_configs")


RETIRED = ["fig_detection", "fig_worstcase", "fig_roc", "fig_strength", "fig_collusion",
           "fig_statistics", "fig_descent", "fig_squeeze", "fig_window", "fig_configs",
           "fig_ablation"]


def composite(stem, parts):
    global save_figure
    heights = {fig_inversion: 2.30, fig_condition: 2.45, fig_selfreport: 3.36, fig_support: 2.35}
    hs = [heights[p] for p in parts]
    master = plt.figure(figsize=(COL1, sum(hs)))
    subs = list(master.subfigures(len(parts), 1, height_ratios=hs))
    queue = list(subs)
    orig_figure, orig_save = plt.figure, save_figure
    plt.figure = lambda *a, **k: queue.pop(0)
    save_figure = lambda fig, name: None
    try:
        for p in parts:
            p()
    finally:
        plt.figure, save_figure = orig_figure, orig_save
    letters = {fig_inversion: "c", fig_selfreport: "c"}[parts[0]]
    for sub in subs[1:]:
        sub.text(0.040, 0.985, letters, fontsize=FS_PANEL, fontweight="bold", ha="left", va="top")
    return save_figure(master, stem)


def prune_retired():
    gone = []
    for stem in RETIRED:
        for ext in ("pdf", "png"):
            p = os.path.join(FIGDIR, f"{stem}.{ext}")
            if os.path.exists(p):
                os.remove(p)
                gone.append(os.path.basename(p))
    if gone:
        print("removed stale figures: " + ", ".join(sorted(gone)))
    else:
        print("no stale figures to remove")
    return gone


def report():
    T = load_temporal_results()
    dets = operating_detectors()
    print("\nlayer 3, worst case over the six attacks (results/temporal_eval.json):")
    print("%-26s%-23s%11s%11s   %s" % ("column", "role", "worst AUC", "worst TPR",
                                       "honest FPR"))
    for col, lab, _c, _h, _m in dets:
        print("%-26s%-23s%11.4f%11.3f   %s"
              % (col, lab, worst_auc(col), worst_tpr(col),
                 honest_fpr(col)["reported"]))
    print("  best single stateful column chosen by the result file: %s"
          % T["verdict"]["best_stateful_baseline"]["column"])
    print("  best memoryless column chosen by the result file:      %s (worst TPR %.3f)"
          % (best_memoryless(), worst_tpr(best_memoryless())))

    print("\nlayer 3, per attack TPR at the frozen threshold:")
    print("%-26s" % "column" + "".join("%14s" % ATTACK_LABEL[a] for a in ATTACKS))
    for col, _lab, _c, _h, _m in dets:
        print("%-26s" % col + "".join("%14.3f" % tpr_of(col, a) for a in ATTACKS))

    print("\nlayer 3, ablation (AUC of one statistic, then worst TPR without it):")
    for s in STATS:
        row = "".join("%10.3f" % auc_of(STAT_COL[s], a) for a in ATTACKS)
        print("  %-6s%s   leave-one-out %.3f" % (s, row, worst_tpr(LOO_COL[s])))
    print("  all five statistics, worst TPR %.3f" % worst_tpr(OURS))

    grid = cost_grid()
    floor, cell = squeeze_floor(grid)
    print("\nlayer 4, the squeeze (results/adaptive_costcurve.json):")
    print("  floor over cheating cells: %.4f at work %g budget %g, attained by %s"
          % (floor, cell[0], cell[1],
             max((d for d in ALL_DETECTORS if d in grid[cell]),
                 key=lambda d: grid_auc(grid[cell], d))))
    print("  %-8s%-8s%10s%10s%10s%10s"
          % ("work", "budget", "ours", "oursTPR", "best X", "max all"))
    for k in sorted(grid):
        v = grid[k]
        print("  %-8g%-8g%10.4f%10.3f%10.4f%10.4f"
              % (k[0], k[1], grid_auc(v, OURS), v[OURS]["tpr"], best_over(v, CROSS),
                 best_over(v, ALL_DETECTORS)))

    neg = load_json("negative_results.json")
    bind = load_json("binding_eval.json")
    g_stab, g_ratio = ghost_medians()
    h_stab, h_ratio, n_honest, warm = honest_probe_medians()
    print("\nlayer 1 and 2:")
    print("  passport entries %d against %d constraints, %.4f unconstrained"
          % (neg["rank_deficiency"]["passport_entries_total"],
             neg["rank_deficiency"]["nominal_constraints_total"],
             neg["rank_deficiency"]["fraction_unconstrained"]))
    hd = neg["N1_exact_forgery"]["headline"]
    print("  forgery: acc %.4f against %.4f, logit shift %.3e, float32 floor %.3e, %.2f s"
          % (hd["acc_forged"], hd["acc_original"], hd["max_abs_dlogit"],
             hd["max_abs_dlogit_float32_noise_floor"], hd["wall_clock_s"]))
    print("  static binding %.4f against %.4f, version: random E %.4f against old E %.4f"
          % (bind["static"]["correct"], bind["static"]["random_theta"],
             bind["version"]["random_E"], bind["version"]["E_first"]))
    print("  ghost: %d of %d rounds accepted, stability %.3f against honest %.3f "
          "(%d honest rounds after warmup %d), drift ratio %.4f against %.4f"
          % (neg["N3_ghost_prover"]["accepted_rounds"], neg["N3_ghost_prover"]["rounds"],
             g_stab, h_stab, n_honest, warm, g_ratio, h_ratio))
    print("  mask sweep (declared support, TPR reading the mask, TPR scoring every class):")
    for s, a, b in mask_shrink_sweep():
        print("    %-6g%10.3f%10.3f" % (s, a, b))
    for s, a, b in mask_churn():
        print("    churn at %-6.2f%10.3f%10.3f" % (s, a, b))
    print("  honest FPR unchanged by the full-row policy: %.4f" % mask_honest_fpr())

    sw = load_json("sweeps.json")["fraction_sweep"]
    print("\nlimits: collusion (ours) "
          + ", ".join("%s:%.3f" % (k, sw[k][OURS]) for k in sorted(sw, key=int)))
    print("  honest flag rate for ours, per target:")
    for t in sorted(load_temporal_results()["results"]["TEST"][OURS], key=float):
        print("    target %-7s round %-22s run %s"
              % (t, honest_fpr(OURS, "client_round", t)["reported"],
                 honest_fpr(OURS, "client_run", t)["reported"]))


def main():
    global FONT, FIGDIR, STYLE
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["inversion", "condition", "selfreport", "support", "operating", "adaptive_summary"], choices=FIGURES)
    ap.add_argument("--keep-retired", action="store_true")
    ap.add_argument("--font-family", default=FONT, choices=["Times New Roman", "DejaVu Serif", "Arial"],
                    help="Explicit plotting font; Times New Roman reproduces the released typography")
    ap.add_argument("--style", default="release", choices=["release", "paper"],
                    help="paper: Arial and bare panel letters, the typography of the manuscript; "
                         "with --only fig2 fig3 it writes the manuscript's Figures 2 and 3 as "
                         "fig2_inversion and fig3_selfreport")
    ap.add_argument("--output-dir", default=FIGDIR, help="Directory for figure files")
    a = ap.parse_args()
    FIGDIR = os.path.abspath(a.output_dir)
    STYLE = a.style
    FONT = "Arial" if STYLE == "paper" else a.font_family

    install_guards()
    assert_fonts()
    set_style()

    for v in (FS_TICK, FS_LABEL, FS_LEGEND, FS_ANNOT, FS_PANEL):
        if not (FS_MIN <= v <= FS_MAX):
            raise SystemExit(f"type size {v} pt is outside the {FS_MIN} to {FS_MAX} band")

    if not a.keep_retired:
        prune_retired()

    want = set(a.only) if a.only else set(FIGURES)
    jobs = [("inversion", fig_inversion),
            ("condition", fig_condition),
            ("adaptive", fig_adaptive),
            ("descent", fig_descent),
            ("selfreport", fig_selfreport),
            ("operating", fig_operating),
            ("ablation", fig_ablation),
            ("squeeze", fig_squeeze),
            ("limits", fig_limits),
            ("configs", fig_configs),
            ("support", fig_support),
            ("utility", fig_utility)]
    for name, fn in jobs:
        if name in want:
            print(f"fig_{name}")
            fn()
    if "fig2" in want:
        print("fig2_inversion (manuscript Figure 2)")
        composite("fig2_inversion", [fig_inversion, fig_condition])
    if "fig3" in want:
        print("fig3_selfreport (manuscript Figure 3)")
        composite("fig3_selfreport", [fig_selfreport, fig_support])

    if "adaptive_summary" in want:
        import make_adaptive_summary
        make_adaptive_summary.draw_adaptive_summary(sys.modules[__name__])

    assert_clean_log()
    assert_no_captions()
    report()


if __name__ == "__main__":
    main()
