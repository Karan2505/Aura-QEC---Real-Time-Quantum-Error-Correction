"""Build the draft paper (Word .docx) from paper/draft_paper.md, the committed result
CSVs in experiments/results/ and the Vivado/co-simulation numbers quoted in the text.

  python paper/build_paper.py            (from aura-qec/)
"""
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor, Cm

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "experiments" / "results"
SRC = HERE / "draft_paper.md"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "draft_paper.docx"
FIG = HERE / "figs"
FIG.mkdir(exist_ok=True)

NAVY = RGBColor(0x1F, 0x3A, 0x5F)
ACCENT = "#2a78d6"
GREY = "#8A8F98"
LIGHT = "#C9CED6"
TEXT = RGBColor(0x26, 0x2B, 0x33)
MUTED = RGBColor(0x5F, 0x66, 0x70)
HEAD_FILL = "1F3A5F"
STRIPE = "F3F6FA"
BODY_FONT = "Calibri"
# validated categorical order (blue, orange, aqua, yellow) + neutral references
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2 = "#1F2A44", "#52514e"

plt.rcParams.update({"font.family": "Calibri", "font.size": 10, "axes.edgecolor": "#B0B5BD",
                     "axes.labelcolor": "#3A3F47", "xtick.color": "#3A3F47", "ytick.color": "#3A3F47",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#E6E9EE", "grid.linewidth": 0.8, "legend.frameon": False})


def rows(name):
    with (RES / name).open() as fh:
        return list(csv.DictReader(fh))


def save(fig, name):
    p = FIG / name
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return p


def eb(ax, x, y, lo, hi, **kw):
    ax.errorbar(x, y, yerr=[[a - b for a, b in zip(y, lo)], [b - a for a, b in zip(y, hi)]], capsize=0,
                elinewidth=1, **kw)


# ----------------------------------------------------------------- figures
def fig_arch():
    """OAM-BP datapath per variable node, inside the fully parallel BP iteration."""
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 48)
    ax.axis("off")

    def box(x, y, w, h, title, sub="", fill="#EEF3FB", edge=C1):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fill, ec=edge,
                                    lw=1.2))
        ax.text(x + w / 2, y + h / 2 + (2.2 if sub else 0), title, ha="center", va="center", fontsize=9,
                color=INK, weight="bold")
        if sub:
            ax.text(x + w / 2, y + h / 2 - 2.6, sub, ha="center", va="center", fontsize=7.5, color=INK2)

    def arrow(x0, y0, x1, y1, txt="", dy=1.6):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1))
        if txt:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, txt, ha="center", fontsize=7.5, color=INK2)

    box(1, 26, 17, 13, "Check nodes", "min-sum, all C\nin parallel")
    box(25, 26, 17, 13, "Variable sum", "Mⱼ = Σ m_cv + Λⱼ")
    box(50, 30, 20, 11, "Flip history", "4-bit shift reg.\n+ prev. sign", fill="#FDF0EA", edge=C2)
    box(50, 12, 20, 11, "State encoder", "popcnt, sign, phase\n→ s = 0..15", fill="#FDF0EA", edge=C2)
    box(77, 12, 20, 11, "Γ ROM (16×9 b)", "learned offline\n(cross-entropy)", fill="#FDF0EA", edge=C2)
    box(77, 30, 20, 11, "Memory mix", "Λⱼ ← P + γ(Mⱼ − P)", fill="#FDF0EA", edge=C2)
    box(25, 3, 17, 11, "Syndrome check", "exit / RL halt")
    box(1, 3, 17, 11, "OSD-0 fallback", "averaged input")
    arrow(18, 32.5, 25, 32.5, "m_cv")
    arrow(42, 35, 50, 35, "sign")
    arrow(60, 30, 60, 23)
    arrow(70, 17.5, 77, 17.5, "s")
    arrow(87, 23, 87, 30, "γ", dy=0)
    arrow(42, 27, 78, 29.4, "")
    ax.text(48, 25.2, "Mⱼ", fontsize=7.5, color=INK2)
    ax.annotate("", (34, 39.5), (87, 41), arrowprops=dict(arrowstyle="-|>", color=C2, lw=1.1,
                                                          connectionstyle="arc3,rad=0.18"))
    ax.text(61, 46.2, "Λⱼ(t+1): next iteration's prior", ha="center", fontsize=7.5, color=C2)
    arrow(33.5, 26, 33.5, 14, "hard", dy=0)
    arrow(25, 8.5, 18, 8.5, "fail")
    ax.text(50, 6, "orange: added by OAM-BP (per variable)", fontsize=7.5, color=C2)
    return save(fig, "p1_arch.png")


def fig_bb72_tradeoff():
    """LER vs mean iterations at lambda = 1, cap 200, all on the same 10k test shots."""
    ens = {r["decoder"]: r for r in rows("oam_ensemble.csv") if float(r["scale"]) == 1.0}
    single = {r["decoder"]: r for r in rows("oam_bb72.csv")
              if float(r["scale"]) == 1.0 and int(r["budget"]) == 200}
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    ax.axhspan(.0634, .0670, color=GREY, alpha=.16, lw=0)
    ax.text(1, .0652, "Relay-BP-1, cap 200 (repeat runs)", ha="left", va="center", color=INK2, fontsize=8.5)
    ax.axhline(.0804, color=GREY, ls="--", lw=1)
    ax.text(58, .0811, "BP+OSD-0, averaged input", ha="right", color=INK2, fontsize=8.5)
    k = [f"ens_fixed{i}" for i in range(1, 5)]
    x = [float(ens[n]["mean_iter"]) for n in k]
    y = [float(ens[n]["ler"]) for n in k]
    eb(ax, x, y, [float(ens[n]["ci_lo"]) for n in k], [float(ens[n]["ci_hi"]) for n in k], color=C3, marker="o",
       ms=5, lw=1.6, ecolor="#9fdcc4", label="Ensemble (OAM first, then fixed γ), 1–4 members")
    for i, (xi, yi) in enumerate(zip(x, y)):
        ax.annotate(f"{i + 1}", (xi, yi), xytext=(5, 5), textcoords="offset points", fontsize=8.5, color=INK2)
    pts = [("fixed_0.125", "Fixed γ = 0.125", C4, "s"), ("fixed_0.25", "Fixed γ = 0.25", C2, "D"),
           ("oam_hand", "OAM, hand-set Γ", GREY, "^"), ("oam_learned", "OAM-BP, learned Γ (single run)", C1, "o")]
    for n, lab, c, m in pts:
        r = single[n]
        eb(ax, [float(r["mean_iter"])], [float(r["ler"])], [float(r["ci_lo"])], [float(r["ci_hi"])], color=c,
           marker=m, ms=8 if n == "oam_learned" else 6, lw=0, ecolor=c, label=lab)
    pair = {(r["decoder"], c): r for c in (100, 200) for r in rows(f"oam_pair_cap{c}.csv")
            if float(r["scale"]) == 1.0}
    r = pair[("oam48", 200)]
    eb(ax, [float(r["mean_iter"])], [float(r["ler"])], [float(r["ci_lo"])], [float(r["ci_hi"])], color=C1,
       marker="o", ms=6, lw=0, ecolor=C1, mfc="white", label="OAM-BP, check-aware 48-state Γ")
    for c, lab, m in ((200, "OAM pair, cap 200 each", "*"), (100, "OAM pair, cap 100 each", "P")):
        r = pair[("pair_both", c)]
        eb(ax, [float(r["mean_iter"])], [float(r["ler"])], [float(r["ci_lo"])], [float(r["ci_hi"])], color="#184f95",
           marker=m, ms=11 if m == "*" else 8, lw=0, ecolor="#184f95", label=lab)
    ax.set_xlabel("Mean BP iterations per shot (sum over runs)")
    ax.set_ylabel("Logical error rate")
    ax.set_xlim(0, 60)
    ax.set_ylim(.050, .085)
    ax.legend(fontsize=8.2, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    return save(fig, "p2_tradeoff.png")


def fig_gamma_heat():
    g = json.loads((RES / "oam_gamma_bb72.json").read_text())["gamma"]
    import numpy as np
    m = np.array(g).reshape(4, 2, 2).reshape(4, 4)           # rows flips; cols (hard, late)
    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("b", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95"])
    im = ax.imshow(m, cmap=cmap, vmin=0, vmax=0.35, aspect="auto")
    ax.grid(False)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if m[i, j] > 0.2 else INK)
    ax.set_xticks(range(4), ["0, early", "0, late", "1, early", "1, late"])
    ax.set_yticks(range(4), ["0", "1", "2", "3+"])
    ax.set_xlabel("Current hard decision (1 = error) and phase (early t ≤ 8, late t > 8)")
    ax.set_ylabel("Sign flips in last 4 iter.")
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("memory strength γ")
    cb.outline.set_visible(False)
    return save(fig, "p3_gamma.png")


def fig_bb72_lambda():
    single = rows("oam_bb72.csv")
    ens = rows("oam_ensemble.csv")
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    series = [("fixed_0.125", "Fixed γ = 0.125", C4, "s", single, True),
              ("fixed_0.25", "Fixed γ = 0.25", C2, "D", single, True),
              ("oam_learned", "OAM-BP, learned Γ", C1, "o", single, True),
              ("ens_fixed4", "Ensemble, 4 members", C3, "v", ens, False),
              ("pair_both", "OAM pair (16 + 48-state Γ)", "#184f95", "*", rows("oam_pair_cap200.csv"), False)]
    for n, lab, c, m, src, cap in series:
        rs = sorted([r for r in src if r["decoder"] == n and (not cap or int(r["budget"]) == 200)],
                    key=lambda r: float(r["scale"]))
        x = [float(r["scale"]) for r in rs]
        eb(ax, x, [float(r["ler"]) for r in rs], [float(r["ci_lo"]) for r in rs], [float(r["ci_hi"]) for r in rs],
           color=c, marker=m, ms=6, lw=1.6, ecolor=c, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("Noise scale λ (1 = calibrated device noise)")
    ax.set_ylabel("Logical error rate")
    ax.set_xticks([0.5, 0.7, 1.0])
    ax.legend(fontsize=8.5, loc="upper left")
    return save(fig, "p4_lambda.png")


def fig_fpga():
    """Measured surface-3 builds on the KV260: resources (share of device) and co-sim latency."""
    dev = {"LUT": 117120, "FF": 234240, "DSP": 1248, "BRAM": 144}
    base = {"LUT": 11698, "FF": 9172, "DSP": 2, "BRAM": 1}
    oam = {"LUT": 11991, "FF": 10857, "DSP": 57, "BRAM": 1}
    import numpy as np
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.8), gridspec_kw={"width_ratios": [1.35, 1]})
    k = list(dev)
    x = np.arange(len(k))
    w = 0.36
    a1.bar(x - w / 2 - 0.01, [100 * base[i] / dev[i] for i in k], w, color=GREY, label="RL-BP baseline")
    a1.bar(x + w / 2 + 0.01, [100 * oam[i] / dev[i] for i in k], w, color=C1, label="OAM-BP")
    a1.set_xticks(x, k)
    a1.set_ylabel("Share of XCK26 (%)")
    a1.legend(fontsize=8.5)
    a1.grid(axis="x", visible=False)
    lat = {"min": (5, 5), "mean": (20, 14), "max": (380, 309)}
    x2 = np.arange(3)
    a2.bar(x2 - w / 2 - 0.01, [v[0] * 3.333 for v in lat.values()], w, color=GREY)
    a2.bar(x2 + w / 2 + 0.01, [v[1] * 3.333 for v in lat.values()], w, color=C1)
    a2.set_yscale("log")
    a2.set_xticks(x2, list(lat))
    a2.set_ylabel("Co-sim latency (ns, 300 MHz)")
    a2.grid(axis="x", visible=False)
    for i, v in enumerate(lat.values()):
        a2.text(i + w / 2, v[1] * 3.333 * 1.12, f"{v[1] * 3.333:.0f}", ha="center", fontsize=8, color=INK)
    fig.tight_layout()
    return save(fig, "p5_fpga.png")


def fig_surface():
    p = RES / "surface_oam.csv"
    if not p.exists():
        return None
    rs = rows("surface_oam.csv")
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.9), sharey=False)
    for ax, code in zip(axs, ("surface:3", "surface:5")):
        for n, lab, c, m in (("mwpm", "MWPM", GREY, "s"), ("bposd_avg2", "BP+OSD-0", C2, "D"),
                             ("oam_osd_avg2", "OAM-BP+OSD-0", C1, "o")):
            rr = sorted([r for r in rs if r["code"] == code and r["decoder"] == n], key=lambda r: float(r["scale"]))
            if not rr:
                continue
            eb(ax, [float(r["scale"]) for r in rr], [float(r["ler"]) for r in rr], [float(r["ci_lo"]) for r in rr],
               [float(r["ci_hi"]) for r in rr], color=c, marker=m, ms=5, lw=1.4, ecolor=c, label=lab)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xticks([0.5, 1, 2], ["0.5", "1", "2"])
        ax.minorticks_off()
        ax.set_title(code.replace("surface:", "Surface code, d = "), fontsize=10, color=INK)
        ax.set_xlabel("Noise scale λ")
    axs[0].set_ylabel("Logical error rate")
    axs[0].legend(fontsize=8.2)
    fig.tight_layout()
    return save(fig, "p6_surface.png")


FIGS = {
    "arch": (fig_arch(), "Figure 1. OAM-BP inside one fully parallel BP iteration. Each variable keeps a 4-bit history of "
                         "hard-decision flips; a 16-state encoder indexes a learned table Γ that sets that variable's "
                         "memory strength γ for the next iteration's prior. Orange blocks are the only additions to the "
                         "RL-BP datapath."),
    "tradeoff": (fig_bb72_tradeoff(), "Figure 2. BB-72 [[72,12,6]] at calibrated device noise (λ = 1), 200-iteration "
                                      "worst case, same 10,000 test shots for every decoder. Whiskers: 95% Wilson "
                                      "intervals. A single learned OAM-BP run sits at the ensemble's and Relay-BP-1's "
                                      "error rate with about half of the ensemble's iterations; the OAM pair (two "
                                      "learned tables, min-weight) is below every other decoder. Iterations are "
                                      "summed over runs."),
    "gamma": (fig_gamma_heat(), "Figure 3. The learned memory table Γ (BB-72, λ = 1). Stable variables keep a weak "
                                "memory (γ ≈ 0.05–0.17); most states of variables that keep flipping late in "
                                "decoding get strong memory (γ ≈ 0.27–0.33), damping the period-2 oscillation that defeats plain BP."),
    "lambda": (fig_bb72_lambda(), "Figure 4. BB-72 logical error rate against noise scale (10,000 shots per point, cap "
                                  "200 per run). The tables are trained only at λ = 1; they transfer to lower noise without "
                                  "retraining."),
    "fpga": (fig_fpga(), "Figure 5. Measured surface-3 builds on the Kria KV260 (XCK26, 300 MHz, post-route). Left: "
                         "share of device resources. Right: co-simulation latency per decode including the OSD-0 "
                         "fallback."),
}
_s = fig_surface()
if _s:
    FIGS["surface"] = (_s, "Figure 6. Surface codes with the BB-72-trained Γ table (no retraining), hardware settings "
                           "(α = 0.75, 30 iterations, OSD-0 fallback). MWPM shown for reference.")
# ----------------------------------------------------------------- docx helpers
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
for side in ("left_margin", "right_margin"):
    setattr(sec, side, Cm(2.3))
sec.top_margin, sec.bottom_margin = Cm(2.2), Cm(2.2)
CONTENT_W = 21.0 - 2 * 2.3

st = doc.styles
st["Normal"].font.name = BODY_FONT
st["Normal"].font.size = Pt(10.5)
st["Normal"].font.color.rgb = TEXT
st["Normal"].element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
st["Normal"].paragraph_format.space_after = Pt(6)
st["Normal"].paragraph_format.line_spacing = 1.12
for name, size, before, after, color in (("Heading 1", 17, 18, 8, NAVY), ("Heading 2", 12.5, 12, 4, NAVY),
                                         ("Title", 28, 0, 6, NAVY)):
    s = st[name]
    s.font.name = BODY_FONT
    s.font.size = Pt(size)
    s.font.bold = True
    s.font.color.rgb = color
    s.paragraph_format.space_before = Pt(before)
    s.paragraph_format.space_after = Pt(after)
    s.paragraph_format.keep_with_next = True
    rf = s.element.get_or_add_rPr().get_or_add_rFonts()
    for k in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rf.set(qn(k), BODY_FONT)
    for k in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        if rf.get(qn(k)) is not None:
            del rf.attrib[qn(k)]


def shade(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def cell_margins(table, top=60, bottom=60, left=90, right=90):
    tblPr = table._tbl.tblPr
    mar = OxmlElement("w:tblCellMar")
    for k, v in (("top", top), ("left", left), ("bottom", bottom), ("right", right)):
        e = OxmlElement(f"w:{k}")
        e.set(qn("w:w"), str(v))
        e.set(qn("w:type"), "dxa")
        mar.append(e)
    tblPr.append(mar)


def borders(table, color="D5DAE1"):
    tblPr = table._tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for k in ("top", "bottom", "insideH"):
        e = OxmlElement(f"w:{k}")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), "4")
        e.set(qn("w:color"), color)
        b.append(e)
    for k in ("left", "right", "insideV"):
        e = OxmlElement(f"w:{k}")
        e.set(qn("w:val"), "nil")
        b.append(e)
    tblPr.append(b)


def repeat_header(row):
    trPr = row._tr.get_or_add_trPr()
    e = OxmlElement("w:tblHeader")
    e.set(qn("w:val"), "true")
    trPr.append(e)


def add_hyperlink(par, text, url, bold=False):
    part = par.part
    r_id = part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
                          is_external=True)
    h = OxmlElement("w:hyperlink")
    h.set(qn("r:id"), r_id)
    r = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    c = OxmlElement("w:color")
    c.set(qn("w:val"), "2F6FDE")
    rPr.append(c)
    u = OxmlElement("w:u")
    u.set(qn("w:val"), "single")
    rPr.append(u)
    if bold:
        rPr.append(OxmlElement("w:b"))
    r.append(rPr)
    t = OxmlElement("w:t")
    t.text = text
    t.set(qn("xml:space"), "preserve")
    r.append(t)
    h.append(r)
    par._p.append(h)


TOKEN = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))")


def add_inline(par, text, size=None, color=None, bold_all=False):
    text = text.replace("\\[", "[").replace("\\]", "]").replace("\\_", "_")
    for part in TOKEN.split(text):
        if not part:
            continue
        if part.startswith("**"):
            r = par.add_run(part[2:-2])
            r.bold = True
        elif part.startswith("`"):
            r = par.add_run(part[1:-1])
            r.font.name = "Consolas"
            r.font.size = Pt((size or 10.5) - 1)
            r.font.color.rgb = RGBColor(0x3A, 0x3F, 0x47)
            continue
        elif part.startswith("["):
            m = re.match(r"\[([^\]]+)\]\(([^)]+)\)", part)
            add_hyperlink(par, m.group(1), m.group(2), bold=bold_all)
            continue
        else:
            r = par.add_run(part)
            if bold_all:
                r.bold = True
        if size:
            r.font.size = Pt(size)
        if color is not None:
            r.font.color.rgb = color


def add_table(rows):
    header, body = rows[0], rows[1:]
    ncol = len(header)
    t = doc.add_table(rows=1 + len(body), cols=ncol)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    # column widths: weight by the longest text in each column
    wts = [max(max(len(r[i]) for r in rows), 10) ** 0.75 for i in range(ncol)]
    tot = sum(wts)
    widths = [Cm(CONTENT_W * w / tot) for w in wts]
    borders(t)
    cell_margins(t)
    size = 9 if ncol <= 5 else 8.5
    for i, row in enumerate(rows):
        tr = t.rows[i]
        if i == 0:
            repeat_header(tr)
        for j, txt in enumerate(row):
            c = tr.cells[j]
            c.width = widths[j]
            p = c.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            if len(rows) <= 8 and i < len(rows) - 1:
                p.paragraph_format.keep_with_next = True
            p.paragraph_format.line_spacing = 1.0
            if i == 0:
                shade(c, HEAD_FILL)
                add_inline(p, txt, size=size, color=RGBColor(0xFF, 0xFF, 0xFF), bold_all=True)
            else:
                if i % 2 == 0:
                    shade(c, STRIPE)
                add_inline(p, txt, size=size)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_figure(key):
    path, caption = FIGS[key]
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(6)
    p.add_run().add_picture(str(path), width=Cm(CONTENT_W - 0.4))
    cp = doc.add_paragraph()
    cp.paragraph_format.space_after = Pt(12)
    r = cp.add_run(caption)
    r.italic = True
    r.font.size = Pt(9)
    r.font.color.rgb = MUTED


def add_field(par, instr):
    r = par.add_run()
    b = OxmlElement("w:fldChar")
    b.set(qn("w:fldCharType"), "begin")
    r._r.append(b)
    r2 = par.add_run()
    it = OxmlElement("w:instrText")
    it.set(qn("xml:space"), "preserve")
    it.text = instr
    r2._r.append(it)
    r3 = par.add_run()
    s = OxmlElement("w:fldChar")
    s.set(qn("w:fldCharType"), "separate")
    r3._r.append(s)
    par.add_run("1")
    r4 = par.add_run()
    e = OxmlElement("w:fldChar")
    e.set(qn("w:fldCharType"), "end")
    r4._r.append(e)


def bottom_rule(par, color="2F6FDE", sz="12"):
    pPr = par._p.get_or_add_pPr()
    bdr = OxmlElement("w:pBdr")
    b = OxmlElement("w:bottom")
    b.set(qn("w:val"), "single")
    b.set(qn("w:sz"), sz)
    b.set(qn("w:space"), "6")
    b.set(qn("w:color"), color)
    bdr.append(b)
    pPr.append(bdr)




# ----------------------------------------------------------------- extra helpers
def add_caption(text):
    """'**Table N.** rest' -> caption paragraph kept with the table below it."""
    p = doc.add_paragraph()
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(4)
    add_inline(p, text, size=9.5, color=NAVY)


def add_equation(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(8)
    r = p.add_run(text)
    r.font.name = "Cambria Math"
    r.font.size = Pt(11.5)


def add_callout(lines, title=None):
    box = doc.add_table(rows=1, cols=1)
    box.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell_margins(box, 140, 140, 200, 200)
    c = box.rows[0].cells[0]
    c.width = Cm(CONTENT_W)
    shade(c, "EEF3FB")
    first = True
    if title:
        cp = c.paragraphs[0]
        r = cp.add_run(title)
        r.bold = True
        r.font.color.rgb = NAVY
        first = False
    for ln in lines:
        cp = c.paragraphs[0] if first else c.add_paragraph()
        first = False
        cp.paragraph_format.space_after = Pt(3)
        add_inline(cp, ln, size=10)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


# ----------------------------------------------------------------- cover page
md = SRC.read_text(encoding="utf-8").splitlines()
meta = {}
while md and md[0].startswith("%"):
    k, _, v = md.pop(0)[1:].partition(":")
    meta[k.strip()] = v.strip()
for _ in range(5):
    doc.add_paragraph()
p = doc.add_paragraph(style="Title")
p.add_run(meta.get("title", "Draft paper"))
bottom_rule(p)
p = doc.add_paragraph()
r = p.add_run(meta.get("subtitle", ""))
r.font.size = Pt(14)
r.font.color.rgb = MUTED
p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(18)
r = p.add_run(meta.get("status", "Draft"))
r.font.size = Pt(11)
r.font.color.rgb = MUTED
for _ in range(2):
    doc.add_paragraph()

# abstract (the '> ' block that opens the file) goes on the cover
i = 0
while i < len(md) and not md[i].strip():
    i += 1
abstract = []
while i < len(md) and md[i].startswith(">"):
    abstract.append(md[i].lstrip("> ").rstrip())
    i += 1
if abstract:
    add_callout([a for a in abstract if a], title="Abstract")
p = doc.add_paragraph()
p.add_run().add_break(WD_BREAK.PAGE)

# ----------------------------------------------------------------- contents
h = doc.add_paragraph(style="Heading 1")
h.add_run("Contents")
p = doc.add_paragraph()
add_field(p, 'TOC \\o "1-2" \\h \\z \\u')
p = doc.add_paragraph()
p.add_run().add_break(WD_BREAK.PAGE)

for s in doc.sections:
    s.different_first_page_header_footer = True
    hp = s.header.paragraphs[0]
    hr = hp.add_run(meta.get("running", "Draft paper"))
    hr.font.size = Pt(8.5)
    hr.font.color.rgb = MUTED
    bottom_rule(hp, color="D5DAE1", sz="4")
    fp = s.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    fr = fp.add_run("Page ")
    fr.font.size = Pt(8.5)
    fr.font.color.rgb = MUTED
    add_field(fp, "PAGE")

# ----------------------------------------------------------------- body
table, callout = [], []


def flush():
    global table, callout
    if table:
        add_table(table)
        table = []
    if callout:
        add_callout(callout[1:], title=callout[0])
        callout = []


while i < len(md):
    line = md[i]
    i += 1
    if line.startswith("|"):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not all(re.fullmatch(r":?-+:?", c) for c in cells):
            table.append(cells)
        continue
    if line.startswith(">"):
        callout.append(line.lstrip("> ").strip().strip("*") if not callout else line.lstrip("> ").rstrip())
        continue
    flush()
    st_ = line.strip()
    if not st_:
        continue
    if st_.startswith("## "):
        doc.add_paragraph(style="Heading 1").add_run(st_[3:])
    elif st_.startswith("### "):
        doc.add_paragraph(style="Heading 2").add_run(st_[4:])
    elif st_.startswith("{{fig:"):
        key = st_[6:-2]
        if key in FIGS:
            add_figure(key)
    elif st_.startswith("$$"):
        add_equation(st_.strip("$ "))
    elif st_.startswith("**Table"):
        add_caption(st_)
    elif st_.startswith("- "):
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(3)
        add_inline(p, st_[2:])
    elif re.match(r"\d+\. ", st_):
        n, rest = st_.split(" ", 1)
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.75)
        p.paragraph_format.first_line_indent = Cm(-0.5)
        p.paragraph_format.space_after = Pt(3)
        r = p.add_run(n + " ")
        r.bold = True
        r.font.color.rgb = NAVY
        add_inline(p, rest)
    else:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        add_inline(p, st_)
flush()

cp_ = doc.core_properties
cp_.author = cp_.last_modified_by = cp_.comments = cp_.keywords = cp_.subject = ""
cp_.title = meta.get("title", "Draft paper")
doc.save(OUT)
print("wrote", OUT)
