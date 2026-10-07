#!/usr/bin/env python3
"""gen_arch_figures.py — architecture figures of the MagiC-Firm control system.

Emits, from ONE layout model per figure:
  * drawio/<name>.drawio   uncompressed draw.io XML (open / import in app.diagrams.net
                           or the desktop app, every box and edge stays editable)
  * control_system_all.drawio   the same figures as pages of one multi-page file
  * svg/<name>.svg         a preview rendered by this script (no draw.io needed)

The figures are derived from the RTL under hardware/rtl/src/control_system (which is a
1:1 port of the SystemC emulator) and the compiler under hardware/compiler/control_system.
Re-run:  python3 arch_figures/gen_arch_figures.py
"""
from __future__ import annotations

import os
import math
from xml.sax.saxutils import escape

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = "Helvetica"

# --------------------------------------------------------------------------------------
# style tables (shared by the draw.io and the SVG emitters)
# --------------------------------------------------------------------------------------
NODE_STYLES = {
    # containers / groups
    "container": dict(fill="#FAFAFA", stroke="#7A7A7A", rounded=1, font=13, bold_first=1,
                      align="left", valign="top", dashed=0),
    "subcontainer": dict(fill="#F1F1F1", stroke="#9A9A9A", rounded=1, font=11, bold_first=1,
                         align="left", valign="top", dashed=1),
    # logic blocks
    "block": dict(fill="#FFFFFF", stroke="#1A1A1A", rounded=1, font=11, bold_first=1),
    "glue": dict(fill="#FFFFFF", stroke="#1A1A1A", rounded=1, font=10, bold_first=1, dashed=1),
    "fsm": dict(fill="#E6F4EA", stroke="#2E7D32", rounded=1, font=11, bold_first=1),
    "mem": dict(fill="#FFF4D6", stroke="#B07D00", rounded=0, font=10, bold_first=1),
    "fifo": dict(fill="#E3F0FB", stroke="#1F5FA8", rounded=0, font=10, bold_first=1),
    "comb": dict(fill="#F0EAFA", stroke="#5E35B1", rounded=1, font=10, bold_first=0),
    "reg": dict(fill="#EDEDED", stroke="#1A1A1A", rounded=0, font=10, bold_first=0),
    "ext": dict(fill="#FFFFFF", stroke="#1A1A1A", rounded=1, font=11, bold_first=1, dashed=1),
    "board": dict(fill="#FFFFFF", stroke="#1A1A1A", rounded=1, font=11, bold_first=1, width=2),
    "port": dict(fill="#FFFFFF", stroke="#555555", rounded=1, arc=50, font=9, bold_first=0),
    "state": dict(fill="#E6F4EA", stroke="#2E7D32", ellipse=1, font=11, bold_first=1),
    "note": dict(fill=None, stroke=None, font=10, bold_first=0, align="left", valign="top"),
    "title": dict(fill=None, stroke=None, font=16, bold_first=1, align="left", valign="top"),
    "bits": dict(fill="#FFFFFF", stroke="#1A1A1A", rounded=0, font=10, bold_first=0),
    "table": dict(fill="#FFFFFF", stroke="#1A1A1A", rounded=0, font=10, bold_first=1, align="left", valign="top", mono=1),
    "swatch": dict(fill="#000000", stroke=None, rounded=0, font=1, bold_first=0),
}

EDGE_STYLES = {
    "data": dict(stroke="#1F3A93", width=1.5, dashed=0),
    "bus": dict(stroke="#1F3A93", width=2.5, dashed=0),
    "ctrl": dict(stroke="#C62828", width=1.5, dashed=1),
    "ps": dict(stroke="#E65100", width=1.5, dashed=1),
    "mem": dict(stroke="#8A6D00", width=1.3, dashed=0),
    "rst": dict(stroke="#6A1B9A", width=1.3, dashed=1),
    "plain": dict(stroke="#444444", width=1.2, dashed=0),
    "fsm": dict(stroke="#2E7D32", width=1.4, dashed=0, curved=1),
}

L, R, T, B = (0, 0.5), (1, 0.5), (0.5, 0), (0.5, 1)


class Fig:
    def __init__(self, name: str, title: str):
        self.name, self.title = name, title
        self.nodes: list[dict] = []
        self.edges: list[dict] = []

    # node
    def n(self, id, label, x, y, w, h, st="block", **kw):
        d = dict(id=id, label=label, x=x, y=y, w=w, h=h, st=st)
        d.update(kw)
        self.nodes.append(d)
        return id

    # edge
    def e(self, src, dst, label="", st="data", exit=None, entry=None, points=None, **kw):
        d = dict(src=src, dst=dst, label=label, st=st, exit=exit, entry=entry, points=points or [])
        d.update(kw)
        self.edges.append(d)

    # a stack of port pills
    def ports(self, prefix, x, y0, items, w=170, h=24, dy=32, st="port"):
        ids = []
        y = y0
        for i, lab in enumerate(items):
            nl = lab.count("\n") + 1
            hh = h + 12 * (nl - 1)
            ids.append(self.n(f"{prefix}{i}", lab, x, y, w, hh, st))
            y += hh + (dy - h)
        return ids

    def node(self, id):
        for d in self.nodes:
            if d["id"] == id:
                return d
        raise KeyError(id)


# --------------------------------------------------------------------------------------
# draw.io emitter
# --------------------------------------------------------------------------------------
def _html_label(label, bold_first):
    # draw.io parses the value as HTML (html=1): escape the text itself first so that
    # e.g. "k<i>_selector" or "MMIO_instr_<p>" are shown literally, then add our tags.
    lines = [escape(ln) for ln in label.split("\n")]
    if bold_first and lines and lines[0]:
        lines[0] = "<b>" + lines[0] + "</b>"
    return "<br>".join(lines)


def node_style_drawio(d):
    s = dict(NODE_STYLES[d["st"]])
    s.update({k: v for k, v in d.items() if k in ("fill", "stroke", "font", "bold_first", "align", "valign", "dashed", "width")})
    parts = []
    if s.get("ellipse"):
        parts.append("ellipse")
    if d["st"] in ("note", "title"):
        parts.append("text")
    parts += ["whiteSpace=wrap", "html=1", f"fontFamily={'Courier New' if s.get('mono') else FONT}", f"fontSize={s['font']}"]
    parts.append("fillColor=" + (s["fill"] or "none"))
    parts.append("strokeColor=" + (s["stroke"] or "none"))
    parts.append(f"rounded={1 if s.get('rounded') else 0}")
    if s.get("arc"):
        parts.append(f"arcSize={s['arc']}")
    if s.get("dashed"):
        parts.append("dashed=1")
    if s.get("width"):
        parts.append(f"strokeWidth={s['width']}")
    parts.append("align=" + s.get("align", "center"))
    parts.append("verticalAlign=" + s.get("valign", "middle"))
    if s.get("align") == "left":
        parts += ["spacingLeft=8", "spacingTop=4"]
    if d["st"] in ("container", "subcontainer"):
        parts.append("container=0")
    return ";".join(parts) + ";"


def edge_style_drawio(d):
    s = dict(EDGE_STYLES[d["st"]])
    parts = ["html=1", f"fontFamily={FONT}", "fontSize=10", "labelBackgroundColor=#FFFFFF",
             "endArrow=block", "endFill=1", f"strokeColor={s['stroke']}", f"strokeWidth={s['width']}"]
    if s.get("curved"):
        parts += ["edgeStyle=none", "curved=1"]
    else:
        parts += ["edgeStyle=orthogonalEdgeStyle", "rounded=0", "orthogonalLoop=1", "jettySize=auto"]
    if s.get("dashed"):
        parts.append("dashed=1")
    if d.get("exit"):
        parts += [f"exitX={d['exit'][0]}", f"exitY={d['exit'][1]}", "exitDx=0", "exitDy=0"]
    if d.get("entry"):
        parts += [f"entryX={d['entry'][0]}", f"entryY={d['entry'][1]}", "entryDx=0", "entryDy=0"]
    if d.get("noarrow"):
        parts[4] = "endArrow=none"
    return ";".join(parts) + ";"


def diagram_xml(fig: Fig, page_id: str) -> str:
    out = []
    W = max(n["x"] + n["w"] for n in fig.nodes) + 40
    H = max(n["y"] + n["h"] for n in fig.nodes) + 40
    out.append(f'  <diagram id="{page_id}" name="{escape(fig.name, {chr(34): "&quot;"})}">')
    out.append(f'    <mxGraphModel dx="1200" dy="800" grid="1" gridSize="10" guides="1" tooltips="1" '
               f'connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{int(W)}" '
               f'pageHeight="{int(H)}" math="0" shadow="0">')
    out.append("      <root>")
    out.append('        <mxCell id="0"/>')
    out.append('        <mxCell id="1" parent="0"/>')
    for d in fig.nodes:
        st = NODE_STYLES[d["st"]]
        bold = d.get("bold_first", st.get("bold_first", 0))
        val = escape(_html_label(d["label"], bold), {'"': "&quot;"})
        out.append(f'        <mxCell id="{d["id"]}" value="{val}" style="{node_style_drawio(d)}" '
                   f'vertex="1" parent="1">')
        out.append(f'          <mxGeometry x="{d["x"]}" y="{d["y"]}" width="{d["w"]}" height="{d["h"]}" as="geometry"/>')
        out.append("        </mxCell>")
    for i, d in enumerate(fig.edges):
        val = escape("<br>".join(escape(ln) for ln in d["label"].split("\n")), {'"': "&quot;"})
        out.append(f'        <mxCell id="e{i}" value="{val}" style="{edge_style_drawio(d)}" edge="1" '
                   f'parent="1" source="{d["src"]}" target="{d["dst"]}">')
        if d["points"]:
            out.append('          <mxGeometry relative="1" as="geometry">')
            out.append('            <Array as="points">')
            for (px, py) in d["points"]:
                out.append(f'              <mxPoint x="{px}" y="{py}"/>')
            out.append("            </Array>")
            out.append("          </mxGeometry>")
        else:
            out.append('          <mxGeometry relative="1" as="geometry"/>')
        out.append("        </mxCell>")
    out.append("      </root>")
    out.append("    </mxGraphModel>")
    out.append("  </diagram>")
    return "\n".join(out)


def mxfile(diagrams: list[str]) -> str:
    return ('<mxfile host="MagiC-Firm/gen_arch_figures.py" version="24.7.0" type="device">\n'
            + "\n".join(diagrams) + "\n</mxfile>\n")


# --------------------------------------------------------------------------------------
# SVG emitter (preview only; draw.io does its own routing from the same anchors)
# --------------------------------------------------------------------------------------
def _anchor(nd, frac):
    return (nd["x"] + frac[0] * nd["w"], nd["y"] + frac[1] * nd["h"])


def _side(frac):
    fx, fy = frac
    if fx == 0:
        return "L"
    if fx == 1:
        return "R"
    if fy == 0:
        return "T"
    if fy == 1:
        return "B"
    return "C"


def _auto_anchors(a, b):
    acx, acy = a["x"] + a["w"] / 2, a["y"] + a["h"] / 2
    bcx, bcy = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
    if abs(bcx - acx) >= abs(bcy - acy):
        return (R if bcx > acx else L), (L if bcx > acx else R)
    return (B if bcy > acy else T), (T if bcy > acy else B)


def route(fig, d):
    a, b = fig.node(d["src"]), fig.node(d["dst"])
    ex, en = d.get("exit"), d.get("entry")
    if ex is None or en is None:
        aex, aen = _auto_anchors(a, b)
        ex = ex or aex
        en = en or aen
    p0, p1 = _anchor(a, ex), _anchor(b, en)
    pts = [p0] + [tuple(p) for p in d["points"]] + [p1]
    if not d["points"] and not EDGE_STYLES[d["st"]].get("curved"):
        s0, s1 = _side(ex), _side(en)
        if s0 in "LR" and s1 in "LR":
            mx = (p0[0] + p1[0]) / 2
            pts = [p0, (mx, p0[1]), (mx, p1[1]), p1]
        elif s0 in "TB" and s1 in "TB":
            my = (p0[1] + p1[1]) / 2
            pts = [p0, (p0[0], my), (p1[0], my), p1]
        elif s0 in "LR":
            pts = [p0, (p1[0], p0[1]), p1]
        elif s0 in "TB":
            pts = [p0, (p0[0], p1[1]), p1]
    elif d["points"] and not EDGE_STYLES[d["st"]].get("curved"):
        # orthogonalise the hops between explicit waypoints
        full = [p0]
        for q in pts[1:]:
            prev = full[-1]
            if prev[0] != q[0] and prev[1] != q[1]:
                full.append((q[0], prev[1]) if _side(ex) in "LR" or len(full) > 1 else (prev[0], q[1]))
            full.append(q)
        pts = full
    dedup = [pts[0]]
    for q in pts[1:]:
        if abs(q[0] - dedup[-1][0]) > 0.01 or abs(q[1] - dedup[-1][1]) > 0.01:
            dedup.append(q)
    return dedup


def _midpoint(pts):
    seg = [math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    total = sum(seg)
    if total == 0:
        return pts[0]
    half, acc = total / 2, 0
    for i, s in enumerate(seg):
        if acc + s >= half:
            t = (half - acc) / s if s else 0
            return (pts[i][0] + t * (pts[i + 1][0] - pts[i][0]), pts[i][1] + t * (pts[i + 1][1] - pts[i][1]))
        acc += s
    return pts[-1]


def _svg_text(lines, cx, y_top, font, bold_first, anchor="middle", color="#111111", weight_all=False, mono=False):
    lh = font * 1.25
    fam = "Courier New, monospace" if mono else f"{FONT}, Arial, sans-serif"
    out = [f'<text font-family="{fam}" font-size="{font}" fill="{color}" text-anchor="{anchor}">']
    for i, ln in enumerate(lines):
        w = ' font-weight="bold"' if (bold_first and i == 0 and ln) or weight_all else ""
        out.append(f'<tspan x="{cx:.1f}" y="{y_top + lh * (i + 1) - font * 0.3:.1f}"{w}>{escape(ln)}</tspan>')
    out.append("</text>")
    return "".join(out)


def svg(fig: Fig) -> str:
    W = max(n["x"] + n["w"] for n in fig.nodes) + 40
    H = max(n["y"] + n["h"] for n in fig.nodes) + 40
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
           '<rect width="100%" height="100%" fill="#FFFFFF"/>', "<defs>"]
    for k, s in EDGE_STYLES.items():
        out.append(f'<marker id="ah-{k}" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto" '
                   f'markerUnits="userSpaceOnUse"><path d="M0,0 L9,4.5 L0,9 z" fill="{s["stroke"]}"/></marker>')
    out.append("</defs>")
    # nodes: containers first (they are listed first by convention), then the rest
    for d in fig.nodes:
        st = dict(NODE_STYLES[d["st"]])
        st.update({k: v for k, v in d.items() if k in ("fill", "stroke", "font", "bold_first", "align", "valign", "dashed", "width")})
        x, y, w, h = d["x"], d["y"], d["w"], d["h"]
        dash = ' stroke-dasharray="6,4"' if st.get("dashed") else ""
        sw = st.get("width", 1)
        if st.get("fill") or st.get("stroke"):
            fill = st["fill"] or "none"
            stroke = st["stroke"] or "none"
            if st.get("ellipse"):
                out.append(f'<ellipse cx="{x + w / 2}" cy="{y + h / 2}" rx="{w / 2}" ry="{h / 2}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash}/>')
            else:
                rx = (min(12.0, min(w, h) * 0.5) if st.get("arc") is None else min(w, h) * st["arc"] / 100.0) if st.get("rounded") else 0
                out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash}/>')
        if d["st"] == "swatch":
            continue
        lines = d["label"].split("\n")
        font = st["font"]
        lh = font * 1.25
        color = d.get("color", "#111111")
        if st.get("align") == "left":
            out.append(_svg_text(lines, x + 8, y + 4, font, st.get("bold_first", 0), anchor="start", color=color, mono=st.get("mono", False)))
        else:
            y_top = y + (h - lh * len(lines)) / 2
            out.append(_svg_text(lines, x + w / 2, y_top, font, st.get("bold_first", 0), color=color))
    # edges
    for d in fig.edges:
        s = EDGE_STYLES[d["st"]]
        pts = route(fig, d)
        dash = ' stroke-dasharray="6,4"' if s.get("dashed") else ""
        marker = "" if d.get("noarrow") else f' marker-end="url(#ah-{d["st"]})"'
        if s.get("curved") and len(pts) > 2:
            path = f"M{pts[0][0]:.1f},{pts[0][1]:.1f} "
            path += " ".join(f"Q{pts[i][0]:.1f},{pts[i][1]:.1f} {(pts[i][0] + pts[i + 1][0]) / 2:.1f},{(pts[i][1] + pts[i + 1][1]) / 2:.1f}"
                             for i in range(1, len(pts) - 1))
            path += f" L{pts[-1][0]:.1f},{pts[-1][1]:.1f}"
            out.append(f'<path d="{path}" fill="none" stroke="{s["stroke"]}" stroke-width="{s["width"]}"{dash}{marker}/>')
        else:
            pstr = " ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in pts)
            out.append(f'<polyline points="{pstr}" fill="none" stroke="{s["stroke"]}" stroke-width="{s["width"]}"{dash}{marker}/>')
        if d["label"]:
            mx, my = _midpoint(pts)
            lines = d["label"].split("\n")
            tw = max(len(ln) for ln in lines) * 5.4 + 8
            th = 12 * len(lines) + 4
            out.append(f'<rect x="{mx - tw / 2:.1f}" y="{my - th / 2:.1f}" width="{tw:.1f}" height="{th:.1f}" fill="#FFFFFF" fill-opacity="0.92"/>')
            out.append(_svg_text(lines, mx, my - th / 2 + 1, 10, 0, color=s["stroke"]))
    out.append("</svg>")
    return "\n".join(out)


# --------------------------------------------------------------------------------------
# legend helper
# --------------------------------------------------------------------------------------
def legend(f: Fig, x, y, entries, w=290):
    f.n("legend", "Legend", x, y, w, 20 + 18 * len(entries) + 8, "container")
    for i, (st, text) in enumerate(entries):
        yy = y + 28 + 18 * i
        s = EDGE_STYLES.get(st) or NODE_STYLES.get(st)
        if st in EDGE_STYLES:
            f.n(f"lg_s{i}", "", x + 10, yy + 5, 28, 3, "swatch", fill=s["stroke"])
        else:
            f.n(f"lg_s{i}", "", x + 10, yy - 1, 28, 14, "swatch", fill=s["fill"], stroke=s["stroke"])
        f.n(f"lg_t{i}", text, x + 44, yy - 6, w - 50, 20, "note")


# ======================================================================================
# FIGURES
# ======================================================================================
def fig00_overview():
    f = Fig("00_system_overview", "Control system — overall architecture (distributed board tree)")
    f.n("ttl", "Control system — overall architecture\nDistributed tree of ControlBoards: events ripple DOWN, data / attempt / finish / post-select ripple UP",
        30, 10, 1000, 40, "title")
    # host layer
    f.n("host", "Host controller\nstart / finish · sees out_finish, discard", 440, 70, 250, 60, "ext")
    f.n("dec", "Decoder + gap estimator\nsoftware MWPM / micro-blossom → gap_post_select", 760, 70, 300, 60, "ext")
    # root
    f.n("root", "Root ControlBoard  (e.g. board 200)\nEVENT_MODE = ORIGINATE · DATA_SRC = EXTERNAL\nBoardControl (attempt / event FSM)  +  DetectorConstructBlock\n"
        "RootOutputSync → detector stream with global indices & round markers", 470, 220, 470, 100, "board")
    # routers
    f.n("r0", "Router ControlBoard  (100)\nFORWARD · EXTERNAL\nRawSelector · DetectorPass · K kernels · OutputSync", 200, 420, 340, 90, "board")
    f.n("r1", "Router = STAGE ControlBoard  (101)\nFORWARD · EXTERNAL · HAS_PS\n+ Postselect → ps_out fast path to root", 870, 420, 340, 90, "board")
    # leaves
    lx = [60, 360, 730, 1030]
    for i, x in enumerate(lx):
        f.n(f"l{i}", f"Leaf ControlBoard  ({i})\nFORWARD · INTERNAL\nDCB: K kernels · RawSelector\nP × PhysicalMMIO (command words)", x, 620, 280, 100, "board")
    # physical
    f.n("phys", "Physical layer: control cores / AWGs → qubits → readout\n(in simulation: one StimReadout per leaf replays recorded measurements — see 11_stim_readout_loop)",
        60, 820, 1250, 60, "ext")
    # toolchain
    f.n("tool", "Compiler toolchain (hardware/compiler/control_system)\nstim circuit + hardware config.json  →\nper-board regfiles (.mem): sync, k<i>_selector,\nk<i>_core, raw_selector, output_sync, global_index,\nround_marker, postselect, MMIO_instr_<p>, MMIO_cw_<p>",
        30, 180, 330, 120, "mem")
    f.n("mono", "Monolithic variant: ONE board = root + leaf fused\n(ORIGINATE · INTERNAL), no children, no raw_out;\nsame ControlBoard module, different parameters.",
        30, 320, 330, 60, "note")
    legend(f, 1090, 60, [("bus", "data plane, UP: det / raw + valid + finish + attempt"),
                         ("ctrl", "control plane, DOWN: event bus {START, ABORT, FINISH} + attempt"),
                         ("ps", "post-select fast path, UP (stage boards) / gap reject"),
                         ("mem", "compiled programs (.mem, $readmemb at elaboration)")], w=340)
    # edges host <-> root
    f.e("host", "root", "start / finish", "ctrl", exit=(0.3, 1), entry=(0.18, 0))
    f.e("root", "host", "out_finish · discard", "ctrl", exit=(0.4, 0), entry=(0.75, 1))
    f.e("root", "dec", "out_det / out_used / out_valid\nout_global_indexes · round markers", "bus", exit=(0.65, 0), entry=(0.3, 1))
    f.e("dec", "root", "gap_post_select", "ps", exit=(0.75, 1), entry=(0.92, 0))
    # root <-> routers
    f.e("root", "r0", "out_ev → ev_*", "ctrl", exit=(0.2, 1), entry=(0.75, 0))
    f.e("r0", "root", "fwd_det / fwd_raw (+valid, finish)\nout_attempt", "bus", exit=(0.45, 0), entry=(0.05, 1))
    f.e("root", "r1", "out_ev → ev_*", "ctrl", exit=(0.8, 1), entry=(0.25, 0))
    f.e("r1", "root", "fwd_det / fwd_raw (+valid, finish)\nout_attempt", "bus", exit=(0.55, 0), entry=(0.95, 1))
    f.e("r1", "root", "ps_out / ps_out_attempt → ps_in", "ps", exit=(0.9, 0), entry=(1, 0.6), points=[(1176, 280)])
    # routers <-> leaves
    for i, (r, ent, ex) in enumerate([("r0", 0.2, 0.5), ("r0", 0.8, 0.5), ("r1", 0.2, 0.5), ("r1", 0.8, 0.5)]):
        f.e(r, f"l{i}", "ev_*", "ctrl", exit=(ent, 1), entry=(0.7, 0))
        f.e(f"l{i}", r, "fwd_det / fwd_raw\nfinish · attempt", "bus", exit=(0.3, 0), entry=(ent - 0.12, 1))
    # leaves <-> physical
    for i in range(4):
        f.e(f"l{i}", "phys", "mmio_out_data / valid\n(command words)", "ctrl", exit=(0.7, 1), entry=((lx[i] + 196 - 60) / 1250, 0))
        f.e("phys", f"l{i}", "in_meas / valid / finish", "data", exit=((lx[i] + 84 - 60) / 1250, 0), entry=(0.3, 1))
    f.e("tool", "root", "per-board .mem files", "mem", exit=(1, 0.4), entry=(0, 0.5))
    return f


def fig01_toolchain():
    f = Fig("01_compiler_toolchain", "Compiler toolchain: stim circuit + config → per-board programs")
    f.n("ttl", "Compiler toolchain (hardware/compiler/control_system)\nstim circuit + hardware config → per-board regfiles / MMIO programs / simulation vectors; validated against stim",
        30, 10, 1100, 40, "title")
    f.n("in1", "stim circuit (.stim)\nQEC / magic-state cultivation", 30, 90, 200, 50, "ext")
    f.n("in2", "hardware config (.json)\nlayers · boards · kernels {n,h} · raw_out\n+ postselect detector set", 30, 170, 200, 70, "ext")
    f.n("parser", "parser.py — DetectorConstructionScanner\nscan() → refine()\ndet_idx = stim detector index\nchannels = spatial coords, windows, stages", 290, 80, 260, 80)
    f.n("config", "config.py — CompilerConfig\nboard tree, path_to_root, lca\npostselect set (explicit indices)", 290, 180, 260, 70)
    f.n("place", "placement.py — Placement\nchannel → LCA of its support (Option B)\nforwarding (normal + obs), stage boards", 610, 120, 260, 80)
    f.n("comp", "compiler.py — Compiler (pure library)\nmask generation → BoardProgram,\nKernelProgram, DetectorOutput\ncore index = emission rank mod h", 930, 120, 260, 90)
    f.n("ser", "serializer.py — Serializer  (PHYSICAL-port layout)\nboard<i>/{json,mem}: qubit_scope (leaf), sync, k<i>_selector,\nk<i>_core, raw_selector[_obs], output_sync, global_index,\nround_marker, postselect (stage) · connections.json\nmeasurement_map.json · manifest.json · report.txt",
        930, 260, 360, 110, "mem")
    f.n("icw", "gen_instr_cw.py  (Phase B)\nMMIO_instr_<p>.mem / MMIO_cw_<p>.mem per core\nwt = inter-measurement time / clock period", 930, 400, 360, 70, "mem")
    f.n("sim", "gen_sim.py  (Phase C)\nstim samples → in_meas / in_valid (DCB layout),\nsim_readout_meas / valid (StimReadout layout),\nexpected_dets, expected_postselect", 930, 500, 360, 80, "mem")
    f.n("cli", "cli.py — front door\norchestrates stages 1–3 (drives Compiler + Serializer)\n--wait-rounds · --wait-row copy-last\n--postselect-layout · --output-sync", 610, 260, 260, 80, "glue")
    f.n("val", "Validation vs stim\ntest_harness.py: in-memory compiler\nvalidate_serialized.py: round-trip from files\n10 topologies × 2 circuits, 0 mismatches", 610, 400, 260, 80, "glue")
    f.n("emu", "SystemC emulator\nhardware/emulator — load_mem_file()\ncycle-accurate model of every block", 290, 650, 260, 70)
    f.n("rtl", "RTL testbenches\nhardware/rtl — gen_dcb_tb.py, gen_cb_tb.py\nVerilator (vsim.sh) / Vivado XSim", 610, 650, 260, 70)
    f.n("fpga", "FPGA synthesis (Vivado)\n$readmemb → LUTRAM (RegFileROM)\n/ BRAM (SyncROM) INIT strings", 930, 650, 260, 70)
    f.n("note", "The SAME .mem files (one word per line, LSB = bit 0) feed the SystemC model, the RTL simulation and synthesis —\nso a program validated in the emulator is bit-identical on the FPGA.",
        290, 560, 600, 40, "note")
    f.e("in1", "parser", "", "data")
    f.e("in2", "config", "", "data")
    f.e("parser", "place", "parsed structs", "data", exit=R, entry=(0, 0.3))
    f.e("config", "place", "board tree", "data", exit=R, entry=(0, 0.7))
    f.e("place", "comp", "", "data")
    f.e("comp", "ser", "programs", "data", exit=(0.5, 1), entry=(0.5, 0))
    f.e("comp", "val", "in-memory masks", "plain", exit=(0, 0.8), entry=(0.5, 0), points=[(895, 192), (895, 380), (740, 380)])
    f.e("ser", "val", "round-trip", "plain", exit=(0, 0.8), entry=(1, 0.7), points=[(915, 348), (915, 456)])
    f.e("ser", "icw", "", "mem", exit=(0.5, 1), entry=(0.5, 0))
    f.e("icw", "sim", "", "mem", exit=(0.5, 1), entry=(0.5, 0))
    f.e("cli", "ser", "", "plain", exit=(1, 0.5), entry=(0, 0.3))
    f.e("sim", "emu", ".mem", "mem", exit=(0, 0.5), entry=(0.5, 0), points=[(420, 540)])
    f.e("sim", "rtl", ".mem", "mem", exit=(0.3, 1), entry=(0.5, 0), points=[(1038, 620), (740, 620)])
    f.e("sim", "fpga", ".mem", "mem", exit=(0.7, 1), entry=(0.5, 0))
    return f


def fig02_control_board():
    f = Fig("02_control_board", "ControlBoard — universal per-board wrapper")
    f.n("ttl", "ControlBoard — the universal per-board wrapper (rtl/src/control_system/ControlBoard.sv)\nOne parameterized module for leaf / router / stage / root / monolithic; all board-to-board glue lives here",
        30, 10, 1200, 40, "title")
    f.n("cb", "ControlBoard  #(IS_ROOT, IS_LEAF, EVENT_MODE, DATA_SRC, HAS_PS, NCHILD, CHILD_DW[], CHILD_RAW[], NPS, P, M, D_IN, K, N, H, RAW_OUT, …)",
        220, 70, 1120, 880, "container")
    # inputs
    f.n("p_ev", "ev_valid / ev_type / ev_payload /\nev_attempt   (from parent)", 20, 120, 180, 36, "port")
    f.n("p_host", "start / finish   (host, root)", 20, 175, 180, 24, "port")
    f.n("p_gap", "gap_post_select   (host, root)", 20, 220, 180, 24, "port")
    f.n("p_psin", "ps_in / ps_in_attempt [NPS]\n(stage boards, root)", 20, 265, 180, 36, "port")
    f.n("p_att", "in_child_attempt [NCHILD]", 20, 340, 180, 24, "port")
    f.n("p_det", "in_det / in_det_valid [D_IN]\n(children, concatenated)", 20, 405, 180, 36, "port")
    f.n("p_meas", "in_meas / in_meas_valid [M]\n(children raw  |  leaf: readout)", 20, 455, 180, 36, "port")
    f.n("p_fin", "in_det_finish_child /\nin_raw_finish_child [NCHILD]", 20, 520, 180, 36, "port")
    f.n("p_mfin", "in_meas_finish [M]   (leaf)", 20, 575, 180, 24, "port")
    f.n("p_clk", "clk / rst   (system)", 20, 890, 180, 24, "port")
    # glue column
    f.n("evmux", "Event-source select\nroot: start/finish → EV_START / EV_FINISH\nnon-root: relay ev_* from parent  (= bc_ev)", 250, 110, 240, 70, "glue")
    f.n("psasm", "Post-select bus assembly (root)\nbc_ps = [ own DCB ps | ps_in[0..NPS) | gap ]\nown + gap slots stamped with cur_attempt", 250, 210, 240, 80, "glue")
    f.n("attor", "Input status\nbc_in_valid = |in_det_valid  |  |in_meas_valid\nbc_data_attempt = |in_child_attempt", 250, 320, 240, 70, "glue")
    f.n("gate", "Stale-valid gating\ndcb_in_*_valid = in_*_valid & ~set_in_data_to_zero", 250, 420, 240, 60, "glue")
    f.n("bcast", "Single-finish broadcast\nchild c's one finish bit → every line in\n[base(c), base(c)+CHILD_DW/RAW[c])  (prefix sums)\nleaf: in_meas_finish passes straight", 250, 510, 240, 90, "glue")
    f.n("mmiomux", "MMIO event source\nORIGINATE: BoardControl.out_ev (registered,\ncarries the internal ABORT)\nFORWARD: bc_ev (received from parent)", 1000, 690, 250, 80, "glue")
    # main blocks
    f.n("bc", "BoardControl  (attempt / event FSM)  →  08_board_control\nIDLE → EXEC → DRAIN;  abort = |(post_select & (ps_att XNOR attempt))\n1-bit attempt: START → 0; abort flips it (ORIGINATE) /\nadopts ev_attempt (FORWARD); stale data = tag mismatch → dropped\nout: reset, set_in_data_to_zero, out_ev_*, out_attempt, discard",
        570, 100, 410, 110, "fsm")
    f.n("dcb", "DetectorConstructBlock (DCB)  →  03_detector_construct_block\nrst = board_reset (local: rst | abort | finish)\nraw path: RawSelector → fwd_raw\nconstruct path: MeasurementSync → K Kernels\npass path: DetectorPass\nd_out = pass ++ construct\nOutputSync / RootOutputSync (+ global index, markers)\nPostselect (HAS_PS) → post_select",
        570, 350, 410, 260, "block")
    f.n("mmio", "PhysicalMMIO × P   (IS_LEAF: leaf / monolithic only)  →  09_physical_mmio\nreset = system rst, NOT board_reset (handles abort/finish via the event bus)",
        570, 690, 410, 160, "subcontainer")
    f.n("c0", "core 0\nInstrSequencer\nInstr RF → Decode\nCW mem", 585, 740, 100, 90)
    f.n("c1", "core 1", 700, 740, 90, 90)
    f.n("cdots", "…", 800, 770, 30, 30, "note")
    f.n("cP", "core P−1", 870, 740, 95, 90)
    f.n("drn", "drain_done  = IS_ROOT ? DCB.out_finish : DCB.fwd_det_finish\n   (the DCB det-finish only fires after every child's finish-tagged\n    detector arrived ⇒ 'my whole subtree has drained')\ndata_valid_output = IS_ROOT ? out_valid : |fwd_det_valid | |fwd_raw_valid",
        1000, 860, 330, 80, "note")
    # outputs
    f.n("o_outev", "out_ev_* → children", 1370, 110, 170, 24, "port")
    f.n("o_disc", "discard → host (root)", 1370, 150, 170, 24, "port")
    f.n("o_att", "out_attempt → parent", 1370, 190, 170, 24, "port")
    f.n("o_psout", "ps_out / ps_out_attempt\n→ root (stage board)", 1370, 240, 170, 36, "port")
    f.n("o_out", "out_det / out_used / out_valid /\nout_finish   (root stream)", 1370, 380, 170, 36, "port")
    f.n("o_gidx", "out_global_indexes, first/last_normal,\nfirst/last_wait   (root)", 1370, 430, 170, 36, "port")
    f.n("o_fwd", "fwd_det / fwd_det_valid /\nfwd_det_finish → parent", 1370, 490, 170, 36, "port")
    f.n("o_raw", "fwd_raw / fwd_raw_valid /\nfwd_raw_finish → parent", 1370, 540, 170, 36, "port")
    f.n("o_mmio", "mmio_out_data / mmio_out_valid [P]\n→ physical control cores", 1370, 720, 170, 36, "port")
    f.n("o_cwf", "cw_gen_finish [P]", 1370, 780, 170, 24, "port")
    # edges
    f.e("p_ev", "evmux", "", "ctrl", exit=R, entry=(0, 0.35))
    f.e("p_host", "evmux", "", "ctrl", exit=R, entry=(0, 0.75))
    f.e("evmux", "bc", "bc_ev_*", "ctrl", exit=R, entry=(0, 0.4))
    f.e("p_gap", "psasm", "", "ps", exit=R, entry=(0, 0.3))
    f.e("p_psin", "psasm", "", "ps", exit=R, entry=(0, 0.75))
    f.e("psasm", "bc", "post_select /\npost_select_attempt [M_PS]", "ps", exit=(1, 0.3), entry=(0, 0.75), points=[(520, 234), (520, 182)])
    f.e("dcb", "psasm", "dcb_post_select (HAS_PS)", "ps", exit=(0, 0.9), entry=(0.85, 1), points=[(540, 584), (540, 300), (454, 300)])
    f.e("p_att", "attor", "", "plain", exit=R, entry=(0, 0.8))
    f.e("attor", "bc", "in_valid, data_attempt", "plain", exit=(1, 0.5), entry=(0, 0.92), points=[(505, 355), (505, 201)])
    f.e("p_det", "gate", "", "bus", exit=R, entry=(0, 0.35))
    f.e("p_meas", "gate", "", "bus", exit=R, entry=(0, 0.75))
    f.e("gate", "attor", "valids", "plain", exit=(0.5, 0), entry=(0.5, 1))
    f.e("gate", "dcb", "in_det + gated valid [D_IN]\nin_meas + gated valid [M]", "bus", exit=R, entry=(0, 0.35))
    f.e("bc", "gate", "set_in_data_to_zero", "rst", exit=(0.08, 1), entry=(1, 0.5), points=[(603, 250), (525, 250), (525, 450)])
    f.e("p_fin", "bcast", "", "plain", exit=R, entry=(0, 0.35))
    f.e("p_mfin", "bcast", "", "plain", exit=R, entry=(0, 0.85))
    f.e("bcast", "dcb", "in_det_finish [D_IN]\nin_meas_finish [M]  (per line)", "data", exit=R, entry=(0, 0.72))
    f.e("bc", "dcb", "board_reset = rst | abort_now | rst_for_finish", "rst", exit=(0.5, 1), entry=(0.5, 0))
    f.e("dcb", "bc", "drain_done, data_valid_output", "plain", exit=(0.85, 0), entry=(0.85, 1))
    f.e("bc", "o_outev", "out_ev_*", "ctrl", exit=(1, 0.2), entry=L)
    f.e("bc", "o_disc", "discard = abort_now", "ctrl", exit=(1, 0.5), entry=L)
    f.e("bc", "o_att", "out_attempt", "plain", exit=(1, 0.8), entry=L)
    f.e("dcb", "o_psout", "ps_out = post_select\nps_out_attempt = cur_attempt", "ps", exit=(1, 0.08), entry=L, points=[(1060, 371), (1060, 258)])
    f.e("dcb", "o_out", "", "bus", exit=(1, 0.3), entry=L)
    f.e("dcb", "o_gidx", "", "bus", exit=(1, 0.42), entry=L)
    f.e("dcb", "o_fwd", "", "bus", exit=(1, 0.6), entry=L)
    f.e("dcb", "o_raw", "", "bus", exit=(1, 0.75), entry=L)
    f.e("bc", "mmiomux", "out_ev (ORIGINATE) / bc_ev (FORWARD)", "ctrl", exit=(0.95, 1), entry=(0.5, 0), points=[(960, 300), (1125, 300)])
    f.e("mmiomux", "mmio", "ev_valid / ev_type / ev_payload", "ctrl", exit=(0, 0.5), entry=(1, 0.25))
    f.e("p_clk", "mmio", "rst (system)", "rst", exit=R, entry=(0.5, 1), points=[(775, 902)])
    f.e("mmio", "o_mmio", "", "ctrl", exit=(1, 0.55), entry=L)
    f.e("mmio", "o_cwf", "", "plain", exit=(1, 0.8), entry=L)
    legend(f, 250, 640, [("bus", "measurement / detector data (+valid, finish)"),
                         ("ctrl", "event bus / command words"),
                         ("ps", "post-select (reject) signals"),
                         ("rst", "local reset / gating"),
                         ("plain", "status / handshake")], w=240)
    return f


def fig03_dcb():
    f = Fig("03_detector_construct_block", "DetectorConstructBlock — per-board detector datapath")
    f.n("ttl", "DetectorConstructBlock — the per-board detector datapath (DetectorConstructBlock.sv)\nwhich sub-blocks exist is chosen by `generate` from the parameters; every regfile is a RegFileROM loaded from this board's .mem files",
        30, 10, 1250, 40, "title")
    f.n("dcb", "DetectorConstructBlock  #(M, D_IN, K, N, H, RAW_OUT, T, NDT, IDX_W, HW_WIDTH, STRIDE, SENTINEL, IS_ROOT, HAS_PS, HAS_OSYNC, COPY_LAST, MEM_DIR)",
        220, 70, 1180, 890, "container")
    f.n("p_meas", "in_meas / in_valid /\nin_meas_finish [M]", 20, 150, 180, 36, "port")
    f.n("p_det", "in_det / in_det_valid /\nin_det_finish [D_IN]", 20, 560, 180, 36, "port")
    f.n("p_rst", "clk / rst (= board_reset)", 20, 900, 180, 24, "port")
    # raw path
    f.n("rawc", "raw_selector.mem  (static)\nRAW_OUT × ⌈log2 M⌉ indices\n$readmemb → packed constant", 260, 100, 220, 60, "mem")
    f.n("raw", "RawSelector  (RAW_OUT > 0)\nforward R of M raw lines up the tree\n1-cycle register; out_finish = OR of\nforwarded (valid & finish)", 540, 100, 250, 80)
    # construct path
    f.n("syncrom", "sync regfile  (RegFileROM)\nT × M  ·  sync.mem\nrow = meas-time, bit = line needed", 260, 230, 220, 60, "mem")
    f.n("ms", "MeasurementSync  (K > 0)\nper-line FIFO (value + finish tag)\nbypass; emit when every needed\nline is available; all-0 mask →\nfast-forward; pc saturates (copy-last)", 540, 210, 250, 100)
    f.n("ker0", "Kernel 0\nselector (N of M) → H cores:\nmasked XOR-accumulate,\nemit-and-clear (one-hot)", 900, 190, 220, 80)
    f.n("ker1", "Kernel 1", 900, 290, 220, 40)
    f.n("kdots", "…", 1000, 338, 30, 20, "note")
    f.n("kerK", "Kernel K−1", 900, 365, 220, 40)
    f.n("core0", "k0 core regfile  (RegFileROM)\nT × H(N+1)  ·  k0_core.mem\n+ k0_selector.mem (static N idx)", 1150, 190, 220, 60, "mem")
    f.n("core1", "k1 core regfile  +  k1 selector", 1150, 290, 220, 40, "mem")
    f.n("coreK", "k(K−1) core regfile  +  selector", 1150, 365, 220, 40, "mem")
    # pass path
    f.n("pass", "DetectorPass  (D_IN > 0)\n1-cycle register of the whole\nchild-detector bus (det, valid, finish)", 540, 540, 250, 70)
    # gather
    f.n("cat", "d_out bus  (D_OUT = D_IN + K)\nline i < D_IN : pass_det[i]\nline D_IN + j : kern_det[j]\n(det, valid, finish per line)", 900, 470, 220, 90, "glue")
    # output path
    f.n("osrom", "output_sync regfile\nNDT × D_OUT  ·  output_sync.mem\nrow = det-time", 260, 660, 220, 60, "mem")
    f.n("girom", "global_index regfile  (root)\nNDT × D_OUT·IDX_W  ·  global_index.mem\nstim detector index per line, SENTINEL = idle", 260, 740, 220, 60, "mem")
    f.n("rmrom", "round_marker regfile  (root)\nNDT × 3  ·  round_marker.mem\n{is_first_normal, is_last_normal, is_wait}", 260, 820, 220, 60, "mem")
    f.n("os", "RootOutputSync  (IS_ROOT)\n= OutputSync + global indices,\n   round markers, copy-last offset\n———\nOutputSync  (non-root, HAS_PS | HAS_OSYNC)\nper-line FIFOs · shared osync_pc\nsynced word = one det-time", 540, 660, 260, 150)
    f.n("psrom", "postselect regfile  (HAS_PS)\nNDT × D_OUT  ·  postselect.mem\naddr = SHARED osync_pc (aligned rows)", 900, 640, 240, 60, "mem")
    f.n("ps", "Postselect  (HAS_PS)\nmask_reg ← mask (1-cycle align)\npost_select = valid & |(mask_reg & det)", 900, 760, 240, 70)
    # outputs
    f.n("o_raw", "fwd_raw / fwd_raw_valid [RAW_OUT]\nfwd_raw_finish → parent", 1430, 110, 190, 36, "port")
    f.n("o_fwd", "fwd_det / fwd_det_valid [D_OUT]\nfwd_det_finish → parent (non-root)", 1430, 680, 190, 36, "port")
    f.n("o_out", "out_det / out_used [D_OUT]\nout_valid / out_finish (root)", 1430, 730, 190, 36, "port")
    f.n("o_gidx", "out_global_indexes [D_OUT × HW_WIDTH]\nfirst/last_normal, first/last_wait", 1430, 780, 190, 36, "port")
    f.n("o_ps", "post_select (stage)", 1430, 840, 190, 24, "port")
    # edges
    f.e("p_meas", "raw", "", "bus", exit=R, entry=(0, 0.6), points=[(510, 168), (510, 148)])
    f.e("p_meas", "ms", "", "bus", exit=R, entry=(0, 0.5), points=[(510, 168), (510, 260)])
    f.e("rawc", "raw", "selector_indexes", "mem", exit=R, entry=(0, 0.3))
    f.e("raw", "o_raw", "", "bus", exit=R, entry=L)
    f.e("ms", "syncrom", "sync_regfile_pc", "mem", exit=(0, 0.8), entry=(1, 0.75), points=[(510, 290), (510, 275)])
    f.e("syncrom", "ms", "sync_mask [M]", "mem", exit=(1, 0.3), entry=(0, 0.25))
    f.e("ms", "ker0", "ms_meas / ms_used [M]\nms_valid, ms_finish", "bus", exit=R, entry=(0, 0.4))
    f.e("ms", "ker1", "", "bus", exit=R, entry=(0, 0.3), points=[(850, 260), (850, 302)])
    f.e("ms", "kerK", "", "bus", exit=R, entry=(0, 0.5), points=[(850, 260), (850, 385)])
    for k, c in (("ker0", "core0"), ("ker1", "core1"), ("kerK", "coreK")):
        f.e(k, c, "core_pc", "mem", exit=(1, 0.7), entry=(0, 0.7))
        f.e(c, k, "core_mask", "mem", exit=(0, 0.3), entry=(1, 0.3))
    f.e("ker0", "cat", "", "data", exit=(1, 0.9), entry=(1, 0.5), points=[(1135, 262), (1135, 515)])
    f.e("ker1", "cat", "", "data", exit=(1, 0.85), entry=(1, 0.5), points=[(1135, 324), (1135, 515)])
    f.e("kerK", "cat", "kern_det / valid / finish [K]", "data", exit=(1, 0.85), entry=(1, 0.5), points=[(1135, 399), (1135, 515)])
    f.e("p_det", "pass", "", "bus", exit=R, entry=L)
    f.e("pass", "cat", "pass_det / valid / finish [D_IN]", "bus", exit=R, entry=(0, 0.6))
    f.e("cat", "os", "dbus_det / valid / finish [D_OUT]", "bus", exit=(0.5, 1), entry=(0.5, 0), points=[(1010, 620), (670, 620)])
    f.e("os", "osrom", "osync_pc (shared)", "mem", exit=(0, 0.2), entry=(1, 0.5), points=[(510, 690)])
    f.e("osrom", "os", "osync_mask", "mem", exit=(1, 0.2), entry=(0, 0.08))
    f.e("girom", "os", "gidx_word", "mem", exit=R, entry=(0, 0.5))
    f.e("rmrom", "os", "rm_word", "mem", exit=R, entry=(0, 0.8))
    f.e("os", "psrom", "osync_pc", "mem", exit=(1, 0.15), entry=(0, 0.5))
    f.e("psrom", "ps", "ps_mask", "mem", exit=(0.5, 1), entry=(0.5, 0))
    f.e("os", "ps", "osync_det, osync_valid", "data", exit=(1, 0.75), entry=(0, 0.5))
    f.e("ps", "o_ps", "", "ps", exit=R, entry=L)
    f.e("os", "o_fwd", "non-root: fwd_det = synced det,\nfwd_det_valid[i] = valid & used[i]", "bus", exit=(1, 0.35), entry=L, points=[(1200, 712), (1200, 698)])
    f.e("os", "o_out", "root", "bus", exit=(1, 0.5), entry=L, points=[(1180, 735), (1180, 748)])
    f.e("os", "o_gidx", "root", "bus", exit=(1, 0.6), entry=L, points=[(1160, 750), (1160, 798)])
    f.e("p_rst", "os", "rst → every sub-block", "rst", exit=R, entry=(0.5, 1), points=[(670, 912)])
    legend(f, 1150, 845, [("bus", "measurement / detector data (+valid, finish)"),
                          ("mem", "regfile address (pc) / word"),
                          ("ps", "post-select"), ("rst", "reset")], w=230)
    return f


def fig04_kernel():
    f = Fig("04_kernel", "Kernel — one detector-construction channel")
    f.n("ttl", "Kernel — one detector-construction channel (Kernel.sv)\nselector picks N of the M synced lines; H cores XOR-accumulate masked subsets and emit-and-clear when a detector window completes",
        30, 10, 1200, 40, "title")
    f.n("k", "Kernel  #(N, H, M, PC_W, SAT_PC)   — 1-cycle latency: datapath combinational, acc / pc / outputs registered", 220, 70, 1060, 780, "container")
    f.n("p_v", "in_valid  (MeasurementSync.out_valid)", 20, 130, 180, 24, "port")
    f.n("p_u", "in_used [M]", 20, 170, 180, 24, "port")
    f.n("p_m", "in_meas [M]", 20, 210, 180, 24, "port")
    f.n("p_f", "in_finish", 20, 250, 180, 24, "port")
    f.n("p_sel", "selector_indexes [N × ⌈log2 M⌉]\n(static, k<i>_selector.mem)", 20, 300, 180, 36, "port")
    f.n("p_cm", "core_mask [H(N+1)]\n(k<i>_core.mem @ core_regfile_pc)", 20, 360, 180, 36, "port")
    f.n("sel", "Selector\nN index muxes over the M lines\nb_m[j] = in_valid & in_used[idx_j]\n            & in_meas[idx_j]", 260, 130, 230, 90)
    # core_mask word layout
    f.n("cml", "core_mask word layout (LSB = bit 0):", 560, 640, 240, 20, "note")
    f.n("cm_h", "core H−1", 560, 665, 90, 30, "bits")
    f.n("cm_d", "…", 650, 665, 50, 30, "bits")
    f.n("cm_1", "core 1", 700, 665, 90, 30, "bits")
    f.n("cm_0", "core 0", 790, 665, 90, 30, "bits")
    f.n("cm_e", "emit", 560, 710, 60, 30, "bits")
    f.n("cm_s", "sel[N−1 … 0]  (which of the N selected bits)", 620, 710, 260, 30, "bits")
    f.n("cmn", "core c occupies bits [c(N+1) +: N+1]:  bit N = emit, bits N−1..0 = select mask", 560, 745, 400, 24, "note")
    # core 0 detail
    f.n("core0", "Core 0", 560, 120, 420, 200, "subcontainer")
    f.n("and0", "AND\nsel_0 & b_m", 580, 170, 90, 50, "comb")
    f.n("xor0", "XOR tree\nres = ^(sel_0 & b_m)", 690, 170, 110, 50, "comb")
    f.n("xacc", "XOR\nb_core = res ^ acc", 820, 170, 100, 50, "comb")
    f.n("acc0", "acc[0]  (1 bit)\n← emit ? 0 : b_core", 700, 250, 130, 40, "reg")
    f.n("core1", "Core 1  (same datapath)", 560, 340, 420, 40, "subcontainer")
    f.n("coreH", "Core H−1  (same datapath)", 560, 400, 420, 40, "subcontainer")
    f.n("onehot", "One-hot select\nout_det = OR_c ( emit_c & b_core_c )\nany_emit = in_valid & |emit\n(≤ 1 core emits per meas-time —\ncompiler guarantee)", 1010, 180, 240, 100, "comb")
    f.n("oreg", "Output registers\nout_valid ← any_emit\nout_det ← out_det_c\nout_finish ← any_emit ? in_finish : 0", 1010, 340, 240, 80, "reg")
    f.n("pc", "core_regfile_pc counter\n+1 per in_valid (in lockstep with\nMeasurementSync); saturates at\nSAT_PC (copy-last wait row)", 260, 560, 260, 80, "reg")
    f.n("nt", "Why this shape: a detector = XOR of measurements spread over several meas-times. Each core accumulates one\ndetector window; the compiler assigns core index = emission rank mod H so windows sharing a core never overlap.",
        260, 790, 900, 40, "note")
    f.n("o_v", "out_valid", 1300, 340, 120, 24, "port")
    f.n("o_d", "out_det", 1300, 375, 120, 24, "port")
    f.n("o_f", "out_finish", 1300, 410, 120, 24, "port")
    f.n("o_pc", "core_regfile_pc", 1300, 585, 120, 24, "port")
    f.e("p_v", "sel", "", "plain", exit=R, entry=(0, 0.25))
    f.e("p_u", "sel", "", "data", exit=R, entry=(0, 0.5))
    f.e("p_m", "sel", "", "bus", exit=R, entry=(0, 0.75))
    f.e("p_sel", "sel", "", "mem", exit=R, entry=(0.5, 1), points=[(375, 318)])
    f.e("sel", "and0", "b_m [N]", "data", exit=R, entry=L)
    f.e("sel", "core1", "", "data", exit=(0.8, 1), entry=(0, 0.5), points=[(444, 360)])
    f.e("sel", "coreH", "", "data", exit=(0.8, 1), entry=(0, 0.5), points=[(444, 420)])
    f.e("p_cm", "and0", "sel_c", "mem", exit=R, entry=(0.3, 1), points=[(530, 378), (530, 240), (607, 240)])
    f.e("and0", "xor0", "", "data")
    f.e("xor0", "xacc", "", "data")
    f.e("acc0", "xacc", "acc", "data", exit=(1, 0.5), entry=(0.5, 1), points=[(870, 270)])
    f.e("xacc", "acc0", "b_core", "data", exit=(0.2, 1), entry=(0.2, 0), points=[(840, 240), (726, 240)])
    f.e("xacc", "onehot", "b_core_0", "data", exit=R, entry=(0, 0.3))
    f.e("p_cm", "onehot", "emit_c", "mem", exit=(0.5, 1), entry=(0.5, 1), points=[(110, 520), (1130, 520)])
    f.e("core1", "onehot", "", "data", exit=R, entry=(0, 0.5), points=[(995, 360), (995, 230)])
    f.e("coreH", "onehot", "", "data", exit=R, entry=(0, 0.7), points=[(1000, 420), (1000, 250)])
    f.e("onehot", "oreg", "", "data", exit=B, entry=T)
    f.e("p_f", "oreg", "in_finish", "plain", exit=R, entry=(0, 0.8), points=[(240, 262), (240, 545), (1005, 545), (1005, 404)])
    f.e("oreg", "o_v", "", "plain", exit=(1, 0.2), entry=L)
    f.e("oreg", "o_d", "", "data", exit=(1, 0.5), entry=L)
    f.e("oreg", "o_f", "", "plain", exit=(1, 0.8), entry=L)
    f.e("p_v", "pc", "in_valid", "plain", exit=(0.5, 1), entry=(0, 0.5), points=[(110, 600)])
    f.e("pc", "o_pc", "", "mem", exit=R, entry=L)
    return f


def fig05_sync():
    f = Fig("05_measurement_sync", "MeasurementSync / OutputSync — per-line FIFO synchronization")
    f.n("ttl", "MeasurementSync (M raw lines) / OutputSync (D detector lines) — identical logic (MeasurementSync.sv, OutputSync.sv)\nasynchronously arriving per-line values are re-aligned into one word per meas-time / det-time, driven by a compiled mask program",
        30, 10, 1250, 40, "title")
    f.n("ms", "MeasurementSync  #(M, D_FIFO, PC_W, SAT_PC)", 220, 70, 1060, 640, "container")
    f.n("p_mask", "sync_mask [M]\n(regfile word @ pc)", 20, 130, 180, 36, "port")
    f.n("p_i", "in_meas[i] / in_valid[i] / in_finish[i]\n(one per line, arrive in any cycle)", 20, 300, 180, 36, "port")
    # regfile + pc row
    f.n("rf", "sync regfile  (RegFileROM, in the parent DCB)\nrow pc → mask: which lines this\nmeas-time needs", 250, 110, 260, 70, "mem")
    f.n("pc", "pc register\nif ready: pc ← (SAT_EN && pc+1 > SAT_PC)\n                  ? SAT_PC : pc + 1", 560, 110, 260, 70, "reg")
    # line i lane
    f.n("lane", "line i  (replicated M times)", 250, 220, 720, 240, "subcontainer")
    f.n("fifo", "FIFO_i  (D_FIFO × {value, finish})\nhead / tail / count\npush: in_valid[i] & ~bypass\npop:  ready & from_fifo[i]", 270, 270, 230, 90, "fifo")
    f.n("src", "source mux\nfrom_fifo[i] = mask[i] & ~empty[i]\nbypass_sel[i] = mask[i] & empty[i] & in_valid[i]\nvalue = from_fifo ? head : in_meas[i]", 540, 260, 260, 90, "comb")
    f.n("avail", "avail[i] = ~mask[i] | from_fifo[i]\n               | bypass_sel[i]", 540, 375, 260, 50, "comb")
    f.n("ovf", "overflow check (sim): want_push & full", 270, 390, 230, 40, "note")
    f.n("lane2", "line i+1 …", 250, 480, 720, 30, "subcontainer")
    f.n("laneM", "line M−1", 250, 525, 720, 30, "subcontainer")
    # shared
    f.n("ready", "ready = AND over all lines of avail[i]\n(all-zero mask ⇒ trivially ready:\nemit all-0 and advance = fast-forward)", 1000, 300, 260, 70, "comb")
    f.n("oreg", "output register (1 pulse per accepted step)\nout_meas[i] ← mask[i] ? value_i : 0\nout_used ← mask\nout_valid ← ready\nout_finish ← OR over used lines of\n   (from_fifo ? head_finish : in_finish[i])", 1000, 400, 260, 110, "reg")
    f.n("nt", "Invariant used by the compiler: a meas-time with an all-0 sync mask has all-0 core words, so one shared pc clocks both\nMeasurementSync and every Kernel; with copy-last the appended wait row saturates the pc so the last round repeats.",
        250, 600, 900, 40, "note")
    f.n("o_pc", "sync_regfile_pc", 1300, 133, 170, 24, "port")
    f.n("o_m", "out_meas [M] / out_used [M]", 1300, 420, 170, 24, "port")
    f.n("o_v", "out_valid / out_finish", 1300, 470, 170, 24, "port")
    f.e("p_i", "fifo", "push", "data", exit=R, entry=(0, 0.4))
    f.e("p_i", "src", "bypass (consumed, not buffered)", "data", exit=R, entry=(0, 0.8), points=[(230, 318), (230, 370), (520, 370), (520, 332)])
    f.e("fifo", "src", "head value / finish", "data", exit=(1, 0.4), entry=(0, 0.4))
    f.e("src", "avail", "", "plain", exit=(0.2, 1), entry=(0.2, 0))
    f.e("avail", "ready", "avail[i]", "plain", exit=R, entry=(0, 0.5))
    f.e("lane2", "ready", "", "plain", exit=R, entry=(0, 0.75), points=[(985, 495), (985, 352)])
    f.e("laneM", "ready", "", "plain", exit=R, entry=(0, 0.9), points=[(990, 540), (990, 363)])
    f.e("ready", "fifo", "pop enable", "plain", exit=(0.5, 0), entry=(0.5, 0), points=[(1130, 200), (385, 200)])
    f.e("src", "oreg", "value_i", "data", exit=R, entry=(0, 0.3), points=[(900, 305), (900, 433)])
    f.e("ready", "oreg", "ready", "plain", exit=B, entry=T)
    f.e("ready", "pc", "ready", "plain", exit=(0.85, 0), entry=(1, 0.8), points=[(1221, 190), (840, 190), (840, 166)])
    f.e("pc", "rf", "pc", "mem", exit=L, entry=R)
    f.e("rf", "p_mask", "", "mem", exit=(0, 0.5), entry=(1, 0.5), noarrow=True)
    f.e("rf", "src", "mask[i]", "mem", exit=(0.7, 1), entry=(0.5, 0), points=[(432, 210), (670, 210)])
    f.e("pc", "o_pc", "", "mem", exit=R, entry=L)
    f.e("oreg", "o_m", "", "bus", exit=(1, 0.3), entry=L)
    f.e("oreg", "o_v", "", "plain", exit=(1, 0.7), entry=L)
    return f


def fig06_root_output_sync():
    f = Fig("06_root_output_sync", "RootOutputSync — global indices, round markers, copy-last offset")
    f.n("ttl", "RootOutputSync — root output path (RootOutputSync.sv)\nwraps OutputSync and stamps every emitted det-time word with stim global detector indices and round markers for the host / decoder",
        30, 10, 1250, 40, "title")
    f.n("ro", "RootOutputSync  #(D, D_FIFO, PC_W, INDEX_W, HW_WIDTH, STRIDE, SENTINEL, SAT_PC)", 220, 70, 1080, 720, "container")
    f.n("p_d", "in_det / in_valid / in_finish [D]\n(= d_out bus: pass ++ construct)", 20, 130, 180, 36, "port")
    f.n("p_sm", "sync_mask [D]  @ pc", 20, 300, 180, 24, "port")
    f.n("p_gi", "global_indexes [D × INDEX_W]  @ pc", 20, 400, 180, 24, "port")
    f.n("p_rm", "round_marker [3]  @ pc\n{is_first_normal, is_last_normal, is_wait}", 20, 470, 180, 36, "port")
    f.n("os", "OutputSync  (see 05)\nper-line FIFOs · ready · pc\n→ out_det, out_used, out_valid, out_finish", 260, 110, 300, 90)
    f.n("mk", "round markers (aligned with out_det)\nfirst_normal = valid & rm_reg[0]\nlast_normal  = valid & rm_reg[1]\nfirst_wait   = valid & rm_reg[2] & ~is_wait_q\nlast_wait    = valid & rm_reg[2] & finish", 620, 230, 300, 110, "comb")
    f.n("align", "1-cycle alignment registers\ngi_reg ← global_indexes\nrm_reg ← round_marker\npc_reg ← sync_regfile_pc\n(line up with OutputSync's registered out_det)", 260, 380, 300, 100, "reg")
    f.n("gidx", "per line l:\nidx = gi_reg[l]\nused = out_valid & out_used[l]\nout_global_indexes[l] =\n  (used && idx != SENTINEL)\n    ? zext(idx) + offset\n    : all-ones  (HW_WIDTH)", 620, 380, 300, 150, "comb")
    f.n("wcnt", "wait counter w\nsat_here = SAT_EN & (pc_reg == SAT_PC)\nif out_valid & sat_here: w ← w + 1\nis_wait_q ← rm_reg[2] on each valid", 260, 520, 300, 90, "reg")
    f.n("off", "offset = sat_here ? w × STRIDE : 0\n(each repeated copy-last wait round\ngets a fresh detector-number block)", 620, 570, 300, 70, "comb")
    f.n("nt", "The host / decoder consumes (out_det, out_global_indexes) per det-time; first_normal / last_normal / first_wait /\nlast_wait tell the control core where the normal rounds end and the (repeated) wait rounds begin and end.",
        260, 690, 900, 40, "note")
    f.n("o_d", "out_det / out_used [D]\nout_valid / out_finish", 1330, 120, 180, 36, "port")
    f.n("o_pc", "sync_regfile_pc\n(→ output_sync, global_index,\nround_marker regfiles)", 1330, 190, 180, 48, "port")
    f.n("o_mk", "first_normal / last_normal\nfirst_wait / last_wait", 1330, 270, 180, 36, "port")
    f.n("o_gi", "out_global_indexes\n[D × HW_WIDTH]", 1330, 440, 180, 36, "port")
    f.e("p_d", "os", "", "bus", exit=R, entry=(0, 0.4))
    f.e("p_sm", "os", "", "mem", exit=R, entry=(0, 0.8), points=[(230, 312), (230, 182)])
    f.e("p_gi", "align", "", "mem", exit=R, entry=(0, 0.35))
    f.e("p_rm", "align", "", "mem", exit=R, entry=(0, 0.65))
    f.e("os", "o_d", "", "bus", exit=(1, 0.3), entry=L)
    f.e("os", "o_pc", "sync_regfile_pc", "mem", exit=(1, 0.75), entry=L, points=[(1000, 177), (1000, 214)])
    f.e("os", "align", "pc", "mem", exit=(0.5, 1), entry=(0.5, 0))
    f.e("os", "mk", "out_valid, out_finish", "plain", exit=(0.8, 1), entry=(0, 0.5), points=[(500, 285)])
    f.e("align", "mk", "rm_reg", "plain", exit=(1, 0.3), entry=(0, 0.7), points=[(590, 410), (590, 307)])
    f.e("align", "gidx", "gi_reg", "mem", exit=(1, 0.5), entry=(0, 0.3))
    f.e("os", "gidx", "out_valid, out_used", "plain", exit=(0.85, 1), entry=(0.5, 0), points=[(515, 360), (770, 360)])
    f.e("align", "wcnt", "pc_reg, rm_reg", "plain", exit=(0.5, 1), entry=(0.5, 0))
    f.e("wcnt", "off", "w, sat_here", "plain", exit=(1, 0.5), entry=(0, 0.5))
    f.e("wcnt", "mk", "is_wait_q", "plain", exit=(0.5, 1), entry=(1, 0.65), points=[(410, 655), (940, 655), (940, 301)])
    f.e("off", "gidx", "offset", "data", exit=(0.5, 0), entry=(0.5, 1))
    f.e("gidx", "o_gi", "", "bus", exit=(1, 0.4), entry=L)
    f.e("mk", "o_mk", "", "plain", exit=(1, 0.35), entry=L)
    return f


def fig07_small_blocks():
    f = Fig("07_small_blocks", "RawSelector · DetectorPass · Postselect · DrainAggregator")
    f.n("ttl", "Small datapath / control blocks\nRawSelector.sv · DetectorPass.sv · Postselect.sv · DrainAggregator.sv",
        30, 10, 1000, 40, "title")
    # RawSelector
    f.n("rs", "RawSelector  #(R, M)  — forward R of the M raw lines up the tree", 30, 70, 600, 220, "container")
    f.n("rs_i", "in_meas / in_valid /\nin_finish [M]", 50, 120, 150, 36, "port")
    f.n("rs_c", "selector_indexes\n[R × ⌈log2 M⌉] (static)", 50, 200, 150, 36, "port")
    f.n("rs_mux", "R index muxes\nsel_meas[i]  = in_meas[idx_i]\nsel_valid[i] = in_valid[idx_i]\nsel_finish = OR_i (in_valid[idx_i] & in_finish[idx_i])", 240, 110, 240, 80, "comb")
    f.n("rs_reg", "1-cycle register", 240, 220, 240, 40, "reg")
    f.n("rs_o", "out_meas / out_valid [R]\nout_finish (single)", 500, 210, 120, 48, "port")
    f.e("rs_i", "rs_mux", "", "bus", exit=R, entry=(0, 0.4))
    f.e("rs_c", "rs_mux", "", "mem", exit=R, entry=(0, 0.8), points=[(220, 218), (220, 174)])
    f.e("rs_mux", "rs_reg", "", "data", exit=B, entry=T)
    f.e("rs_reg", "rs_o", "", "bus", exit=R, entry=L)
    # DetectorPass
    f.n("dp", "DetectorPass  #(D)  — forward children's detectors up the tree (identity form of RawSelector)", 670, 70, 600, 220, "container")
    f.n("dp_i", "in_det / in_valid /\nin_finish [D]", 690, 150, 150, 36, "port")
    f.n("dp_reg", "1-cycle register of the whole D-line bus\nout_det ← in_det · out_valid ← in_valid\nout_finish ← in_finish  (per line, stays aligned)", 880, 130, 260, 80, "reg")
    f.n("dp_o", "out_det / out_valid /\nout_finish [D]", 1160, 150, 100, 48, "port")
    f.e("dp_i", "dp_reg", "", "bus")
    f.e("dp_reg", "dp_o", "", "bus")
    # Postselect
    f.n("ps", "Postselect  #(D)  — stage-board reject: 'a postselect detector fired this det-time'", 30, 330, 600, 240, "container")
    f.n("ps_v", "in_valid  (OutputSync.out_valid)", 50, 380, 170, 24, "port")
    f.n("ps_d", "in_det [D]  (OutputSync.out_det)", 50, 420, 170, 24, "port")
    f.n("ps_m", "postselect_mask [D]\n(ps regfile @ shared osync_pc)", 50, 480, 170, 36, "port")
    f.n("ps_reg", "mask_reg ← postselect_mask\n(1-cycle align, like gi_reg)", 260, 470, 200, 50, "reg")
    f.n("ps_or", "post_select =\nin_valid & |(mask_reg & in_det)", 260, 380, 200, 60, "comb")
    f.n("ps_o", "post_select\n(→ BoardControl abort)", 490, 390, 120, 40, "port")
    f.n("ps_n", "No own pc: the postselect and output_sync regfiles are built row-for-row by det-time, so the\nshared osync_pc keeps them aligned and the copy-last saturation (zero row ⇒ no reject in wait rounds) is inherited.",
        50, 525, 570, 40, "note")
    f.e("ps_v", "ps_or", "", "plain", exit=R, entry=(0, 0.3))
    f.e("ps_d", "ps_or", "", "bus", exit=R, entry=(0, 0.7))
    f.e("ps_m", "ps_reg", "", "mem")
    f.e("ps_reg", "ps_or", "", "mem", exit=(0.5, 0), entry=(0.5, 1))
    f.e("ps_or", "ps_o", "", "ps")
    # DrainAggregator
    f.n("da", "DrainAggregator  #(N)  — join N one-cycle pulses (a leaf's P cw_gen_finish) into one 'all done' pulse", 670, 330, 600, 240, "container")
    f.n("da_i", "drain_done_in [N]\n(pulses, different cycles)", 690, 380, 160, 36, "port")
    f.n("da_c", "clear  (START)", 690, 440, 160, 24, "port")
    f.n("da_l", "latched ← latched | drain_done_in\n(cleared by clear / rst)", 880, 380, 220, 50, "reg")
    f.n("da_e", "all_latched = &latched\nall_drain_done = all_latched & ~all_latched_q\n(edge-aligned, one full cycle wide)", 880, 460, 220, 70, "comb")
    f.n("da_o", "all_drain_done", 1130, 480, 120, 24, "port")
    f.n("da_n", "Fires one cycle after the LAST input arrives. Router-side drain aggregation is retired:\nsubtree drain now rides the DATA path (DCB det out_finish).", 690, 525, 560, 36, "note")
    f.e("da_i", "da_l", "", "plain", exit=R, entry=(0, 0.4))
    f.e("da_c", "da_l", "", "ctrl", exit=R, entry=(0, 0.8), points=[(865, 452), (865, 420)])
    f.e("da_l", "da_e", "", "plain", exit=B, entry=T)
    f.e("da_e", "da_o", "", "plain")
    return f


def fig08_board_control():
    f = Fig("08_board_control", "BoardControl — per-board attempt / event state machine")
    f.n("ttl", "BoardControl — per-board control / attempt state machine (BoardControl.sv)\ntwo independent axes: EVENT_MODE {ORIGINATE, FORWARD} × DATA_SRC {EXTERNAL, INTERNAL}; a 1-bit attempt tag makes stale data drop-only (no purge window)",
        30, 10, 1250, 40, "title")
    f.n("bc", "BoardControl  #(EVENT_MODE, DATA_SRC, M, PAYLOAD_W)", 220, 70, 1100, 760, "container")
    f.n("p_ev", "ev_valid / ev_type / ev_payload /\nev_attempt   (parent | host start/finish)", 20, 110, 180, 36, "port")
    f.n("p_ps", "post_select [M] /\npost_select_attempt [M]", 20, 250, 180, 36, "port")
    f.n("p_iv", "in_valid, data_attempt", 20, 420, 180, 24, "port")
    f.n("p_dd", "drain_done  (DCB det finish)", 20, 560, 180, 24, "port")
    f.n("p_dvo", "data_valid_output", 20, 620, 180, 24, "port")
    # decode
    f.n("dec", "event decode\nis_start  = valid & type==START (00)\nis_abort_ev = valid & type==ABORT (01)\nis_finish = valid & type==FINISH (10)", 250, 100, 260, 70, "comb")
    f.n("abd", "abort detect (ORIGINATE only)\nabort_detected = |( post_select &\n   (post_select_attempt XNOR {M{attempt}}) )\nabort_det_q ← abort_detected & ~abort_det_q  (one-shot)\nabort_now = abort_det_q | is_abort_ev", 250, 220, 300, 100, "comb")
    # FSM states
    f.n("fsmbox", "FSM (all boards)", 620, 100, 660, 300, "subcontainer")
    f.n("s_idle", "IDLE", 660, 210, 110, 50, "state")
    f.n("s_exec", "EXEC", 890, 210, 110, 50, "state")
    f.n("s_drain", "DRAIN", 1120, 210, 110, 50, "state")
    f.e("s_idle", "s_exec", "is_start:\nattempt ← ORIGIN ? 0 : ev_attempt\n(ORIGIN: emit EV_START)", "fsm", exit=(1, 0.5), entry=(0, 0.5), points=[(830, 200)])
    f.e("s_exec", "s_exec", "abort_now: attempt ← ORIGIN ? ~attempt : ev_attempt\n(ORIGIN: emit EV_ABORT with the NEW attempt)", "fsm", exit=(0.3, 0), entry=(0.7, 0), points=[(900, 140), (990, 140)])
    f.e("s_exec", "s_drain", "is_finish & ~drain_done:\nride FINISH down once\n(ORIGIN: emit EV_FINISH)", "fsm", exit=(1, 0.5), entry=(0, 0.5), points=[(1060, 200)])
    f.e("s_drain", "s_idle", "drain_done:\nattempt ← 0, rst_for_finish ← 1", "fsm", exit=(0.5, 1), entry=(0.5, 1), points=[(1175, 330), (715, 330)])
    f.e("s_exec", "s_idle", "is_finish & drain_done\n(children already idle)", "fsm", exit=(0.5, 1), entry=(0.7, 1), points=[(945, 300), (737, 300)])
    # attempt reg & outputs
    f.n("att", "attempt register (1 bit)\nSTART → 0 · abort flips (ORIGINATE) /\nadopts ev_attempt (FORWARD) · FINISH → 0", 620, 430, 300, 70, "reg")
    f.n("outs", "combinational outputs\nreset = rst | abort_now | rst_for_finish      (LOCAL, never propagated)\ndata_att_eff = INTERNAL ? attempt : data_attempt\nset_in_data_to_zero = in_valid & ((data_att_eff != attempt) | abort_now)\nout_attempt = data_valid_output ? attempt : 0\ncur_attempt = attempt   (stamps the post-select fast path)\ndiscard = abort_now      (root → host: reset decoder / gap estimator)",
        620, 530, 420, 130, "comb")
    f.n("oev", "out_ev register (+1 cycle)\ndefault = FORWARD relay of ev_*\nORIGINATE overrides: START (att 0),\nABORT (~attempt), FINISH (attempt)", 1070, 430, 230, 90, "reg")
    f.n("tbl", "board             EVENT_MODE  DATA_SRC\ndistributed root  ORIGINATE   EXTERNAL\ndistributed mid   FORWARD     EXTERNAL\ndistributed leaf  FORWARD     INTERNAL\nmonolithic        ORIGINATE   INTERNAL",
        250, 360, 300, 80, "table")
    f.n("nt", "A parent always changes attempt one cycle before its children can, so incoming data whose attempt tag\nmismatches is simply stale and is dropped (set_in_data_to_zero) — no flush / purge window is needed.\nThe board-generated abort is registered (abort_det_q) to break the reset → clears-post_select → abort loop.",
        250, 690, 620, 60, "note")
    f.n("o_rst", "reset → DCB rst", 1350, 540, 150, 24, "port")
    f.n("o_gate", "set_in_data_to_zero", 1350, 580, 150, 24, "port")
    f.n("o_att", "out_attempt / cur_attempt", 1350, 620, 150, 24, "port")
    f.n("o_disc", "discard", 1350, 660, 150, 24, "port")
    f.n("o_ev", "out_ev_valid / type /\npayload / attempt", 1350, 450, 150, 36, "port")
    f.e("p_ev", "dec", "", "ctrl", exit=R, entry=L)
    f.e("dec", "fsmbox", "is_start / is_finish", "ctrl", exit=R, entry=(0, 0.2))
    f.e("dec", "abd", "is_abort_ev", "ctrl", exit=(0.5, 1), entry=(0.5, 0))
    f.e("p_ps", "abd", "", "ps", exit=R, entry=(0, 0.5))
    f.e("abd", "fsmbox", "abort_now", "ps", exit=R, entry=(0, 0.6))
    f.e("p_dd", "fsmbox", "drain_done", "plain", exit=R, entry=(0, 0.9), points=[(590, 572), (590, 370)])
    f.e("fsmbox", "att", "state, rst_for_finish", "plain", exit=(0.2, 1), entry=(0.5, 0))
    f.e("att", "abd", "attempt", "plain", exit=(0, 0.5), entry=(0.5, 1), points=[(400, 465)])
    f.e("att", "outs", "attempt", "plain", exit=(0.5, 1), entry=(0.5, 0))
    f.e("abd", "outs", "abort_now", "ps", exit=(0.8, 1), entry=(0, 0.4), points=[(490, 595)])
    f.e("p_iv", "outs", "", "plain", exit=R, entry=(0, 0.6), points=[(230, 432), (230, 608)])
    f.e("p_dvo", "outs", "", "plain", exit=R, entry=(0, 0.8), points=[(240, 632), (240, 634)])
    f.e("fsmbox", "oev", "ORIGINATE events", "ctrl", exit=(0.85, 1), entry=(0.5, 0))
    f.e("p_ev", "oev", "FORWARD relay (ev_* delayed one cycle)", "ctrl", exit=(0.5, 0), entry=(1, 0.3), points=[(110, 60), (1310, 60), (1310, 457)])
    f.e("oev", "o_ev", "", "ctrl", exit=(1, 0.6), entry=L)
    f.e("outs", "o_rst", "", "rst", exit=(1, 0.15), entry=L)
    f.e("outs", "o_gate", "", "rst", exit=(1, 0.4), entry=L)
    f.e("outs", "o_att", "", "plain", exit=(1, 0.65), entry=L)
    f.e("outs", "o_disc", "", "ctrl", exit=(1, 0.9), entry=L)
    return f


def fig09_physical_mmio():
    f = Fig("09_physical_mmio", "PhysicalMMIO — command-word generator for one physical control channel")
    f.n("ttl", "PhysicalMMIO — command-word generator for one physical control core (PhysicalMMIO.sv)\none per control core on a leaf (P per board); turns the board's events into a stream of command words; program = instruction regfile + CW memory",
        30, 10, 1250, 40, "title")
    f.n("mm", "PhysicalMMIO  #(INSTR_DEPTH, CW_DEPTH, DATA_W, WT_W, INSTR_FILE, CW_FILE)", 220, 70, 1060, 640, "container")
    f.n("p_ev", "ev_valid / ev_type / ev_payload\n(board event source)", 20, 130, 180, 36, "port")
    f.n("p_rst", "clk / rst  (system reset)", 20, 200, 180, 24, "port")
    f.n("seq", "InstrSequencer  (FSM)  →  10_instr_fsms\nIDLE / EXEC / DRAIN\nowns instruction pointer ptr (saturates at\nDEPTH−1 = the compiled WAIT instruction)\nregisters instr_en / instr_addr (no-stall handoff)\nrst_out = rst | is_abort | rst_for_finish", 250, 110, 330, 130, "fsm")
    f.n("irf", "Instruction regfile  (RegFileROM, async → LUTRAM)\nINSTR_DEPTH × INSTR_W  ·  MMIO_instr_<p>.mem", 700, 110, 330, 60, "mem")
    f.n("unp", "InstrUnpack  (pure combinational)\nsplits the packed word — the ONLY place\nthat knows the instruction encoding", 700, 210, 330, 70, "comb")
    f.n("w_wt", "wt  [WT_W]\nidle cycles before 1st read", 720, 300, 130, 40, "bits")
    f.n("w_st", "start_addr  [ADDR_W]\nfirst CW address", 850, 300, 110, 40, "bits")
    f.n("w_en", "end_addr  [ADDR_W]\nlast CW address", 960, 300, 100, 40, "bits")
    f.n("w_n", "INSTR_W = 2·ADDR_W + WT_W   (MSB … LSB)", 720, 345, 340, 20, "note")
    f.n("dec", "InstrDecode  (Mealy FSM)  →  10_instr_fsms\nIDLE / WAITC / STREAM\nwait wt idle cycles, then r_addr = start … end\n(one per cycle, r_en high); one_instr_end on\nthe cycle r_addr == end", 250, 320, 330, 110, "fsm")
    f.n("cw", "Command-word memory  (SyncROM, 1-cycle → BRAM)\nCW_DEPTH × DATA_W  ·  MMIO_cw_<p>.mem\nNO reset by design: the last word of a finishing\ninstruction is still delivered", 250, 500, 330, 90, "mem")
    f.n("nt", "Output contract: out_valid alone says 'there is a command this cycle'; when low, out_data = 0 and the core does nothing —\nidle cycles cost no CW memory.  Critical path (combinational, by design of the zero-stall handoff):\nInstrSequencer regs → instr_addr mux → instruction regfile (async) → InstrUnpack → InstrDecode → CW memory address.",
        640, 460, 620, 70, "note")
    f.n("nt2", "Abort handling: EV_ABORT restarts the program from instruction 0 (rst_out clears InstrDecode the same cycle);\nEV_FINISH lets the in-flight instruction drain, then cw_gen_finish pulses in the SAME cycle as the final out_valid.",
        640, 560, 620, 50, "note")
    f.n("o_d", "out_data [DATA_W]\nout_valid → physical control core", 1310, 520, 180, 36, "port")
    f.n("o_f", "cw_gen_finish\n→ board (1-cycle pulse)", 1310, 160, 180, 36, "port")
    f.e("p_ev", "seq", "", "ctrl", exit=R, entry=(0, 0.3))
    f.e("p_rst", "seq", "", "rst", exit=R, entry=(0, 0.75))
    f.e("seq", "irf", "instr_en,\ninstr_addr", "mem", exit=(1, 0.3), entry=(0, 0.5))
    f.e("irf", "unp", "instr_word [INSTR_W]", "mem", exit=(0.5, 1), entry=(0.5, 0))
    f.e("unp", "dec", "wt, start_addr, end_addr", "mem", exit=(0, 0.5), entry=(1, 0.35), points=[(640, 245), (640, 358)])
    f.e("seq", "dec", "en = instr_en", "ctrl", exit=(0.3, 1), entry=(0.3, 0))
    f.e("seq", "dec", "rst_out (single reset source)", "rst", exit=(0.75, 1), entry=(0.75, 0))
    f.e("dec", "seq", "one_instr_end", "plain", exit=(0, 0.5), entry=(0, 0.95), points=[(235, 375), (235, 233)])
    f.e("dec", "cw", "r_en, r_addr [ADDR_W]", "mem", exit=(0.5, 1), entry=(0.5, 0))
    f.e("cw", "o_d", "data / data_valid (1 cycle later)", "ctrl", exit=R, entry=L)
    f.e("seq", "o_f", "cw_gen_finish = rst_for_finish", "plain", exit=(0.5, 0), entry=L, points=[(415, 95), (1200, 95), (1200, 178)])
    return f


def fig10_instr_fsms():
    f = Fig("10_instr_fsms", "InstrSequencer and InstrDecode state machines")
    f.n("ttl", "PhysicalMMIO state machines — InstrSequencer.sv (instruction pointer + event FSM) and InstrDecode.sv (one instruction → CW read stream)",
        30, 10, 1250, 40, "title")
    # sequencer
    f.n("sq", "InstrSequencer  (all state registered; outputs: instr_en = en_q, instr_addr = en_q ? ptr : 0, rst_out = rst | is_abort | rst_fin, cw_gen_finish = rst_fin)",
        30, 70, 1260, 370, "container")
    f.n("q_idle", "IDLE\nptr = 0, en = 0", 80, 200, 130, 60, "state")
    f.n("q_exec", "EXEC\nfetch / run", 560, 200, 130, 60, "state")
    f.n("q_drain", "DRAIN\nno new fetch", 1050, 200, 130, 60, "state")
    f.e("q_idle", "q_exec", "is_start: ptr ← 0, en ← 1\n(fetch instruction 0 next cycle)", "fsm", exit=(1, 0.5), entry=(0, 0.5), points=[(385, 185)])
    f.e("q_exec", "q_exec", "one_instr_end: ptr ← min(ptr+1, DEPTH−1), en ← 1   (no-stall: next instr fetched next cycle)\nelse: en ← 0, ptr holds", "fsm", exit=(0.25, 0), entry=(0.75, 0), points=[(575, 120), (675, 120)])
    f.e("q_exec", "q_exec", "is_abort: ptr ← 0, en ← 1, rst_out high THIS cycle\n(InstrDecode cleared before the refetch)", "fsm", exit=(0.25, 1), entry=(0.75, 1), points=[(575, 365), (675, 365)])
    f.e("q_exec", "q_drain", "is_finish & ~one_instr_end:\ninstruction mid-flight → let it finish", "fsm", exit=(1, 0.5), entry=(0, 0.5), points=[(870, 185)])
    f.e("q_drain", "q_idle", "one_instr_end: rst_fin ← 1  (= cw_gen_finish pulse)", "fsm", exit=(0.5, 1), entry=(0.5, 1), points=[(1115, 300), (145, 300)])
    f.e("q_exec", "q_idle", "is_finish & one_instr_end (same cycle): rst_fin ← 1", "fsm", exit=(0.5, 1), entry=(0.85, 1), points=[(625, 290), (190, 290)])
    f.n("sq_n", "WAIT behaviour: ptr saturates at DEPTH−1, so the last (compiled 'wait' = one syndrome-extraction round) instruction repeats until FINISH.",
        80, 405, 1100, 24, "note")
    # decoder
    f.n("dc", "InstrDecode  (Mealy: r_en / r_addr / one_instr_end combinational so CW reads stay contiguous: end_i at T, start_{i+1} at T+1; state registered, async rst = rst_out)",
        30, 470, 1260, 330, "container")
    f.n("d_idle", "IDLE", 80, 600, 130, 60, "state")
    f.n("d_wait", "WAITC\nwcnt idle cycles", 560, 600, 130, 60, "state")
    f.n("d_str", "STREAM\nr_addr walks", 1050, 600, 130, 60, "state")
    f.e("d_idle", "d_wait", "en & wt ≠ 0: latch start/end,\nwcnt ← wt − 1", "fsm", exit=(1, 0.5), entry=(0, 0.5), points=[(385, 585)])
    f.e("d_idle", "d_str", "en & wt == 0 & start ≠ end:  r_en = 1, r_addr = start THIS cycle (fast path); addr ← start + 1", "fsm", exit=(0.5, 0), entry=(0.5, 0), points=[(145, 520), (1115, 520)])
    f.e("d_idle", "d_idle", "en & wt == 0 & start == end:\nsingle read, one_instr_end = 1", "fsm", exit=(0.25, 1), entry=(0.75, 1), points=[(95, 740), (195, 740)])
    f.e("d_wait", "d_wait", "wcnt ≠ 0: wcnt ← wcnt − 1", "fsm", exit=(0.25, 1), entry=(0.75, 1), points=[(575, 740), (675, 740)])
    f.e("d_wait", "d_str", "wcnt == 0: r_en = 1, r_addr = start;\naddr ← start + 1  (or IDLE if start == end)", "fsm", exit=(1, 0.5), entry=(0, 0.5), points=[(870, 585)])
    f.e("d_str", "d_str", "addr ≠ end: r_en = 1, r_addr = addr, addr ← addr + 1", "fsm", exit=(0.25, 1), entry=(0.75, 1), points=[(1065, 740), (1165, 740)])
    f.e("d_str", "d_idle", "addr == end: r_en = 1, r_addr = end, one_instr_end = 1  (→ sequencer registers the next instruction)", "fsm", exit=(0.5, 1), entry=(0.5, 1), points=[(1115, 780), (145, 780)])
    return f


def fig11_stim_readout():
    f = Fig("11_stim_readout_loop", "StimReadout — leaf measurement loopback for whole-system stim tests")
    f.n("ttl", "StimReadout — synthesizable measurement-readout stand-in for hardware-in-the-loop stim tests (StimReadout.sv)\nreplays RECORDED measurements from BRAM back into a leaf ControlBoard; closes the mmio_out → readout → in_meas loop",
        30, 10, 1250, 40, "title")
    f.n("leaf", "Leaf ControlBoard  (FORWARD · INTERNAL)", 60, 90, 420, 420, "container")
    f.n("lbc", "BoardControl", 90, 160, 160, 50, "fsm")
    f.n("lmm", "PhysicalMMIO  (P = 1)\nout_data = ROUND INDEX\n(sim CW type: entry = its index)", 90, 250, 200, 80)
    f.n("ldcb", "DetectorConstructBlock\nin_meas / in_valid / in_meas_finish", 90, 370, 260, 70)
    f.n("lo", "fwd_det / fwd_raw … → parent", 300, 115, 160, 36, "port")
    f.n("sr", "StimReadout  #(M, SHOTS, DEPTH, MEAS_FILE, VALID_FILE)  — GLOBAL rst + cw_gen_finish, never board_reset",
        600, 90, 760, 420, "container")
    f.n("shot", "shot_q / base_q registers\nreset | cw_gen_finish : shot ← 0, base ← 0\nEV_ABORT (guarded shot+1 < SHOTS):\n   shot ← shot+1, base ← (shot+1) × DEPTH\n(pointer moves in LOCKSTEP across all leaves —\n the abort is global and in sync)", 630, 140, 300, 120, "reg")
    f.n("addr", "addr = base_q + mmio_out_data\nrom_addr = active ? addr : 0\nactive = mmio_out_valid & ~rst", 960, 150, 260, 60, "comb")
    f.n("romm", "SyncROM  meas  (BRAM)\nSHOTS × DEPTH rows × M bits\nshot-major · sim_readout_meas.mem", 960, 250, 180, 60, "mem")
    f.n("romv", "SyncROM  valid  (BRAM)\nsame layout\nsim_readout_valid.mem", 1160, 250, 180, 60, "mem")
    f.n("fin", "finish_q ← mmio_out_valid & cw_gen_finish\n(aligned with the 1-cycle BRAM read)\nout_meas_finish = finish_q ? valid_data : 0", 630, 300, 300, 60, "reg")
    f.n("nt", "The ABORT that advances the readout is the SAME event that restarts the CW program in BoardControl / PhysicalMMIO, so a rejected shot is\ndiscarded everywhere on the same cycle. Multi-shot / multi-trial tests (test_control_board_stim, tb_ControlBoard) drive only the root's start / finish / gap_post_select.",
        60, 540, 1300, 50, "note")
    f.e("lbc", "lmm", "ev_* (relayed)", "ctrl", exit=(0.5, 1), entry=(0.5, 0))
    f.e("ldcb", "lo", "", "bus", exit=(0.9, 0), entry=(0.5, 1), points=[(324, 300), (380, 300)])
    f.e("lbc", "shot", "ev_valid & ev_type == ABORT", "ctrl", exit=(1, 0.5), entry=(0, 0.4))
    f.e("lmm", "addr", "mmio_out_data (round index), mmio_out_valid", "ctrl", exit=(1, 0.5), entry=(0, 0.5), points=[(945, 290), (945, 180)])
    f.e("lmm", "fin", "cw_gen_finish", "plain", exit=(1, 0.85), entry=(0, 0.5))
    f.e("shot", "addr", "base_q", "plain", exit=(1, 0.3), entry=(0, 0.3))
    f.e("addr", "romm", "en = active, addr", "mem", exit=(0.5, 1), entry=(0.5, 0))
    f.e("romm", "romv", "", "mem", exit=R, entry=L)
    f.e("romm", "ldcb", "out_meas = meas_data\nout_meas_valid = valid_data\nout_meas_finish", "bus", exit=(0.5, 1), entry=(0.5, 1), points=[(1050, 470), (220, 470)])
    f.e("fin", "ldcb", "", "plain", exit=(0.5, 1), entry=(0.3, 1), points=[(780, 455), (168, 455)])
    return f


def fig12_memories():
    f = Fig("12_memory_primitives", "Memory primitives: RegFileROM, SyncROM, BankedRAM")
    f.n("ttl", "Memory primitives (rtl/src/lib, emulator/include/lib) — every compiled program is a .mem file loaded by $readmemb / $readmemh\nfile format: one word per line, MSB-first, LSB = bit 0 — parsed identically by the SystemC load_mem_file() and by the RTL, baked into LUTRAM / BRAM INIT strings at synthesis",
        30, 10, 1300, 40, "title")
    # RegFileROM
    f.n("rf", "RegFileROM  #(DEPTH, WIDTH, INIT_FILE, RAM_STYLE)  — async read → LUTRAM / distributed RAM", 30, 80, 620, 200, "container")
    f.n("rf_a", "addr [⌈log2 DEPTH⌉]", 50, 150, 150, 24, "port")
    f.n("rf_m", "mem [DEPTH] × WIDTH\ndata = mem[addr]  (combinational, no clock, no enable)", 220, 130, 290, 60, "mem")
    f.n("rf_d", "data [WIDTH]\n(same cycle)", 530, 143, 110, 36, "port")
    f.n("rf_n", "Used for the small compiler regfiles read at a registered pc and consumed in the SAME cycle:\nsync masks, kernel cores, output_sync, global_index, round_marker, postselect, MMIO instruction regfile.",
        50, 210, 590, 50, "note")
    f.e("rf_a", "rf_m", "", "mem")
    f.e("rf_m", "rf_d", "", "mem")
    # SyncROM
    f.n("sr", "SyncROM  #(DEPTH, WIDTH, INIT_FILE, RAM_STYLE)  — sync read, 1-cycle latency → block RAM", 690, 80, 640, 200, "container")
    f.n("sr_a", "en, addr", 710, 150, 110, 24, "port")
    f.n("sr_m", "BRAM read stage (pure, holds when !en)\nif (en) mem_q ← mem[addr];  vld_q ← en\nNO reset (models the BRAM output register)", 840, 120, 300, 70, "mem")
    f.n("sr_x", "fabric mux\ndata = vld_q ? mem_q : 0", 840, 200, 160, 40, "comb")
    f.n("sr_d", "data [WIDTH],\ndata_valid  (+1 cycle)", 1180, 150, 130, 36, "port")
    f.n("sr_n", "Used for large memories: the MMIO command-word memory\nand StimReadout's recorded measurements.", 1010, 235, 310, 40, "note")
    f.e("sr_a", "sr_m", "", "mem")
    f.e("sr_m", "sr_x", "", "mem", exit=(0.3, 1), entry=(0.5, 0))
    f.e("sr_x", "sr_d", "", "mem", exit=R, entry=(0.5, 1), points=[(1245, 220)], )
    f.e("sr_m", "sr_d", "data_valid", "plain", exit=R, entry=(0, 0.5))
    # BankedRAM
    f.n("br", "BankedRAM  #(BANKS, DEPTH, WIDTH, RAM_STYLE)  — banked simple-dual-port RAM, synchronous read, READ_FIRST  (the PGE's M_T column store)", 30, 320, 1300, 300, "container")
    f.n("br_i", "index → bank = index[log2 BANKS−1:0]\n            offset = index[INDEX_W−1 : log2 BANKS]\n(decode done by the CALLER: the compiler's relabelling\n π(c) = offset·BANKS + bank makes the split fixed wiring)", 50, 380, 330, 90, "comb")
    for b, x in enumerate((420, 640, 860)):
        lab = f"bank {b if b < 2 else 'BANKS−1'}\nmem [DEPTH] × WIDTH\nrd_en / rd_addr → rd_data (+1 cycle;\n0 when rd_en was low)\nwr_en / wr_addr / wr_data"
        f.n(f"br_b{b}", lab, x, 370, 200, 110, "mem")
    f.n("br_dots", "…", 845, 415, 20, 20, "note")
    f.n("br_o", "rd_data [BANKS × WIDTH]\n→ consumer XOR tree takes all banks\nunconditionally (rd_en is the mask)", 1090, 380, 220, 70, "port")
    f.n("br_n", "Banking rule (enforced by the PGE compiler): for every fault at most one of its checks lands in any one bank, so all of a fault's columns\nare readable in a single cycle. One read + one write port per bank. Async reset zeroes every bank (note: prevents BRAM inference — revisit for a real target).",
        50, 500, 1260, 60, "note")
    f.e("br_i", "br_b0", "per-bank offset", "mem", exit=R, entry=(0, 0.5))
    f.e("br_b0", "br_b1", "", "mem", noarrow=True)
    f.e("br_b2", "br_o", "", "mem", exit=R, entry=L)
    return f


FIGS = [fig00_overview, fig01_toolchain, fig02_control_board, fig03_dcb, fig04_kernel, fig05_sync,
        fig06_root_output_sync, fig07_small_blocks, fig08_board_control, fig09_physical_mmio,
        fig10_instr_fsms, fig11_stim_readout, fig12_memories]


def main():
    os.makedirs(os.path.join(HERE, "drawio"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "svg"), exist_ok=True)
    pages = []
    for i, mk in enumerate(FIGS):
        fig = mk()
        page = diagram_xml(fig, f"page{i}")
        pages.append(page)
        with open(os.path.join(HERE, "drawio", fig.name + ".drawio"), "w") as fh:
            fh.write(mxfile([page]))
        with open(os.path.join(HERE, "svg", fig.name + ".svg"), "w") as fh:
            fh.write(svg(fig))
        print(f"  {fig.name}: {len(fig.nodes)} nodes, {len(fig.edges)} edges")
    with open(os.path.join(HERE, "control_system_all.drawio"), "w") as fh:
        fh.write(mxfile(pages))
    print(f"wrote {len(pages)} figures -> {HERE}/drawio, {HERE}/svg, control_system_all.drawio")


if __name__ == "__main__":
    main()
