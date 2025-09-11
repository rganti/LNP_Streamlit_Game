#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit App: Molecule Merge — LNP Lab (Chemistry Puzzle Game)

Run:
  pip install streamlit
  streamlit run lnp_streamlit_game.py

Summary:
  Assemble an LNP from 4 component types (IH, HL, CH, PG) to meet level targets
  for Stability, Potency, Toxicity, and sometimes Size.
  The "AI predictor" is a stylized, transparent function with a touch of noise.

Notes:
  - This is edutainment with plausible trends, not a biophysical simulator.
  - Use "Hint" to search your current inventory for promising combos.
  - Levels get stricter; you have limited attempts per level.
"""

import math
import random
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
import streamlit as st
import itertools

# ML advisory (Gaussian Process) for Smart Hints
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel


random.seed(7)  # deterministic runs for the session


# ----------------------------
# Data Structures
# ----------------------------

TYPES = ("IH", "HL", "CH", "PG")  # Ionizable Head, Helper Lipid, Cholesterol, PEG-Lipid


@dataclass
class Component:
    kind: str  # "IH" | "HL" | "CH" | "PG"
    name: str
    pKa: Optional[float] = None  # for IH
    tail_len: Optional[int] = None  # for IH/HL
    unsat: Optional[int] = None  # for IH/HL
    helper_strength: Optional[float] = None  # for HL
    chol_frac: Optional[float] = None  # for CH
    peg_mw: Optional[int] = None  # for PG
    peg_mol_frac: Optional[float] = None  # for PG
    note: str = ""

    def label(self) -> str:
        extra = []
        if self.kind == "IH":
            extra.append(f"pKa {self.pKa:.2f}")
            # extra.append(f"C{self.tail_len}/unsat {self.unsat}")
        elif self.kind == "HL":
            extra.append(f"helper {self.helper_strength:.2f}")
            # extra.append(f"C{self.tail_len}/unsat {self.unsat}")
        elif self.kind == "CH":
            extra.append(f"chol {self.chol_frac:.2f}")
        elif self.kind == "PG":
            # extra.append("")
            extra.append(f"frac {self.peg_mol_frac:.3f}")
        return f"{self.name} — " + " | ".join(extra)


@dataclass
class LNPDesign:
    IH: Component
    HL: Component
    CH: Component
    PG: Component


@dataclass
class LNPProps:
    size_nm: float
    stability: float  # 0-100 (higher is better)
    potency: float  # 0-100 (higher is better)
    toxicity: float  # 0-100 (lower is better)


# ----------------------------
# Component Libraries
# ----------------------------

IONIZABLE_HEADS = [
    Component(
        "IH", "Ionia", pKa=6.2, tail_len=16, unsat=1, note="Lean acidic; mid-tail"
    ),
    Component(
        "IH", "Voltron", pKa=6.5, tail_len=18, unsat=1, note="Sweet spot pKa ~6.5"
    ),
    Component(
        "IH", "Sparklite", pKa=6.8, tail_len=14, unsat=0, note="Basic edge; saturated"
    ),
    Component(
        "IH",
        "Barbie",
        pKa=7.0,
        tail_len=18,
        unsat=2,
        note="More basic; di-unsaturated",
    ),
    Component(
        "IH",
        "Goldilocks",
        pKa=6.4,
        tail_len=16,
        unsat=0,
        note="Near-optimal pKa; saturated",
    ),
]

HELPER_LIPIDS = [
    Component(
        "HL",
        "Tempur Pedic",
        helper_strength=0.40,
        tail_len=18,
        unsat=0,
        note="Feeling firm",
    ),
    Component(
        "HL",
        "Beauty Rest Black",
        helper_strength=0.85,
        tail_len=18,
        unsat=2,
        note="Feeling soft",
    ),
    Component(
        "HL",
        "Casper",
        helper_strength=0.75,
        tail_len=18,
        unsat=2,
        note="Feeling cool",
    ),
    Component(
        "HL",
        "Sleep Number",
        helper_strength=0.50,
        tail_len=14,
        unsat=0,
        note="Feeling comfy",
    ),
]

CHOLESTEROLS = [
    Component("CH", "Thor", chol_frac=0.10, note="Less rigid; low stability"),
    Component("CH", "Captain America", chol_frac=0.20, note="Baseline stability"),
    Component("CH", "Iron Man", chol_frac=0.30, note="More rigid; high stability"),
]

PEGS = [
    Component("PG", "FBI", peg_mw=1000, peg_mol_frac=0.015, note="Short stealth"),
    Component("PG", "CIA", peg_mw=2000, peg_mol_frac=0.020, note="Balanced stealth"),
    Component("PG", "NSA", peg_mw=5000, peg_mol_frac=0.030, note="Long stealth"),
    Component("PG", "Area51", peg_mw=550, peg_mol_frac=0.010, note="Very short"),
]


# ----------------------------
# Predictor (stylized "AI" oracle)
# ----------------------------


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def gaussian(x: float, mu: float, sigma: float) -> float:
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2)


def predict_properties(design: LNPDesign, noise: float = 0.02) -> LNPProps:
    ih, hl, ch, pg = design.IH, design.HL, design.CH, design.PG

    # Size (nm)
    base_size = 60
    size_from_tails = 0.6 * ((ih.tail_len or 16) + (hl.tail_len or 16) - 30)
    size_from_peg = 0.05 * (pg.peg_mw or 2000) ** 0.5
    size_jitter = random.uniform(-4, 4)
    size_nm = clamp(base_size + size_from_tails + size_from_peg + size_jitter, 40, 140)

    # Stability (0-100): ↑ with cholesterol & saturation & PEG fraction
    chol_term = 220 * (ch.chol_frac or 0.2)
    sat_term = 8 * (2 - (ih.unsat or 0)) + 8 * (2 - (hl.unsat or 0))
    peg_term = 900 * (pg.peg_mol_frac or 0.02)
    stability_raw = chol_term + sat_term + peg_term
    stability = clamp(stability_raw + random.gauss(0, noise * 100), 0, 100)

    # Potency (0-100): bell around pKa~6.6; helper strength; size near 82 nm
    pka_score = 100 * max(gaussian((ih.pKa or 6.5), 6.6, 0.25), 0.2)
    helper_score = 60 * (hl.helper_strength or 0.5)
    size_opt = 100 * gaussian(size_nm, 82, 12)
    potency_raw = 0.45 * pka_score + 0.35 * helper_score + 0.30 * size_opt
    potency = clamp(potency_raw + random.gauss(0, noise * 100), 0, 100)

    # Toxicity (0-100, lower is better): ↑ with basic pKa, low PEG, tiny size
    tox_from_pka = 50 * max(0, (ih.pKa or 6.5) - 6.6)
    tox_from_peg = 40 * max(0.03 - (pg.peg_mol_frac or 0.02), 0)
    tox_from_size = 30 * max(0, (75 - size_nm) / 35)
    unsat_buffer = -5 * ((ih.unsat or 0) + (hl.unsat or 0))
    toxicity_raw = 25 + tox_from_pka + tox_from_peg + tox_from_size + unsat_buffer
    toxicity = clamp(toxicity_raw + random.gauss(0, noise * 100), 0, 100)

    return LNPProps(
        size_nm=size_nm, stability=stability, potency=potency, toxicity=toxicity
    )


# ----------------------------
# Levels
# ----------------------------

LEVELS = [
    {
        "name": "Level 1 — Get It Stable",
        "targets": {"stability_min": 70},
        "attempts": 6,
    },
    {
        "name": "Level 2 — Power Without Harm",
        "targets": {"potency_min": 65, "toxicity_max": 30},
        "attempts": 7,
    },
    {
        "name": "Level 3 — Thread the Needle",
        "targets": {"potency_min": 70, "stability_min": 70, "toxicity_max": 28},
        "attempts": 8,
    },
    {
        "name": "Level 4 — Tight Size Window",
        "targets": {
            "potency_min": 68,
            "stability_min": 72,
            "toxicity_max": 28,
            "size_range": (65, 90),
        },
        "attempts": 9,
    },
]


# ----------------------------
# Helpers
# ----------------------------


def describe_targets(t: Dict[str, object]) -> str:
    parts = []
    if "stability_min" in t:
        parts.append(f"Stability ≥ {t['stability_min']}")
    if "potency_min" in t:
        parts.append(f"Potency ≥ {t['potency_min']}")
    if "toxicity_max" in t:
        parts.append(f"Toxicity ≤ {t['toxicity_max']}")
    if "size_range" in t:
        lo, hi = t["size_range"]
        parts.append(f"Size in [{lo}, {hi}] nm")
    return " AND ".join(parts)


def meets_targets(props: LNPProps, targets: Dict[str, object]) -> bool:
    ok = True
    if "stability_min" in targets:
        ok &= props.stability >= targets["stability_min"]
    if "potency_min" in targets:
        ok &= props.potency >= targets["potency_min"]
    if "toxicity_max" in targets:
        ok &= props.toxicity <= targets["toxicity_max"]
    if "size_range" in targets:
        lo, hi = targets["size_range"]
        ok &= lo <= props.size_nm <= hi
    return bool(ok)


def init_state():
    if "level_idx" not in st.session_state:
        st.session_state.level_idx = 0
    if "attempts_left" not in st.session_state:
        st.session_state.attempts_left = LEVELS[0]["attempts"]
    if "history" not in st.session_state:
        st.session_state.history = []  # list of dicts: {design, props, success}
    if "inventory" not in st.session_state:
        st.session_state.inventory = {
            "IH": IONIZABLE_HEADS[:],
            "HL": HELPER_LIPIDS[:],
            "CH": CHOLESTEROLS[:],
            "PG": PEGS[:],
        }


def reset_level(idx: int):
    st.session_state.level_idx = idx
    st.session_state.attempts_left = LEVELS[idx]["attempts"]
    st.session_state.history = []


def next_level():
    if st.session_state.level_idx + 1 < len(LEVELS):
        reset_level(st.session_state.level_idx + 1)
    else:
        st.session_state.game_won = True


def quick_hint(
    inventory: Dict[str, List[Component]],
    targets: Dict[str, object],
    samples: int = 300,
):
    ihs, hls, chs, pgs = (
        inventory["IH"],
        inventory["HL"],
        inventory["CH"],
        inventory["PG"],
    )
    if not (ihs and hls and chs and pgs):
        return []

    def score(p: LNPProps) -> float:
        s = 0.0
        if "stability_min" in targets:
            s += min(1.0, max(0.0, (p.stability - targets["stability_min"]) / 20))
        if "potency_min" in targets:
            s += min(1.0, max(0.0, (p.potency - targets["potency_min"]) / 20))
        if "toxicity_max" in targets:
            s += min(1.0, max(0.0, (targets["toxicity_max"] - p.toxicity) / 15))
        if "size_range" in targets:
            lo, hi = targets["size_range"]
            inside = (
                1.0
                if (lo <= p.size_nm <= hi)
                else 1.0
                - min(abs((p.size_nm - (lo + hi) / 2)) / ((hi - lo) / 2 + 1e-9), 1.0)
            )
            s += 0.7 * inside
        return s

    best = []
    for _ in range(samples):
        d = LNPDesign(
            IH=random.choice(ihs),
            HL=random.choice(hls),
            CH=random.choice(chs),
            PG=random.choice(pgs),
        )
        props = predict_properties(d)
        sc = score(props)
        best.append((sc, d, props))
    best.sort(key=lambda x: x[0], reverse=True)
    return best[:3]


# ----------------------------
# ML Advisor (Smart Hints)
# ----------------------------


def encode_component(c: Component) -> np.ndarray:
    # Numerical features only
    # order: [IH_pKa, IH_tail, IH_unsat, HL_strength, HL_tail, HL_unsat, CH_frac, PG_mw, PG_frac]
    if c.kind == "IH":
        return np.array(
            [c.pKa or 0, c.tail_len or 0, c.unsat or 0, 0, 0, 0, 0, 0, 0], dtype=float
        )
    if c.kind == "HL":
        return np.array(
            [0, 0, 0, c.helper_strength or 0, c.tail_len or 0, c.unsat or 0, 0, 0, 0],
            dtype=float,
        )
    if c.kind == "CH":
        return np.array([0, 0, 0, 0, 0, 0, c.chol_frac or 0, 0, 0], dtype=float)
    if c.kind == "PG":
        return np.array(
            [0, 0, 0, 0, 0, 0, 0, c.peg_mw or 0, c.peg_mol_frac or 0], dtype=float
        )
    return np.zeros(9, dtype=float)


def encode_design(d: LNPDesign) -> np.ndarray:
    return (
        encode_component(d.IH)
        + encode_component(d.HL)
        + encode_component(d.CH)
        + encode_component(d.PG)
    )


def target_score(props: LNPProps, targets: Dict[str, object]) -> float:
    s = 0.0
    if "stability_min" in targets:
        s += min(1.0, max(0.0, (props.stability - targets["stability_min"]) / 20))
    if "potency_min" in targets:
        s += min(1.0, max(0.0, (props.potency - targets["potency_min"]) / 20))
    if "toxicity_max" in targets:
        s += min(1.0, max(0.0, (targets["toxicity_max"] - props.toxicity) / 15))
    if "size_range" in targets:
        lo, hi = targets["size_range"]
        inside = (
            1.0
            if (lo <= props.size_nm <= hi)
            else 1.0
            - min(abs((props.size_nm - (lo + hi) / 2)) / ((hi - lo) / 2 + 1e-9), 1.0)
        )
        s += inside
    return s  # higher is better


def all_combos_from_inventory(inv: Dict[str, List[Component]]):
    return itertools.product(inv["IH"], inv["HL"], inv["CH"], inv["PG"])


def smart_hints_ML(
    inventory: Dict[str, List[Component]],
    targets: Dict[str, object],
    beta: float = 1.0,
    topk: int = 3,
):
    hist = st.session_state.get("history", [])
    if len(hist) < 2:
        return []

    # Training data from history
    X, y = [], []
    for h in hist:
        d: LNPDesign = h["design"]
        p: LNPProps = h["props"]
        X.append(encode_design(d))
        y.append(target_score(p, targets))
    X, y = np.vstack(X), np.asarray(y)

    # Fit a light GP
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
        length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-2, 1e3)
    ) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
    gp = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=0
    )
    gp.fit(X, y)

    # Score all current inventory combos by UCB
    candidates = []
    for ih, hl, ch, pg in all_combos_from_inventory(inventory):
        d = LNPDesign(IH=ih, HL=hl, CH=ch, PG=pg)
        x = encode_design(d).reshape(1, -1)
        mean, std = gp.predict(x, return_std=True)
        ucb = float(mean + beta * std)

        # For display, compute oracle properties
        props = predict_properties(d)
        candidates.append((ucb, d, props))

    # # Diversity boost vs last attempt
    # last_vec = encode_design(hist[-1]["design"])
    # last_norm = np.linalg.norm(last_vec) + 1e-9
    # diversified = []
    # for ucb, d, p in candidates:
    #     vec = encode_design(d)
    #     cos_sim = float(
    #         np.dot(vec, last_vec) / (np.linalg.norm(vec) * last_norm + 1e-9)
    #     )
    #     div = 0.15 * (1.0 - cos_sim)  # prefer different from last try
    #     diversified.append((ucb + div, d, p))

    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[:topk]


# ----------------------------
# UI
# ----------------------------

st.set_page_config(page_title="Molecule Merge — LNP Lab", page_icon="🧪", layout="wide")
st.title("🧪 Molecule Merge — LNP Lab")
st.caption(
    "Assemble an LNP from four components to hit the targets. Edutainment, not a simulator."
)

init_state()

if "game_won" in st.session_state and st.session_state.game_won:
    st.success("🏆 You cleared all levels! Want to replay Level 1?")
    if st.button("Replay from Level 1"):
        st.session_state.pop("game_won")
        reset_level(0)
        st.session_state.pop("celebrated", None)
        st.session_state.pop("just_won", None)
        st.session_state["history"] = []  # or level/global history resets as needed
        for k in ("sel_ih", "sel_hl", "sel_ch", "sel_pg"):
            st.session_state.pop(k, None)
        st.rerun()

level = LEVELS[st.session_state.level_idx]
targets = level["targets"]

left, right = st.columns([2, 1])

with left:
    st.subheader(level["name"])
    st.markdown(f"**Targets:** {describe_targets(targets)}")
    st.markdown(f"**Attempts left:** {st.session_state.attempts_left}")

    # Selection widgets
    with st.form("design_form"):
        col1, col2 = st.columns(2)
        with col1:
            ih = st.selectbox(
                "Ionizable Head (IH)",
                st.session_state.inventory["IH"],
                format_func=lambda x: x.label(),
                key="sel_ih",
            )
            ch = st.selectbox(
                "Cholesterol (CH)",
                st.session_state.inventory["CH"],
                format_func=lambda x: x.label(),
                key="sel_ch",
            )
        with col2:
            hl = st.selectbox(
                "Helper Lipid (HL)",
                st.session_state.inventory["HL"],
                format_func=lambda x: x.label(),
                key="sel_hl",
            )
            pg = st.selectbox(
                "PEG-Lipid (PG)",
                st.session_state.inventory["PG"],
                format_func=lambda x: x.label(),
                key="sel_pg",
            )

        submitted = st.form_submit_button("Evaluate Design ✅")

    if submitted and st.session_state.attempts_left > 0:
        design = LNPDesign(IH=ih, HL=hl, CH=ch, PG=pg)
        props = predict_properties(design)
        success = meets_targets(props, targets)
        st.session_state.history.append(
            {"design": design, "props": props, "success": success}
        )
        st.session_state.attempts_left -= 1

    # Show last result
    if st.session_state.history:
        last = st.session_state.history[-1]
        props = last["props"]
        st.markdown("### Latest Evaluation")
        st.metric("Size (nm)", f"{props.size_nm:.1f}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Stability", f"{props.stability:.1f}")
        m2.metric("Potency", f"{props.potency:.1f}")
        m3.metric("Toxicity (lower better)", f"{props.toxicity:.1f}")

        if last["success"]:
            if not st.session_state.get("celebrated", False):
                st.balloons()
                st.session_state.celebrated = True

            st.success("🎉 Targets met! Advance to the next level.")
            if st.button("Next Level ▶️"):
                next_level()
                st.session_state.celebrated = False
                st.rerun()
        else:
            st.info("Keep iterating or try a hint.")

    # Attempts exhausted?
    if st.session_state.attempts_left == 0 and (
        not st.session_state.history or not st.session_state.history[-1]["success"]
    ):
        st.error("Out of attempts! You can retry this level.")
        if st.button("Retry Level 🔁"):
            reset_level(st.session_state.level_idx)
            st.rerun()

with right:
    # st.subheader("🧠 Smart Hints (learned from your attempts)")
    # disabled = len(st.session_state.history) < 2
    # if st.button(
    #     "Smart Hints (ML) 🔮",
    #     disabled=disabled,
    #     help="Learns from your history; enabled after a few attempts.",
    # ):
    #     suggestions = smart_hints_ML(
    #         st.session_state.inventory, targets, beta=1.0, topk=3
    #     )
    #     if suggestions:
    #         for i, (score, d, p) in enumerate(suggestions, 1):
    #             with st.container(border=True):
    #                 st.markdown(f"**ML Hint #{i} — score {score:.2f}**")
    #                 st.markdown(
    #                     f"- IH: `{d.IH.name}`  \n- HL: `{d.HL.name}`  \n- CH: `{d.CH.name}`  \n- PG: `{d.PG.name}`"
    #                 )
    #                 st.caption(
    #                     f"Size {p.size_nm:.1f} nm — Stability {p.stability:.1f} | Potency {p.potency:.1f} | Toxicity {p.toxicity:.1f}"
    #                 )
    #     else:
    #         st.caption("Not enough history yet. Make a few attempts first.")

    # st.subheader("🧰 Tools & Hints")
    # if st.button("Suggest Combos (Hint) 💡"):
    #     suggestions = quick_hint(st.session_state.inventory, targets, samples=400)
    #     if suggestions:
    #         for i, (score, d, p) in enumerate(suggestions, 1):
    #             with st.container(border=True):
    #                 st.markdown(f"**Hint #{i} — score {score:.2f}**")
    #                 st.markdown(f"- IH: `{d.IH.name}`  \n- HL: `{d.HL.name}`  \n- CH: `{d.CH.name}`  \n- PG: `{d.PG.name}`")
    #                 st.caption(f"Size {p.size_nm:.1f} nm — Stability {p.stability:.1f} | Potency {p.potency:.1f} | Toxicity {p.toxicity:.1f}")
    #     else:
    #         st.write("Inventory incomplete for hints.")

    # st.divider()

    st.subheader("📜 History")
    if st.session_state.history:
        for i, h in enumerate(reversed(st.session_state.history), 1):
            d, p, ok = h["design"], h["props"], h["success"]
            with st.container(border=True):
                st.markdown(
                    f"**Attempt {-i + len(st.session_state.history)}** — {'✅ Success' if ok else '❌ Miss'}"
                )
                st.markdown(
                    f"- IH: `{d.IH.name}` | HL: `{d.HL.name}` | CH: `{d.CH.name}` | PG: `{d.PG.name}`"
                )
                st.caption(
                    f"Size {p.size_nm:.1f} nm — Stability {p.stability:.1f} | Potency {p.potency:.1f} | Toxicity {p.toxicity:.1f}"
                )
    else:
        st.caption("No attempts yet. Build and evaluate a design!")

with st.sidebar:
    st.header("How to Play")
    st.write(
        "Pick one component from each category (IH, HL, CH, PG). "
        "Click **Evaluate Design** to compute properties. "
        "Meet all targets before you run out of attempts to advance."
    )
    st.write("**Trends (roughly):**")
    st.markdown(
        "- Stability ↑ with more cholesterol, more saturation, and PEG fraction.\n"
        "- Potency peaks near IH pKa ~6.6, strong helper lipid, and size ~82 nm.\n"
        "- Toxicity ↑ if IH pKa is too basic (>6.6), very low PEG fraction, or very small size."
    )
    st.write("This is a simplified learning game, not a validated model.")
