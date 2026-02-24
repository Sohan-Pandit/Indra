"""
Stage 11 — Streamlit Demo App

One-page app where a user pastes an abstract and receives:
  1. Extracted JSON impact record (mock or live via Anthropic API)
  2. Hedge terms highlighted inline in the abstract
  3. MC Dropout uncertainty bar (if trained model is available)
  4. Mini interactive knowledge graph for the single abstract

Run with:
    streamlit run src/visualization/app.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

# Make project root importable regardless of launch directory
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.uncertainty.hedge_detector import detect_hedges, HEDGE_TERMS
from src.visualization.knowledge_graph import build_graph, render_html, graph_stats

# ── Page configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Indra",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Indra")
    st.caption("Climate Impact Structured Extractor")
    st.divider()

    st.subheader("Settings")
    use_llm = st.checkbox("Use live LLM extraction", value=False)
    api_key = ""
    if use_llm:
        api_key = st.text_input("Anthropic API Key", type="password")
        st.caption("Requires `anthropic` package and a valid key.")

    n_mc_passes = st.slider("MC Dropout passes", min_value=5, max_value=50, value=20, step=5)
    st.divider()

    st.subheader("About")
    st.markdown(
        """
**Indra** extracts structured climate impact records from scientific abstracts:

- Hazard type + intensity
- Location (normalized)
- Time period (ISO 8601)
- Impact domain + magnitude
- Causal relation triple
- Epistemic uncertainty
        """
    )

# ── Default example ───────────────────────────────────────────────────────────

_DEFAULT = (
    "The 2013 Uttarakhand floods, triggered by extreme rainfall exceeding 200 mm/day, "
    "resulted in over 5,700 deaths and displaced approximately 100,000 people. "
    "The floods caused widespread destruction of road infrastructure and may have "
    "long-term economic consequences for low-income households in the region. "
    "Observed mortality was highest in Rudraprayag district, India."
)

# ── Header ────────────────────────────────────────────────────────────────────

st.title("Indra: Climate Impact Extractor")
st.caption(
    "Paste a scientific abstract about a climate extreme event to extract a structured impact record."
)

# ── Input ─────────────────────────────────────────────────────────────────────

abstract_text = st.text_area(
    "Abstract",
    value=_DEFAULT,
    height=180,
    placeholder="Paste a climate impact abstract here…",
)

col_btn, col_hint = st.columns([1, 5])
with col_btn:
    run = st.button("Extract", type="primary", use_container_width=True)

# ── Mock record (used when LLM is off) ───────────────────────────────────────

_MOCK_RECORD = {
    "abstract_id": "DEMO_001",
    "hazard_type": "flood",
    "hazard_intensity": "extreme rainfall >200 mm/day",
    "location": {"raw": "Uttarakhand, India", "normalized": "Uttarakhand, IN"},
    "time_period": {
        "raw": "2013",
        "normalized": "2013-01-01/2013-12-31",
        "time_type": "historical",
    },
    "impact_domain": "mortality",
    "impact_type": "observed",
    "impact_magnitude": "5700 deaths",
    "magnitude_vague": False,
    "affected_group": "low-income households",
    "causal_relation": {
        "subject": "extreme rainfall",
        "predicate": "caused",
        "object": "5700 deaths",
        "dependency_path": "nsubj → resulted → in",
    },
    "uncertainty_level": "low",
    "uncertainty_source": "observational",
    "hedge_terms": ["may", "approximately"],
    "grounding_quote": "resulted in over 5,700 deaths",
    "grounding_verified": True,
}

# ── Main extraction flow ──────────────────────────────────────────────────────

if run and abstract_text.strip():
    tabs = st.tabs(
        ["Extraction", "Hedge Analysis", "Uncertainty", "Knowledge Graph"]
    )

    # ── Tab 1: Extraction ─────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Extracted Impact Record")

        if use_llm and api_key:
            import os
            os.environ["ANTHROPIC_API_KEY"] = api_key
            from src.annotation.llm_labeler import LLMLabeler

            with st.spinner("Running 2-pass LLM extraction…"):
                try:
                    labeler = LLMLabeler()
                    record = labeler.label_abstract("DEMO_001", abstract_text)
                    if not record:
                        st.error("LLM returned no parseable JSON. Check your API key and try again.")
                        record = None
                except Exception as exc:
                    err_msg = str(exc)
                    if "credit balance" in err_msg.lower() or "billing" in err_msg.lower():
                        st.error(
                            "**Anthropic API — insufficient credits.**  \n"
                            "Top up your balance at [console.anthropic.com/settings/billing]"
                            "(https://console.anthropic.com/settings/billing), then try again.  \n"
                            "The mock record is shown below for reference."
                        )
                        record = _MOCK_RECORD
                    else:
                        st.error(f"LLM error: {exc}")
                        record = None
        else:
            st.info("Running mock extraction. Enable **Live LLM Extraction** in the sidebar for real output.")
            record = _MOCK_RECORD

        if record:
            st.json(record)
        elif record is None and not (use_llm and api_key):
            st.warning("No record to display.")
            record = {}

        # Grounding quote highlight
        quote = (record or {}).get("grounding_quote")
        if quote:
            st.divider()
            st.subheader("Grounding Quote")
            escaped = re.escape(quote)
            highlighted = re.sub(
                escaped,
                f"**:green[{quote}]**",
                abstract_text,
                flags=re.IGNORECASE,
            )
            st.markdown(f"> {highlighted}")
            verified = (record or {}).get("grounding_verified", False)
            st.caption(f"Grounding verified: {'Yes' if verified else 'No'}")

        # Validation errors
        if record and record.get("validation_errors"):
            with st.expander("Schema validation warnings"):
                for err in record["validation_errors"]:
                    st.warning(err)

    # ── Tab 2: Hedge Analysis ────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("Linguistic Hedge Detection")

        hs = detect_hedges(abstract_text)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Hedge Score", f"{hs.score:.3f}")
        c2.metric("Level", hs.level.upper())
        c3.metric("Hedge Terms", hs.term_count)
        c4.metric("Words", hs.word_count)

        # Colour-coded highlight in abstract
        st.subheader("Abstract — hedge terms highlighted")
        _COLORS = {
            "high_uncertainty": "red",
            "medium_uncertainty": "orange",
            "low_uncertainty": "blue",
        }
        highlighted_abs = abstract_text
        for category, terms in hs.found_terms.items():
            color = _COLORS[category]
            for term in set(terms):
                highlighted_abs = re.sub(
                    r"\b" + re.escape(term) + r"\b",
                    f"**:{color}[{term}]**",
                    highlighted_abs,
                    flags=re.IGNORECASE,
                )
        st.markdown(highlighted_abs)

        # Breakdown by tier
        st.subheader("Breakdown by uncertainty tier")
        for cat, color in _COLORS.items():
            found_set = sorted(set(hs.found_terms.get(cat, [])))
            label = cat.replace("_", " ").title()
            if found_set:
                st.markdown(f"**:{color}[{label}]:** {', '.join(found_set)}")
            else:
                st.markdown(f"**{label}:** _none found_")

    # ── Tab 3: Uncertainty ────────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("Epistemic Uncertainty (MC Dropout)")

        model_path = _ROOT / "models" / "jer" / "jer_model.pt"

        if not model_path.exists():
            st.info(
                "No trained JER model found at `models/jer/jer_model.pt`.  "
                "Train the model first to see MC Dropout uncertainty."
            )
            st.markdown(
                """
**How MC Dropout works:**
1. Model is set to `train()` mode (dropout stays active)
2. Run N stochastic forward passes on the same input
3. **Variance** across passes = epistemic uncertainty
4. High variance → model is unsure about this abstract

**Research question this answers:**
Do abstracts with high hedge-term density also produce high MC variance?
Spearman correlation between the two signals is computed across the corpus.
                """
            )
            # Show a representative bar for demo purposes
            st.subheader("Demo uncertainty bars")
            st.progress(hs.score, text=f"Linguistic Hedge Score: {hs.score:.3f}")
            st.progress(0.42, text="MC Dropout Variance (demo): 0.42")
            st.caption("MC Dropout bar is illustrative — train the model for real values.")

        else:
            import torch
            from transformers import AutoTokenizer
            from src.extraction.jer_model import JERModel
            from src.uncertainty.mc_dropout import mc_dropout_predict

            with st.spinner(f"Running {n_mc_passes} MC Dropout passes…"):
                try:
                    tokenizer = AutoTokenizer.from_pretrained(str(model_path.parent))
                    model = JERModel()
                    model.load_state_dict(
                        torch.load(str(model_path), map_location="cpu")
                    )
                    enc = tokenizer(
                        abstract_text,
                        return_tensors="pt",
                        max_length=512,
                        truncation=True,
                    )
                    mean_pred, variance = mc_dropout_predict(
                        model,
                        enc["input_ids"],
                        enc["attention_mask"],
                        n_passes=n_mc_passes,
                    )
                    mc_var = float(variance.mean().item())
                    mc_ent = float(
                        -(mean_pred * (mean_pred + 1e-8).log()).sum(-1).mean().item()
                    )
                except Exception as exc:
                    st.error(f"MC Dropout failed: {exc}")
                    mc_var, mc_ent = 0.0, 0.0

            c1, c2, c3 = st.columns(3)
            c1.metric("Epistemic Variance", f"{mc_var:.5f}")
            c2.metric("Predictive Entropy", f"{mc_ent:.4f}")
            c3.metric("Hedge Score", f"{hs.score:.3f}")

            st.progress(min(mc_var * 20, 1.0), text=f"MC Dropout Variance: {mc_var:.5f}")
            st.progress(hs.score, text=f"Linguistic Hedge Score: {hs.score:.3f}")

            if mc_var > 0 and hs.score > 0:
                if (mc_var > 0.05) == (hs.score > 0.2):
                    st.success(
                        "Both signals agree: high hedge language correlates with high model uncertainty."
                    )
                else:
                    st.warning(
                        "Signals diverge. This is an interesting case — "
                        "model uncertainty and linguistic hedging do not align."
                    )

    # ── Tab 4: Knowledge Graph ────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("Mini Knowledge Graph")

        if record:
            G = build_graph([record])
            stats = graph_stats(G)
            c1, c2, c3 = st.columns(3)
            c1.metric("Nodes", stats["total_nodes"])
            c2.metric("Edges", stats["total_edges"])
            c3.metric("Density", stats["density"])

            graph_html = _ROOT / "outputs" / "demo_graph.html"
            try:
                render_html(G, graph_html, title="Indra — Single Abstract Graph")
                html_content = graph_html.read_text(encoding="utf-8", errors="replace")
                components.html(html_content, height=600, scrolling=False)
            except ImportError:
                st.warning("Install `pyvis` for interactive graph: `pip install pyvis`")
                st.markdown("**Graph nodes:**")
                for node_id, attrs in G.nodes(data=True):
                    st.markdown(f"- `{node_id}` _(type: {attrs.get('node_type')})_")
                st.markdown("**Graph edges:**")
                for src, dst, attrs in G.edges(data=True):
                    st.markdown(f"- `{src}` —[{attrs.get('relation')}]→ `{dst}`")
        else:
            st.info("No extraction result available for graph.")

elif run:
    st.warning("Please enter an abstract before clicking Extract.")
