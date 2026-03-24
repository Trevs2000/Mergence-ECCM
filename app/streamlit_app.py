import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import os
import sys
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from metrics import eccm as eccm_module
from metrics.psc import PSCCalculator
from metrics.fsc import FSCCalculator
from metrics.rsc import RSCCalculator

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mergence – ECCM Platform",
    page_icon="🧬",
    layout="wide",
)

# ── Shared EPC + ECCM loader ──────────────────────────────────────────────────
@st.cache_resource
def load_eccm_tools():
    epc = joblib.load("./models/epc_model.pkl") if Path("./models/epc_model.pkl").exists() else None
    calc = eccm_module.ECCMCalculator()
    return epc, calc

epc_model, eccm_calc = load_eccm_tools()

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_tier(eccm_score: float):
    """Return (label, colour, emoji) for compatibility tier."""
    if eccm_score >= 0.85:
        return "High Compatibility",   "#28a745", "✅"
    elif eccm_score >= 0.65:
        return "Medium Compatibility", "#ffc107", "⚠️"
    else:
        return "Low Compatibility",    "#dc3545", "❌"

def xai_text(psc, fsc, rsc, eccm_score, model_a_name, model_b_name):
    """
    Generate a plain-English explanation of ECCM scores.
    This is the XAI narrative shown below the metrics on page 1.
    """
    tier, _, emoji = get_tier(eccm_score)

    psc_explain = (
        "very similar internal weight structures, meaning they have learned "
        "comparable representations of the data"
        if psc >= 0.9 else
        "moderately similar weight structures, with some divergence in what "
        "each model has learned"
        if psc >= 0.65 else
        "quite different internal weight structures, suggesting they have learned "
        "very different representations"
    )
    fsc_explain = (
        "agree closely on their predictions across the validation set, "
        "indicating consistent decision boundaries"
        if fsc >= 0.9 else
        "agree on most predictions but diverge on harder cases"
        if fsc >= 0.65 else
        "frequently disagree on their predictions, which may reflect "
        "fundamentally different decision strategies"
    )
    rsc_explain = (
        "rank the same features as most important, showing highly aligned "
        "feature utilisation"
        if rsc >= 0.9 else
        "have broadly similar feature importance rankings with some differences"
        if rsc >= 0.65 else
        "prioritise very different features, which is a strong signal of "
        "representational incompatibility"
    )

    verdict = {
        "High Compatibility":   (
            "These two models are strong candidates for merging. "
            "A merged model is likely to be at least as good as the better parent, "
            "and may outperform both by combining complementary knowledge."
        ),
        "Medium Compatibility": (
            "These models have borderline compatibility. "
            "Merging may provide a modest gain but carries some risk. "
            "Review the blend ratio curve on the Analysis tab before proceeding."
        ),
        "Low Compatibility":    (
            "These models are poorly compatible. "
            "Merging is likely to produce a model that performs worse than either parent. "
            "It is recommended to select a different pair unless you have a specific reason to proceed."
        ),
    }[tier]

    lines = [
        f"**Why were these models evaluated?**",
        f"**{model_a_name}** and **{model_b_name}** received an overall ECCM score of "
        f"**{eccm_score:.3f}**, placing them in the **{emoji} {tier}** category.",
        "",
        f"**Parameter Space Compatibility (PSC = {psc:.3f}):** The two models have {psc_explain}.",
        "",
        f"**Functional Similarity (FSC = {fsc:.3f}):** The models {fsc_explain}.",
        "",
        f"**Representational Similarity (RSC = {rsc:.3f}):** The models {rsc_explain}.",
        "",
        f"**Verdict:** {verdict}",
    ]
    return "\n".join(lines)

def feature_importance_chart(model, model_name, feature_names=None):
    """Return a Plotly bar figure of top-15 feature importances."""
    if not hasattr(model, "feature_importances_"):
        return None
    fi = model.feature_importances_
    n  = len(fi)
    names = feature_names if feature_names and len(feature_names) == n else [f"f{i}" for i in range(n)]
    df = pd.DataFrame({"feature": names, "importance": fi})
    df = df.nlargest(15, "importance")
    fig = px.bar(
        df, x="importance", y="feature", orientation="h",
        title=f"Top-15 Feature Importances – {model_name}",
        color="importance", color_continuous_scale="Blues",
        labels={"importance": "Importance", "feature": "Feature"},
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), height=380, showlegend=False)
    return fig

def blend_auc_curve(model_a, model_b, X, y, model_a_name, model_b_name):
    """Return a Plotly line figure of AUC across 21 blend ratios."""
    from sklearn.metrics import roc_auc_score
    pa = model_a.predict_proba(X)[:, 1]
    pb = model_b.predict_proba(X)[:, 1]
    ratios = np.linspace(0, 1, 21)
    aucs   = [roc_auc_score(y, r * pa + (1 - r) * pb) for r in ratios]
    best_r = ratios[int(np.argmax(aucs))]
    best_a = max(aucs)
    fig = px.line(
        x=ratios, y=aucs,
        labels={"x": f"← 100 % {model_b_name}  |  Weight on {model_a_name}  |  100 % {model_a_name} →",
                "y": "AUC"},
        title=f"AUC across blend ratios (optimal = {best_r:.2f} → AUC {best_a:.6f})",
    )
    fig.add_vline(x=best_r, line_dash="dash", line_color="green",
                  annotation_text=f"Optimal {best_r:.2f}", annotation_position="top right")
    fig.update_layout(height=350)
    return fig, best_r, best_a

# ═════════════════════════════════════════════════════════════════════════════
# MAIN TITLE
# ═════════════════════════════════════════════════════════════════════════════
st.title("🧬 Mergence")
st.caption("Evolutionary Compatibility & Co-evolution Metric — Model Merge Compatibility Platform")

tab1, tab2, tab3 = st.tabs([
    "🔬 Compatibility Simulator",
    "📊 Analysis",
    "ℹ️ About",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — COMPATIBILITY SIMULATOR
# Upload models → ECCM scores → XAI explanation → merge action
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Model Compatibility Simulator")
    st.markdown(
        "Upload two trained Random Forest models and a labelled validation CSV. "
        "The platform computes ECCM compatibility live and explains the result in plain English."
    )

    # ── Step 1: Uploads ───────────────────────────────────────────────────────
    st.subheader("Step 1 — Upload Models & Validation Data")
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        file_a    = st.file_uploader("Model A (.pkl)", type=["pkl"], key="s1_a")
        a_name    = st.text_input("Label for Model A", value="Model A", key="s1_aname")
    with col_u2:
        file_b    = st.file_uploader("Model B (.pkl)", type=["pkl"], key="s1_b")
        b_name    = st.text_input("Label for Model B", value="Model B", key="s1_bname")

    val_file   = st.file_uploader("Validation CSV (must contain a label column)", type=["csv"], key="s1_val")
    label_col  = st.text_input("Label column name", value="Class", key="s1_label")

    # ── Step 2: Check compatibility ───────────────────────────────────────────
    if file_a and file_b and val_file:
        model_a_obj = joblib.load(io.BytesIO(file_a.read()))
        model_b_obj = joblib.load(io.BytesIO(file_b.read()))
        val_df      = pd.read_csv(val_file)

        if label_col not in val_df.columns:
            st.error(f"Column '{label_col}' not found. Available: {val_df.columns.tolist()}")
            st.stop()

        feature_names = [c for c in val_df.columns if c != label_col]
        X_sim = val_df.drop(columns=[label_col]).values
        y_sim = val_df[label_col].values

        st.divider()
        st.subheader("Step 2 — Compatibility Check")

        if st.button("▶ Run Compatibility Check", key="s1_run", type="primary"):
            with st.spinner("Computing PSC / FSC / RSC / ECCM…"):
                try:
                    psc_score = PSCCalculator().compute(model_a_obj, model_b_obj)
                    fsc_score = FSCCalculator().compute(model_a_obj, model_b_obj, X_sim)
                    rsc_score = RSCCalculator().compute(model_a_obj, model_b_obj)

                    epc_input = 0.5
                    if epc_model is not None:
                        try:
                            epc_input = float(epc_model.predict([[psc_score, fsc_score, rsc_score]])[0])
                        except Exception:
                            epc_input = 0.5

                    scores = eccm_calc.compute(model_a_obj, model_b_obj, X_sim, epc_pred=epc_input)
                    ec     = scores["eccm"]
                    tier, colour, emoji = get_tier(ec)

                    # Store in session state so the merge button can use them
                    st.session_state["sim_scores"]    = scores
                    st.session_state["sim_ec"]        = ec
                    st.session_state["sim_tier"]      = tier
                    st.session_state["sim_model_a"]   = model_a_obj
                    st.session_state["sim_model_b"]   = model_b_obj
                    st.session_state["sim_X"]         = X_sim
                    st.session_state["sim_y"]         = y_sim
                    st.session_state["sim_a_name"]    = a_name
                    st.session_state["sim_b_name"]    = b_name
                    st.session_state["sim_feat_names"]= feature_names
                    st.session_state["check_done"]    = True
                except Exception as e:
                    st.error(f"Computation failed: {e}")
                    st.session_state["check_done"] = False

        # ── Results (persist across re-runs via session_state) ────────────────
        if st.session_state.get("check_done"):
            scores    = st.session_state["sim_scores"]
            ec        = st.session_state["sim_ec"]
            tier, colour, emoji = get_tier(ec)
            psc_s, fsc_s, rsc_s = scores["psc"], scores["fsc"], scores["rsc"]
            a_n = st.session_state["sim_a_name"]
            b_n = st.session_state["sim_b_name"]

            # Tier banner
            st.markdown(
                f"""<div style='padding:14px 20px; border-radius:8px;
                background:{colour}22; border-left:5px solid {colour};
                font-size:1.2rem; font-weight:600;'>
                {emoji}&nbsp;&nbsp;{tier} — ECCM Score: {ec:.4f}</div>""",
                unsafe_allow_html=True,
            )
            st.markdown("")

            # Four metric cards
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("PSC  (Weights)",   f"{psc_s:.4f}")
            c2.metric("FSC  (Behaviour)", f"{fsc_s:.4f}")
            c3.metric("RSC  (Features)",  f"{rsc_s:.4f}")
            c4.metric("ECCM",             f"{ec:.4f}")

            # ECCM component bar chart
            fig_bar = go.Figure(go.Bar(
                x=["PSC", "FSC", "RSC", "ECCM"],
                y=[psc_s, fsc_s, rsc_s, ec],
                marker_color=["#4c72b0", "#55a868", "#c44e52", "#8172b2"],
                text=[f"{v:.4f}" for v in [psc_s, fsc_s, rsc_s, ec]],
                textposition="outside",
            ))
            fig_bar.update_layout(
                title=f"ECCM Component Scores — {a_n} + {b_n}",
                yaxis=dict(range=[0, 1.12]),
                height=330,
                margin=dict(t=50, b=30),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # ── Feature importance comparison ─────────────────────────────────
            st.subheader("Feature Importance Comparison")
            feat_names = st.session_state["sim_feat_names"]
            col_fi1, col_fi2 = st.columns(2)
            with col_fi1:
                fig_fi_a = feature_importance_chart(
                    st.session_state["sim_model_a"], a_n, feat_names
                )
                if fig_fi_a:
                    st.plotly_chart(fig_fi_a, use_container_width=True)
                else:
                    st.info("Model A has no feature_importances_ attribute.")
            with col_fi2:
                fig_fi_b = feature_importance_chart(
                    st.session_state["sim_model_b"], b_n, feat_names
                )
                if fig_fi_b:
                    st.plotly_chart(fig_fi_b, use_container_width=True)
                else:
                    st.info("Model B has no feature_importances_ attribute.")

            # ── XAI explanation ───────────────────────────────────────────────
            st.subheader("🧠 XAI Explanation")
            explanation = xai_text(psc_s, fsc_s, rsc_s, ec, a_n, b_n)
            st.markdown(explanation)

            # ── Step 3: Merge action ──────────────────────────────────────────
            st.divider()
            st.subheader("Step 3 — Merge Models")

            if tier == "Low Compatibility":
                st.error(
                    "⛔ **High Risk:** These models have low compatibility (ECCM < 0.65). "
                    "Merging is likely to produce a model that performs **worse** than either parent. "
                    "Proceed only if you understand and accept this risk."
                )
            elif tier == "Medium Compatibility":
                st.warning(
                    "⚠️ **Moderate Risk:** These models have borderline compatibility (0.65 ≤ ECCM < 0.85). "
                    "Merging may yield a modest gain but could also reduce performance. "
                    "Check the blend ratio curve on the **Analysis** tab first."
                )

            merge_anyway = True
            if tier in ("Low Compatibility", "Medium Compatibility"):
                merge_anyway = st.checkbox(
                    "I understand the risk and want to merge anyway", key="s1_risk_ack"
                )

            if merge_anyway:
                from sklearn.metrics import roc_auc_score as _auc

                col_ratio, col_merge = st.columns([2, 1])
                with col_ratio:
                    blend_r = st.slider(
                        f"Blend weight for {a_n}  (1 − weight goes to {b_n})",
                        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                        key="s1_blend",
                    )
                with col_merge:
                    st.markdown("<br>", unsafe_allow_html=True)
                    run_merge = st.button("⚗️ Merge & Evaluate", key="s1_merge", type="primary")

                if run_merge:
                    with st.spinner("Merging and evaluating…"):
                        X_m  = st.session_state["sim_X"]
                        y_m  = st.session_state["sim_y"]
                        ma   = st.session_state["sim_model_a"]
                        mb   = st.session_state["sim_model_b"]
                        pa   = ma.predict_proba(X_m)[:, 1]
                        pb   = mb.predict_proba(X_m)[:, 1]
                        blend_proba = blend_r * pa + (1 - blend_r) * pb
                        merged_auc  = _auc(y_m, blend_proba)
                        auc_a       = _auc(y_m, pa)
                        auc_b       = _auc(y_m, pb)
                        best_parent = max(auc_a, auc_b)
                        delta       = merged_auc - best_parent

                    st.success(f"✅ Merge complete — Merged AUC: **{merged_auc:.6f}**")

                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric(f"{a_n} AUC",     f"{auc_a:.6f}")
                    mc2.metric(f"{b_n} AUC",     f"{auc_b:.6f}")
                    mc3.metric("Merged AUC",      f"{merged_auc:.6f}",
                               delta=f"{delta:+.6f} vs best parent")

                    # Download merged predictions as pkl-compatible object
                    class BlendedModel:
                        """Lightweight wrapper exposing predict_proba for the merged blend."""
                        def __init__(self, m_a, m_b, ratio):
                            self.model_a = m_a
                            self.model_b = m_b
                            self.ratio   = ratio
                        def predict_proba(self, X):
                            pa_ = self.model_a.predict_proba(X)[:, 1]
                            pb_ = self.model_b.predict_proba(X)[:, 1]
                            blend_ = self.ratio * pa_ + (1 - self.ratio) * pb_
                            return np.column_stack([1 - blend_, blend_])

                    merged_obj  = BlendedModel(ma, mb, blend_r)
                    buf         = io.BytesIO()
                    joblib.dump(merged_obj, buf)
                    buf.seek(0)
                    st.download_button(
                        label="⬇️ Download Merged Model (.pkl)",
                        data=buf,
                        file_name=f"merged_{a_n}_{b_n}_r{blend_r:.2f}.pkl",
                        mime="application/octet-stream",
                        key="s1_download",
                    )

                    # XAI explanation of the merge outcome
                    st.subheader("🧠 XAI — Merge Outcome Explanation")
                    if delta > 0.001:
                        outcome_text = (
                            f"The merged model (blend ratio **{blend_r:.2f}** on {a_n}) "
                            f"achieved an AUC of **{merged_auc:.6f}**, which is "
                            f"**{delta:+.6f}** better than the best parent ({a_n if auc_a > auc_b else b_n}, "
                            f"AUC {best_parent:.6f}). This confirms that combining the complementary "
                            f"knowledge of both models under the ECCM-guided blend weight produced a "
                            f"net improvement in discriminative performance."
                        )
                    elif delta > -0.001:
                        outcome_text = (
                            f"The merged model achieved an AUC of **{merged_auc:.6f}**, "
                            f"which is essentially equal to the best parent (Δ = {delta:+.6f}). "
                            f"This is consistent with a flat performance landscape around the "
                            f"chosen blend ratio, suggesting both models capture similar patterns. "
                            f"Try the Analysis tab's blend ratio curve to find a sharper optimum."
                        )
                    else:
                        outcome_text = (
                            f"The merged model achieved an AUC of **{merged_auc:.6f}**, "
                            f"which is **{abs(delta):.6f} lower** than the best parent. "
                            f"This is consistent with the {tier} signal from ECCM. "
                            f"Consider adjusting the blend ratio or choosing a higher-compatibility pair."
                        )
                    st.markdown(outcome_text)
    else:
        st.info("⬆️ Upload Model A, Model B, and a validation CSV above to begin.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYSIS
# Shows charts specific to the uploaded pair only — no historical experiment data
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Pair Analysis")

    if not st.session_state.get("check_done"):
        st.info("Run a compatibility check on the **Compatibility Simulator** tab first.")
    else:
        a_n  = st.session_state["sim_a_name"]
        b_n  = st.session_state["sim_b_name"]
        ma   = st.session_state["sim_model_a"]
        mb   = st.session_state["sim_model_b"]
        X_a  = st.session_state["sim_X"]
        y_a  = st.session_state["sim_y"]
        sc   = st.session_state["sim_scores"]
        ec   = st.session_state["sim_ec"]
        feat = st.session_state["sim_feat_names"]
        tier, colour, emoji = get_tier(ec)

        st.markdown(
            f"Showing analysis for: **{a_n}** + **{b_n}** "
            f"&nbsp;&nbsp;|&nbsp;&nbsp; {emoji} {tier} (ECCM = {ec:.4f})"
        )
        st.divider()

        # ── Row 1: ECCM radar + blend AUC curve ──────────────────────────────
        col_r1, col_r2 = st.columns(2)

        with col_r1:
            st.subheader("ECCM Component Radar")
            categories = ["PSC", "FSC", "RSC", "ECCM"]
            values_rad = [sc["psc"], sc["fsc"], sc["rsc"], ec]
            fig_radar  = go.Figure(go.Scatterpolar(
                r=values_rad + [values_rad[0]],
                theta=categories + [categories[0]],
                fill="toself",
                fillcolor=f"{colour}33",
                line_color=colour,
                name="Scores",
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(range=[0, 1])),
                title=f"{a_n} + {b_n} — ECCM Components",
                height=370,
                showlegend=False,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            # XAI text for the radar
            st.caption(
                f"**Reading this chart:** Each axis shows one ECCM sub-metric (0 = no similarity, "
                f"1 = perfect similarity). A larger, more even polygon indicates a well-rounded, "
                f"compatible pair. A narrow or skewed shape suggests specific weaknesses — "
                f"e.g., low FSC means the models disagree in their predictions despite similar weights."
            )

        with col_r2:
            st.subheader("Blend Ratio AUC Curve")
            try:
                fig_blend, best_r, best_auc = blend_auc_curve(ma, mb, X_a, y_a, a_n, b_n)
                st.plotly_chart(fig_blend, use_container_width=True)
                st.caption(
                    f"**Reading this chart:** Each point shows the AUC when blending the two models "
                    f"at a specific ratio. The green dashed line marks the optimal ratio ({best_r:.2f}). "
                    f"A flat curve means both models perform similarly across all ratios. "
                    f"A sharp peak suggests one model is substantially stronger for this dataset."
                )
            except Exception as e:
                st.warning(f"Could not compute blend curve: {e}")

        st.divider()

        # ── Row 2: Side-by-side feature importances ──────────────────────────
        st.subheader("Feature Importance Comparison")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            fig_fi_a = feature_importance_chart(ma, a_n, feat)
            if fig_fi_a:
                st.plotly_chart(fig_fi_a, use_container_width=True)
        with col_f2:
            fig_fi_b = feature_importance_chart(mb, b_n, feat)
            if fig_fi_b:
                st.plotly_chart(fig_fi_b, use_container_width=True)

        # Feature importance overlap XAI
        if hasattr(ma, "feature_importances_") and hasattr(mb, "feature_importances_"):
            fi_a  = ma.feature_importances_
            fi_b  = mb.feature_importances_
            names = feat if len(feat) == len(fi_a) else [f"f{i}" for i in range(len(fi_a))]
            top_a = set(np.argsort(fi_a)[-10:])
            top_b = set(np.argsort(fi_b)[-10:])
            overlap = len(top_a & top_b)
            st.caption(
                f"**Feature overlap:** {overlap} of the top-10 features are shared between "
                f"{a_n} and {b_n}. "
                + (
                    "High overlap suggests the models use similar signals — merging is safer."
                    if overlap >= 7 else
                    "Moderate overlap means the models emphasise different aspects of the data — "
                    "merging may combine complementary strengths."
                    if overlap >= 4 else
                    "Low overlap means the models rely on very different features — "
                    "this is a key driver of the low RSC score."
                )
            )

        st.divider()

        # ── Row 3: Prediction probability distributions ───────────────────────
        st.subheader("Prediction Probability Distributions")
        try:
            pa = ma.predict_proba(X_a)[:, 1]
            pb = mb.predict_proba(X_a)[:, 1]

            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=pa, name=a_n, opacity=0.6,
                marker_color="#4c72b0", nbinsx=40,
            ))
            fig_dist.add_trace(go.Histogram(
                x=pb, name=b_n, opacity=0.6,
                marker_color="#55a868", nbinsx=40,
            ))
            fig_dist.update_layout(
                barmode="overlay",
                xaxis_title="Predicted Probability (class 1)",
                yaxis_title="Count",
                title="Prediction Score Distributions",
                height=330,
            )
            st.plotly_chart(fig_dist, use_container_width=True)

            # Agreement scatter
            fig_agree = px.scatter(
                x=pa, y=pb, opacity=0.3,
                labels={"x": f"{a_n} score", "y": f"{b_n} score"},
                title="Prediction Agreement (each dot = one validation sample)",
                color=y_a.astype(str),
                color_discrete_map={"0": "#4c72b0", "1": "#c44e52"},
            )
            fig_agree.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line=dict(dash="dash", color="grey"),
                name="Perfect agreement",
            ))
            fig_agree.update_layout(height=370)
            st.plotly_chart(fig_agree, use_container_width=True)

            st.caption(
                "**Reading the agreement scatter:** Points on the diagonal mean both models "
                "predict the same probability. Points far off the diagonal reveal cases where "
                "the models strongly disagree — these are the samples that will be most affected "
                "by the choice of blend ratio."
            )
        except Exception as e:
            st.warning(f"Could not compute prediction distributions: {e}")

        # ── Full XAI narrative for the pair ─────────────────────────────────
        st.divider()
        st.subheader("🧠 Full XAI Narrative")
        st.markdown(xai_text(sc["psc"], sc["fsc"], sc["rsc"], ec, a_n, b_n))

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("About Mergence")
    st.markdown("""
### What is ECCM?

**ECCM** (Evolutionary Compatibility & Co-evolution Metric) predicts whether merging
two machine learning models will improve performance. It combines four sub-metrics:

| Metric | Description |
|--------|-------------|
| **PSC** | Parameter Space Compatibility — cosine + Euclidean similarity of model weights |
| **FSC** | Functional Similarity — prediction agreement on validation data |
| **RSC** | Representational Similarity — feature importance rank correlation |
| **EPC** | Evolutionary Pressure Compatibility — meta-learner trained on historical merges |

The composite score is:

> `ECCM = w_psc × PSC + w_fsc × FSC + w_rsc × RSC + w_epc × EPC`

---

### Research Questions addressed by this platform

| RQ | Question | Where |
|----|----------|-------|
| RQ1 | How to quantify evolutionary pressure within a compatibility metric? | Simulator — live EPC scoring |
| RQ2 | Which PSC/FSC/RSC combination is optimal? | Analysis — radar chart + feature overlap |
| RQ3 | What efficiency gains does ECCM provide over baseline? | Simulator — tier-based merge gate |
| RQ4 | How interpretable is ECCM? | Both tabs — XAI narrative + all charts explained |

---

### Compatibility Tiers

| Tier | ECCM Range | Meaning |
|------|-----------|---------|
| ✅ High | ≥ 0.85 | Strong merge candidate, likely to improve on both parents |
| ⚠️ Medium | 0.65 – 0.84 | Borderline, proceed with caution |
| ❌ Low | < 0.65 | Incompatible, merging likely to hurt performance |

---
*Trevin Joseph · w1953285 · BEng Software Engineering · IIT / University of Westminster · 2026*
""")