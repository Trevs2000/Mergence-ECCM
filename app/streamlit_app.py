import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.graph_objects as go
import shap

import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


from metrics import eccm

# Set page config
st.set_page_config(
    page_title="ECCM: Model Merge Predictor",
    page_icon="",
    layout="wide"
)

st.title("ECCM: Evolutionary Context Compatibility Measure")

# Load models and data
@st.cache_resource
def load_assets():
    fraud_models = {}
    for pkl in Path('./models/fraud').glob('*.pkl'):
        fraud_models[pkl.stem] = joblib.load(pkl)

    churn_models = {}
    for pkl in Path('./models/churn').glob('*.pkl'):
        churn_models[pkl.stem] = joblib.load(pkl)
    
    epc = joblib.load('./models/epc_model.pkl')
    eccm_calc = eccm.ECCMCalculator() 
    
    return fraud_models, churn_models, epc, eccm_calc

fraud_models, churn_models, epc, eccm_calc = load_assets()

# Tabs for different views
tab1, tab2, tab3 = st.tabs(["📊 Predict", "📈 Analysis", "ℹ️ About"])

with tab1:
    st.header("Merge Quality Prediction")

    task = st.radio("Task:", ["Fraud", "Churn"], horizontal=True)

    if task == "Fraud":
        model_ids = list(fraud_models.keys())
    else:
        model_ids = list(churn_models.keys())

    colA, colB = st.columns(2)
    with colA:
        model_a_id = st.selectbox("Model A:", model_ids, key="model_a")
    with colB:
        model_b_id = st.selectbox("Model B:", model_ids, key="model_b")

    if st.button("Predict Merge Quality"):
        # 1) Pick correct models and validation data
        if task == "Fraud":
            X_val = pd.read_csv('./data/fraud_val.csv').values  # features only
            model_a = fraud_models[model_a_id]
            model_b = fraud_models[model_b_id]
        else:
            # churn_val_with_churn_col.csv has target 'Churn'
            churn_val = pd.read_csv('./data/churn_val_with_churn_col.csv')
            X_val = churn_val.drop('Churn', axis=1).values
            model_a = churn_models[model_a_id]
            model_b = churn_models[model_b_id]

        # 2) Compute ECCM metrics
        # Example EPC input: use PSC/FSC/RSC later; for now, let EPC
        # internally take PSC/FSC/RSC (if your compute() expects epc_pred
        # pass something like epc.predict([[psc, fsc, rsc]]) instead).
        scores = eccm_calc.compute(
            model_a, model_b, X_val, epc_pred=0.5
        )

        # 3) Display
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("PSC (Weights)", f"{scores['psc']:.3f}")
        col2.metric("FSC (Behavior)", f"{scores['fsc']:.3f}")
        col3.metric("RSC (Features)", f"{scores['rsc']:.3f}")
        col4.metric(
            "ECCM Score",
            f"{scores['eccm']:.3f}",
            delta="Good" if scores['eccm'] > 0.7 else "Fair"
        )

with tab2:
    st.header("Historical Analysis")

    # Choose which history to view
    hist_task = st.radio("History for:", ["Fraud", "Churn"], horizontal=True)

    if hist_task == "Fraud":
        merge_path = './results/merges/fraud/fraud_merge_results.csv'
    else:
        merge_path = './results/merges/churn/churn_merge_results.csv'

    if not os.path.exists(merge_path):
        st.warning(f"No merge history found at {merge_path}")
    else:
        merge_df = pd.read_csv(merge_path)

        fig = go.Figure()
        success = merge_df[merge_df['success'] == 1]
        failure = merge_df[merge_df['success'] == 0]

        fig.add_trace(go.Scatter(
            x=success['eccm'], y=success['auc_merged'],
            mode='markers', name='Successful', marker=dict(color='green')
        ))
        fig.add_trace(go.Scatter(
            x=failure['eccm'], y=failure['auc_merged'],
            mode='markers', name='Failed', marker=dict(color='red')
        ))

        fig.update_layout(
            xaxis_title="ECCM Score",
            yaxis_title="Merged Model AUC"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("About ECCM")
    st.markdown("""
    ### What is ECCM?

    ECCM (Evolutionary Compatibility & Co-evolution metric) predicts whether
    merging two machine learning models will improve performance.

    It combines 4 metrics:
    - **PSC**: Parameter Space Compatibility (weight similarity)
    - **FSC**: Functional Similarity (prediction agreement)
    - **RSC**: Representational Similarity (feature usage)
    - **EPC**: Evolutionary Pressure (meta-learner)

    ### How does it work?

    1. Compute PSC, FSC, RSC for two models
    2. Use the trained EPC model to estimate merge success
    3. Return an ECCM score (0–1, higher is better)
    """)