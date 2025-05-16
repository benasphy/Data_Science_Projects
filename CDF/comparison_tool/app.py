import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy import stats


def run_cdf_comparison():
    st.title("📊 Distribution Comparison Tool")
    st.markdown("""
    ### Compare Cumulative Distribution Functions (CDFs)
    
    Select two distributions and compare their CDFs side by side. Adjust parameters to see how the shapes change.
    """)

    dist_options = ["Normal", "Exponential", "Uniform"]

    col1, col2 = st.columns(2)
    with col1:
        dist1 = st.selectbox("Distribution 1", dist_options, key="comp_dist1")
        param1 = st.slider("Parameter 1 (e.g., mean or rate)", 0.1, 5.0, 1.0, 0.1, key="comp_param1")
    with col2:
        dist2 = st.selectbox("Distribution 2", dist_options, key="comp_dist2")
        param2 = st.slider("Parameter 2 (e.g., mean or rate)", 0.1, 5.0, 2.0, 0.1, key="comp_param2")

    x = np.linspace(-3, 8, 500)
    cdf1, cdf2 = None, None
    label1, label2 = dist1, dist2

    if dist1 == "Normal":
        cdf1 = stats.norm.cdf(x, loc=param1, scale=1)
        label1 = f"Normal (μ={param1}, σ=1)"
    elif dist1 == "Exponential":
        cdf1 = stats.expon.cdf(x, scale=1/param1)
        label1 = f"Exponential (λ={param1})"
    elif dist1 == "Uniform":
        cdf1 = stats.uniform.cdf(x, loc=0, scale=param1)
        label1 = f"Uniform (0, {param1})"

    if dist2 == "Normal":
        cdf2 = stats.norm.cdf(x, loc=param2, scale=1)
        label2 = f"Normal (μ={param2}, σ=1)"
    elif dist2 == "Exponential":
        cdf2 = stats.expon.cdf(x, scale=1/param2)
        label2 = f"Exponential (λ={param2})"
    elif dist2 == "Uniform":
        cdf2 = stats.uniform.cdf(x, loc=0, scale=param2)
        label2 = f"Uniform (0, {param2})"

    fig = go.Figure()
    if cdf1 is not None:
        fig.add_trace(go.Scatter(x=x, y=cdf1, mode='lines', name=label1, line=dict(color='blue')))
    if cdf2 is not None:
        fig.add_trace(go.Scatter(x=x, y=cdf2, mode='lines', name=label2, line=dict(color='red')))

    fig.update_layout(
        title="CDF Comparison",
        xaxis_title="x",
        yaxis_title="CDF",
        yaxis_range=[0, 1],
        legend=dict(x=0.01, y=0.99)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    - **Normal**: Symmetric, bell-shaped CDF.
    - **Exponential**: Rapid rise, models waiting times.
    - **Uniform**: Linear CDF between endpoints.
    """)
