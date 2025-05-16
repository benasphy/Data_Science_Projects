import streamlit as st
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from scipy import stats
import pandas as pd

def run_survival_analysis():
    st.title("🧬 Survival Analysis & Hazard Rates")
    
    st.markdown("""
    ### Understanding Survival and Hazard Functions
    
    Survival analysis explores time-to-event data, revealing 
    how probability of survival changes over time across 
    different scenarios and distributions.
    
    Key Concepts:
    - Survival Function (S(t))
    - Hazard Rate (h(t))
    - Cumulative Hazard Function
    - Censoring and Competing Risks
    """)
    
    # Tabs for different survival analysis perspectives
    tab1, tab2, tab3 = st.tabs([
        "Basic Survival Functions", 
        "Hazard Rate Analysis", 
        "Real-World Applications"
    ])
    
    with tab1:
        st.subheader("Fundamental Survival Functions")
        
        # Distribution parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            distribution_type = st.selectbox(
                "Select Distribution", 
                [
                    "Exponential", 
                    "Weibull", 
                    "Log-Logistic",
                    "Gamma"
                ],
                key="survival_dist_select"
            )
        
        with col2:
            sample_size = st.slider("Sample Size", 100, 10000, 1000, key="survival_sample_size")
        
        with col3:
            scale_param = st.slider("Scale Parameter", 0.1, 5.0, 1.0, 0.1, key="survival_scale_param")
        
        # Always generate and plot survival function on every widget change
        np.random.seed(42)
        time_points = np.linspace(0, 10, 200)
        if distribution_type == "Exponential":
            survival_curve = np.exp(-scale_param * time_points)
            plot_title = f"Exponential Survival Function (λ = {scale_param})"
        elif distribution_type == "Weibull":
            shape = 1.5
            survival_curve = np.exp(-(time_points/scale_param)**shape)
            plot_title = f"Weibull Survival Function (η = {scale_param}, k = {shape})"
        elif distribution_type == "Log-Logistic":
            shape = 2.0
            survival_curve = 1 / (1 + (time_points/scale_param)**shape)
            plot_title = f"Log-Logistic Survival Function (η = {scale_param}, k = {shape})"
        elif distribution_type == "Gamma":
            shape = 2.0
            survival_curve = 1 - stats.gamma.cdf(time_points, a=shape, scale=scale_param)
            plot_title = f"Gamma Survival Function (η = {scale_param}, k = {shape})"
        else:
            survival_curve = np.ones_like(time_points)
            plot_title = "Invalid Distribution Selected"
        
        # Visualization
        fig_survival = go.Figure()
        fig_survival.add_trace(go.Scatter(
            x=time_points,
            y=survival_curve,
            mode='lines',
            name='Survival Function',
            line=dict(color='blue', width=3)
        ))
        fig_survival.update_layout(
            title=plot_title,
            xaxis_title="Time",
            yaxis_title="Survival Probability",
            yaxis_range=[0, 1]
        )
        st.plotly_chart(fig_survival, use_container_width=True)
        
        # Statistical insights
        st.subheader("Survival Function Analysis")
        key_times = [0.5, 1.0, 2.0, 5.0]
        survival_points = [np.interp(t, time_points, survival_curve) for t in key_times]
        stats_df = pd.DataFrame({
            'Time Point': key_times,
            'Survival Probability': survival_points
        })
        st.table(stats_df)
        st.markdown("""
        ### Survival Function Insights
        - Probability of surviving beyond a specific time
        - Decreases monotonically
        - Provides intuition about event occurrence
        """)
    
    with tab2:
        st.subheader("Hazard Rate Analysis")
        
        st.markdown("""
        ### Understanding Hazard Rates
        
        Hazard rate represents instantaneous risk of 
        event occurrence at a specific time, given 
        survival up to that point.
        """)
        
        # Hazard rate parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            hazard_distribution = st.selectbox(
                "Hazard Distribution", 
                [
                    "Constant Hazard", 
                    "Increasing Hazard", 
                    "Decreasing Hazard"
                ],
                key="hazard_dist_select"
            )
        
        with col2:
            hazard_sample_size = st.slider("Sample Size (Hazard)", 500, 10000, 2000, key="hazard_sample_size")
        
        with col3:
            hazard_param = st.slider("Hazard Parameter", 0.1, 5.0, 1.0, 0.1, key="hazard_param")
        
        # Hazard rate simulation
        def simulate_hazard_rate(distribution, sample_size, param):
            time = np.linspace(0, 10, 200)
            
            if distribution == "Constant Hazard":
                # Exponential distribution hazard
                hazard_rate = np.full_like(time, param)
                title = f"Constant Hazard Rate (λ = {param})"
            
            elif distribution == "Increasing Hazard":
                # Weibull-like increasing hazard
                shape = 2.0
                hazard_rate = (shape/param) * (time/param)**(shape-1)
                title = f"Increasing Hazard Rate (η = {param})"
            
            else:  # Decreasing Hazard
                # Inverse power law hazard
                shape = 2.0
                hazard_rate = param / (time + 1)**shape
                title = f"Decreasing Hazard Rate (Scale = {param})"
            
            return time, hazard_rate, title
        
        # Generate hazard rate data
        hazard_time, hazard_curve, hazard_plot_title = simulate_hazard_rate(
            hazard_distribution, hazard_sample_size, hazard_param
        )
        
        # Visualization
        fig_hazard = go.Figure()
        
        # Hazard rate curve
        fig_hazard.add_trace(go.Scatter(
            x=hazard_time,
            y=hazard_curve,
            mode='lines',
            name='Hazard Rate',
            line=dict(color='red', width=3)
        ))
        
        fig_hazard.update_layout(
            title=hazard_plot_title,
            xaxis_title="Time",
            yaxis_title="Hazard Rate",
            yaxis_range=[0, max(hazard_curve) * 1.1]
        )
        
        st.plotly_chart(fig_hazard, use_container_width=True)
        
        # Hazard rate analysis
        st.subheader("Hazard Rate Interpretation")
        
        # Key points on hazard curve
        hazard_points = [np.interp(t, hazard_time, hazard_curve) for t in [0.5, 1.0, 2.0, 5.0]]
        
        hazard_stats_df = pd.DataFrame({
            'Time Point': [0.5, 1.0, 2.0, 5.0],
            'Hazard Rate': hazard_points
        })
        
        st.table(hazard_stats_df)
        
        st.markdown("""
        ### Hazard Rate Insights
        
        - Instantaneous risk of event
        - Varies with time and context
        - Critical in reliability and medical studies
        """)
    
    with tab3:
        st.subheader("Real-World Survival Analysis Applications")
        
        st.markdown("""
        ### Practical Uses of Survival Analysis
        
        Explore survival analysis in:
        - Medical Treatments
        - Machine Reliability
        - Customer Retention
        """)
        
        # Application selection
        application = st.selectbox(
            "Select Application Domain", 
            ["Medical Survival", "Equipment Reliability", "Customer Churn"]
        )
        
        # Parameters
        col1, col2 = st.columns(2)
        
        with col1:
            app_sample_size = st.slider("Sample Size (Application)", 500, 10000, 2000)
        
        with col2:
            app_hazard_param = st.slider("Hazard Parameter (Application)", 0.1, 5.0, 1.0, 0.1)
        
        # Real-world survival data simulation
        def simulate_real_world_survival(application, sample_size, param):
            time = np.linspace(0, 10, 200)
            
            if application == "Medical Survival":
                # Treatment effectiveness
                survival_func = np.exp(-(time/param)**1.5)
                title = f"Medical Treatment Survival Curve (η = {param})"
            
            elif application == "Equipment Reliability":
                # Mechanical failure probability
                survival_func = 1 / (1 + (time/param)**2)
                title = f"Equipment Reliability Survival Curve (η = {param})"
            
            else:  # Customer Churn
                # Customer retention probability
                survival_func = np.exp(-param * time)
                title = f"Customer Retention Survival Curve (λ = {param})"
            
            return time, survival_func, title
        
        # Generate application-specific survival data
        app_time, app_survival_curve, app_plot_title = simulate_real_world_survival(
            application, app_sample_size, app_hazard_param
        )
        
        # Visualization
        fig_app_survival = go.Figure()
        
        # Survival curve
        fig_app_survival.add_trace(go.Scatter(
            x=app_time,
            y=app_survival_curve,
            mode='lines',
            name='Survival Function',
            line=dict(color='green', width=3)
        ))
        
        fig_app_survival.update_layout(
            title=app_plot_title,
            xaxis_title="Time",
            yaxis_title="Survival Probability",
            yaxis_range=[0, 1]
        )
        
        st.plotly_chart(fig_app_survival, use_container_width=True)
        
        # Application-specific survival analysis
        st.subheader(f"{application} Survival Analysis")
        
        # Key points on survival curve
        app_key_times = [0.5, 1.0, 2.0, 5.0]
        app_survival_points = [np.interp(t, app_time, app_survival_curve) for t in app_key_times]
        
        app_stats_df = pd.DataFrame({
            'Time Point': app_key_times,
            'Survival Probability': app_survival_points
        })
        
        st.table(app_stats_df)
        
        st.markdown(f"""
        ### {application} Survival Analysis Insights
        
        **Key Observations**:
        - Demonstrates practical survival analysis techniques
        - Reveals probabilistic behavior over time
        - Helps in understanding risk and retention
        """)

if __name__ == "__main__":
    run_survival_analysis()
