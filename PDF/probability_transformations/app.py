import streamlit as st
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from scipy import stats
import pandas as pd

def run_probability_transformations():
    st.title("🔄 Probability Distribution Transformations")
    
    st.markdown("""
    ### Understanding Probability Distribution Transformations
    
    Probability transformations are techniques to convert 
    one probability distribution into another, revealing 
    deep connections between different random variables.
    
    Key Concepts:
    - Inverse Transform Sampling
    - Moment Generating Functions
    - Distribution Mapping
    - Probabilistic Transformations
    """)
    
    # Tabs for different transformation explorations
    tab1, tab2, tab3 = st.tabs([
        "Basic Transformations", 
        "Inverse Transform Sampling", 
        "Real-World Applications"
    ])
    
    with tab1:
        st.subheader("Fundamental Distribution Transformations")
        
        # Transformation parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            transformation_type = st.selectbox(
                "Select Transformation", 
                [
                    "Linear Scaling", 
                    "Exponential Mapping", 
                    "Logarithmic Transformation",
                    "Power Transformation"
                ]
            )
        
        with col2:
            sample_size = st.slider("Sample Size", 100, 10000, 1000)
        
        with col3:
            transformation_param = st.slider("Transformation Parameter", 0.1, 5.0, 1.0, 0.1)
        
        # Generate transformed distributions
        def transform_distribution(transformation_type, sample_size, param):
            np.random.seed(42)
            
            # Base normal distribution
            base_samples = np.random.normal(0, 1, sample_size)
            
            # Apply transformations
            if transformation_type == "Linear Scaling":
                transformed_samples = param * base_samples
                title = f"Linear Scaling (Multiplier = {param})"
            
            elif transformation_type == "Exponential Mapping":
                transformed_samples = np.exp(base_samples * param)
                title = f"Exponential Mapping (Scale = {param})"
            
            elif transformation_type == "Logarithmic Transformation":
                transformed_samples = np.log(np.abs(base_samples) + 1) * param
                title = f"Logarithmic Transformation (Scale = {param})"
            
            else:  # Power Transformation
                transformed_samples = np.power(np.abs(base_samples), param)
                title = f"Power Transformation (Exponent = {param})"
            
            return base_samples, transformed_samples, title
        
        # Generate data
        base_samples, transformed_samples, plot_title = transform_distribution(
            transformation_type, sample_size, transformation_param
        )
        
        # Visualization
        fig_transform = go.Figure()
        
        # Base distribution
        fig_transform.add_trace(go.Histogram(
            x=base_samples,
            name='Base Distribution',
            opacity=0.7
        ))
        
        # Transformed distribution
        fig_transform.add_trace(go.Histogram(
            x=transformed_samples,
            name='Transformed Distribution',
            opacity=0.7
        ))
        
        fig_transform.update_layout(
            title=plot_title,
            xaxis_title="Value",
            yaxis_title="Frequency",
            barmode='overlay'
        )
        
        st.plotly_chart(fig_transform, use_container_width=True)
        
        # Statistical comparison
        st.subheader("Transformation Analysis")
        
        stats_df = pd.DataFrame({
            'Statistic': ['Mean', 'Standard Deviation', 'Skewness', 'Kurtosis'],
            'Base Distribution': [
                np.mean(base_samples),
                np.std(base_samples),
                stats.skew(base_samples),
                stats.kurtosis(base_samples)
            ],
            'Transformed Distribution': [
                np.mean(transformed_samples),
                np.std(transformed_samples),
                stats.skew(transformed_samples),
                stats.kurtosis(transformed_samples)
            ]
        })
        
        st.table(stats_df)
        
        st.markdown("""
        ### Transformation Insights
        
        - Reveals complex distribution behaviors
        - Changes distribution shape and characteristics
        - Useful in data preprocessing and statistical modeling
        """)
    
    with tab2:
        st.subheader("Inverse Transform Sampling")
        
        st.markdown("""
        ### Generating Non-Uniform Distributions
        
        Inverse transform sampling allows generating 
        samples from any cumulative distribution function.
        """)
        
        # Distribution selection
        col1, col2, col3 = st.columns(3)
        
        with col1:
            target_distribution = st.selectbox(
                "Target Distribution", 
                [
                    "Weibull", 
                    "Beta", 
                    "Gamma", 
                    "Log-Normal"
                ]
            )
        
        with col2:
            inv_sample_size = st.slider("Sample Size (Inverse Transform)", 500, 10000, 2000)
        
        with col3:
            distribution_param = st.slider("Distribution Parameter", 0.1, 5.0, 1.0, 0.1)
        
        # Inverse transform sampling
        def inverse_transform_sampling(distribution, sample_size, param):
            np.random.seed(42)
            
            # Uniform samples
            u = np.random.uniform(0, 1, sample_size)
            
            if distribution == "Weibull":
                samples = (-np.log(1 - u))**(1/param)
                title = f"Weibull Distribution (Shape = {param})"
            
            elif distribution == "Beta":
                samples = u**(1/param)
                title = f"Beta Distribution (Shape = {param})"
            
            elif distribution == "Gamma":
                samples = -np.log(u) / param
                title = f"Gamma Distribution (Rate = {param})"
            
            else:  # Log-Normal
                samples = np.exp(stats.norm.ppf(u) * param)
                title = f"Log-Normal Distribution (Scale = {param})"
            
            return samples, title
        
        # Generate samples
        inv_samples, inv_plot_title = inverse_transform_sampling(
            target_distribution, inv_sample_size, distribution_param
        )
        
        # Visualization
        fig_inv_transform = go.Figure()
        
        # Histogram of samples
        fig_inv_transform.add_trace(go.Histogram(
            x=inv_samples,
            name='Transformed Samples',
            opacity=0.7
        ))
        
        fig_inv_transform.update_layout(
            title=inv_plot_title,
            xaxis_title="Value",
            yaxis_title="Frequency"
        )
        
        st.plotly_chart(fig_inv_transform, use_container_width=True)
        
        # Statistical analysis
        st.subheader("Inverse Transform Sampling Analysis")
        
        stats_inv_df = pd.DataFrame({
            'Statistic': ['Mean', 'Standard Deviation', 'Skewness', 'Kurtosis'],
            'Transformed Samples': [
                np.mean(inv_samples),
                np.std(inv_samples),
                stats.skew(inv_samples),
                stats.kurtosis(inv_samples)
            ]
        })
        
        st.table(stats_inv_df)
        
        st.markdown("""
        ### Inverse Transform Sampling Insights
        
        - Generates samples from complex distributions
        - Uses uniform random variables
        - Powerful technique in simulation and sampling
        """)
    
    with tab3:
        st.subheader("Real-World Probability Transformations")
        
        st.markdown("""
        ### Practical Applications of Distribution Transformations
        
        Explore transformations in:
        - Financial Modeling
        - Signal Processing
        - Environmental Data Analysis
        """)
        
        # Application selection
        application = st.selectbox(
            "Select Application Domain", 
            ["Stock Returns", "Signal Normalization", "Climate Data"]
        )
        
        # Parameters
        col1, col2 = st.columns(2)
        
        with col1:
            app_sample_size = st.slider("Sample Size (Application)", 500, 10000, 2000)
        
        with col2:
            app_transformation_param = st.slider("Transformation Parameter (Application)", 0.1, 5.0, 1.0, 0.1)
        
        # Real-world data simulation
        def simulate_real_world_transformation(application, sample_size, param):
            np.random.seed(42)
            
            if application == "Stock Returns":
                # Simulate log returns
                base_samples = np.random.normal(0, 0.02, sample_size)
                transformed_samples = np.exp(base_samples * param) - 1
                title = f"Stock Returns Transformation (Volatility = {param})"
            
            elif application == "Signal Normalization":
                # Simulate signal processing transformation
                base_samples = np.random.laplace(0, 1, sample_size)
                transformed_samples = np.tanh(base_samples * param)
                title = f"Signal Normalization (Scale = {param})"
            
            else:  # Climate Data
                # Simulate temperature anomalies
                base_samples = np.random.normal(0, 1, sample_size)
                transformed_samples = base_samples * param + 1.5
                title = f"Climate Data Transformation (Scaling = {param})"
            
            return base_samples, transformed_samples, title
        
        # Generate application-specific transformation
        base_samples, transformed_samples, app_plot_title = simulate_real_world_transformation(
            application, app_sample_size, app_transformation_param
        )
        
        # Visualization
        fig_app_transform = go.Figure()
        
        # Base distribution
        fig_app_transform.add_trace(go.Histogram(
            x=base_samples,
            name='Base Distribution',
            opacity=0.7
        ))
        
        # Transformed distribution
        fig_app_transform.add_trace(go.Histogram(
            x=transformed_samples,
            name='Transformed Distribution',
            opacity=0.7
        ))
        
        fig_app_transform.update_layout(
            title=app_plot_title,
            xaxis_title="Value",
            yaxis_title="Frequency",
            barmode='overlay'
        )
        
        st.plotly_chart(fig_app_transform, use_container_width=True)
        
        # Statistical analysis
        st.subheader(f"{application} Transformation Analysis")
        
        stats_app_df = pd.DataFrame({
            'Statistic': ['Mean', 'Standard Deviation', 'Skewness', 'Kurtosis'],
            'Base Distribution': [
                np.mean(base_samples),
                np.std(base_samples),
                stats.skew(base_samples),
                stats.kurtosis(base_samples)
            ],
            'Transformed Distribution': [
                np.mean(transformed_samples),
                np.std(transformed_samples),
                stats.skew(transformed_samples),
                stats.kurtosis(transformed_samples)
            ]
        })
        
        st.table(stats_app_df)
        
        st.markdown(f"""
        ### {application} Transformation Insights
        
        **Key Observations**:
        - Demonstrates distribution transformation techniques
        - Shows how data can be reshaped and normalized
        - Reveals underlying statistical properties
        """)

if __name__ == "__main__":
    run_probability_transformations()
