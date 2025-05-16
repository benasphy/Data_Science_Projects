import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import pandas as pd

def run_dispersion_measures():
    st.title("📏 Dispersion Measures Explorer")
    
    st.markdown("""
    ### Understanding Measures of Dispersion
    
    Dispersion measures describe the spread or variability of a dataset.
    Key measures include:
    
    - Variance
    - Standard Deviation
    - Range
    - Interquartile Range (IQR)
    
    These measures help understand data distribution and variability.
    """)
    
    # Tabs for different dispersion analyses
    tab1, tab2, tab3 = st.tabs([
        "Basic Dispersion Measures", 
        "Distribution Comparison", 
        "Real-World Data"
    ])
    
    with tab1:
        st.subheader("Basic Dispersion Measures")
        
        # Data generation options
        distribution = st.selectbox(
            "Select Distribution", 
            ["Normal", "Uniform", "Exponential", "Skewed"]
        )
        
        # Parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sample_size = st.slider("Sample Size", 50, 10000, 1000)
        
        with col2:
            if distribution == "Normal":
                mean = st.slider("Mean", -10.0, 10.0, 0.0, 0.1)
                std_dev = st.slider("Standard Deviation", 0.1, 5.0, 1.0, 0.1)
            elif distribution == "Uniform":
                dist_min = st.slider("Minimum", -10.0, 0.0, -5.0, 0.1)
                dist_max = st.slider("Maximum", 0.0, 10.0, 5.0, 0.1)
            elif distribution == "Exponential":
                rate = st.slider("Rate (λ)", 0.1, 5.0, 1.0, 0.1)
            else:  # Skewed
                skew_param = st.slider("Skewness Parameter", 0.1, 10.0, 2.0, 0.1)
        
        with col3:
            num_simulations = st.slider("Number of Simulations", 10, 100, 50)
        
        # Data generation function
        def generate_data(distribution, sample_size):
            np.random.seed(42)
            
            if distribution == "Normal":
                data = np.random.normal(mean, std_dev, sample_size)
            
            elif distribution == "Uniform":
                data = np.random.uniform(dist_min, dist_max, sample_size)
            
            elif distribution == "Exponential":
                data = np.random.exponential(1/rate, sample_size)
            
            else:  # Skewed
                data = np.random.gamma(skew_param, 1.0, sample_size)
            
            return data
        
        # Compute dispersion measures
        def compute_dispersion_measures(data):
            return {
                'Variance': np.var(data),
                'Standard Deviation': np.std(data),
                'Range': np.ptp(data),
                'Interquartile Range': np.percentile(data, 75) - np.percentile(data, 25)
            }
        
        # Simulation and analysis
        dispersion_results = []
        for _ in range(num_simulations):
            data = generate_data(distribution, sample_size)
            dispersion_results.append(compute_dispersion_measures(data))
        
        # Convert to DataFrame
        results_df = pd.DataFrame(dispersion_results)
        
        # Visualization
        fig_dispersion = go.Figure()
        
        for measure in ['Variance', 'Standard Deviation', 'Range', 'Interquartile Range']:
            fig_dispersion.add_trace(go.Box(
                y=results_df[measure],
                name=measure
            ))
        
        fig_dispersion.update_layout(
            title=f"Dispersion Measures for {distribution} Distribution",
            yaxis_title="Value"
        )
        
        st.plotly_chart(fig_dispersion, use_container_width=True)
        
        # Summary statistics
        st.subheader("Dispersion Measures Analysis")
        
        summary_stats = results_df.agg(['mean', 'std'])
        st.table(summary_stats)
        
        st.markdown("""
        ### Interpretation
        
        - **Variance**: Average squared deviation from the mean
        - **Standard Deviation**: Square root of variance
        - **Range**: Difference between maximum and minimum values
        - **Interquartile Range**: Spread of the middle 50% of data
        
        Different measures provide insights into data variability.
        """)
    
    with tab2:
        st.subheader("Dispersion Across Distributions")
        
        st.markdown("""
        ### Compare Dispersion Measures
        
        Explore how different distributions exhibit varying levels of spread.
        """)
        
        # Distribution comparison
        distributions_to_compare = st.multiselect(
            "Select Distributions", 
            ["Normal", "Uniform", "Exponential", "Skewed"],
            default=["Normal", "Uniform"]
        )
        
        # Comparative analysis
        def compare_distributions_dispersion(distributions, sample_size):
            np.random.seed(42)
            results = []
            
            for dist in distributions:
                if dist == "Normal":
                    data = np.random.normal(0, 1, sample_size)
                elif dist == "Uniform":
                    data = np.random.uniform(-5, 5, sample_size)
                elif dist == "Exponential":
                    data = np.random.exponential(1, sample_size)
                else:  # Skewed
                    data = np.random.gamma(2, 1, sample_size)
                
                results.append({
                    'Distribution': dist,
                    'Variance': np.var(data),
                    'Standard Deviation': np.std(data),
                    'Range': np.ptp(data),
                    'Interquartile Range': np.percentile(data, 75) - np.percentile(data, 25)
                })
            
            return pd.DataFrame(results)
        
        # Parameters
        col1, col2 = st.columns(2)
        
        with col1:
            compare_sample_size = st.slider("Sample Size (Comparison)", 100, 10000, 1000)
        
        with col2:
            num_comparisons = st.slider("Number of Comparisons", 10, 200, 50)
        
        # Generate comparative data
        comparison_results = []
        for _ in range(num_comparisons):
            comparison_results.append(
                compare_distributions_dispersion(distributions_to_compare, compare_sample_size)
            )
        
        # Aggregate results
        aggregated_results = pd.concat(comparison_results)
        
        # Visualization
        fig_compare = go.Figure()
        
        for measure in ['Variance', 'Standard Deviation', 'Range', 'Interquartile Range']:
            fig_compare.add_trace(go.Box(
                x=aggregated_results['Distribution'],
                y=aggregated_results[measure],
                name=measure
            ))
        
        fig_compare.update_layout(
            title="Dispersion Measures Across Distributions",
            xaxis_title="Distribution",
            yaxis_title="Value"
        )
        
        st.plotly_chart(fig_compare, use_container_width=True)
        
        # Detailed analysis
        st.subheader("Distribution Dispersion Analysis")
        
        summary_analysis = aggregated_results.groupby('Distribution').agg(['mean', 'std'])
        st.table(summary_analysis)
        
        st.markdown("""
        ### Insights
        
        - **Normal Distribution**: Symmetric, well-defined spread
        - **Uniform Distribution**: Constant spread across range
        - **Exponential Distribution**: Right-skewed, increasing spread
        - **Skewed Distributions**: Asymmetric variability
        
        Dispersion measures help characterize different distribution types.
        """)
    
    with tab3:
        st.subheader("Real-World Data Variability")
        
        st.markdown("""
        ### Applying Dispersion Measures
        
        Explore variability in real-world datasets:
        - Height Measurements
        - Income Distribution
        - Test Scores
        """)
        
        # Dataset selection
        dataset = st.selectbox(
            "Select Dataset", 
            ["Height", "Income", "Test Scores"]
        )
        
        # Parameters
        col1, col2 = st.columns(2)
        
        with col1:
            sample_size_real = st.slider("Sample Size (Real-World)", 100, 10000, 1000)
        
        with col2:
            num_simulations_real = st.slider("Number of Simulations (Real-World)", 10, 200, 50)
        
        # Real-world data simulation
        def simulate_real_world_data(dataset, sample_size):
            np.random.seed(42)
            
            if dataset == "Height":
                # Height in cm, slightly skewed
                data = np.random.normal(170, 10, sample_size) + np.random.exponential(2, sample_size)
            
            elif dataset == "Income":
                # Income with right-skewed distribution
                data = np.random.lognormal(10, 1, sample_size)
            
            else:  # Test Scores
                # Normally distributed test scores
                data = np.random.normal(75, 10, sample_size)
            
            return data
        
        # Simulation and analysis
        real_world_results = []
        for _ in range(num_simulations_real):
            data = simulate_real_world_data(dataset, sample_size_real)
            
            real_world_results.append({
                'Variance': np.var(data),
                'Standard Deviation': np.std(data),
                'Range': np.ptp(data),
                'Interquartile Range': np.percentile(data, 75) - np.percentile(data, 25)
            })
        
        # Convert to DataFrame
        real_world_df = pd.DataFrame(real_world_results)
        
        # Visualization
        fig_real = go.Figure()
        
        for measure in ['Variance', 'Standard Deviation', 'Range', 'Interquartile Range']:
            fig_real.add_trace(go.Box(
                y=real_world_df[measure],
                name=measure
            ))
        
        fig_real.update_layout(
            title=f"Dispersion Measures in {dataset} Data",
            yaxis_title="Value"
        )
        
        st.plotly_chart(fig_real, use_container_width=True)
        
        # Summary statistics
        st.subheader(f"{dataset} Data Variability")
        
        summary_real = real_world_df.agg(['mean', 'std'])
        st.table(summary_real)
        
        st.markdown(f"""
        ### {dataset} Data Insights
        
        **Interpretation**:
        - Demonstrates variability in real-world data
        - Shows practical application of dispersion measures
        - Highlights importance of understanding data spread
        """)

if __name__ == "__main__":
    run_dispersion_measures()
