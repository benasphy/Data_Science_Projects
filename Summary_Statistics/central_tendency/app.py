import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import pandas as pd

def run_central_tendency():
    st.title("📊 Central Tendency Explorer")
    
    st.markdown("""
    ### Understanding Measures of Central Tendency
    
    Central tendency measures describe the center or typical value 
    of a dataset. Key measures include:
    
    - Mean: Arithmetic average
    - Median: Middle value
    - Mode: Most frequent value
    
    Each measure provides unique insights into data distribution.
    """)
    
    # Tabs for different central tendency analyses
    tab1, tab2, tab3 = st.tabs([
        "Basic Measures", 
        "Distribution Impact", 
        "Real-World Data"
    ])
    
    with tab1:
        st.subheader("Basic Central Tendency Measures")
        
        # Data generation options
        distribution = st.selectbox(
            "Select Distribution", 
            ["Normal", "Skewed", "Bimodal", "Uniform"]
        )
        
        # Parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sample_size = st.slider("Sample Size", 50, 10000, 1000)
        
        with col2:
            if distribution == "Normal":
                mean = st.slider("Mean", -10.0, 10.0, 0.0, 0.1)
                std_dev = st.slider("Standard Deviation", 0.1, 5.0, 1.0, 0.1)
            elif distribution == "Skewed":
                skew_param = st.slider("Skewness Parameter", 0.1, 10.0, 2.0, 0.1)
            elif distribution == "Bimodal":
                mode1 = st.slider("First Mode", -10.0, 10.0, -3.0, 0.1)
                mode2 = st.slider("Second Mode", -10.0, 10.0, 3.0, 0.1)
            else:  # Uniform
                dist_min = st.slider("Minimum", -10.0, 0.0, -5.0, 0.1)
                dist_max = st.slider("Maximum", 0.0, 10.0, 5.0, 0.1)
        
        with col3:
            num_simulations = st.slider("Number of Simulations", 10, 100, 50)
        
        # Data generation function
        def generate_data(distribution, sample_size):
            np.random.seed(42)
            
            if distribution == "Normal":
                data = np.random.normal(mean, std_dev, sample_size)
            
            elif distribution == "Skewed":
                # Generate right-skewed distribution using gamma distribution
                data = np.random.gamma(skew_param, 1.0, sample_size)
            
            elif distribution == "Bimodal":
                # Create bimodal distribution
                half_size = sample_size // 2
                data = np.concatenate([
                    np.random.normal(mode1, 1.0, half_size),
                    np.random.normal(mode2, 1.0, sample_size - half_size)
                ])
            
            else:  # Uniform
                data = np.random.uniform(dist_min, dist_max, sample_size)
            
            return data
        
        # Simulation to show variability
        def compute_central_tendencies(data):
            return {
                'Mean': np.mean(data),
                'Median': np.median(data),
                'Mode': stats.mode(data, keepdims=True).mode[0]
            }
        
        # Generate and analyze data
        simulation_results = []
        for _ in range(num_simulations):
            data = generate_data(distribution, sample_size)
            simulation_results.append(compute_central_tendencies(data))
        
        # Convert to DataFrame
        results_df = pd.DataFrame(simulation_results)
        
        # Visualization
        fig_measures = go.Figure()
        
        # Add boxplot for each measure
        for measure in ['Mean', 'Median', 'Mode']:
            fig_measures.add_trace(go.Box(
                y=results_df[measure],
                name=measure
            ))
        
        fig_measures.update_layout(
            title=f"Central Tendency Measures for {distribution} Distribution",
            yaxis_title="Value"
        )
        
        st.plotly_chart(fig_measures, use_container_width=True)
        
        # Summary statistics
        st.subheader("Central Tendency Analysis")
        
        summary_stats = results_df.agg(['mean', 'std'])
        st.table(summary_stats)
        
        st.markdown("""
        ### Interpretation
        
        - **Mean**: Sensitive to extreme values
        - **Median**: Robust to outliers
        - **Mode**: Indicates most frequent value
        
        Different measures provide complementary insights into data distribution.
        """)
    
    with tab2:
        st.subheader("Impact of Distribution Shape")
        
        st.markdown("""
        ### How Distribution Shape Affects Central Tendency
        
        Explore how different distribution characteristics 
        influence mean, median, and mode.
        """)
        
        # Distribution shape parameters
        distributions_to_compare = st.multiselect(
            "Select Distributions", 
            ["Symmetric", "Right-Skewed", "Left-Skewed", "Uniform"],
            default=["Symmetric", "Right-Skewed"]
        )
        
        # Comparative analysis
        def compare_distributions(distributions, sample_size):
            np.random.seed(42)
            results = []
            
            for dist in distributions:
                if dist == "Symmetric":
                    data = np.random.normal(0, 1, sample_size)
                elif dist == "Right-Skewed":
                    data = np.random.gamma(2, 1, sample_size)
                elif dist == "Left-Skewed":
                    # Simulate left-skewed using beta distribution
                    data = np.random.beta(2, 5, sample_size)
                else:  # Uniform
                    data = np.random.uniform(-5, 5, sample_size)
                
                results.append({
                    'Distribution': dist,
                    'Mean': np.mean(data),
                    'Median': np.median(data),
                    'Mode': stats.mode(data, keepdims=True).mode[0]
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
                compare_distributions(distributions_to_compare, compare_sample_size)
            )
        
        # Aggregate results
        aggregated_results = pd.concat(comparison_results)
        
        # Visualization
        fig_compare = go.Figure()
        
        for measure in ['Mean', 'Median', 'Mode']:
            fig_compare.add_trace(go.Box(
                x=aggregated_results['Distribution'],
                y=aggregated_results[measure],
                name=measure
            ))
        
        fig_compare.update_layout(
            title="Central Tendency Across Distribution Shapes",
            xaxis_title="Distribution Shape",
            yaxis_title="Value"
        )
        
        st.plotly_chart(fig_compare, use_container_width=True)
        
        # Detailed analysis
        st.subheader("Distribution Shape Analysis")
        
        summary_analysis = aggregated_results.groupby('Distribution').agg(['mean', 'std'])
        st.table(summary_analysis)
        
        st.markdown("""
        ### Insights
        
        - **Symmetric Distributions**: 
          Mean ≈ Median ≈ Mode
        
        - **Right-Skewed Distributions**:
          Mean > Median > Mode
        
        - **Left-Skewed Distributions**:
          Mean < Median < Mode
        
        Distribution shape significantly impacts central tendency measures.
        """)
    
    with tab3:
        st.subheader("Real-World Data Analysis")
        
        st.markdown("""
        ### Applying Central Tendency in Practice
        
        Explore central tendency in various real-world datasets:
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
                'Mean': np.mean(data),
                'Median': np.median(data),
                'Mode': stats.mode(data, keepdims=True).mode[0],
                'Std Dev': np.std(data)
            })
        
        # Convert to DataFrame
        real_world_df = pd.DataFrame(real_world_results)
        
        # Visualization
        fig_real = go.Figure()
        
        for measure in ['Mean', 'Median', 'Mode']:
            fig_real.add_trace(go.Box(
                y=real_world_df[measure],
                name=measure
            ))
        
        fig_real.update_layout(
            title=f"Central Tendency in {dataset} Data",
            yaxis_title="Value"
        )
        
        st.plotly_chart(fig_real, use_container_width=True)
        
        # Summary statistics
        st.subheader(f"{dataset} Data Analysis")
        
        summary_real = real_world_df.agg(['mean', 'std'])
        st.table(summary_real)
        
        st.markdown(f"""
        ### {dataset} Data Insights
        
        **Interpretation**:
        - Demonstrates variability in central tendency measures
        - Shows practical application of statistical concepts
        - Highlights importance of understanding data distribution
        """)

if __name__ == "__main__":
    run_central_tendency()
