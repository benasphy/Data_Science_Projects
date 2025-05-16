import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import pandas as pd
import seaborn as sns

def run_correlation_analysis():
    st.title("🔗 Correlation Analysis Explorer")
    
    st.markdown("""
    ### Understanding Correlation Measures
    
    Correlation measures the strength and direction of relationships 
    between variables. Key correlation coefficients include:
    
    - Pearson Correlation (Linear Relationship)
    - Spearman Rank Correlation (Monotonic Relationship)
    - Kendall's Tau (Rank-Based Correlation)
    
    Explore how different types of relationships can be quantified.
    """)
    
    # Tabs for different correlation analyses
    tab1, tab2, tab3 = st.tabs([
        "Basic Correlation", 
        "Correlation Types", 
        "Real-World Data"
    ])
    
    with tab1:
        st.subheader("Basic Correlation Exploration")
        
        # Relationship type selection
        relationship_type = st.selectbox(
            "Select Relationship Type", 
            ["Linear", "Nonlinear", "Noisy Linear", "No Correlation"]
        )
        
        # Parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sample_size = st.slider("Sample Size", 50, 10000, 1000)
        
        with col2:
            noise_level = st.slider("Noise Level", 0.0, 2.0, 0.5, 0.1)
        
        with col3:
            num_simulations = st.slider("Number of Simulations", 10, 100, 50)
        
        # Data generation function
        def generate_correlated_data(relationship_type, sample_size, noise_level):
            np.random.seed(42)
            
            if relationship_type == "Linear":
                x = np.linspace(0, 10, sample_size)
                y = 2 * x + np.random.normal(0, noise_level, sample_size)
            
            elif relationship_type == "Nonlinear":
                x = np.linspace(0, 10, sample_size)
                y = x**2 + np.random.normal(0, noise_level * 10, sample_size)
            
            elif relationship_type == "Noisy Linear":
                x = np.random.uniform(0, 10, sample_size)
                y = 2 * x + np.random.normal(0, noise_level * 5, sample_size)
            
            else:  # No Correlation
                x = np.random.uniform(0, 10, sample_size)
                y = np.random.uniform(0, 10, sample_size)
            
            return x, y
        
        # Compute correlation measures
        def compute_correlations(x, y):
            return {
                'Pearson Correlation': stats.pearsonr(x, y)[0],
                'Spearman Correlation': stats.spearmanr(x, y)[0],
                'Kendall Tau': stats.kendalltau(x, y)[0]
            }
        
        # Simulation and analysis
        correlation_results = []
        for _ in range(num_simulations):
            x, y = generate_correlated_data(relationship_type, sample_size, noise_level)
            correlation_results.append(compute_correlations(x, y))
        
        # Convert to DataFrame
        results_df = pd.DataFrame(correlation_results)
        
        # Visualization
        fig_corr = go.Figure()
        
        for measure in ['Pearson Correlation', 'Spearman Correlation', 'Kendall Tau']:
            fig_corr.add_trace(go.Box(
                y=results_df[measure],
                name=measure
            ))
        
        fig_corr.update_layout(
            title=f"Correlation Measures for {relationship_type} Relationship",
            yaxis_title="Correlation Coefficient"
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Scatter plot of last generated data
        fig_scatter = px.scatter(
            x=x, y=y, 
            title=f"{relationship_type} Relationship Scatter Plot",
            labels={'x': 'X Variable', 'y': 'Y Variable'}
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Summary statistics
        st.subheader("Correlation Analysis")
        
        summary_stats = results_df.agg(['mean', 'std'])
        st.table(summary_stats)
        
        st.markdown("""
        ### Interpretation
        
        - **Pearson Correlation**: Measures linear relationship
        - **Spearman Correlation**: Measures monotonic relationship
        - **Kendall Tau**: Measures ordinal relationship
        
        Correlation coefficients range from -1 to 1:
        - 1: Perfect positive correlation
        - 0: No correlation
        - -1: Perfect negative correlation
        """)
    
    with tab2:
        st.subheader("Correlation Types Comparison")
        
        st.markdown("""
        ### Exploring Different Correlation Types
        
        Compare how different relationship types affect correlation measures.
        """)
        
        # Relationship types to compare
        relationship_types = st.multiselect(
            "Select Relationship Types", 
            ["Linear", "Nonlinear", "Exponential", "Logarithmic"],
            default=["Linear", "Nonlinear"]
        )
        
        # Comparative analysis
        def compare_correlation_types(relationship_types, sample_size):
            np.random.seed(42)
            results = []
            
            for rel_type in relationship_types:
                if rel_type == "Linear":
                    x = np.linspace(0, 10, sample_size)
                    y = 2 * x + np.random.normal(0, 0.5, sample_size)
                
                elif rel_type == "Nonlinear":
                    x = np.linspace(0, 10, sample_size)
                    y = x**2 + np.random.normal(0, 5, sample_size)
                
                elif rel_type == "Exponential":
                    x = np.linspace(0, 10, sample_size)
                    y = np.exp(x) + np.random.normal(0, 10, sample_size)
                
                else:  # Logarithmic
                    x = np.linspace(1, 10, sample_size)
                    y = np.log(x) + np.random.normal(0, 0.5, sample_size)
                
                correlations = compute_correlations(x, y)
                correlations['Relationship Type'] = rel_type
                results.append(correlations)
            
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
                compare_correlation_types(relationship_types, compare_sample_size)
            )
        
        # Aggregate results
        aggregated_results = pd.concat(comparison_results)
        
        # Visualization
        fig_compare = go.Figure()
        
        for measure in ['Pearson Correlation', 'Spearman Correlation', 'Kendall Tau']:
            fig_compare.add_trace(go.Box(
                x=aggregated_results['Relationship Type'],
                y=aggregated_results[measure],
                name=measure
            ))
        
        fig_compare.update_layout(
            title="Correlation Measures Across Relationship Types",
            xaxis_title="Relationship Type",
            yaxis_title="Correlation Coefficient"
        )
        
        st.plotly_chart(fig_compare, use_container_width=True)
        
        # Detailed analysis
        st.subheader("Correlation Type Analysis")
        
        summary_analysis = aggregated_results.groupby('Relationship Type').agg(['mean', 'std'])
        st.table(summary_analysis)
        
        st.markdown("""
        ### Insights
        
        - **Linear Relationships**: 
          High Pearson correlation, consistent across measures
        
        - **Nonlinear Relationships**:
          Lower Pearson correlation
          Higher Spearman/Kendall correlations
        
        - **Complex Relationships**:
          Highlight limitations of linear correlation measures
        """)
    
    with tab3:
        st.subheader("Real-World Correlation Analysis")
        
        st.markdown("""
        ### Exploring Correlations in Practical Datasets
        
        Analyze correlations in various domains:
        - Economic Indicators
        - Environmental Measurements
        - Social Science Data
        """)
        
        # Dataset selection
        dataset = st.selectbox(
            "Select Dataset", 
            ["Economic Growth", "Climate Data", "Social Indicators"]
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
            
            if dataset == "Economic Growth":
                # GDP and Investment correlation
                x = np.random.normal(50000, 10000, sample_size)  # GDP per capita
                y = 0.7 * x + np.random.normal(0, 5000, sample_size)  # Investment
            
            elif dataset == "Climate Data":
                # Temperature and CO2 levels
                x = np.linspace(1900, 2020, sample_size)  # Years
                y = 0.5 * x - 1000 + np.random.normal(0, 50, sample_size)  # CO2 levels
            
            else:  # Social Indicators
                # Education and Income correlation
                x = np.random.normal(12, 2, sample_size)  # Years of education
                y = 5000 * x + np.random.normal(0, 10000, sample_size)  # Income
            
            return x, y
        
        # Simulation and analysis
        real_world_results = []
        for _ in range(num_simulations_real):
            x, y = simulate_real_world_data(dataset, sample_size_real)
            real_world_results.append(compute_correlations(x, y))
        
        # Convert to DataFrame
        real_world_df = pd.DataFrame(real_world_results)
        
        # Visualization
        fig_real = go.Figure()
        
        for measure in ['Pearson Correlation', 'Spearman Correlation', 'Kendall Tau']:
            fig_real.add_trace(go.Box(
                y=real_world_df[measure],
                name=measure
            ))
        
        fig_real.update_layout(
            title=f"Correlation Measures in {dataset}",
            yaxis_title="Correlation Coefficient"
        )
        
        st.plotly_chart(fig_real, use_container_width=True)
        
        # Scatter plot of last generated data
        fig_scatter_real = px.scatter(
            x=x, y=y, 
            title=f"{dataset} Scatter Plot",
            labels={'x': 'X Variable', 'y': 'Y Variable'}
        )
        
        st.plotly_chart(fig_scatter_real, use_container_width=True)
        
        # Summary statistics
        st.subheader(f"{dataset} Correlation Analysis")
        
        summary_real = real_world_df.agg(['mean', 'std'])
        st.table(summary_real)
        
        st.markdown(f"""
        ### {dataset} Data Insights
        
        **Interpretation**:
        - Demonstrates correlation in complex real-world scenarios
        - Shows how variables can be interconnected
        - Highlights the importance of choosing appropriate correlation measures
        """)

if __name__ == "__main__":
    run_correlation_analysis()
