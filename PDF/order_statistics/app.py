import streamlit as st
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from scipy import stats
import pandas as pd

def run_order_statistics():
    st.title("📊 Order Statistics Explorer")
    
    st.markdown("""
    ### Understanding Order Statistics
    
    Order statistics are the values of a sample arranged in ascending 
    or descending order. They provide insights into data distribution, 
    extreme values, and probabilistic rankings.
    
    Key Concepts:
    - Minimum and Maximum
    - Median and Quartiles
    - Extreme Value Theory
    - Probabilistic Rankings
    """)
    
    # Tabs for different order statistics analyses
    tab1, tab2, tab3 = st.tabs([
        "Basic Order Statistics", 
        "Extreme Value Analysis", 
        "Real-World Applications"
    ])
    
    with tab1:
        st.subheader("Fundamental Order Statistics")
        
        # Distribution parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            distribution_type = st.selectbox(
                "Select Distribution", 
                [
                    "Normal", 
                    "Uniform", 
                    "Exponential",
                    "Log-Normal"
                ]
            )
        
        with col2:
            sample_size = st.slider("Sample Size", 100, 10000, 1000)
        
        with col3:
            rank_selection = st.slider("Order Statistic Rank", 1, 10, 3)
        
        # Generate order statistics
        def generate_order_statistics(distribution_type, sample_size, rank):
            np.random.seed(42)
            
            # Generate samples based on distribution
            if distribution_type == "Normal":
                samples = np.random.normal(0, 1, sample_size)
                title = "Normal Distribution Order Statistics"
            
            elif distribution_type == "Uniform":
                samples = np.random.uniform(0, 1, sample_size)
                title = "Uniform Distribution Order Statistics"
            
            elif distribution_type == "Exponential":
                samples = np.random.exponential(1, sample_size)
                title = "Exponential Distribution Order Statistics"
            
            else:  # Log-Normal
                samples = np.random.lognormal(0, 1, sample_size)
                title = "Log-Normal Distribution Order Statistics"
            
            # Sort samples
            sorted_samples = np.sort(samples)
            
            # Select specific order statistic
            order_stat_samples = sorted_samples[rank-1::rank]
            
            return samples, sorted_samples, order_stat_samples, title
        
        # Generate data
        base_samples, sorted_samples, order_stat_samples, plot_title = generate_order_statistics(
            distribution_type, sample_size, rank_selection
        )
        
        # Visualization
        fig_order_stats = go.Figure()
        
        # Original distribution
        fig_order_stats.add_trace(go.Histogram(
            x=base_samples,
            name='Original Distribution',
            opacity=0.7
        ))
        
        # Sorted distribution
        fig_order_stats.add_trace(go.Histogram(
            x=sorted_samples,
            name='Sorted Distribution',
            opacity=0.7
        ))
        
        # Order statistic highlights
        fig_order_stats.add_trace(go.Histogram(
            x=order_stat_samples,
            name=f'{rank_selection}th Order Statistic',
            opacity=1,
            marker_color='red'
        ))
        
        fig_order_stats.update_layout(
            title=plot_title,
            xaxis_title="Value",
            yaxis_title="Frequency",
            barmode='overlay'
        )
        
        st.plotly_chart(fig_order_stats, use_container_width=True)
        
        # Statistical analysis
        st.subheader("Order Statistics Analysis")
        
        stats_df = pd.DataFrame({
            'Statistic': [
                'Minimum', 
                'Maximum', 
f'{rank_selection}th Order Statistic', 
                'Median', 
                'Mean'
            ],
            'Value': [
                np.min(sorted_samples),
                np.max(sorted_samples),
                order_stat_samples[0],
                np.median(sorted_samples),
                np.mean(sorted_samples)
            ]
        })
        
        st.table(stats_df)
        
        st.markdown("""
        ### Order Statistics Insights
        
        - Reveals data distribution characteristics
        - Helps understand extreme values
        - Useful in ranking and probabilistic analysis
        """)
    
    with tab2:
        st.subheader("Extreme Value Analysis")
        
        st.markdown("""
        ### Exploring Extreme Value Distributions
        
        Investigate how extreme values behave in different 
        probability distributions.
        """)
        
        # Extreme value parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            extreme_distribution = st.selectbox(
                "Extreme Value Distribution", 
                [
                    "Gumbel (Type I)", 
                    "Fréchet (Type II)", 
                    "Weibull (Type III)"
                ]
            )
        
        with col2:
            extreme_sample_size = st.slider("Sample Size (Extreme)", 500, 10000, 2000)
        
        with col3:
            extreme_param = st.slider("Shape Parameter", 0.1, 5.0, 1.0, 0.1)
        
        # Extreme value distribution simulation
        def simulate_extreme_distribution(distribution, sample_size, param):
            np.random.seed(42)
            
            if distribution == "Gumbel (Type I)":
                # Location-scale distribution
                location = 0
                scale = param
                samples = np.random.gumbel(location, scale, sample_size)
                title = f"Gumbel Distribution (Scale = {param})"
            
            elif distribution == "Fréchet (Type II)":
                # Heavy-tailed distribution
                shape = param
                samples = np.random.pareto(shape, sample_size)
                title = f"Fréchet Distribution (Shape = {param})"
            
            else:  # Weibull (Type III)
                # Bounded distribution
                shape = param
                samples = np.random.weibull(shape, sample_size)
                title = f"Weibull Distribution (Shape = {param})"
            
            # Sort samples
            sorted_samples = np.sort(samples)
            
            return samples, sorted_samples, title
        
        # Generate extreme value data
        extreme_samples, sorted_extreme_samples, extreme_plot_title = simulate_extreme_distribution(
            extreme_distribution, extreme_sample_size, extreme_param
        )
        
        # Visualization
        fig_extreme = go.Figure()
        
        # Original distribution
        fig_extreme.add_trace(go.Histogram(
            x=extreme_samples,
            name='Original Distribution',
            opacity=0.7
        ))
        
        # Sorted distribution
        fig_extreme.add_trace(go.Histogram(
            x=sorted_extreme_samples,
            name='Sorted Distribution',
            opacity=0.7
        ))
        
        fig_extreme.update_layout(
            title=extreme_plot_title,
            xaxis_title="Value",
            yaxis_title="Frequency",
            barmode='overlay'
        )
        
        st.plotly_chart(fig_extreme, use_container_width=True)
        
        # Extreme value analysis
        st.subheader("Extreme Value Analysis")
        
        extreme_stats_df = pd.DataFrame({
            'Statistic': [
                'Minimum', 
                'Maximum', 
                '95th Percentile', 
                '99th Percentile', 
                'Mean'
            ],
            'Value': [
                np.min(sorted_extreme_samples),
                np.max(sorted_extreme_samples),
                np.percentile(sorted_extreme_samples, 95),
                np.percentile(sorted_extreme_samples, 99),
                np.mean(sorted_extreme_samples)
            ]
        })
        
        st.table(extreme_stats_df)
        
        st.markdown("""
        ### Extreme Value Insights
        
        - Characterizes tail behavior of distributions
        - Important in risk assessment
        - Helps understand rare events
        """)
    
    with tab3:
        st.subheader("Real-World Order Statistics Applications")
        
        st.markdown("""
        ### Practical Uses of Order Statistics
        
        Explore order statistics in:
        - Environmental Monitoring
        - Quality Control
        - Performance Evaluation
        """)
        
        # Application selection
        application = st.selectbox(
            "Select Application Domain", 
            ["Climate Data", "Manufacturing Quality", "Athletic Performance"]
        )
        
        # Parameters
        col1, col2 = st.columns(2)
        
        with col1:
            app_sample_size = st.slider("Sample Size (Application)", 500, 10000, 2000)
        
        with col2:
            app_order_rank = st.slider("Order Statistic Rank (Application)", 1, 20, 5)
        
        # Real-world data simulation
        def simulate_real_world_order_statistics(application, sample_size, rank):
            np.random.seed(42)
            
            if application == "Climate Data":
                # Simulate temperature variations
                samples = np.random.normal(20, 5, sample_size)
                title = "Temperature Variations Order Statistics"
            
            elif application == "Manufacturing Quality":
                # Simulate product measurements
                samples = np.random.lognormal(0, 0.5, sample_size)
                title = "Product Measurement Order Statistics"
            
            else:  # Athletic Performance
                # Simulate performance times
                samples = np.random.exponential(10, sample_size)
                title = "Athletic Performance Order Statistics"
            
            # Sort samples
            sorted_samples = np.sort(samples)
            
            # Select specific order statistic
            order_stat_samples = sorted_samples[rank-1::rank]
            
            return samples, sorted_samples, order_stat_samples, title
        
        # Generate application-specific order statistics
        base_samples, sorted_samples, order_stat_samples, app_plot_title = simulate_real_world_order_statistics(
            application, app_sample_size, app_order_rank
        )
        
        # Visualization
        fig_app_order_stats = go.Figure()
        
        # Original distribution
        fig_app_order_stats.add_trace(go.Histogram(
            x=base_samples,
            name='Original Distribution',
            opacity=0.7
        ))
        
        # Sorted distribution
        fig_app_order_stats.add_trace(go.Histogram(
            x=sorted_samples,
            name='Sorted Distribution',
            opacity=0.7
        ))
        
        # Order statistic highlights
        fig_app_order_stats.add_trace(go.Histogram(
            x=order_stat_samples,
            name=f'{app_order_rank}th Order Statistic',
            opacity=1,
            marker_color='red'
        ))
        
        fig_app_order_stats.update_layout(
            title=app_plot_title,
            xaxis_title="Value",
            yaxis_title="Frequency",
            barmode='overlay'
        )
        
        st.plotly_chart(fig_app_order_stats, use_container_width=True)
        
        # Statistical analysis
        st.subheader(f"{application} Order Statistics Analysis")
        
        stats_app_df = pd.DataFrame({
            'Statistic': [
                'Minimum', 
                'Maximum', 
f'{app_order_rank}th Order Statistic', 
                'Median', 
                'Mean'
            ],
            'Value': [
                np.min(sorted_samples),
                np.max(sorted_samples),
                order_stat_samples[0],
                np.median(sorted_samples),
                np.mean(sorted_samples)
            ]
        })
        
        st.table(stats_app_df)
        
        st.markdown(f"""
        ### {application} Order Statistics Insights
        
        **Key Observations**:
        - Demonstrates practical use of order statistics
        - Reveals distribution characteristics
        - Helps in understanding variability and extreme values
        """)

if __name__ == "__main__":
    run_order_statistics()
