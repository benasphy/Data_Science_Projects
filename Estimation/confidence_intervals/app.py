import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import pandas as pd

def run_confidence_intervals():
    st.title("🎯 Confidence Intervals Explorer")
    
    st.markdown("""
    ### Understanding Confidence Intervals
    
    Confidence intervals provide a range of plausible values for a population parameter, 
    based on sample data. They help quantify the uncertainty in statistical estimates.
    
    Key concepts:
    - Measures the precision of an estimate
    - Accounts for sampling variability
    - Provides a range, not a point estimate
    """)
    
    # Tabs for different types of confidence intervals
    tab1, tab2, tab3 = st.tabs([
        "Mean Estimation", 
        "Proportion Estimation", 
        "Simulation"
    ])
    
    with tab1:
        st.subheader("Confidence Interval for Population Mean")
        
        # Parameters for mean estimation
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sample_size = st.slider("Sample Size", 10, 1000, 100)
        
        with col2:
            confidence_level = st.select_slider(
                "Confidence Level", 
                options=[0.80, 0.90, 0.95, 0.99],
                value=0.95,
                key="conf_level_slider_mean"
            )
        
        with col3:
            population_std = st.slider("Population Standard Deviation", 0.1, 10.0, 2.0, 0.1)
        
        # Simulate sample data
        np.random.seed(42)
        sample_mean = 10  # True population mean
        sample_data = np.random.normal(sample_mean, population_std, sample_size)
        
        # Calculate sample statistics
        x_bar = np.mean(sample_data)
        standard_error = population_std / np.sqrt(sample_size)
        
        # Calculate confidence interval
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        margin_of_error = z_score * standard_error
        ci_lower = x_bar - margin_of_error
        ci_upper = x_bar + margin_of_error
        
        # Visualization
        fig = go.Figure()
        
        # Sample data histogram
        fig.add_trace(go.Histogram(
            x=sample_data,
            name='Sample Data',
            opacity=0.7
        ))
        
        # Compute y_max for vertical lines
        counts, _ = np.histogram(sample_data, bins='auto')
        y_max = 1.1 * counts.max()
        # Add vertical lines for mean and confidence interval
        fig.add_shape(
            type="line",
            x0=x_bar,
            y0=0,
            x1=x_bar,
            y1=y_max,
            line=dict(color="red", width=2, dash="dash")
        )
        
        fig.add_shape(
            type="line",
            x0=ci_lower,
            y0=0,
            x1=ci_lower,
            y1=y_max,
            line=dict(color="green", width=2, dash="dot")
        )
        
        fig.add_shape(
            type="line",
            x0=ci_upper,
            y0=0,
            x1=ci_upper,
            y1=y_max,
            line=dict(color="green", width=2, dash="dot")
        )
        
        fig.update_layout(
            title=f"Sample Distribution with {confidence_level*100:.0f}% Confidence Interval",
            xaxis_title="Value",
            yaxis_title="Frequency"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display results
        st.markdown(f"""
        ### Confidence Interval Results
        
        - **Sample Mean**: {x_bar:.4f}
        - **Confidence Level**: {confidence_level*100:.0f}%
        - **Margin of Error**: ±{margin_of_error:.4f}
        - **Confidence Interval**: [{ci_lower:.4f}, {ci_upper:.4f}]
        
        #### Interpretation
        We are {confidence_level*100:.0f}% confident that the true population mean 
        falls within the interval [{ci_lower:.4f}, {ci_upper:.4f}].
        """)
    
    with tab2:
        st.subheader("Confidence Interval for Population Proportion")
        
        # Parameters for proportion estimation
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sample_size_prop = st.slider("Sample Size", 50, 10000, 1000)
        
        with col2:
            true_proportion = st.slider("True Proportion", 0.0, 1.0, 0.5, 0.01)
        
        with col3:
            confidence_level_prop = st.select_slider(
                "Confidence Level", 
                options=[0.80, 0.90, 0.95, 0.99],
                value=0.95,
                key="conf_level_slider_prop"
            )
        
        # Simulate sample data
        np.random.seed(42)
        sample_data_prop = np.random.binomial(1, true_proportion, sample_size_prop)
        
        # Calculate sample proportion
        sample_prop = np.mean(sample_data_prop)
        
        # Calculate confidence interval (Wilson score interval)
        z = stats.norm.ppf((1 + confidence_level_prop) / 2)
        
        def wilson_interval(p, n, confidence):
            z = stats.norm.ppf((1 + confidence) / 2)
            denominator = 1 + z**2/n
            center_adjusted_prop = p + z**2 / (2*n)
            prop_variance = p * (1 - p) / n + z**2 / (4*n**2)
            
            lower = (center_adjusted_prop - z * np.sqrt(prop_variance)) / denominator
            upper = (center_adjusted_prop + z * np.sqrt(prop_variance)) / denominator
            
            return max(0, lower), min(1, upper)
        
        # Calculate interval
        ci_lower_prop, ci_upper_prop = wilson_interval(sample_prop, sample_size_prop, confidence_level_prop)
        
        # Visualization
        fig_prop = go.Figure()
        
        # Pie chart of sample data
        prop_counts = pd.Series(sample_data_prop).value_counts(normalize=True)
        
        fig_prop = go.Figure(data=[go.Pie(
            labels=['Success', 'Failure'],
            values=prop_counts.values,
            textinfo='percent+label'
        )])
        
        fig_prop.update_layout(
            title=f"Sample Proportion with {confidence_level_prop*100:.0f}% Confidence Interval"
        )
        
        st.plotly_chart(fig_prop, use_container_width=True)
        
        # Display results
        st.markdown(f"""
        ### Proportion Confidence Interval Results
        
        - **Sample Size**: {sample_size_prop}
        - **Sample Proportion**: {sample_prop:.4f}
        - **Confidence Level**: {confidence_level_prop*100:.0f}%
        - **Confidence Interval**: [{ci_lower_prop:.4f}, {ci_upper_prop:.4f}]
        
        #### Interpretation
        We are {confidence_level_prop*100:.0f}% confident that the true population 
        proportion falls within the interval [{ci_lower_prop:.4f}, {ci_upper_prop:.4f}].
        """)
    
    with tab3:
        st.subheader("Confidence Interval Simulation")
        
        st.markdown("""
        ### Exploring Confidence Interval Behavior
        
        This simulation demonstrates how confidence intervals change with:
        - Sample size
        - Confidence level
        - Underlying population parameters
        """)
        
        # Simulation parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            num_simulations = st.slider("Number of Simulations", 10, 1000, 100)
        
        with col2:
            sim_sample_size = st.slider("Sample Size per Simulation", 10, 500, 100)
        
        with col3:
            true_mean = st.slider("True Population Mean", -10.0, 10.0, 0.0, 0.1)
        
        # Run simulations
        sim_results = []
        
        for _ in range(num_simulations):
            # Generate sample
            sample = np.random.normal(true_mean, 1, sim_sample_size)
            
            # Calculate confidence interval
            sample_mean = np.mean(sample)
            sample_std = np.std(sample, ddof=1)
            standard_error = sample_std / np.sqrt(sim_sample_size)
            
            # 95% confidence interval
            t_value = stats.t.ppf(0.975, df=sim_sample_size-1)
            ci_lower = sample_mean - t_value * standard_error
            ci_upper = sample_mean + t_value * standard_error
            
            sim_results.append({
                'Sample Mean': sample_mean,
                'CI Lower': ci_lower,
                'CI Upper': ci_upper,
                'Contains True Mean': ci_lower <= true_mean <= ci_upper
            })
        
        # Convert to DataFrame
        sim_df = pd.DataFrame(sim_results)
        
        # Visualization
        fig_sim = go.Figure()
        
        # Scatter plot of confidence intervals
        for i, row in sim_df.iterrows():
            color = 'green' if row['Contains True Mean'] else 'red'
            fig_sim.add_trace(go.Scatter(
                x=[row['CI Lower'], row['CI Upper']],
                y=[i, i],
                mode='lines',
                line=dict(color=color, width=2),
                showlegend=False
            ))
        
        # True mean line
        fig_sim.add_shape(
            type="line",
            x0=true_mean,
            y0=-1,
            x1=true_mean,
            y1=num_simulations,
            line=dict(color="blue", width=2, dash="dash")
        )
        
        fig_sim.update_layout(
            title="Confidence Intervals Across Simulations",
            xaxis_title="Value",
            yaxis_title="Simulation Number",
            height=600
        )
        
        st.plotly_chart(fig_sim, use_container_width=True)
        
        # Summary statistics
        coverage_rate = sim_df['Contains True Mean'].mean()
        
        st.markdown(f"""
        ### Simulation Summary
        
        - **Number of Simulations**: {num_simulations}
        - **True Population Mean**: {true_mean}
        - **Coverage Rate**: {coverage_rate:.2%}
        
        #### Interpretation
        The coverage rate shows the proportion of confidence intervals 
        that contain the true population mean. Theoretically, this should 
        be close to the chosen confidence level (95% in this case).
        
        A coverage rate significantly different from the expected level 
        might indicate:
        - Violation of underlying assumptions
        - Non-representative sampling
        - Need for more robust estimation methods
        """)

if __name__ == "__main__":
    run_confidence_intervals()
