import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import pandas as pd

def run_binomial_experiments():
    st.title("🎲 Binomial Experiments Simulator")
    
    st.markdown("""
    ### Understanding Binomial Experiments
    
    A binomial experiment is a statistical experiment with the following characteristics:
    - Fixed number of trials
    - Each trial is independent
    - Each trial has two possible outcomes (success/failure)
    - Probability of success remains constant across trials
    
    Explore how different parameters affect binomial distributions:
    """)
    
    # Experiment setup
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_trials = st.slider("Number of Trials (n)", 1, 100, 20)
    
    with col2:
        success_prob = st.slider("Probability of Success (p)", 0.0, 1.0, 0.5, 0.01)
    
    with col3:
        num_experiments = st.slider("Number of Experiments", 10, 1000, 100)
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs([
        "Probability Distribution", 
        "Experiment Simulation", 
        "Confidence Intervals"
    ])
    
    with tab1:
        st.subheader("Binomial Probability Distribution")
        
        # Calculate PMF
        x = list(range(num_trials + 1))
        pmf = [stats.binom.pmf(k, num_trials, success_prob) for k in x]
        
        # Create bar plot of PMF
        fig = go.Figure(data=[
            go.Bar(
                x=x, 
                y=pmf, 
                text=[f'{p:.2%}' for p in pmf],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=f"Binomial PMF (n = {num_trials}, p = {success_prob:.2f})",
            xaxis_title="Number of Successes",
            yaxis_title="Probability"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Expected value and variance
        expected_value = num_trials * success_prob
        variance = num_trials * success_prob * (1 - success_prob)
        
        st.markdown(f"""
        ### Distribution Properties
        
        - **Expected Value (Mean)**: {expected_value:.2f}
        - **Variance**: {variance:.2f}
        - **Standard Deviation**: {np.sqrt(variance):.2f}
        
        #### Probability Calculation
        
        The probability of exactly k successes in n trials is given by:
        
        $$P(X = k) = \binom{{n}}{{k}} p^k (1-p)^{{n-k}}$$
        
        Where:
        - n = number of trials
        - k = number of successes
        - p = probability of success in each trial
        """)
    
    with tab2:
        st.subheader("Binomial Experiment Simulation")
        
        # Run multiple experiments
        experiments = np.random.binomial(
            n=num_trials, 
            p=success_prob, 
            size=num_experiments
        )
        
        # Create histogram of experiment results
        fig = go.Figure(data=[
            go.Histogram(
                x=experiments, 
                nbinsx=num_trials + 1,
                histnorm='probability'
            )
        ])
        
        fig.update_layout(
            title=f"Binomial Experiment Simulation (n = {num_trials}, p = {success_prob:.2f})",
            xaxis_title="Number of Successes",
            yaxis_title="Proportion of Experiments"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        summary = pd.Series(experiments)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean", f"{summary.mean():.2f}")
        col2.metric("Std Dev", f"{summary.std():.2f}")
        col3.metric("Median", f"{summary.median():.2f}")
        
        # Theoretical vs Empirical comparison
        st.subheader("Theoretical vs Empirical Distribution")
        
        # Calculate theoretical PMF
        x_theo = list(range(num_trials + 1))
        pmf_theo = [stats.binom.pmf(k, num_trials, success_prob) for k in x_theo]
        
        # Create comparison plot
        fig_comp = go.Figure()
        
        # Theoretical distribution
        fig_comp.add_trace(go.Scatter(
            x=x_theo, 
            y=pmf_theo, 
            mode='lines+markers',
            name='Theoretical PMF',
            line=dict(color='red', dash='dot')
        ))
        
        # Empirical distribution
        empirical_dist = summary.value_counts(normalize=True).sort_index()
        
        fig_comp.add_trace(go.Bar(
            x=empirical_dist.index, 
            y=empirical_dist.values,
            name='Empirical Distribution',
            opacity=0.7
        ))
        
        fig_comp.update_layout(
            title="Theoretical vs Empirical Distribution",
            xaxis_title="Number of Successes",
            yaxis_title="Probability/Proportion"
        )
        
        st.plotly_chart(fig_comp, use_container_width=True)
    
    with tab3:
        st.subheader("Confidence Intervals")
        
        st.markdown("""
        ### Understanding Confidence Intervals for Binomial Proportion
        
        Confidence intervals help us estimate the true probability of success 
        based on our sample of experiments.
        """)
        
        # Confidence level selection
        confidence_level = st.select_slider(
            "Confidence Level", 
            options=[0.80, 0.90, 0.95, 0.99],
            value=0.95
        )
        
        # Calculate confidence interval
        successes = np.sum(experiments >= num_trials * success_prob)
        sample_prop = successes / num_experiments
        
        # Wilson score interval (more accurate for small samples)
        def wilson_interval(p, n, confidence):
            z = stats.norm.ppf((1 + confidence) / 2)
            denominator = 1 + z**2/n
            center_adjusted_prop = p + z**2 / (2*n)
            prop_variance = p * (1 - p) / n + z**2 / (4*n**2)
            
            lower = (center_adjusted_prop - z * np.sqrt(prop_variance)) / denominator
            upper = (center_adjusted_prop + z * np.sqrt(prop_variance)) / denominator
            
            return max(0, lower), min(1, upper)
        
        # Calculate interval
        lower, upper = wilson_interval(sample_prop, num_experiments, confidence_level)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Sample Proportion", f"{sample_prop:.2%}")
        col2.metric("Lower Bound", f"{lower:.2%}")
        col3.metric("Upper Bound", f"{upper:.2%}")
        
        # Visualization of confidence interval
        fig_ci = go.Figure()
        
        # Add point estimate
        fig_ci.add_trace(go.Scatter(
            x=[sample_prop],
            y=[0.5],
            mode='markers',
            marker=dict(color='red', size=20),
            name='Point Estimate'
        ))
        
        # Add confidence interval
        fig_ci.add_trace(go.Scatter(
            x=[lower, upper],
            y=[0.5, 0.5],
            mode='lines',
            line=dict(color='blue', width=5),
            name=f'{confidence_level*100:.0f}% Confidence Interval'
        ))
        
        fig_ci.update_layout(
            title=f"{confidence_level*100:.0f}% Confidence Interval for Proportion",
            xaxis_title="Proportion",
            yaxis_title="",
            yaxis=dict(showticklabels=False),
            height=300
        )
        
        st.plotly_chart(fig_ci, use_container_width=True)
        
        st.markdown("""
        ### Interpreting the Confidence Interval
        
        - The interval provides a range of plausible values for the true probability of success
        - A {confidence_level*100:.0f}% confidence interval means that if we repeated this experiment 
          many times, {confidence_level*100:.0f}% of the intervals would contain the true probability
        
        #### Factors Affecting Confidence Interval:
        - Sample size
        - Variability in the data
        - Chosen confidence level
        """)

if __name__ == "__main__":
    run_binomial_experiments()
