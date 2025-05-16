import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import pandas as pd

def run_law_of_large_numbers():
    st.title("📊 Law of Large Numbers Explorer")
    
    st.markdown("""
    ### Understanding the Law of Large Numbers
    
    The Law of Large Numbers states that as the number of trials increases, 
    the sample mean converges to the expected (true) population mean.
    
    Key Insights:
    - Convergence of sample mean to population mean
    - Applies to independent, identically distributed random variables
    - Fundamental to probability and statistics
    """)
    
    # Tabs for different LLN explorations
    tab1, tab2, tab3 = st.tabs([
        "Convergence Demonstration", 
        "Different Distributions", 
        "Practical Applications"
    ])
    
    with tab1:
        st.subheader("Mean Convergence Demonstration")
        
        # Distribution selection
        distribution = st.selectbox(
            "Select Distribution", 
            ["Uniform", "Exponential", "Binomial"]
        )
        
        # Parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_trials = st.slider("Maximum Number of Trials", 100, 10000, 1000)
        
        with col2:
            if distribution == "Uniform":
                dist_min = st.slider("Distribution Minimum", -10.0, 0.0, -5.0, 0.1)
                dist_max = st.slider("Distribution Maximum", 0.0, 10.0, 5.0, 0.1)
            elif distribution == "Exponential":
                rate = st.slider("Rate (λ)", 0.1, 5.0, 1.0, 0.1)
            else:  # Binomial
                num_trials = st.slider("Number of Trials", 1, 100, 10)
                prob_success = st.slider("Probability of Success", 0.0, 1.0, 0.5, 0.01)
        
        # Generate cumulative means
        def generate_cumulative_means(distribution, max_trials):
            np.random.seed(42)
            cumulative_means = []
            
            for n in range(1, max_trials + 1):
                if distribution == "Uniform":
                    sample = np.random.uniform(dist_min, dist_max, n)
                    true_mean = (dist_min + dist_max) / 2
                elif distribution == "Exponential":
                    sample = np.random.exponential(1/rate, n)
                    true_mean = 1 / rate
                else:  # Binomial
                    sample = np.random.binomial(num_trials, prob_success, n)
                    true_mean = num_trials * prob_success
                
                cumulative_means.append({
                    'Trials': n,
                    'Sample Mean': np.mean(sample),
                    'True Mean': true_mean
                })
            
            return pd.DataFrame(cumulative_means)
        
        # Generate data
        convergence_data = generate_cumulative_means(distribution, max_trials)
        
        # Visualization
        fig_conv = go.Figure()
        
        # Sample mean
        fig_conv.add_trace(go.Scatter(
            x=convergence_data['Trials'],
            y=convergence_data['Sample Mean'],
            mode='lines',
            name='Sample Mean',
            line=dict(color='blue')
        ))
        
        # True mean
        fig_conv.add_trace(go.Scatter(
            x=convergence_data['Trials'],
            y=convergence_data['True Mean'],
            mode='lines',
            name='True Mean',
            line=dict(color='red', dash='dot')
        ))
        
        fig_conv.update_layout(
            title=f"Mean Convergence for {distribution} Distribution",
            xaxis_title="Number of Trials",
            yaxis_title="Mean Value"
        )
        
        st.plotly_chart(fig_conv, use_container_width=True)
        
        # Final statistics
        final_sample_mean = convergence_data['Sample Mean'].iloc[-1]
        true_mean = convergence_data['True Mean'].iloc[-1]
        
        st.markdown(f"""
        ### Convergence Analysis
        
        **Final Sample Mean**: {final_sample_mean:.4f}
        **True Population Mean**: {true_mean:.4f}
        **Absolute Error**: {abs(final_sample_mean - true_mean):.4f}
        
        #### Observations
        - Sample mean approaches true mean as trials increase
        - Convergence rate depends on distribution characteristics
        """)
    
    with tab2:
        st.subheader("Convergence Across Distributions")
        
        st.markdown("""
        ### Comparing Convergence Behavior
        
        Explore how different distributions converge:
        - Symmetric vs. Skewed distributions
        - Discrete vs. Continuous distributions
        """)
        
        # Distribution comparison
        distributions_to_compare = st.multiselect(
            "Select Distributions", 
            ["Uniform", "Exponential", "Normal", "Poisson"],
            default=["Uniform", "Exponential"]
        )
        
        # Parameters
        col1, col2 = st.columns(2)
        
        with col1:
            compare_max_trials = st.slider("Maximum Trials (Comparison)", 100, 10000, 1000)
        
        with col2:
            num_simulations = st.slider("Number of Simulations", 5, 50, 10)
        
        # Comparative convergence
        def compare_distributions_convergence(distributions, max_trials, num_sims):
            convergence_results = []
            
            for dist in distributions:
                for sim in range(num_sims):
                    np.random.seed(sim)
                    
                    if dist == "Uniform":
                        dist_min, dist_max = -5.0, 5.0
                        true_mean = (dist_min + dist_max) / 2
                        sample = np.random.uniform(dist_min, dist_max, max_trials)
                    
                    elif dist == "Exponential":
                        rate = 1.0
                        true_mean = 1 / rate
                        sample = np.random.exponential(1/rate, max_trials)
                    
                    elif dist == "Normal":
                        true_mean = 0.0
                        std_dev = 1.0
                        sample = np.random.normal(true_mean, std_dev, max_trials)
                    
                    else:  # Poisson
                        true_lambda = 5.0
                        true_mean = true_lambda
                        sample = np.random.poisson(true_lambda, max_trials)
                    
                    # Calculate cumulative means
                    cumulative_means = np.cumsum(sample) / np.arange(1, max_trials + 1)
                    
                    for n, mean in enumerate(cumulative_means, 1):
                        convergence_results.append({
                            'Distribution': dist,
                            'Trials': n,
                            'Sample Mean': mean,
                            'True Mean': true_mean,
                            'Simulation': sim
                        })
            
            return pd.DataFrame(convergence_results)
        
        # Generate comparative data
        comparison_data = compare_distributions_convergence(
            distributions_to_compare, 
            compare_max_trials, 
            num_simulations
        )
        
        # Visualization
        fig_compare = go.Figure()
        
        for dist in distributions_to_compare:
            dist_data = comparison_data[comparison_data['Distribution'] == dist]
            
            # Aggregate across simulations
            agg_data = dist_data.groupby('Trials').agg({
                'Sample Mean': 'mean',
                'True Mean': 'first'
            }).reset_index()
            
            # Mean line
            fig_compare.add_trace(go.Scatter(
                x=agg_data['Trials'],
                y=agg_data['Sample Mean'],
                mode='lines',
                name=f'{dist} Sample Mean',
                line=dict(width=2)
            ))
            
            # True mean line
            fig_compare.add_trace(go.Scatter(
                x=agg_data['Trials'],
                y=agg_data['True Mean'],
                mode='lines',
                name=f'{dist} True Mean',
                line=dict(dash='dot', width=2)
            ))
        
        fig_compare.update_layout(
            title="Mean Convergence Across Distributions",
            xaxis_title="Number of Trials",
            yaxis_title="Mean Value"
        )
        
        st.plotly_chart(fig_compare, use_container_width=True)
        
        # Convergence error analysis
        st.subheader("Convergence Error Analysis")
        
        error_data = []
        for dist in distributions_to_compare:
            dist_data = comparison_data[comparison_data['Distribution'] == dist]
            
            # Calculate error at different trial points
            trial_points = [100, 500, 1000, compare_max_trials]
            
            for trials in trial_points:
                subset = dist_data[dist_data['Trials'] == trials]
                mean_error = np.mean(np.abs(subset['Sample Mean'] - subset['True Mean']))
                
                error_data.append({
                    'Distribution': dist,
                    'Trials': trials,
                    'Mean Absolute Error': mean_error
                })
        
        error_df = pd.DataFrame(error_data)
        st.table(error_df)
    
    with tab3:
        st.subheader("Practical Applications")
        
        st.markdown("""
        ### Real-World Law of Large Numbers Applications
        
        Explore how LLN applies in various domains:
        - Gambling and Probability
        - Quality Control
        - Financial Risk Assessment
        """)
        
        # Application selection
        application = st.selectbox(
            "Select Application Domain", 
            ["Coin Flips", "Insurance Claims", "Stock Returns"]
        )
        
        # Parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_trials_app = st.slider("Maximum Number of Trials (Application)", 100, 10000, 1000)
        
        with col2:
            if application == "Coin Flips":
                prob_heads = st.slider("Probability of Heads", 0.0, 1.0, 0.5, 0.01)
            elif application == "Insurance Claims":
                avg_claim_amount = st.slider("Average Claim Amount ($)", 1000.0, 50000.0, 10000.0, 100.0)
                claim_std_dev = st.slider("Claim Amount Std Dev ($)", 500.0, 20000.0, 5000.0, 100.0)
            else:  # Stock Returns
                avg_return = st.slider("Average Annual Return (%)", -10.0, 20.0, 7.0, 0.1)
                return_std_dev = st.slider("Return Std Dev (%)", 1.0, 30.0, 15.0, 0.1)
        
        with col3:
            num_simulations_app = st.slider("Number of Simulations (Application)", 5, 50, 10)
        
        # Practical application simulation
        def simulate_practical_application(application, max_trials, num_sims):
            np.random.seed(42)
            results = []
            
            for sim in range(num_sims):
                if application == "Coin Flips":
                    # Simulate coin flips
                    flips = np.random.binomial(1, prob_heads, max_trials)
                    cumulative_means = np.cumsum(flips) / np.arange(1, max_trials + 1)
                    true_mean = prob_heads
                
                elif application == "Insurance Claims":
                    # Simulate insurance claims
                    claims = np.random.normal(avg_claim_amount, claim_std_dev, max_trials)
                    cumulative_means = np.cumsum(claims) / np.arange(1, max_trials + 1)
                    true_mean = avg_claim_amount
                
                else:  # Stock Returns
                    # Simulate stock returns
                    returns = np.random.normal(avg_return, return_std_dev, max_trials)
                    cumulative_means = np.cumsum(returns) / np.arange(1, max_trials + 1)
                    true_mean = avg_return
                
                for n, mean in enumerate(cumulative_means, 1):
                    results.append({
                        'Trials': n,
                        'Sample Mean': mean,
                        'True Mean': true_mean,
                        'Simulation': sim
                    })
            
            return pd.DataFrame(results)
        
        # Generate application data
        app_data = simulate_practical_application(
            application, 
            max_trials_app, 
            num_simulations_app
        )
        
        # Visualization
        fig_app = go.Figure()
        
        # Aggregate across simulations
        agg_data = app_data.groupby('Trials').agg({
            'Sample Mean': 'mean',
            'True Mean': 'first'
        }).reset_index()
        
        # Sample mean line
        fig_app.add_trace(go.Scatter(
            x=agg_data['Trials'],
            y=agg_data['Sample Mean'],
            mode='lines',
            name='Sample Mean',
            line=dict(color='blue')
        ))
        
        # True mean line
        fig_app.add_trace(go.Scatter(
            x=agg_data['Trials'],
            y=agg_data['True Mean'],
            mode='lines',
            name='True Mean',
            line=dict(color='red', dash='dot')
        ))
        
        fig_app.update_layout(
            title=f"Law of Large Numbers in {application}",
            xaxis_title="Number of Trials",
            yaxis_title="Mean Value"
        )
        
        st.plotly_chart(fig_app, use_container_width=True)
        
        # Final statistics
        final_sample_mean = agg_data['Sample Mean'].iloc[-1]
        true_mean = agg_data['True Mean'].iloc[-1]
        
        st.markdown(f"""
        ### Practical Application Analysis
        
        **Final Sample Mean**: {final_sample_mean:.4f}
        **True Mean**: {true_mean:.4f}
        **Absolute Error**: {abs(final_sample_mean - true_mean):.4f}
        
        #### Insights
        - Demonstrates convergence of sample mean to true mean
        - Highlights importance of large sample sizes in estimation
        """)

if __name__ == "__main__":
    run_law_of_large_numbers()
