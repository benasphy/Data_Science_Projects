import streamlit as st
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from scipy import stats
import pandas as pd

def run_bayesian_inference():
    st.title("🔬 Bayesian Inference Visualization")
    
    st.markdown("""
    ### Understanding Bayesian Inference
    
    Bayesian inference is a method of statistical inference 
    that updates the probability of a hypothesis as more evidence becomes available.
    
    Key Concepts:
    - Prior Distribution
    - Likelihood Function
    - Posterior Distribution
    - Bayesian Update
    """)
    
    # Tabs for different Bayesian inference analyses
    tab1, tab2, tab3 = st.tabs([
        "Basic Bayesian Update", 
        "Parameter Estimation", 
        "Real-World Applications"
    ])
    
    with tab1:
        st.subheader("Bayesian Updating Process")
        
        # Distribution parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            prior_mean = st.slider("Prior Mean", -10.0, 10.0, 0.0, 0.1)
        
        with col2:
            prior_std = st.slider("Prior Standard Deviation", 0.1, 5.0, 1.0, 0.1)
        
        with col3:
            num_observations = st.slider("Number of Observations", 1, 100, 10)
        
        # Bayesian update simulation
        def simulate_bayesian_update(prior_mean, prior_std, num_observations):
            np.random.seed(42)
            
            # True underlying parameter (unknown in real scenario)
            true_parameter = np.random.normal(prior_mean, prior_std)
            
            # Generate observations
            observations = np.random.normal(true_parameter, 1.0, num_observations)
            
            # Prior distribution
            x = np.linspace(prior_mean - 4*prior_std, prior_mean + 4*prior_std, 300)
            prior_pdf = stats.norm.pdf(x, prior_mean, prior_std)
            
            # Likelihood and posterior calculations
            def bayesian_update(prior_mean, prior_std, observations):
                # Compute likelihood
                likelihood_mean = np.mean(observations)
                likelihood_std = 1.0 / np.sqrt(len(observations))
                
                # Posterior calculation (conjugate prior for normal distribution)
                posterior_var = 1 / (1/prior_std**2 + len(observations)/(1.0**2))
                posterior_mean = posterior_var * (prior_mean/prior_std**2 + 
                                                  len(observations)*likelihood_mean/(1.0**2))
                posterior_std = np.sqrt(posterior_var)
                
                return posterior_mean, posterior_std
            
            # Compute posterior
            posterior_mean, posterior_std = bayesian_update(prior_mean, prior_std, observations)
            
            # Posterior distribution
            posterior_pdf = stats.norm.pdf(x, posterior_mean, posterior_std)
            
            return x, prior_pdf, posterior_pdf, observations, true_parameter, posterior_mean
        
        # Generate data
        x, prior_pdf, posterior_pdf, observations, true_parameter, posterior_mean = simulate_bayesian_update(
            prior_mean, prior_std, num_observations
        )
        
        # Visualization
        fig_bayes = go.Figure()
        
        # Prior distribution
        fig_bayes.add_trace(go.Scatter(
            x=x,
            y=prior_pdf,
            mode='lines',
            name='Prior Distribution',
            line=dict(color='blue', width=3)
        ))
        
        # Posterior distribution
        fig_bayes.add_trace(go.Scatter(
            x=x,
            y=posterior_pdf,
            mode='lines',
            name='Posterior Distribution',
            line=dict(color='red', dash='dot')
        ))
        
        fig_bayes.update_layout(
            title="Bayesian Updating: Prior vs Posterior",
            xaxis=dict(title="Parameter Value"),
            yaxis=dict(title="Probability Density")
        )
        
        st.plotly_chart(fig_bayes, use_container_width=True)
        
        # Observations and results
        st.subheader("Bayesian Update Results")
        
        results_df = pd.DataFrame({
            'Metric': ['True Parameter', 'Prior Mean', 'Posterior Mean', 'Observations'],
            'Value': [
                true_parameter, 
                prior_mean, 
                posterior_mean, 
                f"[{', '.join(map(str, observations[:5]))}{'...' if len(observations) > 5 else ''}]"
            ]
        })
        
        st.table(results_df)
        
        st.markdown("""
        ### Bayesian Update Interpretation
        
        - **Prior**: Initial belief about parameter
        - **Likelihood**: Information from observations
        - **Posterior**: Updated belief after observing data
        """)
    
    with tab2:
        st.subheader("Bayesian Parameter Estimation")
        
        st.markdown("""
        ### Exploring Parameter Estimation
        
        Investigate how:
        - Sample size affects estimation
        - Prior knowledge impacts inference
        - Uncertainty reduces with more data
        """)
        
        # Parameter estimation exploration
        col1, col2 = st.columns(2)
        
        with col1:
            sample_sizes = st.multiselect(
                "Select Sample Sizes", 
                [10, 50, 100, 500, 1000],
                default=[10, 50, 100]
            )
        
        with col2:
            prior_variations = st.multiselect(
                "Select Prior Variations", 
                ["Narrow Prior", "Broad Prior", "Informative Prior"],
                default=["Narrow Prior", "Broad Prior"]
            )
        
        # Comparative parameter estimation
        def compare_parameter_estimation(sample_sizes, prior_variations):
            np.random.seed(42)
            
            # True underlying parameter
            true_parameter = 5.0
            
            results = []
            
            for prior_type in prior_variations:
                if prior_type == "Narrow Prior":
                    prior_mean, prior_std = true_parameter, 0.5
                elif prior_type == "Broad Prior":
                    prior_mean, prior_std = true_parameter, 5.0
                else:  # Informative Prior
                    prior_mean, prior_std = true_parameter, 1.0
                
                for sample_size in sample_sizes:
                    # Generate observations
                    observations = np.random.normal(true_parameter, 1.0, sample_size)
                    
                    # Bayesian update
                    posterior_var = 1 / (1/prior_std**2 + sample_size/(1.0**2))
                    posterior_mean = posterior_var * (prior_mean/prior_std**2 + 
                                                      sample_size*np.mean(observations)/(1.0**2))
                    posterior_std = np.sqrt(posterior_var)
                    
                    results.append({
                        'Prior Type': prior_type,
                        'Sample Size': sample_size,
                        'Posterior Mean': posterior_mean,
                        'Posterior Std': posterior_std
                    })
            
            return pd.DataFrame(results)
        
        # Generate comparative data
        comparison_results = compare_parameter_estimation(sample_sizes, prior_variations)
        
        # Visualization
        fig_estimation = go.Figure()
        
        for prior_type in prior_variations:
            prior_data = comparison_results[comparison_results['Prior Type'] == prior_type]
            
            fig_estimation.add_trace(go.Scatter(
                x=prior_data['Sample Size'],
                y=prior_data['Posterior Mean'],
                mode='lines+markers',
                name=prior_type,
                error_y=dict(
                    type='data',
                    array=prior_data['Posterior Std'],
                    visible=True
                )
            ))
        
        fig_estimation.update_layout(
            title="Bayesian Parameter Estimation",
            xaxis=dict(title="Sample Size"),
            yaxis=dict(title="Posterior Mean")
        )
        
        st.plotly_chart(fig_estimation, use_container_width=True)
        
        # Detailed analysis
        st.subheader("Parameter Estimation Analysis")
        st.table(comparison_results)
        
        st.markdown("""
        ### Insights
        
        - **Narrow Prior**: Quick convergence
        - **Broad Prior**: More exploration
        - **Sample Size**: Reduces uncertainty
        """)
    
    with tab3:
        st.subheader("Real-World Bayesian Inference")
        
        st.markdown("""
        ### Practical Applications of Bayesian Inference
        
        Explore Bayesian methods in:
        - Medical Diagnosis
        - A/B Testing
        - Machine Learning
        """)
        
        # Application selection
        application = st.selectbox(
            "Select Application Domain", 
            ["Medical Test Accuracy", "Website Conversion Rate", "Machine Learning Model Performance"]
        )
        
        # Parameters
        col1, col2 = st.columns(2)
        
        with col1:
            app_prior_mean = st.slider("Prior Mean (Application)", -10.0, 10.0, 0.0, 0.1)
        
        with col2:
            app_prior_std = st.slider("Prior Standard Deviation (Application)", 0.1, 5.0, 1.0, 0.1)
        
        # Real-world data simulation
        def simulate_real_world_bayesian(application, prior_mean, prior_std):
            np.random.seed(42)
            
            if application == "Medical Test Accuracy":
                # Simulate medical test sensitivity
                true_sensitivity = np.random.normal(0.9, 0.05)
                observations = np.random.normal(true_sensitivity, 0.05, 100)
            
            elif application == "Website Conversion Rate":
                # Simulate conversion rate
                true_conversion = np.random.normal(0.05, 0.01)
                observations = np.random.normal(true_conversion, 0.02, 200)
            
            else:  # Machine Learning Model Performance
                # Simulate model accuracy
                true_accuracy = np.random.normal(0.75, 0.1)
                observations = np.random.normal(true_accuracy, 0.05, 150)
            
            # Bayesian update
            posterior_var = 1 / (1/prior_std**2 + len(observations)/(0.05**2))
            posterior_mean = posterior_var * (prior_mean/prior_std**2 + 
                                              len(observations)*np.mean(observations)/(0.05**2))
            posterior_std = np.sqrt(posterior_var)
            
            return observations, true_sensitivity, posterior_mean, posterior_std
        
        # Generate application-specific Bayesian inference
        observations, true_parameter, posterior_mean, posterior_std = simulate_real_world_bayesian(
            application, app_prior_mean, app_prior_std
        )
        
        # Visualization
        x = np.linspace(0, 1, 300)
        prior_pdf = stats.norm.pdf(x, app_prior_mean, app_prior_std)
        posterior_pdf = stats.norm.pdf(x, posterior_mean, posterior_std)
        
        fig_app_bayes = go.Figure()
        
        # Prior distribution
        fig_app_bayes.add_trace(go.Scatter(
            x=x,
            y=prior_pdf,
            mode='lines',
            name='Prior Distribution',
            line=dict(color='blue', width=3)
        ))
        
        # Posterior distribution
        fig_app_bayes.add_trace(go.Scatter(
            x=x,
            y=posterior_pdf,
            mode='lines',
            name='Posterior Distribution',
            line=dict(color='red', dash='dot')
        ))
        
        fig_app_bayes.update_layout(
            title=f"Bayesian Inference: {application}",
            xaxis=dict(title="Parameter Value"),
            yaxis=dict(title="Probability Density")
        )
        
        st.plotly_chart(fig_app_bayes, use_container_width=True)
        
        # Results summary
        st.subheader(f"{application} Bayesian Inference")
        
        results_df = pd.DataFrame({
            'Metric': ['True Parameter', 'Prior Mean', 'Posterior Mean', 'Observations'],
            'Value': [
                true_parameter, 
                app_prior_mean, 
                posterior_mean, 
                f"[{', '.join(map(str, observations[:5]))}{'...' if len(observations) > 5 else ''}]"
            ]
        })
        
        st.table(results_df)
        
        st.markdown(f"""
        ### {application} Insights
        
        **Bayesian Inference Demonstrates**:
        - Updating beliefs with new evidence
        - Quantifying uncertainty
        - Probabilistic reasoning
        """)

if __name__ == "__main__":
    run_bayesian_inference()
