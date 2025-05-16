import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import pandas as pd

def run_bayesian_estimation():
    st.title("🔮 Bayesian Estimation Explorer")
    
    st.markdown("""
    ### Understanding Bayesian Estimation
    
    Bayesian estimation combines prior knowledge with observed data 
    to estimate population parameters.
    
    Key Concepts:
    - Prior Distribution
    - Likelihood Function
    - Posterior Distribution
    """)
    
    # Tabs for different Bayesian estimation techniques
    tab1, tab2, tab3 = st.tabs([
        "Parameter Estimation", 
        "Conjugate Priors", 
        "Posterior Inference"
    ])
    
    with tab1:
        st.subheader("Bayesian Parameter Estimation")
        
        # Distribution selection
        distribution = st.selectbox(
            "Select Distribution", 
            ["Normal", "Binomial", "Poisson"]
        )
        
        # Parameters for Bayesian estimation
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sample_size = st.slider("Sample Size", 10, 1000, 100)
        
        with col2:
            if distribution == "Normal":
                true_mean = st.slider("True Mean", -10.0, 10.0, 0.0, 0.1)
                true_std = st.slider("True Standard Deviation", 0.1, 5.0, 1.0, 0.1)
                # Prior for mean
                prior_mean = st.slider("Prior Mean", -10.0, 10.0, 0.0, 0.1)
                prior_std = st.slider("Prior Std Dev", 0.1, 5.0, 1.0, 0.1)
            elif distribution == "Binomial":
                true_prob = st.slider("True Probability", 0.0, 1.0, 0.5, 0.01)
                # Prior for probability
                prior_alpha = st.slider("Prior Alpha", 0.1, 20.0, 1.0, 0.1)
                prior_beta = st.slider("Prior Beta", 0.1, 20.0, 1.0, 0.1)
            else:  # Poisson
                true_lambda = st.slider("True Lambda", 0.1, 20.0, 5.0, 0.1)
                # Prior for lambda
                prior_shape = st.slider("Prior Shape", 0.1, 20.0, 1.0, 0.1)
                prior_rate = st.slider("Prior Rate", 0.1, 20.0, 1.0, 0.1)
        
        with col3:
            num_simulations = st.slider("Number of Simulations", 10, 500, 100)
        
        # Bayesian estimation function
        def bayesian_estimation(distribution):
            np.random.seed(42)
            posterior_samples = []
            
            for _ in range(num_simulations):
                # Generate sample data
                if distribution == "Normal":
                    # Generate sample from true distribution
                    sample = np.random.normal(true_mean, true_std, sample_size)
                    
                    # Bayesian update for normal distribution with known variance
                    # Posterior mean is weighted average of prior and sample mean
                    sample_mean = np.mean(sample)
                    posterior_mean = (prior_mean / (prior_std**2) + sample_mean * sample_size / (true_std**2)) / \
                                     (1 / (prior_std**2) + sample_size / (true_std**2))
                    
                    posterior_samples.append(posterior_mean)
                
                elif distribution == "Binomial":
                    # Generate sample from true distribution
                    sample = np.random.binomial(1, true_prob, sample_size)
                    
                    # Bayesian update for binomial (Beta-Binomial conjugate prior)
                    successes = np.sum(sample)
                    posterior_alpha = prior_alpha + successes
                    posterior_beta = prior_beta + (sample_size - successes)
                    
                    # Sample from posterior Beta distribution
                    posterior_samples.append(np.random.beta(posterior_alpha, posterior_beta))
                
                else:  # Poisson
                    # Generate sample from true distribution
                    sample = np.random.poisson(true_lambda, sample_size)
                    
                    # Bayesian update for Poisson (Gamma-Poisson conjugate prior)
                    total_events = np.sum(sample)
                    posterior_shape = prior_shape + total_events
                    posterior_rate = prior_rate + sample_size
                    
                    # Sample from posterior Gamma distribution
                    posterior_samples.append(np.random.gamma(posterior_shape, 1/posterior_rate))
            
            return posterior_samples
        
        # Perform Bayesian estimation
        posterior_samples = bayesian_estimation(distribution)
        
        # Visualization
        fig = go.Figure(data=[
            go.Histogram(
                x=posterior_samples, 
                name='Posterior Samples',
                opacity=0.7
            )
        ])
        
        # Add true parameter line
        if distribution == "Normal":
            fig.add_shape(
                type="line",
                x0=true_mean,
                y0=0,
                x1=true_mean,
                y1=fig.layout.yaxis.range[1],
                line=dict(color="red", width=2, dash="dash")
            )
            title = f"Posterior Distribution for {distribution} Mean"
            x_title = "Estimated Mean"
        elif distribution == "Binomial":
            fig.add_shape(
                type="line",
                x0=true_prob,
                y0=0,
                x1=true_prob,
                y1=fig.layout.yaxis.range[1],
                line=dict(color="red", width=2, dash="dash")
            )
            title = f"Posterior Distribution for {distribution} Probability"
            x_title = "Estimated Probability"
        else:  # Poisson
            fig.add_shape(
                type="line",
                x0=true_lambda,
                y0=0,
                x1=true_lambda,
                y1=fig.layout.yaxis.range[1],
                line=dict(color="red", width=2, dash="dash")
            )
            title = f"Posterior Distribution for {distribution} Lambda"
            x_title = "Estimated Lambda"
        
        fig.update_layout(
            title=title,
            xaxis_title=x_title,
            yaxis_title="Frequency"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        posterior_mean = np.mean(posterior_samples)
        posterior_std = np.std(posterior_samples)
        
        # Credible interval
        credible_interval = np.percentile(posterior_samples, [2.5, 97.5])
        
        if distribution == "Normal":
            st.markdown(f"""
            ### Bayesian Estimation Results
            
            - **True Mean**: {true_mean}
            - **Posterior Mean**: {posterior_mean:.4f}
            - **Posterior Std Dev**: {posterior_std:.4f}
            - **95% Credible Interval**: [{credible_interval[0]:.4f}, {credible_interval[1]:.4f}]
            """)
        elif distribution == "Binomial":
            st.markdown(f"""
            ### Bayesian Estimation Results
            
            - **True Probability**: {true_prob}
            - **Posterior Mean**: {posterior_mean:.4f}
            - **Posterior Std Dev**: {posterior_std:.4f}
            - **95% Credible Interval**: [{credible_interval[0]:.4f}, {credible_interval[1]:.4f}]
            """)
        else:  # Poisson
            st.markdown(f"""
            ### Bayesian Estimation Results
            
            - **True Lambda**: {true_lambda}
            - **Posterior Mean**: {posterior_mean:.4f}
            - **Posterior Std Dev**: {posterior_std:.4f}
            - **95% Credible Interval**: [{credible_interval[0]:.4f}, {credible_interval[1]:.4f}]
            """)
    
    with tab2:
        st.subheader("Conjugate Priors")
        
        st.markdown("""
        ### Exploring Conjugate Prior Distributions
        
        Conjugate priors simplify Bayesian inference by ensuring the 
        posterior distribution belongs to the same family as the prior.
        
        Common Conjugate Prior Pairs:
        - Beta prior for Binomial likelihood
        - Gamma prior for Poisson likelihood
        - Normal prior for Normal likelihood with known variance
        """)
        
        # Conjugate prior selection
        prior_type = st.selectbox(
            "Select Conjugate Prior", 
            ["Beta-Binomial", "Gamma-Poisson", "Normal-Normal"]
        )
        
        if prior_type == "Beta-Binomial":
            # Beta-Binomial conjugate prior
            col1, col2 = st.columns(2)
            
            with col1:
                alpha = st.slider("Prior Alpha", 0.1, 20.0, 1.0, 0.1)
            
            with col2:
                beta = st.slider("Prior Beta", 0.1, 20.0, 1.0, 0.1)
            
            # Visualization of Beta prior
            x = np.linspace(0, 1, 200)
            prior_pdf = stats.beta.pdf(x, alpha, beta)
            
            fig_prior = go.Figure(data=[
                go.Scatter(
                    x=x, 
                    y=prior_pdf, 
                    mode='lines',
                    name='Prior Distribution'
                )
            ])
            
            fig_prior.update_layout(
                title=f"Beta Prior Distribution (α={alpha}, β={beta})",
                xaxis_title="Probability",
                yaxis_title="Density"
            )
            
            st.plotly_chart(fig_prior, use_container_width=True)
            
            st.markdown(f"""
            ### Beta Prior Characteristics
            
            - **Mean**: {alpha / (alpha + beta):.4f}
            - **Variance**: {(alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1)):.4f}
            
            #### Interpretation
            - α and β control the shape of the prior distribution
            - When α = β = 1, it's a uniform prior
            - Higher values concentrate probability around the mean
            """)
        
        elif prior_type == "Gamma-Poisson":
            # Gamma-Poisson conjugate prior
            col1, col2 = st.columns(2)
            
            with col1:
                shape = st.slider("Prior Shape", 0.1, 20.0, 1.0, 0.1)
            
            with col2:
                rate = st.slider("Prior Rate", 0.1, 20.0, 1.0, 0.1)
            
            # Visualization of Gamma prior
            x = np.linspace(0, 10, 200)
            prior_pdf = stats.gamma.pdf(x, shape, scale=1/rate)
            
            fig_prior = go.Figure(data=[
                go.Scatter(
                    x=x, 
                    y=prior_pdf, 
                    mode='lines',
                    name='Prior Distribution'
                )
            ])
            
            fig_prior.update_layout(
                title=f"Gamma Prior Distribution (k={shape}, θ={1/rate})",
                xaxis_title="Lambda",
                yaxis_title="Density"
            )
            
            st.plotly_chart(fig_prior, use_container_width=True)
            
            st.markdown(f"""
            ### Gamma Prior Characteristics
            
            - **Mean**: {shape / rate:.4f}
            - **Variance**: {shape / (rate**2):.4f}
            
            #### Interpretation
            - Shape (k) and rate (θ) control the distribution
            - Useful for modeling rate parameters in Poisson processes
            """)
        
        else:  # Normal-Normal
            # Normal-Normal conjugate prior
            col1, col2 = st.columns(2)
            
            with col1:
                prior_mean = st.slider("Prior Mean", -10.0, 10.0, 0.0, 0.1)
            
            with col2:
                prior_std = st.slider("Prior Std Dev", 0.1, 5.0, 1.0, 0.1)
            
            # Visualization of Normal prior
            x = np.linspace(prior_mean - 4*prior_std, prior_mean + 4*prior_std, 200)
            prior_pdf = stats.norm.pdf(x, prior_mean, prior_std)
            
            fig_prior = go.Figure(data=[
                go.Scatter(
                    x=x, 
                    y=prior_pdf, 
                    mode='lines',
                    name='Prior Distribution'
                )
            ])
            
            fig_prior.update_layout(
                title=f"Normal Prior Distribution (μ={prior_mean}, σ={prior_std})",
                xaxis_title="Mean",
                yaxis_title="Density"
            )
            
            st.plotly_chart(fig_prior, use_container_width=True)
            
            st.markdown(f"""
            ### Normal Prior Characteristics
            
            - **Mean**: {prior_mean}
            - **Variance**: {prior_std**2:.4f}
            
            #### Interpretation
            - Centered at prior_mean
            - Spread controlled by prior_std
            - Represents prior belief about the parameter
            """)
    
    with tab3:
        st.subheader("Posterior Inference")
        
        st.markdown("""
        ### Bayesian Inference Techniques
        
        Explore different methods of drawing insights from the posterior distribution:
        - Point Estimation
        - Interval Estimation
        - Hypothesis Testing
        """)
        
        # Inference method selection
        inference_method = st.selectbox(
            "Select Inference Method", 
            ["Point Estimates", "Credible Intervals", "Probability Statements"]
        )
        
        # Simulation parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            inf_distribution = st.selectbox(
                "Distribution", 
                ["Normal", "Binomial", "Poisson"]
            )
        
        with col2:
            inf_sample_size = st.slider("Sample Size", 10, 1000, 100)
        
        with col3:
            inf_num_simulations = st.slider("Number of Simulations", 10, 500, 100)
        
        # True parameters
        if inf_distribution == "Normal":
            true_mean = st.slider("True Mean", -10.0, 10.0, 0.0, 0.1)
            true_std = st.slider("True Std Dev", 0.1, 5.0, 1.0, 0.1)
            prior_mean = st.slider("Prior Mean", -10.0, 10.0, 0.0, 0.1)
            prior_std = st.slider("Prior Std Dev", 0.1, 5.0, 1.0, 0.1)
        elif inf_distribution == "Binomial":
            true_prob = st.slider("True Probability", 0.0, 1.0, 0.5, 0.01)
            prior_alpha = st.slider("Prior Alpha", 0.1, 20.0, 1.0, 0.1)
            prior_beta = st.slider("Prior Beta", 0.1, 20.0, 1.0, 0.1)
        else:  # Poisson
            true_lambda = st.slider("True Lambda", 0.1, 20.0, 5.0, 0.1)
            prior_shape = st.slider("Prior Shape", 0.1, 20.0, 1.0, 0.1)
            prior_rate = st.slider("Prior Rate", 0.1, 20.0, 1.0, 0.1)
        
        # Posterior inference function
        def posterior_inference(distribution):
            np.random.seed(42)
            posterior_samples = []
            
            for _ in range(inf_num_simulations):
                # Generate sample data
                if distribution == "Normal":
                    sample = np.random.normal(true_mean, true_std, inf_sample_size)
                    
                    # Bayesian update
                    sample_mean = np.mean(sample)
                    posterior_mean = (prior_mean / (prior_std**2) + sample_mean * inf_sample_size / (true_std**2)) / \
                                     (1 / (prior_std**2) + inf_sample_size / (true_std**2))
                    
                    posterior_samples.append(posterior_mean)
                
                elif distribution == "Binomial":
                    sample = np.random.binomial(1, true_prob, inf_sample_size)
                    
                    # Bayesian update
                    successes = np.sum(sample)
                    posterior_alpha = prior_alpha + successes
                    posterior_beta = prior_beta + (inf_sample_size - successes)
                    
                    posterior_samples.append(np.random.beta(posterior_alpha, posterior_beta))
                
                else:  # Poisson
                    sample = np.random.poisson(true_lambda, inf_sample_size)
                    
                    # Bayesian update
                    total_events = np.sum(sample)
                    posterior_shape = prior_shape + total_events
                    posterior_rate = prior_rate + inf_sample_size
                    
                    posterior_samples.append(np.random.gamma(posterior_shape, 1/posterior_rate))
            
            return posterior_samples
        
        # Perform posterior inference
        posterior_samples = posterior_inference(inf_distribution)
        
        if inference_method == "Point Estimates":
            # Different point estimation methods
            point_methods = {
                "Posterior Mean": np.mean(posterior_samples),
                "Posterior Median": np.median(posterior_samples),
                "Maximum A Posteriori (MAP)": stats.mode(posterior_samples)[0][0]
            }
            
            st.subheader("Point Estimation Comparison")
            
            for method, estimate in point_methods.items():
                st.metric(method, f"{estimate:.4f}")
            
            # Visualization of point estimates
            fig_point = go.Figure()
            
            for method, estimate in point_methods.items():
                fig_point.add_trace(go.Scatter(
                    x=[method],
                    y=[estimate],
                    mode='markers',
                    name=method
                ))
            
            # Add true parameter line
            if inf_distribution == "Normal":
                fig_point.add_shape(
                    type="line",
                    x0=-1,
                    y0=true_mean,
                    x1=len(point_methods),
                    y1=true_mean,
                    line=dict(color="red", width=2, dash="dash")
                )
                y_title = "Estimated Mean"
            elif inf_distribution == "Binomial":
                fig_point.add_shape(
                    type="line",
                    x0=-1,
                    y0=true_prob,
                    x1=len(point_methods),
                    y1=true_prob,
                    line=dict(color="red", width=2, dash="dash")
                )
                y_title = "Estimated Probability"
            else:  # Poisson
                fig_point.add_shape(
                    type="line",
                    x0=-1,
                    y0=true_lambda,
                    x1=len(point_methods),
                    y1=true_lambda,
                    line=dict(color="red", width=2, dash="dash")
                )
                y_title = "Estimated Lambda"
            
            fig_point.update_layout(
                title="Comparison of Point Estimation Methods",
                yaxis_title=y_title
            )
            
            st.plotly_chart(fig_point, use_container_width=True)
        
        elif inference_method == "Credible Intervals":
            # Calculate credible intervals
            credible_levels = [0.50, 0.80, 0.95]
            
            st.subheader("Credible Intervals")
            
            interval_results = []
            
            for level in credible_levels:
                lower, upper = np.percentile(posterior_samples, [(1-level)/2 * 100, (1+level)/2 * 100])
                interval_results.append({
                    'Level': f"{level*100:.0f}%",
                    'Lower Bound': lower,
                    'Upper Bound': upper
                })
            
            # Display as table
            st.table(pd.DataFrame(interval_results))
            
            # Visualization of credible intervals
            fig_ci = go.Figure()
            
            for result in interval_results:
                fig_ci.add_trace(go.Scatter(
                    x=[result['Lower Bound'], result['Upper Bound']],
                    y=[result['Level'], result['Level']],
                    mode='lines',
                    name=result['Level']
                ))
            
            # Add true parameter line
            if inf_distribution == "Normal":
                fig_ci.add_shape(
                    type="line",
                    x0=true_mean,
                    y0=0,
                    x1=true_mean,
                    y1=1,
                    line=dict(color="red", width=2, dash="dash")
                )
                x_title = "Estimated Mean"
            elif inf_distribution == "Binomial":
                fig_ci.add_shape(
                    type="line",
                    x0=true_prob,
                    y0=0,
                    x1=true_prob,
                    y1=1,
                    line=dict(color="red", width=2, dash="dash")
                )
                x_title = "Estimated Probability"
            else:  # Poisson
                fig_ci.add_shape(
                    type="line",
                    x0=true_lambda,
                    y0=0,
                    x1=true_lambda,
                    y1=1,
                    line=dict(color="red", width=2, dash="dash")
                )
                x_title = "Estimated Lambda"
            
            fig_ci.update_layout(
                title="Credible Intervals",
                xaxis_title=x_title,
                yaxis_title="Credible Level"
            )
            
            st.plotly_chart(fig_ci, use_container_width=True)
        
        else:  # Probability Statements
            st.subheader("Probability Statements")
            
            # Probability threshold selection
            if inf_distribution == "Normal":
                threshold = st.slider("Threshold for Mean", 
                    float(np.min(posterior_samples)), 
                    float(np.max(posterior_samples)), 
                    float(np.mean(posterior_samples)), 
                    float((np.max(posterior_samples) - np.min(posterior_samples)) / 100)
                )
                prob_above = np.mean(np.array(posterior_samples) > threshold)
                prob_below = np.mean(np.array(posterior_samples) < threshold)
                
                st.markdown(f"""
                ### Probability Statements
                
                - **P(Mean > {threshold:.4f})**: {prob_above:.4f} ({prob_above*100:.2f}%)
                - **P(Mean < {threshold:.4f})**: {prob_below:.4f} ({prob_below*100:.2f}%)
                """)
            
            elif inf_distribution == "Binomial":
                threshold = st.slider("Threshold for Probability", 0.0, 1.0, 0.5, 0.01)
                prob_above = np.mean(np.array(posterior_samples) > threshold)
                prob_below = np.mean(np.array(posterior_samples) < threshold)
                
                st.markdown(f"""
                ### Probability Statements
                
                - **P(Probability > {threshold:.4f})**: {prob_above:.4f} ({prob_above*100:.2f}%)
                - **P(Probability < {threshold:.4f})**: {prob_below:.4f} ({prob_below*100:.2f}%)
                """)
            
            else:  # Poisson
                threshold = st.slider("Threshold for Lambda", 
                    float(np.min(posterior_samples)), 
                    float(np.max(posterior_samples)), 
                    float(np.mean(posterior_samples)), 
                    float((np.max(posterior_samples) - np.min(posterior_samples)) / 100)
                )
                prob_above = np.mean(np.array(posterior_samples) > threshold)
                prob_below = np.mean(np.array(posterior_samples) < threshold)
                
                st.markdown(f"""
                ### Probability Statements
                
                - **P(Lambda > {threshold:.4f})**: {prob_above:.4f} ({prob_above*100:.2f}%)
                - **P(Lambda < {threshold:.4f})**: {prob_below:.4f} ({prob_below*100:.2f}%)
                """)
            
            # Visualization of probability distribution
            fig_prob = go.Figure(data=[
                go.Histogram(
                    x=posterior_samples, 
                    name='Posterior Distribution',
                    opacity=0.7,
                    cumulative_enabled=True
                )
            ])
            
            # Add threshold line
            fig_prob.add_shape(
                type="line",
                x0=threshold,
                y0=0,
                x1=threshold,
                y1=fig_prob.layout.yaxis.range[1],
                line=dict(color="red", width=2, dash="dash")
            )
            
            fig_prob.update_layout(
                title="Cumulative Posterior Distribution",
                xaxis_title="Parameter Value",
                yaxis_title="Cumulative Probability"
            )
            
            st.plotly_chart(fig_prob, use_container_width=True)

if __name__ == "__main__":
    run_bayesian_estimation()
