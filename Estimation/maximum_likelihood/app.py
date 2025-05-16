import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import pandas as pd

def run_maximum_likelihood_estimation():
    st.title("📊 Maximum Likelihood Estimation Explorer")
    
    st.markdown("""
    ### Understanding Maximum Likelihood Estimation (MLE)
    
    Maximum Likelihood Estimation is a method for estimating the parameters 
    of a statistical model by maximizing the likelihood function.
    
    Key Concepts:
    - Likelihood Function
    - Parameter Estimation
    - Model Selection
    """)
    
    # Tabs for different MLE techniques
    tab1, tab2, tab3 = st.tabs([
        "Parameter Estimation", 
        "Likelihood Surfaces", 
        "Model Comparison"
    ])
    
    with tab1:
        st.subheader("MLE Parameter Estimation")
        
        # Distribution selection
        distribution = st.selectbox(
            "Select Distribution", 
            ["Normal", "Exponential", "Poisson"]
        )
        
        # Parameters for MLE
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sample_size = st.slider("Sample Size", 10, 1000, 100)
        
        with col2:
            if distribution == "Normal":
                true_mean = st.slider("True Mean", -10.0, 10.0, 0.0, 0.1)
                true_std = st.slider("True Standard Deviation", 0.1, 5.0, 1.0, 0.1)
            elif distribution == "Exponential":
                true_rate = st.slider("True Rate (λ)", 0.1, 5.0, 1.0, 0.1)
            else:  # Poisson
                true_lambda = st.slider("True Lambda (λ)", 0.1, 20.0, 5.0, 0.1)
        
        with col3:
            num_simulations = st.slider("Number of Simulations", 10, 500, 100)
        
        # MLE estimation function
        def mle_estimation(distribution):
            np.random.seed(42)
            results = []
            
            for _ in range(num_simulations):
                # Generate sample data
                if distribution == "Normal":
                    sample = np.random.normal(true_mean, true_std, sample_size)
                    
                    # MLE for normal is sample mean and sample std
                    mle_mean = np.mean(sample)
                    mle_std = np.std(sample, ddof=1)
                    
                    results.append({
                        'True Mean': true_mean, 
                        'Estimated Mean': mle_mean,
                        'True Std': true_std, 
                        'Estimated Std': mle_std
                    })
                
                elif distribution == "Exponential":
                    sample = np.random.exponential(1/true_rate, sample_size)
                    
                    # MLE for exponential is 1/sample_mean
                    mle_rate = 1 / np.mean(sample)
                    
                    results.append({
                        'True Rate': true_rate, 
                        'Estimated Rate': mle_rate
                    })
                
                else:  # Poisson
                    sample = np.random.poisson(true_lambda, sample_size)
                    
                    # MLE for Poisson is sample mean
                    mle_lambda = np.mean(sample)
                    
                    results.append({
                        'True Lambda': true_lambda, 
                        'Estimated Lambda': mle_lambda
                    })
            
            return pd.DataFrame(results)
        
        # Perform MLE estimation
        mle_results = mle_estimation(distribution)
        
        # Visualization
        fig = go.Figure()
        
        if distribution == "Normal":
            # Histogram of estimated means
            fig = go.Figure(data=[
                go.Histogram(
                    x=mle_results['Estimated Mean'], 
                    name='Estimated Mean',
                    opacity=0.7
                )
            ])
            
            # Compute y_max for vertical lines
            counts, _ = np.histogram(mle_results['Estimated Mean'], bins='auto')
            y_max = 1.1 * counts.max()
            # Add true mean line
            fig.add_shape(
                type="line",
                x0=true_mean,
                y0=0,
                x1=true_mean,
                y1=y_max,
                line=dict(color="red", width=2, dash="dash")
            )
            
            fig.update_layout(
                title=f"MLE Estimates for {distribution} Distribution Mean",
                xaxis_title="Estimated Mean",
                yaxis_title="Frequency"
            )
            
            # Display summary statistics
            st.markdown(f"""
            ### MLE Estimation Results
            
            - **True Mean**: {true_mean}
            - **Estimated Mean (Average)**: {mle_results['Estimated Mean'].mean():.4f}
            - **Mean Estimation Error**: {abs(mle_results['Estimated Mean'].mean() - true_mean):.4f}
            
            - **True Std Dev**: {true_std}
            - **Estimated Std Dev (Average)**: {mle_results['Estimated Std'].mean():.4f}
            - **Std Dev Estimation Error**: {abs(mle_results['Estimated Std'].mean() - true_std):.4f}
            """)
        
        elif distribution == "Exponential":
            # Histogram of estimated rates
            fig = go.Figure(data=[
                go.Histogram(
                    x=mle_results['Estimated Rate'], 
                    name='Estimated Rate',
                    opacity=0.7
                )
            ])
            
            # Add true rate line
            fig.add_shape(
                type="line",
                x0=true_rate,
                y0=0,
                x1=true_rate,
                y1=fig.layout.yaxis.range[1],
                line=dict(color="red", width=2, dash="dash")
            )
            
            fig.update_layout(
                title=f"MLE Estimates for {distribution} Distribution Rate",
                xaxis_title="Estimated Rate",
                yaxis_title="Frequency"
            )
            
            # Display summary statistics
            st.markdown(f"""
            ### MLE Estimation Results
            
            - **True Rate (λ)**: {true_rate}
            - **Estimated Rate (Average)**: {mle_results['Estimated Rate'].mean():.4f}
            - **Rate Estimation Error**: {abs(mle_results['Estimated Rate'].mean() - true_rate):.4f}
            """)
        
        else:  # Poisson
            # Histogram of estimated lambdas
            fig = go.Figure(data=[
                go.Histogram(
                    x=mle_results['Estimated Lambda'], 
                    name='Estimated Lambda',
                    opacity=0.7
                )
            ])
            
            # Add true lambda line
            fig.add_shape(
                type="line",
                x0=true_lambda,
                y0=0,
                x1=true_lambda,
                y1=fig.layout.yaxis.range[1],
                line=dict(color="red", width=2, dash="dash")
            )
            
            fig.update_layout(
                title=f"MLE Estimates for {distribution} Distribution Lambda",
                xaxis_title="Estimated Lambda",
                yaxis_title="Frequency"
            )
            
            # Display summary statistics
            st.markdown(f"""
            ### MLE Estimation Results
            
            - **True Lambda (λ)**: {true_lambda}
            - **Estimated Lambda (Average)**: {mle_results['Estimated Lambda'].mean():.4f}
            - **Lambda Estimation Error**: {abs(mle_results['Estimated Lambda'].mean() - true_lambda):.4f}
            """)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Likelihood Surfaces")
        
        st.markdown("""
        ### Exploring Likelihood Functions
        
        Visualize how the likelihood changes with different parameter values.
        """)
        
        # Likelihood surface parameters
        col1, col2 = st.columns(2)
        
        with col1:
            surface_distribution = st.selectbox(
                "Distribution for Likelihood Surface", 
                ["Normal", "Exponential"]
            )
        
        with col2:
            surface_sample_size = st.slider("Sample Size for Surface", 10, 500, 100)
        
        # Generate sample data
        if surface_distribution == "Normal":
            # Parameters for surface
            col1, col2 = st.columns(2)
            
            with col1:
                true_mean_surface = st.slider("True Mean", -10.0, 10.0, 0.0, 0.1, key="likelihood_surface_true_mean")
            
            with col2:
                true_std_surface = st.slider("True Std Dev", 0.1, 5.0, 1.0, 0.1, key="likelihood_surface_true_std")
            
            # Generate sample
            np.random.seed(42)
            sample_surface = np.random.normal(true_mean_surface, true_std_surface, surface_sample_size)
            
            # Likelihood function for normal distribution
            def normal_likelihood(mu, sigma, data):
                return np.prod(stats.norm.pdf(data, mu, sigma))
            
            # Create grid of parameter values
            mean_range = np.linspace(true_mean_surface - 3, true_mean_surface + 3, 100)
            std_range = np.linspace(true_std_surface - 2, true_std_surface + 2, 100)
            
            # Compute likelihood surface
            likelihood_surface = np.zeros((len(mean_range), len(std_range)))
            
            for i, mu in enumerate(mean_range):
                for j, sigma in enumerate(std_range):
                    likelihood_surface[i, j] = normal_likelihood(mu, sigma, sample_surface)
            
            # Create 3D surface plot
            fig_surface = go.Figure(data=[
                go.Surface(
                    z=np.log(likelihood_surface),  # Log-likelihood for better visualization
                    x=mean_range,
                    y=std_range,
                    colorscale='Viridis'
                )
            ])
            
            fig_surface.update_layout(
                title="Log-Likelihood Surface for Normal Distribution",
                scene=dict(
                    xaxis_title="Mean (μ)",
                    yaxis_title="Std Dev (σ)",
                    zaxis_title="Log-Likelihood"
                )
            )
            
            st.plotly_chart(fig_surface, use_container_width=True)
            
            # Find maximum likelihood estimates
            max_idx = np.unravel_index(np.argmax(likelihood_surface), likelihood_surface.shape)
            mle_mean_surface = mean_range[max_idx[0]]
            mle_std_surface = std_range[max_idx[1]]
            
            st.markdown(f"""
            ### Likelihood Surface Analysis
            
            - **True Mean**: {true_mean_surface}
            - **MLE Mean**: {mle_mean_surface:.4f}
            
            - **True Std Dev**: {true_std_surface}
            - **MLE Std Dev**: {mle_std_surface:.4f}
            """)
        
        else:  # Exponential
            # Parameters for surface
            true_rate_surface = st.slider("True Rate (λ)", 0.1, 5.0, 1.0, 0.1)
            
            # Generate sample
            np.random.seed(42)
            sample_surface = np.random.exponential(1/true_rate_surface, surface_sample_size)
            
            # Likelihood function for exponential distribution
            def exponential_likelihood(rate, data):
                return np.prod(stats.expon.pdf(data, scale=1/rate))
            
            # Create grid of parameter values
            rate_range = np.linspace(true_rate_surface - 2, true_rate_surface + 2, 200)
            
            # Compute likelihood surface
            likelihood_surface = [exponential_likelihood(rate, sample_surface) for rate in rate_range]
            
            # Create likelihood plot
            fig_surface = go.Figure(data=[
                go.Scatter(
                    x=rate_range, 
                    y=np.log(likelihood_surface),  # Log-likelihood
                    mode='lines',
                    name='Log-Likelihood'
                )
            ])
            
            # Add true rate line
            fig_surface.add_shape(
                type="line",
                x0=true_rate_surface,
                y0=min(np.log(likelihood_surface)),
                x1=true_rate_surface,
                y1=max(np.log(likelihood_surface)),
                line=dict(color="red", width=2, dash="dash")
            )
            
            fig_surface.update_layout(
                title="Log-Likelihood for Exponential Distribution",
                xaxis_title="Rate (λ)",
                yaxis_title="Log-Likelihood"
            )
            
            st.plotly_chart(fig_surface, use_container_width=True)
            
            # Find maximum likelihood estimate
            max_idx = np.argmax(likelihood_surface)
            mle_rate_surface = rate_range[max_idx]
            
            st.markdown(f"""
            ### Likelihood Surface Analysis
            
            - **True Rate (λ)**: {true_rate_surface}
            - **MLE Rate**: {mle_rate_surface:.4f}
            """)
    
    with tab3:
        st.subheader("Model Comparison")
        
        st.markdown("""
        ### Comparing Estimation Models
        
        Use likelihood-based methods to compare and select the best model.
        
        Key Metrics:
        - Log-Likelihood
        - Akaike Information Criterion (AIC)
        - Bayesian Information Criterion (BIC)
        """)
        
        # Model comparison parameters
        col1, col2 = st.columns(2)
        
        with col1:
            num_models = st.slider("Number of Models to Compare", 2, 5, 3)
        
        with col2:
            comparison_sample_size = st.slider("Sample Size for Comparison", 50, 1000, 200)
        
        # Generate models and data
        models = []
        for i in range(num_models):
            model_type = st.selectbox(
                f"Model {i+1} Distribution", 
                ["Normal", "Exponential", "Poisson"],
                key=f"model_type_{i}"
            )
            
            if model_type == "Normal":
                mean = st.slider(f"Model {i+1} Mean", -10.0, 10.0, 0.0, 0.1, key=f"mean_{i}")
                std = st.slider(f"Model {i+1} Std Dev", 0.1, 5.0, 1.0, 0.1, key=f"std_{i}")
                models.append({
                    'type': 'Normal',
                    'params': {'mean': mean, 'std': std}
                })
            
            elif model_type == "Exponential":
                rate = st.slider(f"Model {i+1} Rate (λ)", 0.1, 5.0, 1.0, 0.1, key=f"rate_{i}")
                models.append({
                    'type': 'Exponential',
                    'params': {'rate': rate}
                })
            
            else:  # Poisson
                lambda_param = st.slider(f"Model {i+1} Lambda (λ)", 0.1, 20.0, 5.0, 0.1, key=f"lambda_{i}")
                models.append({
                    'type': 'Poisson',
                    'params': {'lambda': lambda_param}
                })
        
        # Compute model comparison metrics
        def compute_model_metrics(models, sample_size):
            np.random.seed(42)
            model_metrics = []
            
            for model in models:
                # Generate sample data
                if model['type'] == 'Normal':
                    sample = np.random.normal(
                        model['params']['mean'], 
                        model['params']['std'], 
                        sample_size
                    )
                    
                    # Compute log-likelihood
                    log_likelihood = np.sum(stats.norm.logpdf(
                        sample, 
                        model['params']['mean'], 
                        model['params']['std']
                    ))
                    
                    # Compute AIC and BIC
                    num_params = 2  # mean and std
                
                elif model['type'] == 'Exponential':
                    sample = np.random.exponential(
                        1/model['params']['rate'], 
                        sample_size
                    )
                    
                    # Compute log-likelihood
                    log_likelihood = np.sum(stats.expon.logpdf(
                        sample, 
                        scale=1/model['params']['rate']
                    ))
                    
                    # Compute AIC and BIC
                    num_params = 1  # rate
                
                else:  # Poisson
                    sample = np.random.poisson(
                        model['params']['lambda'], 
                        sample_size
                    )
                    
                    # Compute log-likelihood
                    log_likelihood = np.sum(stats.poisson.logpmf(
                        sample, 
                        model['params']['lambda']
                    ))
                    
                    # Compute AIC and BIC
                    num_params = 1  # lambda
                
                # Compute information criteria
                aic = 2 * num_params - 2 * log_likelihood
                bic = num_params * np.log(sample_size) - 2 * log_likelihood
                
                model_metrics.append({
                    'Model': model['type'],
                    'Log-Likelihood': log_likelihood,
                    'AIC': aic,
                    'BIC': bic
                })
            
            return pd.DataFrame(model_metrics)
        
        # Compute and display model metrics
        model_metrics = compute_model_metrics(models, comparison_sample_size)
        
        st.subheader("Model Comparison Metrics")
        st.table(model_metrics)
        
        # Visualization of metrics
        fig_metrics = go.Figure()
        
        # Log-Likelihood
        fig_metrics.add_trace(go.Bar(
            x=model_metrics['Model'],
            y=model_metrics['Log-Likelihood'],
            name='Log-Likelihood'
        ))
        
        fig_metrics.update_layout(
            title="Model Comparison: Log-Likelihood",
            xaxis_title="Model",
            yaxis_title="Log-Likelihood"
        )
        
        st.plotly_chart(fig_metrics, use_container_width=True)
        
        # Information Criteria
        fig_ic = go.Figure()
        
        # AIC
        fig_ic.add_trace(go.Bar(
            x=model_metrics['Model'],
            y=model_metrics['AIC'],
            name='AIC'
        ))
        
        # BIC
        fig_ic.add_trace(go.Bar(
            x=model_metrics['Model'],
            y=model_metrics['BIC'],
            name='BIC'
        ))
        
        fig_ic.update_layout(
            title="Model Comparison: Information Criteria",
            xaxis_title="Model",
            yaxis_title="Information Criterion Value"
        )
        
        st.plotly_chart(fig_ic, use_container_width=True)
        
        # Interpretation
        st.markdown("""
        ### Interpreting Model Comparison Metrics
        
        #### Log-Likelihood
        - Higher values indicate better model fit
        - Measures how well the model explains the observed data
        
        #### Akaike Information Criterion (AIC)
        - Lower values indicate better models
        - Balances model fit and model complexity
        - Penalizes models with more parameters
        
        #### Bayesian Information Criterion (BIC)
        - Lower values indicate better models
        - More stringent penalty for model complexity
        - Tends to select simpler models
        
        **Choose the model with the lowest AIC or BIC**
        """)

if __name__ == "__main__":
    run_maximum_likelihood_estimation()
