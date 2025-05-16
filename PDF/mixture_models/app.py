import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import pandas as pd
from sklearn.mixture import GaussianMixture

def run_mixture_models():
    st.title("🔬 Mixture Models Explorer")
    
    st.markdown("""
    ### Understanding Mixture Models
    
    Mixture models are probabilistic models for representing 
    populations that are composed of subpopulations with different characteristics.
    
    Key Concepts:
    - Gaussian Mixture Models (GMM)
    - Component Distributions
    - Probability Density Estimation
    - Clustering and Classification
    """)
    
    # Tabs for different mixture model analyses
    tab1, tab2, tab3 = st.tabs([
        "Basic Mixture Models", 
        "Model Complexity", 
        "Real-World Applications"
    ])
    
    with tab1:
        st.subheader("Gaussian Mixture Model Exploration")
        
        # Distribution configuration
        col1, col2, col3 = st.columns(3)
        
        with col1:
            num_components = st.slider("Number of Components", 2, 5, 3)
        
        with col2:
            sample_size = st.slider("Sample Size", 100, 10000, 1000)
        
        with col3:
            noise_level = st.slider("Noise Level", 0.0, 2.0, 0.5, 0.1)
        
        # Mixture model generation
        def generate_mixture_model(num_components, sample_size, noise_level):
            np.random.seed(42)
            
            # Component means and standard deviations
            means = np.linspace(-10, 10, num_components)
            stds = np.random.uniform(0.5, 2.0, num_components)
            
            # Mixing weights
            weights = np.random.dirichlet(np.ones(num_components))
            
            # Generate samples
            samples = []
            labels = []
            
            for i in range(num_components):
                component_samples = np.random.normal(
                    means[i], 
                    stds[i], 
                    int(sample_size * weights[i])
                )
                samples.extend(component_samples)
                labels.extend([i] * len(component_samples))
            
            # Add noise
            samples = np.array(samples)
            samples += np.random.normal(0, noise_level, len(samples))
            
            return samples, labels, means, stds, weights
        
        # Generate mixture model data
        samples, true_labels, true_means, true_stds, true_weights = generate_mixture_model(
            num_components, sample_size, noise_level
        )
        
        # Fit Gaussian Mixture Model
        def fit_gmm(samples, num_components):
            gmm = GaussianMixture(
                n_components=num_components, 
                random_state=42
            )
            gmm.fit(samples.reshape(-1, 1))
            return gmm
        
        # Fit GMM
        gmm = fit_gmm(samples, num_components)
        
        # Visualization
        x = np.linspace(min(samples), max(samples), 300)
        
        # Overall mixture PDF
        mixture_pdf = np.zeros_like(x)
        
        # Individual component PDFs
        fig_pdf = go.Figure()
        
        for i in range(num_components):
            # Component PDF
            component_pdf = stats.norm.pdf(
                x, 
                gmm.means_[i][0], 
                np.sqrt(gmm.covariances_[i][0][0])
            ) * gmm.weights_[i]
            
            # Add component PDF trace
            fig_pdf.add_trace(go.Scatter(
                x=x,
                y=component_pdf,
                mode='lines',
                name=f'Component {i+1}',
                line=dict(width=2)
            ))
            
            mixture_pdf += component_pdf
        
        # Overall mixture PDF
        fig_pdf.add_trace(go.Scatter(
            x=x,
            y=mixture_pdf,
            mode='lines',
            name='Mixture PDF',
            line=dict(color='black', width=3, dash='dot')
        ))
        
        fig_pdf.update_layout(
            title="Mixture Model Probability Density Functions",
            xaxis_title="Value",
            yaxis_title="Probability Density"
        )
        
        st.plotly_chart(fig_pdf, use_container_width=True)
        
        # Scatter plot with component assignments
        fig_scatter = go.Figure()
        
        # Scatter plot of samples
        fig_scatter.add_trace(go.Scatter(
            x=samples,
            y=np.zeros_like(samples),
            mode='markers',
            name='Samples',
            marker=dict(
                color=gmm.predict(samples.reshape(-1, 1)),
                colorscale='Viridis',
                showscale=True
            )
        ))
        
        fig_scatter.update_layout(
            title="Mixture Model Sample Distribution",
            xaxis_title="Value",
            yaxis_title="Component Assignment"
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Model parameter comparison
        st.subheader("Mixture Model Parameters")
        
        # Create comparison DataFrame
        param_df = pd.DataFrame({
            'Component': range(1, num_components + 1),
            'True Mean': true_means,
            'Estimated Mean': gmm.means_.flatten(),
            'True Std Dev': true_stds,
            'Estimated Std Dev': np.sqrt(gmm.covariances_.flatten()),
            'True Weight': true_weights,
            'Estimated Weight': gmm.weights_
        })
        
        st.table(param_df)
        
        st.markdown("""
        ### Interpretation
        
        - **Mixture Model**: Represents complex, multi-modal distributions
        - **Components**: Represent different subpopulations
        - **Weights**: Proportion of each component in the mixture
        """)
    
    with tab2:
        st.subheader("Model Complexity Analysis")
        
        st.markdown("""
        ### Exploring Mixture Model Complexity
        
        Investigate how the number of components affects:
        - Model fit
        - Probability density estimation
        - Model selection criteria
        """)
        
        # Complexity parameters
        col1, col2 = st.columns(2)
        
        with col1:
            max_components = st.slider("Maximum Number of Components", 2, 10, 6)
        
        with col2:
            complexity_sample_size = st.slider("Sample Size (Complexity)", 500, 10000, 2000)
        
        # Model complexity analysis
        def analyze_model_complexity(max_components, sample_size):
            np.random.seed(42)
            
            # Generate complex mixture model
            samples, _, _, _, _ = generate_mixture_model(
                min(4, max_components), 
                sample_size, 
                noise_level=0.5
            )
            
            # Compute model selection criteria
            aic_scores = []
            bic_scores = []
            
            for n_components in range(1, max_components + 1):
                gmm = GaussianMixture(
                    n_components=n_components, 
                    random_state=42
                )
                gmm.fit(samples.reshape(-1, 1))
                
                aic_scores.append(gmm.aic(samples.reshape(-1, 1)))
                bic_scores.append(gmm.bic(samples.reshape(-1, 1)))
            
            return {
                'Components': list(range(1, max_components + 1)),
                'AIC': aic_scores,
                'BIC': bic_scores
            }
        
        # Compute model complexity metrics
        complexity_results = analyze_model_complexity(max_components, complexity_sample_size)
        
        # Visualization
        fig_complexity = go.Figure()
        
        # AIC
        fig_complexity.add_trace(go.Scatter(
            x=complexity_results['Components'],
            y=complexity_results['AIC'],
            mode='lines+markers',
            name='Akaike Information Criterion (AIC)'
        ))
        
        # BIC
        fig_complexity.add_trace(go.Scatter(
            x=complexity_results['Components'],
            y=complexity_results['BIC'],
            mode='lines+markers',
            name='Bayesian Information Criterion (BIC)'
        ))
        
        fig_complexity.update_layout(
            title="Model Complexity: Information Criteria",
            xaxis_title="Number of Components",
            yaxis_title="Information Criterion Value"
        )
        
        st.plotly_chart(fig_complexity, use_container_width=True)
        
        # Create DataFrame for detailed analysis
        complexity_df = pd.DataFrame(complexity_results)
        st.table(complexity_df)
        
        st.markdown("""
        ### Model Selection Insights
        
        #### Akaike Information Criterion (AIC)
        - Balances model complexity and goodness of fit
        - Lower values indicate better models
        
        #### Bayesian Information Criterion (BIC)
        - More stringent penalty for model complexity
        - Tends to select simpler models
        
        **Choose the model with the lowest AIC or BIC**
        """)
    
    with tab3:
        st.subheader("Real-World Mixture Model Applications")
        
        st.markdown("""
        ### Practical Uses of Mixture Models
        
        Explore mixture models in various domains:
        - Customer Segmentation
        - Anomaly Detection
        - Biometric Data Analysis
        """)
        
        # Application selection
        application = st.selectbox(
            "Select Application Domain", 
            ["Customer Segments", "Sensor Data", "Biological Measurements"]
        )
        
        # Parameters
        col1, col2 = st.columns(2)
        
        with col1:
            app_sample_size = st.slider("Sample Size (Application)", 500, 10000, 2000)
        
        with col2:
            app_noise_level = st.slider("Noise Level (Application)", 0.0, 2.0, 0.5, 0.1)
        
        # Real-world data simulation
        def simulate_real_world_mixture(application, sample_size, noise_level):
            np.random.seed(42)
            
            if application == "Customer Segments":
                # Simulate customer spending
                means = [30000, 75000, 150000]  # Income levels
                stds = [10000, 20000, 30000]
                weights = [0.4, 0.4, 0.2]
            
            elif application == "Sensor Data":
                # Simulate sensor readings with different noise characteristics
                means = [50, 100, 200]
                stds = [5, 15, 30]
                weights = [0.3, 0.5, 0.2]
            
            else:  # Biological Measurements
                # Simulate cell size or protein concentrations
                means = [10, 25, 50]
                stds = [2, 5, 10]
                weights = [0.5, 0.3, 0.2]
            
            # Generate samples
            samples = []
            labels = []
            
            for i in range(len(means)):
                component_samples = np.random.normal(
                    means[i], 
                    stds[i], 
                    int(sample_size * weights[i])
                )
                samples.extend(component_samples)
                labels.extend([i] * len(component_samples))
            
            # Add noise
            samples = np.array(samples)
            samples += np.random.normal(0, noise_level, len(samples))
            
            return samples, labels, means, stds, weights
        
        # Generate application-specific mixture model
        app_samples, true_labels, true_means, true_stds, true_weights = simulate_real_world_mixture(
            application, app_sample_size, app_noise_level
        )
        
        # Fit Gaussian Mixture Model
        app_gmm = fit_gmm(app_samples, len(true_means))
        
        # Visualization
        app_x = np.linspace(min(app_samples), max(app_samples), 300)
        
        # Overall mixture PDF
        app_mixture_pdf = np.zeros_like(app_x)
        
        # Individual component PDFs
        fig_app_pdf = go.Figure()
        
        for i in range(len(true_means)):
            # Component PDF
            component_pdf = stats.norm.pdf(
                app_x, 
                app_gmm.means_[i][0], 
                np.sqrt(app_gmm.covariances_[i][0][0])
            ) * app_gmm.weights_[i]
            
            # Add component PDF trace
            fig_app_pdf.add_trace(go.Scatter(
                x=app_x,
                y=component_pdf,
                mode='lines',
                name=f'Component {i+1}',
                line=dict(width=2)
            ))
            
            app_mixture_pdf += component_pdf
        
        # Overall mixture PDF
        fig_app_pdf.add_trace(go.Scatter(
            x=app_x,
            y=app_mixture_pdf,
            mode='lines',
            name='Mixture PDF',
            line=dict(color='black', width=3, dash='dot')
        ))
        
        fig_app_pdf.update_layout(
            title=f"Mixture Model PDF for {application}",
            xaxis_title="Value",
            yaxis_title="Probability Density"
        )
        
        st.plotly_chart(fig_app_pdf, use_container_width=True)
        
        # Scatter plot with component assignments
        fig_app_scatter = go.Figure()
        
        # Scatter plot of samples
        fig_app_scatter.add_trace(go.Scatter(
            x=app_samples,
            y=np.zeros_like(app_samples),
            mode='markers',
            name='Samples',
            marker=dict(
                color=app_gmm.predict(app_samples.reshape(-1, 1)),
                colorscale='Viridis',
                showscale=True
            )
        ))
        
        fig_app_scatter.update_layout(
            title=f"{application} Mixture Model Sample Distribution",
            xaxis_title="Value",
            yaxis_title="Component Assignment"
        )
        
        st.plotly_chart(fig_app_scatter, use_container_width=True)
        
        # Model parameter comparison
        st.subheader(f"{application} Mixture Model Parameters")
        
        # Create comparison DataFrame
        app_param_df = pd.DataFrame({
            'Component': range(1, len(true_means) + 1),
            'True Mean': true_means,
            'Estimated Mean': app_gmm.means_.flatten(),
            'True Std Dev': true_stds,
            'Estimated Std Dev': np.sqrt(app_gmm.covariances_.flatten()),
            'True Weight': true_weights,
            'Estimated Weight': app_gmm.weights_
        })
        
        st.table(app_param_df)
        
        st.markdown(f"""
        ### {application} Mixture Model Insights
        
        **Practical Applications**:
        - Identify distinct subpopulations
        - Understand complex data distributions
        - Perform probabilistic clustering
        """)

if __name__ == "__main__":
    run_mixture_models()
