import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import pandas as pd

def run_point_estimation():
    st.title("🎯 Point Estimation Explorer")
    
    st.markdown("""
    ### Understanding Point Estimation
    
    Point estimation is a method of statistical inference that produces a single 
    value (point estimate) as an approximation of a population parameter.
    
    Key estimation methods:
    - Maximum Likelihood Estimation (MLE)
    - Method of Moments
    - Bayesian Point Estimation
    """)
    
    # Tabs for different estimation techniques
    tab1, tab2, tab3 = st.tabs([
        "Maximum Likelihood", 
        "Method of Moments", 
        "Estimation Comparison"
    ])
    
    with tab1:
        st.subheader("Maximum Likelihood Estimation (MLE)")
        
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
        
        # Simulate data and estimate parameters
        def mle_estimation(distribution):
            np.random.seed(42)
            results = []
            
            for _ in range(num_simulations):
                # Generate sample data based on distribution
                if distribution == "Normal":
                    sample = np.random.normal(true_mean, true_std, sample_size)
                    # MLE for normal is simply sample mean and sample std
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
        fig_mle = go.Figure()
        
        if distribution == "Normal":
            # Histogram of estimated means
            fig_mle = go.Figure(data=[
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
            fig_mle.add_shape(
                type="line",
                x0=true_mean,
                y0=0,
                x1=true_mean,
                y1=y_max,
                line=dict(color="red", width=2, dash="dash")
            )
            
            fig_mle.update_layout(
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
            fig_mle = go.Figure(data=[
                go.Histogram(
                    x=mle_results['Estimated Rate'], 
                    name='Estimated Rate',
                    opacity=0.7
                )
            ])
            
            # Add true rate line
            fig_mle.add_shape(
                type="line",
                x0=true_rate,
                y0=0,
                x1=true_rate,
                y1=fig_mle.layout.yaxis.range[1],
                line=dict(color="red", width=2, dash="dash")
            )
            
            fig_mle.update_layout(
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
            fig_mle = go.Figure(data=[
                go.Histogram(
                    x=mle_results['Estimated Lambda'], 
                    name='Estimated Lambda',
                    opacity=0.7
                )
            ])
            
            # Add true lambda line
            fig_mle.add_shape(
                type="line",
                x0=true_lambda,
                y0=0,
                x1=true_lambda,
                y1=fig_mle.layout.yaxis.range[1],
                line=dict(color="red", width=2, dash="dash")
            )
            
            fig_mle.update_layout(
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
        
        st.plotly_chart(fig_mle, use_container_width=True)
    
    with tab2:
        st.subheader("Method of Moments Estimation")
        
        # Distribution selection
        mom_distribution = st.selectbox(
            "Select Distribution for Method of Moments", 
            ["Normal", "Exponential", "Gamma"]
        )
        
        # Parameters for Method of Moments
        col1, col2, col3 = st.columns(3)
        
        with col1:
            mom_sample_size = st.slider("Sample Size (MoM)", 10, 1000, 100)
        
        with col2:
            if mom_distribution == "Normal":
                mom_true_mean = st.slider("True Mean (MoM)", -10.0, 10.0, 0.0, 0.1)
                mom_true_std = st.slider("True Standard Deviation (MoM)", 0.1, 5.0, 1.0, 0.1)
            elif mom_distribution == "Exponential":
                mom_true_rate = st.slider("True Rate (λ) (MoM)", 0.1, 5.0, 1.0, 0.1)
            else:  # Gamma
                mom_true_shape = st.slider("True Shape (k)", 0.5, 10.0, 2.0, 0.1)
                mom_true_scale = st.slider("True Scale (θ)", 0.1, 5.0, 1.0, 0.1)
        
        with col3:
            mom_num_simulations = st.slider("Number of Simulations (MoM)", 10, 500, 100)
        
        # Simulate data and estimate parameters using Method of Moments
        def method_of_moments_estimation(distribution):
            np.random.seed(42)
            results = []
            
            for _ in range(mom_num_simulations):
                # Generate sample data based on distribution
                if distribution == "Normal":
                    sample = np.random.normal(mom_true_mean, mom_true_std, mom_sample_size)
                    # Method of Moments for normal: first moment = mean, second moment = variance
                    mom_mean = np.mean(sample)
                    mom_std = np.std(sample, ddof=1)
                    results.append({
                        'True Mean': mom_true_mean, 
                        'Estimated Mean': mom_mean,
                        'True Std': mom_true_std, 
                        'Estimated Std': mom_std
                    })
                
                elif distribution == "Exponential":
                    sample = np.random.exponential(1/mom_true_rate, mom_sample_size)
                    # Method of Moments for exponential: first moment = 1/rate
                    mom_rate = 1 / np.mean(sample)
                    results.append({
                        'True Rate': mom_true_rate, 
                        'Estimated Rate': mom_rate
                    })
                
                else:  # Gamma
                    sample = np.random.gamma(mom_true_shape, mom_true_scale, mom_sample_size)
                    # Method of Moments for Gamma: 
                    # First moment = shape * scale
                    # Second moment = shape * scale^2
                    sample_mean = np.mean(sample)
                    sample_var = np.var(sample)
                    
                    # Estimate parameters
                    mom_shape = (sample_mean**2) / sample_var
                    mom_scale = sample_var / sample_mean
                    
                    results.append({
                        'True Shape': mom_true_shape, 
                        'Estimated Shape': mom_shape,
                        'True Scale': mom_true_scale, 
                        'Estimated Scale': mom_scale
                    })
            
            return pd.DataFrame(results)
        
        # Perform Method of Moments estimation
        mom_results = method_of_moments_estimation(mom_distribution)
        
        # Visualization and summary
        fig_mom = go.Figure()
        
        if mom_distribution == "Normal":
            # Histogram of estimated means
            fig_mom = go.Figure(data=[
                go.Histogram(
                    x=mom_results['Estimated Mean'], 
                    name='Estimated Mean',
                    opacity=0.7
                )
            ])
            
            # Compute y_max for vertical lines
            counts, _ = np.histogram(mom_results['Estimated Mean'], bins='auto')
            y_max = 1.1 * counts.max()
            # Add true mean line
            fig_mom.add_shape(
                type="line",
                x0=mom_true_mean,
                y0=0,
                x1=mom_true_mean,
                y1=y_max,
                line=dict(color="red", width=2, dash="dash")
            )
            
            fig_mom.update_layout(
                title=f"Method of Moments Estimates for {mom_distribution} Distribution Mean",
                xaxis_title="Estimated Mean",
                yaxis_title="Frequency"
            )
            
            # Display summary statistics
            st.markdown(f"""
            ### Method of Moments Estimation Results
            
            - **True Mean**: {mom_true_mean}
            - **Estimated Mean (Average)**: {mom_results['Estimated Mean'].mean():.4f}
            - **Mean Estimation Error**: {abs(mom_results['Estimated Mean'].mean() - mom_true_mean):.4f}
            
            - **True Std Dev**: {mom_true_std}
            - **Estimated Std Dev (Average)**: {mom_results['Estimated Std'].mean():.4f}
            - **Std Dev Estimation Error**: {abs(mom_results['Estimated Std'].mean() - mom_true_std):.4f}
            """)
        
        elif mom_distribution == "Exponential":
            # Histogram of estimated rates
            fig_mom = go.Figure(data=[
                go.Histogram(
                    x=mom_results['Estimated Rate'], 
                    name='Estimated Rate',
                    opacity=0.7
                )
            ])
            
            # Add true rate line
            fig_mom.add_shape(
                type="line",
                x0=mom_true_rate,
                y0=0,
                x1=mom_true_rate,
                y1=fig_mom.layout.yaxis.range[1],
                line=dict(color="red", width=2, dash="dash")
            )
            
            fig_mom.update_layout(
                title=f"Method of Moments Estimates for {mom_distribution} Distribution Rate",
                xaxis_title="Estimated Rate",
                yaxis_title="Frequency"
            )
            
            # Display summary statistics
            st.markdown(f"""
            ### Method of Moments Estimation Results
            
            - **True Rate (λ)**: {mom_true_rate}
            - **Estimated Rate (Average)**: {mom_results['Estimated Rate'].mean():.4f}
            - **Rate Estimation Error**: {abs(mom_results['Estimated Rate'].mean() - mom_true_rate):.4f}
            """)
        
        else:  # Gamma
            # Histogram of estimated shape and scale
            fig_mom = go.Figure()
            
            # Shape parameter
            fig_mom.add_trace(go.Histogram(
                x=mom_results['Estimated Shape'], 
                name='Estimated Shape',
                opacity=0.7
            ))
            
            # Add true shape line
            fig_mom.add_shape(
                type="line",
                x0=mom_true_shape,
                y0=0,
                x1=mom_true_shape,
                y1=fig_mom.layout.yaxis.range[1],
                line=dict(color="red", width=2, dash="dash")
            )
            
            fig_mom.update_layout(
                title=f"Method of Moments Estimates for {mom_distribution} Distribution Parameters",
                xaxis_title="Estimated Shape",
                yaxis_title="Frequency"
            )
            
            # Display summary statistics
            st.markdown(f"""
            ### Method of Moments Estimation Results
            
            - **True Shape (k)**: {mom_true_shape}
            - **Estimated Shape (Average)**: {mom_results['Estimated Shape'].mean():.4f}
            - **Shape Estimation Error**: {abs(mom_results['Estimated Shape'].mean() - mom_true_shape):.4f}
            
            - **True Scale (θ)**: {mom_true_scale}
            - **Estimated Scale (Average)**: {mom_results['Estimated Scale'].mean():.4f}
            - **Scale Estimation Error**: {abs(mom_results['Estimated Scale'].mean() - mom_true_scale):.4f}
            """)
        
        st.plotly_chart(fig_mom, use_container_width=True)
    
    with tab3:
        st.subheader("Estimation Method Comparison")
        
        st.markdown("""
        ### Comparing Estimation Techniques
        
        Different estimation methods can yield different results based on:
        - Underlying distribution
        - Sample characteristics
        - Estimation method properties
        """)
        
        # Comparison parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            comp_distribution = st.selectbox(
                "Distribution for Comparison", 
                ["Normal", "Exponential"]
            )
        
        with col2:
            comp_sample_size = st.slider("Sample Size (Comparison)", 10, 1000, 100)
        
        with col3:
            comp_num_simulations = st.slider("Number of Simulations (Comparison)", 10, 500, 100)
        
        # True parameters
        if comp_distribution == "Normal":
            true_mean = st.slider("True Mean (Comparison)", -10.0, 10.0, 0.0, 0.1)
            true_std = st.slider("True Standard Deviation (Comparison)", 0.1, 5.0, 1.0, 0.1)
        else:  # Exponential
            true_rate = st.slider("True Rate (λ) (Comparison)", 0.1, 5.0, 1.0, 0.1)
        
        # Comparison function
        def compare_estimation_methods(distribution):
            np.random.seed(42)
            mle_results = []
            mom_results = []
            
            for _ in range(comp_num_simulations):
                # Generate sample data
                if distribution == "Normal":
                    sample = np.random.normal(true_mean, true_std, comp_sample_size)
                    
                    # MLE Estimation
                    mle_mean = np.mean(sample)
                    mle_std = np.std(sample, ddof=1)
                    
                    # Method of Moments Estimation
                    mom_mean = np.mean(sample)
                    mom_std = np.std(sample, ddof=1)
                    
                    mle_results.append({
                        'Estimated Mean': mle_mean,
                        'Estimated Std': mle_std
                    })
                    
                    mom_results.append({
                        'Estimated Mean': mom_mean,
                        'Estimated Std': mom_std
                    })
                
                else:  # Exponential
                    sample = np.random.exponential(1/true_rate, comp_sample_size)
                    
                    # MLE Estimation (1/sample_mean)
                    mle_rate = 1 / np.mean(sample)
                    
                    # Method of Moments Estimation (1/sample_mean)
                    mom_rate = 1 / np.mean(sample)
                    
                    mle_results.append({
                        'Estimated Rate': mle_rate
                    })
                    
                    mom_results.append({
                        'Estimated Rate': mom_rate
                    })
            
            return pd.DataFrame(mle_results), pd.DataFrame(mom_results)
        
        # Perform comparison
        mle_comp, mom_comp = compare_estimation_methods(comp_distribution)
        
        # Visualization
        if comp_distribution == "Normal":
            # Create side-by-side boxplots for mean estimation
            fig_comp_mean = go.Figure()
            
            fig_comp_mean.add_trace(go.Box(
                y=mle_comp['Estimated Mean'], 
                name='MLE Mean Estimates'
            ))
            
            fig_comp_mean.add_trace(go.Box(
                y=mom_comp['Estimated Mean'], 
                name='MoM Mean Estimates'
            ))
            
            # Add true mean line
            fig_comp_mean.add_shape(
                type="line",
                x0=-1,
                y0=true_mean,
                x1=1,
                y1=true_mean,
                line=dict(color="red", width=2, dash="dash")
            )
            
            fig_comp_mean.update_layout(
                title="Comparison of Mean Estimation Methods",
                yaxis_title="Estimated Mean"
            )
            
            st.plotly_chart(fig_comp_mean, use_container_width=True)
            
            # Mean estimation summary
            st.markdown(f"""
            ### Mean Estimation Comparison
            
            **MLE Mean Estimates**:
            - Average: {mle_comp['Estimated Mean'].mean():.4f}
            - Standard Deviation: {mle_comp['Estimated Mean'].std():.4f}
            
            **Method of Moments Mean Estimates**:
            - Average: {mom_comp['Estimated Mean'].mean():.4f}
            - Standard Deviation: {mom_comp['Estimated Mean'].std():.4f}
            
            **True Mean**: {true_mean}
            """)
            
            # Create side-by-side boxplots for std estimation
            fig_comp_std = go.Figure()
            
            fig_comp_std.add_trace(go.Box(
                y=mle_comp['Estimated Std'], 
                name='MLE Std Estimates'
            ))
            
            fig_comp_std.add_trace(go.Box(
                y=mom_comp['Estimated Std'], 
                name='MoM Std Estimates'
            ))
            
            # Add true std line
            fig_comp_std.add_shape(
                type="line",
                x0=-1,
                y0=true_std,
                x1=1,
                y1=true_std,
                line=dict(color="red", width=2, dash="dash")
            )
            
            fig_comp_std.update_layout(
                title="Comparison of Standard Deviation Estimation Methods",
                yaxis_title="Estimated Std Dev"
            )
            
            st.plotly_chart(fig_comp_std, use_container_width=True)
            
            # Std estimation summary
            st.markdown(f"""
            ### Standard Deviation Estimation Comparison
            
            **MLE Std Estimates**:
            - Average: {mle_comp['Estimated Std'].mean():.4f}
            - Standard Deviation: {mle_comp['Estimated Std'].std():.4f}
            
            **Method of Moments Std Estimates**:
            - Average: {mom_comp['Estimated Std'].mean():.4f}
            - Standard Deviation: {mom_comp['Estimated Std'].std():.4f}
            
            **True Std Dev**: {true_std}
            """)
        
        else:  # Exponential
            # Create side-by-side boxplots for rate estimation
            fig_comp_rate = go.Figure()
            
            fig_comp_rate.add_trace(go.Box(
                y=mle_comp['Estimated Rate'], 
                name='MLE Rate Estimates'
            ))
            
            fig_comp_rate.add_trace(go.Box(
                y=mom_comp['Estimated Rate'], 
                name='MoM Rate Estimates'
            ))
            
            # Add true rate line
            fig_comp_rate.add_shape(
                type="line",
                x0=-1,
                y0=true_rate,
                x1=1,
                y1=true_rate,
                line=dict(color="red", width=2, dash="dash")
            )
            
            fig_comp_rate.update_layout(
                title="Comparison of Rate Estimation Methods",
                yaxis_title="Estimated Rate"
            )
            
            st.plotly_chart(fig_comp_rate, use_container_width=True)
            
            # Rate estimation summary
            st.markdown(f"""
            ### Rate Estimation Comparison
            
            **MLE Rate Estimates**:
            - Average: {mle_comp['Estimated Rate'].mean():.4f}
            - Standard Deviation: {mle_comp['Estimated Rate'].std():.4f}
            
            **Method of Moments Rate Estimates**:
            - Average: {mom_comp['Estimated Rate'].mean():.4f}
            - Standard Deviation: {mom_comp['Estimated Rate'].std():.4f}
            
            **True Rate**: {true_rate}
            """)

if __name__ == "__main__":
    run_point_estimation()
