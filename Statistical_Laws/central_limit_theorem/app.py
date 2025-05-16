import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import pandas as pd

def run_central_limit_theorem():
    st.title("🔬 Central Limit Theorem Explorer")
    
    st.markdown("""
    ### Understanding the Central Limit Theorem (CLT)
    
    The Central Limit Theorem states that the distribution of sample means 
    approaches a normal distribution as sample size increases, 
    regardless of the underlying population distribution.
    
    Key Insights:
    - Works for any distribution with finite variance
    - Sample size matters
    - Convergence to normal distribution
    """)
    
    # Tabs for different CLT explorations
    tab1, tab2, tab3 = st.tabs([
        "Distribution Convergence", 
        "Sample Size Impact", 
        "Real-World Distributions"
    ])
    
    with tab1:
        st.subheader("Distribution Convergence")
        
        # Original distribution selection
        original_dist = st.selectbox(
            "Original Population Distribution", 
            ["Uniform", "Exponential", "Skewed"]
        )
        
        # Parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            num_simulations = st.slider("Number of Simulations", 100, 10000, 1000)
        
        with col2:
            sample_size = st.slider("Sample Size", 10, 500, 50)
        
        with col3:
            if original_dist == "Uniform":
                dist_min = st.slider("Distribution Minimum", -10.0, 0.0, -5.0, 0.1)
                dist_max = st.slider("Distribution Maximum", 0.0, 10.0, 5.0, 0.1)
            elif original_dist == "Exponential":
                rate = st.slider("Rate (λ)", 0.1, 5.0, 1.0, 0.1)
            else:  # Skewed
                skew_param = st.slider("Skewness Parameter", 0.1, 10.0, 2.0, 0.1)
        
        # Simulate sample means
        def generate_sample_means(distribution, num_sims, sample_size, dist_min=None, dist_max=None, rate=None, skew_param=None):
            sample_means = []
            for _ in range(num_sims):
                if distribution == "Uniform":
                    sample = np.random.uniform(dist_min, dist_max, sample_size)
                elif distribution == "Exponential":
                    sample = np.random.exponential(1/rate, sample_size)
                else:  # Skewed
                    sample = np.random.gamma(skew_param, 1.0, sample_size)
                sample_means.append(np.mean(sample))
            return sample_means
        
        # Generate sample means
        if original_dist == "Uniform":
            sample_means = generate_sample_means("Uniform", num_simulations, sample_size, dist_min=dist_min, dist_max=dist_max)
        elif original_dist == "Exponential":
            sample_means = generate_sample_means("Exponential", num_simulations, sample_size, rate=rate)
        else:  # Skewed
            sample_means = generate_sample_means("Skewed", num_simulations, sample_size, skew_param=skew_param)
        
        # Visualization
        fig_dist = go.Figure()
        
        # Original distribution
        x_orig = np.linspace(min(sample_means), max(sample_means), 200)
        
        # Theoretical normal distribution
        if original_dist == "Uniform":
            true_mean = (dist_min + dist_max) / 2
            true_std = np.sqrt((dist_max - dist_min)**2 / 12)
        elif original_dist == "Exponential":
            true_mean = 1 / rate
            true_std = 1 / rate
        else:  # Skewed
            true_mean = skew_param
            true_std = np.sqrt(skew_param)
        
        # Theoretical normal distribution for sample means
        theo_mean = true_mean
        theo_std = true_std / np.sqrt(sample_size)
        theo_pdf = stats.norm.pdf(x_orig, theo_mean, theo_std)
        
        # Histogram of sample means
        fig_dist.add_trace(go.Histogram(
            x=sample_means, 
            name='Sample Means',
            histnorm='probability density',
            opacity=0.7
        ))
        
        # Theoretical normal distribution
        fig_dist.add_trace(go.Scatter(
            x=x_orig,
            y=theo_pdf,
            mode='lines',
            name='Theoretical Normal',
            line=dict(color='red', dash='dot')
        ))
        
        fig_dist.update_layout(
            title=f"Sample Means Distribution (n = {sample_size})",
            xaxis_title="Sample Mean",
            yaxis_title="Probability Density"
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Summary statistics
        sample_mean_dist = np.mean(sample_means)
        sample_std_dist = np.std(sample_means)
        
        st.markdown(f"""
        ### Distribution Convergence Analysis
        
        **Theoretical Parameters**:
        - Population Mean: {true_mean:.4f}
        - Population Std Dev: {true_std:.4f}
        
        **Sample Means Distribution**:
        - Mean of Sample Means: {sample_mean_dist:.4f}
        - Std Dev of Sample Means: {sample_std_dist:.4f}
        
        **Theoretical Sample Mean Distribution**:
        - Expected Mean: {theo_mean:.4f}
        - Expected Std Dev: {theo_std:.4f}
        """)
    
    with tab2:
        st.subheader("Impact of Sample Size")
        
        st.markdown("""
        ### How Sample Size Affects Distribution
        
        Explore how increasing sample size influences:
        - Convergence to normal distribution
        - Reduction in sampling variability
        """)
        
        # Sample size comparison
        sample_sizes = st.multiselect(
            "Select Sample Sizes to Compare", 
            [10, 30, 50, 100, 200, 500],
            default=[10, 50, 200]
        )
        
        # Comparison of sample means distributions
        fig_sample_size = go.Figure()
        
        for size in sample_sizes:
            # Generate sample means
            sample_means_comp = generate_sample_means("Uniform", num_simulations, size, dist_min=dist_min, dist_max=dist_max)
            
            # Add histogram
            fig_sample_size.add_trace(go.Histogram(
                x=sample_means_comp, 
                name=f'n = {size}',
                histnorm='probability density',
                opacity=0.5
            ))
        
        fig_sample_size.update_layout(
            title="Sample Means Distribution for Different Sample Sizes",
            xaxis_title="Sample Mean",
            yaxis_title="Probability Density"
        )
        
        st.plotly_chart(fig_sample_size, use_container_width=True)
        
        # Detailed analysis
        st.subheader("Sample Size Impact Analysis")
        
        analysis_data = []
        for size in sample_sizes:
            sample_means_analysis = generate_sample_means("Uniform", num_simulations, size, dist_min=dist_min, dist_max=dist_max)
            analysis_data.append({
                'Sample Size': size,
                'Mean of Sample Means': np.mean(sample_means_analysis),
                'Std Dev of Sample Means': np.std(sample_means_analysis)
            })
        
        analysis_df = pd.DataFrame(analysis_data)
        st.table(analysis_df)
        
        st.markdown("""
        ### Observations
        
        - As sample size increases, sample means distribution:
          1. Becomes more normally distributed
          2. Concentrates around the true population mean
          3. Reduces in variability
        
        #### Mathematical Explanation
        
        For sample mean X̄ from a population with:
        - Mean μ
        - Standard deviation σ
        
        The sampling distribution of X̄ has:
        - Mean = μ
        - Standard deviation = σ / √n
        """)
    
    with tab3:
        st.subheader("Real-World Distribution Examples")
        
        st.markdown("""
        ### Applying Central Limit Theorem in Practice
        
        Explore how CLT works with various real-world distributions:
        - Height measurements
        - Income distributions
        - Biological measurements
        """)
        
        # Real-world distribution selection
        distribution_type = st.selectbox(
            "Select Real-World Distribution", 
            ["Height", "Income", "Plant Growth"]
        )
        
        # Simulation parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            real_num_simulations = st.slider("Number of Simulations (Real-World)", 100, 10000, 1000)
        
        with col2:
            real_sample_size = st.slider("Sample Size (Real-World)", 10, 500, 50)
        
        with col3:
            if distribution_type == "Height":
                population_mean = st.slider("Population Mean Height (cm)", 150.0, 190.0, 170.0, 0.1)
                population_std = st.slider("Population Std Dev (cm)", 5.0, 20.0, 10.0, 0.1)
            elif distribution_type == "Income":
                population_mean = st.slider("Population Mean Income ($)", 30000.0, 100000.0, 50000.0, 100.0)
                population_std = st.slider("Population Std Dev ($)", 5000.0, 50000.0, 15000.0, 100.0)
            else:  # Plant Growth
                population_mean = st.slider("Population Mean Growth (cm)", 10.0, 50.0, 25.0, 0.1)
                population_std = st.slider("Population Std Dev (cm)", 2.0, 15.0, 5.0, 0.1)
        
        # Simulate real-world sample means
        def generate_real_world_sample_means(num_sims, sample_size, mean, std):
            sample_means = []
            
            for _ in range(num_sims):
                # Simulate slightly skewed distribution
                sample = np.random.normal(mean, std) + 0.5 * np.random.normal(0, std, sample_size)
                sample_means.append(np.mean(sample))
            
            return sample_means
        
        # Generate sample means
        real_sample_means = generate_real_world_sample_means(
            real_num_simulations, 
            real_sample_size, 
            population_mean, 
            population_std
        )
        
        # Visualization
        fig_real = go.Figure()
        
        # Histogram of sample means
        fig_real.add_trace(go.Histogram(
            x=real_sample_means, 
            name='Sample Means',
            histnorm='probability density',
            opacity=0.7
        ))
        
        # Theoretical normal distribution
        x_real = np.linspace(min(real_sample_means), max(real_sample_means), 200)
        theo_mean_real = population_mean
        theo_std_real = population_std / np.sqrt(real_sample_size)
        theo_pdf_real = stats.norm.pdf(x_real, theo_mean_real, theo_std_real)
        
        fig_real.add_trace(go.Scatter(
            x=x_real,
            y=theo_pdf_real,
            mode='lines',
            name='Theoretical Normal',
            line=dict(color='red', dash='dot')
        ))
        
        fig_real.update_layout(
            title=f"{distribution_type} Sample Means Distribution (n = {real_sample_size})",
            xaxis_title="Sample Mean",
            yaxis_title="Probability Density"
        )
        
        st.plotly_chart(fig_real, use_container_width=True)
        
        # Summary statistics
        sample_mean_real = np.mean(real_sample_means)
        sample_std_real = np.std(real_sample_means)
        
        st.markdown(f"""
        ### Real-World Distribution Analysis
        
        **Population Parameters**:
        - Mean: {population_mean:.2f}
        - Standard Deviation: {population_std:.2f}
        
        **Sample Means Distribution**:
        - Mean of Sample Means: {sample_mean_real:.2f}
        - Std Dev of Sample Means: {sample_std_real:.2f}
        
        **Theoretical Sample Mean Distribution**:
        - Expected Mean: {theo_mean_real:.2f}
        - Expected Std Dev: {theo_std_real:.2f}
        
        #### Practical Implications
        The Central Limit Theorem allows us to make statistical 
        inferences about population parameters using sample statistics, 
        even when the underlying distribution is not perfectly normal.
        """)

if __name__ == "__main__":
    run_central_limit_theorem()
