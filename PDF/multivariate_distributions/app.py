import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
import pandas as pd

def run_multivariate_distributions():
    st.title("🌐 Multivariate Probability Distributions")
    
    st.markdown("""
    ### Understanding Multivariate Distributions
    
    Multivariate probability distributions describe the probability of multiple random variables 
    occurring together. These distributions are essential for modeling relationships between 
    variables and understanding their joint behavior.
    
    This app allows you to explore:
    - Bivariate normal distributions
    - Correlation and covariance
    - Conditional distributions
    - Multivariate sampling
    """)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs([
        "Bivariate Normal Distribution", 
        "Correlation Explorer", 
        "Conditional Distributions"
    ])
    
    with tab1:
        st.subheader("Bivariate Normal Distribution")
        
        st.markdown("""
        The bivariate normal distribution is characterized by:
        - Mean vector (μ₁, μ₂)
        - Standard deviations (σ₁, σ₂)
        - Correlation coefficient (ρ)
        
        Adjust the parameters below to see how they affect the distribution:
        """)
        
        # Parameters for bivariate normal
        col1, col2, col3 = st.columns(3)
        
        with col1:
            mu_x = st.slider("Mean X (μ₁)", -3.0, 3.0, 0.0, 0.1)
            sigma_x = st.slider("Std Dev X (σ₁)", 0.1, 3.0, 1.0, 0.1)
        
        with col2:
            mu_y = st.slider("Mean Y (μ₂)", -3.0, 3.0, 0.0, 0.1)
            sigma_y = st.slider("Std Dev Y (σ₂)", 0.1, 3.0, 1.0, 0.1)
        
        with col3:
            rho = st.slider("Correlation (ρ)", -0.99, 0.99, 0.5, 0.01)
        
        # Create grid for bivariate normal
        x = np.linspace(mu_x - 3*sigma_x, mu_x + 3*sigma_x, 100)
        y = np.linspace(mu_y - 3*sigma_y, mu_y + 3*sigma_y, 100)
        X, Y = np.meshgrid(x, y)
        
        # Calculate PDF values
        Z = bivariate_normal_pdf(X, Y, mu_x, mu_y, sigma_x, sigma_y, rho)
        
        # Create 3D surface plot
        fig1 = go.Figure(data=[go.Surface(z=Z, x=x, y=y, colorscale='Viridis')])
        
        fig1.update_layout(
            title="Bivariate Normal PDF",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="f(x,y)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            height=600
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # Create contour plot
        fig2 = go.Figure(data=[
            go.Contour(
                z=Z,
                x=x,
                y=y,
                colorscale='Viridis',
                contours=dict(
                    showlabels=True,
                    labelfont=dict(size=12, color='white')
                )
            )
        ])
        
        fig2.update_layout(
            title="Contour Plot of Bivariate Normal Distribution",
            xaxis_title="X",
            yaxis_title="Y",
            height=500
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Mathematical formulation
        st.subheader("Mathematical Formulation")
        
        st.markdown(r"""
        The probability density function of a bivariate normal distribution is given by:
        
        $$f(x,y) = \frac{1}{2\pi\sigma_x\sigma_y\sqrt{1-\rho^2}} \exp\left(-\frac{1}{2(1-\rho^2)}\left[\frac{(x-\mu_x)^2}{\sigma_x^2} + \frac{(y-\mu_y)^2}{\sigma_y^2} - \frac{2\rho(x-\mu_x)(y-\mu_y)}{\sigma_x\sigma_y}\right]\right)$$
        
        where:
        - $\mu_x, \mu_y$ are the means
        - $\sigma_x, \sigma_y$ are the standard deviations
        - $\rho$ is the correlation coefficient
        """)
        
        # Generate sample data
        st.subheader("Sample Data")
        
        num_samples = st.slider("Number of samples", 50, 1000, 200)
        
        # Create covariance matrix
        cov = [[sigma_x**2, rho*sigma_x*sigma_y], 
               [rho*sigma_x*sigma_y, sigma_y**2]]
        
        # Generate random samples
        samples = np.random.multivariate_normal([mu_x, mu_y], cov, num_samples)
        
        # Create scatter plot
        fig3 = px.scatter(
            x=samples[:, 0], 
            y=samples[:, 1],
            color_discrete_sequence=['blue'],
            opacity=0.7
        )
        
        fig3.update_layout(
            title=f"{num_samples} Random Samples from Bivariate Normal Distribution",
            xaxis_title="X",
            yaxis_title="Y",
            height=500
        )
        
        # Add contour lines to the scatter plot
        fig3.add_trace(
            go.Contour(
                z=Z,
                x=x,
                y=y,
                colorscale='Viridis',
                opacity=0.3,
                showscale=False,
                contours=dict(
                    showlabels=False,
                    coloring='lines'
                )
            )
        )
        
        st.plotly_chart(fig3, use_container_width=True)
    
    with tab2:
        st.subheader("Correlation Explorer")
        
        st.markdown("""
        Correlation measures the strength and direction of the linear relationship between two variables.
        
        - **Correlation coefficient (ρ)** ranges from -1 to 1
        - ρ = 1: Perfect positive correlation
        - ρ = 0: No linear correlation
        - ρ = -1: Perfect negative correlation
        
        Explore how different correlation values affect the joint distribution:
        """)
        
        # Create a grid of correlation values
        correlation_values = [-0.95, -0.5, 0, 0.5, 0.95]
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=len(correlation_values),
            subplot_titles=[f"ρ = {rho}" for rho in correlation_values]
        )
        
        # Generate samples for each correlation value
        for i, rho in enumerate(correlation_values):
            # Create covariance matrix
            cov = [[1, rho], [rho, 1]]
            
            # Generate random samples
            samples = np.random.multivariate_normal([0, 0], cov, 200)
            
            # Add scatter plot
            fig.add_trace(
                go.Scatter(
                    x=samples[:, 0],
                    y=samples[:, 1],
                    mode='markers',
                    marker=dict(
                        color='blue',
                        opacity=0.5
                    ),
                    showlegend=False
                ),
                row=1, col=i+1
            )
            
            # Set axis ranges
            fig.update_xaxes(range=[-3, 3], row=1, col=i+1)
            fig.update_yaxes(range=[-3, 3], row=1, col=i+1)
        
        fig.update_layout(
            height=300,
            title="Effect of Correlation on Bivariate Normal Distribution"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interactive correlation demo
        st.subheader("Interactive Correlation Demo")
        
        # Parameters
        demo_rho = st.slider("Correlation coefficient (ρ)", -0.99, 0.99, 0.0, 0.01, key="demo_rho")
        
        # Generate data
        n_points = 200
        cov = [[1, demo_rho], [demo_rho, 1]]
        samples = np.random.multivariate_normal([0, 0], cov, n_points)
        
        # Calculate correlation statistics
        pearson_corr = np.corrcoef(samples[:, 0], samples[:, 1])[0, 1]
        spearman_corr = stats.spearmanr(samples[:, 0], samples[:, 1])[0]
        
        # Create DataFrame for scatter plot
        df = pd.DataFrame({
            'X': samples[:, 0],
            'Y': samples[:, 1]
        })
        
        # Create scatter plot with regression line
        fig = px.scatter(
            df, x='X', y='Y',
            trendline='ols',
            trendline_color_override='red'
        )
        
        fig.update_layout(
            title=f"Correlation Demonstration (ρ = {demo_rho:.2f})",
            xaxis_title="X",
            yaxis_title="Y",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display correlation statistics
        col1, col2 = st.columns(2)
        col1.metric("Pearson Correlation", f"{pearson_corr:.4f}")
        col2.metric("Spearman Rank Correlation", f"{spearman_corr:.4f}")
        
        # Explanation of correlation
        st.markdown("""
        ### Interpreting Correlation
        
        - **Pearson correlation** measures linear relationships
        - **Spearman correlation** measures monotonic relationships (can detect non-linear associations)
        
        #### Important Notes:
        
        1. Correlation does not imply causation
        2. Correlation only measures linear relationships
        3. Correlation is sensitive to outliers
        4. Zero correlation does not mean independence (except for multivariate normal)
        """)
        
        # Correlation vs Causation example
        st.subheader("Correlation vs. Causation")
        
        st.markdown("""
        Just because two variables are correlated doesn't mean one causes the other.
        
        Common scenarios:
        - **Direct causation**: X → Y
        - **Reverse causation**: Y → X
        - **Common cause**: Z → X and Z → Y
        - **Coincidental correlation**: No causal relationship
        
        Always investigate the underlying mechanism before inferring causation.
        """)
    
    with tab3:
        st.subheader("Conditional Distributions")
        
        st.markdown("""
        A conditional distribution shows the distribution of one variable given a specific value of another variable.
        
        For a bivariate normal distribution, the conditional distribution of Y given X=x is also normal with:
        - Mean: μ_Y|X = μ_Y + ρ(σ_Y/σ_X)(x - μ_X)
        - Variance: σ²_Y|X = σ²_Y(1 - ρ²)
        
        Explore how conditioning affects the distribution:
        """)
        
        # Parameters for bivariate normal
        cond_mu_x = st.slider("Mean X (μ₁)", -3.0, 3.0, 0.0, 0.1, key="cond_mu_x")
        cond_mu_y = st.slider("Mean Y (μ₂)", -3.0, 3.0, 0.0, 0.1, key="cond_mu_y")
        cond_sigma_x = st.slider("Std Dev X (σ₁)", 0.1, 3.0, 1.0, 0.1, key="cond_sigma_x")
        cond_sigma_y = st.slider("Std Dev Y (σ₂)", 0.1, 3.0, 1.0, 0.1, key="cond_sigma_y")
        cond_rho = st.slider("Correlation (ρ)", -0.99, 0.99, 0.7, 0.01, key="cond_rho")
        
        # Condition value
        x_condition = st.slider(
            "Condition: X = ", 
            float(cond_mu_x - 3*cond_sigma_x), 
            float(cond_mu_x + 3*cond_sigma_x), 
            float(cond_mu_x + cond_sigma_x)
        )
        
        # Calculate conditional mean and std dev
        cond_mean = cond_mu_y + cond_rho * (cond_sigma_y/cond_sigma_x) * (x_condition - cond_mu_x)
        cond_std = cond_sigma_y * np.sqrt(1 - cond_rho**2)
        
        # Create grid for bivariate normal
        x = np.linspace(cond_mu_x - 3*cond_sigma_x, cond_mu_x + 3*cond_sigma_x, 100)
        y = np.linspace(cond_mu_y - 3*cond_sigma_y, cond_mu_y + 3*cond_sigma_y, 100)
        X, Y = np.meshgrid(x, y)
        
        # Calculate PDF values
        Z = bivariate_normal_pdf(X, Y, cond_mu_x, cond_mu_y, cond_sigma_x, cond_sigma_y, cond_rho)
        
        # Create contour plot
        fig = go.Figure()
        
        # Add contour plot
        fig.add_trace(
            go.Contour(
                z=Z,
                x=x,
                y=y,
                colorscale='Viridis',
                contours=dict(showlabels=True)
            )
        )
        
        # Add vertical line for condition
        fig.add_shape(
            type="line",
            x0=x_condition,
            y0=min(y),
            x1=x_condition,
            y1=max(y),
            line=dict(color="red", width=2, dash="dash")
        )
        
        # Add conditional distribution
        y_cond = np.linspace(cond_mu_y - 3*cond_sigma_y, cond_mu_y + 3*cond_sigma_y, 100)
        pdf_cond = stats.norm.pdf(y_cond, cond_mean, cond_std)
        
        # Scale the conditional PDF for visualization
        max_pdf = np.max(pdf_cond)
        scale_factor = (max(x) - x_condition) / 3
        pdf_cond_scaled = pdf_cond / max_pdf * scale_factor
        
        fig.add_trace(
            go.Scatter(
                x=x_condition + pdf_cond_scaled,
                y=y_cond,
                mode='lines',
                line=dict(color='red', width=2),
                name=f'P(Y|X={x_condition:.2f})'
            )
        )
        
        fig.update_layout(
            title=f"Conditional Distribution: P(Y|X={x_condition:.2f})",
            xaxis_title="X",
            yaxis_title="Y",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display conditional distribution properties
        st.markdown(f"""
        ### Conditional Distribution Properties
        
        For X = {x_condition:.2f}, the conditional distribution of Y is normal with:
        
        - **Conditional Mean**: μ_Y|X = {cond_mean:.4f}
        - **Conditional Standard Deviation**: σ_Y|X = {cond_std:.4f}
        - **Conditional Variance**: σ²_Y|X = {cond_std**2:.4f}
        
        The conditional standard deviation is always smaller than or equal to the marginal standard deviation.
        """)
        
        # Mathematical formulation
        st.markdown(r"""
        ### Mathematical Formulation
        
        For a bivariate normal distribution, the conditional distribution of Y given X=x is:
        
        $$P(Y|X=x) \sim \mathcal{N}\left(\mu_Y + \rho\frac{\sigma_Y}{\sigma_X}(x - \mu_X), \sigma_Y^2(1-\rho^2)\right)$$
        
        This means that:
        - The conditional distribution is also normal
        - The conditional mean depends on the value of x
        - The conditional variance is reduced by a factor of (1-ρ²)
        - The stronger the correlation, the more the conditional variance is reduced
        """)

def bivariate_normal_pdf(x, y, mu_x, mu_y, sigma_x, sigma_y, rho):
    """Calculate PDF values for bivariate normal distribution"""
    z = ((x - mu_x) / sigma_x) ** 2 + ((y - mu_y) / sigma_y) ** 2 - \
        2 * rho * ((x - mu_x) / sigma_x) * ((y - mu_y) / sigma_y)
    
    return (1 / (2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho**2))) * \
           np.exp(-z / (2 * (1 - rho**2)))

if __name__ == "__main__":
    run_multivariate_distributions()
