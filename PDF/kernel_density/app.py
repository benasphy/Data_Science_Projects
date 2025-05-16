import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
import plotly.express as px

def run_kernel_density():
    st.title("🔍 Kernel Density Estimation Explorer")
    
    st.markdown("""
    ### Understanding Kernel Density Estimation (KDE)
    
    Kernel Density Estimation is a non-parametric method to estimate the probability density function 
    of a random variable. KDE works by placing a kernel (a smooth, symmetric function like a Gaussian) 
    on each data point and then summing these kernels to create a smooth curve.
    
    KDE is useful when:
    - You want to visualize the distribution of data without assuming a specific parametric form
    - You need to identify modes (peaks) in the data distribution
    - You want to estimate the PDF from a limited sample
    
    Explore how different kernels and bandwidths affect the density estimation:
    """)
    
    # Create tabs for different modes
    tab1, tab2 = st.tabs(["Interactive Demo", "Upload Your Data"])
    
    with tab1:
        st.subheader("Interactive KDE Demo")
        
        # Distribution selection for generating sample data
        dist_type = st.selectbox(
            "Select distribution for sample data",
            ["Normal", "Bimodal Normal", "Skewed (Gamma)", "Uniform", "Custom Points"]
        )
        
        # Sample size
        sample_size = st.slider("Sample size", 10, 1000, 200)
        
        # Generate sample data based on selected distribution
        if dist_type == "Normal":
            mean = st.slider("Mean", -5.0, 5.0, 0.0, 0.1)
            std = st.slider("Standard Deviation", 0.1, 5.0, 1.0, 0.1)
            data = np.random.normal(mean, std, sample_size)
            true_pdf = lambda x: stats.norm.pdf(x, mean, std)
            x_range = (mean - 4*std, mean + 4*std)
            
        elif dist_type == "Bimodal Normal":
            mean1 = st.slider("Mean of First Mode", -10.0, 0.0, -2.0, 0.1)
            mean2 = st.slider("Mean of Second Mode", 0.0, 10.0, 2.0, 0.1)
            std = st.slider("Standard Deviation of Both Modes", 0.1, 2.0, 0.5, 0.1)
            mix = st.slider("Mixing Proportion", 0.0, 1.0, 0.5, 0.01)
            
            # Generate mixture of two normals
            data1 = np.random.normal(mean1, std, int(sample_size * mix))
            data2 = np.random.normal(mean2, std, sample_size - int(sample_size * mix))
            data = np.concatenate([data1, data2])
            
            # True PDF for bimodal normal
            true_pdf = lambda x: mix * stats.norm.pdf(x, mean1, std) + (1-mix) * stats.norm.pdf(x, mean2, std)
            x_range = (min(mean1, mean2) - 4*std, max(mean1, mean2) + 4*std)
            
        elif dist_type == "Skewed (Gamma)":
            shape = st.slider("Shape Parameter (α)", 0.5, 10.0, 2.0, 0.1)
            scale = st.slider("Scale Parameter (θ)", 0.1, 5.0, 1.0, 0.1)
            data = np.random.gamma(shape, scale, sample_size)
            true_pdf = lambda x: stats.gamma.pdf(x, shape, scale=scale)
            x_range = (0, shape*scale*4)
            
        elif dist_type == "Uniform":
            low = st.slider("Lower Bound", -10.0, 0.0, -5.0, 0.1)
            high = st.slider("Upper Bound", 0.0, 10.0, 5.0, 0.1)
            data = np.random.uniform(low, high, sample_size)
            true_pdf = lambda x: stats.uniform.pdf(x, low, high-low)
            x_range = (low - 1, high + 1)
            
        else:  # Custom Points
            st.markdown("""
            Click on the chart below to add data points. Double-click to clear all points.
            """)
            
            # Create an empty chart for adding points
            if 'custom_data' not in st.session_state:
                st.session_state.custom_data = []
            
            # Create a scatter plot for adding points
            fig = px.scatter(
                x=[0] if not st.session_state.custom_data else st.session_state.custom_data,
                y=[0] if not st.session_state.custom_data else [0.1] * len(st.session_state.custom_data),
                title="Click to Add Points",
                height=300
            )
            
            fig.update_layout(
                xaxis=dict(range=[-10, 10]),
                yaxis=dict(range=[0, 0.2], showticklabels=False),
                clickmode='event+select'
            )
            
            # Display the plot and capture clicks
            scatter_plot = st.plotly_chart(fig, use_container_width=True)
            
            # Use custom data if available, otherwise use a default normal distribution
            if st.session_state.custom_data:
                data = np.array(st.session_state.custom_data)
                true_pdf = None  # No true PDF for custom points
                x_range = (min(data) - 2, max(data) + 2)
            else:
                data = np.random.normal(0, 1, sample_size)
                true_pdf = lambda x: stats.norm.pdf(x, 0, 1)
                x_range = (-4, 4)
        
        # KDE parameters
        st.subheader("KDE Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            kernel = st.selectbox(
                "Kernel Function",
                ["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"]
            )
        with col2:
            bandwidth_method = st.selectbox(
                "Bandwidth Selection Method",
                ["scott", "silverman", "manual"]
            )
        
        if bandwidth_method == "manual":
            bandwidth = st.slider("Bandwidth", 0.01, 2.0, 0.5, 0.01)
        else:
            # Use the selected method
            bandwidth = bandwidth_method
        
        # Create KDE
        kde = stats.gaussian_kde(data, bw_method=bandwidth)
        
        # Generate points for plotting
        x_plot = np.linspace(x_range[0], x_range[1], 1000)
        kde_values = kde(x_plot)
        
        # Create the plot
        fig = go.Figure()
        
        # Add histogram of data
        fig.add_trace(go.Histogram(
            x=data,
            histnorm='probability density',
            name='Histogram',
            opacity=0.5,
            nbinsx=30
        ))
        
        # Add KDE curve
        fig.add_trace(go.Scatter(
            x=x_plot,
            y=kde_values,
            mode='lines',
            name='KDE',
            line=dict(color='red', width=2)
        ))
        
        # Add true PDF if available
        if true_pdf is not None:
            fig.add_trace(go.Scatter(
                x=x_plot,
                y=[true_pdf(x) for x in x_plot],
                mode='lines',
                name='True PDF',
                line=dict(color='green', width=2, dash='dash')
            ))
        
        # Update layout
        fig.update_layout(
            title="Kernel Density Estimation",
            xaxis_title="Value",
            yaxis_title="Density",
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Explanation of the current settings
        st.subheader("Explanation")
        
        st.markdown(f"""
        ### Current KDE Settings:
        
        - **Kernel**: {kernel.capitalize()} - {get_kernel_description(kernel)}
        - **Bandwidth**: {bandwidth if isinstance(bandwidth, float) else f"{bandwidth.capitalize()} method"} - {get_bandwidth_description(bandwidth_method)}
        - **Sample Size**: {sample_size} data points
        
        ### Effect of Parameters:
        
        - **Smaller bandwidth** → More detail, potentially overfitting (high variance)
        - **Larger bandwidth** → Smoother curve, potentially underfitting (high bias)
        - **Different kernels** → Usually have minor effects compared to bandwidth selection
        
        The optimal bandwidth balances bias and variance to best represent the underlying distribution.
        """)
        
        # Add information about kernel density estimation
        st.subheader("Mathematical Formulation")
        
        st.markdown(r"""
        The kernel density estimator is given by:
        
        $$\hat{f}_h(x) = \frac{1}{nh} \sum_{i=1}^{n} K\left(\frac{x - x_i}{h}\right)$$
        
        where:
        - $K$ is the kernel function
        - $h$ is the bandwidth (smoothing parameter)
        - $x_1, x_2, \ldots, x_n$ are the data samples
        - $n$ is the sample size
        
        The choice of bandwidth $h$ is critical for the performance of the estimator. Too small a bandwidth 
        leads to an undersmoothed estimate with high variance, while too large a bandwidth leads to an 
        oversmoothed estimate with high bias.
        """)
    
    with tab2:
        st.subheader("Upload Your Data")
        
        st.markdown("""
        Upload a CSV or Excel file to perform kernel density estimation on your own data.
        
        The file should contain at least one column of numerical data.
        """)
        
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])
        
        if uploaded_file is not None:
            try:
                # Determine file type and read accordingly
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Display the first few rows of the data
                st.write("Data Preview:")
                st.dataframe(df.head())
                
                # Select column for KDE
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if not numeric_columns:
                    st.error("No numeric columns found in the uploaded file.")
                else:
                    selected_column = st.selectbox("Select column for KDE", numeric_columns)
                    
                    # Remove missing values
                    data = df[selected_column].dropna().values
                    
                    if len(data) < 2:
                        st.error("Not enough valid data points for KDE.")
                    else:
                        # KDE parameters
                        col1, col2 = st.columns(2)
                        with col1:
                            kernel = st.selectbox(
                                "Kernel Function",
                                ["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"],
                                key="upload_kernel"
                            )
                        with col2:
                            bandwidth_method = st.selectbox(
                                "Bandwidth Selection Method",
                                ["scott", "silverman", "manual"],
                                key="upload_bw_method"
                            )
                        
                        if bandwidth_method == "manual":
                            # Calculate a reasonable range for bandwidth slider
                            data_range = np.max(data) - np.min(data)
                            max_bw = data_range / 2
                            init_bw = data_range / 10
                            
                            bandwidth = st.slider(
                                "Bandwidth", 
                                min_value=data_range/100, 
                                max_value=max_bw, 
                                value=init_bw,
                                key="upload_bw"
                            )
                        else:
                            # Use the selected method
                            bandwidth = bandwidth_method
                        
                        # Create KDE
                        kde = stats.gaussian_kde(data, bw_method=bandwidth)
                        
                        # Generate points for plotting
                        x_min, x_max = np.min(data), np.max(data)
                        padding = (x_max - x_min) * 0.2
                        x_plot = np.linspace(x_min - padding, x_max + padding, 1000)
                        kde_values = kde(x_plot)
                        
                        # Create the plot
                        fig = go.Figure()
                        
                        # Add histogram of data
                        fig.add_trace(go.Histogram(
                            x=data,
                            histnorm='probability density',
                            name='Histogram',
                            opacity=0.5,
                            nbinsx=30
                        ))
                        
                        # Add KDE curve
                        fig.add_trace(go.Scatter(
                            x=x_plot,
                            y=kde_values,
                            mode='lines',
                            name='KDE',
                            line=dict(color='red', width=2)
                        ))
                        
                        # Update layout
                        fig.update_layout(
                            title=f"Kernel Density Estimation for {selected_column}",
                            xaxis_title=selected_column,
                            yaxis_title="Density",
                            height=500,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        
                        # Display the plot
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display summary statistics
                        st.subheader("Summary Statistics")
                        
                        # Calculate statistics
                        stats_df = pd.DataFrame({
                            "Statistic": ["Count", "Mean", "Median", "Std Dev", "Min", "Max", "Skewness", "Kurtosis"],
                            "Value": [
                                len(data),
                                np.mean(data),
                                np.median(data),
                                np.std(data),
                                np.min(data),
                                np.max(data),
                                stats.skew(data),
                                stats.kurtosis(data)
                            ]
                        })
                        
                        st.table(stats_df)
                        
                        # Add information about the KDE parameters
                        st.markdown(f"""
                        ### KDE Parameters:
                        
                        - **Kernel**: {kernel.capitalize()}
                        - **Bandwidth**: {bandwidth if isinstance(bandwidth, float) else f"{bandwidth.capitalize()} method"}
                        - **Sample Size**: {len(data)} data points
                        
                        The bandwidth value determines the smoothness of the KDE curve. 
                        Adjust it to find the right balance between capturing the true structure 
                        of your data and avoiding noise.
                        """)
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    # Add educational content
    st.subheader("Applications of Kernel Density Estimation")
    
    st.markdown("""
    ### Real-world Applications:
    
    1. **Data Exploration**: Visualizing the distribution of data without assuming a parametric form
    
    2. **Anomaly Detection**: Identifying data points in low-density regions
    
    3. **Multimodal Distribution Analysis**: Detecting and characterizing multiple peaks in data
    
    4. **Finance**: Estimating the distribution of returns for risk management
    
    5. **Machine Learning**: Used in non-parametric classification methods and clustering
    
    6. **Image Processing**: Edge detection and feature extraction
    
    7. **Ecology**: Estimating animal home ranges and habitat use
    
    ### Advantages and Limitations:
    
    **Advantages:**
    - No assumption about the underlying distribution
    - Captures multimodality and skewness
    - Provides a smooth estimate of the PDF
    
    **Limitations:**
    - Bandwidth selection is critical and can be challenging
    - Less efficient than parametric methods when the true distribution is known
    - Suffers from the "curse of dimensionality" in high-dimensional spaces
    - Boundary bias issues near the edges of the data range
    """)

def get_kernel_description(kernel):
    descriptions = {
        "gaussian": "The most common kernel, using a normal distribution shape. Provides a smooth density estimate.",
        "tophat": "A uniform kernel (rectangular shape). Simple but can create discontinuities in the density estimate.",
        "epanechnikov": "A parabolic kernel that is optimal in a mean squared error sense. Good balance of efficiency and smoothness.",
        "exponential": "An exponential decay kernel. Gives more weight to points near the estimation point.",
        "linear": "A triangular kernel that decreases linearly from the center. Provides a moderately smooth estimate.",
        "cosine": "A cosine-based kernel that provides a smooth estimate similar to the Epanechnikov kernel."
    }
    return descriptions.get(kernel, "")

def get_bandwidth_description(method):
    descriptions = {
        "scott": "Scott's rule of thumb (h ∝ n^(-1/5)), which is optimal for normally distributed data.",
        "silverman": "Silverman's rule of thumb, which is slightly more robust to non-normal data.",
        "manual": "Manually specified bandwidth, allowing fine control over the smoothness of the estimate."
    }
    return descriptions.get(method, "")

if __name__ == "__main__":
    run_kernel_density()
