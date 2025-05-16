import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy import stats

def normal_cdf(x, mu, sigma):
    """Calculate the CDF of a normal distribution at point x"""
    return stats.norm.cdf(x, loc=mu, scale=sigma)

def normal_pdf(x, mu, sigma):
    """Calculate the PDF of a normal distribution at point x"""
    return stats.norm.pdf(x, loc=mu, scale=sigma)

def run_normal_cdf():
    st.title("📈 Normal Distribution CDF Explorer")
    
    st.markdown("""
    ### Understanding the Cumulative Distribution Function (CDF)
    
    The CDF of a normal distribution gives the probability that a random variable X is less than or equal to x.
    
    Mathematically, for a normal distribution with mean μ and standard deviation σ:
    
    F(x) = P(X ≤ x) = ∫<sub>-∞</sub><sup>x</sup> (1/(σ√2π)) * e<sup>-(t-μ)²/(2σ²)</sup> dt
    
    This integral doesn't have a closed-form solution but can be expressed in terms of the error function.
    
    Explore how changing the parameters affects the CDF:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        mu = st.slider("Mean (μ)", -5.0, 5.0, 0.0, 0.1)
    with col2:
        sigma = st.slider("Standard Deviation (σ)", 0.1, 5.0, 1.0, 0.1)
    
    # Generate x values
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
    
    # Calculate CDF and PDF values
    cdf_values = [normal_cdf(val, mu, sigma) for val in x]
    pdf_values = [normal_pdf(val, mu, sigma) for val in x]
    
    # Create figure with two subplots
    fig = go.Figure()
    
    # Add CDF trace
    fig.add_trace(go.Scatter(
        x=x,
        y=cdf_values,
        mode='lines',
        name='CDF',
        line=dict(color='blue', width=2)
    ))
    
    # Update layout
    fig.update_layout(
        title="Normal Distribution CDF",
        xaxis_title="x",
        yaxis_title="F(x) = P(X ≤ x)",
        yaxis=dict(range=[0, 1.05]),
        height=500
    )
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Add interactive probability calculator
    st.subheader("Probability Calculator")
    
    calc_type = st.radio(
        "Calculate probability:",
        ["P(X ≤ a)", "P(X > a)", "P(a < X ≤ b)"]
    )
    
    if calc_type == "P(X ≤ a)":
        a = st.slider("Value a:", float(min(x)), float(max(x)), mu)
        prob = normal_cdf(a, mu, sigma)
        
        # Visualization with shaded area
        fig2 = go.Figure()
        
        # Add CDF curve
        fig2.add_trace(go.Scatter(
            x=x,
            y=cdf_values,
            mode='lines',
            name='CDF',
            line=dict(color='blue', width=2)
        ))
        
        # Add vertical line at a
        fig2.add_shape(
            type="line",
            x0=a, y0=0,
            x1=a, y1=normal_cdf(a, mu, sigma),
            line=dict(color="red", width=2, dash="dash")
        )
        
        # Add horizontal line from y-axis to (a, F(a))
        fig2.add_shape(
            type="line",
            x0=min(x), y0=normal_cdf(a, mu, sigma),
            x1=a, y1=normal_cdf(a, mu, sigma),
            line=dict(color="red", width=2, dash="dash")
        )
        
        # Add shaded area for P(X ≤ a)
        x_fill = [val for val in x if val <= a]
        y_fill = [normal_cdf(val, mu, sigma) for val in x_fill]
        
        fig2.add_trace(go.Scatter(
            x=x_fill,
            y=y_fill,
            fill='tozeroy',
            fillcolor='rgba(0, 0, 255, 0.2)',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig2.update_layout(
            title=f"P(X ≤ {a:.2f}) = {prob:.4f}",
            xaxis_title="x",
            yaxis_title="F(x)",
            yaxis=dict(range=[0, 1.05]),
            height=400
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
    elif calc_type == "P(X > a)":
        a = st.slider("Value a:", float(min(x)), float(max(x)), mu)
        prob = 1 - normal_cdf(a, mu, sigma)
        
        # Visualization with shaded area
        fig2 = go.Figure()
        
        # Add CDF curve
        fig2.add_trace(go.Scatter(
            x=x,
            y=cdf_values,
            mode='lines',
            name='CDF',
            line=dict(color='blue', width=2)
        ))
        
        # Add vertical line at a
        fig2.add_shape(
            type="line",
            x0=a, y0=0,
            x1=a, y1=normal_cdf(a, mu, sigma),
            line=dict(color="red", width=2, dash="dash")
        )
        
        # Add horizontal line from (a, F(a)) to (max(x), F(a))
        fig2.add_shape(
            type="line",
            x0=a, y0=normal_cdf(a, mu, sigma),
            x1=max(x), y1=normal_cdf(a, mu, sigma),
            line=dict(color="red", width=2, dash="dash")
        )
        
        # Add shaded area for P(X > a)
        x_fill = [val for val in x if val >= a]
        y_fill_bottom = [normal_cdf(a, mu, sigma)] * len(x_fill)
        y_fill_top = [normal_cdf(val, mu, sigma) for val in x_fill]
        
        fig2.add_trace(go.Scatter(
            x=x_fill,
            y=y_fill_top,
            fill='tonexty',
            fillcolor='rgba(0, 0, 255, 0.2)',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig2.add_trace(go.Scatter(
            x=x_fill,
            y=y_fill_bottom,
            line=dict(width=0),
            showlegend=False
        ))
        
        fig2.update_layout(
            title=f"P(X > {a:.2f}) = {prob:.4f}",
            xaxis_title="x",
            yaxis_title="F(x)",
            yaxis=dict(range=[0, 1.05]),
            height=400
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
    else:  # P(a < X ≤ b)
        col1, col2 = st.columns(2)
        with col1:
            a = st.slider("Lower bound (a):", float(min(x)), float(max(x)), mu - sigma)
        with col2:
            b = st.slider("Upper bound (b):", float(min(x)), float(max(x)), mu + sigma, key="b_slider")
        
        if a >= b:
            st.error("Lower bound must be less than upper bound!")
        else:
            prob = normal_cdf(b, mu, sigma) - normal_cdf(a, mu, sigma)
            
            # Visualization with shaded area
            fig2 = go.Figure()
            
            # Add CDF curve
            fig2.add_trace(go.Scatter(
                x=x,
                y=cdf_values,
                mode='lines',
                name='CDF',
                line=dict(color='blue', width=2)
            ))
            
            # Add vertical lines at a and b
            fig2.add_shape(
                type="line",
                x0=a, y0=0,
                x1=a, y1=normal_cdf(a, mu, sigma),
                line=dict(color="red", width=2, dash="dash")
            )
            
            fig2.add_shape(
                type="line",
                x0=b, y0=0,
                x1=b, y1=normal_cdf(b, mu, sigma),
                line=dict(color="red", width=2, dash="dash")
            )
            
            # Add horizontal lines
            fig2.add_shape(
                type="line",
                x0=a, y0=normal_cdf(a, mu, sigma),
                x1=b, y1=normal_cdf(a, mu, sigma),
                line=dict(color="red", width=2, dash="dash")
            )
            
            fig2.add_shape(
                type="line",
                x0=b, y0=normal_cdf(b, mu, sigma),
                x1=b, y1=normal_cdf(a, mu, sigma),
                line=dict(color="red", width=2, dash="dash")
            )
            
            # Add shaded area for P(a < X ≤ b)
            x_fill = [val for val in x if a <= val <= b]
            y_fill_bottom = [normal_cdf(a, mu, sigma)] * len(x_fill)
            y_fill_top = [normal_cdf(val, mu, sigma) for val in x_fill]
            
            fig2.add_trace(go.Scatter(
                x=x_fill,
                y=y_fill_top,
                fill='tonexty',
                fillcolor='rgba(0, 0, 255, 0.2)',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig2.add_trace(go.Scatter(
                x=x_fill,
                y=y_fill_bottom,
                line=dict(width=0),
                showlegend=False
            ))
            
            fig2.update_layout(
                title=f"P({a:.2f} < X ≤ {b:.2f}) = {prob:.4f}",
                xaxis_title="x",
                yaxis_title="F(x)",
                yaxis=dict(range=[0, 1.05]),
                height=400
            )
            
            st.plotly_chart(fig2, use_container_width=True)
    
    # Add information about quantiles/percentiles
    st.subheader("Quantiles (Inverse CDF)")
    
    st.markdown("""
    The inverse of the CDF gives us quantiles - values below which a certain proportion of the data falls.
    
    For example, the 0.5 quantile (50th percentile) is the median.
    """)
    
    p = st.slider("Probability (p):", 0.01, 0.99, 0.5, 0.01)
    quantile = stats.norm.ppf(p, loc=mu, scale=sigma)
    
    st.markdown(f"""
    For p = {p:.2f}:
    - The {p*100:.0f}th percentile is x = {quantile:.4f}
    - This means P(X ≤ {quantile:.4f}) = {p:.2f}
    """)
    
    # Visualization of the quantile
    fig3 = go.Figure()
    
    # Add CDF curve
    fig3.add_trace(go.Scatter(
        x=x,
        y=cdf_values,
        mode='lines',
        name='CDF',
        line=dict(color='blue', width=2)
    ))
    
    # Add horizontal line at p
    fig3.add_shape(
        type="line",
        x0=min(x), y0=p,
        x1=quantile, y1=p,
        line=dict(color="red", width=2, dash="dash")
    )
    
    # Add vertical line at quantile
    fig3.add_shape(
        type="line",
        x0=quantile, y0=0,
        x1=quantile, y1=p,
        line=dict(color="red", width=2, dash="dash")
    )
    
    # Add shaded area
    x_fill = [val for val in x if val <= quantile]
    y_fill = [normal_cdf(val, mu, sigma) for val in x_fill]
    
    fig3.add_trace(go.Scatter(
        x=x_fill,
        y=y_fill,
        fill='tozeroy',
        fillcolor='rgba(0, 0, 255, 0.2)',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig3.update_layout(
        title=f"Quantile for p = {p:.2f}",
        xaxis_title="x",
        yaxis_title="F(x)",
        yaxis=dict(range=[0, 1.05]),
        height=400
    )
    
    st.plotly_chart(fig3, use_container_width=True)
    
    # Add educational content
    st.subheader("Properties of the Normal CDF")
    
    st.markdown("""
    ### Key Properties of the Normal CDF:
    
    1. **Range**: The CDF ranges from 0 to 1
    2. **Monotonicity**: The CDF is always non-decreasing
    3. **Limits**: As x approaches -∞, F(x) approaches 0; as x approaches ∞, F(x) approaches 1
    4. **Symmetry**: For the standard normal distribution (μ=0, σ=1), the CDF has the property F(-x) = 1 - F(x)
    5. **Inflection Point**: The CDF has an inflection point at x = μ, where the slope is maximum
    
    ### Applications:
    
    - **Risk Assessment**: Calculating the probability of a value falling below a critical threshold
    - **Quality Control**: Determining specification limits for manufacturing processes
    - **Finance**: Computing Value at Risk (VaR) for investment portfolios
    - **Statistics**: Performing hypothesis tests and constructing confidence intervals
    """)

if __name__ == "__main__":
    run_normal_cdf()
