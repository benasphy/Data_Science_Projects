import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy import stats

def run_distribution_explorer():
    st.title("📊 Probability Density Function Explorer")
    
    st.markdown("""
    ### Understanding Probability Density Functions (PDFs)
    
    A probability density function (PDF) describes the relative likelihood of a continuous random variable 
    taking on a given value. The probability of the random variable falling within a range is the integral 
    of the PDF over that range.
    
    Explore different probability distributions and their properties:
    """)
    
    # Distribution selection
    distribution = st.selectbox(
        "Select Distribution",
        ["Normal", "Exponential", "Gamma", "Beta", "Lognormal", "Weibull", "t-Distribution"]
    )
    
    # Parameters based on distribution
    if distribution == "Normal":
        col1, col2 = st.columns(2)
        with col1:
            mu = st.slider("Mean (μ)", -5.0, 5.0, 0.0, 0.1)
        with col2:
            sigma = st.slider("Standard Deviation (σ)", 0.1, 5.0, 1.0, 0.1)
        
        # Generate x values
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
        
        # Calculate PDF values
        pdf_values = stats.norm.pdf(x, loc=mu, scale=sigma)
        
        # Distribution properties
        properties = {
            "Mean": mu,
            "Median": mu,
            "Mode": mu,
            "Variance": sigma**2,
            "Skewness": 0,
            "Kurtosis": 3,
            "Support": "(-∞, ∞)"
        }
        
        formula = r"f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}"
        
    elif distribution == "Exponential":
        lambda_param = st.slider("Rate Parameter (λ)", 0.1, 5.0, 1.0, 0.1)
        
        # Generate x values
        x = np.linspace(0, 5/lambda_param, 1000)
        
        # Calculate PDF values
        pdf_values = stats.expon.pdf(x, scale=1/lambda_param)
        
        # Distribution properties
        properties = {
            "Mean": 1/lambda_param,
            "Median": np.log(2)/lambda_param,
            "Mode": 0,
            "Variance": 1/(lambda_param**2),
            "Skewness": 2,
            "Kurtosis": 9,
            "Support": "[0, ∞)"
        }
        
        formula = r"f(x) = \lambda e^{-\lambda x} \text{ for } x \geq 0"
        
    elif distribution == "Gamma":
        col1, col2 = st.columns(2)
        with col1:
            alpha = st.slider("Shape (α)", 0.1, 10.0, 2.0, 0.1)
        with col2:
            beta = st.slider("Rate (β)", 0.1, 10.0, 1.0, 0.1)
        
        # Generate x values
        x = np.linspace(0, max(20, alpha/beta*5), 1000)
        
        # Calculate PDF values
        pdf_values = stats.gamma.pdf(x, a=alpha, scale=1/beta)
        
        # Distribution properties
        properties = {
            "Mean": alpha/beta,
            "Median": "≈ α/β (no closed form)",
            "Mode": (alpha-1)/beta if alpha >= 1 else 0,
            "Variance": alpha/(beta**2),
            "Skewness": 2/np.sqrt(alpha),
            "Kurtosis": 6/alpha + 3,
            "Support": "[0, ∞)"
        }
        
        formula = r"f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x} \text{ for } x \geq 0"
        
    elif distribution == "Beta":
        col1, col2 = st.columns(2)
        with col1:
            alpha = st.slider("Shape α", 0.1, 10.0, 2.0, 0.1)
        with col2:
            beta = st.slider("Shape β", 0.1, 10.0, 2.0, 0.1)
        
        # Generate x values
        x = np.linspace(0, 1, 1000)
        
        # Calculate PDF values
        pdf_values = stats.beta.pdf(x, a=alpha, b=beta)
        
        # Distribution properties
        properties = {
            "Mean": alpha/(alpha+beta),
            "Median": "No closed form",
            "Mode": (alpha-1)/(alpha+beta-2) if alpha > 1 and beta > 1 else (0 if alpha < 1 else 1 if beta < 1 else "0 or 1"),
            "Variance": (alpha*beta)/((alpha+beta)**2 * (alpha+beta+1)),
            "Skewness": 2*(beta-alpha)*np.sqrt(alpha+beta+1)/((alpha+beta+2)*np.sqrt(alpha*beta)),
            "Support": "[0, 1]"
        }
        
        formula = r"f(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)} \text{ for } 0 \leq x \leq 1"
        
    elif distribution == "Lognormal":
        col1, col2 = st.columns(2)
        with col1:
            mu = st.slider("Log-mean (μ)", -2.0, 2.0, 0.0, 0.1)
        with col2:
            sigma = st.slider("Log-standard deviation (σ)", 0.1, 2.0, 0.5, 0.1)
        
        # Generate x values
        x = np.linspace(0.001, np.exp(mu + 4*sigma), 1000)
        
        # Calculate PDF values
        pdf_values = stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu))
        
        # Distribution properties
        properties = {
            "Mean": np.exp(mu + sigma**2/2),
            "Median": np.exp(mu),
            "Mode": np.exp(mu - sigma**2),
            "Variance": (np.exp(sigma**2) - 1) * np.exp(2*mu + sigma**2),
            "Skewness": (np.exp(sigma**2) + 2) * np.sqrt(np.exp(sigma**2) - 1),
            "Support": "(0, ∞)"
        }
        
        formula = r"f(x) = \frac{1}{x\sigma\sqrt{2\pi}} e^{-\frac{(\ln x-\mu)^2}{2\sigma^2}} \text{ for } x > 0"
        
    elif distribution == "Weibull":
        col1, col2 = st.columns(2)
        with col1:
            k = st.slider("Shape (k)", 0.1, 5.0, 1.5, 0.1)
        with col2:
            scale = st.slider("Scale (λ)", 0.1, 5.0, 1.0, 0.1)
        
        # Generate x values
        x = np.linspace(0, scale*5, 1000)
        
        # Calculate PDF values
        pdf_values = stats.weibull_min.pdf(x, c=k, scale=scale)
        
        # Distribution properties
        properties = {
            "Mean": scale * np.exp(np.log(1 + 1/k)),
            "Median": scale * np.log(2)**(1/k),
            "Mode": scale * ((k-1)/k)**(1/k) if k > 1 else 0,
            "Variance": scale**2 * (np.exp(np.log(1 + 2/k)) - np.exp(np.log(1 + 1/k))**2),
            "Support": "[0, ∞)"
        }
        
        formula = r"f(x) = \frac{k}{\lambda}\left(\frac{x}{\lambda}\right)^{k-1}e^{-(x/\lambda)^k} \text{ for } x \geq 0"
        
    elif distribution == "t-Distribution":
        df = st.slider("Degrees of Freedom (ν)", 1, 30, 5)
        
        # Generate x values
        x = np.linspace(-5, 5, 1000)
        
        # Calculate PDF values
        pdf_values = stats.t.pdf(x, df)
        
        # Distribution properties
        properties = {
            "Mean": 0 if df > 1 else "Undefined",
            "Median": 0,
            "Mode": 0,
            "Variance": df/(df-2) if df > 2 else "Undefined",
            "Skewness": 0 if df > 3 else "Undefined",
            "Kurtosis": 6/(df-4) + 3 if df > 4 else "Undefined",
            "Support": "(-∞, ∞)"
        }
        
        formula = r"f(x) = \frac{\Gamma(\frac{\nu+1}{2})}{\sqrt{\nu\pi}\,\Gamma(\frac{\nu}{2})}\left(1+\frac{x^2}{\nu}\right)^{-\frac{\nu+1}{2}}"
    
    # Create plot
    fig = go.Figure()
    
    # Add PDF trace
    fig.add_trace(go.Scatter(
        x=x,
        y=pdf_values,
        mode='lines',
        name='PDF',
        line=dict(color='blue', width=2)
    ))
    
    # Update layout
    fig.update_layout(
        title=f"{distribution} Distribution PDF",
        xaxis_title="x",
        yaxis_title="f(x)",
        height=500
    )
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Display formula
    st.subheader("PDF Formula")
    st.latex(formula)
    
    # Display properties
    st.subheader("Distribution Properties")
    
    # Create a DataFrame for the properties
    properties_df = {
        "Property": list(properties.keys()),
        "Value": list(properties.values())
    }
    
    # Display as a table
    col1, col2 = st.columns(2)
    for i, (prop, val) in enumerate(properties.items()):
        if i % 2 == 0:
            col1.metric(prop, val)
        else:
            col2.metric(prop, val)
    
    # Add interactive probability calculator
    st.subheader("Probability Calculator")
    
    calc_type = st.radio(
        "Calculate probability:",
        ["P(a < X < b)", "P(X < a)", "P(X > b)"]
    )
    
    if calc_type == "P(a < X < b)":
        col1, col2 = st.columns(2)
        with col1:
            a = st.slider("Lower bound (a):", float(min(x)), float(max(x)), float(min(x)))
        with col2:
            b = st.slider("Upper bound (b):", float(min(x)), float(max(x)), float(max(x)))
        
        if a >= b:
            st.error("Lower bound must be less than upper bound!")
        else:
            # Calculate probability based on distribution
            if distribution == "Normal":
                prob = stats.norm.cdf(b, loc=mu, scale=sigma) - stats.norm.cdf(a, loc=mu, scale=sigma)
            elif distribution == "Exponential":
                prob = stats.expon.cdf(b, scale=1/lambda_param) - stats.expon.cdf(a, scale=1/lambda_param)
            elif distribution == "Gamma":
                prob = stats.gamma.cdf(b, a=alpha, scale=1/beta) - stats.gamma.cdf(a, a=alpha, scale=1/beta)
            elif distribution == "Beta":
                prob = stats.beta.cdf(b, a=alpha, b=beta) - stats.beta.cdf(a, a=alpha, b=beta)
            elif distribution == "Lognormal":
                prob = stats.lognorm.cdf(b, s=sigma, scale=np.exp(mu)) - stats.lognorm.cdf(a, s=sigma, scale=np.exp(mu))
            elif distribution == "Weibull":
                prob = stats.weibull_min.cdf(b, c=k, scale=scale) - stats.weibull_min.cdf(a, c=k, scale=scale)
            elif distribution == "t-Distribution":
                prob = stats.t.cdf(b, df) - stats.t.cdf(a, df)
            
            # Visualization with shaded area
            fig2 = go.Figure()
            
            # Add PDF curve
            fig2.add_trace(go.Scatter(
                x=x,
                y=pdf_values,
                mode='lines',
                name='PDF',
                line=dict(color='blue', width=2)
            ))
            
            # Add shaded area for P(a < X < b)
            x_fill = [val for val in x if a <= val <= b]
            y_fill = [pdf_values[list(x).index(val)] if val in x else 0 for val in x_fill]
            
            fig2.add_trace(go.Scatter(
                x=x_fill,
                y=y_fill,
                fill='tozeroy',
                fillcolor='rgba(0, 0, 255, 0.2)',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig2.update_layout(
                title=f"P({a:.2f} < X < {b:.2f}) = {prob:.4f}",
                xaxis_title="x",
                yaxis_title="f(x)",
                height=400
            )
            
            st.plotly_chart(fig2, use_container_width=True)
    
    elif calc_type == "P(X < a)":
        a = st.slider("Upper bound (a):", float(min(x)), float(max(x)), float(min(x) + (max(x) - min(x))/2))
        
        # Calculate probability based on distribution
        if distribution == "Normal":
            prob = stats.norm.cdf(a, loc=mu, scale=sigma)
        elif distribution == "Exponential":
            prob = stats.expon.cdf(a, scale=1/lambda_param)
        elif distribution == "Gamma":
            prob = stats.gamma.cdf(a, a=alpha, scale=1/beta)
        elif distribution == "Beta":
            prob = stats.beta.cdf(a, a=alpha, b=beta)
        elif distribution == "Lognormal":
            prob = stats.lognorm.cdf(a, s=sigma, scale=np.exp(mu))
        elif distribution == "Weibull":
            prob = stats.weibull_min.cdf(a, c=k, scale=scale)
        elif distribution == "t-Distribution":
            prob = stats.t.cdf(a, df)
        
        # Visualization with shaded area
        fig2 = go.Figure()
        
        # Add PDF curve
        fig2.add_trace(go.Scatter(
            x=x,
            y=pdf_values,
            mode='lines',
            name='PDF',
            line=dict(color='blue', width=2)
        ))
        
        # Add shaded area for P(X < a)
        x_fill = [val for val in x if val <= a]
        y_fill = [pdf_values[list(x).index(val)] if val in x else 0 for val in x_fill]
        
        fig2.add_trace(go.Scatter(
            x=x_fill,
            y=y_fill,
            fill='tozeroy',
            fillcolor='rgba(0, 0, 255, 0.2)',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig2.update_layout(
            title=f"P(X < {a:.2f}) = {prob:.4f}",
            xaxis_title="x",
            yaxis_title="f(x)",
            height=400
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    else:  # P(X > b)
        b = st.slider("Lower bound (b):", float(min(x)), float(max(x)), float(min(x) + (max(x) - min(x))/2))
        
        # Calculate probability based on distribution
        if distribution == "Normal":
            prob = 1 - stats.norm.cdf(b, loc=mu, scale=sigma)
        elif distribution == "Exponential":
            prob = 1 - stats.expon.cdf(b, scale=1/lambda_param)
        elif distribution == "Gamma":
            prob = 1 - stats.gamma.cdf(b, a=alpha, scale=1/beta)
        elif distribution == "Beta":
            prob = 1 - stats.beta.cdf(b, a=alpha, b=beta)
        elif distribution == "Lognormal":
            prob = 1 - stats.lognorm.cdf(b, s=sigma, scale=np.exp(mu))
        elif distribution == "Weibull":
            prob = 1 - stats.weibull_min.cdf(b, c=k, scale=scale)
        elif distribution == "t-Distribution":
            prob = 1 - stats.t.cdf(b, df)
        
        # Visualization with shaded area
        fig2 = go.Figure()
        
        # Add PDF curve
        fig2.add_trace(go.Scatter(
            x=x,
            y=pdf_values,
            mode='lines',
            name='PDF',
            line=dict(color='blue', width=2)
        ))
        
        # Add shaded area for P(X > b)
        x_fill = [val for val in x if val >= b]
        y_fill = [pdf_values[list(x).index(val)] if val in x else 0 for val in x_fill]
        
        fig2.add_trace(go.Scatter(
            x=x_fill,
            y=y_fill,
            fill='tozeroy',
            fillcolor='rgba(0, 0, 255, 0.2)',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig2.update_layout(
            title=f"P(X > {b:.2f}) = {prob:.4f}",
            xaxis_title="x",
            yaxis_title="f(x)",
            height=400
        )
        
        st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    run_distribution_explorer()
