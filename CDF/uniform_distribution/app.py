import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy import stats

def uniform_cdf(x, a, b):
    """Calculate the CDF of a uniform distribution at point x"""
    if x < a:
        return 0
    elif x > b:
        return 1
    else:
        return (x - a) / (b - a)

def uniform_pdf(x, a, b):
    """Calculate the PDF of a uniform distribution at point x"""
    if a <= x <= b:
        return 1 / (b - a)
    else:
        return 0

def run_uniform_cdf():
    st.title("📊 Uniform Distribution CDF Explorer")
    
    st.markdown("""
    ### Understanding the Uniform Distribution and its CDF
    
    The uniform distribution is the simplest continuous probability distribution, where all intervals 
    of equal length within the distribution's support have equal probability.
    
    For a random variable X uniformly distributed between a and b:
    - PDF: f(x) = 1/(b-a) for a ≤ x ≤ b, and 0 elsewhere
    - CDF: F(x) = 0 for x < a, (x-a)/(b-a) for a ≤ x ≤ b, and 1 for x > b
    
    The uniform distribution is often used to model random selection from a continuous interval, 
    such as random number generation or modeling uncertainty when all values in a range are equally likely.
    """)
    
    # Parameter selection
    col1, col2 = st.columns(2)
    with col1:
        a = st.slider("Lower bound (a)", -10.0, 10.0, 0.0, 0.1)
    with col2:
        b = st.slider("Upper bound (b)", -10.0, 10.0, 1.0, 0.1)
    
    if a >= b:
        st.error("Lower bound must be less than upper bound!")
        return
    
    # Calculate mean and variance
    mean = (a + b) / 2
    variance = (b - a) ** 2 / 12
    
    st.markdown(f"""
    **Distribution Properties:**
    - Mean (Expected Value): (a + b)/2 = {mean:.2f}
    - Variance: (b - a)²/12 = {variance:.2f}
    - Standard Deviation: (b - a)/√12 = {np.sqrt(variance):.2f}
    - Median: (a + b)/2 = {mean:.2f}
    - Range: b - a = {b - a:.2f}
    """)
    
    # Generate x values with padding
    padding = (b - a) * 0.3
    x = np.linspace(a - padding, b + padding, 1000)
    
    # Calculate CDF and PDF values
    cdf_values = [uniform_cdf(val, a, b) for val in x]
    pdf_values = [uniform_pdf(val, a, b) for val in x]
    
    # Create figure with two y-axes
    fig = go.Figure()
    
    # Add CDF trace
    fig.add_trace(go.Scatter(
        x=x,
        y=cdf_values,
        mode='lines',
        name='CDF',
        line=dict(color='blue', width=2)
    ))
    
    # Add PDF trace on secondary y-axis
    fig.add_trace(go.Scatter(
        x=x,
        y=pdf_values,
        mode='lines',
        name='PDF',
        line=dict(color='red', width=2),
        yaxis="y2"
    ))
    
    # Update layout with secondary y-axis
    fig.update_layout(
        title="Uniform Distribution: CDF and PDF",
        xaxis_title="x",
        yaxis=dict(
            title="F(x) = P(X ≤ x)",
            range=[0, 1.05],
            titlefont=dict(color="blue"),
            tickfont=dict(color="blue")
        ),
        yaxis2=dict(
            title="f(x)",
            titlefont=dict(color="red"),
            tickfont=dict(color="red"),
            anchor="x",
            overlaying="y",
            side="right",
            range=[0, max(pdf_values) * 1.1]
        ),
        height=500,
        legend=dict(x=0.01, y=0.99)
    )
    
    # Add vertical lines at a and b
    fig.add_shape(
        type="line",
        x0=a, y0=0,
        x1=a, y1=uniform_cdf(a, a, b),
        line=dict(color="green", width=2, dash="dash")
    )
    
    fig.add_shape(
        type="line",
        x0=b, y0=0,
        x1=b, y1=uniform_cdf(b, a, b),
        line=dict(color="green", width=2, dash="dash")
    )
    
    # Add annotations for a and b
    fig.add_annotation(
        x=a,
        y=0.05,
        text="a",
        showarrow=False,
        yshift=10
    )
    
    fig.add_annotation(
        x=b,
        y=0.05,
        text="b",
        showarrow=False,
        yshift=10
    )
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Add interactive probability calculator
    st.subheader("Probability Calculator")
    
    calc_type = st.radio(
        "Calculate probability:",
        ["P(X ≤ c)", "P(X > c)", "P(c₁ < X ≤ c₂)"]
    )
    
    if calc_type == "P(X ≤ c)":
        c = st.slider("Value c:", float(a - padding), float(b + padding), (a + b) / 2)
        prob = uniform_cdf(c, a, b)
        
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
        
        # Add vertical line at c
        fig2.add_shape(
            type="line",
            x0=c, y0=0,
            x1=c, y1=prob,
            line=dict(color="red", width=2, dash="dash")
        )
        
        # Add horizontal line to y-axis
        fig2.add_shape(
            type="line",
            x0=a - padding, y0=prob,
            x1=c, y1=prob,
            line=dict(color="red", width=2, dash="dash")
        )
        
        # Add shaded area for P(X ≤ c)
        x_fill = [val for val in x if val <= c]
        y_fill = [uniform_cdf(val, a, b) for val in x_fill]
        
        fig2.add_trace(go.Scatter(
            x=x_fill,
            y=y_fill,
            fill='tozeroy',
            fillcolor='rgba(0, 0, 255, 0.2)',
            line=dict(width=0),
            showlegend=False
        ))
        
        # Add vertical lines at a and b
        fig2.add_shape(
            type="line",
            x0=a, y0=0,
            x1=a, y1=uniform_cdf(a, a, b),
            line=dict(color="green", width=2, dash="dash")
        )
        
        fig2.add_shape(
            type="line",
            x0=b, y0=0,
            x1=b, y1=uniform_cdf(b, a, b),
            line=dict(color="green", width=2, dash="dash")
        )
        
        fig2.update_layout(
            title=f"P(X ≤ {c:.2f}) = {prob:.4f}",
            xaxis_title="x",
            yaxis_title="F(x)",
            yaxis=dict(range=[0, 1.05]),
            height=400
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Calculate the probability analytically
        if c <= a:
            prob_formula = "0"
        elif c >= b:
            prob_formula = "1"
        else:
            prob_formula = f"(c - a)/(b - a) = ({c:.2f} - {a:.2f})/({b:.2f} - {a:.2f}) = {(c - a)/(b - a):.4f}"
        
        st.markdown(f"""
        **Analytical Calculation:**
        
        P(X ≤ {c:.2f}) = {prob_formula}
        
        This represents the probability that X is less than or equal to {c:.2f}.
        """)
        
    elif calc_type == "P(X > c)":
        c = st.slider("Value c:", float(a - padding), float(b + padding), (a + b) / 2)
        prob = 1 - uniform_cdf(c, a, b)
        
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
        
        # Add vertical line at c
        fig2.add_shape(
            type="line",
            x0=c, y0=0,
            x1=c, y1=uniform_cdf(c, a, b),
            line=dict(color="red", width=2, dash="dash")
        )
        
        # Add shaded area for P(X > c)
        x_fill = [val for val in x if val >= c]
        y_fill_bottom = [uniform_cdf(c, a, b)] * len(x_fill)
        y_fill_top = [uniform_cdf(val, a, b) for val in x_fill]
        
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
        
        # Add vertical lines at a and b
        fig2.add_shape(
            type="line",
            x0=a, y0=0,
            x1=a, y1=uniform_cdf(a, a, b),
            line=dict(color="green", width=2, dash="dash")
        )
        
        fig2.add_shape(
            type="line",
            x0=b, y0=0,
            x1=b, y1=uniform_cdf(b, a, b),
            line=dict(color="green", width=2, dash="dash")
        )
        
        fig2.update_layout(
            title=f"P(X > {c:.2f}) = {prob:.4f}",
            xaxis_title="x",
            yaxis_title="F(x)",
            yaxis=dict(range=[0, 1.05]),
            height=400
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Calculate the probability analytically
        if c <= a:
            prob_formula = "1"
        elif c >= b:
            prob_formula = "0"
        else:
            prob_formula = f"1 - (c - a)/(b - a) = 1 - ({c:.2f} - {a:.2f})/({b:.2f} - {a:.2f}) = {1 - (c - a)/(b - a):.4f}"
        
        st.markdown(f"""
        **Analytical Calculation:**
        
        P(X > {c:.2f}) = {prob_formula}
        
        This represents the probability that X is greater than {c:.2f}.
        """)
        
    else:  # P(c₁ < X ≤ c₂)
        col1, col2 = st.columns(2)
        with col1:
            c1 = st.slider("Lower bound (c₁):", float(a - padding), float(b + padding), a)
        with col2:
            c2 = st.slider("Upper bound (c₂):", float(a - padding), float(b + padding), b, key="c2_slider")
        
        if c1 >= c2:
            st.error("Lower bound must be less than upper bound!")
        else:
            prob = uniform_cdf(c2, a, b) - uniform_cdf(c1, a, b)
            
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
            
            # Add vertical lines at c1 and c2
            fig2.add_shape(
                type="line",
                x0=c1, y0=0,
                x1=c1, y1=uniform_cdf(c1, a, b),
                line=dict(color="red", width=2, dash="dash")
            )
            
            fig2.add_shape(
                type="line",
                x0=c2, y0=0,
                x1=c2, y1=uniform_cdf(c2, a, b),
                line=dict(color="red", width=2, dash="dash")
            )
            
            # Add horizontal lines
            fig2.add_shape(
                type="line",
                x0=c1, y0=uniform_cdf(c1, a, b),
                x1=c2, y1=uniform_cdf(c1, a, b),
                line=dict(color="red", width=2, dash="dash")
            )
            
            fig2.add_shape(
                type="line",
                x0=c2, y0=uniform_cdf(c2, a, b),
                x1=c2, y1=uniform_cdf(c1, a, b),
                line=dict(color="red", width=2, dash="dash")
            )
            
            # Add shaded area for P(c1 < X ≤ c2)
            x_fill = [val for val in x if c1 <= val <= c2]
            y_fill_bottom = [uniform_cdf(c1, a, b)] * len(x_fill)
            y_fill_top = [uniform_cdf(val, a, b) for val in x_fill]
            
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
            
            # Add vertical lines at a and b
            fig2.add_shape(
                type="line",
                x0=a, y0=0,
                x1=a, y1=uniform_cdf(a, a, b),
                line=dict(color="green", width=2, dash="dash")
            )
            
            fig2.add_shape(
                type="line",
                x0=b, y0=0,
                x1=b, y1=uniform_cdf(b, a, b),
                line=dict(color="green", width=2, dash="dash")
            )
            
            fig2.update_layout(
                title=f"P({c1:.2f} < X ≤ {c2:.2f}) = {prob:.4f}",
                xaxis_title="x",
                yaxis_title="F(x)",
                yaxis=dict(range=[0, 1.05]),
                height=400
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Calculate the probability analytically
            # Several cases to consider
            if c1 >= b or c2 <= a:
                prob_formula = "0"
            elif c1 <= a and c2 >= b:
                prob_formula = "1"
            elif c1 <= a and c2 < b:
                prob_formula = f"(c₂ - a)/(b - a) = ({c2:.2f} - {a:.2f})/({b:.2f} - {a:.2f}) = {(c2 - a)/(b - a):.4f}"
            elif c1 > a and c2 >= b:
                prob_formula = f"(b - c₁)/(b - a) = ({b:.2f} - {c1:.2f})/({b:.2f} - {a:.2f}) = {(b - c1)/(b - a):.4f}"
            else:  # a < c1 < c2 < b
                prob_formula = f"(c₂ - c₁)/(b - a) = ({c2:.2f} - {c1:.2f})/({b:.2f} - {a:.2f}) = {(c2 - c1)/(b - a):.4f}"
            
            st.markdown(f"""
            **Analytical Calculation:**
            
            P({c1:.2f} < X ≤ {c2:.2f}) = {prob_formula}
            
            This represents the probability that X is between {c1:.2f} and {c2:.2f}.
            """)
    
    # Add information about quantiles/percentiles
    st.subheader("Quantiles (Inverse CDF)")
    
    st.markdown("""
    The inverse of the CDF gives us quantiles - values below which a certain proportion of the data falls.
    
    For a uniform distribution, the quantile function is:
    
    Q(p) = a + p(b - a) for 0 ≤ p ≤ 1
    """)
    
    p = st.slider("Probability (p):", 0.0, 1.0, 0.5, 0.01)
    quantile = a + p * (b - a)
    
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
        x0=a - padding, y0=p,
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
    y_fill = [uniform_cdf(val, a, b) for val in x_fill]
    
    fig3.add_trace(go.Scatter(
        x=x_fill,
        y=y_fill,
        fill='tozeroy',
        fillcolor='rgba(0, 0, 255, 0.2)',
        line=dict(width=0),
        showlegend=False
    ))
    
    # Add vertical lines at a and b
    fig3.add_shape(
        type="line",
        x0=a, y0=0,
        x1=a, y1=uniform_cdf(a, a, b),
        line=dict(color="green", width=2, dash="dash")
    )
    
    fig3.add_shape(
        type="line",
        x0=b, y0=0,
        x1=b, y1=uniform_cdf(b, a, b),
        line=dict(color="green", width=2, dash="dash")
    )
    
    fig3.update_layout(
        title=f"Quantile for p = {p:.2f}",
        xaxis_title="x",
        yaxis_title="F(x)",
        yaxis=dict(range=[0, 1.05]),
        height=400
    )
    
    st.plotly_chart(fig3, use_container_width=True)
    
    # Add educational content
    st.subheader("Uniform Distribution in Practice")
    
    st.markdown("""
    ### Applications of the Uniform Distribution:
    
    1. **Random Number Generation**: The basis for generating random numbers in computing
    
    2. **Simulation**: Modeling scenarios where all outcomes in a range are equally likely
    
    3. **Uncertainty Modeling**: Representing uncertainty when only the bounds are known
    
    4. **Cryptography**: Used in various cryptographic algorithms
    
    5. **Statistical Testing**: Generating random samples for testing statistical hypotheses
    
    ### Key Properties:
    
    1. **Equal Probability**: All intervals of the same length within [a,b] have the same probability
    
    2. **Maximum Entropy**: Among all continuous distributions with support [a,b], the uniform distribution has the maximum entropy
    
    3. **Relationship to Other Distributions**:
       - If U is uniform on [0,1], then a + (b-a)U is uniform on [a,b]
       - If X is any continuous random variable with CDF F(x), then F(X) is uniform on [0,1]
    
    4. **Sum of Uniform Variables**: The sum of uniform random variables follows an Irwin-Hall distribution
    
    ### Interesting Facts:
    
    - The uniform distribution is the only continuous distribution where the PDF is constant over its support
    - It is one of the simplest continuous distributions but plays a fundamental role in probability theory and statistics
    """)
    
    # Add a section on the relationship between uniform distribution and other distributions
    st.subheader("Generating Other Distributions from Uniform")
    
    st.markdown("""
    The uniform distribution is fundamental in probability theory because it can be used to generate 
    random variables from any other distribution using techniques such as:
    
    1. **Inverse Transform Sampling**: If U is uniform on [0,1] and F is a CDF with inverse F⁻¹, 
       then F⁻¹(U) follows the distribution with CDF F
    
    2. **Box-Muller Transform**: Two independent uniform random variables can be transformed to 
       generate two independent standard normal random variables
    
    3. **Acceptance-Rejection Method**: Uses uniform random variables to generate samples from 
       more complex distributions
    
    These techniques are the foundation of random number generation in computational statistics and simulation.
    """)

if __name__ == "__main__":
    run_uniform_cdf()
