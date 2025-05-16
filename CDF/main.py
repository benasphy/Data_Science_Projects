import streamlit as st
import sys
import os

# Add the current directory to the path so we can import the modules
sys.path.append(os.path.dirname(__file__))

# Import the run functions from each app
from normal_distribution.app import run_normal_cdf
from exponential_distribution.app import run_exponential_cdf
from uniform_distribution.app import run_uniform_cdf

def main():
    st.set_page_config(
        page_title="CDF Interactive Projects",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 Cumulative Distribution Function (CDF) Explorer")
    
    st.markdown("""
    ### Understanding Cumulative Distribution Functions
    
    The Cumulative Distribution Function (CDF) is a fundamental concept in probability and statistics that 
    describes the probability that a random variable X takes on a value less than or equal to x.
    
    Mathematically, for a random variable X, the CDF is defined as:
    
    F(x) = P(X ≤ x)
    
    CDFs have several important properties:
    - They are non-decreasing (monotonically increasing)
    - They range from 0 to 1
    - For continuous distributions, F(x) = ∫<sub>-∞</sub><sup>x</sup> f(t) dt, where f(t) is the PDF
    - For discrete distributions, F(x) = ∑<sub>t≤x</sub> P(X = t)
    
    Explore different distribution CDFs through our interactive applications:
    """)
    
    # Create a container for the project cards
    project_container = st.container()
    
    with project_container:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🔔 Normal Distribution
            
            Explore the bell-shaped normal distribution's CDF, which is fundamental in statistics due to the Central Limit Theorem.
            
            - Visualize how parameters affect the shape
            - Calculate probabilities for different ranges
            - Understand quantiles and percentiles
            """)
            if st.button("Explore Normal Distribution", key="normal"):
                st.session_state.app = "normal"
        
        with col2:
            st.markdown("""
            ### ⏱️ Exponential Distribution
            
            Investigate the exponential distribution's CDF, which models the time between events in a Poisson process.
            
            - Understand the memoryless property
            - Calculate waiting time probabilities
            - Simulate random events
            """)
            if st.button("Explore Exponential Distribution", key="exponential"):
                st.session_state.app = "exponential"
        
        with col3:
            st.markdown("""
            ### 📊 Uniform Distribution
            
            Examine the uniform distribution's CDF, the simplest continuous distribution where all intervals of equal length have equal probability.
            
            - Visualize the linear CDF
            - Calculate probabilities for different ranges
            - Understand how uniform distributions generate other distributions
            """)
            if st.button("Explore Uniform Distribution", key="uniform"):
                st.session_state.app = "uniform"
    
    # Display the selected app
    if "app" not in st.session_state:
        st.session_state.app = None
    
    if st.session_state.app == "normal":
        run_normal_cdf()
    elif st.session_state.app == "exponential":
        run_exponential_cdf()
    elif st.session_state.app == "uniform":
        run_uniform_cdf()
    
    # Add a button to return to the main menu
    if st.session_state.app is not None:
        if st.button("Return to Main Menu"):
            st.session_state.app = None
            st.experimental_rerun()
    
    # Add educational content if no app is selected
    if st.session_state.app is None:
        st.markdown("""
        ---
        
        ### Why are CDFs Important?
        
        Cumulative Distribution Functions (CDFs) are powerful tools in probability and statistics for several reasons:
        
        1. **Calculating Probabilities**: CDFs directly give the probability that a random variable is less than or equal to a specific value
        
        2. **Finding Percentiles**: The inverse of the CDF gives percentiles of the distribution
        
        3. **Comparing Distributions**: CDFs provide a visual way to compare different distributions
        
        4. **Generating Random Variables**: CDFs are used in the inverse transform sampling method to generate random variables
        
        5. **Statistical Tests**: Many statistical tests, such as the Kolmogorov-Smirnov test, use CDFs
        
        6. **Risk Assessment**: In finance, CDFs help calculate Value at Risk (VaR) and other risk measures
        
        ### Relationship Between PDF and CDF
        
        For continuous distributions:
        - The CDF is the integral of the PDF: F(x) = ∫<sub>-∞</sub><sup>x</sup> f(t) dt
        - The PDF is the derivative of the CDF: f(x) = d/dx F(x)
        
        For discrete distributions:
        - The CDF is the sum of the PMF: F(x) = ∑<sub>t≤x</sub> P(X = t)
        - The PMF can be derived from the CDF as P(X = x) = F(x) - F(x-)
        
        ### Applications in Data Science
        
        - **Anomaly Detection**: Identifying values in the tails of a distribution
        - **Quantile Regression**: Modeling the relationship between variables at different quantiles
        - **Survival Analysis**: Modeling time-to-event data
        - **Monte Carlo Simulation**: Generating random samples for simulation studies
        """)

if __name__ == "__main__":
    main()
