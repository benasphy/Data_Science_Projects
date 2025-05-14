import streamlit as st
import numpy as np
import plotly.graph_objects as go

def calculate_probabilities(prevalence, sensitivity, specificity):
    # Calculate probabilities using Bayes' Theorem
    true_positive = prevalence * sensitivity
    false_positive = (1 - prevalence) * (1 - specificity)
    true_negative = (1 - prevalence) * specificity
    false_negative = prevalence * (1 - sensitivity)
    
    # Calculate positive predictive value
    ppv = true_positive / (true_positive + false_positive)
    
    return {
        'True Positive': true_positive,
        'False Positive': false_positive,
        'True Negative': true_negative,
        'False Negative': false_negative,
        'PPV': ppv
    }

def run_drug_testing():
    st.title("💊 Drug Testing and Bayes' Theorem")
    st.markdown("""
    ### Understanding False Positives in Medical Testing

    This application demonstrates how Bayes' Theorem helps us understand the true meaning of a positive drug test result.
    It shows why we need to consider the base rate (prevalence) of drug use in the population when interpreting test results.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        prevalence = st.slider("Drug use prevalence (%)", 0.1, 20.0, 5.0) / 100
    with col2:
        sensitivity = st.slider("Test sensitivity (%)", 80.0, 99.9, 95.0) / 100
    with col3:
        specificity = st.slider("Test specificity (%)", 80.0, 99.9, 90.0) / 100

    st.markdown("""
    - **Prevalence**: Percentage of population that uses drugs
    - **Sensitivity**: Probability of a positive test if person uses drugs
    - **Specificity**: Probability of a negative test if person doesn't use drugs
    """)

    # Calculate probabilities
    probs = calculate_probabilities(prevalence, sensitivity, specificity)

    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = ["Population", "Users", "Non-Users", "Positive Test", "Negative Test"],
            color = ["blue", "red", "green", "purple", "gray"]
        ),
        link = dict(
            source = [0, 0, 1, 1, 2, 2],
            target = [1, 2, 3, 4, 3, 4],
            value = [prevalence, 1-prevalence, 
                     probs['True Positive'], probs['False Negative'],
                     probs['False Positive'], probs['True Negative']],
            color = ["red", "green", "red", "pink", "orange", "green"]
        )
    )])

    fig.update_layout(title_text="Drug Testing Flow Diagram", font_size=10)
    st.plotly_chart(fig)

    st.markdown(f"""
    ### Results

    If someone tests positive, what's the probability they actually use drugs?

    Using Bayes' Theorem:
    P(User|Positive) = P(Positive|User) × P(User) / P(Positive)

    **Positive Predictive Value = {probs['PPV']:.1%}**

    This means that if someone tests positive:
    - There's a {probs['PPV']:.1%} chance they actually use drugs
    - There's a {(1 - probs['PPV']):.1%} chance it's a false positive

    This counterintuitive result occurs because:
    1. The prevalence of drug use is low ({prevalence:.1%})
    2. Even with good test accuracy, the absolute number of false positives can exceed true positives in low-prevalence populations
    """)

    # Add simulation option
    if st.checkbox("Run Monte Carlo Simulation"):
        n_people = st.slider("Number of people to simulate", 1000, 100000, 10000)
        
        # Simulate population
        users = np.random.random(n_people) < prevalence
        
        # Simulate test results
        positive_tests = np.where(
            users,
            np.random.random(n_people) < sensitivity,  # True positives
            np.random.random(n_people) > specificity   # False positives
        )
        
        # Calculate simulated PPV
        simulated_ppv = np.sum(users & positive_tests) / np.sum(positive_tests)
        
        st.markdown(f"""
        ### Simulation Results
        With {n_people:,} simulated people:
        - Simulated Positive Predictive Value: {simulated_ppv:.1%}
        - Theoretical Positive Predictive Value: {probs['PPV']:.1%}
        """)

if __name__ == "__main__":
    run_drug_testing()
