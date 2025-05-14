import streamlit as st
import numpy as np
import plotly.graph_objects as go
from itertools import product

def simulate_two_child_problem(n_families):
    # Generate families with two children
    # 0 represents girl, 1 represents boy
    children = np.random.randint(2, size=(n_families, 2))
    
    # Count different scenarios
    total_at_least_one_girl = np.sum(np.any(children == 0, axis=1))
    total_both_girls = np.sum(np.all(children == 0, axis=1))
    
    # Calculate probability
    prob_both_girls_given_one_girl = total_both_girls / total_at_least_one_girl
    return prob_both_girls_given_one_girl

def run_two_child():
    st.title("👨‍👧‍👦 The Two Child Problem")
    st.markdown("""
    ### A Probability Puzzle Using Bayes' Theorem

    The Two Child Problem is a probability puzzle that demonstrates how additional information affects probability calculations.

    **The Problem:**
    - A family has two children
    - We know at least one of them is a girl
    - What's the probability that both children are girls?

    Many people intuitively answer 1/2, but let's see why this is incorrect using simulation and Bayes' Theorem!
    """)

    n_simulations = st.slider("Number of families to simulate", 1000, 100000, 10000)

    # Run simulation
    probability = simulate_two_child_problem(n_simulations)

    # Create visualization of possible outcomes
    outcomes = list(product(['Girl', 'Boy'], repeat=2))
    probabilities = {
        'Girl-Girl': 0.25,
        'Girl-Boy': 0.25,
        'Boy-Girl': 0.25,
        'Boy-Boy': 0.25
    }

    # Update probabilities given we know at least one is a girl
    total_prob = sum(v for k, v in probabilities.items() if 'Girl' in k)
    conditional_probs = {
        k: (v/total_prob if 'Girl' in k else 0) 
        for k, v in probabilities.items()
    }

    fig = go.Figure()

    # Original probabilities
    fig.add_trace(go.Bar(
        name='Original',
        x=list(probabilities.keys()),
        y=list(probabilities.values()),
        text=[f"{v:.1%}" for v in probabilities.values()],
        textposition='auto',
    ))

    # Conditional probabilities
    fig.add_trace(go.Bar(
        name='Given at least one girl',
        x=list(conditional_probs.keys()),
        y=list(conditional_probs.values()),
        text=[f"{v:.1%}" for v in conditional_probs.values()],
        textposition='auto',
    ))

    fig.update_layout(
        title="Probability Distribution of Two-Child Combinations",
        barmode='group',
        yaxis_title="Probability",
        yaxis_range=[0, 0.5]
    )

    st.plotly_chart(fig)

    st.markdown(f"""
    ### Simulation Results
    In our simulation with {n_simulations:,} families:
    - The probability of both children being girls, given that at least one is a girl: **{probability:.1%}**

    ### Explanation using Bayes' Theorem

    P(Both Girls | At least one Girl) = P(Both Girls) / P(At least one Girl)

    - P(Both Girls) = 1/4
    - P(At least one Girl) = 3/4

    Therefore: P(Both Girls | At least one Girl) = (1/4)/(3/4) = 1/3 ≈ 33%

    This demonstrates how conditional probability can lead to counterintuitive results!
    """)

if __name__ == "__main__":
    run_two_child()
