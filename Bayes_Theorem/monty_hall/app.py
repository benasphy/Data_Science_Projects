import streamlit as st
import numpy as np
import plotly.graph_objects as go
from collections import defaultdict

def simulate_monty_hall(n_simulations, switch):
    wins = 0
    for _ in range(n_simulations):
        # Set up doors (0 = goat, 1 = car)
        doors = [0, 0, 0]
        car_position = np.random.randint(0, 3)
        doors[car_position] = 1
        
        # Player's initial choice
        initial_choice = np.random.randint(0, 3)
        
        # Monty opens a door
        available_doors = [i for i in range(3) 
                         if i != initial_choice and doors[i] == 0]
        monty_opens = np.random.choice(available_doors)
        
        if switch:
            # Switch to the remaining door
            final_choice = [i for i in range(3) 
                          if i != initial_choice and i != monty_opens][0]
        else:
            final_choice = initial_choice
            
        if doors[final_choice] == 1:
            wins += 1
            
    return wins / n_simulations

def run_monty_hall():
    st.title("🚪 The Monty Hall Problem")
    st.markdown("""
    ### Understanding Bayes' Theorem through a Classic Puzzle

    The Monty Hall problem is a probability puzzle named after the host of the TV show 'Let's Make a Deal'.
    Here's the scenario:
    - There are three doors, behind one is a car, behind the others are goats
    - You pick a door
    - The host (who knows what's behind each door) opens another door showing a goat
    - You can either stick with your original choice or switch to the remaining door

    What should you do? Let's simulate to find out!
    """)

    col1, col2 = st.columns(2)
    with col1:
        n_simulations = st.slider("Number of simulations", 100, 10000, 1000)
    with col2:
        show_theory = st.checkbox("Show theoretical probability")

    results = {
        "Switch": simulate_monty_hall(n_simulations, True),
        "Stay": simulate_monty_hall(n_simulations, False)
    }

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(results.keys()),
        y=list(results.values()),
        text=[f"{v:.1%}" for v in results.values()],
        textposition='auto',
    ))

    fig.update_layout(
        title="Win Probability: Switch vs Stay",
        yaxis_title="Probability of Winning",
        yaxis_range=[0, 1],
        showlegend=False
    )

    if show_theory:
        fig.add_hline(y=2/3, line_dash="dash", annotation_text="Theoretical (Switch): 2/3")
        fig.add_hline(y=1/3, line_dash="dash", annotation_text="Theoretical (Stay): 1/3")

    st.plotly_chart(fig)

    st.markdown("""
    ### Why does switching give you a better chance?

    Using Bayes' Theorem, we can explain this counterintuitive result:

    P(Win|Switch) = P(Car in Other Door|Initial Choice Wrong) × P(Initial Choice Wrong)

    - P(Initial Choice Wrong) = 2/3
    - P(Car in Other Door|Initial Choice Wrong) = 1

    Therefore, P(Win|Switch) = 2/3 ≈ 67%

    While P(Win|Stay) = P(Initial Choice Right) = 1/3 ≈ 33%
    """)

if __name__ == "__main__":
    run_monty_hall()
