import streamlit as st
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

# Import the individual apps
from monty_hall.app import run_monty_hall
from two_child_problem.app import run_two_child
from drug_testing.app import run_drug_testing

st.set_page_config(
    page_title="Bayes' Theorem Projects",
    page_icon="🎲",
    layout="wide"
)

def main():
    st.title("🎲 Bayes' Theorem Interactive Projects")
    
    st.markdown("""
    Welcome to the Bayes' Theorem projects dashboard! This collection demonstrates the power and 
    applications of Bayes' Theorem through interactive simulations and real-world examples.
    
    ### Available Projects:
    """)

    # Project selection
    project = st.selectbox(
        "Choose a project to explore:",
        ["Home", "Monty Hall Problem", "Two Child Problem", "Drug Testing"],
        format_func=lambda x: f"📊 {x}" if x != "Home" else "🏠 Home"
    )

    # Display project descriptions on home
    if project == "Home":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            #### 🚪 Monty Hall Problem
            A classic probability puzzle where switching doors counter-intuitively increases your chances of winning.
            
            [Launch Project →](/?project=Monty+Hall+Problem)
            """)
            
        with col2:
            st.markdown("""
            #### 👨‍👧‍👦 Two Child Problem
            Explore how additional information affects probability calculations in this famous puzzle.
            
            [Launch Project →](/?project=Two+Child+Problem)
            """)
            
        with col3:
            st.markdown("""
            #### 💊 Drug Testing
            Understanding false positives and the importance of base rates in medical testing.
            
            [Launch Project →](/?project=Drug+Testing)
            """)
            
        st.markdown("""
        ### About Bayes' Theorem
        
        Bayes' Theorem is a fundamental principle in probability theory that describes the probability of an event 
        based on prior knowledge of conditions that might be related to the event. The theorem is stated mathematically as:
        
        P(A|B) = P(B|A) × P(A) / P(B)
        
        Where:
        - P(A|B) is the posterior probability
        - P(B|A) is the likelihood
        - P(A) is the prior probability
        - P(B) is the normalizing constant
        
        These projects demonstrate various applications of this powerful theorem in both recreational mathematics 
        and real-world scenarios.
        """)
    
    else:
        # Run the selected project
        if project == "Monty Hall Problem":
            run_monty_hall()
        elif project == "Two Child Problem":
            run_two_child()
        elif project == "Drug Testing":
            run_drug_testing()

if __name__ == "__main__":
    main()
