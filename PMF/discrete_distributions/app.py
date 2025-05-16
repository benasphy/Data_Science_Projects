import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy import stats
import pandas as pd

def run_discrete_distributions():
    st.title("🎲 Discrete Probability Distributions Explorer")
    
    st.markdown("""
    ### Understanding Discrete Probability Distributions
    
    Discrete probability distributions model the probabilities of discrete (countable) random variables. 
    Unlike continuous distributions, these distributions have specific, distinct possible values.
    
    Explore common discrete distributions and their properties:
    """)
    
    # Distribution selection
    distribution = st.selectbox(
        "Select Discrete Distribution",
        [
            "Bernoulli", 
            "Binomial", 
            "Poisson", 
            "Geometric", 
            "Negative Binomial", 
            "Hypergeometric"
        ]
    )
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Distribution", "Probability Calculator", "Simulation"])
    
    with tab1:
        st.subheader(f"{distribution} Distribution")
        
        if distribution == "Bernoulli":
            # Bernoulli distribution parameters
            p = st.slider("Probability of Success (p)", 0.0, 1.0, 0.5, 0.01)
            
            # Calculate probabilities
            prob_success = p
            prob_failure = 1 - p
            
            # Create probability mass function
            x = [0, 1]
            pmf = [prob_failure, prob_success]
            
            # Create bar plot
            fig = go.Figure(data=[
                go.Bar(
                    x=x, 
                    y=pmf, 
                    text=[f'{p:.2%}' for p in pmf],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="Bernoulli Distribution PMF",
                xaxis_title="Outcome",
                yaxis_title="Probability",
                xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['Failure', 'Success'])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Distribution properties
            st.markdown(f"""
            ### Bernoulli Distribution Properties
            - **Mean**: {p}
            - **Variance**: {p * (1-p)}
            - **Success Probability**: {p:.2%}
            - **Failure Probability**: {1-p:.2%}
            #### Mathematical Formulation
            """)
            st.markdown(r"$$P(X = k) = p^k(1-p)^{1-k}, \quad k \in \{0, 1\}$$")
            st.markdown("The Bernoulli distribution models a single trial with two possible outcomes (success/failure, yes/no, true/false) with probability p of success.")
        
        elif distribution == "Binomial":
            # Binomial distribution parameters
            n = st.slider("Number of Trials (n)", 1, 50, 10)
            p = st.slider("Probability of Success (p)", 0.0, 1.0, 0.5, 0.01)
            
            # Calculate probabilities for each possible number of successes
            x = list(range(n + 1))
            pmf = [stats.binom.pmf(k, n, p) for k in x]
            
            # Create bar plot
            fig = go.Figure(data=[
                go.Bar(
                    x=x, 
                    y=pmf, 
                    text=[f'{prob:.2%}' for prob in pmf],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="Binomial Distribution PMF",
                xaxis_title="Number of Successes",
                yaxis_title="Probability"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Distribution properties
            st.markdown(f"""
            ### Binomial Distribution Properties
            - **Number of Trials**: {n}
            - **Success Probability**: {p:.2%}
            - **Mean**: {n * p}
            - **Variance**: {n * p * (1-p)}
            #### Mathematical Formulation
            """)
            st.markdown(r"$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0, 1, \ldots, n$$")
            st.markdown("The Binomial distribution models the number of successes in n independent Bernoulli trials, each with probability p of success.")
        
        elif distribution == "Poisson":
            # Poisson distribution parameters
            rate = st.slider("Average Rate (λ)", 0.1, 20.0, 5.0, 0.1)
            
            # Calculate probabilities for first 20 events
            x = list(range(20))
            pmf = [stats.poisson.pmf(k, rate) for k in x]
            
            # Create bar plot
            fig = go.Figure(data=[
                go.Bar(
                    x=x, 
                    y=pmf, 
                    text=[f'{prob:.2%}' for prob in pmf],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="Poisson Distribution PMF",
                xaxis_title="Number of Events",
                yaxis_title="Probability"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Distribution properties
            st.markdown(f"""
            ### Poisson Distribution Properties
            
            - **Average Rate (λ)**: {rate}
            - **Mean**: {rate}
            - **Variance**: {rate}
            
            #### Mathematical Formulation
            
            $$P(X = k) = \frac{{\lambda^k e^{{-\lambda}}}}{{k!}}, \quad k = 0, 1, 2, \ldots$$
            
            The Poisson distribution models the number of events occurring in a fixed 
            interval of time or space, given a known average rate.
            """)
        
        elif distribution == "Geometric":
            # Geometric distribution parameters
            p = st.slider("Probability of Success (p)", 0.0, 1.0, 0.5, 0.01, key="geom_dist_p")
            x = list(range(0, 15))
            pmf = [stats.geom.pmf(k+1, p) for k in x]  # k+1 because scipy's geom is 1-based
            fig = go.Figure(data=[
                go.Bar(x=x, y=pmf, text=[f'{prob:.2%}' for prob in pmf], textposition='auto')
            ])
            fig.update_layout(
                title="Geometric Distribution PMF",
                xaxis_title="Number of Failures Before First Success",
                yaxis_title="Probability"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"""
            ### Geometric Distribution Properties
            - **Mean**: {(1-p)/p:.2f}
            - **Variance**: {(1-p)/p**2:.2f}
            - **Success Probability**: {p:.2%}
            - **Failure Probability**: {1-p:.2%}
            #### Mathematical Formulation
            """)
            st.markdown(r"$$P(X = k) = (1-p)^k p, \quad k = 0, 1, 2, \ldots$$")
            st.markdown("The Geometric distribution models the number of failures before the first success in a sequence of independent Bernoulli trials.")
        elif distribution == "Negative Binomial":
            r = st.slider("Number of Successes (r)", 1, 20, 5, key="nbinom_dist_r")
            p = st.slider("Probability of Success (p)", 0.0, 1.0, 0.5, 0.01, key="nbinom_dist_p")
            x = list(range(0, 30))
            pmf = [stats.nbinom.pmf(k, r, p) for k in x]
            fig = go.Figure(data=[
                go.Bar(x=x, y=pmf, text=[f'{prob:.2%}' for prob in pmf], textposition='auto')
            ])
            fig.update_layout(
                title="Negative Binomial Distribution PMF",
                xaxis_title="Number of Failures Before r-th Success",
                yaxis_title="Probability"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"""
            ### Negative Binomial Distribution Properties
            - **Mean**: {r * (1-p)/p:.2f}
            - **Variance**: {r * (1-p)/p**2:.2f}
            - **Success Probability**: {p:.2%}
            - **Number of Successes (r)**: {r}
            #### Mathematical Formulation
            """)
            st.markdown(r"$$P(X = k) = \binom{k + r - 1}{k} p^r (1-p)^k, \quad k = 0, 1, 2, \ldots$$")
            st.markdown("The Negative Binomial distribution models the number of failures before achieving r successes in a sequence of independent Bernoulli trials.")
        elif distribution == "Hypergeometric":
            N = st.slider("Population Size (N)", 10, 200, 50, key="hypergeom_dist_N")
            K = st.slider("Number of Success States in Population (K)", 1, N, min(10, N), key="hypergeom_dist_K")
            n = st.slider("Number of Draws (n)", 1, N, min(10, N), key="hypergeom_dist_n")
            x = list(range(max(0, n+K-N), min(n, K)+1))
            pmf = [stats.hypergeom.pmf(k, N, K, n) for k in x]
            fig = go.Figure(data=[
                go.Bar(x=x, y=pmf, text=[f'{prob:.2%}' for prob in pmf], textposition='auto')
            ])
            fig.update_layout(
                title="Hypergeometric Distribution PMF",
                xaxis_title="Number of Successes in Draws",
                yaxis_title="Probability"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"""
            ### Hypergeometric Distribution Properties
            - **Population Size (N)**: {N}
            - **Number of Success States (K)**: {K}
            - **Number of Draws (n)**: {n}
            - **Mean**: {n * K / N:.2f}
            - **Variance**: {n * K * (N-K) * (N-n) / (N**2 * (N-1)):.2f}
            #### Mathematical Formulation
            """)
            st.markdown(r"$$P(X = k) = \frac{\binom{K}{k} \binom{N-K}{n-k}}{\binom{N}{n}}, \quad k = \max(0, n+K-N), \ldots, \min(n, K)$$")
            st.markdown("The Hypergeometric distribution models the number of successes in a fixed number of draws without replacement from a finite population.")
        
        # Add more distributions as needed
    
    with tab2:
        st.subheader("Probability Calculator")
        
        # Probability calculation based on selected distribution
        if distribution == "Bernoulli":
            calc_type = st.radio(
                "Calculate Probability",
                ["P(X = k)", "P(X ≤ k)", "P(X > k)"]
            )
            
            k = st.slider("k (Outcome)", 0, 1, 1)
            
            if calc_type == "P(X = k)":
                prob = p if k == 1 else 1 - p
                st.metric("Probability", f"{prob:.2%}")
            elif calc_type == "P(X ≤ k)":
                prob = p if k == 1 else 0
                st.metric("Probability", f"{prob:.2%}")
            else:
                prob = 1 - p if k == 1 else p
                st.metric("Probability", f"{prob:.2%}")
        
        elif distribution == "Binomial":
            calc_type = st.radio(
                "Calculate Probability",
                ["P(X = k)", "P(X ≤ k)", "P(X > k)"]
            )
            
            k = st.slider("k (Number of Successes)", 0, n, n//2)
            
            if calc_type == "P(X = k)":
                prob = stats.binom.pmf(k, n, p)
                st.metric("Probability", f"{prob:.2%}")
            elif calc_type == "P(X ≤ k)":
                prob = stats.binom.cdf(k, n, p)
                st.metric("Probability", f"{prob:.2%}")
            else:
                prob = 1 - stats.binom.cdf(k, n, p)
                st.metric("Probability", f"{prob:.2%}")
        
        elif distribution == "Poisson":
            calc_type = st.radio(
                "Calculate Probability",
                ["P(X = k)", "P(X ≤ k)", "P(X > k)"]
            )
            
            k = st.slider("k (Number of Events)", 0, 20, 5)
            
            if calc_type == "P(X = k)":
                prob = stats.poisson.pmf(k, rate)
                st.metric("Probability", f"{prob:.2%}")
            elif calc_type == "P(X ≤ k)":
                prob = stats.poisson.cdf(k, rate)
                st.metric("Probability", f"{prob:.2%}")
            else:
                prob = 1 - stats.poisson.cdf(k, rate)
                st.metric("Probability", f"{prob:.2%}")
        
        elif distribution == "Geometric":
            p = st.slider("Probability of Success (p)", 0.0, 1.0, 0.5, 0.01, key="geom_prob_p")
            k = st.slider("k (Number of Failures Before First Success)", 0, 15, 3, key="geom_prob_k")
            calc_type = st.radio("Calculate Probability", ["P(X = k)", "P(X ≤ k)", "P(X > k)"])
            if calc_type == "P(X = k)":
                prob = stats.geom.pmf(k+1, p)
            elif calc_type == "P(X ≤ k)":
                prob = stats.geom.cdf(k+1, p)
            else:
                prob = 1 - stats.geom.cdf(k+1, p)
            st.metric("Probability", f"{prob:.2%}")
        elif distribution == "Negative Binomial":
            r = st.slider("Number of Successes (r)", 1, 20, 5, key="nbinom_prob_r")
            p = st.slider("Probability of Success (p)", 0.0, 1.0, 0.5, 0.01, key="nbinom_prob_p")
            k = st.slider("k (Number of Failures Before r-th Success)", 0, 30, 5, key="nbinom_prob_k")
            calc_type = st.radio("Calculate Probability", ["P(X = k)", "P(X ≤ k)", "P(X > k)"])
            if calc_type == "P(X = k)":
                prob = stats.nbinom.pmf(k, r, p)
            elif calc_type == "P(X ≤ k)":
                prob = stats.nbinom.cdf(k, r, p)
            else:
                prob = 1 - stats.nbinom.cdf(k, r, p)
            st.metric("Probability", f"{prob:.2%}")
        elif distribution == "Hypergeometric":
            N = st.slider("Population Size (N)", 10, 200, 50, key="hypergeom_prob_N")
            K = st.slider("Number of Success States in Population (K)", 1, N, min(10, N), key="hypergeom_prob_K")
            n = st.slider("Number of Draws (n)", 1, N, min(10, N), key="hypergeom_prob_n")
            x_min = max(0, n+K-N)
            x_max = min(n, K)
            k = st.slider("k (Number of Successes in Draws)", x_min, x_max, x_min, key="hypergeom_prob_k")
            calc_type = st.radio("Calculate Probability", ["P(X = k)", "P(X ≤ k)", "P(X > k)"])
            if calc_type == "P(X = k)":
                prob = stats.hypergeom.pmf(k, N, K, n)
            elif calc_type == "P(X ≤ k)":
                prob = stats.hypergeom.cdf(k, N, K, n)
            else:
                prob = 1 - stats.hypergeom.cdf(k, N, K, n)
            st.metric("Probability", f"{prob:.2%}")
    
    with tab3:
        st.subheader("Simulation")
        
        # Simulation based on selected distribution
        num_simulations = st.slider("Number of Simulations", 100, 10000, 1000)
        
        if distribution == "Bernoulli":
            # Simulate Bernoulli trials
            simulations = np.random.binomial(1, p, num_simulations)
            
            # Create bar plot of simulation results
            sim_counts = pd.Series(simulations).value_counts(normalize=True)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=sim_counts.index, 
                    y=sim_counts.values, 
                    text=[f'{val:.2%}' for val in sim_counts.values],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title=f"Bernoulli Simulation (p = {p:.2f})",
                xaxis_title="Outcome",
                xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['Failure', 'Success']),
                yaxis_title="Proportion"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif distribution == "Binomial":
            # Simulate Binomial trials
            simulations = np.random.binomial(n, p, num_simulations)
            
            # Create histogram of simulation results
            fig = go.Figure(data=[
                go.Histogram(
                    x=simulations, 
                    nbinsx=n+1,
                    histnorm='probability'
                )
            ])
            
            fig.update_layout(
                title=f"Binomial Simulation (n = {n}, p = {p:.2f})",
                xaxis_title="Number of Successes",
                yaxis_title="Proportion"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif distribution == "Poisson":
            # Simulate Poisson events
            simulations = np.random.poisson(rate, num_simulations)
            
            # Create histogram of simulation results
            fig = go.Figure(data=[
                go.Histogram(
                    x=simulations, 
                    nbinsx=20,
                    histnorm='probability'
                )
            ])
            
            fig.update_layout(
                title=f"Poisson Simulation (λ = {rate})",
                xaxis_title="Number of Events",
                yaxis_title="Proportion"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        elif distribution == "Geometric":
            # Probability slider for Geometric distribution
            p = st.slider("Probability of Success (p)", 0.0, 1.0, 0.5, 0.01, key="geom_sim_p")
            # Simulate Geometric trials (number of failures before first success)
            simulations = np.random.geometric(p, num_simulations) - 1
            fig = go.Figure(data=[
                go.Histogram(
                    x=simulations,
                    nbinsx=20,
                    histnorm='probability'
                )
            ])
            fig.update_layout(
                title=f"Geometric Simulation (p = {p:.2f})",
                xaxis_title="Number of Failures Before First Success",
                yaxis_title="Proportion"
            )
            st.plotly_chart(fig, use_container_width=True)
        elif distribution == "Negative Binomial":
            r = st.slider("Number of Successes (r)", 1, 20, 5, key="nbinom_sim_r")
            p = st.slider("Probability of Success (p)", 0.0, 1.0, 0.5, 0.01, key="nbinom_sim_p")
            simulations = np.random.negative_binomial(r, p, num_simulations)
            fig = go.Figure(data=[
                go.Histogram(
                    x=simulations,
                    nbinsx=20,
                    histnorm='probability'
                )
            ])
            fig.update_layout(
                title=f"Negative Binomial Simulation (r = {r}, p = {p:.2f})",
                xaxis_title="Number of Failures Before r-th Success",
                yaxis_title="Proportion"
            )
            st.plotly_chart(fig, use_container_width=True)
        elif distribution == "Hypergeometric":
            N = st.slider("Population Size (N)", 10, 200, 50, key="hypergeom_sim_N")
            K = st.slider("Number of Success States in Population (K)", 1, N, min(10, N), key="hypergeom_sim_K")
            n = st.slider("Number of Draws (n)", 1, N, min(10, N), key="hypergeom_sim_n")
            simulations = np.random.hypergeometric(K, N-K, n, num_simulations)
            fig = go.Figure(data=[
                go.Histogram(
                    x=simulations,
                    nbinsx=n+1,
                    histnorm='probability'
                )
            ])
            fig.update_layout(
                title=f"Hypergeometric Simulation (N = {N}, K = {K}, n = {n})",
                xaxis_title="Number of Successes in Draws",
                yaxis_title="Proportion"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Display summary statistics
        st.subheader("Simulation Summary")
        
        # Calculate and display summary statistics
        summary_stats = pd.Series(simulations).describe()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean", f"{summary_stats['mean']:.2f}")
        col2.metric("Std Dev", f"{summary_stats['std']:.2f}")
        col3.metric("Unique Values", len(set(simulations)))

if __name__ == "__main__":
    run_discrete_distributions()
