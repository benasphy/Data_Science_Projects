import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import pandas as pd

def run_poisson_process():
    st.title("🌟 Poisson Process Simulator")
    
    st.markdown("""
    ### Understanding Poisson Processes
    
    A Poisson process is a model for random events that occur independently 
    at a constant average rate in a fixed interval of time or space.
    
    Key characteristics:
    - Events occur randomly and independently
    - Average rate of events is constant
    - Number of events in non-overlapping intervals are independent
    
    Explore the fascinating world of random event generation!
    """)
    
    # Tabs for different explorations
    tab1, tab2, tab3 = st.tabs([
        "Event Rate Simulation", 
        "Waiting Time Distribution", 
        "Multiple Interval Analysis"
    ])
    
    with tab1:
        st.subheader("Poisson Event Rate Simulation")
        
        # Parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_rate = st.slider("Average Events per Interval (λ)", 0.1, 20.0, 5.0, 0.1)
        
        with col2:
            total_intervals = st.slider("Total Intervals", 10, 1000, 200)
        
        with col3:
            interval_duration = st.slider("Interval Duration", 0.1, 10.0, 1.0, 0.1)
        
        # Simulate Poisson events
        events_per_interval = np.random.poisson(avg_rate, total_intervals)
        
        # Create histogram of event counts
        fig = go.Figure(data=[
            go.Histogram(
                x=events_per_interval, 
                nbinsx=20,
                histnorm='probability'
            )
        ])
        
        fig.update_layout(
            title=f"Poisson Event Distribution (λ = {avg_rate})",
            xaxis_title="Number of Events per Interval",
            yaxis_title="Proportion of Intervals"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Theoretical vs Empirical comparison
        x_theo = list(range(max(events_per_interval) + 1))
        pmf_theo = [stats.poisson.pmf(k, avg_rate) for k in x_theo]
        
        fig_comp = go.Figure()
        
        # Theoretical distribution
        fig_comp.add_trace(go.Scatter(
            x=x_theo, 
            y=pmf_theo, 
            mode='lines+markers',
            name='Theoretical PMF',
            line=dict(color='red', dash='dot')
        ))
        
        # Empirical distribution
        empirical_dist = pd.Series(events_per_interval).value_counts(normalize=True).sort_index()
        
        fig_comp.add_trace(go.Bar(
            x=empirical_dist.index, 
            y=empirical_dist.values,
            name='Empirical Distribution',
            opacity=0.7
        ))
        
        fig_comp.update_layout(
            title="Theoretical vs Empirical Poisson Distribution",
            xaxis_title="Number of Events",
            yaxis_title="Probability/Proportion"
        )
        
        st.plotly_chart(fig_comp, use_container_width=True)
        
        # Summary statistics
        summary = pd.Series(events_per_interval)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Events", f"{summary.mean():.2f}")
        col2.metric("Std Dev", f"{summary.std():.2f}")
        col3.metric("Max Events", f"{summary.max():.0f}")
        
        # Theoretical explanation
        st.markdown(f"""
        ### Poisson Distribution Insights
        
        For λ = {avg_rate}:
        - Expected number of events per interval: {avg_rate}
        - Variance of events: {avg_rate}
        
        #### Probability Mass Function
        
        $$P(X = k) = \frac{{\lambda^k e^{{-\lambda}}}}{{k!}}$$
        
        Where:
        - λ (lambda) is the average rate of events
        - k is the number of events
        """)
    
    with tab2:
        st.subheader("Waiting Time Between Events")
        
        st.markdown("""
        In a Poisson process, the time between events follows an exponential distribution.
        
        Key properties:
        - Memoryless property
        - Waiting time is independent of past events
        """)
        
        # Exponential distribution parameters
        rate = st.slider("Event Rate (λ)", 0.1, 10.0, 2.0, 0.1)
        
        # Simulate waiting times
        num_samples = st.slider("Number of Waiting Times", 100, 10000, 1000)
        waiting_times = np.random.exponential(1/rate, num_samples)
        
        # Create histogram of waiting times
        fig_wait = go.Figure(data=[
            go.Histogram(
                x=waiting_times, 
                nbinsx=30,
                histnorm='probability density'
            )
        ])
        
        fig_wait.update_layout(
            title=f"Waiting Time Distribution (λ = {rate})",
            xaxis_title="Time Between Events",
            yaxis_title="Probability Density"
        )
        
        st.plotly_chart(fig_wait, use_container_width=True)
        
        # Theoretical PDF overlay
        x = np.linspace(0, max(waiting_times), 200)
        pdf_theo = rate * np.exp(-rate * x)
        
        fig_theo = go.Figure()
        
        # Histogram
        fig_theo.add_trace(go.Histogram(
            x=waiting_times, 
            nbinsx=30,
            histnorm='probability density',
            opacity=0.7,
            name='Simulated Waiting Times'
        ))
        
        # Theoretical PDF
        fig_theo.add_trace(go.Scatter(
            x=x,
            y=pdf_theo,
            mode='lines',
            name='Theoretical PDF',
            line=dict(color='red', width=2)
        ))
        
        fig_theo.update_layout(
            title=f"Waiting Time: Theoretical vs Simulated (λ = {rate})",
            xaxis_title="Time Between Events",
            yaxis_title="Probability Density"
        )
        
        st.plotly_chart(fig_theo, use_container_width=True)
        
        # Summary statistics
        summary_wait = pd.Series(waiting_times)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Waiting Time", f"{summary_wait.mean():.2f}")
        col2.metric("Median Waiting Time", f"{summary_wait.median():.2f}")
        col3.metric("Std Dev", f"{summary_wait.std():.2f}")
        
        # Theoretical explanation
        st.markdown(f"""
        ### Exponential Distribution Insights
        
        For λ = {rate}:
        - Expected waiting time: {1/rate:.2f}
        - Variance of waiting time: {1/(rate**2):.2f}
        
        #### Probability Density Function
        
        $$f(t) = \lambda e^{{-\lambda t}}, \quad t \geq 0$$
        
        Where:
        - λ (lambda) is the event rate
        - t is the waiting time
        """)
    
    with tab3:
        st.subheader("Multiple Interval Analysis")
        
        st.markdown("""
        Explore how Poisson processes behave across different time intervals.
        
        Compare event occurrences in:
        - Short vs. long intervals
        - Different average rates
        """)
        
        # Multiple interval parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            interval_type = st.selectbox(
                "Interval Comparison",
                ["Different Rates", "Different Durations"]
            )
        
        with col2:
            if interval_type == "Different Rates":
                rates = st.multiselect(
                    "Event Rates (λ)",
                    [1.0, 2.0, 5.0, 10.0],
                    default=[1.0, 5.0]
                )
                intervals = [100] * len(rates)
            else:
                rates = [5.0] * 3
                intervals = st.multiselect(
                    "Interval Sizes",
                    [50, 100, 200, 500],
                    default=[50, 100, 200]
                )
        
        with col3:
            num_simulations = st.slider("Simulations per Interval", 100, 1000, 500)
        
        # Simulate events for multiple intervals
        event_data = []
        
        for i, (rate, interval_size) in enumerate(zip(rates, intervals)):
            events = np.random.poisson(rate, num_simulations)
            event_data.append(pd.DataFrame({
                'Rate/Interval': f'λ={rate}/n={interval_size}',
                'Events': events
            }))
        
        # Combine data
        combined_data = pd.concat(event_data)
        
        # Create box plot
        fig_multi = px.box(
            combined_data, 
            x='Rate/Interval', 
            y='Events',
            title='Event Distribution Across Intervals'
        )
        
        st.plotly_chart(fig_multi, use_container_width=True)
        
        # Statistical summary
        st.subheader("Statistical Summary")
        
        summary_table = combined_data.groupby('Rate/Interval')['Events'].agg([
            'mean', 'median', 'std', 'min', 'max'
        ])
        
        st.table(summary_table)
        
        # Theoretical vs Empirical probabilities
        st.subheader("Probability of Event Counts")
        
        # Create probability comparison
        prob_data = []
        
        for rate in rates:
            x_theo = list(range(max(combined_data['Events']) + 1))
            pmf_theo = [stats.poisson.pmf(k, rate) for k in x_theo]
            
            prob_data.append(pd.DataFrame({
                'Rate': rate,
                'Events': x_theo,
                'Theoretical Probability': pmf_theo
            }))
        
        prob_df = pd.concat(prob_data)
        
        fig_prob = px.line(
            prob_df, 
            x='Events', 
            y='Theoretical Probability', 
            color='Rate',
            title='Theoretical Probability of Event Counts'
        )
        
        st.plotly_chart(fig_prob, use_container_width=True)

if __name__ == "__main__":
    run_poisson_process()
