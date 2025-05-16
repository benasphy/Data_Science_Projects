import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy import stats
import pandas as pd

def exponential_cdf(x, lambda_param):
    """Calculate the CDF of an exponential distribution at point x"""
    return 1 - np.exp(-lambda_param * x)

def exponential_pdf(x, lambda_param):
    """Calculate the PDF of an exponential distribution at point x"""
    return lambda_param * np.exp(-lambda_param * x)

def run_exponential_cdf():
    st.title("📈 Exponential Distribution Explorer")
    
    st.markdown("""
    ### Understanding the Exponential Distribution and its CDF
    
    The exponential distribution models the time between events in a Poisson process, where events occur 
    continuously and independently at a constant average rate.
    
    For a random variable X with rate parameter λ (lambda):
    - PDF: f(x) = λe<sup>-λx</sup> for x ≥ 0
    - CDF: F(x) = 1 - e<sup>-λx</sup> for x ≥ 0
    
    The exponential distribution has the unique **memoryless property**: P(X > s+t | X > s) = P(X > t)
    
    This means the probability of waiting an additional time t is independent of how long you've already waited.
    """)
    
    # Parameter selection
    lambda_param = st.slider("Rate parameter (λ)", 0.1, 5.0, 1.0, 0.1)
    
    # Calculate mean and variance
    mean = 1 / lambda_param
    variance = 1 / (lambda_param ** 2)
    
    st.markdown(f"""
    **Distribution Properties:**
    - Mean (Expected Value): 1/λ = {mean:.2f}
    - Variance: 1/λ² = {variance:.2f}
    - Standard Deviation: 1/λ = {mean:.2f}
    - Median: ln(2)/λ = {np.log(2)/lambda_param:.2f}
    """)
    
    # Generate x values
    max_x = min(20, 5 / lambda_param)  # Adjust range based on lambda
    x = np.linspace(0, max_x, 1000)
    
    # Calculate CDF and PDF values
    cdf_values = [exponential_cdf(val, lambda_param) for val in x]
    pdf_values = [exponential_pdf(val, lambda_param) for val in x]
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["CDF Visualization", "Memoryless Property", "Waiting Time Simulation"])
    
    with tab1:
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
            title="Exponential Distribution: CDF and PDF",
            xaxis_title="x (time)",
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
                side="right"
            ),
            height=500,
            legend=dict(x=0.01, y=0.99)
        )
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Add interactive probability calculator
        st.subheader("Probability Calculator")
        
        calc_type = st.radio(
            "Calculate probability:",
            ["P(X ≤ t)", "P(X > t)", "P(t₁ < X ≤ t₂)"]
        )
        
        if calc_type == "P(X ≤ t)":
            t = st.slider("Time t:", 0.0, float(max_x), mean)
            prob = exponential_cdf(t, lambda_param)
            
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
            
            # Add vertical line at t
            fig2.add_shape(
                type="line",
                x0=t, y0=0,
                x1=t, y1=prob,
                line=dict(color="red", width=2, dash="dash")
            )
            
            # Add horizontal line to y-axis
            fig2.add_shape(
                type="line",
                x0=0, y0=prob,
                x1=t, y1=prob,
                line=dict(color="red", width=2, dash="dash")
            )
            
            # Add shaded area for P(X ≤ t)
            x_fill = [val for val in x if val <= t]
            y_fill = [exponential_cdf(val, lambda_param) for val in x_fill]
            
            fig2.add_trace(go.Scatter(
                x=x_fill,
                y=y_fill,
                fill='tozeroy',
                fillcolor='rgba(0, 0, 255, 0.2)',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig2.update_layout(
                title=f"P(X ≤ {t:.2f}) = {prob:.4f}",
                xaxis_title="x (time)",
                yaxis_title="F(x)",
                yaxis=dict(range=[0, 1.05]),
                height=400
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            st.markdown(f"""
            This represents the probability that the event occurs within {t:.2f} time units.
            
            For example, if λ = {lambda_param:.2f} events per unit time:
            - There's a {prob:.2%} chance that the event will occur within {t:.2f} time units
            - The expected time until the event is {mean:.2f} time units
            """)
            
        elif calc_type == "P(X > t)":
            t = st.slider("Time t:", 0.0, float(max_x), mean)
            prob = 1 - exponential_cdf(t, lambda_param)
            
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
            
            # Add vertical line at t
            fig2.add_shape(
                type="line",
                x0=t, y0=0,
                x1=t, y1=exponential_cdf(t, lambda_param),
                line=dict(color="red", width=2, dash="dash")
            )
            
            # Add shaded area for P(X > t)
            x_fill = [val for val in x if val >= t]
            y_fill_bottom = [exponential_cdf(t, lambda_param)] * len(x_fill)
            y_fill_top = [exponential_cdf(val, lambda_param) for val in x_fill]
            
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
                title=f"P(X > {t:.2f}) = {prob:.4f}",
                xaxis_title="x (time)",
                yaxis_title="F(x)",
                yaxis=dict(range=[0, 1.05]),
                height=400
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            st.markdown(f"""
            This represents the probability that the event occurs after {t:.2f} time units (survival function).
            
            For example, if λ = {lambda_param:.2f} events per unit time:
            - There's a {prob:.2%} chance that you'll need to wait more than {t:.2f} time units
            - This is also written as e<sup>-{lambda_param:.2f}×{t:.2f}</sup> = {np.exp(-lambda_param * t):.4f}
            """)
            
        else:  # P(t₁ < X ≤ t₂)
            col1, col2 = st.columns(2)
            with col1:
                t1 = st.slider("Lower bound (t₁):", 0.0, float(max_x), 0.0)
            with col2:
                t2 = st.slider("Upper bound (t₂):", 0.0, float(max_x), mean, key="t2_slider")
            
            if t1 >= t2:
                st.error("Lower bound must be less than upper bound!")
            else:
                prob = exponential_cdf(t2, lambda_param) - exponential_cdf(t1, lambda_param)
                
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
                
                # Add vertical lines at t1 and t2
                fig2.add_shape(
                    type="line",
                    x0=t1, y0=0,
                    x1=t1, y1=exponential_cdf(t1, lambda_param),
                    line=dict(color="red", width=2, dash="dash")
                )
                
                fig2.add_shape(
                    type="line",
                    x0=t2, y0=0,
                    x1=t2, y1=exponential_cdf(t2, lambda_param),
                    line=dict(color="red", width=2, dash="dash")
                )
                
                # Add horizontal lines
                fig2.add_shape(
                    type="line",
                    x0=t1, y0=exponential_cdf(t1, lambda_param),
                    x1=t2, y1=exponential_cdf(t1, lambda_param),
                    line=dict(color="red", width=2, dash="dash")
                )
                
                fig2.add_shape(
                    type="line",
                    x0=t2, y0=exponential_cdf(t2, lambda_param),
                    x1=t2, y1=exponential_cdf(t1, lambda_param),
                    line=dict(color="red", width=2, dash="dash")
                )
                
                # Add shaded area for P(t1 < X ≤ t2)
                x_fill = [val for val in x if t1 <= val <= t2]
                y_fill_bottom = [exponential_cdf(t1, lambda_param)] * len(x_fill)
                y_fill_top = [exponential_cdf(val, lambda_param) for val in x_fill]
                
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
                    title=f"P({t1:.2f} < X ≤ {t2:.2f}) = {prob:.4f}",
                    xaxis_title="x (time)",
                    yaxis_title="F(x)",
                    yaxis=dict(range=[0, 1.05]),
                    height=400
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                
                st.markdown(f"""
                This represents the probability that the event occurs between {t1:.2f} and {t2:.2f} time units.
                
                For example, if λ = {lambda_param:.2f} events per unit time:
                - There's a {prob:.2%} chance that the event will occur between {t1:.2f} and {t2:.2f} time units
                - This can be calculated as e<sup>-{lambda_param:.2f}×{t1:.2f}</sup> - e<sup>-{lambda_param:.2f}×{t2:.2f}</sup> = {np.exp(-lambda_param * t1) - np.exp(-lambda_param * t2):.4f}
                """)
    
    with tab2:
        st.subheader("Memoryless Property Demonstration")
        
        st.markdown("""
        The exponential distribution is the only continuous probability distribution that is memoryless.
        
        This means that if you've already waited some time s, the probability of waiting an additional 
        time t is the same as the original probability of waiting time t:
        
        P(X > s+t | X > s) = P(X > t)
        
        This is why the exponential distribution is often used to model the lifetime of components that 
        don't "age" or "wear out" - their future lifetime is independent of their current age.
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            s = st.slider("Time already waited (s):", 0.0, float(max_x/2), 1.0)
        with col2:
            t = st.slider("Additional time (t):", 0.0, float(max_x/2), 1.0, key="memoryless_t")
        
        # Calculate probabilities
        p_greater_t = np.exp(-lambda_param * t)
        p_greater_s_plus_t_given_greater_s = np.exp(-lambda_param * t)
        
        # Create visualization
        fig_memoryless = go.Figure()
        
        # Add the survival function curve
        fig_memoryless.add_trace(go.Scatter(
            x=x,
            y=[np.exp(-lambda_param * val) for val in x],
            mode='lines',
            name='Survival Function P(X > x)',
            line=dict(color='blue', width=2)
        ))
        
        # Add vertical lines
        fig_memoryless.add_shape(
            type="line",
            x0=t, y0=0,
            x1=t, y1=p_greater_t,
            line=dict(color="red", width=2, dash="dash")
        )
        
        fig_memoryless.add_shape(
            type="line",
            x0=s, y0=0,
            x1=s, y1=np.exp(-lambda_param * s),
            line=dict(color="green", width=2, dash="dash")
        )
        
        fig_memoryless.add_shape(
            type="line",
            x0=s+t, y0=0,
            x1=s+t, y1=np.exp(-lambda_param * (s+t)),
            line=dict(color="purple", width=2, dash="dash")
        )
        
        # Add horizontal lines
        fig_memoryless.add_shape(
            type="line",
            x0=0, y0=p_greater_t,
            x1=t, y1=p_greater_t,
            line=dict(color="red", width=2, dash="dash")
        )
        
        fig_memoryless.add_shape(
            type="line",
            x0=0, y0=np.exp(-lambda_param * s),
            x1=s, y1=np.exp(-lambda_param * s),
            line=dict(color="green", width=2, dash="dash")
        )
        
        fig_memoryless.add_shape(
            type="line",
            x0=0, y0=np.exp(-lambda_param * (s+t)),
            x1=s+t, y1=np.exp(-lambda_param * (s+t)),
            line=dict(color="purple", width=2, dash="dash")
        )
        
        fig_memoryless.update_layout(
            title="Memoryless Property of Exponential Distribution",
            xaxis_title="x (time)",
            yaxis_title="P(X > x)",
            yaxis=dict(range=[0, 1.05]),
            height=500,
            annotations=[
                dict(
                    x=t/2,
                    y=p_greater_t + 0.05,
                    text="P(X > t)",
                    showarrow=False
                ),
                dict(
                    x=s/2,
                    y=np.exp(-lambda_param * s) + 0.05,
                    text="P(X > s)",
                    showarrow=False
                ),
                dict(
                    x=s + t/2,
                    y=np.exp(-lambda_param * (s+t)) + 0.05,
                    text="P(X > s+t)",
                    showarrow=False
                )
            ]
        )
        
        st.plotly_chart(fig_memoryless, use_container_width=True)
        
        st.markdown(f"""
        **Demonstration of the Memoryless Property:**
        
        - P(X > t) = e<sup>-λt</sup> = e<sup>-{lambda_param:.2f}×{t:.2f}</sup> = {p_greater_t:.4f}
        - P(X > s+t | X > s) = P(X > s+t) / P(X > s) = e<sup>-λ(s+t)</sup> / e<sup>-λs</sup> = e<sup>-λt</sup> = {p_greater_s_plus_t_given_greater_s:.4f}
        
        As you can see: P(X > s+t | X > s) = P(X > t) = {p_greater_t:.4f}
        
        This demonstrates the memoryless property of the exponential distribution.
        """)
    
    with tab3:
        st.subheader("Waiting Time Simulation")
        
        st.markdown("""
        Let's simulate waiting times between events in a Poisson process with rate parameter λ.
        
        Each simulation generates random waiting times between consecutive events, following an exponential distribution.
        """)
        
        # Simulation parameters
        num_events = st.slider("Number of events to simulate:", 5, 100, 20)
        num_simulations = st.slider("Number of simulations:", 1, 10, 1)
        
        # Run simulations
        all_simulations = []
        
        for sim in range(num_simulations):
            # Generate random waiting times
            waiting_times = np.random.exponential(scale=1/lambda_param, size=num_events)
            
            # Calculate event times (cumulative sum of waiting times)
            event_times = np.cumsum(waiting_times)
            
            # Create dataframe for this simulation
            df_sim = pd.DataFrame({
                'Event': range(1, num_events + 1),
                'Waiting Time': waiting_times,
                'Event Time': event_times,
                'Simulation': f"Simulation {sim+1}"
            })
            
            all_simulations.append(df_sim)
        
        # Combine all simulations
        df_all = pd.concat(all_simulations)
        
        # Create visualization
        fig_sim = go.Figure()
        
        for sim in range(num_simulations):
            sim_data = df_all[df_all['Simulation'] == f"Simulation {sim+1}"]
            
            # Add trace for event times
            fig_sim.add_trace(go.Scatter(
                x=sim_data['Event Time'],
                y=[sim+1] * len(sim_data),
                mode='markers',
                name=f"Simulation {sim+1}",
                marker=dict(size=10)
            ))
        
        fig_sim.update_layout(
            title="Simulated Event Times",
            xaxis_title="Time",
            yaxis_title="Simulation",
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(1, num_simulations + 1)),
                ticktext=[f"Sim {i+1}" for i in range(num_simulations)]
            ),
            height=300 + 50 * num_simulations
        )
        
        st.plotly_chart(fig_sim, use_container_width=True)
        
        # Display statistics
        st.subheader("Simulation Statistics")
        
        # Calculate statistics for waiting times
        mean_waiting_time = df_all['Waiting Time'].mean()
        std_waiting_time = df_all['Waiting Time'].std()
        
        st.markdown(f"""
        **Waiting Time Statistics:**
        - Theoretical Mean: 1/λ = {1/lambda_param:.4f}
        - Simulated Mean: {mean_waiting_time:.4f}
        - Theoretical Standard Deviation: 1/λ = {1/lambda_param:.4f}
        - Simulated Standard Deviation: {std_waiting_time:.4f}
        """)
        
        # Histogram of waiting times
        fig_hist = go.Figure()
        
        fig_hist.add_trace(go.Histogram(
            x=df_all['Waiting Time'],
            nbinsx=20,
            opacity=0.7,
            name="Simulated Waiting Times"
        ))
        
        # Add theoretical PDF curve
        x_pdf = np.linspace(0, max(df_all['Waiting Time']), 1000)
        y_pdf = [lambda_param * np.exp(-lambda_param * val) for val in x_pdf]
        
        fig_hist.add_trace(go.Scatter(
            x=x_pdf,
            y=[y * len(df_all['Waiting Time']) * (max(df_all['Waiting Time'])/20) for y in y_pdf],
            mode='lines',
            name='Theoretical PDF',
            line=dict(color='red', width=2)
        ))
        
        fig_hist.update_layout(
            title="Distribution of Waiting Times",
            xaxis_title="Waiting Time",
            yaxis_title="Frequency",
            height=400
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Show the data
        if st.checkbox("Show simulation data"):
            st.dataframe(df_all)
    
    # Add educational content
    st.subheader("Applications of the Exponential Distribution")
    
    st.markdown("""
    ### Real-world Applications:
    
    1. **Reliability Engineering**: Modeling the lifetime of electronic components that fail due to random events rather than wear and tear
    
    2. **Queueing Theory**: Modeling the time between arrivals in a Poisson process (e.g., customers arriving at a service counter)
    
    3. **Nuclear Physics**: Modeling the time between radioactive decay events
    
    4. **Telecommunications**: Modeling the time between incoming calls to a call center
    
    5. **Survival Analysis**: Modeling lifetimes when the hazard rate (instantaneous risk of failure) is constant
    
    ### Key Properties:
    
    1. **Memoryless**: The only continuous distribution with the memoryless property
    
    2. **Constant Hazard Rate**: The instantaneous failure rate λ is constant over time
    
    3. **Maximum Entropy**: Among all continuous distributions with support [0,∞) and a fixed mean, the exponential distribution has the maximum entropy
    
    4. **Relationship to Poisson Process**: If events occur according to a Poisson process with rate λ, then the time between consecutive events follows an exponential distribution with parameter λ
    """)

if __name__ == "__main__":
    run_exponential_cdf()
