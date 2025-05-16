import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import pandas as pd

def run_regression_to_mean():
    st.title("📊 Regression to the Mean Explorer")
    
    st.markdown("""
    ### Understanding Regression to the Mean
    
    Regression to the mean is a statistical phenomenon where:
    
    - Extreme measurements tend to be closer to the average in subsequent measurements
    - Occurs due to natural variability and random chance
    - Important in fields like sports, education, and scientific research
    
    Explore how this concept works through interactive simulations.
    """)
    
    # Tabs for different regression to mean analyses
    tab1, tab2, tab3 = st.tabs([
        "Basic Demonstration", 
        "Variability Impact", 
        "Real-World Applications"
    ])
    
    with tab1:
        st.subheader("Regression to Mean Demonstration")
        
        # Scenario selection
        scenario = st.selectbox(
            "Select Scenario", 
            ["Height", "Test Scores", "Athletic Performance"]
        )
        
        # Parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            population_mean = st.slider("Population Mean", 50.0, 200.0, 100.0, 0.1)
        
        with col2:
            population_std = st.slider("Population Standard Deviation", 5.0, 50.0, 15.0, 0.1)
        
        with col3:
            num_trials = st.slider("Number of Trials", 2, 10, 5)
        
        # Data generation function
        def generate_regression_data(population_mean, population_std, num_trials):
            np.random.seed(42)
            
            # Simulate multiple measurements
            measurements = np.zeros((1000, num_trials))
            
            for trial in range(num_trials):
                measurements[:, trial] = np.random.normal(
                    population_mean, 
                    population_std, 
                    1000
                )
            
            return measurements
        
        # Generate data
        regression_data = generate_regression_data(population_mean, population_std, num_trials)
        
        # Identify extreme performers
        def find_extreme_performers(data, percentile=10):
            first_trial = data[:, 0]
            extreme_indices = np.where(
                (first_trial <= np.percentile(first_trial, percentile)) | 
                (first_trial >= np.percentile(first_trial, 100 - percentile))
            )[0]
            
            return extreme_indices
        
        # Extreme performers analysis
        extreme_indices = find_extreme_performers(regression_data)
        extreme_data = regression_data[extreme_indices, :]
        
        # Visualization
        fig_regression = go.Figure()
        
        # Add traces for each trial
        for trial in range(num_trials):
            fig_regression.add_trace(go.Box(
                y=extreme_data[:, trial],
                name=f'Trial {trial + 1}',
                boxmean=True
            ))
        
        fig_regression.update_layout(
            title="Regression to Mean for Extreme Performers",
            yaxis_title="Measurement Value"
        )
        
        st.plotly_chart(fig_regression, use_container_width=True)
        
        # Statistical analysis
        st.subheader("Regression to Mean Analysis")
        
        # Compute mean and standard deviation for each trial
        trial_means = np.mean(extreme_data, axis=0)
        trial_stds = np.std(extreme_data, axis=0)
        
        # Create DataFrame for analysis
        analysis_df = pd.DataFrame({
            'Trial': range(1, num_trials + 1),
            'Mean': trial_means,
            'Std Dev': trial_stds
        })
        
        st.table(analysis_df)
        
        st.markdown("""
        ### Interpretation
        
        - Extreme performers tend to move closer to population mean
        - Variability decreases across subsequent trials
        - Demonstrates natural statistical phenomenon
        """)
    
    with tab2:
        st.subheader("Impact of Variability")
        
        st.markdown("""
        ### How Variability Affects Regression to Mean
        
        Explore how different levels of variability influence 
        regression to the mean phenomenon.
        """)
        
        # Variability comparison
        variability_levels = st.multiselect(
            "Select Variability Levels", 
            ["Low", "Medium", "High", "Very High"],
            default=["Low", "High"]
        )
        
        # Comparative analysis
        def compare_variability(variability_levels, num_trials):
            np.random.seed(42)
            results = []
            
            variability_map = {
                "Low": 5.0,
                "Medium": 15.0,
                "High": 30.0,
                "Very High": 50.0
            }
            
            for level in variability_levels:
                # Simulate measurements with different variability
                measurements = np.zeros((1000, num_trials))
                std_dev = variability_map[level]
                
                for trial in range(num_trials):
                    measurements[:, trial] = np.random.normal(100, std_dev, 1000)
                
                # Find extreme performers
                first_trial = measurements[:, 0]
                extreme_indices = np.where(
                    (first_trial <= np.percentile(first_trial, 10)) | 
                    (first_trial >= np.percentile(first_trial, 90))
                )[0]
                
                extreme_data = measurements[extreme_indices, :]
                
                # Compute regression metrics
                trial_means = np.mean(extreme_data, axis=0)
                
                for t, mean in enumerate(trial_means):
                    results.append({
                        'Variability Level': level,
                        'Trial': t + 1,
                        'Mean': mean
                    })
            
            return pd.DataFrame(results)
        
        # Parameters
        col1, col2 = st.columns(2)
        
        with col1:
            compare_trials = st.slider("Number of Trials (Comparison)", 2, 10, 5)
        
        with col2:
            num_comparisons = st.slider("Number of Comparisons", 10, 200, 50)
        
        # Generate comparative data
        comparison_results = []
        for _ in range(num_comparisons):
            comparison_results.append(
                compare_variability(variability_levels, compare_trials)
            )
        
        # Aggregate results
        aggregated_results = pd.concat(comparison_results)
        
        # Visualization
        fig_compare = go.Figure()
        
        for level in variability_levels:
            level_data = aggregated_results[aggregated_results['Variability Level'] == level]
            
            fig_compare.add_trace(go.Box(
                x=level_data['Trial'],
                y=level_data['Mean'],
                name=level
            ))
        
        fig_compare.update_layout(
            title="Regression to Mean Across Variability Levels",
            xaxis_title="Trial Number",
            yaxis_title="Mean Value"
        )
        
        st.plotly_chart(fig_compare, use_container_width=True)
        
        # Detailed analysis
        st.subheader("Variability Impact Analysis")
        
        summary_analysis = aggregated_results.groupby(['Variability Level', 'Trial']).agg(['mean', 'std'])
        st.table(summary_analysis)
        
        st.markdown("""
        ### Insights
        
        - **Low Variability**: 
          Minimal regression to mean
          Consistent performance across trials
        
        - **High Variability**:
          Pronounced regression to mean
          Significant performance changes
        
        Variability plays a crucial role in regression to mean phenomenon.
        """)
    
    with tab3:
        st.subheader("Real-World Regression to Mean")
        
        st.markdown("""
        ### Practical Applications
        
        Explore regression to mean in various domains:
        - Sports Performance
        - Academic Achievement
        - Medical Treatment Outcomes
        """)
        
        # Application selection
        application = st.selectbox(
            "Select Application Domain", 
            ["Sports", "Education", "Medical Treatment"]
        )
        
        # Parameters
        col1, col2 = st.columns(2)
        
        with col1:
            sample_size_real = st.slider("Sample Size (Real-World)", 100, 10000, 1000)
        
        with col2:
            num_simulations_real = st.slider("Number of Simulations (Real-World)", 10, 200, 50)
        
        # Real-world data simulation
        def simulate_real_world_regression(application, sample_size):
            np.random.seed(42)
            
            if application == "Sports":
                # Simulate athlete performance
                base_performance = np.random.normal(100, 20, sample_size)
                performance_trials = np.zeros((sample_size, 5))
                
                for trial in range(5):
                    performance_trials[:, trial] = base_performance + np.random.normal(0, 10, sample_size)
            
            elif application == "Education":
                # Simulate test scores
                base_score = np.random.normal(75, 15, sample_size)
                performance_trials = np.zeros((sample_size, 5))
                
                for trial in range(5):
                    performance_trials[:, trial] = base_score + np.random.normal(0, 5, sample_size)
            
            else:  # Medical Treatment
                # Simulate treatment effectiveness
                base_health = np.random.normal(0, 1, sample_size)
                performance_trials = np.zeros((sample_size, 5))
                
                for trial in range(5):
                    performance_trials[:, trial] = base_health + np.random.normal(0, 0.5, sample_size)
            
            return performance_trials
        
        # Simulation and analysis
        real_world_data = simulate_real_world_regression(application, sample_size_real)
        
        # Find extreme performers
        first_trial = real_world_data[:, 0]
        extreme_indices = np.where(
            (first_trial <= np.percentile(first_trial, 10)) | 
            (first_trial >= np.percentile(first_trial, 90))
        )[0]
        
        extreme_data = real_world_data[extreme_indices, :]
        
        # Visualization
        fig_real = go.Figure()
        
        for trial in range(extreme_data.shape[1]):
            fig_real.add_trace(go.Box(
                y=extreme_data[:, trial],
                name=f'Trial {trial + 1}',
                boxmean=True
            ))
        
        fig_real.update_layout(
            title=f"Regression to Mean in {application}",
            yaxis_title="Performance Value"
        )
        
        st.plotly_chart(fig_real, use_container_width=True)
        
        # Statistical analysis
        st.subheader(f"{application} Regression to Mean Analysis")
        
        # Compute mean and standard deviation for each trial
        trial_means = np.mean(extreme_data, axis=0)
        trial_stds = np.std(extreme_data, axis=0)
        
        # Create DataFrame for analysis
        analysis_df = pd.DataFrame({
            'Trial': range(1, extreme_data.shape[1] + 1),
            'Mean': trial_means,
            'Std Dev': trial_stds
        })
        
        st.table(analysis_df)
        
        st.markdown(f"""
        ### {application} Data Insights
        
        **Interpretation**:
        - Demonstrates regression to mean in practical scenarios
        - Shows how extreme performances tend to normalize
        - Highlights the importance of understanding statistical variation
        """)

if __name__ == "__main__":
    run_regression_to_mean()
