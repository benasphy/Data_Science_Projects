import streamlit as st
import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

# Import modules from each project
# These will be imported dynamically when needed

st.set_page_config(
    page_title="Data Science Interactive Projects",
    page_icon="📊",
    layout="wide"
)

# Define project structure
PROJECTS = {
    "Bayes' Theorem": {
        "icon": "🎲",
        "description": "Interactive demonstrations of Bayes' Theorem through puzzles and applications",
        "projects": {
            "Monty Hall Problem": {
                "path": "Bayes_Theorem.monty_hall.app",
                "function": "run_monty_hall",
                "description": "A classic probability puzzle where switching doors counter-intuitively increases your chances of winning."
            },
            "Two Child Problem": {
                "path": "Bayes_Theorem.two_child_problem.app",
                "function": "run_two_child",
                "description": "Explore how additional information affects probability calculations in this famous puzzle."
            },
            "Drug Testing": {
                "path": "Bayes_Theorem.drug_testing.app",
                "function": "run_drug_testing",
                "description": "Understanding false positives and the importance of base rates in medical testing."
            }
        }
    },
    "CDF (Cumulative Distribution Functions)": {
        "icon": "📈",
        "description": "Explore cumulative distribution functions and their properties",
        "projects": {
            "Normal Distribution Explorer": {
                "path": "CDF.normal_distribution.app",
                "function": "run_normal_cdf",
                "description": "Interactive visualization of the normal distribution CDF with adjustable parameters."
            },
            "Survival Analysis": {
                "path": "CDF.survival_analysis.app",
                "function": "run_survival_analysis",
                "description": "Explore survival functions, hazard rates, and time-to-event analysis."
            },
            "Distribution Comparison Tool": {
                "path": "CDF.comparison_tool.app",
                "function": "run_cdf_comparison",
                "description": "Compare multiple CDFs side by side with statistical tests."
            }
        }
    },
    "PDF (Probability Density Functions)": {
        "icon": "🕯",
        "description": "Visualize and understand probability density functions",
        "projects": {
            "Distribution Explorer": {
                "path": "PDF.distribution_explorer.app",
                "function": "run_distribution_explorer",
                "description": "Interactive tool to explore common probability distributions and their properties."
            },
            "Kernel Density Estimation": {
                "path": "PDF.kernel_density.app",
                "function": "run_kernel_density",
                "description": "Learn how to estimate probability density functions from data using kernels."
            },
            "Multivariate Distributions": {
                "path": "PDF.multivariate_distributions.app",
                "function": "run_multivariate_distributions",
                "description": "Explore joint probability distributions and their properties in multiple dimensions."
            },
            "Mixture Models": {
                "path": "PDF.mixture_models.app",
                "function": "run_mixture_models",
                "description": "Explore probabilistic models for representing complex, multi-modal distributions."
            },
            "Probability Transformations": {
                "path": "PDF.probability_transformations.app",
                "function": "run_probability_transformations",
                "description": "Explore techniques for transforming probability distributions and generating complex random variables."
            },
            "Order Statistics": {
                "path": "PDF.order_statistics.app",
                "function": "run_order_statistics",
                "description": "Explore statistical ranking, extreme values, and distribution characteristics."
            }
        }
    },
    "PMF (Probability Mass Functions)": {
        "icon": "🎯",
        "description": "Understand discrete probability distributions",
        "projects": {
            "Discrete Distributions": {
                "path": "PMF.discrete_distributions.app",
                "function": "run_discrete_distributions",
                "description": "Explore common discrete distributions like Bernoulli, Binomial, and Poisson."
            },
            "Binomial Experiments": {
                "path": "PMF.binomial_experiments.app",
                "function": "run_binomial_experiments",
                "description": "Simulate binomial experiments and visualize the results."
            },
            "Poisson Process": {
                "path": "PMF.poisson_process.app",
                "function": "run_poisson_process",
                "description": "Visualize and understand the Poisson process for modeling random events."
            }
        }
    },
    "Summary Statistics": {
        "icon": "📊",
        "description": "Learn about measures that summarize data distributions",
        "projects": {
            "Central Tendency": {
                "path": "Summary_Statistics.central_tendency.app",
                "function": "run_central_tendency",
                "description": "Explore measures of central tendency in data."
            },
            "Dispersion Measures": {
                "path": "Summary_Statistics.dispersion_measures.app",
                "function": "run_dispersion_measures",
                "description": "Understand variance, standard deviation, and other measures of spread."
            },
            "Correlation Analysis": {
                "path": "Summary_Statistics.correlation_analysis.app",
                "function": "run_correlation_analysis",
                "description": "Visualize and calculate different correlation measures between variables."
            }
        }
    },
    "Estimation": {
        "icon": "🔍",
        "description": "Statistical methods for estimating population parameters",
        "projects": {
            "Confidence Intervals": {
                "path": "Estimation.confidence_intervals.app",
                "function": "run_confidence_intervals",
                "description": "Visualize and understand confidence interval estimation."
            },
            "Maximum Likelihood": {
                "path": "Estimation.maximum_likelihood.app",
                "function": "run_maximum_likelihood_estimation",
                "description": "Learn about maximum likelihood estimation techniques."
            },
            "Point Estimation": {
                "path": "Estimation.point_estimation.app",
                "function": "run_point_estimation",
                "description": "Explore various methods of point parameter estimation."
            }
        }
    },
    "Statistical Laws": {
        "icon": "⚖️",
        "description": "Fundamental laws and theorems in statistics",
        "projects": {
            "Central Limit Theorem": {
                "path": "Statistical_Laws.central_limit_theorem.app",
                "function": "run_central_limit_theorem",
                "description": "Interactive demonstration of the Central Limit Theorem with various distributions."
            },
            "Law of Large Numbers": {
                "path": "Statistical_Laws.law_of_large_numbers.app",
                "function": "run_law_of_large_numbers",
                "description": "Visualize how sample means converge to the population mean as sample size increases."
            },
            "Regression to the Mean": {
                "path": "Statistical_Laws.regression_to_mean.app",
                "function": "run_regression_to_mean",
                "description": "Understand the statistical phenomenon of regression toward the mean."
            }
        }
    }
}

def main():
    st.title("📊 Data Science Interactive Projects")
    
    st.markdown("""
    Welcome to the Data Science Interactive Projects dashboard! This collection demonstrates various 
    statistical concepts through interactive simulations and visualizations.
    
    Choose a topic from the sidebar to explore different projects.
    """)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    
    # Topic selection
    topic = st.sidebar.selectbox(
        "Choose a topic:",
        list(PROJECTS.keys()),
        format_func=lambda x: f"{PROJECTS[x]['icon']} {x}"
    )
    
    # Show topic description
    st.sidebar.markdown(f"**{PROJECTS[topic]['description']}**")
    
    # Project selection within the topic
    project_names = list(PROJECTS[topic]['projects'].keys())
    
    if len(project_names) > 0:
        project = st.sidebar.radio(
            f"Select a project in {topic}:",
            project_names
        )
        
        project_info = PROJECTS[topic]['projects'][project]
        
        # Display project info on sidebar
        st.sidebar.markdown(f"**Description:** {project_info['description']}")
        
        # Main content - immediately load the selected project
        if project:
            try:
                module_path = project_info['path']
                function_name = project_info['function']

                # Dynamic import
                module = __import__(module_path, fromlist=[function_name])
                run_function = getattr(module, function_name)

                # Run the project
                run_function()
            except ImportError as e:
                st.error(f"Could not load project: {e}")
                st.info("This project may not be implemented yet.")

if __name__ == "__main__":
    main()
