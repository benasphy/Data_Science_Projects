# Probability Density Function (PDF) Interactive Projects

This collection of interactive applications helps visualize and understand probability density functions and related concepts.

## Projects

### 1. Distribution Explorer
An interactive tool to explore common probability distributions and their properties. Features include:
- Visualize PDFs for various distributions (Normal, Exponential, Gamma, Beta, etc.)
- Adjust distribution parameters and see real-time updates
- Calculate probabilities for different ranges
- View distribution properties and formulas

### 2. Kernel Density Estimation
Learn how to estimate probability density functions from data using kernels:
- Explore different kernel functions and bandwidths
- Upload your own data for KDE analysis
- Compare KDE with histograms and true PDFs
- Understand the mathematical foundations of KDE

### 3. Multivariate Distributions
Explore joint probability distributions and their properties in multiple dimensions:
- Visualize bivariate normal distributions in 3D and contour plots
- Understand correlation and its effects on joint distributions
- Explore conditional distributions
- Generate random samples from multivariate distributions

## Running the Applications

Each application can be run individually using Streamlit:

```bash
cd /path/to/PDF/distribution_explorer
streamlit run app.py
```

Or access all applications through the main dashboard:

```bash
cd /path/to/Data_Science_Projects
streamlit run main.py
```

## Dependencies

All required packages are listed in the `requirements.txt` file. Install them using:

```bash
pip install -r requirements.txt
```

## Learning Objectives

These applications are designed to help you:
- Understand the concept of probability density functions
- Visualize how distribution parameters affect shape and properties
- Learn to calculate probabilities using PDFs
- Explore relationships between multiple random variables
- Understand non-parametric density estimation techniques
