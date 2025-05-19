# Volatility analysis with the GARCH(1, 1) model

This project demonstrates and compares volatility clustering in both simulated and real market data using GARCH(1, 1) (Generalized Autoregressive Conditional Heteroskedasticity) model. The analysis includes simulated GARCH processes, volatility shock impacts, and real S&P 500 market behavior.

## Project structure and codebase

- `simulate_data.py`: Generate synthetic financial data with volatility clustering and shocks
- `garch_analysis.py`: Implementation of GARCH model analysis
- `visualization.py`: Functions for visualizing results and analyzing shock impacts
- `main.py`: Main script to run the complete analysis
- `notebooks/`: Jupyter notebook following along the main.py script

## Overview

1. Simulated Data Analysis
   - GARCH(1,1) process simulation
   - Homoskedastic (constant volatility) process
   - Comparison of volatility patterns

2. Volatility Shock Analysis
   - Simulate and analyze large shocks
   - Study shock propagation and persistence
   - Compare pre- and post-shock GARCH parameters

3. Real Market Analysis
   - S&P 500 daily returns analysis
   - Comparison with simulated data
   - Real market volatility clustering patterns

4. Visualizations
   - Returns and volatility plots
   - Volatility clustering evidence via the Auto correlation function
   - Returns distribution
   - Shock impact visualization
   - Real vs. simulated data comparison

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies

- numpy: Numerical computations
- pandas: Data manipulation
- matplotlib: Plotting and visualization
- seaborn: Enhanced plotting
- arch: GARCH model implementation
- yfinance: Market data fetching
- statsmodels: Statistical analysis

## Running the Analysis

1. Run the main script:
```bash
python main.py
```

2. Or explore the Jupyter notebook in the `notebooks/` directory.

## Methodology

### GARCH model implementation

The GARCH(1,1) process for the volatility squared time series is defined as follows:

σ²ₜ = ω + α₁ε²ₜ₋₁ + β₁σ²ₜ₋₁

Where:
- σ²ₜ is the conditional variance at time t
- ω is the constant term (omega)
- α₁ is the ARCH parameter (impact of past returns)
- β₁ is the GARCH parameter (persistence of volatility)
- ε²ₜ₋₁ is the squared return from the previous period

Key Parameters Used:
- ω = 0.05 (constant term)
- α = 0.15 (ARCH effect)
- β = 0.80 (GARCH effect)
- Total Persistence = 0.95

### Shock Analysis Configuration

The shock analysis introduces controlled volatility spikes with:
- Shock Magnitude: 8x normal volatility
- Shock Timing: Mid-sample period
- Analysis Window: Pre/post-shock comparison


## Main analysis components

1. Simulated GARCH Process
   - Generate synthetic returns with volatility clustering
   - Control parameters for volatility persistence
   - Compare with homoskedastic process

2. Volatility Shock Analysis
   - Introduce significant market shocks
   - Analyze shock decay
   - Study changes in volatility dynamics
   - Compare pre- and post-shock behavior

3. GARCH Model Analysis
   - Fit GARCH(1,1) model to both simulated and real data
   - Estimate and compare parameters
   - Analyze volatility persistence
   - Study model performance on different data types

4. Real Market Integration
   - Fetch and analyze S&P 500 data
   - Compare market behavior with simulations
   - Validate GARCH model assumptions
   - Study real volatility clustering patterns

5. Visualization & Diagnostics
   - Time series plots of returns and volatility
   - Autocorrelation analysis for clustering
   - Returns distribution vs. normal
   - Shock impact visualization
   - Comparative analysis plots

## Summary of key findings

1. **Volatility Clustering**
   - Strong evidence in S&P 500 data as seen from ACF
   - Evidence also in simulated data
   - Clustering intensifies post-shock

2. **Model Accuracy**
   - High correlation with true volatility (0.98)
   - Low estimation error (MAE: 0.103)
   - Robust performance across different market conditions

3. **Shock Impact**
   - Significant increase in mean volatility
   - Persistence increases post-shock
   - Quick initial response followed by gradual decay

4. **Real vs. Simulated Data**
   - Similar persistence patterns
   - Higher kurtosis in real data (6.94 vs 2.53)



## Things for the future

- EGARCH for asymmetric volatility
- Alternative volatility models (Heston, Rough Heston, Rough Fractional Volatility)

- Analyse more real world market data 
   - CBOE Volatility Index (VIX), Options data, Historical Stock Prices
   - Analysis during crisis period 

- More advanced stuff ... 
   - Integration of real-time market monitoring
   - Automated shock detection
   - Study various risk metrics (Implied Volatility, Volatility of Volatility, VIX Index, Value-at-Risk, Maximum Drawdown, Expected Shortfall, Realized Skewness & Kurtosis)