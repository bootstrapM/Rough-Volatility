# Rough Fractional Stochastic Volatility 

This project implements and analyzes fractional random walks of various Hurst parameters with applications to rough volatility in financial time series. The implementation follows a structured approach covering three main sections: Gaussian Random Variables, Fractional Random Walk, and Price Returns with Rough Volatilities.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── src/                                # Source code
│   ├── gaussian_rv.py                  # Generates Gaussian random variables
│   ├── fractional_random_walk.py       # Fractional Random Walk implementation and analysis
│   ├── price_returns.py                # Price returns with rough volatility and analysis
│   ├── analyze_frw.py                  # Some analysis scripts for FRW
│   └── analyze_price_returns.py        # Some analysis scripts for price returns
├── notebooks/                          # Contains notebooks with various implementations
└── docs/
    └── images/                         # Visualization plots for readme file
```

## Setup

1. Within the folder containing the above file system create a virtual environment:
```bash
python3 -m venv venv
```

2. Activate the virtual environment:
```bash
source venv/bin/activate 
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project modules

1. `gaussian_rv.py`
   - Generates Gaussian random variables and contains variance analysis that will be needed for the next module

2. `fractional_random_walk.py` 
   - Generates fractional random walk time series with different Hurst parameters (H)
   - Calculates the auto-covariance function of the time series that shows a linear dependence on the lag for small values of lag compared to the length of the time series
   - Variogram and Higher-moment analysis of the time series 

3. Price Returns with fractional rough stochastic volatilities
   - Model implementation
   - Return variance analysis
   - Drawdown analysis
   - Return distribution properties
   - Correlation structure analysis
   - Price variogram analysis

## Dependencies

- numpy
- matplotlib
- scipy
- pandas
- PyPDF2
- pdfplumber

## Usage

To run the analysis:
```bash
python src/analyze_price_returns.py  # For price returns analysis
python src/analyze_frw.py           # For FRW analysis
```

## Technical Implementation Details

### Numerical Stability
- Log-space calculations for price process
- Standardization of volatility process
- Careful handling of high-moment calculations

### Performance Optimization
- Vectorized implementations where possible
- Efficient correlation computation
- Smart memory management for large N

## Conclusions

1. **Rough Volatility Properties**
   - Successfully implemented rough volatility with H = 0.1
   - Captured key empirical features of financial returns
   - Demonstrated impact of vol-of-vol parameter

2. **Statistical Properties**
   - Verified theoretical predictions
   - Documented deviations from Gaussian behavior
   - Quantified impact of parameters

3. **Market Implications**
   - Insights into drawdown risk
   - Understanding of volatility feedback
   - Implications for option pricing

## Usage

[To be added as we implement the functionality] 

## Section 1: Gaussian Random Variables

### Implementation
- Generated N independent Gaussian random variables (ωk, εk)
- Variance scaling: E[ωk²] = E[εk²] = 1/k²
- Implemented variance calculations and numerical validation

### Key Results
- Verified the theoretical variance of the sum ϑ = Σ(ωk + εk)
- Numerical validation showed excellent agreement with theory
- Implementation is numerically stable for large N

## Fractional Random Walk

### Implementation
The `FractionalRandomWalk` class implements:
- Time series generation with different Hurst parameters (H)
- Correlation function computation
- Variogram analysis
- Higher-moment analysis

### Key Findings

#### Time Series Properties
![FRW Paths](docs/images/frw_paths.png)
*Figure 1: Fractional Random Walk paths for different Hurst parameters (H = 0.7, 0.5, 0.3, 0.1)*

- Generated FRW for H = 0.7, 0.5, 0.3, 0.1
- Observed increasing roughness with decreasing H
- Validated scaling behavior through variogram analysis

#### Scaling Analysis
![FRW Variograms](docs/images/frw_variograms.png)
*Figure 2: Variogram scaling for different Hurst parameters. Dashed lines show theoretical scaling.*

- Computed variograms for different moment orders (q = 2,3,4)
- Confirmed scaling behavior V(ρ) ~ ρ^(2H)
- Higher moments showed expected scaling Vq(ρ) ~ ρ^(qH)

## Section 3: Price Returns and Rough Volatilities

### Model Implementation
The price returns model is implemented as:
```python
rt = σ * φt * exp(γ * θt - γ²/2 * E[θt²])
```
where:
- σ: base volatility level
- γ: vol-of-vol parameter
- θt: rough volatility process (H = 0.1)
- φt: standard normal innovations

### Analysis Results

#### 1. Return Series Properties
![Return Series](docs/images/return_series.png)
*Figure 3: Return series for different vol-of-vol parameters (γ = 0.0, 0.3, 0.6)*

| γ Value | Empirical Variance | Theoretical Variance | Relative Error |
|---------|-------------------|---------------------|----------------|
| 0.0     | 1.000454         | 1.000000           | 0.045%        |
| 0.3     | 1.093028         | 1.000000           | 9.303%        |
| 0.6     | 1.383764         | 1.000000           | 38.376%       |

#### 2. Volatility Dynamics
![Volatility Processes](docs/images/volatility_processes.png)
*Figure 4: Volatility processes for different vol-of-vol parameters*

#### 3. Return Distributions
![Return Distributions](docs/images/return_distributions.png)
*Figure 5: Return distributions compared to normal distribution*

Key observations:
- Heavy tails emerge with increasing γ
- Volatility clustering visible for γ > 0
- Return distributions deviate from normality

#### 4. Drawdown Analysis
| γ Value | Maximum Drawdown | Duration (steps) |
|---------|-----------------|------------------|
| 0.0     | 89.55%         | 7,245           |
| 0.3     | 60.17%         | 2,742           |
| 0.6     | 87.96%         | 19,076          |

Findings:
- Moderate vol-of-vol (γ = 0.3) leads to smaller but more frequent drawdowns
- High vol-of-vol (γ = 0.6) results in extended drawdown periods
- Significant drawdowns possible even with constant volatility 