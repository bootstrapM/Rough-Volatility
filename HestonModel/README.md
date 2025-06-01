# Heston Model Implementation and Analysis

This project implements and analyzes the Heston Stochastic Volatility Model, a widely used model in quantitative finance for pricing options and understanding market dynamics.

## Project Structure

```
HestonModel/
├── src/
│   ├── heston_model.py      # Core Heston model implementation
│   ├── heston_pricing.py    # Option pricing methods
│   └── main.py             # Analysis and visualization
├── figures/                # Generated plots and analysis
└── README.md              # This file
```

## Features

1. **Heston Model Implementation**
   - Stochastic volatility dynamics
   - Mean reversion characteristics
   - Correlation between asset returns and volatility
   - Feller condition verification

2. **Option Pricing Methods**
   - Characteristic function method
   - Monte Carlo simulation
   - Black-Scholes comparison
   - Implied volatility surface

3. **Analysis Tools**
   - Volatility clustering analysis
   - Mean reversion studies
   - Parameter sensitivity analysis
   - Autocorrelation function analysis

## Key Findings

### Model Parameters
- Mean reversion speed (κ): 1.20
- Long-term variance (θ): 0.10 (31.6% volatility)
- Volatility of variance (σ): 0.30
- Correlation (ρ): -0.80
- Initial variance (v₀): 0.10

### Feller Condition Analysis
- 2κθ = 0.240
- σ² = 0.090
- Condition satisfied with margin: 0.150

### Volatility Characteristics
1. **Volatility Clustering**
   - Strong persistence in volatility
   - Positive autocorrelation in absolute returns
   - Decay pattern following exponential form

2. **Mean Reversion**
   - Variance reverts to long-term level θ
   - Speed of mean reversion varies with initial conditions
   - Estimated mean reversion speed matches theoretical κ

3. **Leverage Effect**
   - Strong negative correlation (-0.8) between returns and volatility
   - Captures market asymmetry
   - Explains volatility skew in option prices

### Option Pricing Analysis
1. **Pricing Methods Comparison**
   - Characteristic function method provides benchmark prices
   - Monte Carlo convergence with increasing paths
   - Black-Scholes comparison using implied volatility

2. **Implied Volatility Surface**
   - Smile/skew patterns
   - Term structure effects
   - Strike price dependency

## Running the Analysis

To run the complete analysis:

```bash
python src/main.py
```

This will generate:
1. Price and volatility paths
2. Volatility clustering analysis
3. Mean reversion studies
4. Parameter sensitivity analysis
5. Option pricing comparisons
6. Implied volatility surface

## Generated Figures

The analysis produces several key visualizations in the `figures/` directory:

1. `price_and_vol_paths.png`
   - Sample price paths
   - Corresponding volatility paths
   - Mean reversion visualization

2. `volatility_clustering_single_instance_acf.png`
   - Autocorrelation of absolute returns
   - Autocorrelation of squared returns
   - Confidence intervals

3. `mean_reversion.png`
   - Variance paths from different initial levels
   - Convergence to long-term variance
   - Confidence bands

4. `sensitivity_*.png`
   - Parameter sensitivity analysis
   - Impact of κ, θ, and σ changes
   - Mean volatility paths

5. `pricing_methods_comparison.png`
   - Monte Carlo vs Characteristic function prices
   - Convergence analysis
   - Error bounds

6. `implied_volatility_surface.png`
   - 3D surface of implied volatilities
   - Strike price dependency
   - Time to maturity effects

## Key Insights

1. **Volatility Dynamics**
   - The model successfully captures volatility clustering
   - Mean reversion is evident in variance paths
   - Strong leverage effect is maintained

2. **Pricing Accuracy**
   - Characteristic function method provides stable prices
   - Monte Carlo converges with sufficient paths
   - Implied volatility surface shows realistic patterns

3. **Model Stability**
   - Feller condition is satisfied
   - No negative variances in simulation
   - Parameter sensitivity is well-behaved

## Future Enhancements

1. **Model Extensions**
   - Jump diffusion component
   - Time-dependent parameters
   - Multi-factor volatility

2. **Analysis Tools**
   - Calibration to market data
   - Risk metrics calculation
   - Greeks computation

3. **Performance Optimization**
   - Parallel Monte Carlo simulation
   - GPU acceleration
   - Variance reduction techniques

## Dependencies

- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- Seaborn
- statsmodels

## References

1. Heston, S. L. (1993). "A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options"
2. Gatheral, J. (2006). "The Volatility Surface: A Practitioner's Guide"
3. Cont, R. (2001). "Empirical Properties of Asset Returns: Stylized Facts and Statistical Issues" 