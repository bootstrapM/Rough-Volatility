# Function Documentation

This document provides a comprehensive overview of all functions implemented in the Rough Fractional Stochastic Volatility project.

## Module: `gaussian_random_variables.py`

This module creates sets of gaussian random variables that will be needed for generating the rough volatility process. It also performs elementary variance analysis on these random variables. It contains the following functions.

### `generate_random_variables(N: int) -> Tuple[np.ndarray, np.ndarray]`
Generates two sets of independent Gaussian random variables $\omega$ and $\epsilon$ that form the basis of the rough volatility process. These variables are used to construct the fractional Brownian motion components of the volatility model. 
$$
\omega_k \sim \mathcal{N}(0, 1/k^2)~,\qquad \epsilon_k \sim \mathcal{N}(0, 1/k^2)~,\qquad k = 1,2, \cdots N/2~.
$$
- **Args:**
  - `N`: Number of variables (will generate N/2 variables for each set)
- **Returns:**
  - Return the sets $\{\omega_k\}$ and $\{\epsilon_k\}$.

### `calculate_theoretical_variance(N: int) -> float`
Returns the variance of the following random variable
$$
x(N) = \sum_{k=1}^{N/2} \omega_k + \sum_{k=1}^{N/2} \beta_k~.
$$
- **Args:**
  - `N`: Number of variables
- **Returns:**
  - Theoretical variance of x(N): the total sum of the two sets.

### `calculate_empirical_variance(N: int, num_realizations: int = 10000) -> float`
Return the empirical variance of $x(N)$ by averaging over many realizations
- **Args:**
  - `N`: Number of variables
  - `num_realizations`: Number of realizations
- **Returns:**
  - Empirical variance of $x(N)$. 

### `validate_variance(N: int, num_realizations: int = 10000) -> Tuple[float, float, float]`
This function implements all the functionalities of this module in a single call. Returns theoretical variance and empirical variance along with its error
- **Args:**
  - `N`: Number of variables
  - `num_realizations`: Number of realizations for computing empirical variance
- **Returns:**
  - Tuple containing theoretical variance, empirical variance, and relative error
