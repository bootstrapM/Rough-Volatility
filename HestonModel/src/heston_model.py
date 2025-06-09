import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy import stats

@dataclass
class HestonParameters:
    """Parameters for the Heston model."""
    kappa: float  # Mean reversion rate
    theta: float  # Long-term variance
    sigma: float  # Volatility of variance
    rho: float    # Correlation between price and variance processes
    v0: float     # Initial variance
    mu: float     # Drift rate

class HestonModel:
    """
    Implementation of the Heston stochastic volatility model.
    
    The model is defined by the following SDEs:
    dS(t) = μS(t)dt + √v(t)S(t)dW₁(t)
    dv(t) = κ(θ - v(t))dt + σ√v(t)dW₂(t)
    """
    
    def __init__(self, params: HestonParameters):
        self.params = params
        self._validate_parameters()
        
    def _validate_parameters(self):
        """Validate Feller condition and other parameter constraints."""
            
        # Check basic parameter constraints
        if not (self.params.kappa > 0 and self.params.theta > 0 and 
                self.params.sigma > 0 and abs(self.params.rho) <= 1 and
                self.params.v0 > 0):
            raise ValueError("Invalid parameter values.")
        
        # Check Feller condition for positivity of the variance process (important for the model to be well-defined)
        feller_condition = 2 * self.params.kappa * self.params.theta > self.params.sigma**2
        if not feller_condition:
            print("Warning: Feller condition not satisfied. Variance process might hit zero.")

    def simulate_paths(self, S0: float, T: float, N: int, M: int,
                      return_vol: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Simulate price paths using the full truncation Euler scheme.
        
        Args:
            S0: Initial price
            T: Time horizon
            N: Number of time steps
            M: Number of paths
            return_vol: Whether to return variance paths
            
        Returns:
            Tuple of arrays (prices, variances) if return_vol=True, else just prices
        """
        dt = T/N
        sqrt_dt = np.sqrt(dt)
        
        # Initialize paths
        prices = np.zeros((M, N+1))
        variances = np.zeros((M, N+1))
        prices[:, 0] = S0
        variances[:, 0] = self.params.v0
        
        # Below Z1 and Z2 are correlated standard normal random variables with correlation coefficient rho
        Z1 = np.random.standard_normal((M, N))
        Z2 = (self.params.rho * Z1 + 
              np.sqrt(1 - self.params.rho**2) * np.random.standard_normal((M, N)))
        
        # Vectorized code for simulation
        for i in range(N):
            # Get current values
            v_current = variances[:, i]
            p_current = prices[:, i]
            
            # Ensure positive variance (full truncation)
            v_positive = np.maximum(v_current, 0)
            
            # Calculate variance update
            var_update = (v_current + 
                         self.params.kappa * (self.params.theta - v_positive) * dt +
                         self.params.sigma * np.sqrt(v_positive) * Z2[:, i] * sqrt_dt)
            
            # Apply full truncation to the entire update
            variances[:, i+1] = np.maximum(var_update, 0)
            
            # Update prices using the truncated variance
            prices[:, i+1] = (p_current * 
                             np.exp((self.params.mu - 0.5 * v_positive) * dt +
                                   np.sqrt(v_positive) * Z1[:, i] * sqrt_dt))
        
        return (prices, variances) if return_vol else prices

    def analyze_volatility_clustering(self, prices: np.ndarray, lags: int = 50):
        """
        Analyze volatility clustering through autocorrelation of absolute and squared returns.
        Averages ACF over all sample paths.
        
        Args:
            prices: 2D array of prices (M x N+1) where M is number of paths
            lags: Number of lags for autocorrelation
            
        Returns:
            Tuple of (lags, autocorrelations) for both absolute and squared returns
        """
        M, N_plus_1 = prices.shape
        N = N_plus_1 - 1
        
        # Calculate returns (log returns) for all paths
        returns = np.log(prices[:, 1:] / prices[:, :-1])  # Shape: (M, N)
        
        # Calculate absolute and squared returns
        abs_returns = np.abs(returns)
        squared_returns = returns ** 2
        
        # Calculate autocorrelation for both measures
        def calculate_acf(data):
            # Initialize array to store ACF for each path
            acf_all_paths = np.zeros((M, lags))
            
            # Calculate ACF for each path
            for m in range(M):
                # Get data for this path
                path_data = data[m, :]
                # Remove any NaN or infinite values
                path_data = path_data[np.isfinite(path_data)]
                
                # Calculate autocorrelation
                acf = np.zeros(lags)
                acf[0] = 1.0  # Lag 0 autocorrelation is always 1
                
                for lag in range(1, lags):
                    if len(path_data) > lag:
                        correlation = np.corrcoef(path_data[lag:], path_data[:-lag])[0, 1]
                        acf[lag] = correlation if not np.isnan(correlation) else 0.0
                
                acf_all_paths[m, :] = acf
            
            # Average ACF across all paths
            mean_acf = np.mean(acf_all_paths, axis=0)
            std_acf = np.std(acf_all_paths, axis=0)
            
            return mean_acf, std_acf
        
        # Calculate ACF for both measures
        acf_abs, std_abs = calculate_acf(abs_returns)
        acf_squared, std_squared = calculate_acf(squared_returns)
        
        # Plot results
        plt.figure(figsize=(15, 10))
        
        # ACF of absolute returns
        plt.subplot(2, 1, 1)
        lags_to_plot = np.arange(1, lags)  # Start from 1 instead of 0
        conf_level = 1.96/np.sqrt(N)
        
        plt.fill_between(lags_to_plot, 
                         -conf_level, 
                         conf_level,
                         alpha=0.4,
                         color='orange',
                         label='95% Confidence Interval')
        
        # Plot mean ACF with error bars (excluding lag 0)
        plt.errorbar(lags_to_plot, acf_abs[1:], yerr=std_abs[1:], 
                    fmt='bo-', markersize=6, linewidth=2, 
                    label='Mean ACF of Absolute Returns',
                    capsize=5)
        
        plt.title('Autocorrelation of Absolute Returns (Averaged over Paths)', fontsize=12, pad=20)
        plt.xlabel('Lag', fontsize=10)
        plt.ylabel('Autocorrelation', fontsize=10)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # ACF of squared returns
        plt.subplot(2, 1, 2)
        plt.fill_between(lags_to_plot, 
                         -conf_level, 
                         conf_level,
                         alpha=0.4,
                         color='orange',
                         label='95% Confidence Interval')
        
        # Plot mean ACF with error bars (excluding lag 0)
        plt.errorbar(lags_to_plot, acf_squared[1:], yerr=std_squared[1:],
                    fmt='ro-', markersize=6, linewidth=2,
                    label='Mean ACF of Squared Returns',
                    capsize=5)
        
        plt.title('Autocorrelation of Squared Returns (Averaged over Paths)', fontsize=12, pad=20)
        plt.xlabel('Lag', fontsize=10)
        plt.ylabel('Autocorrelation', fontsize=10)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../figures/volatility_clustering_acf.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        # Print statistics
        print("\nVolatility Clustering Analysis (Averaged over Paths):")
        print(f"First-order autocorrelation of absolute returns: {acf_abs[1]:.3f} ± {std_abs[1]:.3f}")
        print(f"First-order autocorrelation of squared returns: {acf_squared[1]:.3f} ± {std_squared[1]:.3f}")
        print(f"Mean autocorrelation of absolute returns (lags 1-10): {np.mean(acf_abs[1:11]):.3f} ± {np.mean(std_abs[1:11]):.3f}")
        print(f"Mean autocorrelation of squared returns (lags 1-10): {np.mean(acf_squared[1:11]):.3f} ± {np.mean(std_squared[1:11]):.3f}")
        
        return lags_to_plot, acf_abs, acf_squared

    def plot_volatility_decay(self, prices: np.ndarray, T: float, N: int, 
                            lags: int = 50) -> None:
        """
        Plot the decay of volatility clustering through ACF of absolute and squared returns.
        Includes both exponential and log-linear decay analysis.
        
        Args:
            prices: 2D array of prices (M x N+1) where M is number of paths
            T: Time horizon
            N: Number of time steps
            lags: Number of lags for autocorrelation
        """
        M, N_plus_1 = prices.shape
        
        # Calculate returns (log returns) for all paths
        returns = np.log(prices[:, 1:] / prices[:, :-1])  # Shape: (M, N)
        
        # Calculate absolute and squared returns
        abs_returns = np.abs(returns)
        squared_returns = returns ** 2
        
        # Calculate autocorrelation for both measures
        def calculate_acf(data):
            # Initialize array to store ACF for each path
            acf_all_paths = np.zeros((M, lags))
            
            # Calculate ACF for each path
            for m in range(M):
                # Get data for this path
                path_data = data[m, :]
                # Remove any NaN or infinite values
                path_data = path_data[np.isfinite(path_data)]
                
                # Calculate autocorrelation
                acf = np.zeros(lags)
                acf[0] = 1.0  # Lag 0 autocorrelation is always 1
                
                for lag in range(1, lags):
                    if len(path_data) > lag:
                        correlation = np.corrcoef(path_data[lag:], path_data[:-lag])[0, 1]
                        acf[lag] = correlation if not np.isnan(correlation) else 0.0
                
                acf_all_paths[m, :] = acf
            
            # Average ACF across all paths
            mean_acf = np.mean(acf_all_paths, axis=0)
            std_acf = np.std(acf_all_paths, axis=0)
            
            return mean_acf, std_acf
        
        # Calculate ACF for both measures
        acf_abs, std_abs = calculate_acf(abs_returns)
        acf_squared, std_squared = calculate_acf(squared_returns)
        
        # Create time points for x-axis
        time_points = np.arange(1, lags)  # Exclude lag 0
        
        # Plot results - Exponential decay
        plt.figure(figsize=(15, 10))
        
        # ACF of absolute returns
        plt.subplot(2, 1, 1)
        plt.errorbar(time_points, acf_abs[1:], yerr=std_abs[1:],
                    fmt='bo-', markersize=6, linewidth=2,
                    label='ACF of Absolute Returns',
                    capsize=5)
        
        # Fit exponential decay
        def exp_decay(x, a, b):
            return a * np.exp(-b * x)
        
        try:
            popt_abs, _ = curve_fit(exp_decay, time_points, acf_abs[1:])
            a_abs, b_abs = popt_abs
            plt.plot(time_points, exp_decay(time_points, a_abs, b_abs),
                    'r--', label=f'Exponential fit: {a_abs:.3f}exp(-{b_abs:.3f}x)')
        except:
            print("Warning: Could not fit exponential decay to absolute returns ACF")
        
        plt.title('Decay of Absolute Returns Autocorrelation', fontsize=12, pad=20)
        plt.xlabel('Lag', fontsize=10)
        plt.ylabel('Autocorrelation', fontsize=10)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # ACF of squared returns
        plt.subplot(2, 1, 2)
        plt.errorbar(time_points, acf_squared[1:], yerr=std_squared[1:],
                    fmt='ro-', markersize=6, linewidth=2,
                    label='ACF of Squared Returns',
                    capsize=5)
        
        # Fit exponential decay
        try:
            popt_squared, _ = curve_fit(exp_decay, time_points, acf_squared[1:])
            a_squared, b_squared = popt_squared
            plt.plot(time_points, exp_decay(time_points, a_squared, b_squared),
                    'b--', label=f'Exponential fit: {a_squared:.3f}exp(-{b_squared:.3f}x)')
        except:
            print("Warning: Could not fit exponential decay to squared returns ACF")
        
        plt.title('Decay of Squared Returns Autocorrelation', fontsize=12, pad=20)
        plt.xlabel('Lag', fontsize=10)
        plt.ylabel('Autocorrelation', fontsize=10)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../figures/volatility_decay_acf.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        # New plot for log-linear analysis
        plt.figure(figsize=(15, 10))
        
        # Log-linear analysis for absolute returns
        plt.subplot(2, 1, 1)
        log_acf_abs = np.log(acf_abs[1:])
        plt.errorbar(time_points, log_acf_abs, yerr=std_abs[1:]/acf_abs[1:],
                    fmt='bo-', markersize=6, linewidth=2,
                    label='Log ACF of Absolute Returns',
                    capsize=5)
        
        # Fit linear regression to log ACF
        slope_abs, intercept_abs, r_value_abs, p_value_abs, std_err_abs = stats.linregress(time_points, log_acf_abs)
        plt.plot(time_points, slope_abs * time_points + intercept_abs,
                'r--', label=f'Linear fit: {slope_abs:.3f}x + {intercept_abs:.3f}\nR² = {r_value_abs**2:.3f}')
        
        plt.title('Log-Linear Decay of Absolute Returns Autocorrelation', fontsize=12, pad=20)
        plt.xlabel('Lag', fontsize=10)
        plt.ylabel('Log Autocorrelation', fontsize=10)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Log-linear analysis for squared returns
        plt.subplot(2, 1, 2)
        log_acf_squared = np.log(acf_squared[1:])
        plt.errorbar(time_points, log_acf_squared, yerr=std_squared[1:]/acf_squared[1:],
                    fmt='ro-', markersize=6, linewidth=2,
                    label='Log ACF of Squared Returns',
                    capsize=5)
        
        # Fit linear regression to log ACF
        slope_squared, intercept_squared, r_value_squared, p_value_squared, std_err_squared = stats.linregress(time_points, log_acf_squared)
        plt.plot(time_points, slope_squared * time_points + intercept_squared,
                'b--', label=f'Linear fit: {slope_squared:.3f}x + {intercept_squared:.3f}\nR² = {r_value_squared**2:.3f}')
        
        plt.title('Log-Linear Decay of Squared Returns Autocorrelation', fontsize=12, pad=20)
        plt.xlabel('Lag', fontsize=10)
        plt.ylabel('Log Autocorrelation', fontsize=10)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../figures/volatility_decay_log_acf.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        # Print decay statistics
        print("\nVolatility Clustering Decay Analysis:")
        print("Absolute Returns ACF:")
        print(f"Initial decay rate (lag 1): {acf_abs[1]:.3f}")
        print(f"Half-life (lag where ACF < 0.5): {np.where(acf_abs[1:] < 0.5)[0][0] + 1 if np.any(acf_abs[1:] < 0.5) else 'N/A'}")
        print(f"Exponential decay rate: {b_abs:.3f}")
        print(f"Log-linear decay rate: {slope_abs:.3f}")
        print(f"Log-linear R²: {r_value_abs**2:.3f}")
        
        print("\nSquared Returns ACF:")
        print(f"Initial decay rate (lag 1): {acf_squared[1]:.3f}")
        print(f"Half-life (lag where ACF < 0.5): {np.where(acf_squared[1:] < 0.5)[0][0] + 1 if np.any(acf_squared[1:] < 0.5) else 'N/A'}")
        print(f"Exponential decay rate: {b_squared:.3f}")
        print(f"Log-linear decay rate: {slope_squared:.3f}")
        print(f"Log-linear R²: {r_value_squared**2:.3f}")

    def estimate_mean_reversion_speed(self, variances: np.ndarray, dt: float) -> float:
        """
        Estimate the speed of mean reversion from simulated variance paths.
        
        Args:
            variances: Array of variance paths
            dt: Time step size
            
        Returns:
            Estimated kappa (mean reversion rate)
        """
        # Calculate log variance changes
        dv = variances[:, 1:] - variances[:, :-1]
        v = variances[:, :-1]
        
        # Linear regression of dv/dt on (theta - v)
        X = self.params.theta - v.flatten()
        y = (dv/dt).flatten()
        
        # Estimate kappa using least squares
        kappa_est = np.sum(X * y) / np.sum(X**2)

        print(f"True mean reversion rate: f{self.params.kappa}")
        print(f"Extimated mean reversion rate from the simulated variance paths: f{kappa_est}")
        
        return kappa_est
    
    def analyze_leverage_effect(self, prices: np.ndarray, variances: np.ndarray, 
                                max_lag: int = 50) -> None:
        """
        Demonstrate the leverage effect by comuting correlation between current returns and future volatility squared.
        
        Args:
            prices: 2D array of prices (M x N+1) where M is number of paths
            variances: 2D array of variances (M x N+1)
            max_lag: Maximum number of time steps to look ahead
        """
        M, N_plus_1 = prices.shape
        N = N_plus_1 - 1
        
        # Calculate returns (log returns) for all paths
        returns = np.log(prices[:, 1:] / prices[:, :-1])  # Shape: (M, N)
        
        # Calculate volatility squared (variance)
        vol_squared = variances[:, :-1]  # Shape: (M, N)
        
        # Initialize array for correlations at different lags
        correlations = np.zeros(max_lag)
        correlation_std = np.zeros(max_lag)
        
        # Calculate correlation for each lag
        for lag in range(max_lag):
            lag_correlations = np.zeros(M)
            
            for m in range(M):
                # Get current returns and future volatility
                # current_returns = returns[m, :-lag-1] if lag > 0 else returns[m, :]
                # future_vol = vol_squared[m, lag-1:] if lag > 0 else vol_squared[m, 1:]
                current_returns = returns[m, :-(lag+1)]  # Remove last (lag+1) elements
                future_vol = vol_squared[m, (lag+1):]   # Remove first (lag+1) elements
                
                # Calculate correlation for this path
                if len(current_returns) > 1 and len(future_vol) > 1:
                    lag_correlations[m] = np.corrcoef(current_returns, future_vol)[0, 1]
            
            # Store mean and std of correlations across paths
            correlations[lag] = np.mean(lag_correlations)
            correlation_std[lag] = np.std(lag_correlations)
        
        # Create time points for x-axis (in terms of time steps)
        time_points = np.arange(max_lag)
        
        # Plot results
        plt.figure(figsize=(12, 8))
        
        # Plot correlation with confidence bands
        plt.plot(time_points, correlations, 'b-', 
                label='Mean Correlation', linewidth=2)
        plt.fill_between(time_points, 
                        correlations - 2*correlation_std,
                        correlations + 2*correlation_std,
                        alpha=0.2,
                        color='blue',
                        label='95% Confidence Interval')
        
        # Add horizontal line at true correlation (rho)
        plt.axhline(y=self.params.rho, color='r', linestyle='--',
                   label=f'True Correlation (ρ = {self.params.rho:.2f})')
        
        plt.title('Leverage Effect: Correlation between Returns and Future Volatility', 
                 fontsize=12, pad=20)
        plt.xlabel('Time Lag (steps)', fontsize=10)
        plt.ylabel('Correlation', fontsize=10)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('../figures/leverage_corr.png', dpi=300, bbox_inches='tight')
        
        # Add scatter plot for lag=0 (contemporaneous correlation)
        plt.figure(figsize=(10, 6))
        sample_path = 0  # Use first path for visualization
        plt.scatter(returns[sample_path], vol_squared[sample_path], 
                   alpha=0.5, label='Returns vs Volatility²')
        
        plt.title('Leverage Effect: Returns vs Volatility² (Sample Path)', fontsize=12, pad=20)
        plt.xlabel('Returns', fontsize=10)
        plt.ylabel('Volatility²', fontsize=10)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        # plt.savefig('../figures/leverage_lead_lag.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        # Print statistics
        print("\nLeverage Effect Lead-Lag Analysis:")
        print(f"True correlation parameter (ρ): {self.params.rho:.3f}")
        print(f"Maximum correlation: {np.min(correlations):.3f} at lag {np.argmin(correlations)}")
        print(f"Correlation at lag 0: {correlations[0]:.3f}")
        print(f"Correlation at lag {max_lag-1}: {correlations[-1]:.3f}") 