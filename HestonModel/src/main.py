import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List
from heston_model import HestonModel, HestonParameters
from heston_pricing import HestonPricer, BlackScholesPricer, test_pricing

def setup_model(params: HestonParameters = None) -> Tuple[HestonModel, HestonPricer]:
    """Set up the Heston model and pricer with default or custom parameters."""
    if params is None:
        params = HestonParameters(
            kappa=1.2,    # Mean reversion
            theta=0.10,   # Long-term variance
            sigma=0.3,    # Volatility of variance
            rho=-0.8,     # Correlation
            v0=0.10,      # Initial variance
            mu=0.1        # Drift rate
        )
    model = HestonModel(params)
    pricer = HestonPricer(model)
    return model, pricer

def plot_price_paths(model: HestonModel, S0: float, T: float, n_paths: int = 5, n_steps: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Plot sample price paths from the Heston model."""
    prices, vols = model.simulate_paths(S0, T, n_steps, n_paths, return_vol=True)
    time = np.linspace(0, T, n_steps + 1)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    for i in range(n_paths):
        plt.plot(time, prices[i], label=f'Path {i+1}')
    plt.title('Stock Price Paths')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    for i in range(n_paths):
        plt.plot(time, np.sqrt(vols[i]), label=f'Path {i+1}')
    plt.title('Volatility Paths')
    plt.xlabel('Time')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('../figures/price_and_vol_paths.png')
    plt.show()
    plt.close()


def plot_implied_volatility_surface(pricer: HestonPricer, S0: float, T_range: List[float], 
                                  K_range: List[float], r: float):
    """Plot the implied volatility surface for different strikes and maturities."""
    T_grid, K_grid = np.meshgrid(T_range, K_range)
    iv_surface = np.zeros_like(T_grid)
    
    for i, T in enumerate(T_range):
        for j, K in enumerate(K_range):
            price, _ = pricer.price_option_CF(S0, K, T, r, 10000, 'call')
            iv_surface[j, i] = pricer.implied_volatility(price, S0, K, T, r, 'call')
    
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(T_grid, K_grid, iv_surface, cmap='viridis')
    plt.colorbar(surf)
    ax.set_xlabel('Time to Maturity')
    ax.set_ylabel('Strike Price')
    ax.set_zlabel('Implied Volatility')
    plt.title('Implied Volatility Surface')
    plt.savefig('../figures/implied_volatility_surface.png')
    plt.show()
    plt.close()

def compare_pricing_methods(pricer: HestonPricer, S0: float, K: float, T: float, r: float,
                          n_paths_list: List[int] = [10000, 50000, 100000]):
    """Compare different pricing methods and generate comparison plots."""
    # Get prices from different methods
    cf_call, cf_err = pricer.price_option_CF(S0, K, T, r, 10000, 'call')
    cf_put, _ = pricer.price_option_CF(S0, K, T, r, 10000, 'put')
    
    mc_prices = []
    mc_errors = []
    for n_paths in n_paths_list:
        mc_call, mc_err = pricer.price_option_MC(S0, K, T, r, 'call', n_paths=n_paths)
        mc_prices.append(mc_call)
        mc_errors.append(abs(mc_err))  # Use absolute value for error bars
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.errorbar(n_paths_list, mc_prices, yerr=mc_errors, label='Monte Carlo', marker='o')
    plt.axhline(y=cf_call, color='r', linestyle='--', label='Characteristic Function')
    plt.xlabel('Number of Paths')
    plt.ylabel('Call Option Price')
    plt.title('Comparison of Pricing Methods')
    plt.legend()
    plt.grid(True)
    plt.savefig('../figures/pricing_methods_comparison.png')
    plt.show()
    plt.close()

def analyze_volatility_clustering_single_instance(prices: np.ndarray, max_lag: int = 50):
    """
    Analyze volatility clustering through non-linear autocorrelations the return series.
    
    Args:
        prices: 1D array of prices (length N+1)
        max_lag: Max number of lags for autocorrelation (default 50)
        
    Returns:
        Tuple of (lags, autocorrelations) for both absolute and squared returns
    """
    # Calculate returns (log returns)
    returns = np.log(prices[1:] / prices[:-1])  # Shape: (N,)
    
    # Calculate absolute and squared returns
    abs_returns = np.abs(returns)
    squared_returns = returns ** 2

    abs_returns_mean = np.mean(abs_returns)
    squared_returns_mean = np.mean(squared_returns)

    abs_returns_centered = abs_returns - abs_returns_mean
    squared_returns_centered = squared_returns - squared_returns_mean

    abs_returns_var = np.var(abs_returns)
    squared_returns_var = np.var(squared_returns)

    lags = np.arange(max_lag)
    
    # Calculate autocorrelation for both measures
    acf_abs = np.zeros(max_lag)
    acf_squared = np.zeros(max_lag)

    for lag in lags:
        if lag == 0:
            acf_abs[lag] = 1.0
        else:
            acf_abs[lag] = np.mean(abs_returns_centered[lag:] * abs_returns_centered[:-lag]) / abs_returns_var
            acf_squared[lag] = np.mean(squared_returns_centered[lag:] * squared_returns_centered[:-lag]) / squared_returns_var
        
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # ACF of absolute returns
    plt.subplot(2, 1, 1)
    lags_to_plot = lags[1:]
    conf_level = 1.96/np.sqrt(len(abs_returns.flatten()))
    
    plt.fill_between(lags_to_plot, 
                     -conf_level, 
                     conf_level,
                     alpha=0.4,
                     color='orange',
                     label='95% Confidence Interval')
    plt.plot(lags_to_plot, acf_abs[1:], 'bo-', markersize=6, linewidth=2, label='ACF of Absolute Returns')
    plt.title('Autocorrelation of Absolute Returns', fontsize=12, pad=20)
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
    plt.plot(lags_to_plot, acf_squared[1:], 'ro-', markersize=6, linewidth=2, label='ACF of Squared Returns')
    plt.title('Autocorrelation of Squared Returns', fontsize=12, pad=20)
    plt.xlabel('Lag', fontsize=10)
    plt.ylabel('Autocorrelation', fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../figures/volatility_clustering_single_instance_acf.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Print statistics
    print("\nVolatility Clustering Analysis:")
    print(f"First-order autocorrelation of absolute returns: {acf_abs[1]:.3f}")
    print(f"First-order autocorrelation of squared returns: {acf_squared[1]:.3f}")
    print(f"Mean autocorrelation of absolute returns (lags 1-10): {np.mean(acf_abs[1:11]):.3f}")
    print(f"Mean autocorrelation of squared returns (lags 1-10): {np.mean(acf_squared[1:11]):.3f}")
    
def study_mean_reversion(model, T, N, variances):
    """Study and plot mean reversion characteristics."""
    # Analyze decay from different initial levels
    initial_levels = [0.01, 0.04, 0.09]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Better color scheme
    
    plt.figure(figsize=(12, 6))
    time_points = np.linspace(0, 1.0, 251)
    
    for v0, color in zip(initial_levels, colors):
        # Create temporary model with new initial variance
        params_temp = HestonParameters(
            kappa=model.params.kappa,
            theta=model.params.theta,
            sigma=model.params.sigma,
            rho=model.params.rho,
            v0=v0,
            mu=model.params.mu
        )
        model_temp = HestonModel(params_temp)
        
        # Simulate paths
        _, var_paths = model_temp.simulate_paths(100, T=1.0, N=250, M=1000, return_vol=True)
        
        # Calculate mean variance path
        mean_var = np.mean(var_paths, axis=0)
        std_var = np.std(var_paths, axis=0)
        
        # Plot mean and confidence bands
        plt.plot(time_points, mean_var, color=color, label=f'v0 = {v0:.2f}')
        plt.fill_between(time_points, 
                        mean_var - 2*std_var,
                        mean_var + 2*std_var,
                        alpha=0.2,
                        color=color)
    
    # Add reference line for long-term variance
    plt.axhline(y=model.params.theta, color='red', linestyle='--',
                label='Long-term variance (θ)')
    
    plt.title('Variance Mean Reversion')
    plt.xlabel('Time')
    plt.ylabel('Variance')
    plt.legend()
    plt.grid(True)
    plt.savefig('../figures/mean_reversion.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Estimate mean reversion speed
    dt = T/N
    kappa_est = model.estimate_mean_reversion_speed(variances, dt)
    print(f"\nMean Reversion Analysis:")
    print(f"Estimated kappa: {kappa_est:.3f} (True kappa: {model.params.kappa:.3f})")
    print("Mean reversion plot saved as 'figures/mean_reversion.png'")

def analyze_parameter_sensitivity(base_params):
    """Analyze and plot sensitivity to different parameters."""
    def plot_sensitivity(base_params, param_name, values):
        plt.figure(figsize=(12, 6))
        
        colors = sns.color_palette("husl", len(values))
        for value, color in zip(values, colors):
            # Create new parameters with modified value
            param_dict = base_params.__dict__.copy()
            param_dict[param_name] = value
            new_params = HestonParameters(**param_dict)
            model = HestonModel(new_params)
            
            # Simulate and plot
            _, variances = model.simulate_paths(100, T=1.0, N=250, M=500, return_vol=True)
            mean_vol = np.sqrt(np.mean(variances, axis=0))
            time_points = np.linspace(0, 1.0, 251)
            
            plt.plot(time_points, mean_vol, label=f'{param_name}={value:.2f}', color=color)
        
        plt.title(f'Sensitivity to {param_name}')
        plt.xlabel('Time')
        plt.ylabel('Mean Volatility')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'../figures/sensitivity_{param_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        print(f"Parameter sensitivity plot for {param_name} saved as 'figures/sensitivity_{param_name}.png'")
    
    print("\nAnalyzing parameter sensitivity...")
    # Test different parameter values
    plot_sensitivity(base_params, 'kappa', [1.0, 2.0, 3.0])
    plot_sensitivity(base_params, 'theta', [0.02, 0.04, 0.06])
    plot_sensitivity(base_params, 'sigma', [0.2, 0.3, 0.4])

def main():
    # Default parameters
    S0 = 100.0
    K = 100.0
    T = 5.0
    r = 0.05
    
    # Set up model
    model, pricer = setup_model()

    # Verify Feller condition
    feller_value = 2 * model.params.kappa * model.params.theta
    sigma_squared = model.params.sigma ** 2
    print("\nFeller Condition Analysis:")
    print(f"2κθ = {feller_value:.3f}")
    print(f"σ² = {sigma_squared:.3f}")
    print(f"Feller condition {'satisfied' if feller_value > sigma_squared else 'not satisfied'}")
    print(f"Margin: {feller_value - sigma_squared:.3f}")
    print("\nParameter Interpretation:")
    print(f"- κ = {model.params.kappa:.2f}: Moderate mean reversion for persistence and stability")
    print(f"- θ = {model.params.theta:.3f}: Long-term variance (corresponds to {np.sqrt(model.params.theta)*100:.1f}% volatility)")
    print(f"- σ = {model.params.sigma:.2f}: Volatility of variance")
    print(f"- ρ = {model.params.rho:.1f}: Strong negative correlation captures leverage effect")

    # generate paths
    print("Generating price and volatility paths...")
    plot_price_paths(model, S0, T)

    prices, variances = model.simulate_paths(S0, T, 1250, 1000, return_vol=True)

    # Check for any negative variances in simulation
    neg_var_count = np.sum(variances < 0)
    if neg_var_count > 0:
        print(f"\nWarning: {neg_var_count} negative variance values detected in simulation")
        print(f"Percentage of negative values: {100 * neg_var_count / variances.size:.4f}%")
    else:
        print("\nNo negative variance values detected in simulation")

    # ACF of absolute and squared returns
    analyze_volatility_clustering_single_instance(prices=prices[25], max_lag=220)

    # ACF of absolute and squared returns averaged over all paths
    model.analyze_volatility_clustering(prices, 150)

    # Fit the above ACF with an exponential decay
    model.plot_volatility_decay(prices, T, N=1000, lags=200)

    # Study and plot mean reversion characteristics
    study_mean_reversion(model, T, N=1000, variances=variances)
    model.estimate_mean_reversion_speed(variances, dt=T/1000)

    # Analyze and plot sensitivity to different parameters
    analyze_parameter_sensitivity(model.params)

    # Pricing
    test_pricing()
    
    print("Comparing pricing methods...")
    compare_pricing_methods(pricer, S0, K, T, r)

    # generate Volatility surface
    print("Generating implied volatility surface...")
    T_range = np.linspace(0.1, 2.0, 10)
    K_range = np.linspace(80, 120, 10)
    plot_implied_volatility_surface(pricer, S0, T_range, K_range, r)

if __name__ == "__main__":
    main() 