import numpy as np
import matplotlib.pyplot as plt
from src.fractional_random_walk import FractionalRandomWalk

class PriceReturns:
    def __init__(self, N, H_val=0.1, sigma=1.0, gamma=0.3):
        """
        Initialize the Price Returns model with rough volatility.
        
        Args:
            N (int): Number of time steps
            H_val (float): Hurst parameter for volatility process, typically 0.1 for rough volatility
            sigma (float): Base volatility level
            gamma (float): Vol of vol parameter
        """
        self.N = N
        self.H_val = H_val
        self.sigma = sigma
        self.gamma = gamma
        self.frw = None
        self.eta = None
        self.eta_var = None
        self.volatility = None
        self.returns = None
        self.prices = None
        self.phi = None  # Standard normal innovations
        
    def get_eta_series(self):
        """
        Get the eta series from the FractionalRandomWalk class methods.
        """
        self.frw = FractionalRandomWalk(self.N, self.H_val)
        self.eta = self.frw.generate_eta_series()
        self.eta_var = np.mean(self.eta**2)
        return self.eta
    
    def generate_returns(self):
        """
        Generate price returns according to the model:
        rt = σ * φt * exp(γ * ηt - γ² * E[ηt²])
        
        Returns:
            numpy.ndarray: Generated returns
        """
        if self.eta is None:
            self.get_eta_series()
            
        # Generate standard normal innovations
        self.phi = np.random.normal(0, 1, self.N)
        
        # Compute volatility adjustment term
        vol_adjustment = self.gamma**2 * self.eta_var
        
        # Compute log volatility
        log_vol = self.gamma * self.eta - vol_adjustment
        
        self.volatility = self.sigma * np.exp(log_vol) 
        
        # Finally generate returns
        self.returns = self.volatility * self.phi
        return self.returns
    
    def generate_prices(self):
        """
        Generate cumulative price process from returns.
        
        Returns:
            numpy.ndarray: Generated price data from return data
        """
        if self.returns is None:
            self.generate_returns()

        self.prices = np.cumsum(self.returns)
        return self.prices
    
    def verify_return_variance(self, n_simulations=1000):
        """
        Verify that E[rt²] = σ² by averaging over several realizations
        """
        total_var = 0
        for _ in range(n_simulations):
            returns = self.generate_returns()
            total_var += np.mean(returns**2)
            
        empirical_var = total_var / n_simulations
        theoretical_var = self.sigma**2
        
        print(f"Empirical variance: {empirical_var:.6f}")
        print(f"Theoretical variance: {theoretical_var:.6f}")
        print(f"Relative error: {abs(empirical_var - theoretical_var)/theoretical_var:.6f}")
        
    def compute_abs_return_correlation(self, max_lag=None):
        """
        Compute correlation function of absolute returns D(t-s) = E[|rt*rs|]-E[|rt|]².
        """
        if max_lag is None:
            max_lag = self.N // 4
            
        if self.returns is None:
            self.generate_returns()
            
        abs_returns = np.abs(self.returns)
        mean_abs_return = np.mean(abs_returns)
        
        # Compute correlations
        lags = np.arange(max_lag)
        corr = np.zeros(max_lag)
        
        for lag in lags:
            if lag == 0:
                corr[lag] = 0.0
            else:
                corr[lag] = np.mean(abs_returns[:-lag] * abs_returns[lag:]) - mean_abs_return**2
                
        return lags, corr
    
    def compute_price_variogram(self, q_values=None, max_lag=None):
        """
        Compute price variograms of different powers: Wq(t-s) = E[|pt - ps|^q].
        
        Args:
            q_values (list): List of moment orders to analyze
            max_lag (int): Maximum lag to compute
        """
        if q_values is None:
            q_values = [2, 3, 4]
            
        if max_lag is None:
            max_lag = self.N // 4
            
        if self.prices is None:
            self.generate_prices()
            
        lags = np.arange(1, max_lag)
        variograms = {q: np.zeros(len(lags)) for q in q_values}
        
        for i, lag in enumerate(lags):
            price_differences = self.prices[lag:] - self.prices[:-lag]
            
            for q in q_values:
                variograms[q][i] = np.mean(np.abs(price_differences)**q)
                
        return lags, variograms
    
    def analyze_drawdowns(self):
        """
        Analyze the largest drawdowns in the price series.
        """
        if self.prices is None:
            self.generate_prices()
            
        # Track running maximum
        running_max = np.maximum.accumulate(self.prices)
        
        # Calculate drawdowns. This gives values between 0 (no drawdown) and 1 (total loss)
        drawdowns = running_max - self.prices
        
        # Find largest drawdown
        max_drawdown = np.max(drawdowns)
        max_drawdown_idx = np.argmax(drawdowns)
        
        # Find the start of the drawdown period
        start_idx = np.where(self.prices[:max_drawdown_idx] == running_max[max_drawdown_idx])[0][-1]

        return {
            'max_drawdown': max_drawdown,
            'start_idx': start_idx,
            'bottom_idx': max_drawdown_idx,
            'drawdown_length': max_drawdown_idx - start_idx,
            'drawdown_series': drawdowns
        }
    
    def plot_return_distribution(self):
        """
        Analyze and plot the distribution of returns.
        """
        if self.returns is None:
            self.generate_returns()
            
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Return distribution
        plt.subplot(1, 2, 1)
        plt.hist(self.returns, bins=50, density=True, alpha=0.7)
        
        # Add normal distribution for comparison
        x = np.linspace(min(self.returns), max(self.returns), 100)
        plt.plot(x, 1/np.sqrt(2*np.pi*self.sigma**2) * np.exp(-x**2/(2*self.sigma**2)), 
                'r--', label='Normal')
        
        plt.title(f'Return Distribution (γ={self.gamma})')
        plt.xlabel('Return')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: QQ plot
        plt.subplot(1, 2, 2)
        from scipy import stats
        stats.probplot(self.returns, dist="norm", plot=plt)
        plt.title('Q-Q Plot vs Normal')
        
        plt.tight_layout()
        plt.show()
        
    def plot_price_process(self):
        """
        Plot the price process and its components.
        """
        if self.prices is None:
            self.generate_prices()
            
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot volatility
        axes[0].plot(self.volatility)
        axes[0].set_title(f'Stochastic Volatility (H={self.H_val}, γ={self.gamma})')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('σ(t)')
        axes[0].grid(True)
        
        # Plot returns
        axes[1].plot(self.returns)
        axes[1].set_title('Returns')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('r(t)')
        axes[1].grid(True)
        
        # Plot prices
        axes[2].plot(self.prices)
        axes[2].set_title('Price Process')
        axes[2].set_xlabel('Time')
        axes[2].set_ylabel('S(t)')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.show() 

    def plot_abs_return_correlation(self, max_lag=None):
        """
        Plot the correlation function of absolute returns and its scaling behavior.
        
        Args:
            max_lag (int, optional): Maximum lag to compute. Defaults to N//4.
        """
        lags, corr = self.compute_abs_return_correlation(max_lag)
        
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Regular correlation vs lag
        plt.subplot(1, 2, 1)
        plt.plot(lags[1:], corr[1:], 'b-', label='Empirical')
        plt.title('Absolute Return Correlation')
        plt.xlabel('Lag')
        plt.ylabel('D(t-s)')
        plt.grid(True)
        plt.legend()
        
        # Plot 2: Correlation vs lag^(2H) to check scaling
        plt.subplot(1, 2, 2)
        nonzero_lags = lags[1:]
        plt.plot(nonzero_lags**(2*self.H_val), corr[1:], 'b.', label='Empirical')
        plt.title('Correlation vs lag^(2H)')
        plt.xlabel('lag^(2H)')
        plt.ylabel('D(t-s)')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_price_variogram(self, q_values=None, max_lag=None):
        """
        Plot price variograms for different powers and their scaling behavior.
        
        Args:
            q_values (list, optional): List of moment orders to analyze. Defaults to [2,3,4].
            max_lag (int, optional): Maximum lag to compute. Defaults to N//4.
        """
        lags, variograms = self.compute_price_variogram(q_values, max_lag)
        
        plt.figure(figsize=(15, 5))
        colors = ['b', 'r', 'g']  # Different colors for different q values
        
        # Plot 1: Raw variograms
        # plt.subplot(1, 3, 1)
        for i, q in enumerate(variograms.keys()):
            plt.plot(lags, variograms[q], f'{colors[i]}-', label=f'q={q}')
        plt.title('Price Variograms')
        plt.xlabel('Lag')
        plt.ylabel('Wq(t-s)')
        plt.grid(True)
        plt.legend()
        
        # # Plot 2: Normalized variograms
        # plt.subplot(1, 3, 2)
        # for i, q in enumerate(variograms.keys()):
        #     # Normalize by the first value to compare shapes
        #     norm_variogram = variograms[q] / variograms[q][0]
        #     plt.loglog(lags, norm_variogram, f'{colors[i]}-', label=f'q={q}')
        # plt.title('Normalized Variograms')
        # plt.xlabel('Lag')
        # plt.ylabel('Wq(t-s)/Wq(1)')
        # plt.grid(True)
        # plt.legend()
        
        # Plot 3: Scaling exponents
        # plt.subplot(1, 3, 3)
        # q_values = list(variograms.keys())
        # scaling_exponents = []
        
        # for q in q_values:
        #     # Fit power law in log-log space
        #     log_lags = np.log(lags)
        #     log_var = np.log(variograms[q])
        #     coeffs = np.polyfit(log_lags, log_var, 1)
        #     scaling_exponents.append(coeffs[0])
        
        # plt.plot(q_values, scaling_exponents, 'bo-', label='Empirical')
        # plt.plot(q_values, [q/2 for q in q_values], 'r--', label='Brownian')
        # plt.title('Scaling Exponents')
        # plt.xlabel('q')
        # plt.ylabel('ζ(q)')
        # plt.grid(True)
        # plt.legend()
        
        # plt.tight_layout()
        plt.show()
        
        # return scaling_exponents
    
    def compute_autocorrelation(self, max_lag=None):
        """
        Compute autocorrelation function for either returns or prices.
        
        Args:
            max_lag (int, optional): Maximum lag to compute. Defaults to N//4.
            process (str): Either 'returns' or 'prices'. Defaults to 'returns'.
            
        Returns:
            tuple: (lags, autocorrelation values)
        """
        if max_lag is None:
            max_lag = self.N // 4
            
        
        if self.returns is None:
            self.generate_returns()
            
        
        # Compute variance for normalization
        data_squared = self.returns**2 
        mean_data_squared = np.mean(data_squared)
        variance_data_squared = np.var(data_squared)

        data_abs = np.abs(self.returns)
        mean_data_abs = np.mean(data_abs)
        variance_data_abs = np.var(data_abs)
        
        # Compute autocorrelations
        lags = np.arange(max_lag)
        squared_acf = np.zeros(max_lag)
        abs_acf = np.zeros(max_lag)
        
        for lag in lags[1:]:
            squared_acf[lag] = (np.mean(data_squared[:-lag] * data_squared[lag:]) - mean_data_squared**2) / variance_data_squared
            abs_acf[lag] = (np.mean(data_abs[:-lag] * data_abs[lag:]) - mean_data_abs**2) / variance_data_abs
                
        return lags[1:], squared_acf[1:], abs_acf[1:]
        
    def plot_autocorrelation(self, max_lag=None):
        """
        Plot autocorrelation functions for both returns and prices.
        
        Args:
            max_lag (int, optional): Maximum lag to compute. Defaults to N//4.
        """
        if max_lag is None:
            max_lag = self.N // 4
            
        # Compute autocorrelations
        lags_r, acf_r, acf_r1 = self.compute_autocorrelation(max_lag)
        
        plt.figure(figsize=(12, 5))
        
        # Plot squared returns autocorrelation
        plt.subplot(1, 2, 1)
        plt.plot(lags_r, acf_r, 'b-', label='Empirical')
        plt.axhline(y=0, color='r', linestyle='--')
        # Add confidence bands (±1.96/√N)
        conf_level = 1.96 / np.sqrt(self.N)
        plt.axhline(y=conf_level, color='gray', linestyle=':')
        plt.axhline(y=-conf_level, color='gray', linestyle=':')
        plt.title('Squared Returns Autocorrelation')
        plt.xlabel('Lag')
        plt.ylabel('ACF')
        plt.grid(True)
        plt.legend()
        
        # Plot abs returns autocorrelation
        plt.subplot(1, 2, 2)
        plt.plot(lags_r, acf_r1, 'r-', label='Empirical')
        plt.axhline(y=0, color='b', linestyle='--')
        plt.axhline(y=conf_level, color='gray', linestyle=':')
        plt.axhline(y=-conf_level, color='gray', linestyle=':')
        plt.title('Absolute Returns Autocorrelation')
        plt.xlabel('Lag')
        plt.ylabel('ACF')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.show()