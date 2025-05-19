import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import statsmodels.api as sm
import os

class VolatilityVisualizer:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize visualizer with data containing returns and volatility.
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame containing returns and volatility
        """
        self.data = data
        # Set style
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
        # Create plots directory if it doesn't exist
        os.makedirs('plots', exist_ok=True)
        
    def plot_returns_and_volatility(
        self,
        figsize: Tuple[int, int] = (15, 12),
        true_vol: bool = True,
        estimated_vol: bool = True,
        title_prefix: str = ""
    ) -> None:
        """
        Plot returns and volatility in a subplot format.
        
        Parameters:
        -----------
        figsize : Tuple[int, int]
            Figure size
        true_vol : bool
            Whether to plot true volatility
        estimated_vol : bool
            Whether to plot estimated volatility
        title_prefix : str
            Prefix for the plot title and filename
        """
        # Determine if we have true volatility (simulated data) or not (market data)
        has_true_volatility = 'volatility' in self.data.columns
        
        # For market data, we only need two subplots instead of three
        if has_true_volatility:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize)
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Plot returns
        ax1.plot(self.data.index, self.data['returns'], 
                label='Returns', alpha=0.7, color='blue')
        ax1.set_title('Returns Time Series', fontsize=12, pad=10)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Returns')
        ax1.legend()
        
        # Plot volatilities
        if has_true_volatility and true_vol:
            ax2.plot(self.data.index, self.data['volatility'],
                    label='True Volatility', alpha=0.7, color='green')
            ax2.set_title('True (Simulated) Volatility\nThis is what we used to generate the returns', 
                         fontsize=12, pad=10)
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Volatility')
            ax2.legend()
            
            if estimated_vol and 'conditional_volatility' in self.data.columns:
                # Plot estimated volatility with comparison
                ax3.plot(self.data.index, self.data['conditional_volatility'],
                        label='Estimated Volatility', alpha=0.7, color='red')
                ax3.plot(self.data.index, self.data['volatility'],
                        label='True Volatility', alpha=0.4, color='green', linestyle='--')
                
                # Calculate and plot estimation error
                error = np.abs(self.data['conditional_volatility'] - self.data['volatility'])
                ax3.fill_between(self.data.index, 
                               self.data['conditional_volatility'] - error,
                               self.data['conditional_volatility'] + error,
                               color='red', alpha=0.2, 
                               label='Estimation Error Band')
                
                ax3.set_title('Estimated vs True Volatility\nComparing GARCH model estimates with actual volatility',
                             fontsize=12, pad=10)
                ax3.set_xlabel('Date')
                ax3.set_ylabel('Volatility')
                ax3.legend()
        else:
            # For market data, only plot estimated volatility
            if estimated_vol and 'conditional_volatility' in self.data.columns:
                ax2.plot(self.data.index, self.data['conditional_volatility'],
                        label='Estimated Volatility', alpha=0.7, color='red')
                ax2.set_title('GARCH Estimated Volatility', fontsize=12, pad=10)
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Volatility')
                ax2.legend()
        
        plt.tight_layout()
        # Save the plot
        plt.savefig(f'plots/{title_prefix}_returns_volatility.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print comparison statistics only for simulated data
        if has_true_volatility and true_vol and estimated_vol and 'conditional_volatility' in self.data.columns:
            error = self.data['conditional_volatility'] - self.data['volatility']
            mae = np.abs(error).mean()
            rmse = np.sqrt((error ** 2).mean())
            correlation = self.data['volatility'].corr(self.data['conditional_volatility'])
            
            print("\nVolatility Estimation Statistics:")
            print(f"Mean Absolute Error: {mae:.6f}")
            print(f"Root Mean Square Error: {rmse:.6f}")
            print(f"Correlation between True and Estimated: {correlation:.6f}")
        elif estimated_vol and 'conditional_volatility' in self.data.columns:
            # For market data, print basic volatility statistics
            vol = self.data['conditional_volatility']
            print("\nEstimated Volatility Statistics:")
            print(f"Mean: {vol.mean():.6f}")
            print(f"Std Dev: {vol.std():.6f}")
            print(f"Min: {vol.min():.6f}")
            print(f"Max: {vol.max():.6f}")
            print(f"Median: {vol.median():.6f}")
        
    def plot_volatility_clustering(self, lags: int = 20, title_prefix: str = "") -> None:
        """
        Plot ACF of squared returns to show volatility clustering.
        
        Parameters:
        -----------
        lags : int
            Number of lags to plot
        title_prefix : str
            Prefix for the plot title and filename
        """
        squared_returns = self.data['returns'] ** 2
        
        fig, ax = plt.subplots(figsize=(12, 8))
        pd.plotting.autocorrelation_plot(squared_returns, ax=ax)
        ax.set_xlim([0, lags])
        
        ax.set_title('Autocorrelation of Squared Returns\nEvidence of Volatility Clustering', 
                    fontsize=12, pad=10)
        ax.set_xlabel('Lag (Number of periods)')
        ax.set_ylabel('Autocorrelation')
        
        # Add explanation text
        explanation = (
            "• Solid black line: Zero correlation reference\n"
            "• Solid gray line: 95% confidence bounds\n"
            "• Dashed gray line: 99% confidence bounds\n"
            "• Bars beyond bounds indicate significant volatility clustering\n"
            "• More bars beyond bounds = stronger evidence of clustering"
        )
        plt.figtext(0.99, 0.02, explanation, ha='right', fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        plt.tight_layout()
        # Save the plot
        plt.savefig(f'plots/{title_prefix}_volatility_clustering.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate and print statistics using 95% confidence bound
        n = len(squared_returns)
        conf_bound_95 = 1.96/np.sqrt(n)  # 95% confidence bound
        
        significant_lags = sum(1 for i in range(1, lags+1) 
                             if abs(pd.Series(squared_returns).autocorr(lag=i)) > conf_bound_95)
        first_lag_acf = pd.Series(squared_returns).autocorr(lag=1)
        
        print(f"\nVolatility Clustering Statistics:")
        print(f"Number of significant lags: {significant_lags} out of {lags}")
        print(f"First lag autocorrelation: {first_lag_acf:.4f}")
        print(f"95% Confidence bound: ±{conf_bound_95:.4f}")
        if abs(first_lag_acf) > conf_bound_95:
            print("Strong evidence of volatility clustering (significant first-lag autocorrelation)")
        else:
            print("Weak or no evidence of volatility clustering")
        
    def plot_returns_distribution(self, title_prefix: str = "") -> None:
        """
        Plot the distribution of returns against normal distribution.
        
        Parameters:
        -----------
        title_prefix : str
            Prefix for the plot title and filename
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot histogram of returns
        sns.histplot(data=self.data['returns'], stat='density', 
                    alpha=0.5, label='Returns', ax=ax)
        
        # Plot normal distribution
        x = np.linspace(self.data['returns'].min(), self.data['returns'].max(), 100)
        mu = self.data['returns'].mean()
        sigma = self.data['returns'].std()
        normal_dist = (1/(sigma * np.sqrt(2 * np.pi))) * \
                     np.exp(-(x - mu)**2 / (2 * sigma**2))
        
        ax.plot(x, normal_dist, 'r-', label='Normal Distribution')
        ax.set_title('Returns Distribution vs Normal Distribution')
        ax.legend()
        plt.tight_layout()
        # Save the plot
        plt.savefig(f'plots/{title_prefix}_returns_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_qq_plot(self, title_prefix: str = "") -> None:
        """
        Create Q-Q plot of standardized residuals.
        
        Parameters:
        -----------
        title_prefix : str
            Prefix for the plot title and filename
        """
        if 'conditional_volatility' not in self.data.columns:
            raise ValueError("Conditional volatility required for Q-Q plot")
            
        # Calculate standardized residuals
        std_residuals = self.data['returns'] / self.data['conditional_volatility']
        
        fig, ax = plt.subplots(figsize=(10, 10))
        sm.graphics.qqplot(std_residuals, line='45', fit=True, ax=ax)
        ax.set_title('Q-Q Plot of Standardized Residuals')
        plt.tight_layout()
        # Save the plot
        plt.savefig(f'plots/{title_prefix}_qq_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_shock_analysis(self, shock_time: pd.Timestamp, window: int = 50, title_prefix: str = "") -> None:
        """
        Analyze and visualize the impact of a volatility shock.
        
        Parameters:
        -----------
        shock_time : pd.Timestamp
            Time when the shock occurred
        window : int
            Number of periods to analyze before and after the shock
        title_prefix : str
            Prefix for the plot title and filename
        """
        # Get data around the shock
        start_time = shock_time - pd.Timedelta(days=window)
        end_time = shock_time + pd.Timedelta(days=window)
        
        shock_data = self.data[start_time:end_time].copy()
        shock_idx = shock_data.index.get_loc(shock_time)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Plot returns around shock
        ax1.plot(shock_data.index, shock_data['returns'], 
                label='Returns', color='blue', alpha=0.7)
        ax1.axvline(x=shock_time, color='red', linestyle='--', 
                   label='Shock Time')
        ax1.set_title('Returns Around Volatility Shock', fontsize=12, pad=10)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Returns')
        ax1.legend()
        
        # Plot volatility around shock
        if 'volatility' in shock_data.columns:
            ax2.plot(shock_data.index, shock_data['volatility'],
                    label='True Volatility', color='green', alpha=0.7)
        if 'conditional_volatility' in shock_data.columns:
            ax2.plot(shock_data.index, shock_data['conditional_volatility'],
                    label='Estimated Volatility', color='red', alpha=0.7)
        ax2.axvline(x=shock_time, color='red', linestyle='--', 
                   label='Shock Time')
        ax2.set_title('Volatility Around Shock', fontsize=12, pad=10)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Volatility')
        ax2.legend()
        
        plt.tight_layout()
        # Save the plot
        plt.savefig(f'plots/{title_prefix}_shock_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate and print shock impact statistics
        pre_shock = shock_data[:shock_time]
        post_shock = shock_data[shock_time:]
        
        print("\nShock Impact Analysis:")
        print("Pre-shock statistics:")
        print(f"Mean volatility: {pre_shock['volatility'].mean():.4f}")
        print(f"Max volatility: {pre_shock['volatility'].max():.4f}")
        
        print("\nPost-shock statistics:")
        print(f"Mean volatility: {post_shock['volatility'].mean():.4f}")
        print(f"Max volatility: {post_shock['volatility'].max():.4f}")
        print(f"Peak-to-pre-shock ratio: {post_shock['volatility'].max() / pre_shock['volatility'].mean():.2f}")
        
        # Calculate half-life of shock
        peak_vol = post_shock['volatility'].max()
        pre_shock_mean = pre_shock['volatility'].mean()
        half_effect = (peak_vol - pre_shock_mean) / 2 + pre_shock_mean
        
        days_to_half = None
        for i, vol in enumerate(post_shock['volatility']):
            if vol <= half_effect:
                days_to_half = i
                break
                
        if days_to_half is not None:
            print(f"\nShock half-life: {days_to_half} days")
        else:
            print("\nShock effect persisted beyond the analysis window") 