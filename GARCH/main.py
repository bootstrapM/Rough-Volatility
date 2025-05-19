import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from simulate_data import generate_sample_data
from garch_analysis import GARCHAnalyzer
from visualization import VolatilityVisualizer

def get_market_data(symbol: str = "^GSPC", years: int = 4) -> pd.DataFrame:
    """
    Fetch market data from Yahoo Finance and prepare it for GARCH analysis.
    
    Parameters:
    -----------
    symbol : str
        Yahoo Finance ticker symbol (default: ^GSPC for S&P 500)
    years : int
        Number of years of historical data to fetch
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing returns and prices
    """
    # Calculate start date
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)
    
    # Fetch data
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date)
    
    # Calculate returns and handle NaN values
    data['returns'] = data['Close'].pct_change()
    
    # Remove any NaN values
    data = data.dropna()
    
    # Create DataFrame with required format
    df = pd.DataFrame({
        'returns': data['returns'],
        'prices': data['Close']
    })
    
    print(f"\nMarket Data Summary:")
    print(f"Total observations: {len(df)}")
    print(f"Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"Average daily return: {df['returns'].mean():.6f}")
    print(f"Daily volatility: {df['returns'].std():.6f}")
    
    return df

def analyze_data(data: pd.DataFrame, title: str) -> GARCHAnalyzer:
    """
    Analyze the given dataset and create visualizations.
    """
    print(f"\nAnalyzing {title}...")
    
    # Create analyzer and fit model
    analyzer = GARCHAnalyzer(data['returns'])
    params = analyzer.fit_model()
    
    # Get conditional volatility
    data['conditional_volatility'] = analyzer.get_conditional_volatility()
    
    # Print model parameters
    print("\nEstimated GARCH parameters:")
    for key, value in params.items():
        print(f"{key}: {value:.4f}")
    
    # Calculate return statistics
    stats = {
        'mean': data['returns'].mean(),
        'std': data['returns'].std(),
        'skewness': data['returns'].skew(),
        'kurtosis': data['returns'].kurtosis(),
        'min': data['returns'].min(),
        'max': data['returns'].max(),
        'acf_squared_returns': pd.Series(data['returns']**2).autocorr()
    }
    
    print("\nReturn series statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")
    
    # Create visualizations
    prefix = title.lower().replace(' ', '_').replace('(', '').replace(')', '')
    visualizer = VolatilityVisualizer(data)
    
    print(f"\nPlotting returns and volatility for {title}...")
    visualizer.plot_returns_and_volatility(title_prefix=prefix)
    
    print(f"\nPlotting evidence of volatility clustering for {title}...")
    visualizer.plot_volatility_clustering(title_prefix=prefix)
    
    print(f"\nPlotting returns distribution for {title}...")
    visualizer.plot_returns_distribution(title_prefix=prefix)
    
    if 'conditional_volatility' in data.columns:
        visualizer.plot_qq_plot(title_prefix=prefix)
    
    return analyzer

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("1. Generating simulated data...")
    
    # Generate data with volatility clustering (GARCH process)
    print("\nGenerating GARCH process data (with volatility clustering)...")
    garch_params = {
        'omega': 0.05,   # Small constant term
        'alpha': 0.15,   # Impact of past squared returns
        'beta': 0.80     # High persistence in volatility
    }
    
    # First analyze without shock
    print("\nAnalyzing GARCH process without shock...")
    data_garch = generate_sample_data(n_samples=1000, garch_params=garch_params)
    analyzer_no_shock = analyze_data(data_garch, "GARCH Process (without shock)")
    
    # Generate homoskedastic data (no volatility clustering)
    print("\nGenerating homoskedastic data (no volatility clustering)...")
    data_homo = generate_sample_data(n_samples=1000, homoskedastic=True)
    
    print("\n2. Analyzing and comparing datasets...")
    
    # Analyze homoskedastic data
    analyzer_homo = analyze_data(data_homo, "Homoskedastic Process (no volatility clustering)")
    
    # Then analyze with shock
    print("\n3. Analyzing GARCH process with shock...")
    shock_params = {
        'time': 500,      # Shock at day 500
        'magnitude': 8.0  # 8x normal volatility (increased from 5x)
    }
    data_garch_shock = generate_sample_data(
        n_samples=1000, 
        garch_params=garch_params,
        add_shock=True,
        shock_params=shock_params
    )
    analyzer_shock = analyze_data(data_garch_shock, "GARCH Process (with shock)")
    
    # Analyze shock impact
    print("\n4. Analyzing shock impact...")
    shock_date = data_garch_shock.index[shock_params['time']]
    visualizer = VolatilityVisualizer(data_garch_shock)
    visualizer.plot_shock_analysis(shock_date)
    
    # Compare persistence before and after shock
    print("\nComparing GARCH persistence:")
    print("\nPre-shock model:")
    pre_shock_data = data_garch_shock.iloc[:shock_params['time']]
    analyzer_pre = GARCHAnalyzer(pre_shock_data['returns'])
    pre_params = analyzer_pre.fit_model()
    print(f"Alpha: {pre_params['alpha']:.4f}")
    print(f"Beta: {pre_params['beta']:.4f}")
    print(f"Persistence: {pre_params['persistence']:.4f}")
    
    print("\nPost-shock model:")
    post_shock_data = data_garch_shock.iloc[shock_params['time']:]
    analyzer_post = GARCHAnalyzer(post_shock_data['returns'])
    post_params = analyzer_post.fit_model()
    print(f"Alpha: {post_params['alpha']:.4f}")
    print(f"Beta: {post_params['beta']:.4f}")
    print(f"Persistence: {post_params['persistence']:.4f}")
    
    # Analyze real market data
    print("\n5. Analyzing real market data (S&P 500)...")
    market_data = get_market_data("^GSPC", years=4)
    analyzer_market = analyze_data(market_data, "S&P 500")
    
    # Compare simulated vs real data characteristics
    print("\nComparison of Real vs Simulated Data:")
    print("\nReal S&P 500 GARCH parameters:")
    market_params = analyzer_market.fit_model()
    print(f"Alpha: {market_params['alpha']:.4f}")
    print(f"Beta: {market_params['beta']:.4f}")
    print(f"Persistence: {market_params['persistence']:.4f}")
    
    print("\nSimulated GARCH parameters:")
    print(f"Alpha: {garch_params['alpha']:.4f}")
    print(f"Beta: {garch_params['beta']:.4f}")
    print(f"Persistence: {garch_params['alpha'] + garch_params['beta']:.4f}")
    
    print("\nAnalysis complete! Key observations:")
    print("1. GARCH process shows clear volatility clustering")
    print("2. Homoskedastic process shows constant volatility")
    print("3. GARCH process has more realistic fat-tailed returns distribution")
    print("4. Shock creates a significant spike in volatility (8x normal)")
    print("5. Post-shock volatility persistence may differ from pre-shock")
    print("6. Autocorrelation in squared returns is much stronger in GARCH process")
    print("7. Real market data shows similar volatility clustering patterns")
    print("8. S&P 500 GARCH parameters indicate market volatility characteristics")

if __name__ == "__main__":
    main() 