# Rough-Volatility

## Some Background

In 1900, Bachelier introduced the first mathematical model of prices, the Brownian motion. For several years the only modification to Bachelier model was to consider log-prices (instead of prices themselves) that are described by Brownian motion. While this modification prevents the prices from becoming negative, for a long time none of the drawbacks of Bachelier model were addressed. The major drawbacks being failing to capture now widely accepted *stylized facts* about financial time series. At daily frequency granularity these are:

 - fat tails in return distributions (unconditional return distribution tend to have fatter tails than conditional return distribution)
 - absence of autocorrelation in returns (at time scales above which the market microstructure becomes important)
 - volatility clustering (strong positive autocorrelation in the absolute or squared return series)
 - aggregational normality (at longer time scales say, weekly or monthly
returns, as opposed to daily or intraday, the distribution of
log-returns becomes closer to a Gaussian distribution.)
 - skewness in returns (Negative skewness, commonly observed in stock returns, indicates a longer or fatter left tail, implying that large downward movements are more prevalent or severe than large upward movements.)
 
 In 1973 Black and Scholes published their famous paper where it was shown that perfect delta-hedging was possible (i.e. it is possible to construct a portfolio where the risk from small movements in the price of the underlying asset can be completely eliminated). However the condition under which this is possible is

- Continuous trading (the hedge is adjusted instantly and continuously)
- No Transaction cost 
- No market impact 
- No Price jumps and no market crashes

Of course none of these conditions are realized in the real world markets. The last point is a particularly problematic assumption since the fat-tailed nature of return distributions were known as early as 1963 by Mandelbrot in “The Variation of Certain Speculative Prices” (he was studying studied cotton prices). In the same paper, he made the observation that "large changes tend to be followed by large changes, of either sign, and small changes tend to be followed by small changes”. This phenomenon which is now referred to as **volatility clustering** is captured by GARCH models. Later in 1976, Black discovered that a down-move in the asset price is positively correlated with an up-move in the volatility. This implies an asymmetric response of volatility to shocks and is referred to as the **leverage effect**. While the standard GARCH(1,1) model captures fat tails, votality clustering and also mean reversion in the volatility series it fails to capture the leverage effect. Modifications (such as the EGARCH model) are needed to include volatility asymmetry.

As eluded to above, a limitation of the Black-Scholes model was that the volatility parameter was constant. This issue was addressed in a famous paper by Heston in 1993 which was among the most famous post-Black–Scholes models, that encapsulated volatility clustering as well as the leverage effect within a continuous time, Brownian motion formalism that we will explore in some detail here. 

However, both Heston and GARCH models share a limitation in their formulation namely volatility fluctuations decay at a rate governed by a single time scale. This implies that periods of high or low volatility are expected to have a relatively defined duration. However, this feature is inconsistent with empirical market data. Observations (particularly in the high frequency data) show that volatility bursts in financial markets do not adhere to a single time scale, but can persist for vastly different periods, ranging from mere hours to several years. This suggests that volatility fluctuations exhibit a **scale-invariant property**, lacking a clear, inherent time scale. Consequently, the need arose for a version of the Heston model that could capture this scale-invariant behavior of volatility.

Towards this end, in 2014, Gatheral, Jaisson, and Rosenbaum introduced their now famous “Rough Volatility” model. This model can be viewed as an extension of what is known as Multifractal Random Walk (MRW) with an extra parameter for tuning the roughness of volatility. In these pages, we will study the GARCH(1,1), the Heston model as well as the Rough Volatility model of Gatheral et.al. in a somewhat simplified setting. 

## Repository Structure

The structure of this evolving repository is as follows
```
Rough-Volatility/
├── GARCH/                         # GARCH(1,1) implementation and analysis
│   ├── garch_analysis.py
│   ├── main.py
│   ├── simulate_data.py
│   ├── visualization.py
│   ├── requirements.txt
│   └── README.md
│   ├── plots/
│
├── HestonModel/                   # Heston stochastic volatility model
│   ├── src/
│   │   ├── heston_model.py
|   |   ├── heston_pricing.py
│   ├── figures/
│   ├── requirements.txt
│   ├── README.md
│
├── RFSV model/                    # Rough Fractional Stochastic Volatility
│   ├── src/
│   │   ├── fractional_random_walk.py
│   │   ├── gaussian_rv.py
│   │   └── price_returns.py
│   ├── figures/
│   ├── requirements.txt
│   └── README.md
│
└── README.md                      # Main project documentation
```


## Some questions for the future

* Derivative pricing under rough volatility
* Zumbach pointed out in 2009 that financial time series are not statistically invariant upon time reversal. How is this fact incorporated in volatility modeling?
* Is rough volatility is an inevitable consequence of electronic trading?
* Can rough volatility be derived from market microstructure (e.g., order flow models, limit order books)?