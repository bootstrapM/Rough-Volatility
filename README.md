# Rough-Volatility

## Some Background

In 1900, Bachelier introduced the first mathematical model of prices, the Brownian motion. For several years the only modification to Bachelier model was to consider log-prices (instead of prices themselves) that are described by Brownian motion. While this modification prevents the prices from becoming negative, for a long time none of the drawbacks of Bachelier model were addressed (a major drawback being failing to capture features like **fat tails**, **volatility clustering**, and **skewness** in returns). Then in 1973 Black and Scholes published their famous paper where it was shown that perfect delta-hedging was possible (i.e. it is possible to construct a portfolio where the risk from small movements in the price of the underlying asset can be completely eliminated). However the condition under which this is possible is

- Continuous trading (the hedge is adjusted instantly and continuously)
- No Transaction cost 
- No market impact 
- No Price jumps and no market crashes

Of course none of these conditions are realized in the real world markets. The last point is a particularly problematic assumption since the fat-tailed nature of return distributions were known as early as 1963 by Mandelbrot in “The Variation of Certain Speculative Prices” (he was studying studied cotton prices). He made the observation that "large changes tend to be followed by large changes, of either sign, and small changes tend to be followed by small changes”. This phenomenon which is now referred to as **volatility clustering** is captured by GARCH models. 

As eluded to above, a limitation of the Black-Scholes model was that the volatility parameter was constant. To correct this aspect, Heston published a paper in 1993. This was among the most famous post-Black–Scholes models, that encapsulated volatility clustering within a continuous time, Brownian motion formalism. However, Both the Heston and GARCH models share a similar limitation in that in their framework, volatility fluctuations decay at a rate governed by a single time scale. This implies that periods of high or low volatility are expected to have a relatively defined duration. However, this feature is inconsistent with empirical market data. Observations show that volatility bursts in financial markets do not adhere to a single time scale, but can persist for vastly different periods, ranging from mere hours to several years. This suggests that volatility fluctuations exhibit a **scale-invariant property**, lacking a clear, inherent time scale. Consequently, the need arose for a version of the Heston model that could capture this scale-invariant behavior of volatility.

Towards this end, in 2014, Gatheral, Jaisson, and Rosenbaum introduced their now famous “Rough Volatility” model. This model can be viewed as an extension of what is known as Multifractal Random Walk (MRW) with an extra parameter for tuning the roughness of volatility. In these pages, we will study the GARCH(1,1), the Heston model and the Rough Volatility model of Gatheral et.al. in a somewhat simplified setting.

## Some questions

* Is rough volatility is an inevitable consequence of electronic trading?
* In 2009 when Zumbach noticed that financial time series are not statistically invariant upon time reversal. How is this fact incorporated in volatility modeling? 