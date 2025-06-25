
# Glossary of Trading Terms, Definitions, and Formulas

## QuoteData

### mark_price
- **Formula:** (bid + ask) / 2  
- **Definition:**  
  The mark price is the midpoint between the bid price (the highest price a buyer is willing to pay) and the ask price (the lowest price a seller is willing to accept). It represents a fair estimate of a security’s value at a specific moment and is often used to mark-to-market open positions.

### net_change
- **Formula:** last_price - close_price  
- **Definition:**  
  Net change is the difference between the last traded price and the previous session’s closing price. It indicates how much the price has moved since the last trading day, showing upward or downward momentum.

### percent_change
- **Formula:** (net_change / close_price) * 100  
- **Definition:**  
  Percent change expresses the net change as a percentage of the previous closing price. It’s useful for quickly assessing the magnitude of a price move relative to its value.

## OptionData

### mark_price_option
- **Formula:** (bid + ask) / 2  
- **Definition:**  
  The midpoint between the bid and ask prices for an option contract. This provides a fair value for the option’s price, smoothing out the spread between buyers and sellers.

### intrinsic_value_call
- **Formula:** max(last_price - strike_price, 0)  
- **Definition:**  
  The actual value of a call option if exercised immediately. It’s the amount by which the underlying asset price exceeds the strike price, or zero if the option is out of the money.

### intrinsic_value_put
- **Formula:** max(strike_price - last_price, 0)  
- **Definition:**  
  The value of a put option if exercised immediately. It’s the amount by which the strike price exceeds the current asset price, or zero if the option is out of the money.

### extrinsic_value
- **Formula:** option_price - intrinsic_value  
- **Definition:**  
  Also called "time value", extrinsic value is the part of the option’s price that exceeds its intrinsic value. It reflects factors like volatility, time until expiration, and market demand.

### covered_return
- **Formula:** (extrinsic / mark_price) * (365 / days_to_expiration)  
- **Definition:**  
  The annualized return earned from selling a covered call option, calculated as the extrinsic value received divided by the mark price, adjusted for the remaining time until expiration.

### return_on_capital
- **Formula:** (mark_price * dv / -bp_effect) * (365 / days_to_expiration)  
- **Definition:**  
  This measures the annualized return on the capital used to enter a position, factoring in buying power and the effect of holding the position.

### return_on_risk
- **Formula:** (mark_price / max_risk) * (365 / days_to_expiration)  
- **Definition:**  
  The annualized return divided by the maximum risk for the position, showing potential profitability relative to worst-case losses.

## VolatilityData

### volatility_difference
- **Formula:** front_vol - back_vol  
- **Definition:**  
  The difference between near-term (front month) and longer-term (back month) implied volatilities. It is often used to assess volatility skew or term structure.

### weighted_back_volatility
- **Formula:** sqrt(((back_vol² * t2) - (front_vol² * t1)) / (t2 - t1))  
- **Definition:**  
  A volatility measure that combines front and back volatilities, weighted by their respective times to expiration, giving a time-adjusted volatility estimate.

### norm_cdf
- **Formula:** Normal CDF of x  
- **Definition:**  
  The cumulative distribution function (CDF) of the standard normal distribution, often used in option pricing to determine probabilities.

### expected_move
- **Formula:** last_price * exp(vol² / 2) * (2 * norm_cdf(volatility) - 1)  
- **Definition:**  
  The projected move in price, based on implied volatility and the standard normal distribution. Used to estimate how much an asset might move by a certain date.

### front_expected_move
- **Formula:** expected_move(sqrt(t1) * front_vol)  
- **Definition:**  
  Expected price move for the nearest-term expiration using its implied volatility and time to expiration.

### back_expected_move
- **Formula:** expected_move(sqrt(t2) * back_vol)  
- **Definition:**  
  Expected price move for the further expiration date, using the corresponding volatility.

### expected_move_difference
- **Formula:** expected_move(sqrt(t2 - t1) * wbv²)  
- **Definition:**  
  The difference in expected moves between two expiration periods, using weighted back volatility.

### market_maker_move
- **Formula:** expected_move(sqrt(t1 * (front_vol² - wbv²)))  
- **Definition:**  
  The expected price movement implied by market makers, derived from the difference between front volatility and weighted back volatility.

## FundamentalData

### pe_ratio
- **Formula:** last_price / earnings_per_share  
- **Definition:**  
  The price-to-earnings ratio compares a company’s current share price to its earnings per share, commonly used to value stocks.

### dividend_yield
- **Formula:** (dividend * freq_multiplier) / last_price  
- **Definition:**  
  The dividend yield shows the annual dividend income as a percentage of the current share price.

### market_cap
- **Formula:** last_price * shares_outstanding  
- **Definition:**  
  Market capitalization is the total market value of all outstanding shares, representing the company’s overall value as perceived by the market.

## VolumeData

### put_call_ratio
- **Formula:** put_volume / call_volume  
- **Definition:**  
  This ratio compares the trading volume of put options to call options, serving as a sentiment indicator. Values above 1 may indicate bearish sentiment, while values below 1 may indicate bullish sentiment.

## HistoricalData

### historical_volatility
- **Formula:** std(log_returns) * sqrt(252)  
- **Definition:**  
  Historical volatility measures how much an asset's price fluctuated, calculated as the annualized standard deviation of daily log returns.

## SwingTradeAnalytics

### sma
- **Formula:** Simple Moving Average of close over period  
- **Definition:**  
  The average of a security's closing prices over a specified number of periods, smoothing out short-term price fluctuations.

### ema
- **Formula:** Exponential Moving Average of close over period  
- **Definition:**  
  Similar to SMA but gives more weight to recent prices, making it more responsive to new information.

### atr
- **Formula:** Avg. True Range over period  
- **Definition:**  
  The average of the true range (high minus low, considering gaps) over a given period. It is a measure of volatility.

### rsi
- **Formula:** 100 - (100 / (1 + RS)); RS = avg_gain / avg_loss  
- **Definition:**  
  The Relative Strength Index measures the speed and change of price movements. RSI values above 70 suggest overbought conditions; below 30 suggest oversold.

### price_change_percent
- **Formula:** % change in close over period  
- **Definition:**  
  The percentage increase or decrease in closing price over a chosen period.

### support_level
- **Formula:** rolling min of low over lookback  
- **Definition:**  
  The lowest price observed over a specified lookback period. It often acts as a "floor" where buying interest appears.

### resistance_level
- **Formula:** rolling max of high over lookback  
- **Definition:**  
  The highest price observed over a specified lookback period, often acting as a "ceiling" for price advances.
