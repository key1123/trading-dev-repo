Glossary of Trading Terms, Definitions, and Mathematical Formulas

1. QuoteData:
- mark_price = (bid + ask) / 2
- net_change = last_price - close_price
- percent_change = (net_change / close_price) * 100

2. OptionData:
- mark_price_option = (bid + ask) / 2
- intrinsic_value_call = max(last_price - strike_price, 0)
- intrinsic_value_put = max(strike_price - last_price, 0)
- extrinsic_value = option_price - intrinsic_value
- covered_return = (extrinsic / mark_price) * (365 / days_to_expiration)
- return_on_capital = (mark_price * dv / -bp_effect) * (365 / days_to_expiration)
- return_on_risk = (mark_price / max_risk) * (365 / days_to_expiration)

3. VolatilityData:
- volatility_difference = front_vol - back_vol
- weighted_back_volatility = sqrt(((back_vol² * t2) - (front_vol² * t1)) / (t2 - t1))
- norm_cdf = Normal CDF of x
- expected_move = last_price * exp(vol² / 2) * (2 * norm_cdf(volatility) - 1)
- front_expected_move = expected_move(sqrt(t1) * front_vol)
- back_expected_move = expected_move(sqrt(t2) * back_vol)
- expected_move_difference = expected_move(sqrt(t2 - t1) * wbv²)
- market_maker_move = expected_move(sqrt(t1 * (front_vol² - wbv²)))

4. FundamentalData:
- pe_ratio = last_price / earnings_per_share
- dividend_yield = (dividend * freq_multiplier) / last_price
- market_cap = last_price * shares_outstanding

5. VolumeData:
- put_call_ratio = put_volume / call_volume

6. HistoricalData:
- historical_volatility = std(log_returns) * sqrt(252)

7. SwingTradeAnalytics:
- sma = Simple Moving Average of close over period
- ema = Exponential Moving Average of close over period
- atr = Avg. True Range over period
- rsi = 100 - (100 / (1 + RS)); RS = avg_gain / avg_loss
- price_change_percent = % change in close over period
- support_level = rolling min of low over lookback
- resistance_level = rolling max of high over lookback