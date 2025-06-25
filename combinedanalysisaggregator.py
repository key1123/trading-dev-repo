import math
import numpy as np
import requests
import csv
import json

"""
Stock Analytics Aggregator

This module provides classes and methods to analyze stock market data, including quotes, options,
volatility, fundamentals, volume, and historical price data.

Core functionalities:
- Data classes represent different aspects of market data (QuoteData, OptionData, etc.).
- Each class provides methods to compute relevant financial metrics (e.g., mark price, intrinsic value).
- The StockAnalyticsAggregator class combines all available data for a symbol and computes a
  comprehensive set of analytics, risk scores, and trade signals.
- Supports batch evaluation of multiple symbols.
- Provides summary reporting and export to CSV/JSON.

The aggregator safely handles missing data and exceptions, allowing partial datasets without breaking.
It also includes sample risk scoring and trade signal logic which can be customized or extended.

Usage example is included at the bottom under __main__.
"""


# Class to represent basic quote data for a stock
class QuoteData:
    def __init__(self, bid=None, ask=None, last_price=None, close_price=None):
        """
        Initialize QuoteData with bid, ask, last traded price, and previous close price.

        Args:
            bid (float): Current bid price.
            ask (float): Current ask price.
            last_price (float): Last traded price.
            close_price (float): Previous closing price.
        """
        self.bid = bid
        self.ask = ask
        self.last_price = last_price
        self.close_price = close_price

    def mark_price(self):
        """
        Calculate the mark price as the midpoint between bid and ask.

        Returns:
            float or None: Midpoint price or None if calculation fails.
        """
        try:
            return (self.bid + self.ask) / 2
        except:
            return None

    def net_change(self):
        """
        Calculate the net change in price from previous close to last price.

        Returns:
            float or None: Net change or None if calculation fails.
        """
        try:
            return self.last_price - self.close_price
        except:
            return None

    def percent_change(self):
        """
        Calculate the percentage change from previous close to last price.

        Returns:
            float or None: Percent change (%) or None if calculation fails.
        """
        try:
            net = self.net_change()
            return (net / self.close_price) * 100 if net is not None and self.close_price else None
        except:
            return None


# Class to represent option-specific data and calculations
class OptionData:
    def __init__(self, strike_price=None, option_price=None, days_to_expiration=None,
                 bid=None, ask=None, last_price=None,
                 dv=None, bp_effect=None, max_risk=None):
        """
        Initialize OptionData with option parameters and risk metrics.

        Args:
            strike_price (float): Option strike price.
            option_price (float): Market price of the option.
            days_to_expiration (int): Days until option expiration.
            bid (float): Option bid price.
            ask (float): Option ask price.
            last_price (float): Option last traded price.
            dv (float): Delta value or similar metric for return calculations.
            bp_effect (float): Basis point effect for return calculations.
            max_risk (float): Maximum risk exposure for return on risk.
        """
        self.strike_price = strike_price
        self.option_price = option_price
        self.days_to_expiration = days_to_expiration
        self.bid = bid
        self.ask = ask
        self.last_price = last_price
        self.dv = dv
        self.bp_effect = bp_effect
        self.max_risk = max_risk

    def mark_price_option(self):
        """
        Calculate the option mark price as the midpoint of bid and ask.

        Returns:
            float or None: Midpoint option price or None if calculation fails.
        """
        try:
            return (self.bid + self.ask) / 2
        except:
            return None

    def intrinsic_value_call(self, last_price):
        """
        Calculate intrinsic value for a call option.

        Args:
            last_price (float): Current underlying asset price.

        Returns:
            float: Intrinsic value (max(last_price - strike_price, 0))
        """
        try:
            return max(last_price - self.strike_price, 0)
        except:
            return None

    def intrinsic_value_put(self, last_price):
        """
        Calculate intrinsic value for a put option.

        Args:
            last_price (float): Current underlying asset price.

        Returns:
            float: Intrinsic value (max(strike_price - last_price, 0))
        """
        try:
            return max(self.strike_price - last_price, 0)
        except:
            return None

    def extrinsic_value(self, intrinsic):
        """
        Calculate the extrinsic (time) value of the option.

        Args:
            intrinsic (float): Intrinsic value of the option.

        Returns:
            float: Extrinsic value = option price - intrinsic value
        """
        try:
            return self.option_price - intrinsic
        except:
            return None

    def covered_return(self, extrinsic, mark_price):
        """
        Calculate annualized covered call return based on extrinsic value.

        Args:
            extrinsic (float): Extrinsic value of the option.
            mark_price (float): Mark price of the option.

        Returns:
            float: Annualized covered return.
        """
        try:
            return (extrinsic / mark_price) * (365 / self.days_to_expiration)
        except:
            return None

    def return_on_capital(self, mark_price):
        """
        Calculate return on capital for the option position.

        Args:
            mark_price (float): Mark price of the option.

        Returns:
            float: Annualized return on capital.
        """
        try:
            return (mark_price * self.dv / -self.bp_effect) * (365 / self.days_to_expiration)
        except:
            return None

    def return_on_risk(self, mark_price):
        """
        Calculate return on risk.

        Args:
            mark_price (float): Mark price of the option.

        Returns:
            float: Annualized return on risk.
        """
        try:
            return (mark_price / self.max_risk) * (365 / self.days_to_expiration)
        except:
            return None


# Class to handle volatility data and related computations
class VolatilityData:
    def __init__(self, front_vol=None, back_vol=None, t1=None, t2=None, last_price=None):
        """
        Initialize VolatilityData with implied volatilities and times.

        Args:
            front_vol (float): Front implied volatility.
            back_vol (float): Back implied volatility.
            t1 (float): Time period 1.
            t2 (float): Time period 2.
            last_price (float): Current price of the underlying asset.
        """
        self.front_vol = front_vol
        self.back_vol = back_vol
        self.t1 = t1
        self.t2 = t2
        self.last_price = last_price

    def volatility_difference(self):
        """
        Calculate the difference between front and back volatilities.

        Returns:
            float or None: Difference or None if error.
        """
        try:
            return self.front_vol - self.back_vol
        except:
            return None

    def weighted_back_volatility(self):
        """
        Calculate weighted back volatility using volatilities and time periods.

        Returns:
            float or None: Weighted back volatility or None if error.
        """
        try:
            numerator = (self.back_vol ** 2 * self.t2) - (self.front_vol ** 2 * self.t1)
            denominator = self.t2 - self.t1
            return math.sqrt(numerator / denominator)
        except:
            return None

    def norm_cdf(self, x):
        """
        Compute the normal cumulative distribution function at x.

        Args:
            x (float): Value to evaluate.

        Returns:
            float or None: CDF value or None if error.
        """
        try:
            return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
        except:
            return None

    def expected_move(self, volatility):
        """
        Calculate expected move based on volatility.

        Args:
            volatility (float): Volatility value.

        Returns:
            float or None: Expected move or None if error.
        """
        try:
            n_vol = 2 * self.norm_cdf(volatility) - 1
            return self.last_price * math.exp(volatility**2 / 2) * n_vol
        except:
            return None

    def front_expected_move(self):
        """
        Expected move for front period volatility.

        Returns:
            float or None: Front expected move or None if error.
        """
        try:
            return self.expected_move(math.sqrt(self.t1) * self.front_vol)
        except:
            return None

    def back_expected_move(self):
        """
        Expected move for back period volatility.

        Returns:
            float or None: Back expected move or None if error.
        """
        try:
            return self.expected_move(math.sqrt(self.t2) * self.back_vol)
        except:
            return None

    def expected_move_difference(self, wbv):
        """
        Difference in expected move between periods using weighted back volatility.

        Args:
            wbv (float): Weighted back volatility.

        Returns:
            float or None: Expected move difference or None if error.
        """
        try:
            return self.expected_move(math.sqrt(self.t2 - self.t1) * (wbv ** 2))
        except:
            return None

    def market_maker_move(self, wbv):
        """
        Calculate market maker expected move.

        Args:
            wbv (float): Weighted back volatility.

        Returns:
            float or None: Market maker expected move or None if error.
        """
        try:
            return self.expected_move(math.sqrt(self.t1 * (self.front_vol ** 2 - wbv ** 2)))
        except:
            return None


# Class for fundamental stock data and calculations
class FundamentalData:
    def __init__(self, earnings_per_share=None, dividend=None, dividend_frequency=None,
                 shares_outstanding=None, last_price=None):
        """
        Initialize fundamental data including earnings, dividends, shares outstanding.

        Args:
            earnings_per_share (float): EPS value.
            dividend (float): Dividend per period.
            dividend_frequency (str): Frequency code ('A', 'Q', 'M').
            shares_outstanding (float): Number of shares outstanding.
            last_price (float): Last traded price.
        """
        self.earnings_per_share = earnings_per_share
        self.dividend = dividend
        self.dividend_frequency = dividend_frequency
        self.shares_outstanding = shares_outstanding
        self.last_price = last_price

    def pe_ratio(self, last_price):
        """
        Calculate Price-to-Earnings (P/E) ratio.

        Args:
            last_price (float): Last price.

        Returns:
            float or None: P/E ratio or None if error.
        """
        try:
            return last_price / self.earnings_per_share
        except:
            return None

    def dividend_yield(self, last_price):
        """
        Calculate annualized dividend yield.

        Args:
            last_price (float): Last price.

        Returns:
            float or None: Dividend yield or None if error.
        """
        try:
            multiplier = {"A": 1, "Q": 4, "M": 12}.get(self.dividend_frequency.upper(), 1)
            return (self.dividend * multiplier) / last_price
        except:
            return None

    def market_cap(self, last_price):
        """
        Calculate market capitalization.

        Args:
            last_price (float): Last price.

        Returns:
            float or None: Market cap or None if error.
        """
        try:
            return last_price * self.shares_outstanding
        except:
            return None


# Class for volume-based data such as put-call ratio
class VolumeData:
    def __init__(self, put_volume=None, call_volume=None):
        """
        Initialize VolumeData with put and call option volumes.

        Args:
            put_volume (int): Put option volume.
            call_volume (int): Call option volume.
        """
        self.put_volume = put_volume
        self.call_volume = call_volume

    def put_call_ratio(self):
        """
        Calculate put-call volume ratio.

        Returns:
            float or None: Put-call ratio or None if error.
        """
        try:
            return self.put_volume / self.call_volume
        except:
            return None


# Class for historical price data and volatility calculation
class HistoricalData:
    def __init__(self, closes=None):
        """
        Initialize HistoricalData with a list of closing prices.

        Args:
            closes (list of float): Historical closing prices.
        """
        self.closes = closes or []

    def historical_volatility(self):
        """
        Calculate annualized historical volatility from log returns.

        Returns:
            float or None: Annualized volatility or None if error.
        """
        try:
            log_returns = np.log(np.array(self.closes[1:]) / np.array(self.closes[:-1]))
            return np.std(log_returns, ddof=1) * math.sqrt(252)  # 252 trading days per year
        except:
            return None


# Aggregator class to combine data types and run analytics
class StockAnalyticsAggregator:
    def __init__(self, quote_data=None, option_data=None, volatility_data=None,
                 fundamental_data=None, volume_data=None, historical_data=None, swing_data=None):
        """
        Initialize aggregator with all relevant data objects for a stock.

        Args:
            quote_data (QuoteData): Quote data object.
            option_data (OptionData): Option data object.
            volatility_data (VolatilityData): Volatility data object.
            fundamental_data (FundamentalData): Fundamental data object.
            volume_data (VolumeData): Volume data object.
            historical_data (HistoricalData): Historical data object.
            swing_data (object): Optional swing trade indicator data.
        """
        self.quote = quote_data
        self.option = option_data
        self.volatility = volatility_data
        self.fundamental = fundamental_data
        self.volume = volume_data
        self.historical = historical_data
        self.swing = swing_data  # Optional, for custom swing indicators

    def compute_risk_score(self, results):
        """
        Compute a basic risk score based on return_on_risk and historical_volatility.

        Args:
            results (dict): Dictionary of evaluated metrics.

        Returns:
            float or None: Risk score or None if insufficient data.
        """
        try:
            if results.get("return_on_risk") and results.get("historical_volatility"):
                return round(results["return_on_risk"] / (results["historical_volatility"] + 1e-6), 2)
        except:
            pass
        return None

    def compute_custom_risk_score(self, results):
        """
        Compute a custom risk score adjusted by put-call ratio (market sentiment).

        Args:
            results (dict): Dictionary of evaluated metrics.

        Returns:
            float or None: Adjusted risk score or None if insufficient data.
        """
        try:
            base_score = self.compute_risk_score(results)
            pcr = results.get("put_call_ratio")
            if base_score is not None and pcr is not None:
                adjusted_score = base_score * (1 / (pcr + 1e-6))
                return round(adjusted_score, 2)
        except:
            pass
        return None

    def compute_trade_signal(self, results):
        """
        Determine a simple trade signal based on RSI, moving averages, and resistance level.

        Args:
            results (dict): Dictionary of evaluated metrics including swing indicators.

        Returns:
            str or None: Trade signal such as 'BUY', 'SELL', 'HOLD', or 'BUY_BREAKOUT'.
        """
        try:
            rsi = results.get("RSI_14")
            sma = results.get("SMA_20")
            ema = results.get("EMA_20")
            resistance = results.get("Resistance_20")
            last_price = self.quote.last_price if self.quote else None

            if rsi is not None:
                if rsi < 30:
                    return "BUY"
                elif rsi > 70:
                    return "SELL"

            if sma is not None and ema is not None:
                if sma > ema:
                    return "HOLD/BUY"
                else:
                    return "HOLD/SELL"

            if last_price is not None and resistance is not None:
                if last_price >= resistance:
                    return "BUY_BREAKOUT"
                else:
                    return "HOLD"

            return "HOLD"
        except:
            pass
        return None

    def evaluate(self):
        """
        Evaluate all available analytics for the stock and compute risk scores and trade signals.

        Returns:
            dict: Dictionary of calculated metrics, scores, and signals.
        """
        results = {}

        # Evaluate quote-related metrics
        try:
            if self.quote:
                results["mark_price"] = self.quote.mark_price()
                results["net_change"] = self.quote.net_change()
                results["percent_change"] = self.quote.percent_change()
        except:
            pass

        # Evaluate option-related metrics
        try:
            if self.option and self.quote:
                intrinsic_call = self.option.intrinsic_value_call(self.quote.last_price)
                intrinsic_put = self.option.intrinsic_value_put(self.quote.last_price)
                extrinsic = self.option.extrinsic_value(intrinsic_call)
                mark_price = self.option.mark_price_option() or self.quote.mark_price()
                results.update({
                    "intrinsic_call": intrinsic_call,
                    "intrinsic_put": intrinsic_put,
                    "extrinsic_value": extrinsic,
                    "covered_return": self.option.covered_return(extrinsic, mark_price),
                    "return_on_capital": self.option.return_on_capital(mark_price),
                    "return_on_risk": self.option.return_on_risk(mark_price)
                })
        except:
            pass

        # Evaluate volatility-related metrics
        try:
            if self.volatility and self.quote:
                wbv = self.volatility.weighted_back_volatility()
                results.update({
                    "volatility_diff": self.volatility.volatility_difference(),
                    "weighted_back_volatility": wbv,
                    "front_expected_move": self.volatility.front_expected_move(),
                    "back_expected_move": self.volatility.back_expected_move(),
                    "expected_move_diff": self.volatility.expected_move_difference(wbv),
                    "market_maker_move": self.volatility.market_maker_move(wbv)
                })
        except:
            pass

        # Evaluate fundamental metrics
        try:
            if self.fundamental and self.quote:
                results.update({
                    "pe_ratio": self.fundamental.pe_ratio(self.quote.last_price),
                    "dividend_yield": self.fundamental.dividend_yield(self.quote.last_price),
                    "market_cap": self.fundamental.market_cap(self.quote.last_price)
                })
        except:
            pass

        # Evaluate volume metrics
        try:
            if self.volume:
                results["put_call_ratio"] = self.volume.put_call_ratio()
        except:
            pass

        # Evaluate historical data metrics
        try:
            if self.historical:
                results["historical_volatility"] = self.historical.historical_volatility()
        except:
            pass

        # Add swing trade indicator results if available
        if self.swing:
            try:
                results.update(self.swing)
            except:
                pass

        # Calculate risk scores based on evaluated metrics
        results["risk_score"] = self.compute_risk_score(results)
        results["custom_risk_score"] = self.compute_custom_risk_score(results)

        # Generate trade signal based on indicators
        results["trade_signal"] = self.compute_trade_signal(results)

        return results

    @staticmethod
    def batch_evaluate(symbol_data_list):
        """
        Evaluate a batch of stocks given a dictionary of symbol -> data objects.

        Args:
            symbol_data_list (dict): Dictionary mapping symbol strings to dicts of data objects.

        Returns:
            dict: Dictionary mapping symbol to evaluation results.
        """
        results = {}
        for symbol, data in symbol_data_list.items():
            aggregator = StockAnalyticsAggregator(
                quote_data=data.get("quote"),
                option_data=data.get("option"),
                volatility_data=data.get("volatility"),
                fundamental_data=data.get("fundamental"),
                volume_data=data.get("volume"),
                historical_data=data.get("historical"),
                swing_data=data.get("swing")
            )
            results[symbol] = aggregator.evaluate()
        return results

    @staticmethod
    def summary_report(results):
        """
        Generate a summary report with average and top performer for each numeric metric.

        Args:
            results (dict): Dictionary mapping symbols to evaluation results.

        Returns:
            dict: Summary dictionary with averages and top symbols for each metric.
        """
        summary = {}
        numeric_metrics = {}
        for symbol, metrics in results.items():
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    numeric_metrics.setdefault(k, []).append((symbol, v))

        for metric, values in numeric_metrics.items():
            avg = sum(val for _, val in values) / len(values)
            top = max(values, key=lambda x: x[1])
            summary[metric] = {
                "average": avg,
                "top_symbol": top[0],
                "top_value": top[1]
            }
        return summary

    @staticmethod
    def export_to_csv(results, filename="results.csv"):
        """
        Export the batch evaluation results to a CSV file.

        Args:
            results (dict): Dictionary mapping symbols to evaluation results.
            filename (str): Filename for the CSV file.
        """
        keys = sorted({k for symbol in results for k in results[symbol]})
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Symbol"] + keys)
            for symbol, metrics in results.items():
                row = [symbol] + [metrics.get(k, "") for k in keys]
                writer.writerow(row)

    @staticmethod
    def export_to_json(results, filename="results.json"):
        """
        Export the batch evaluation results to a JSON file.

        Args:
            results (dict): Dictionary mapping symbols to evaluation results.
            filename (str): Filename for the JSON file.
        """
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)


# Example usage with sample data
if __name__ == "__main__":
    symbols_data = {
        "AAPL": {
            "quote": QuoteData(ask=175.2, bid=174.8, last_price=175.0, close_price=173.0),
            "option": OptionData(strike_price=170, option_price=6, days_to_expiration=30, bid=5.9, ask=6.1,
                                 dv=1.0, bp_effect=-0.5, max_risk=10),
            "fundamental": FundamentalData(earnings_per_share=6.5, dividend=0.88, dividend_frequency='Q', shares_outstanding=16e9),
            "volume": VolumeData(put_volume=5000, call_volume=6000),
            "historical": HistoricalData(closes=[170, 172, 171, 173, 175, 174, 176]),
            "swing": {
                "RSI_14": 25,
                "SMA_20": 174,
                "EMA_20": 175,
                "Resistance_20": 176
            }
        },
        "MSFT": {
            "quote": QuoteData(ask=330.1, bid=329.9, last_price=330.0, close_price=325.0),
            "volume": VolumeData(put_volume=4000, call_volume=5000),
            "historical": HistoricalData(closes=[320, 322, 325, 328, 330, 332, 331]),
            # No options, fundamentals, or swing data provided
        }
    }

    # Run batch evaluation
    results = StockAnalyticsAggregator.batch_evaluate(symbols_data)
    print(results)

    # Generate and print summary report
    summary = StockAnalyticsAggregator.summary_report(results)
    print("\nSummary Report:")
    for metric,
