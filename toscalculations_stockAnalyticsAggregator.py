import math
import numpy as np
import requests
import csv
import json

# Class to represent basic quote data for a stock
class QuoteData:
    def __init__(self, bid=None, ask=None, last_price=None, close_price=None):
        """
        Initialize QuoteData with bid, ask, last traded price, and previous close price.
        """
        self.bid = bid
        self.ask = ask
        self.last_price = last_price
        self.close_price = close_price

    def mark_price(self):
        """
        Calculate the mark price as the midpoint between bid and ask.
        Returns None if calculation fails.
        """
        try:
            return (self.bid + self.ask) / 2
        except:
            return None

    def net_change(self):
        """
        Calculate the net change in price from previous close to last price.
        Returns None if calculation fails.
        """
        try:
            return self.last_price - self.close_price
        except:
            return None

    def percent_change(self):
        """
        Calculate the percentage change from previous close to last price.
        Returns None if calculation fails.
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
        Initialize OptionData with option parameters such as strike price,
        option market price, time to expiration, bid/ask prices, and other risk metrics.
        """
        self.strike_price = strike_price
        self.option_price = option_price
        self.days_to_expiration = days_to_expiration
        self.bid = bid
        self.ask = ask
        self.last_price = last_price
        self.dv = dv  # Delta value or similar metric for return calculations
        self.bp_effect = bp_effect  # Basis point effect for return calculations
        self.max_risk = max_risk  # Maximum risk exposure for return on risk

    def mark_price_option(self):
        """
        Calculate the option mark price as the midpoint of bid and ask.
        Returns None if calculation fails.
        """
        try:
            return (self.bid + self.ask) / 2
        except:
            return None

    def intrinsic_value_call(self, last_price):
        """
        Calculate intrinsic value for a call option.
        Intrinsic value = max(last_price - strike_price, 0)
        """
        try:
            return max(last_price - self.strike_price, 0)
        except:
            return None

    def intrinsic_value_put(self, last_price):
        """
        Calculate intrinsic value for a put option.
        Intrinsic value = max(strike_price - last_price, 0)
        """
        try:
            return max(self.strike_price - last_price, 0)
        except:
            return None

    def extrinsic_value(self, intrinsic):
        """
        Calculate the extrinsic (time) value of the option.
        Extrinsic value = option price - intrinsic value
        """
        try:
            return self.option_price - intrinsic
        except:
            return None

    def covered_return(self, extrinsic, mark_price):
        """
        Calculate covered call return annualized based on extrinsic value and mark price.
        Formula: (extrinsic / mark_price) * (365 / days_to_expiration)
        """
        try:
            return (extrinsic / mark_price) * (365 / self.days_to_expiration)
        except:
            return None

    def return_on_capital(self, mark_price):
        """
        Calculate return on capital for the option position.
        Formula: (mark_price * dv / -bp_effect) * (365 / days_to_expiration)
        """
        try:
            return (mark_price * self.dv / -self.bp_effect) * (365 / self.days_to_expiration)
        except:
            return None

    def return_on_risk(self, mark_price):
        """
        Calculate return on risk.
        Formula: (mark_price / max_risk) * (365 / days_to_expiration)
        """
        try:
            return (mark_price / self.max_risk) * (365 / self.days_to_expiration)
        except:
            return None


# Class to handle volatility data and related computations
class VolatilityData:
    def __init__(self, front_vol=None, back_vol=None, t1=None, t2=None, last_price=None):
        """
        Initialize VolatilityData with front and back implied volatilities,
        corresponding time periods t1 and t2, and current last price.
        """
        self.front_vol = front_vol
        self.back_vol = back_vol
        self.t1 = t1
        self.t2 = t2
        self.last_price = last_price

    def volatility_difference(self):
        """
        Calculate the difference between front and back volatilities.
        """
        try:
            return self.front_vol - self.back_vol
        except:
            return None

    def weighted_back_volatility(self):
        """
        Calculate weighted back volatility using squared volatilities and time periods.
        Formula: sqrt( ((back_vol^2 * t2) - (front_vol^2 * t1)) / (t2 - t1) )
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
        Used internally for expected move calculations.
        """
        try:
            return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
        except:
            return None

    def expected_move(self, volatility):
        """
        Calculate expected move based on volatility.
        Formula involves normal CDF and lognormal adjustment.
        """
        try:
            n_vol = 2 * self.norm_cdf(volatility) - 1
            return self.last_price * math.exp(volatility**2 / 2) * n_vol
        except:
            return None

    def front_expected_move(self):
        """
        Expected move for front period volatility.
        """
        try:
            return self.expected_move(math.sqrt(self.t1) * self.front_vol)
        except:
            return None

    def back_expected_move(self):
        """
        Expected move for back period volatility.
        """
        try:
            return self.expected_move(math.sqrt(self.t2) * self.back_vol)
        except:
            return None

    def expected_move_difference(self, wbv):
        """
        Difference in expected move between periods using weighted back volatility.
        """
        try:
            return self.expected_move(math.sqrt(self.t2 - self.t1) * (wbv ** 2))
        except:
            return None

    def market_maker_move(self, wbv):
        """
        Calculate market maker expected move.
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
        Initialize fundamental data including earnings, dividend info, shares outstanding.
        """
        self.earnings_per_share = earnings_per_share
        self.dividend = dividend
        self.dividend_frequency = dividend_frequency
        self.shares_outstanding = shares_outstanding
        self.last_price = last_price

    def pe_ratio(self, last_price):
        """
        Calculate Price-to-Earnings (P/E) ratio.
        """
        try:
            return last_price / self.earnings_per_share
        except:
            return None

    def dividend_yield(self, last_price):
        """
        Calculate dividend yield annualized based on frequency.
        Frequency multipliers: A=1, Q=4, M=12
        """
        try:
            multiplier = {"A": 1, "Q": 4, "M": 12}.get(self.dividend_frequency.upper(), 1)
            return (self.dividend * multiplier) / last_price
        except:
            return None

    def market_cap(self, last_price):
        """
        Calculate market capitalization.
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
        """
        self.put_volume = put_volume
        self.call_volume = call_volume

    def put_call_ratio(self):
        """
        Calculate put-call volume ratio.
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
        """
        self.closes = closes or []

    def historical_volatility(self):
        """
        Calculate annualized historical volatility from log returns of closing prices.
        """
        try:
            log_returns = np.log(np.array(self.closes[1:]) / np.array(self.closes[:-1]))
            return np.std(log_returns, ddof=1) * math.sqrt(252)  # 252 trading days in a year
        except:
            return None


# Aggregator class to combine data types and run analytics
class StockAnalyticsAggregator:
    def __init__(self, quote_data=None, option_data=None, volatility_data=None,
                 fundamental_data=None, volume_data=None, historical_data=None):
        """
        Initialize aggregator with all relevant data objects for a stock.
        """
        self.quote = quote_data
        self.option = option_data
        self.volatility = volatility_data
        self.fundamental = fundamental_data
        self.volume = volume_data
        self.historical = historical_data

    def evaluate(self):
        """
        Evaluate all available analytics for the stock.
        Each calculation is wrapped in try/except to safely skip failures.
        Returns a dictionary of results.
        """
        results = {}

        try:
            if self.quote:
                results["mark_price"] = self.quote.mark_price()
                results["net_change"] = self.quote.net_change()
                results["percent_change"] = self.quote.percent_change()
        except:
            pass

        try:
            if self.option and self.quote:
                intrinsic_call = self.option.intrinsic_value_call(self.quote.last_price)
                intrinsic_put = self.option.intrinsic_value_put(self.quote.last_price)
                extrinsic = self.option.extrinsic_value(intrinsic_call)
                # Prefer option's mark price, fallback to quote mark price
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

        try:
            if self.fundamental and self.quote:
                results.update({
                    "pe_ratio": self.fundamental.pe_ratio(self.quote.last_price),
                    "dividend_yield": self.fundamental.dividend_yield(self.quote.last_price),
                    "market_cap": self.fundamental.market_cap(self.quote.last_price)
                })
        except:
            pass

        try:
            if self.volume:
                results["put_call_ratio"] = self.volume.put_call_ratio()
        except:
            pass

        try:
            if self.historical:
                results["historical_volatility"] = self.historical.historical_volatility()
        except:
            pass

        return results

    @staticmethod
    def batch_evaluate(symbol_data_list):
        """
        Evaluate a batch of stocks given a dict of symbol->data objects.
        Returns a dictionary of symbol -> evaluation results.
        """
        results = {}
        for symbol, data in symbol_data_list.items():
            aggregator = StockAnalyticsAggregator(
                quote_data=data.get("quote"),
                option_data=data.get("option"),
                volatility_data=data.get("volatility"),
                fundamental_data=data.get("fundamental"),
                volume_data=data.get("volume"),
                historical_data=data.get("historical")
            )
            results[symbol] = aggregator.evaluate()
        return results

    @staticmethod
    def summary_report(results):
        """
        Generate a summary report with average and top performer for each numeric metric.
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
        """
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)


# Example usage with sample data
symbols_data = {
    "AAPL": {
        "quote": QuoteData(ask=175.2, bid=174.8, last_price=175.0, close_price=173.0),
        "option": OptionData(strike_price=170, option_price=6, days_to_expiration=30, bid=5.9, ask=6.1,
                             dv=1.0, bp_effect=-0.5, max_risk=10),
        "fundamental": FundamentalData(earnings_per_share=6.5, dividend=0.88, dividend_frequency='Q', shares_outstanding=16e9),
    },
    "MSFT": {
        "quote": QuoteData(ask=330.1, bid=329.9, last_price=330.0, close_price=325.0),
    }
}

# Run batch evaluation
results = StockAnalyticsAggregator.batch_evaluate(symbols_data)
print(results)

# Generate and print summary report
summary = StockAnalyticsAggregator.summary_report(results)
print("\nSummary Report:")
for metric, data in summary.items():
    print(f"{metric}: Avg = {data['average']:.2f}, Top = {data['top_symbol']} ({data['top_value']:.2f})")

# Export results
StockAnalyticsAggregator.export_to_csv(results, "analytics_results.csv")
StockAnalyticsAggregator.export_to_json(results, "analytics_results.json")

