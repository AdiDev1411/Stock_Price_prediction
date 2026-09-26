import unittest
import pandas as pd

from indicators import add_indicators


class AddIndicatorsTest(unittest.TestCase):
    def test_add_indicators_returns_expected_columns(self):
        df = pd.DataFrame(
            {
                "Close": [100, 101, 102, 103, 104],
                "Volume": [1000, 1100, 1200, 1300, 1400],
            }
        )

        result = add_indicators(df)

        expected_columns = {
            "RSI",
            "MACD",
            "MACD_Signal",
            "EMA20",
            "EMA50",
            "SMA50",
            "SMA100",
            "EMA100",
            "Return_1d",
            "Return_5d",
            "Volume_Change",
            "Volatility_20d",
            "Price_vs_EMA20",
        }

        self.assertTrue(expected_columns.issubset(set(result.columns)))
        self.assertGreaterEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
