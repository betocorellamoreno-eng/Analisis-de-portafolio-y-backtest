"""Pruebas independientes de RF histórica y ratios netos de entrada."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = Path(sys.argv.pop(1)) if len(sys.argv) > 1 else ROOT / "Análisis y backtesting de portafolios.ipynb"
document = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
source = next("".join(cell["source"]) for cell in document["cells"]
              if "".join(cell["source"]).startswith(
                  "# Métricas ajustadas por riesgo: referencia diaria y costos de entrada."))
namespace = {"np": np, "pd": pd}
exec(compile(source, str(NOTEBOOK), "exec"), namespace)
align_rf = namespace["alinear_rf_diario"]
net_returns = namespace["rendimientos_netos_con_entrada"]
ratios = namespace["calcular_ratios"]


class RiskMetricsTests(unittest.TestCase):
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])

    def series(self, values, dates=None):
        return pd.Series(values, index=self.dates if dates is None else dates, dtype=float)

    def test_cme_example_uses_all_observations_for_downside(self):
        # CME/Red Rock annual example: two shortfalls among eight observations.
        dates = pd.date_range("2010-12-31", periods=8, freq="YE")
        returns = self.series([.17, .15, .23, -.05, .12, .09, .13, -.04], dates)
        result = ratios(returns, returns * 0, periodos_anuales=1)
        expected_downside = np.sqrt((.05 ** 2 + .04 ** 2) / 8)
        self.assertEqual(set(result.index), {
            "Sharpe", "Sortino", "Exceso medio anualizado", "Volatilidad del exceso",
            "Downside anualizado", "Intervalos",
        })
        self.assertAlmostEqual(result["Sortino"], .10 / expected_downside, places=10)
        self.assertAlmostEqual(result["Downside anualizado"], expected_downside, places=12)
        self.assertEqual(result["Intervalos"], 8)

    def test_sharpe_uses_excess_and_sample_standard_deviation(self):
        returns = self.series([.011, -.018, .033, -.006])
        rf = self.series([.001, .002, .003, .004])
        result = ratios(returns, rf, periodos_anuales=252)
        excess = np.array([.01, -.02, .03, -.01])
        sample_sd = np.sqrt(np.sum((excess - excess.mean()) ** 2) / 3)
        self.assertAlmostEqual(result["Sharpe"], np.sqrt(252) * excess.mean() / sample_sd)
        self.assertAlmostEqual(result["Exceso medio anualizado"], 252 * excess.mean())
        self.assertAlmostEqual(result["Volatilidad del exceso"], np.sqrt(252) * sample_sd)
        self.assertAlmostEqual(result["Sortino"],
                               np.sqrt(252) * excess.mean() / np.sqrt((.02 ** 2 + .01 ** 2) / 4))

    def test_constant_returns_leave_undefined_ratios_nan(self):
        zero = self.series([0, 0, 0, 0])
        negative = ratios(zero - .1, zero, periodos_anuales=1)
        self.assertTrue(np.isnan(negative["Sharpe"]))
        self.assertAlmostEqual(negative["Sortino"], -1.)
        positive = ratios(zero + .1, zero, periodos_anuales=1)
        self.assertTrue(np.isnan(positive["Sharpe"]))
        self.assertTrue(np.isnan(positive["Sortino"]))
        flat = ratios(zero, zero)
        self.assertTrue(np.isnan(flat["Sharpe"]))
        self.assertTrue(np.isnan(flat["Sortino"]))

    def test_initial_cost_is_in_first_real_interval_exactly_once(self):
        nav = self.series([99, 103, 97, 110])
        result = net_returns({"patrimonio": nav, "capital_inicial": 100.})
        pd.testing.assert_index_equal(result.index, self.dates[1:])
        np.testing.assert_allclose(result.to_numpy(), [103 / 100 - 1, 97 / 103 - 1, 110 / 97 - 1])
        self.assertAlmostEqual(np.prod(1 + result), 110 / 100, places=12)
        self.assertEqual(len(result), len(nav) - 1)
        self.assertAlmostEqual(nav.iloc[0] / 100 * np.prod(1 + nav.pct_change().iloc[1:]),
                               np.prod(1 + result), places=12)

    def test_rf_percent_conversion_no_extra_weekend_accrual(self):
        extra_dates = pd.to_datetime(["2019-12-31", *self.dates.strftime("%Y-%m-%d"), "2020-01-08"])
        rf = self.series([.9, .01, .02, .03, .04, .9], extra_dates)
        aligned = align_rf(self.dates, rf)
        pd.testing.assert_index_equal(aligned.index, self.dates[1:])
        np.testing.assert_allclose(aligned.to_numpy(), [.0002, .0003, .0004])
        # Monday's published daily RF is used once, without multiplying by three.
        self.assertAlmostEqual(aligned.loc[pd.Timestamp("2020-01-06")], .0003)

    def test_rf_calendar_rejects_missing_or_extra_interior_dates(self):
        rf = self.series([.01, .01, .01, .01])
        with self.assertRaises(ValueError):
            align_rf(self.dates, rf.drop(self.dates[2]))
        with self.assertRaises(ValueError):
            align_rf(self.dates, pd.concat([rf, pd.Series([.01], index=pd.to_datetime(["2020-01-04"]))]).sort_index())
        with self.assertRaises(ValueError):
            align_rf(self.dates, rf.iloc[::-1])
        with self.assertRaises(ValueError):
            align_rf(self.dates, pd.concat([rf.iloc[:1], rf]))
        for value in [np.nan, np.inf, -100.]:
            with self.subTest(rf=value), self.assertRaises(ValueError):
                align_rf(self.dates, self.series([.01, value, .01, .01]))

    def test_ratios_reject_shifted_unsorted_duplicate_or_short_indices(self):
        returns = self.series([.01, -.02, .03, -.01])
        with self.assertRaises(ValueError):
            ratios(returns, self.series([0, 0, 0, 0], self.dates + pd.Timedelta(days=1)))
        for bad in [returns.iloc[::-1], pd.concat([returns.iloc[:1], returns]),
                    returns.iloc[:1], pd.Series([.1, .2], index=[0, 1]),
                    pd.Series([.1, .2], index=pd.to_datetime(["2020-01-01", None]))]:
            with self.subTest(index=str(bad.index)), self.assertRaises(ValueError):
                ratios(bad, bad * 0)

    def test_ratios_reject_invalid_values_and_annualization(self):
        returns = self.series([.01, -.02, .03, -.01])
        for factor in [0, -1, np.nan, np.inf, True, "252"]:
            with self.subTest(factor=factor), self.assertRaises(ValueError):
                ratios(returns, returns * 0, periodos_anuales=factor)
        for bad in [np.nan, np.inf, -1., -1.1]:
            invalid = self.series([.01, bad, .03, -.01])
            with self.subTest(returns=bad), self.assertRaises(ValueError):
                ratios(invalid, returns * 0)
            with self.subTest(rf=bad), self.assertRaises(ValueError):
                ratios(returns, invalid)

    def test_net_returns_and_price_calendar_reject_invalid_inputs(self):
        nav = self.series([99, 103, 97, 110])
        for capital in [0, -100, np.inf, np.nan, True]:
            with self.subTest(capital=capital), self.assertRaises(ValueError):
                net_returns({"patrimonio": nav, "capital_inicial": capital})
        for bad in [nav.iloc[:2], nav.iloc[::-1], self.series([99, 0, 97, 110]),
                    self.series([99, np.nan, 97, 110])]:
            with self.subTest(nav=str(bad)), self.assertRaises(ValueError):
                net_returns({"patrimonio": bad, "capital_inicial": 100.})
        rf = self.series([.01] * 4)
        for dates in [self.dates[:2], self.dates[::-1], self.dates[[0, 1, 1, 3]],
                      pd.Index([0, 1, 2]), pd.to_datetime(["2020-01-02", None, "2020-01-07"])]:
            with self.subTest(dates=str(dates)), self.assertRaises(ValueError):
                align_rf(dates, rf)


if __name__ == "__main__":
    unittest.main(verbosity=2)
