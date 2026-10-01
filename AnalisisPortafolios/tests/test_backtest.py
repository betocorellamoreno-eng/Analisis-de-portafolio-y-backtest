"""Pruebas del motor del notebook: ejecutar con el entorno portfolio-analysis."""
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
              if "".join(cell["source"]).startswith("# Motor del backtest:"))
namespace = {}
exec(compile(source, str(NOTEBOOK), "exec"), namespace)
simulate = namespace["simular_portafolio"]
summarize = namespace["resumir_backtest"]


class BacktestTests(unittest.TestCase):
    def prices(self, a, b):
        return pd.DataFrame({"A": a, "B": b}, index=pd.to_datetime([
            "2020-01-29", "2020-01-30", "2020-01-31", "2020-02-03", "2020-02-28"
        ]))

    def test_drift_and_timing_with_hand_calculation(self):
        # A se duplica antes de fin de mes: 50/50 -> 100/50, y luego 75/75.
        # Al caer A a la mitad después del rebalanceo: 37.5 + 75 = 112.5.
        prices = self.prices([1, 2, 2, 1, 1], [1, 1, 1, 1, 1])
        result = simulate(prices, {"A": .5, "B": .5}, 100, 0)
        np.testing.assert_allclose(result["patrimonio"], [100, 150, 150, 112.5, 112.5])
        np.testing.assert_allclose(result["pesos"].iloc[1], [2/3, 1/3])
        np.testing.assert_allclose(result["pesos"].iloc[2], [.5, .5])
        self.assertEqual(list(result["operaciones"].index), list(prices.index[[0, 2]]))
        hold = simulate(prices, {"A": .5, "B": .5}, 100, 0, False)
        self.assertAlmostEqual(hold["patrimonio"].iloc[-1], 100)

    def test_costs_paid_from_holdings_and_exact_targets(self):
        prices = self.prices([1, 2, 2, 1, 1], [1, 1, 1, 1, 1])
        result = simulate(prices, {"A": .5, "B": .5}, 100, 100)
        # Entrada: 100/1.01; rebalanceo 50/50 de dos posiciones: volumen = diferencia.
        initial = 100 / 1.01
        fee = .01 * (.5 * initial)
        after = 1.5 * initial - fee
        self.assertAlmostEqual(result["patrimonio"].iloc[2], after)
        self.assertAlmostEqual(result["patrimonio"].iloc[-1], after * .75)
        trade = result["operaciones"].iloc[1]
        self.assertAlmostEqual(trade["Ventas"] - trade["Compras"], trade["Costo"])
        self.assertAlmostEqual(trade["Costo"], fee)
        np.testing.assert_allclose(result["pesos"].iloc[2], [.5, .5])

    def test_flat_prices_no_rebalance_charge_and_entry_drawdown(self):
        result = simulate(self.prices([1]*5, [1]*5), {"A": .5, "B": .5}, 100, 100)
        np.testing.assert_allclose(result["patrimonio"], 100 / 1.01)
        self.assertAlmostEqual(result["operaciones"]["Costo"].iloc[1], 0)
        self.assertAlmostEqual(result["drawdown"].min(), 1 / 1.01 - 1)
        self.assertAlmostEqual(summarize(result)["Volatilidad anualizada"], 0)

    def test_buy_hold_identity_scaling_and_return_compounding(self):
        prices = self.prices([1, 1.2, .8, 2, 1.5], [1, .9, 1.1, 1, 1.2])
        result = simulate(prices, {"A": .6, "B": .4}, 100, 10, False)
        expected = 100 / 1.001 * (.6 * prices["A"] + .4 * prices["B"])
        np.testing.assert_allclose(result["patrimonio"], expected)
        self.assertAlmostEqual((1 + result["rendimientos"]).prod(), expected.iloc[-1]/100)
        larger = simulate(prices, {"A": .6, "B": .4}, 1000, 10, False)
        np.testing.assert_allclose(larger["patrimonio"], result["patrimonio"] * 10)

    def test_future_prices_do_not_change_prior_results(self):
        prices = self.prices([1, 2, 2, 1, 1], [1]*5)
        changed = prices.copy()
        changed.iloc[3:, 0] *= 20
        a = simulate(prices, {"A": .5, "B": .5})
        b = simulate(changed, {"A": .5, "B": .5})
        pd.testing.assert_series_equal(a["patrimonio"].iloc[:3], b["patrimonio"].iloc[:3])
        pd.testing.assert_frame_equal(a["operaciones"], b["operaciones"])

    def test_invalid_inputs_fail_explicitly(self):
        prices = self.prices([1]*5, [1]*5)
        for weights, capital, cost in [({"A": .3, "B": .3},100,10),
                                      ({"A": 1.2, "B": -.2},100,10),
                                      ({"A": .5, "C": .5},100,10),
                                      ({"A": .5, "B": .5},0,10),
                                      ({"A": .5, "B": .5},100,-1),
                                      ({"A": .5, "B": .5},100,10000)]:
            with self.subTest(weights=weights, capital=capital, cost=cost):
                with self.assertRaises(ValueError):
                    simulate(prices, weights, capital, cost)
        for bad in [prices.iloc[::-1], pd.concat([prices, prices.iloc[-1:]]),
                    prices.assign(A=np.nan), prices.assign(A=0)]:
            with self.assertRaises(ValueError):
                simulate(bad, {"A": .5, "B": .5})

    def test_quarterly_and_annual_calendar_and_counts(self):
        dates = pd.bdate_range("2020-01-02", "2022-01-04")
        prices = pd.DataFrame({"A": np.linspace(1, 2, len(dates)), "B": 1.0}, index=dates)
        expected = {
            "trimestral": ["2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31",
                           "2021-03-31", "2021-06-30", "2021-09-30", "2021-12-31"],
            "anual": ["2020-12-31", "2021-12-31"],
        }
        for frequency, dates_expected in expected.items():
            with self.subTest(frequency=frequency):
                result = simulate(prices, {"A": .5, "B": .5}, frecuencia=frequency)
                trades = result["operaciones"].iloc[1:]
                self.assertEqual(list(trades.index), list(pd.to_datetime(dates_expected)))
                self.assertEqual(summarize(result)["Rebalanceos"], len(dates_expected))
                np.testing.assert_allclose(result["pesos"].loc[trades.index], .5, atol=1e-12)
                np.testing.assert_allclose(trades["Ventas"] - trades["Compras"], trades["Costo"], atol=1e-9)

    def test_quarterly_timing_hand_calculation_and_terminal_exclusion(self):
        dates = pd.to_datetime(["2020-01-02", "2020-01-31", "2020-03-31", "2020-04-01", "2020-06-30"])
        prices = pd.DataFrame({"A": [1, 2, 2, 1, 1], "B": 1.0}, index=dates)
        result = simulate(prices, {"A": .5, "B": .5}, 100, 0, frecuencia="trimestral")
        np.testing.assert_allclose(result["patrimonio"], [100, 150, 150, 112.5, 112.5])
        np.testing.assert_allclose(result["pesos"].iloc[1], [2/3, 1/3])
        self.assertEqual(list(result["operaciones"].index), list(dates[[0, 2]]))
        annual = simulate(prices, {"A": .5, "B": .5}, 100, 0, frecuencia="anual")
        self.assertEqual(len(annual["operaciones"]), 1)
        self.assertAlmostEqual(annual["patrimonio"].iloc[-1], 100)

    def test_explicit_frequency_compatibility_and_invalid_value(self):
        prices = self.prices([1, 2, 2, 1, 1], [1]*5)
        for flag, frequency in [(True, "mensual"), (False, "sin rebalanceo")]:
            old = simulate(prices, {"A": .5, "B": .5}, rebalanceo_mensual=flag)
            new = simulate(prices, {"A": .5, "B": .5}, frecuencia=frequency)
            pd.testing.assert_series_equal(old["patrimonio"], new["patrimonio"])
            pd.testing.assert_frame_equal(old["operaciones"], new["operaciones"])
        with self.assertRaises(ValueError):
            simulate(prices, {"A": .5, "B": .5}, frecuencia="semanal")

    def test_initial_date_at_period_end_is_not_rebalanced(self):
        prices = pd.DataFrame({"A": [1, 2, 3], "B": [1, 1, 1]},
                              index=pd.to_datetime(["2020-12-31", "2021-01-04", "2021-01-05"]))
        for frequency in ["mensual", "trimestral", "anual"]:
            result = simulate(prices, {"A": .5, "B": .5}, frecuencia=frequency)
            self.assertEqual(len(result["operaciones"]), 1)


if __name__ == "__main__":
    unittest.main()
