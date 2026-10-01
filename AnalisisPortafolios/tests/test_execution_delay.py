"""Pruebas independientes de señales al cierre y ejecución una sesión después."""
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


class ExecutionDelayTests(unittest.TestCase):
    weights = {"A": .5, "B": .5}
    dates = pd.to_datetime([
        "2020-01-29", "2020-01-30", "2020-01-31", "2020-02-03", "2020-02-04",
    ])

    def prices(self, a, dates=None):
        return pd.DataFrame({"A": a, "B": 1.0},
                            index=self.dates if dates is None else dates, dtype=float)

    def assert_results_equal(self, first, second):
        self.assertEqual(set(first), set(second))
        for key in first:
            with self.subTest(result_key=key):
                if isinstance(first[key], pd.Series):
                    pd.testing.assert_series_equal(first[key], second[key])
                elif isinstance(first[key], pd.DataFrame):
                    pd.testing.assert_frame_equal(first[key], second[key])
                else:
                    self.assertEqual(first[key], second[key])

    def test_default_preserves_results_and_return_schema(self):
        prices = self.prices([1, 2, 2, 1, 1])
        for frequency in ["sin rebalanceo", "mensual", "trimestral", "anual", "bandas"]:
            policy = {"frecuencia": frequency}
            if frequency == "bandas":
                policy["banda_pp"] = 10
            with self.subTest(frequency=frequency):
                default = simulate(prices, self.weights, 100, 10, **policy)
                explicit = simulate(prices, self.weights, 100, 10,
                                    ejecucion="mismo_cierre", **policy)
                self.assertNotIn("senales", default)
                self.assert_results_equal(default, explicit)

    def test_calendar_waits_for_next_session_return_before_trading(self):
        calendars = {
            "mensual": self.dates,
            "trimestral": pd.to_datetime([
                "2020-03-27", "2020-03-30", "2020-03-31", "2020-04-01", "2020-04-02",
            ]),
            "anual": pd.to_datetime([
                "2020-12-28", "2020-12-29", "2020-12-31", "2021-01-04", "2021-01-05",
            ]),
        }
        for frequency, dates in calendars.items():
            with self.subTest(frequency=frequency):
                prices = self.prices([1, 2, 2, 1, 1], dates)
                delayed = simulate(prices, self.weights, 100, 0, frecuencia=frequency,
                                   ejecucion="siguiente_cierre")
                immediate = simulate(prices, self.weights, 100, 0, frecuencia=frequency)
                # La caída de A se aplica a las posiciones originales 100/50.
                # Retraso: 50 + 50 = 100; inmediato: 37.5 + 75 = 112.5.
                np.testing.assert_allclose(delayed["patrimonio"], [100, 150, 150, 100, 100])
                np.testing.assert_allclose(immediate["patrimonio"], [100, 150, 150, 112.5, 112.5])
                np.testing.assert_allclose(delayed["pesos"].iloc[2], [2 / 3, 1 / 3])
                self.assertEqual(list(delayed["operaciones"].index), list(dates[[0, 3]]))
                signal = delayed["senales"].iloc[0]
                self.assertEqual(signal["Fecha señal"], dates[2])
                self.assertEqual(signal["Fecha ejecución prevista"], dates[3])
                self.assertEqual(signal["Fecha ejecución"], dates[3])
                self.assertEqual(signal["Estado"], "Ejecutada")
                self.assertEqual(signal["Motivo"], f"Rebalanceo {frequency}")
                self.assertAlmostEqual(signal["Desvío señal (pp)"], 100 / 6)

    def test_pending_band_order_is_not_cancelled_when_weights_normalize(self):
        prices = self.prices([1, 2, 1, 1, 1])
        result = simulate(prices, self.weights, 100, 100, frecuencia="bandas",
                          banda_pp=10, ejecucion="siguiente_cierre")
        invested = 100 / 1.01
        np.testing.assert_allclose(result["patrimonio"],
                                   invested * np.array([1, 1.5, 1, 1, 1]))
        self.assertEqual(list(result["operaciones"].index), list(self.dates[[0, 2]]))
        trade = result["operaciones"].iloc[1]
        self.assertAlmostEqual(trade["Costo"], 0)
        self.assertAlmostEqual(trade["Volumen operado"], 0)
        self.assertEqual(len(result["senales"]), 1)
        self.assertEqual(result["senales"].iloc[0]["Estado"], "Ejecutada")

    def test_costs_use_execution_holdings_and_post_trade_weights_prevent_resignal(self):
        prices = self.prices([1, 2, 3, 3, 3])
        result = simulate(prices, self.weights, 100, 100, frecuencia="bandas",
                          banda_pp=10, ejecucion="siguiente_cierre")
        invested = 100 / 1.01
        # Al ejecutar, posiciones 1.5*I y .5*I: volumen I y costo .01*I.
        after = 1.99 * invested
        np.testing.assert_allclose(result["patrimonio"],
                                   [invested, 1.5 * invested, after, after, after])
        np.testing.assert_allclose(result["pesos_antes"].iloc[2], [.75, .25])
        np.testing.assert_allclose(result["pesos"].iloc[2:], .5, atol=1e-12)
        self.assertEqual(len(result["senales"]), 1)
        self.assertEqual(list(result["operaciones"].index), list(self.dates[[0, 2]]))
        trade = result["operaciones"].iloc[1]
        self.assertAlmostEqual(trade["Valor antes"], 2 * invested)
        self.assertAlmostEqual(trade["Volumen operado"], invested)
        self.assertAlmostEqual(trade["Costo"], .01 * invested)
        self.assertAlmostEqual(trade["Valor después"] + trade["Costo"], trade["Valor antes"])
        self.assertAlmostEqual(trade["Ventas"] - trade["Compras"], trade["Costo"])
        self.assertAlmostEqual(result["costos"].sum(), 100 - invested + .01 * invested)

    def test_initial_and_terminal_dates_are_not_rebalanced_and_late_signal_is_audited(self):
        short_dates = pd.to_datetime(["2020-01-30", "2020-01-31", "2020-02-03"])
        for policy in [{"frecuencia": "mensual"}, {"frecuencia": "bandas", "banda_pp": 10}]:
            with self.subTest(policy=policy):
                prices = self.prices([1, 2, 3], short_dates)
                result = simulate(prices, self.weights, 100, 0,
                                  ejecucion="siguiente_cierre", **policy)
                np.testing.assert_allclose(result["patrimonio"], [100, 150, 200])
                self.assertEqual(list(result["operaciones"].index), [short_dates[0]])
                self.assertEqual(len(result["senales"]), 1)
                signal = result["senales"].iloc[0]
                self.assertEqual(signal["Fecha señal"], short_dates[1])
                self.assertEqual(signal["Fecha ejecución prevista"], short_dates[2])
                self.assertTrue(pd.isna(signal["Fecha ejecución"]))
                self.assertEqual(signal["Estado"], "Sin sesión interior para ejecutar")
        # Comprar en fin de mes no genera otra orden. Cruzar solo al final tampoco.
        edge_dates = pd.to_datetime(["2020-01-31", "2020-02-03", "2020-02-04"])
        for policy in [{"frecuencia": "mensual"}, {"frecuencia": "bandas", "banda_pp": 10}]:
            result = simulate(self.prices([1, 1, 3], edge_dates), self.weights, 100, 0,
                              ejecucion="siguiente_cierre", **policy)
            self.assertEqual(list(result["operaciones"].index), [edge_dates[0]])
            self.assertTrue(result["senales"].empty)

    def test_future_prices_do_not_change_prior_signals_or_holdings(self):
        prices = self.prices([1, 2, 3, 3, 3])
        changed = prices.copy()
        changed.iloc[2:, 0] = [1, .5, .5]
        results = [simulate(p, self.weights, 100, 10, frecuencia="bandas", banda_pp=10,
                            ejecucion="siguiente_cierre") for p in [prices, changed]]
        for key in ["patrimonio", "rendimientos", "costos", "drawdown"]:
            pd.testing.assert_series_equal(results[0][key].iloc[:2], results[1][key].iloc[:2])
        for key in ["pesos", "pesos_antes"]:
            pd.testing.assert_frame_equal(results[0][key].iloc[:2], results[1][key].iloc[:2])
        pd.testing.assert_series_equal(results[0]["senales"].iloc[0], results[1]["senales"].iloc[0])
        for result in results:
            self.assertEqual(result["operaciones"].index[1], self.dates[2])

    def test_buy_hold_and_initial_purchase_are_identical_in_both_modes(self):
        prices = self.prices([1, 2, 3, 1, 1.5])
        immediate = simulate(prices, self.weights, 100, 10, frecuencia="sin rebalanceo")
        delayed = simulate(prices, self.weights, 100, 10, frecuencia="sin rebalanceo",
                           ejecucion="siguiente_cierre")
        signals = delayed.pop("senales")
        self.assertTrue(signals.empty)
        self.assertEqual(list(signals.columns), [
            "Fecha señal", "Fecha ejecución prevista", "Fecha ejecución",
            "Estado", "Motivo", "Desvío señal (pp)",
        ])
        self.assert_results_equal(immediate, delayed)
        for policy in [{"frecuencia": "mensual"}, {"frecuencia": "bandas", "banda_pp": 10}]:
            result = simulate(prices, self.weights, 100, 10,
                              ejecucion="siguiente_cierre", **policy)
            pd.testing.assert_series_equal(result["operaciones"].iloc[0],
                                           immediate["operaciones"].iloc[0])
            self.assertEqual(result["patrimonio"].iloc[0], immediate["patrimonio"].iloc[0])

    def test_invalid_execution_modes_raise_value_error(self):
        prices = self.prices([1, 2, 3, 1, 1.5])
        for mode in [None, True, False, 0, 1, "", "siguiente_apertura", "Siguiente_cierre", []]:
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                simulate(prices, self.weights, ejecucion=mode)


if __name__ == "__main__":
    unittest.main()
