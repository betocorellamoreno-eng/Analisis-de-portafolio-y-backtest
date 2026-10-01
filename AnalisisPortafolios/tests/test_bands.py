"""Pruebas independientes del rebalanceo por desviaciones en puntos porcentuales."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = Path(sys.argv.pop(1)) if len(sys.argv) > 1 else ROOT / "Análisis y backtesting de portafolios.ipynb"
document = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
namespace = {}
for prefix in ["# Motor del backtest:",
               "# Investigación por bandas: resúmenes y comparación pareada en ventanas."]:
    source = next("".join(cell["source"]) for cell in document["cells"]
                  if "".join(cell["source"]).startswith(prefix))
    exec(compile(source, str(NOTEBOOK), "exec"), namespace)
simulate = namespace["simular_portafolio"]
summarize = namespace["resumir_backtest"]
build_policies = namespace["construir_politicas_bandas"]


class BandBacktestTests(unittest.TestCase):
    weights = {"A": .5, "B": .5}

    @staticmethod
    def prices(a, b=None):
        return pd.DataFrame({"A": a, "B": [1.] * 5 if b is None else b},
                            index=pd.to_datetime([
                                "2020-01-29", "2020-01-30", "2020-01-31",
                                "2020-02-03", "2020-02-04",
                            ]), dtype=float)

    def test_strict_threshold_after_return_uses_percentage_points(self):
        prices = self.prices([1, 1.5, 2, 2, 2])
        result = simulate(prices, self.weights, 100, 0, frecuencia="bandas", banda_pp=10)
        # 50/50 -> 60/40: desviación EXACTA de 10 pp; aún no hay operación.
        # 50/50 -> 2/3, 1/3 al día siguiente: 16.67 pp activa el rebalanceo.
        np.testing.assert_allclose(result["patrimonio"], [100, 125, 150, 150, 150])
        np.testing.assert_allclose(result["pesos"].iloc[1], [.6, .4], atol=1e-12)
        np.testing.assert_allclose(result["pesos_antes"].iloc[2], [2 / 3, 1 / 3], atol=1e-12)
        np.testing.assert_allclose(result["pesos"].iloc[2], [.5, .5], atol=1e-12)
        self.assertEqual(list(result["operaciones"].index), list(prices.index[[0, 2]]))
        self.assertEqual(summarize(result)["Rebalanceos"], 1)

    def test_costs_are_self_financed_and_restore_all_target_weights(self):
        prices = self.prices([1, 2, 2, 2, 2])
        result = simulate(prices, self.weights, 100, 100, frecuencia="bandas", banda_pp=10)
        invested = 100 / 1.01
        # Tras duplicarse A, el volumen para volver a 50/50 es .5 * invertido.
        fee = .01 * .5 * invested
        expected = 1.5 * invested - fee
        np.testing.assert_allclose(result["patrimonio"].iloc[1:], expected)
        trade = result["operaciones"].iloc[1]
        self.assertAlmostEqual(trade["Costo"], fee)
        self.assertAlmostEqual(trade["Valor antes"], 1.5 * invested)
        self.assertAlmostEqual(trade["Valor después"] + trade["Costo"], trade["Valor antes"])
        self.assertAlmostEqual(trade["Ventas"] - trade["Compras"], trade["Costo"])
        self.assertAlmostEqual(trade["Costo"], .01 * (trade["Compras"] + trade["Ventas"]))
        np.testing.assert_allclose(result["pesos"].iloc[1:], .5, atol=1e-12)
        self.assertAlmostEqual(result["costos"].sum(), 100 - invested + fee)
        self.assertEqual(summarize(result)["Rebalanceos"], 1)

    def test_synchronous_assets_and_wide_band_match_buy_and_hold(self):
        cases = [
            (self.prices([1, 2, 3, 2, 4], [1, 2, 3, 2, 4]), 1),
            (self.prices([1, 2, 3, 4, 5]), 49),
        ]
        for prices, band in cases:
            with self.subTest(band=band):
                result = simulate(prices, self.weights, 100, 10, frecuencia="bandas", banda_pp=band)
                expected = 100 / 1.001 * prices.div(prices.iloc[0]).mul(pd.Series(self.weights)).sum(axis=1)
                np.testing.assert_allclose(result["patrimonio"], expected)
                self.assertEqual(list(result["operaciones"].index), [prices.index[0]])
                self.assertEqual(summarize(result)["Rebalanceos"], 0)
                self.assertAlmostEqual(result["costos"].sum(), 100 - 100 / 1.001)

    def test_crossing_only_on_terminal_date_does_not_trade(self):
        prices = self.prices([1, 1, 1, 1, 3])
        result = simulate(prices, self.weights, 100, 0, frecuencia="bandas", banda_pp=10)
        self.assertEqual(list(result["operaciones"].index), [prices.index[0]])
        self.assertEqual(summarize(result)["Rebalanceos"], 0)
        self.assertAlmostEqual(result["patrimonio"].iloc[-1], 200)
        np.testing.assert_allclose(result["pesos"].iloc[-1], [.75, .25], atol=1e-12)

    def test_future_prices_do_not_change_past_band_decisions(self):
        prices = self.prices([1, 2, 2, 2, 2])
        changed = prices.copy()
        changed.iloc[3:, 0] *= 20
        original = simulate(prices, self.weights, 100, 10, frecuencia="bandas", banda_pp=10)
        altered = simulate(changed, self.weights, 100, 10, frecuencia="bandas", banda_pp=10)
        for key in ["patrimonio", "rendimientos", "costos", "drawdown"]:
            pd.testing.assert_series_equal(original[key].iloc[:3], altered[key].iloc[:3])
        for key in ["pesos", "pesos_antes"]:
            pd.testing.assert_frame_equal(original[key].iloc[:3], altered[key].iloc[:3])
        cutoff = prices.index[2]
        pd.testing.assert_frame_equal(original["operaciones"].loc[:cutoff],
                                      altered["operaciones"].loc[:cutoff])

    def test_legacy_monthly_and_buy_hold_api_are_preserved(self):
        prices = self.prices([1, 2, 2, 1, 1])
        for flag, frequency in [(True, "mensual"), (False, "sin rebalanceo")]:
            with self.subTest(frequency=frequency):
                legacy = simulate(prices, self.weights, 100, 10, flag)
                explicit = simulate(prices, self.weights, 100, 10,
                                    frecuencia=frequency, banda_pp=None)
                pd.testing.assert_series_equal(legacy["patrimonio"], explicit["patrimonio"])
                pd.testing.assert_frame_equal(legacy["operaciones"], explicit["operaciones"])
                pd.testing.assert_frame_equal(legacy["pesos"], explicit["pesos"])
                pd.testing.assert_series_equal(summarize(legacy), summarize(explicit))

    def test_invalid_thresholds_and_bands_on_other_frequencies_fail(self):
        prices = self.prices([1, 2, 2, 1, 1])
        for band in [None, True, False, np.bool_(True), 0, -1, 100, 101,
                     np.nan, np.inf, -np.inf, "10", [10]]:
            with self.subTest(band=band), self.assertRaises(ValueError):
                simulate(prices, self.weights, frecuencia="bandas", banda_pp=band)
        for frequency in [None, "sin rebalanceo", "mensual", "trimestral", "anual"]:
            with self.subTest(frequency=frequency), self.assertRaises(ValueError):
                simulate(prices, self.weights, frecuencia=frequency, banda_pp=10)

    def test_policy_builder_validates_original_types_and_bounds(self):
        invalid = [True, False, np.bool_(True), None, "5", np.nan, np.inf,
                   -np.inf, 0, -1, 100]
        for value in invalid:
            with self.subTest(value=value, location="lista"), self.assertRaises(ValueError):
                build_policies([2.5, value], 5)
            with self.subTest(value=value, location="principal"), self.assertRaises(ValueError):
                build_policies([2.5, 5, 10], value)

    def test_policy_builder_preserves_close_distinct_thresholds(self):
        first, second = 2.5000001, 2.5000002
        policies, primary = build_policies([first, second], first)
        self.assertEqual(list(policies), ["Banda ±2.5000001 pp", "Banda ±2.5000002 pp"])
        self.assertEqual(primary, "Banda ±2.5000001 pp")
        self.assertEqual(policies[primary]["banda_pp"], first)
        self.assertEqual(policies["Banda ±2.5000002 pp"]["banda_pp"], second)

    def test_policy_builder_includes_primary_sorts_and_deduplicates(self):
        expected = {
            "Banda ±2.5 pp": {"frecuencia": "bandas", "banda_pp": 2.5},
            "Banda ±5 pp": {"frecuencia": "bandas", "banda_pp": 5.0},
            "Banda ±10 pp": {"frecuencia": "bandas", "banda_pp": 10.0},
        }
        for candidates in [[10, 2.5, 10], [10, 5, 2.5, 5.0, 10.0]]:
            with self.subTest(candidates=candidates):
                policies, primary = build_policies(candidates, 5)
                self.assertEqual(policies, expected)
                self.assertEqual(list(policies), list(expected))
                self.assertEqual(primary, "Banda ±5 pp")


if __name__ == "__main__":
    unittest.main()
