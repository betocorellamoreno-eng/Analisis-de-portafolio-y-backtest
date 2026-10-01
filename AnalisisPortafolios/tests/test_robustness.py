"""Controles independientes para escenarios de subperiodos y costos."""
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
for prefix in ["# Motor del backtest:", "# Investigación de robustez: escenarios independientes."]:
    source = next("".join(cell["source"]) for cell in document["cells"]
                  if "".join(cell["source"]).startswith(prefix))
    exec(compile(source, str(NOTEBOOK), "exec"), namespace)
investigate = namespace["investigar_escenarios"]
simulate = namespace["simular_portafolio"]
summarize = namespace["resumir_backtest"]


class RobustnessTests(unittest.TestCase):
    def setUp(self):
        # Pocas observaciones bastan para comprobar contabilidad y límites;
        # estas series sintéticas no se utilizan para interpretar volatilidad.
        dates = pd.to_datetime([
            "2020-01-02", "2020-03-31", "2020-04-01", "2020-06-30", "2020-09-30", "2020-12-31",
            "2021-01-04", "2021-03-31", "2021-04-01", "2021-06-30", "2021-09-30", "2021-12-31",
        ])
        self.prices = pd.DataFrame({
            "SPY": [100, 120, 110, 135, 130, 160, 180, 165, 175, 170, 190, 195],
            "IEF": [100, 101, 103, 102, 105, 110, 115, 119, 121, 125, 126, 130],
        }, index=dates, dtype=float)
        self.weights = {"SPY": .6, "IEF": .4}
        self.capital = 1000.0
        self.periods = {
            "2020": ("2020-01-01", "2020-12-31"),
            "2021": ("2021-01-01", "2021-12-31"),
            "Muestra completa": ("2020-01-01", "2021-12-31"),
        }
        self.frequencies = {
            "Sin rebalanceo": "sin rebalanceo", "Mensual": "mensual",
            "Trimestral": "trimestral", "Anual": "anual",
        }

    def run_scenarios(self, prices=None, costs=(10,), periods=None):
        return investigate(
            self.prices if prices is None else prices, self.weights, self.capital,
            costs, self.periods if periods is None else periods, self.frequencies,
        )

    def test_cartesian_grid_adds_zero_and_deduplicates_costs(self):
        table = self.run_scenarios(costs=[50, 10, 50])
        self.assertEqual(table.index.names, ["Periodo", "Costo (pb)", "Estrategia"])
        expected = pd.MultiIndex.from_product([
            self.periods, [0.0, 10.0, 50.0], [*self.frequencies, "SPY"]
        ], names=table.index.names)
        self.assertTrue(table.index.is_unique)
        self.assertEqual(len(table), len(expected))
        self.assertEqual(set(table.index), set(expected))
        for period in self.periods:
            costs = table.xs((period, "SPY"), level=("Periodo", "Estrategia")).index.tolist()
            self.assertEqual(costs, [0.0, 10.0, 50.0])

    def test_each_block_restarts_from_its_own_prices_and_pays_entry(self):
        table = self.run_scenarios()
        for period, (start, end) in self.periods.items():
            block = self.prices.loc[start:end]
            growth = block.iloc[-1] / block.iloc[0]
            for name, factor in [
                ("Sin rebalanceo", .6 * growth["SPY"] + .4 * growth["IEF"]),
                ("SPY", growth["SPY"]),
            ]:
                with self.subTest(period=period, strategy=name):
                    row = table.loc[(period, 10.0, name)]
                    invested = self.capital / 1.001
                    expected = invested * factor
                    years = (block.index[-1] - block.index[0]).days / 365.25
                    self.assertEqual(pd.Timestamp(row["Inicio real"]), block.index[0])
                    self.assertEqual(pd.Timestamp(row["Fin real"]), block.index[-1])
                    self.assertEqual(row["Observaciones"], len(block))
                    self.assertAlmostEqual(row["Valor final (USD)"], expected)
                    self.assertAlmostEqual(row["CAGR"], (expected / self.capital) ** (1 / years) - 1)
                    self.assertAlmostEqual(row["Costos pagados (USD)"], self.capital - invested)
                    self.assertAlmostEqual(row["Impacto en valor final (USD)"], self.capital * factor - expected)
                    self.assertAlmostEqual(row["Volumen relativo anual"], 0.0)
                    self.assertEqual(row["Rebalanceos"], 0)

    def test_prices_before_block_do_not_change_that_block(self):
        original = self.run_scenarios()
        changed = self.prices.copy()
        changed.loc[:"2020-12-31", "SPY"] *= 8.0
        changed.loc[:"2020-12-31", "IEF"] *= .2
        altered = self.run_scenarios(prices=changed)
        pd.testing.assert_frame_equal(original.xs("2021"), altered.xs("2021"))

    def test_full_sample_matches_direct_motor_and_turnover(self):
        table = self.run_scenarios()
        years = (self.prices.index[-1] - self.prices.index[0]).days / 365.25
        for name in [*self.frequencies, "SPY"]:
            with self.subTest(strategy=name):
                benchmark = name == "SPY"
                direct = simulate(
                    self.prices[["SPY"]] if benchmark else self.prices,
                    {"SPY": 1.0} if benchmark else self.weights,
                    self.capital, 10.0,
                    frecuencia="sin rebalanceo" if benchmark else self.frequencies[name],
                )
                expected = summarize(direct)
                row = table.loc[("Muestra completa", 10.0, name)]
                np.testing.assert_allclose(row.loc[expected.index].to_numpy(dtype=float), expected)
                trades = direct["operaciones"].iloc[1:]
                relative_volume = (trades["Compras"] + trades["Ventas"]) / trades["Valor antes"]
                self.assertAlmostEqual(row["Volumen relativo anual"], relative_volume.sum() / years)

    def test_cost_sensitivity_reduces_final_wealth_and_cagr(self):
        table = self.run_scenarios(costs=[100, 50, 10])
        for period in self.periods:
            for name in [*self.frequencies, "SPY"]:
                with self.subTest(period=period, strategy=name):
                    rows = table.xs((period, name), level=("Periodo", "Estrategia")).sort_index()
                    self.assertTrue(np.all(np.diff(rows["Valor final (USD)"]) <= 1e-9))
                    self.assertTrue(np.all(np.diff(rows["CAGR"]) <= 1e-12))
                    self.assertAlmostEqual(rows.loc[0.0, "Impacto en valor final (USD)"], 0.0)
                    np.testing.assert_allclose(
                        rows["Impacto en valor final (USD)"],
                        rows.loc[0.0, "Valor final (USD)"] - rows["Valor final (USD)"], atol=1e-9,
                    )
                    self.assertEqual(rows["Rebalanceos"].nunique(), 1)

    def test_invalid_costs_periods_and_missing_benchmark_fail(self):
        for costs in [[-1], [np.nan], [np.inf], [10000], ["no numérico"]]:
            with self.subTest(costs=costs), self.assertRaises(ValueError):
                self.run_scenarios(costs=costs)
        for start, end in [
            ("2022-01-01", "2022-12-31"),  # Vacío.
            ("2021-12-31", "2021-01-01"),  # Invertido.
            ("2020-01-01", "2020-03-31"),  # Solo dos observaciones.
        ]:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                self.run_scenarios(periods={"Inválido": (start, end)})
        with self.assertRaises(ValueError):
            investigate(self.prices[["IEF"]], {"IEF": 1.0}, self.capital,
                        [0, 10], self.periods, self.frequencies)


if __name__ == "__main__":
    unittest.main()
