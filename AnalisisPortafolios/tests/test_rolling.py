"""Controles independientes de ventanas móviles y sensibilidad a la entrada."""
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
for prefix in ["# Motor del backtest:", "# Ventanas móviles: generación y evaluación."]:
    source = next("".join(cell["source"]) for cell in document["cells"]
                  if "".join(cell["source"]).startswith(prefix))
    exec(compile(source, str(NOTEBOOK), "exec"), namespace)
build_windows = namespace["construir_ventanas_mensuales"]
evaluate = namespace["evaluar_ventanas_moviles"]
simulate = namespace["simular_portafolio"]
summarize = namespace["resumir_backtest"]


class RollingWindowTests(unittest.TestCase):
    weights = {"SPY": .6, "IEF": .4}
    frequencies = {
        "Sin rebalanceo": "sin rebalanceo", "Mensual": "mensual",
        "Trimestral": "trimestral", "Anual": "anual",
    }
    capital = 1000.0
    cost = 25.0

    @staticmethod
    def prices(start="2020-01-02", end="2020-09-30"):
        dates = pd.bdate_range(start, end)
        t = np.arange(len(dates), dtype=float)
        return pd.DataFrame({
            "SPY": 100 * np.exp(.001 * t + .025 * np.sin(t / 6)),
            "IEF": 100 * np.exp(.0002 * t + .010 * np.cos(t / 8)),
        }, index=dates)

    def run_windows(self, prices, windows, frequencies=None):
        return evaluate(prices, self.weights, self.capital, self.cost, windows,
                        self.frequencies if frequencies is None else frequencies)

    def test_calendar_count_monthly_steps_and_actual_boundaries(self):
        prices = self.prices("2010-01-04", "2025-12-31")
        windows = build_windows(prices)
        expected = pd.period_range("2010-01", "2022-01", freq="M").astype(str)
        self.assertEqual(windows.index.name, "Ventana")
        self.assertEqual(list(windows.index), list(expected))
        self.assertEqual(len(windows), 192 - 48 + 1)
        first = windows.iloc[0]
        self.assertEqual(first["Inicio solicitado"], pd.Timestamp("2010-01-01"))
        self.assertEqual(first["Fin solicitado"], pd.Timestamp("2013-12-31"))
        self.assertEqual(first["Inicio real"], pd.Timestamp("2010-01-04"))
        self.assertEqual(first["Fin real"], pd.Timestamp("2013-12-31"))
        self.assertEqual(windows.iloc[-1]["Fin real"], pd.Timestamp("2025-12-31"))
        for _, row in windows.iterrows():
            block = prices.loc[row["Inicio solicitado"]:row["Fin solicitado"]]
            self.assertEqual(row["Observaciones"], len(block))
            years = (block.index[-1] - block.index[0]).days / 365.25
            self.assertAlmostEqual(row["Años efectivos"], years)
        stepped = build_windows(prices, meses=48, paso_meses=3)
        self.assertEqual(list(stepped.index), list(expected[::3]))

    def test_conservative_terminal_month_cutoff_and_missing_month(self):
        # Julio 2021 termina en sábado: el viernes 30 no cubre el cierre nominal.
        for last in ["2021-07-15", "2021-07-30"]:
            with self.subTest(last=last):
                windows = build_windows(self.prices("2021-01-04", last), meses=2)
                self.assertEqual(list(windows.index),
                                 ["2021-01", "2021-02", "2021-03", "2021-04", "2021-05"])
                self.assertEqual(windows.iloc[-1]["Fin solicitado"], pd.Timestamp("2021-06-30"))
        # La ausencia completa de marzo solo elimina febrero-marzo y marzo-abril.
        prices = self.prices("2020-01-02", "2020-06-30")
        prices = prices.loc[prices.index.month != 3]
        windows = build_windows(prices, meses=2)
        self.assertEqual(list(windows.index), ["2020-01", "2020-04", "2020-05"])

    def test_buy_hold_analytic_wealth_risk_cost_and_actual_year_cagr(self):
        prices = self.prices()
        windows = build_windows(prices, meses=3)
        table = self.run_windows(prices, windows)
        self.assertEqual(table.index.names, ["Ventana", "Estrategia"])
        self.assertTrue(table.index.is_unique)
        self.assertEqual(len(table), len(windows) * 5)
        invested = self.capital / (1 + self.cost / 10_000)
        for name, window in windows.iterrows():
            block = prices.loc[window["Inicio real"]:window["Fin real"]]
            growth = block.div(block.iloc[0])
            for strategy, factors in [
                ("Sin rebalanceo", .6 * growth["SPY"] + .4 * growth["IEF"]),
                ("SPY", growth["SPY"]),
            ]:
                with self.subTest(window=name, strategy=strategy):
                    row = table.loc[(name, strategy)]
                    wealth = invested * factors
                    years = (block.index[-1] - block.index[0]).days / 365.25
                    expected_cagr = (wealth.iloc[-1] / self.capital) ** (1 / years) - 1
                    self.assertAlmostEqual(row["Valor final (USD)"], wealth.iloc[-1])
                    self.assertAlmostEqual(row["CAGR"], expected_cagr)
                    self.assertAlmostEqual(row["Años efectivos"], years)
                    self.assertAlmostEqual(row["Costos pagados (USD)"], self.capital - invested)
                    self.assertEqual(row["Rebalanceos"], 0)
                    returns = wealth.pct_change(fill_method=None).iloc[1:]
                    self.assertAlmostEqual(row["Volatilidad anualizada"], returns.std(ddof=1) * np.sqrt(252))
                    drawdown = wealth / wealth.cummax().clip(lower=self.capital) - 1
                    self.assertAlmostEqual(row["Máxima caída"], drawdown.min())
                    self.assertEqual(row["Inicio real"], block.index[0])
                    self.assertEqual(row["Fin real"], block.index[-1])

    def test_prices_outside_window_do_not_change_its_results(self):
        prices = self.prices()
        windows = build_windows(prices, meses=3).loc[["2020-03"]]
        expected = self.run_windows(prices, windows)
        changed = prices.copy()
        outside = (changed.index < windows.iloc[0]["Inicio real"]) | (changed.index > windows.iloc[0]["Fin real"])
        changed.loc[outside, "SPY"] *= 7
        changed.loc[outside, "IEF"] *= .2
        actual = self.run_windows(changed, windows)
        pd.testing.assert_frame_equal(actual, expected)

    def test_calendar_rules_and_direct_motor_equivalence(self):
        prices = self.prices("2010-01-04", "2014-01-31")
        windows = build_windows(prices)
        self.assertEqual(list(windows.index), ["2010-01", "2010-02"])
        table = self.run_windows(prices, windows)
        expected_events = {
            "2010-01": {"Mensual": 47, "Trimestral": 15, "Anual": 3},
            "2010-02": {"Mensual": 47, "Trimestral": 16, "Anual": 4},
        }
        for name, counts in expected_events.items():
            for strategy, count in counts.items():
                self.assertEqual(table.loc[(name, strategy), "Rebalanceos"], count)
        window = windows.loc["2010-02"]
        block = prices.loc[window["Inicio real"]:window["Fin real"]]
        direct = simulate(block, self.weights, self.capital, self.cost, frecuencia="mensual")
        summary = summarize(direct)
        actual = table.loc[("2010-02", "Mensual"), summary.index]
        np.testing.assert_allclose(actual.to_numpy(dtype=float), summary.to_numpy(dtype=float), rtol=1e-12)
        self.assertNotIn(block.index[-1], direct["operaciones"].index)

    def test_equal_assets_produce_tied_strategies_with_entry_cost(self):
        prices = self.prices()
        prices["IEF"] = prices["SPY"]
        windows = build_windows(prices, meses=3).iloc[[0]]
        table = self.run_windows(prices, windows).droplevel("Ventana")
        columns = ["Valor final (USD)", "CAGR", "Volatilidad anualizada", "Máxima caída"]
        reference = table.loc["SPY", columns].to_numpy(dtype=float)
        np.testing.assert_allclose(table[columns].to_numpy(dtype=float), np.tile(reference, (5, 1)), atol=1e-12)

    def test_window_builder_rejects_invalid_parameters_indices_and_empty_result(self):
        prices = self.prices()
        for keyword in ["meses", "paso_meses"]:
            for value in [True, False, 0, -1, 2.0, "2"]:
                with self.subTest(keyword=keyword, value=value), self.assertRaises(ValueError):
                    build_windows(prices, **{keyword: value})
        bad_nat = prices.copy()
        bad_nat.index = pd.DatetimeIndex([pd.NaT, *prices.index[1:]])
        for bad in [prices.iloc[::-1], pd.concat([prices, prices.iloc[-1:]]),
                    bad_nat, prices.reset_index(drop=True), prices.iloc[:0]]:
            with self.subTest(index_type=type(bad.index).__name__, n=len(bad)), self.assertRaises(ValueError):
                build_windows(bad, meses=2)
        with self.assertRaises(ValueError):
            build_windows(prices, meses=48)
        sparse = prices.iloc[[0, -1]]
        with self.assertRaises(ValueError):
            build_windows(sparse, meses=9)

    def test_evaluation_rejects_bad_metadata_duplicate_windows_and_reference(self):
        prices = self.prices()
        windows = build_windows(prices, meses=3)
        bad_frames = [windows.iloc[:0], pd.concat([windows, windows.iloc[[0]]])]
        for column, delta in [
            ("Inicio real", pd.Timedelta(days=1)),
            ("Fin real", pd.Timedelta(days=-1)),
            ("Observaciones", 1),
            ("Años efectivos", .1),
        ]:
            altered = windows.copy()
            altered.loc[altered.index[0], column] += delta
            bad_frames.append(altered)
        for bad in bad_frames:
            with self.subTest(rows=len(bad)), self.assertRaises(ValueError):
                self.run_windows(prices, bad)
        with self.assertRaises(ValueError):
            evaluate(prices[["IEF"]], {"IEF": 1}, self.capital, self.cost, windows, self.frequencies)
        with self.assertRaises(ValueError):
            self.run_windows(prices, windows, frequencies={"SPY": "mensual"})


if __name__ == "__main__":
    unittest.main()
