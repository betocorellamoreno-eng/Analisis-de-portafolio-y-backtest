"""Prepara RF diario en USD desde el ZIP oficial de Kenneth French.

Reutiliza el archivo original si existe. --refresh vuelve a descargarlo.
No descarga precios ni modifica los CSV existentes de activos.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.request
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
DETAILS = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_factors.html"


def main(refresh=False):
    archive = ROOT / "data" / "F-F_Research_Data_Factors_daily_CSV.zip"
    if refresh or not archive.exists():
        payload = urllib.request.urlopen(URL, timeout=60).read()
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            assert len(z.namelist()) == 1
        archive.write_bytes(payload)
    payload = archive.read_bytes()
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        member = z.namelist()[0]
        source = z.read(member).decode("utf-8-sig")
    rows = []
    for row in csv.reader(source.splitlines()):
        if len(row) == 5 and re.fullmatch(r"\d{8}", row[0].strip()):
            rows.append((pd.to_datetime(row[0].strip(), format="%Y%m%d"), float(row[4])))
    rf = pd.DataFrame(rows, columns=["Date", "RF_pct"]).set_index("Date")
    prices = pd.read_csv(ROOT / "data" / "precios_ajustados.csv", index_col="Date", parse_dates=["Date"])
    rf = rf.loc[prices.index[0]:prices.index[-1]]
    if rf.empty or not rf.index.is_unique or not rf.index.is_monotonic_increasing:
        raise ValueError("Cobertura RF vacía o fechas inválidas.")
    if not np.isfinite(rf.to_numpy()).all() or (rf["RF_pct"] <= -99).any():
        raise ValueError("La muestra RF contiene faltantes o marcadores del proveedor.")
    missing = prices.index.difference(rf.index)
    extra = rf.index.difference(prices.index)
    if len(missing) or len(extra):
        raise ValueError(f"Calendarios distintos. Precios sin RF: {list(missing)}; RF sin precios: {list(extra)}")
    output = ROOT / "data" / "rf_diario_usd.csv"
    rf.to_csv(output, float_format="%.8f")
    metadata = {
        "provider": "Kenneth R. French Data Library",
        "dataset": "Fama/French 3 Factors [Daily] - RF column",
        "download_url": URL, "methodology_url": DETAILS,
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "archive_modified_utc": datetime.fromtimestamp(archive.stat().st_mtime, timezone.utc).isoformat(),
        "archive_file": archive.name, "archive_member": member,
        "archive_sha256": hashlib.sha256(payload).hexdigest(),
        "csv_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "source_vintage": source.splitlines()[0],
        "source_units": "Percent per trading day; simple daily T-bill return derived from monthly rate.",
        "decimal_conversion": "RF_pct / 100; do not divide by 252 or add weekend interest.",
        "source_change": "Ibbotson Associates through 2024-05; ICE BofA US 1-Month Treasury Bill Index from 2024-06.",
        "snapshot_note": "Retrospective downloaded vintage; reference for ex-post metrics only, not a trading signal.",
        "start": rf.index[0].strftime("%Y-%m-%d"), "end": rf.index[-1].strftime("%Y-%m-%d"),
        "observations": len(rf), "calendar_matches_prices": True,
    }
    (ROOT / "data" / "rf_diario_usd_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"RF preparado: {len(rf)} observaciones, {metadata['start']} a {metadata['end']}; calendario idéntico a precios.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    main(parser.parse_args().refresh)
