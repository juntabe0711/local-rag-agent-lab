#!/usr/bin/env python3
"""エージェントのデモ用に、架空のウェーハ工場の品質データベース (SQLite) を作る。

    python scripts/make_sample_db.py            # data/factory.sqlite (乱数シード固定で毎回同じ)

中身はすべて乱数で作った架空のデータ。実在の工場・製品・装置とは関係ない。
原因調査のデモになるよう、次の「仕込み」がある (答え合わせは docs/sample_data.md):
  - 研磨装置 POL-03 は 2026-09-15 の定期保全でパッドを交換したあと、パーティクル (LPD) が増える
  - 洗浄装置 CLN-02 は薬液温度が高めの日に、金属汚染 (Fe) がやや増える
"""
from __future__ import annotations

import argparse
import math
import random
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

from _common import DEFAULT_CONFIG, setup_logging

from ragcore.config import data_path, load_config

SCHEMA = """
CREATE TABLE products (
    product_code TEXT PRIMARY KEY,
    diameter_mm  INTEGER NOT NULL,
    grade        TEXT NOT NULL,
    spec_lpd_max INTEGER NOT NULL       -- 1 枚あたり LPD (≥45nm) の上限
);
CREATE TABLE equipment (
    equipment_id TEXT PRIMARY KEY,
    process      TEXT NOT NULL,         -- slicing / lapping / polishing / cleaning / inspection
    model        TEXT NOT NULL,
    installed_on TEXT NOT NULL
);
CREATE TABLE lots (
    lot_id       TEXT PRIMARY KEY,
    product_code TEXT NOT NULL REFERENCES products(product_code),
    wafer_count  INTEGER NOT NULL,
    started_at   TEXT NOT NULL,
    shipped_on   TEXT
);
CREATE TABLE process_history (
    lot_id       TEXT NOT NULL REFERENCES lots(lot_id),
    step_no      INTEGER NOT NULL,
    process      TEXT NOT NULL,
    equipment_id TEXT NOT NULL REFERENCES equipment(equipment_id),
    start_time   TEXT NOT NULL,
    end_time     TEXT NOT NULL,
    PRIMARY KEY (lot_id, step_no)
);
CREATE TABLE inspection_results (
    lot_id         TEXT NOT NULL REFERENCES lots(lot_id),
    wafer_no       INTEGER NOT NULL,
    inspected_at   TEXT NOT NULL,
    lpd_45nm       INTEGER NOT NULL,    -- 45nm 以上のパーティクル数
    lpd_65nm       INTEGER NOT NULL,
    flatness_gbir  REAL NOT NULL,       -- 平坦度 GBIR (µm)
    fe_conc_e9     REAL NOT NULL,       -- 表面 Fe 濃度 (1e9 atoms/cm2)
    judgement      TEXT NOT NULL,       -- OK / NG
    PRIMARY KEY (lot_id, wafer_no)
);
CREATE TABLE maintenance_events (
    event_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    equipment_id TEXT NOT NULL REFERENCES equipment(equipment_id),
    event_date   TEXT NOT NULL,
    kind         TEXT NOT NULL,         -- PM (定期保全) / repair / consumable
    description  TEXT NOT NULL
);
CREATE TABLE process_params (
    equipment_id TEXT NOT NULL REFERENCES equipment(equipment_id),
    run_date     TEXT NOT NULL,
    param_name   TEXT NOT NULL,
    value        REAL NOT NULL,
    PRIMARY KEY (equipment_id, run_date, param_name)
);
CREATE TABLE app_settings (
    key          TEXT PRIMARY KEY,
    value        TEXT,
    api_key      TEXT                   -- 機密列の伏せ字・参照拒否のデモ用 (中身はダミー)
);
CREATE INDEX idx_hist_equipment ON process_history(equipment_id, start_time);
CREATE INDEX idx_insp_time ON inspection_results(inspected_at);
CREATE INDEX idx_lots_started ON lots(started_at);
"""

PRODUCTS = [("P300-PRIME", 300, "prime", 30), ("P300-EPI", 300, "epi", 20),
            ("P200-PRIME", 200, "prime", 40), ("P300-TEST", 300, "test", 200)]
EQUIPMENT = [("SLC-01", "slicing", "WS-9000", "2019-04-01"), ("SLC-02", "slicing", "WS-9000", "2021-06-01"),
             ("LAP-01", "lapping", "LP-500", "2018-10-01"), ("LAP-02", "lapping", "LP-500", "2020-03-01"),
             ("POL-01", "polishing", "CMP-X3", "2019-01-15"), ("POL-02", "polishing", "CMP-X3", "2020-07-01"),
             ("POL-03", "polishing", "CMP-X5", "2023-02-01"),
             ("CLN-01", "cleaning", "SC-1200", "2019-01-15"), ("CLN-02", "cleaning", "SC-1200", "2022-05-01"),
             ("INS-01", "inspection", "SP-7", "2021-01-10")]
FLOW = ["slicing", "lapping", "polishing", "cleaning", "inspection"]
START = date(2026, 7, 1)
END = date(2026, 10, 10)
POL03_PAD_CHANGE = date(2026, 9, 15)


def build(path: Path, seed: int = 20261017) -> dict:
    rnd = random.Random(seed)
    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.executescript(SCHEMA)
    conn.executemany("INSERT INTO products VALUES (?,?,?,?)", PRODUCTS)
    conn.executemany("INSERT INTO equipment VALUES (?,?,?,?)", EQUIPMENT)
    by_proc: dict = {}
    for eq, proc, *_ in EQUIPMENT:
        by_proc.setdefault(proc, []).append(eq)

    # 保全の記録 (POL-03 のパッド交換が仕込み。ほかは通常の保全)
    events = [("POL-03", POL03_PAD_CHANGE.isoformat(), "PM", "定期保全: 研磨パッドを新ロット (PAD-B 系) に交換、コンディショナ調整"),
              ("POL-01", "2026-08-05", "PM", "定期保全: 研磨パッド交換 (従来品 PAD-A)"),
              ("POL-02", "2026-08-26", "PM", "定期保全: 研磨パッド交換 (従来品 PAD-A)"),
              ("CLN-02", "2026-08-12", "repair", "薬液ヒーターの温調器を交換"),
              ("CLN-01", "2026-09-02", "PM", "定期保全: フィルタ交換"),
              ("SLC-01", "2026-07-20", "consumable", "ワイヤー交換"),
              ("SLC-02", "2026-09-28", "consumable", "ワイヤー交換"),
              ("LAP-02", "2026-08-18", "PM", "定盤の面出し"),
              ("INS-01", "2026-07-30", "PM", "標準粒子による感度校正")]
    conn.executemany("INSERT INTO maintenance_events (equipment_id, event_date, kind, description) VALUES (?,?,?,?)",
                     events)

    # 装置パラメータ (日ごと)。CLN-02 の薬液温度は 8/12 の修理後に高め・ばらつき大
    params = []
    d = START
    cln02_temp = {}
    while d <= END:
        for eq in by_proc["cleaning"]:
            base, sd = (65.0, 0.6) if eq == "CLN-01" else ((65.5, 1.6) if d >= date(2026, 8, 12) else (65.0, 0.6))
            t = round(rnd.gauss(base, sd), 2)
            if eq == "CLN-02":
                cln02_temp[d] = t
            params.append((eq, d.isoformat(), "chem_temp_c", t))
            params.append((eq, d.isoformat(), "megasonic_w", round(rnd.gauss(1200, 15), 1)))
        for eq in by_proc["polishing"]:
            params.append((eq, d.isoformat(), "pad_pressure_kpa", round(rnd.gauss(18.0, 0.3), 2)))
            params.append((eq, d.isoformat(), "slurry_flow_ml_min", round(rnd.gauss(250, 4), 1)))
        d += timedelta(days=1)
    conn.executemany("INSERT INTO process_params VALUES (?,?,?,?)", params)

    # ロット・工程の履歴・検査結果
    lots, hist, insp = [], [], []
    n_lot = 0
    d = START
    while d <= END:
        for _ in range(rnd.randint(4, 7)):
            n_lot += 1
            prod = rnd.choices(PRODUCTS, weights=[5, 3, 2, 1])[0]
            lot_id = f"L{d.strftime('%y%m%d')}{n_lot % 1000:03d}"
            started = datetime.combine(d, datetime.min.time()) + timedelta(hours=rnd.randint(6, 20))
            wafers = 25
            t = started
            used = {}
            for step, proc in enumerate(FLOW, 1):
                eq = rnd.choice(by_proc[proc])
                dur = timedelta(hours=rnd.uniform(1.5, 5.0))
                hist.append((lot_id, step, proc, eq, t.isoformat(sep=" ", timespec="minutes"),
                             (t + dur).isoformat(sep=" ", timespec="minutes")))
                used[proc] = (eq, t.date())
                t = t + dur + timedelta(hours=rnd.uniform(0.5, 8))
            shipped = (t + timedelta(days=rnd.randint(1, 3))).date()
            lots.append((lot_id, prod[0], wafers, started.isoformat(sep=" ", timespec="minutes"),
                         shipped.isoformat() if shipped <= END else None))
            pol_eq, pol_day = used["polishing"]
            cln_eq, cln_day = used["cleaning"]
            lpd_mean = {"prime": 8.0, "epi": 6.0, "test": 25.0}[prod[2]]
            if pol_eq == "POL-03" and pol_day >= POL03_PAD_CHANGE:        # 仕込み 1
                lpd_mean *= 2.6
            fe_mean = 0.8
            if cln_eq == "CLN-02" and cln02_temp.get(cln_day, 65.0) > 66.5:  # 仕込み 2
                fe_mean = 1.9
            for w in range(1, wafers + 1):
                lpd45 = max(0, int(round(rnd.gammavariate(4.0, lpd_mean / 4.0))))
                lpd65 = max(0, int(round(lpd45 * rnd.uniform(0.25, 0.45))))
                gbir = round(max(0.05, rnd.gauss(0.32 if prod[1] == 300 else 0.40, 0.04)), 3)
                fe = round(max(0.05, rnd.lognormvariate(math.log(fe_mean), 0.35)), 2)
                ng = lpd45 > prod[3] or gbir > 0.5 or fe > 3.0
                insp.append((lot_id, w, (t + timedelta(minutes=w)).isoformat(sep=" ", timespec="minutes"),
                             lpd45, lpd65, gbir, fe, "NG" if ng else "OK"))
        d += timedelta(days=1)
    conn.executemany("INSERT INTO lots VALUES (?,?,?,?,?)", lots)
    conn.executemany("INSERT INTO process_history VALUES (?,?,?,?,?,?)", hist)
    conn.executemany("INSERT INTO inspection_results VALUES (?,?,?,?,?,?,?,?)", insp)
    conn.executemany("INSERT INTO app_settings VALUES (?,?,?)",
                     [("report_mail_to", "quality-team@example.com", None),
                      ("external_api", "https://api.example.com", "sk-dummy-not-a-real-key")])
    conn.commit()
    counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("lots", "process_history", "inspection_results", "maintenance_events", "process_params")}
    conn.close()
    return counts


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--seed", type=int, default=20261017)
    args = p.parse_args(argv)
    setup_logging()
    cfg, base = load_config(args.config)
    path = data_path(cfg, base, (cfg.get("agent") or {}).get("db_path", "factory.sqlite"))
    counts = build(path, args.seed)
    print(f"wrote {path}")
    for k, v in counts.items():
        print(f"  {k:20s} {v:>8,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
