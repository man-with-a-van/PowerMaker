#!/usr/bin/env python3
"""
office_supply_analysis.py
-------------------------
Answers: "Is the battery typically powering the office in the mornings and early
evenings when solar output is low or zero?"

Re-runs the PowerMaker backtest (importing everything from arbitrage_regression)
and breaks the building's electricity supply down by hour of day into:
  solar (direct)  |  battery (discharge)  |  grid (import)

Run: python3 office_supply_analysis.py
"""
import os
import numpy as np
import pandas as pd

import arbitrage_regression as A


def run(strategy="co2"):
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, A.DATA_FILE)
    df = A.load_data(path)
    df = df.set_index(pd.RangeIndex(len(df)))
    base = A.base_price_stats(df)
    th = A.apply_margin(base, A.MIN_MARGIN)
    planned, nights = A.plan_truck_grid_intervals(df)
    if strategy == "co2":
        r = A.simulate_co2(df, th, planned, nights)
    else:
        r = A.simulate(df, th, planned, nights, with_battery=True,
                       solar_priority=(strategy == "solar_priority"))
    return df, r


def main():
    strategy = "co2"        # the active goal: CO2-max + aggressive cycling
    df, r = run(strategy)
    days = r["days"]
    print(f"(strategy: {strategy})")

    solar = r["load_from_solar"] / days   # kWh/day, by hour
    batt = r["load_from_batt"] / days
    grid = r["load_from_grid"] / days
    total = solar + batt + grid

    print("=" * 74)
    print("HOW THE OFFICE/WAREHOUSE LOAD IS SUPPLIED, BY HOUR OF DAY (avg kWh/day)")
    print("CO2-max + aggressive-cycling mode | solar x{:g}".format(A.SOLAR_SCALE))
    print("=" * 74)
    print(f"  {'hour':>5} {'solar':>7} {'battery':>8} {'grid':>7} {'load':>7}   battery share")
    print("  " + "-" * 64)
    for h in range(24):
        bshare = (batt[h] / total[h] * 100) if total[h] > 0 else 0.0
        bar = "#" * int(round(bshare / 5))
        print(f"  {h:02d}:00 {solar[h]:7.1f} {batt[h]:8.2f} {grid[h]:7.1f} {total[h]:7.1f}   "
              f"{bshare:4.0f}% |{bar:<20}|")

    def windowed(hours):
        s = solar[hours].sum(); b = batt[hours].sum(); g = grid[hours].sum()
        t = s + b + g
        return s, b, g, t

    print("\n" + "=" * 74)
    print("FOCUS WINDOWS (avg kWh/day, and % of that window's load)")
    print("=" * 74)
    windows = {
        "Morning 06:00-09:00 (sun low/rising)": [6, 7, 8],
        "Midday   10:00-15:00 (sun high)     ": [10, 11, 12, 13, 14],
        "Evening  17:00-20:00 (sun low/zero) ": [17, 18, 19],
        "Overnight 21:00-05:00 (no sun)      ": [21, 22, 23, 0, 1, 2, 3, 4, 5],
    }
    for name, hrs in windows.items():
        s, b, g, t = windowed(hrs)
        print(f"  {name}")
        print(f"     solar {s:6.1f} ({s/t*100:3.0f}%) | battery {b:5.2f} ({b/t*100:3.0f}%) | "
              f"grid {g:6.1f} ({g/t*100:3.0f}%)")

    # whole-day battery contribution to load
    tb = batt.sum(); tt = total.sum()
    print("\n" + "=" * 74)
    print(f"  Battery supplies {tb:.2f} kWh/day of the {tt:.0f} kWh/day office load "
          f"= {tb/tt*100:.1f}% overall.")
    print("=" * 74)


if __name__ == "__main__":
    main()
