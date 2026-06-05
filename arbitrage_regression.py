#!/usr/bin/env python3
"""
arbitrage_regression.py
=======================

Backtest / "regression" of the PowerMaker spot-price arbitrage algorithm against
two years of historical 5-minute interval data for a Victorian office + warehouse
with a rooftop solar array (but, in the source data, NO battery and NO EVs).

On top of the historical building load + solar we simulate adding:

  * A 200 kWh stationary battery behind a 100 kW inverter, traded with the
    PowerMaker import/export algorithm (5-day rolling price quantiles + an
    exponential charge/discharge rate curve, ported from powermakerfunctions.py).
  * Two electric trucks (40 kWh each) that must charge 0->100 % overnight
    (17:00 -> 06:00) on the nights they are used (4 days/week), replacing diesel
    trucks (15 L/100 km, <=100 km/day).

It answers:
  Q1. Expected annual profit from the electrical arbitrage (vs. having no battery).
  Q2. The daily import/export cycle of the battery storage.
  Q3. Annual kg CO2 saved by the electric trucks, given the solar/grid mix that
      actually charged them.

NOTE ON "REGRESSION": the core analysis is a deterministic backtest over the real
price/load/solar series. As an explicit statistical regression we also fit annual
arbitrage profit as a linear function of daily price spread (see the regression
section at the end) and report R^2.

Run:    python3 arbitrage_regression.py
Deps:   pandas, numpy   (matplotlib optional -> writes charts to static/ if present)

------------------------------------------------------------------------------
ALL TUNABLE ASSUMPTIONS ARE IN THE CONFIG BLOCK BELOW. The headline numbers are
most sensitive to: ROUND_TRIP_EFFICIENCY, VIC_GRID_KGCO2_PER_KWH, MIN_MARGIN,
and the truck usage days. Edit and re-run.
------------------------------------------------------------------------------
"""

import os
import sys
import math
from datetime import time as dtime

import numpy as np
import pandas as pd

# =============================================================================
# CONFIG / ASSUMPTIONS  (edit these)
# =============================================================================
DATA_FILE = "Victorian Freight Decarbonisation Grant Data Analysis - ActualUsage5Min_SpotPrice.csv"

# ---- Solar array ------------------------------------------------------------
SOLAR_SCALE            = 2.0      # multiply historical solar production. 2.0 = double
                                  # the existing array (per brief). 1.0 = as-built.

# ---- Battery storage system -------------------------------------------------
BATTERY_CAPACITY_KWH   = 200.0    # usable nameplate capacity
INVERTER_KW            = 100.0    # max charge/discharge power (bidirectional)
BATTERY_RESERVE_FRAC   = 0.05     # keep >= 5% as a floor (cycle-life protection)
ROUND_TRIP_EFFICIENCY  = 0.90     # AC->storage->AC; split sqrt() each way
BATTERY_WEAR_COST_KWH  = 0.00     # $/kWh throughput wear charge. Upfront cost = $0
                                  # per brief, so default 0; MIN_MARGIN already
                                  # protects export economics. Set e.g. 0.05 to model wear.

# ---- Solar-priority truck charging mode -------------------------------------
# In solar-priority mode the battery (a) caps grid-arbitrage charging to leave
# headroom for the day's solar, and (b) protects stored solar (green energy) from
# being sold to the grid, reserving it to charge the trucks. Trades some arbitrage
# profit for a higher solar share / lower truck CO2.
SOLAR_RESERVE_FRAC     = 0.40     # headroom kept free for solar (= ~80 kWh = truck demand)

# ---- MIN_MARGIN sensitivity sweep ------------------------------------------
MARGIN_SWEEP = [0.00, 0.02, 0.05, 0.08, 0.11, 0.14, 0.20, 0.30]

# ---- CO2-max + aggressive-cycling strategy ("co2_cycle") --------------------
# Objective order: (1) supply office + trucks first, (2) MAXIMISE CO2 reduction by
# reserving stored solar (green energy) for the trucks/office and never selling it,
# (3) MAXIMISE battery cycling by buying grid energy cheap and re-selling it at ANY
# profit (price above the round-trip break-even below), at full inverter power.
CO2_SELL_FLOOR = 0.05     # $/kWh: sell grid-bought (grey) energy at any price above
                          # this. Must exceed buy-price/round-trip + wear to profit.
                          # ~$0.05 covers losses+wear when buying near $0.

# ---- PowerMaker algorithm parameters (ported from exampleconfig.py) ---------
IMPORT_QUANTILE = 0.25     # buy when price <= 25th percentile of trailing 5 days
EXPORT_QUANTILE = 0.75     # sell when price >= 75th percentile of trailing 5 days
PRICE_WINDOW_DAYS = 5      # trailing window for the quantile thresholds
EXP_INPUT_MIN = 0.0        # exponential rate-curve domain (see calc_*_rate)
EXP_INPUT_MAX = 4.0
IE_MIN_RATE_KW = 1.0       # min trade power
IE_MAX_RATE_KW = INVERTER_KW   # max trade power (capped at inverter, vs 120 kW default)
LOW_PRICE_IMPORT = 0.01    # $/kWh: at or below this, import hard regardless of quantile
MIN_MARGIN = 0.14          # $/kWh: floor on export margin above 5-day average (battery wear)
#  Per PowerMaker git history the *import* margin floor was removed (they import
#  whenever cheap, only gate exports by MIN_MARGIN); we mirror that here.

# ---- Electric trucks --------------------------------------------------------
NUM_TRUCKS              = 2
TRUCK_BATTERY_KWH       = 40.0    # each; charged 0->100% on a used night
TRUCK_USAGE_DAYS_PER_WK = 4       # nights requiring a full charge
TRUCK_USAGE_WEEKDAYS    = {0, 1, 2, 3}  # Mon-Thu (0=Mon). Drives which nights charge.
TRUCK_CHARGE_WINDOW     = (dtime(17, 0), dtime(6, 0))  # 17:00 -> 06:00 next day
TRUCK_CHARGER_KW        = 50.0    # combined truck charger power (separate from battery inverter)
TRUCK_KM_PER_DAY        = 100.0   # <=100 km/day per the brief (used for diesel displaced)
DIESEL_L_PER_100KM      = 15.0    # diesel truck being replaced
DIESEL_PRICE_PER_L      = 1.85    # AUD/L. Current (2025-26) VIC bulk/retail diesel ~$1.80-2.00.
                                  # Pre fuel-tax-credit; a freight operator's net cost may be
                                  # ~$0.20/L lower. Edit to your actual delivered price.

# ---- Emissions factors ------------------------------------------------------
DIESEL_KGCO2_PER_L      = 2.68    # tailpipe (scope 1) combustion of diesel
# Set DIESEL_INCLUDE_UPSTREAM=True for well-to-wheel (~3.0-3.2 kg/L incl. scope 3)
DIESEL_INCLUDE_UPSTREAM = False
DIESEL_UPSTREAM_KGCO2_PER_L = 0.55
VIC_GRID_KGCO2_PER_KWH  = 0.79    # Victorian grid scope-2 emissions factor (~2023-24)
SOLAR_KGCO2_PER_KWH     = 0.0     # on-site solar treated as zero operational emissions

# Derived efficiency split
ETA_C = math.sqrt(ROUND_TRIP_EFFICIENCY)   # charge efficiency
ETA_D = math.sqrt(ROUND_TRIP_EFFICIENCY)   # discharge efficiency
INTERVAL_HOURS = 5.0 / 60.0                 # 5-minute intervals
INVERTER_KWH_PER_INTERVAL = INVERTER_KW * INTERVAL_HOURS
CHARGER_KWH_PER_INTERVAL  = TRUCK_CHARGER_KW * INTERVAL_HOURS
RESERVE_KWH = BATTERY_CAPACITY_KWH * BATTERY_RESERVE_FRAC


# =============================================================================
# DATA LOADING
# =============================================================================
def load_data(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["dt"] = pd.to_datetime(df["Date Time"], format="%d/%m/%Y %H:%M:%S")
    # Solar production: small negative standby values -> clip to 0. Wh -> kWh.
    # Scaled by SOLAR_SCALE to model a larger array.
    df["solar_kwh"] = pd.to_numeric(df["Energy Produced Wh"], errors="coerce").clip(lower=0) / 1000.0 * SOLAR_SCALE
    df["load_kwh"]  = pd.to_numeric(df["Energy Consumed Wh"], errors="coerce").clip(lower=0) / 1000.0
    # Price $/MWh -> $/kWh
    p = df["Spot Price $/MWh"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False)
    df["price"] = pd.to_numeric(p, errors="coerce") / 1000.0
    df = df.dropna(subset=["dt", "solar_kwh", "load_kwh", "price"]).reset_index(drop=True)
    df["date"] = df["dt"].dt.normalize()
    return df


# =============================================================================
# POWERMAKER ALGORITHM (ported / adapted from powermakerfunctions.py)
# =============================================================================
def base_price_stats(df):
    """Replicate get_spot_price_stats() quantiles: trailing 5-day stats per calendar
    day, WITHOUT the MIN_MARGIN export floor (applied separately so a sweep can vary
    the margin cheaply). Returns dict date -> (avg, pmin, pmax, import_price, exp_raw)."""
    daily = df.groupby("date")["price"].apply(list)
    dates = list(daily.index)
    out = {}
    for i, d in enumerate(dates):
        lo = max(0, i - PRICE_WINDOW_DAYS)
        window = [p for dd in dates[lo:i + 1] for p in daily[dd]]  # trailing window incl. today
        arr = np.array(window, dtype=float)
        avg, pmin, pmax = arr.mean(), arr.min(), arr.max()
        imp = np.quantile(arr, IMPORT_QUANTILE)
        exp_raw = np.quantile(arr, EXPORT_QUANTILE) + 0.1   # PowerMaker adds 0.1 cushion
        out[d] = (avg, pmin, pmax, imp, exp_raw)
    return out


def apply_margin(base, min_margin):
    """Apply the MIN_MARGIN export floor to base stats: export_price is raised to at
    least (avg + min_margin). Returns dict date -> (avg, pmin, pmax, import, export)."""
    out = {}
    for d, (avg, pmin, pmax, imp, exp_raw) in base.items():
        exp = max(exp_raw, avg + min_margin)
        out[d] = (avg, pmin, pmax, imp, exp)
    return out


def calc_charge_rate_kw(price, import_price, price_min):
    """Exponential import rate curve (kW). Lower price -> harder charge."""
    if import_price <= price_min:
        return IE_MAX_RATE_KW
    scaled = np.interp(price, [price_min, import_price], [EXP_INPUT_MAX, EXP_INPUT_MIN])
    margin_exp = math.exp(scaled) - math.exp(EXP_INPUT_MIN)
    mult = (IE_MAX_RATE_KW - IE_MIN_RATE_KW) / math.exp(EXP_INPUT_MAX)
    return IE_MIN_RATE_KW + margin_exp * mult


def calc_discharge_rate_kw(price, export_price, price_max):
    """Exponential export rate curve (kW). Higher price -> harder discharge."""
    if price_max <= export_price:
        return IE_MAX_RATE_KW
    scaled = np.interp(price, [export_price, price_max], [EXP_INPUT_MIN, EXP_INPUT_MAX])
    margin_exp = math.exp(scaled) - math.exp(EXP_INPUT_MIN)
    mult = (IE_MAX_RATE_KW - IE_MIN_RATE_KW) / math.exp(EXP_INPUT_MAX)
    return IE_MIN_RATE_KW + margin_exp * mult


# =============================================================================
# TRUCK CHARGING SCHEDULE
# =============================================================================
def in_charge_window(ts):
    """True if timestamp is inside the overnight 17:00->06:00 window."""
    t = ts.time()
    start, end = TRUCK_CHARGE_WINDOW
    return t >= start or t < end


def charging_night_key(ts):
    """A night is keyed by the date it STARTS (the 17:00 side).
    Returns the start-date if ts is in a window, else None."""
    if not in_charge_window(ts):
        return None
    # If before 06:00, the night started on the previous calendar day.
    if ts.time() < TRUCK_CHARGE_WINDOW[1]:
        return (ts - pd.Timedelta(days=1)).normalize()
    return ts.normalize()


def upcoming_night(ts):
    """The charging-night key whose truck demand we should reserve solar for right
    now: the active night if inside the window, otherwise tonight (during daytime).
    Returns None if tonight is not a truck usage night."""
    nk = charging_night_key(ts)
    if nk is not None:
        return nk
    d = ts.normalize()                                  # daytime 06:00-17:00 -> tonight
    return d if d.weekday() in TRUCK_USAGE_WEEKDAYS else None


def plan_truck_grid_intervals(df):
    """For each charging night, pick the cheapest grid intervals sufficient to
    deliver the full truck demand from the grid (a fair, already-smart baseline).
    Battery-solar will opportunistically displace some of this during the sim.
    Returns: set of df-row indices flagged as planned grid-charge intervals, and
    a dict night_key -> required kWh."""
    truck_demand = NUM_TRUCKS * TRUCK_BATTERY_KWH
    df = df.copy()
    df["night"] = df["dt"].apply(charging_night_key)
    planned = set()
    nights = {}
    for night, grp in df.groupby("night"):
        if night is None:
            continue
        # Only nights whose start weekday is a usage day require a charge.
        if night.weekday() not in TRUCK_USAGE_WEEKDAYS:
            continue
        nights[night] = truck_demand
        grp_sorted = grp.sort_values("price")          # cheapest first
        acc = 0.0
        for idx, row in grp_sorted.iterrows():
            if acc >= truck_demand:
                break
            planned.add(idx)
            acc += CHARGER_KWH_PER_INTERVAL
    return planned, nights


# =============================================================================
# SIMULATION ENGINE
# =============================================================================
def simulate(df, thresholds, planned_idx, nights, with_battery=True, solar_priority=False):
    """Run the 5-minute backtest.

    Battery is modelled as two energy buckets (STORED kWh):
      green_kwh  - energy that originated from on-site solar surplus
      grey_kwh   - energy bought from the grid for arbitrage
    Truck charging prefers green (solar) energy, then grid, to maximise solar
    utilisation for decarbonisation. Arbitrage exports sell grey first.

    solar_priority=True:
      * grid-arbitrage charging is capped at cap*(1-SOLAR_RESERVE_FRAC) so the
        upper band of the battery stays free to absorb the day's solar, and
      * stored solar (green) is NOT sold to the grid -- it is reserved for the
        trucks. Trades some arbitrage profit for a higher solar/truck share.
    """
    cap = BATTERY_CAPACITY_KWH if with_battery else 0.0
    grid_charge_cap = cap * (1 - SOLAR_RESERVE_FRAC) if solar_priority else cap
    green = 0.0
    grey = 0.0

    # accumulators -----------------------------------------------------------
    grid_import_cost = 0.0      # $ paid to import (building + battery charging)
    grid_export_rev  = 0.0      # $ earned exporting solar + battery to grid
    batt_charge_cost = 0.0      # $ paid specifically to grid-charge the battery
    batt_export_rev  = 0.0      # $ earned specifically discharging battery to grid
    wear_cost        = 0.0

    truck_grid_cost  = 0.0      # $ paid to charge trucks from the grid
    truck_kwh_solar  = 0.0      # truck energy delivered from solar (via battery)
    truck_kwh_grid   = 0.0      # truck energy delivered from the grid

    batt_grid_import_kwh = 0.0  # battery energy taken FROM grid (metered input)
    batt_grid_export_kwh = 0.0  # battery energy sold TO grid (metered output)
    batt_solar_charge_kwh = 0.0 # solar energy stored into battery (input)
    batt_throughput_kwh   = 0.0 # delivered output, for cycle counting

    # per-night remaining truck demand
    truck_need = dict(nights) if with_battery or True else {}
    # (trucks must charge in both scenarios; battery only changes the source)

    # hourly import/export profile (kWh summed by hour-of-day)
    hour_import = np.zeros(24)
    hour_export = np.zeros(24)

    # how the BUILDING (office/warehouse) load is served, by hour of day (kWh)
    load_from_solar = np.zeros(24)   # solar used directly behind the meter
    load_from_batt  = np.zeros(24)   # discharged from the battery to cover load
    load_from_grid  = np.zeros(24)   # imported from the grid to cover load

    daily_rows = []
    cur_date = None
    d_imp = d_exp = d_batt_rev = d_batt_cost = 0.0

    prices = df["price"].values
    solars = df["solar_kwh"].values
    loads = df["load_kwh"].values
    dts = df["dt"].values
    idxs = df.index.values

    for i in range(len(df)):
        ts = pd.Timestamp(dts[i])
        price = prices[i]
        s = solars[i]
        L = loads[i]
        date = ts.normalize()
        hr = ts.hour

        if cur_date is None:
            cur_date = date
        if date != cur_date:
            daily_rows.append((cur_date, d_imp, d_exp, d_batt_rev - d_batt_cost))
            d_imp = d_exp = d_batt_rev = d_batt_cost = 0.0
            cur_date = date

        avg, pmin, pmax, imp_price, exp_price = thresholds[date]
        inv_budget = INVERTER_KWH_PER_INTERVAL    # shared bidirectional budget

        # --- 1. Solar directly serves building load (free, behind meter) -----
        direct = min(s, L)
        s -= direct
        L -= direct
        load_from_solar[hr] += direct

        # --- 2. Surplus solar -> charge battery (green), else export ---------
        if s > 0:
            room = cap - (green + grey)
            charge_in = min(s, room / ETA_C if ETA_C > 0 else 0, inv_budget)
            if charge_in > 0:
                green += charge_in * ETA_C
                batt_solar_charge_kwh += charge_in
                inv_budget -= charge_in
                s -= charge_in
            if s > 0:   # remaining surplus exported
                grid_export_rev += s * price
                d_exp += s
                hour_export[hr] += s
                s = 0.0

        # --- 3. Remaining building load: battery if price high, else grid ----
        if L > 0:
            if with_battery and price >= exp_price and (green + grey) > RESERVE_KWH:
                # In solar-priority mode only grey energy may discharge here (green
                # is reserved for the trucks); otherwise the whole battery is usable.
                pool = grey if solar_priority else (green + grey)
                avail = min(pool, (green + grey) - RESERVE_KWH, inv_budget)
                draw = min(L / ETA_D, avail) if avail > 0 else 0.0  # stored kWh to draw
                if draw > 0:
                    # use grey first (keep green for trucks)
                    g_grey = min(grey, draw); grey -= g_grey
                    g_green = draw - g_grey; green -= g_green
                    delivered = draw * ETA_D
                    L -= delivered
                    inv_budget -= draw
                    batt_throughput_kwh += delivered
                    wear_cost += draw * BATTERY_WEAR_COST_KWH
                    load_from_batt[hr] += delivered
            if L > 0:                                    # rest from grid
                grid_import_cost += L * price
                d_imp += L
                hour_import[hr] += L
                load_from_grid[hr] += L
                L = 0.0

        # --- 4. Arbitrage with remaining inverter budget ---------------------
        if with_battery and inv_budget > 0:
            total = green + grey
            # In solar-priority mode grid charging stops at grid_charge_cap, leaving
            # the top band free for solar; solar (step 2) may still fill to full cap.
            charging = total < grid_charge_cap - 1e-9 and (price <= imp_price or price <= LOW_PRICE_IMPORT)
            if charging:
                # PowerMaker: at/below LOW_PRICE_IMPORT charge at FULL rate; otherwise
                # use the exponential curve that trades harder the cheaper it gets.
                if price <= LOW_PRICE_IMPORT:
                    rate_kw = IE_MAX_RATE_KW
                else:
                    rate_kw = calc_charge_rate_kw(price, imp_price, pmin)
                want = min(rate_kw * INTERVAL_HOURS, inv_budget, (grid_charge_cap - total) / ETA_C)
                if want > 0:
                    grey += want * ETA_C
                    cost = want * price
                    batt_charge_cost += cost
                    grid_import_cost += cost
                    batt_grid_import_kwh += want
                    d_imp += want
                    d_batt_cost += cost
                    hour_import[hr] += want
                    inv_budget -= want
            elif price >= exp_price and total > RESERVE_KWH:
                rate_kw = calc_discharge_rate_kw(price, exp_price, pmax)
                # Solar-priority: never sell stored solar (green) -- only grey.
                sell_pool = grey if solar_priority else total
                cap_draw = min(sell_pool, total - RESERVE_KWH)
                draw = min(rate_kw * INTERVAL_HOURS, inv_budget, cap_draw)
                if draw > 0:
                    g_grey = min(grey, draw); grey -= g_grey      # sell grey first
                    g_green = draw - g_grey; green -= g_green
                    delivered = draw * ETA_D
                    rev = delivered * price
                    batt_export_rev += rev
                    grid_export_rev += rev
                    batt_grid_export_kwh += delivered
                    batt_throughput_kwh += delivered
                    wear_cost += draw * BATTERY_WEAR_COST_KWH
                    d_exp += delivered
                    d_batt_rev += rev
                    hour_export[hr] += delivered
                    inv_budget -= draw

        # --- 5. Truck charging (overnight, used nights) ----------------------
        night = charging_night_key(ts)
        if night is not None and night in truck_need and truck_need[night] > 1e-9:
            need = truck_need[night]
            want = min(CHARGER_KWH_PER_INTERVAL, need)

            # 5a. Prefer GREEN battery energy (stored solar) -> trucks
            if with_battery and green > 0 and inv_budget > 0:
                draw = min(want / ETA_D, green, inv_budget)
                if draw > 0:
                    delivered = draw * ETA_D
                    green -= draw
                    inv_budget -= draw
                    batt_throughput_kwh += delivered
                    wear_cost += draw * BATTERY_WEAR_COST_KWH
                    truck_kwh_solar += delivered
                    need -= delivered
                    want -= delivered

            # 5b. Remaining from grid, on planned-cheap intervals or forced top-up
            if want > 0 and need > 0:
                # remaining intervals until 06:00 to guarantee completion
                end_dt = (night + pd.Timedelta(days=1)).replace(hour=6, minute=0)
                hrs_left = max((end_dt - ts).total_seconds() / 3600.0, INTERVAL_HOURS)
                must_finish = need >= TRUCK_CHARGER_KW * hrs_left * 0.999
                if (idxs[i] in planned_idx) or must_finish:
                    g = min(want, need)
                    truck_grid_cost += g * price
                    truck_kwh_grid += g
                    grid_import_cost += g * price
                    d_imp += g
                    hour_import[hr] += g
                    need -= g
            truck_need[night] = need

    daily_rows.append((cur_date, d_imp, d_exp, d_batt_rev - d_batt_cost))

    days = (df["dt"].max() - df["dt"].min()).total_seconds() / 86400.0
    years = days / 365.0

    return {
        "years": years, "days": days,
        "grid_import_cost": grid_import_cost,
        "grid_export_rev": grid_export_rev,
        "batt_charge_cost": batt_charge_cost,
        "batt_export_rev": batt_export_rev,
        "wear_cost": wear_cost,
        "net_energy_cost": grid_import_cost - grid_export_rev + wear_cost,
        "truck_grid_cost": truck_grid_cost,
        "truck_kwh_solar": truck_kwh_solar,
        "truck_kwh_grid": truck_kwh_grid,
        "batt_grid_import_kwh": batt_grid_import_kwh,
        "batt_grid_export_kwh": batt_grid_export_kwh,
        "batt_solar_charge_kwh": batt_solar_charge_kwh,
        "batt_throughput_kwh": batt_throughput_kwh,
        "hour_import": hour_import,
        "hour_export": hour_export,
        "load_from_solar": load_from_solar,
        "load_from_batt": load_from_batt,
        "load_from_grid": load_from_grid,
        "daily_rows": daily_rows,
    }


def simulate_co2(df, thresholds, planned_idx, nights):
    """CO2-MAX + AGGRESSIVE-CYCLING strategy.

    Priority each 5-minute interval:
      1. Solar serves the office directly.
      2. Surplus solar is stored as GREEN energy (may use the whole battery).
      3. Office shortfall is covered from the battery before the grid -- GREY first
         when the grid is pricey (peak-shave), then GREEN that is surplus to tonight's
         truck reserve. GREEN is otherwise held for the trucks.
      4. Trucks charge from GREEN (stored solar) first, then grid -- BEFORE any
         arbitrage selling, so on-site demand is always served first.
      5. Battery cycling: buy GREY when cheap (<= import quantile) at full power, and
         sell GREY at ANY price above CO2_SELL_FLOOR at full power. GREEN is NEVER
         sold (reserved to decarbonise on-site load). This maximises cycling/profit
         on the grid-sourced inventory without eroding the CO2 goal.
    """
    cap = BATTERY_CAPACITY_KWH
    grid_charge_cap = cap * (1 - SOLAR_RESERVE_FRAC)   # cap on GREY; GREEN may use all
    green = 0.0
    grey = 0.0

    grid_import_cost = grid_export_rev = 0.0
    batt_charge_cost = batt_export_rev = wear_cost = 0.0
    truck_grid_cost = truck_kwh_solar = truck_kwh_grid = 0.0
    batt_grid_import_kwh = batt_grid_export_kwh = 0.0
    batt_solar_charge_kwh = batt_throughput_kwh = 0.0
    solar_export_kwh = 0.0

    truck_need = dict(nights)
    hour_import = np.zeros(24); hour_export = np.zeros(24)
    load_from_solar = np.zeros(24); load_from_batt = np.zeros(24); load_from_grid = np.zeros(24)

    daily_rows = []
    cur_date = None
    d_imp = d_exp = d_batt_rev = d_batt_cost = 0.0

    prices = df["price"].values; solars = df["solar_kwh"].values
    loads = df["load_kwh"].values; dts = df["dt"].values; idxs = df.index.values

    for i in range(len(df)):
        ts = pd.Timestamp(dts[i])
        price = prices[i]; s = solars[i]; L = loads[i]
        date = ts.normalize(); hr = ts.hour
        if cur_date is None:
            cur_date = date
        if date != cur_date:
            daily_rows.append((cur_date, d_imp, d_exp, d_batt_rev - d_batt_cost))
            d_imp = d_exp = d_batt_rev = d_batt_cost = 0.0
            cur_date = date
        avg, pmin, pmax, imp_price, exp_price = thresholds[date]
        inv = INVERTER_KWH_PER_INTERVAL

        # 1. Solar -> office (free, behind meter)
        direct = min(s, L); s -= direct; L -= direct
        load_from_solar[hr] += direct

        # 2. Surplus solar -> GREEN (whole battery available), overflow exported
        if s > 0:
            room = cap - (green + grey)
            cin = min(s, room / ETA_C if ETA_C > 0 else 0, inv)
            if cin > 0:
                green += cin * ETA_C; batt_solar_charge_kwh += cin; inv -= cin; s -= cin
            if s > 0:
                grid_export_rev += s * price; solar_export_kwh += s
                d_exp += s; hour_export[hr] += s; s = 0.0

        # reserve GREEN for tonight's trucks so the office doesn't drain it
        nk = upcoming_night(ts)
        reserve_green = truck_need.get(nk, 0.0) if nk is not None else 0.0

        # 3. Office shortfall -> battery before grid
        if L > 0 and inv > 0:
            # 3a. peak-shave with GREY when the grid is pricey
            if price >= CO2_SELL_FLOOR and grey > 0:
                draw = min(L / ETA_D, grey, inv, (green + grey) - RESERVE_KWH)
                if draw > 0:
                    grey -= draw; delivered = draw * ETA_D; L -= delivered
                    inv -= draw; batt_throughput_kwh += delivered
                    wear_cost += draw * BATTERY_WEAR_COST_KWH; load_from_batt[hr] += delivered
            # 3b. GREEN surplus to the truck reserve may serve the office
            usable_green = max(0.0, green - reserve_green)
            if L > 0 and usable_green > 0 and inv > 0:
                draw = min(L / ETA_D, usable_green, inv, (green + grey) - RESERVE_KWH)
                if draw > 0:
                    green -= draw; delivered = draw * ETA_D; L -= delivered
                    inv -= draw; batt_throughput_kwh += delivered
                    wear_cost += draw * BATTERY_WEAR_COST_KWH; load_from_batt[hr] += delivered
            if L > 0:                                    # remainder from grid
                grid_import_cost += L * price; d_imp += L
                hour_import[hr] += L; load_from_grid[hr] += L; L = 0.0

        # 4. Trucks (window): GREEN first, then grid -- served before arbitrage selling
        night = charging_night_key(ts)
        if night is not None and night in truck_need and truck_need[night] > 1e-9:
            need = truck_need[night]
            want = min(CHARGER_KWH_PER_INTERVAL, need)
            if green > 0 and inv > 0:                    # 4a. stored solar -> trucks
                draw = min(want / ETA_D, green, inv)
                if draw > 0:
                    delivered = draw * ETA_D; green -= draw; inv -= draw
                    batt_throughput_kwh += delivered; wear_cost += draw * BATTERY_WEAR_COST_KWH
                    truck_kwh_solar += delivered; need -= delivered; want -= delivered
            if want > 0 and need > 0:                    # 4b. grid (planned-cheap or forced)
                end_dt = (night + pd.Timedelta(days=1)).replace(hour=6, minute=0)
                hrs_left = max((end_dt - ts).total_seconds() / 3600.0, INTERVAL_HOURS)
                must_finish = need >= TRUCK_CHARGER_KW * hrs_left * 0.999
                if (idxs[i] in planned_idx) or must_finish:
                    g = min(want, need)
                    truck_grid_cost += g * price; truck_kwh_grid += g
                    grid_import_cost += g * price; d_imp += g
                    hour_import[hr] += g; need -= g
            truck_need[night] = need

        # 5. Battery cycling on GREY: buy cheap / sell at any profit, full power
        total = green + grey
        if price <= imp_price and total < grid_charge_cap - 1e-9 and inv > 0:
            want = min(IE_MAX_RATE_KW * INTERVAL_HOURS, inv, (grid_charge_cap - total) / ETA_C)
            if want > 0:
                grey += want * ETA_C; cost = want * price
                batt_charge_cost += cost; grid_import_cost += cost
                batt_grid_import_kwh += want; d_imp += want; d_batt_cost += cost
                hour_import[hr] += want; inv -= want
        elif price >= CO2_SELL_FLOOR and grey > 0 and inv > 0:
            draw = min(IE_MAX_RATE_KW * INTERVAL_HOURS, inv, grey, total - RESERVE_KWH)
            if draw > 0:
                grey -= draw; delivered = draw * ETA_D; rev = delivered * price
                batt_export_rev += rev; grid_export_rev += rev
                batt_grid_export_kwh += delivered; batt_throughput_kwh += delivered
                wear_cost += draw * BATTERY_WEAR_COST_KWH
                d_exp += delivered; d_batt_rev += rev; hour_export[hr] += delivered; inv -= draw

    daily_rows.append((cur_date, d_imp, d_exp, d_batt_rev - d_batt_cost))
    days = (df["dt"].max() - df["dt"].min()).total_seconds() / 86400.0

    return {
        "years": days / 365.0, "days": days,
        "grid_import_cost": grid_import_cost, "grid_export_rev": grid_export_rev,
        "batt_charge_cost": batt_charge_cost, "batt_export_rev": batt_export_rev,
        "wear_cost": wear_cost,
        "net_energy_cost": grid_import_cost - grid_export_rev + wear_cost,
        "truck_grid_cost": truck_grid_cost,
        "truck_kwh_solar": truck_kwh_solar, "truck_kwh_grid": truck_kwh_grid,
        "batt_grid_import_kwh": batt_grid_import_kwh, "batt_grid_export_kwh": batt_grid_export_kwh,
        "batt_solar_charge_kwh": batt_solar_charge_kwh, "batt_throughput_kwh": batt_throughput_kwh,
        "solar_export_kwh": solar_export_kwh,
        "hour_import": hour_import, "hour_export": hour_export,
        "load_from_solar": load_from_solar, "load_from_batt": load_from_batt,
        "load_from_grid": load_from_grid, "daily_rows": daily_rows,
    }


# =============================================================================
# REPORTING
# =============================================================================
def money(x):
    return f"${x:,.0f}"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, DATA_FILE)
    if not os.path.exists(path):
        path = DATA_FILE
    print("Loading", os.path.basename(path), "...")
    df = load_data(path)
    df = df.set_index(pd.RangeIndex(len(df)))  # clean integer index for planned_idx
    n_years = (df["dt"].max() - df["dt"].min()).total_seconds() / 86400.0 / 365.0
    n_days = n_years * 365.0
    avg_solar_day = df["solar_kwh"].sum() / n_days
    avg_load_day = df["load_kwh"].sum() / n_days
    print(f"  {len(df):,} intervals | {df['dt'].min()} -> {df['dt'].max()} | {n_years:.2f} years")
    print(f"  Solar x{SOLAR_SCALE:g} -> {avg_solar_day:.0f} kWh/day avg | "
          f"building load {avg_load_day:.0f} kWh/day avg\n")

    print("Computing 5-day rolling price thresholds ...")
    base_stats = base_price_stats(df)
    thresholds = apply_margin(base_stats, MIN_MARGIN)

    print("Planning overnight truck charging schedule ...")
    planned_idx, nights = plan_truck_grid_intervals(df)
    truck_charge_nights = len(nights)
    print(f"  {truck_charge_nights} charging nights over the period "
          f"(~{truck_charge_nights / n_years:.0f}/yr, target {TRUCK_USAGE_DAYS_PER_WK}/wk)\n")

    print("Running scenario: pure arbitrage (PowerMaker default) ...")
    with_b = simulate(df, thresholds, planned_idx, nights, with_battery=True, solar_priority=False)
    print("Running scenario: solar-priority truck charging ...")
    with_sp = simulate(df, thresholds, planned_idx, nights, with_battery=True, solar_priority=True)
    print("Running scenario: CO2-MAX + aggressive cycling (NEW goal) ...")
    with_co2 = simulate_co2(df, thresholds, planned_idx, nights)
    print("Running BASELINE scenario (no battery) ...\n")
    no_b = simulate(df, thresholds, planned_idx, nights, with_battery=False)

    head = with_co2          # the strategy the report headlines (the new goal)
    y = head["years"]

    # ---------------- Diesel + baseline economics ---------------------------
    diesel_ef0 = DIESEL_KGCO2_PER_L + (DIESEL_UPSTREAM_KGCO2_PER_L if DIESEL_INCLUDE_UPSTREAM else 0.0)
    diesel_l_per_year = (truck_charge_nights / y) * NUM_TRUCKS * TRUCK_KM_PER_DAY / 100.0 * DIESEL_L_PER_100KM
    diesel_co2_0 = diesel_l_per_year * diesel_ef0
    diesel_cost_saved = diesel_l_per_year * DIESEL_PRICE_PER_L     # $/yr fuel no longer bought

    # Clean reference baseline: office with 2x solar, NO battery, NO EV load (diesel trucks).
    # Office-only electricity bill computed directly: solar serves load, surplus exported.
    imp_office = (df["load_kwh"] - df["solar_kwh"]).clip(lower=0)
    exp_office = (df["solar_kwh"] - df["load_kwh"]).clip(lower=0)
    office_only_net = float((imp_office * df["price"] - exp_office * df["price"]).sum()) / y

    # Total annual operating profit vs that baseline =
    #   diesel fuel avoided  +  (baseline office elec bill  -  proposed elec bill incl. trucks)
    def total_profit(scn):
        return diesel_cost_saved + (office_only_net - scn["net_energy_cost"] / y)

    def summarise(scn):
        tot = total_profit(scn)
        elec = office_only_net - scn["net_energy_cost"] / y   # electricity component only
        cyc = (scn["batt_throughput_kwh"] / scn["days"]) / BATTERY_CAPACITY_KWH
        tsolar = scn["truck_kwh_solar"]; tgrid = scn["truck_kwh_grid"]
        tfrac = tsolar / (tsolar + tgrid) if (tsolar + tgrid) else 0.0
        net_co2 = diesel_co2_0 - tgrid / y * VIC_GRID_KGCO2_PER_KWH
        return tot, elec, cyc, tfrac, net_co2

    print("=" * 90)
    print("STRATEGY COMPARISON  (annualised, vs office+2x-solar+diesel-trucks baseline)")
    print("=" * 90)
    print(f"  {'strategy':32}{'TOTAL $/yr':>11}{'(elec)':>9}{'(diesel)':>10}"
          f"{'cycles/d':>10}{'trk solar':>11}{'CO2 t/yr':>10}")
    print("  " + "-" * 86)
    for label, scn in [("1. Pure arbitrage (default)", with_b),
                       ("2. Solar-priority trucks", with_sp),
                       ("3. CO2-max + aggressive cycle", with_co2)]:
        tot, elec, c, f, co2 = summarise(scn)
        print(f"  {label:32}{money(tot):>11}{money(elec):>9}{money(diesel_cost_saved):>10}"
              f"{c:>10.2f}{f*100:>10.0f}%{co2/1000:>10.1f}")
    print()
    print(f"  Diesel avoided is {money(diesel_cost_saved)}/yr for all strategies "
          f"({diesel_l_per_year:,.0f} L @ ${DIESEL_PRICE_PER_L}/L); strategies differ on the")
    print("  electricity component. Strategy 3 (active goal) details in Q1-Q3 below.")
    print()

    # ---------------- Q1: total operating profit (electricity + diesel) -----
    net_with = head["net_energy_cost"] / y
    elec_saving = office_only_net - net_with          # electricity component vs baseline
    batt_benefit = (no_b["net_energy_cost"] - head["net_energy_cost"]) / y   # value of the battery
    annual_trade_margin = (head["batt_export_rev"] - head["batt_charge_cost"] - head["wear_cost"]) / y
    total_op = diesel_cost_saved + elec_saving

    print("=" * 78)
    print("Q1.  ANNUAL OPERATING PROFIT  (electricity arbitrage + diesel displacement)")
    print("=" * 78)
    print("  Baseline = office with 2x solar, NO battery, DIESEL trucks.")
    print()
    print("  A) DIESEL FUEL avoided (trucks now electric):")
    print(f"       {diesel_l_per_year:,.0f} L/yr @ ${DIESEL_PRICE_PER_L}/L"
          f"               = {money(diesel_cost_saved)} /yr")
    print()
    print("  B) ELECTRICITY bill change vs baseline:")
    print(f"       baseline office-only elec cost          = {money(office_only_net)} /yr")
    print(f"       proposed elec cost (office+trucks+batt) = {money(net_with)} /yr")
    print(f"       -> electricity saving                   = {money(elec_saving)} /yr")
    print(f"          (incl. battery's own benefit of {money(batt_benefit)}/yr & grid")
    print(f"           trading margin of {money(annual_trade_margin)}/yr on grey energy)")
    print()
    print(f"  => TOTAL ANNUAL OPERATING PROFIT  (A + B)    = {money(total_op)} /yr   <== headline")
    print()

    # ---------------- Q2: daily import/export cycle -------------------------
    days = head["days"]
    imp_day = head["batt_grid_import_kwh"] / days
    exp_day = head["batt_grid_export_kwh"] / days
    solar_day = head["batt_solar_charge_kwh"] / days
    cycles_day = (head["batt_throughput_kwh"] / days) / BATTERY_CAPACITY_KWH
    arb_cyc = (with_b["batt_throughput_kwh"] / with_b["days"]) / BATTERY_CAPACITY_KWH
    print("=" * 78)
    print("Q2.  DAILY BATTERY IMPORT / EXPORT CYCLE  (CO2-max + aggressive cycling)")
    print("=" * 78)
    print(f"  Avg grid energy IMPORTED to battery : {imp_day:6.1f} kWh/day (grey, for cycling)")
    print(f"  Avg solar energy stored to battery  : {solar_day:6.1f} kWh/day (green, for on-site)")
    print(f"  Avg GREY energy SOLD to grid         : {exp_day:6.1f} kWh/day")
    print(f"  Equivalent full cycles              : {cycles_day:6.2f} cycles/day "
          f"(~{cycles_day*365:.0f}/yr)")
    print()
    print(f"  This is {cycles_day/arb_cyc:.0f}x the {arb_cyc:.2f} cycles/day of the default arbitrage")
    print("  strategy. Selling grey energy at ANY price above the ${:.2f}/kWh break-even".format(CO2_SELL_FLOOR))
    print("  (at full inverter power, instead of PowerMaker's spike-throttled rate curve)")
    print("  trades far more often. Stored SOLAR (green) is never sold -- it is held to")
    print("  decarbonise the trucks and office, which is why solar storage is separate above.")
    print()
    print("  Average 24h shape (kWh charged vs discharged, by hour of day):")
    hi = head["hour_import"] / days
    ho = head["hour_export"] / days
    peak = max(hi.max(), ho.max()) or 1.0
    for h in range(24):
        ci = "#" * int(round(20 * hi[h] / peak))
        co = "=" * int(round(20 * ho[h] / peak))
        print(f"   {h:02d}:00  charge {hi[h]:5.1f} |{ci:<20}|  discharge {ho[h]:5.1f} |{co:<20}|")
    print()

    # ---------------- Q3: truck CO2 savings ---------------------------------
    truck_days_per_year = truck_charge_nights / y
    km_per_year = truck_days_per_year * NUM_TRUCKS * TRUCK_KM_PER_DAY
    diesel_l_per_year = km_per_year / 100.0 * DIESEL_L_PER_100KM
    diesel_ef = DIESEL_KGCO2_PER_L + (DIESEL_UPSTREAM_KGCO2_PER_L if DIESEL_INCLUDE_UPSTREAM else 0.0)
    diesel_co2 = diesel_l_per_year * diesel_ef

    def co2_for(scn):
        t_solar = scn["truck_kwh_solar"] / y
        t_grid = scn["truck_kwh_grid"] / y
        t_total = t_solar + t_grid
        frac = t_solar / t_total if t_total else 0.0
        ev = t_grid * VIC_GRID_KGCO2_PER_KWH + t_solar * SOLAR_KGCO2_PER_KWH
        return dict(solar=t_solar, grid=t_grid, total=t_total, frac=frac,
                    ev_co2=ev, net=diesel_co2 - ev)

    print("=" * 78)
    print("Q3.  ANNUAL CO2 SAVED BY THE ELECTRIC TRUCKS")
    print("=" * 78)
    print(f"  Truck operation : {truck_days_per_year:.0f} days/yr x {NUM_TRUCKS} trucks "
          f"x {TRUCK_KM_PER_DAY:.0f} km = {km_per_year:,.0f} km/yr")
    print(f"  Diesel displaced: {diesel_l_per_year:,.0f} L/yr @ {diesel_ef:.2f} kgCO2/L"
          f"  -> {diesel_co2:,.0f} kg CO2/yr GROSS (tailpipe) avoided")
    print()
    print("  Charging mix and NET saving depend on the charging strategy:")
    print(f"  {'':30}{'solar kWh':>11}{'grid kWh':>10}{'solar %':>9}"
          f"{'grid CO2':>11}{'NET saved':>12}")
    for label, scn in [("1. Pure arbitrage", with_b),
                       ("2. Solar-priority trucks", with_sp),
                       ("3. CO2-max + cycle (active)", with_co2)]:
        m = co2_for(scn)
        print(f"  {label:30}{m['solar']:>11,.0f}{m['grid']:>10,.0f}{m['frac']*100:>8.0f}%"
              f"{m['ev_co2']:>9,.0f}kg{m['net']:>9,.0f}kg")
    print()
    base = co2_for(with_b)
    co2m = co2_for(with_co2)
    print(f"  -> NET CO2 SAVED (this strategy)    : {co2m['net']:,.0f} kg/yr "
          f"({co2m['net']/1000:.1f} t)   <== headline")
    print(f"     Lifts truck solar share {base['frac']*100:.0f}% -> {co2m['frac']*100:.0f}% vs the default,")
    print(f"     saving an extra {(co2m['net']-base['net']):,.0f} kg/yr of CO2 while also cycling harder.")
    print()
    print(f"  NOTE: even doubled, solar (~{avg_solar_day:.0f} kWh/day) is close to building")
    print(f"  load (~{avg_load_day:.0f} kWh/day) and trucks charge overnight (no sun), so solar")
    print("  reaches them only via stored daytime surplus. VIC's high-emission grid (0.79)")
    print("  means grid-charged EV km save far less CO2 than the gross diesel figure implies.")
    print()

    # ---------------- Explicit statistical regression -----------------------
    # Fit annual-scaled daily arbitrage proxy vs daily price spread.
    print("=" * 78)
    print("REGRESSION:  daily battery profit  ~  daily price spread  (arbitrage strategy)")
    print("=" * 78)
    print("  (Fitted on the spread-sensitive pure-arbitrage strategy; the CO2 strategy")
    print("   sells on volume at a flat break-even, so its profit is spread-independent.)")
    daily = pd.DataFrame(with_b["daily_rows"], columns=["date", "imp_kwh", "exp_kwh", "batt_profit"])
    pstats = df.groupby("date")["price"].agg(["min", "max", "mean", "std"])
    daily = daily.merge(pstats, left_on="date", right_index=True, how="left")
    daily["spread"] = daily["max"] - daily["min"]
    # Regress daily battery arbitrage profit ($) on the daily price spread ($/kWh).
    x = daily["spread"].values
    yv = daily["batt_profit"].values
    mask = np.isfinite(x) & np.isfinite(yv)
    x, yv = x[mask], yv[mask]
    if len(x) > 2 and x.std() > 0:
        slope, intercept = np.polyfit(x, yv, 1)
        pred = slope * x + intercept
        ss_res = np.sum((yv - pred) ** 2)
        ss_tot = np.sum((yv - yv.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
        corr = np.corrcoef(x, yv)[0, 1]
        print(f"  daily battery profit ($) = {slope:.2f} * (daily $/kWh spread) + {intercept:.2f}")
        print(f"  R^2 = {r2:.3f} | Pearson r = {corr:.3f} | n = {len(x)} days")
        print(f"  mean daily battery profit = ${yv.mean():.2f} | mean daily spread = ${x.mean():.3f}/kWh")
        print("  (Daily arbitrage profit rises with the day's price spread/volatility.)")
    print()

    # ---------------- MIN_MARGIN sensitivity sweep --------------------------
    print("=" * 78)
    print("SENSITIVITY:  MIN_MARGIN  (export gate above 5-day average $/kWh)")
    print("=" * 78)
    print("  Lower margin = trade more often = more profit & cycling, but more wear.")
    print("  (Pure-arbitrage mode; baseline-no-battery cost is margin-independent.)\n")
    print(f"  {'MIN_MARGIN':>10}{'annual benefit':>16}{'trade margin':>14}"
          f"{'cycles/day':>12}{'export kWh/d':>14}")
    print("  " + "-" * 64)
    best = None
    for m in MARGIN_SWEEP:
        th_m = apply_margin(base_stats, m)
        r = simulate(df, th_m, planned_idx, nights, with_battery=True, solar_priority=False)
        benefit = (no_b["net_energy_cost"] - r["net_energy_cost"]) / y
        margin = (r["batt_export_rev"] - r["batt_charge_cost"] - r["wear_cost"]) / y
        cyc = (r["batt_throughput_kwh"] / r["days"]) / BATTERY_CAPACITY_KWH
        exp_d = r["batt_grid_export_kwh"] / r["days"]
        flag = "  <- default" if abs(m - MIN_MARGIN) < 1e-9 else ""
        print(f"  {m:>10.2f}{money(benefit):>16}{money(margin):>14}"
              f"{cyc:>12.2f}{exp_d:>14.1f}{flag}")
        if best is None or benefit > best[1]:
            best = (m, benefit)
    print(f"\n  Best annual benefit in sweep: {money(best[1])}/yr at MIN_MARGIN=${best[0]:.2f}")
    print("  Note how cycles/day barely move as the margin drops: the binding constraint")
    print("  is the discharge-RATE curve (throttled by spikes), not the export gate. Above")
    print("  ~$0.14 the gate starts to bite (fewer, higher-price sells). The bigger lever is")
    print("  the rate curve; if you add BATTERY_WEAR_COST_KWH, the optimum margin rises.")
    print()

    # ---------------- Assumptions echo --------------------------------------
    print("=" * 78)
    print("KEY ASSUMPTIONS USED (edit at top of script and re-run)")
    print("=" * 78)
    print(f"  Solar array scaled x{SOLAR_SCALE:g} ({avg_solar_day:.0f} kWh/day avg)")
    print(f"  Battery {BATTERY_CAPACITY_KWH:.0f} kWh / {INVERTER_KW:.0f} kW inverter | "
          f"round-trip eff {ROUND_TRIP_EFFICIENCY*100:.0f}% | reserve {BATTERY_RESERVE_FRAC*100:.0f}%")
    print(f"  Wear cost {money(BATTERY_WEAR_COST_KWH)}/kWh | MIN_MARGIN ${MIN_MARGIN}/kWh | "
          f"import<=Q{IMPORT_QUANTILE} export>=Q{EXPORT_QUANTILE}")
    print(f"  Solar/CO2 modes reserve {SOLAR_RESERVE_FRAC*100:.0f}% of the battery for solar "
          f"& protect stored solar for on-site use")
    print(f"  CO2-max mode sells grid-bought energy at any price >= ${CO2_SELL_FLOOR}/kWh "
          f"(full inverter power)")
    print(f"  Trucks {NUM_TRUCKS} x {TRUCK_BATTERY_KWH:.0f} kWh, {TRUCK_USAGE_DAYS_PER_WK} days/wk, "
          f"{TRUCK_KM_PER_DAY:.0f} km/day, charge 17:00-06:00")
    print(f"  Diesel {DIESEL_L_PER_100KM} L/100km @ {diesel_ef:.2f} kgCO2/L & ${DIESEL_PRICE_PER_L}/L | "
          f"VIC grid {VIC_GRID_KGCO2_PER_KWH} kgCO2/kWh")
    print(f"  Upfront capital cost: $0 (per brief) -> all figures are operating cash flow")
    print("=" * 78)

    # Save daily detail for further analysis
    out_csv = os.path.join(here, "arbitrage_daily_results.csv")
    daily.to_csv(out_csv, index=False)
    print(f"\nDaily detail written to {out_csv}")


if __name__ == "__main__":
    main()
