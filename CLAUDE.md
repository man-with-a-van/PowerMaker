# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

PowerMaker arbitrages the New Zealand spot electricity market with a stationary battery array: it polls the live spot price, decides whether to import (charge), export (discharge), or idle, and drives an inverter over Modbus TCP. A Flask web app shows status and exposes a manual override. Built with Python, Flask, MySQL, pymodbus, numpy and matplotlib; designed to run on Linux (tested on Ubuntu 20.04).

## Setup & running

There is **no package manifest** — dependencies are installed manually:

```bash
sudo pip install pymodbus pymysql flask matplotlib numpy requests
```

`config.py` is gitignored and **must be created before anything runs**:

```bash
cp exampleconfig.py config.py   # then edit credentials, IPs, thresholds
./setupdb.py                    # drops & recreates the MySQL tables (destructive)
./powermaker.py                 # control loop: import/export decisions
./webapp.py                     # Flask dev server on 0.0.0.0:5000
```

`powermaker.py` and `webapp.py` are two independent long-running processes that communicate only through the MySQL database — there is no shared in-memory state. In production the web app is served via `webapp.wsgi` under Apache + mod_wsgi (see README.md).

There is no test suite, linter, or build step. The bottom of `powermakerfunctions.py` has a block of commented-out function calls used as an ad-hoc REPL for manually exercising individual functions.

## PROD vs TEST mode

`config.PROD` is the single most important flag in the codebase. Nearly every hardware/market function in `powermakerfunctions.py` branches on it:

- **`PROD = True`**: talks to the real Modbus inverter (`config.MODBUS_CLIENT_IP`) and the live WITS / electricityinfo.co.nz spot-price API. The Modbus `client` is only constructed when `PROD` is true.
- **`PROD = False`**: returns randomized fake data (battery %, solar, load) and derives a synthetic spot price by jittering the last DB value. Lets the full loop run with no hardware attached.

When editing functions like `get_battery_status`, `get_solar_generation`, `is_CPD`, `charge_from_grid`, `discharge_to_grid`, preserve both branches. Note: `get_actual_IE` has the `PROD` check inverted relative to the others (reads hardware when `not config.PROD`) — likely a bug, be careful before "fixing" it.

## Architecture

**Control loop (`powermaker.py`)** — an infinite `while True` that, every `config.DELAY` seconds:
1. Gathers state via `powermakerfunctions.py`: spot price, 5-day price stats, solar, load, CPD status, battery charge, manual override.
2. Runs a single prioritized `if/elif` decision chain to pick an action. Priority order: manual override → CPD active → spot below `LOW_PRICE_IMPORT` → high demand → export-when-high → import-when-low → winter CPD night-charging → idle. Each branch calls `charge_from_grid(rate)`, `discharge_to_grid(rate)`, or `reset_to_default()`.
3. Writes a `DataPoint` row (price, charge, status string, suggested/actual I/E) and commits.

All errors are caught at the loop level; on any exception it attempts `reset_to_default()` to stop import/export and logs an error DataPoint, so a failure never leaves the inverter in an active state.

**Functions (`powermakerfunctions.py`)** — all the I/O: Modbus reads/writes, spot-price fetch, DB access, the charge/discharge rate math, and matplotlib graph generation. This is the module to read first.

**Rate calculation** — `calc_charge_rate` / `calc_discharge_rate` map the spot price onto an exponential curve (via `numpy.interp` + `np.exp`) between `IE_MIN_RATE` and `IE_MAX_RATE`, so the system trades harder the further the price is from the import/export thresholds. Import/export thresholds come from quantiles of the 5-day price history in `get_spot_price_stats`, floored by `MIN_MARGIN` to cover battery wear.

**Modbus register conventions** — rates are 32-bit values split across two holding registers (2703 = upper, 2704 = lower) via `utils/utilfunctions.split_into_ushorts`. Export is encoded as `max_int_32 - rate` (a large unsigned value signals negative/discharge to the inverter). `reset_to_default()` writes 0 to register 2703 to stop all I/E. Other registers: 843 = battery %, 808–810 = solar, 817–819 = load, 820–822 = grid load, 3422 = CPD status (value 3 = active).

**Web app (`webapp.py`)** — three routes: `/` (dashboard, regenerates graphs on each load via `update_graphs`), `/admin` (gated by checking the requester's IP shares the first three octets of `config.SERVER_IP`), and `/override` (POST that sets/clears the manual override in the `Config` table).

**Database** — two MySQL tables created by `setupdb.py`: `DataPoint` (append-only time series of every loop iteration) and `Config` (key/value; currently just the `Override` row). The override mechanism is the only channel from the web app back to the control loop: `update_override` writes the rate, the loop reads it via `get_override` (`'N'` = automatic, otherwise an int rate; negative = export, positive = import).

## Domain notes

- **CPD (Control Period / "Control Point Dispatch")** — a network peak-loading window where the lines company penalizes grid draw. `is_CPD()` reads it live from the inverter; `is_CPD_period()` is the seasonal window (May–Sep) used for pre-emptive overnight charging. During CPD the loop avoids importing and biases toward discharge.
- Prices throughout are dollars per kWh (the API returns $/MWh and is divided by 1000). Rates are in watts (e.g. `IE_MAX_RATE = 120000` = 120 kW).
- `utils/` holds older/standalone experiments (`spotpricechecker.py`, `whatsthespotprice.py`, `ea.py`) and the shared `utilfunctions.py`. `spotpricechecker.py` references modules (`keys`) that aren't in the repo — treat the non-`utilfunctions` files as legacy reference, not live code.

## Gotchas

- SQL is built with f-strings / string interpolation throughout (e.g. the `INSERT INTO DataPoint` statements and `update_override`). Status strings flow straight into queries — be mindful when changing what goes into a `status` value.
- `config.py`, `static/*.png` graph outputs, and `__pycache__` are gitignored; don't commit them.
