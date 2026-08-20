import os
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
import MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone

HISTORY_MONTHS = 8
SAVE_INCREMENTAL = False

MACRO_TICKERS = {
    'US_10Y': 'DGS10',
    'DE_10Y': 'IRLTLT01DEM156N',
    'DXY': 'DX-Y.NYB',
}

MT5_SYMBOLS = {
    'EURUSD': 'EURUSD',
    'GBPUSD': 'GBPUSD',
    'USDJPY': 'USDJPY',
    'SP500': 'SPX500',
    'GOLD': 'XAUUSD',
}

RAW_DATA_DIR = 'data/raw'

def setup_environment():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

def save_data(new_df, filename):
    path = f"{RAW_DATA_DIR}/{filename}"
    if new_df.index.tz is not None:
        new_df.index = new_df.index.tz_localize(None)

    if SAVE_INCREMENTAL and os.path.exists(path):
        existing = pd.read_csv(path, index_col=0, parse_dates=True)
        if existing.index.tz is not None:
            existing.index = existing.index.tz_localize(None)
        combined = pd.concat([existing, new_df])
        combined = combined[~combined.index.duplicated(keep='last')].sort_index()
        combined.to_csv(path)
        print(f"{filename}: merged, {len(combined)} rows total")
    else:
        new_df.to_csv(path)
        print(f"{filename}: saved {len(new_df)} rows")

def fetch_macro_data(days=730):
    end = datetime.now()
    start = end - timedelta(days=days)

    try:
        us_yield = web.DataReader(MACRO_TICKERS['US_10Y'], 'fred', start, end)
        de_yield = web.DataReader(MACRO_TICKERS['DE_10Y'], 'fred', start, end)
    except Exception as e:
        print(f"FRED fetch failed: {e}")
        return None

    yahoo_raw = yf.download(MACRO_TICKERS['DXY'], start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(yahoo_raw.columns, pd.MultiIndex):
        yahoo_raw.columns = yahoo_raw.columns.get_level_values(0)
    dxy_close = yahoo_raw['Close']

    combined = us_yield.join(de_yield, how='outer').join(dxy_close, how='outer')
    combined = combined.ffill().dropna()

    if combined.empty:
        return None

    combined['Yield_Spread'] = combined['DGS10'] - combined['IRLTLT01DEM156N']
    final = combined[['DGS10', 'IRLTLT01DEM156N', 'Close', 'Yield_Spread']].rename(
        columns={'DGS10': 'US_10Y', 'IRLTLT01DEM156N': 'DE_10Y', 'Close': 'DXY'}
    )
    save_data(final, "macro_yields.csv")

def get_mt5_symbol_data(symbol, timeframe, n_months):
    utc_to = datetime.now(timezone.utc)
    utc_from = utc_to - timedelta(days=n_months * 30)
    rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_to)

    if rates is None or len(rates) == 0:
        print(f"{symbol}: fetch failed ({mt5.last_error()})")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df[['open', 'high', 'low', 'close']]

def fetch_mt5_nexus_data():
    if not mt5.initialize():
        print("MT5 initialization failed")
        return

    tf_map = {
        '15m': (mt5.TIMEFRAME_M15, 'eurusd_nexus_15m.csv'),
        '30m': (mt5.TIMEFRAME_M30, 'eurusd_30m.csv'),
        '1h': (mt5.TIMEFRAME_H1, 'eurusd_1h.csv'),
        '1d': (mt5.TIMEFRAME_D1, 'eurusd_daily.csv'),
    }

    for tf_name, (mt5_tf, filename) in tf_map.items():
        main_df = get_mt5_symbol_data(MT5_SYMBOLS['EURUSD'], mt5_tf, HISTORY_MONTHS)
        if main_df.empty:
            print(f"{tf_name}: no EURUSD data, skipping")
            continue

        nexus_df = main_df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})
        for asset in ['GBPUSD', 'USDJPY', 'SP500', 'GOLD']:
            asset_df = get_mt5_symbol_data(MT5_SYMBOLS[asset], mt5_tf, HISTORY_MONTHS)
            if not asset_df.empty:
                nexus_df[asset] = asset_df['close']

        before = len(nexus_df)
        nexus_df = nexus_df.ffill().dropna()
        if before != len(nexus_df):
            print(f"{tf_name}: dropped {before - len(nexus_df)} incomplete rows")

        save_data(nexus_df, filename)

    mt5.shutdown()

setup_environment()
fetch_macro_data(days=HISTORY_MONTHS * 30 + 60)
fetch_mt5_nexus_data()
