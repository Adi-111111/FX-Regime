import pandas as pd
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from vae_policy import VAE
from data_engine import DataEngine
from regime_engine import RegimeEngine
from meta_filter import MetaFilter
from risk_manager import RiskManager
from pipeline_fx import TRAINING_CUTOFF

REALISTIC_MODE = True
ALPHA_STATE = 3
CONFIDENCE_REQ = 0.9
META_THRESHOLD = 0.4
MIN_STABILITY = 2
FLOAT_SPREAD = 1.0

USE_TRAILING_STOP = False
TRAIL_TRIGGER = 2
TRAIL_DISTANCE = 1.0

# starts at the training cutoff so the audit window can't overlap training data
START_DATE = TRAINING_CUTOFF
END_DATE = "2026-01-17"

COMMISSION_PER_LOT = 3.0 if REALISTIC_MODE else 0.0
SWAP_LONG_POINTS = -13.472 if REALISTIC_MODE else 0.0
SWAP_SHORT_POINTS = 0.107 if REALISTIC_MODE else 0.0

LATENT_DIM = 6
N_REGIMES = 5
CAPITAL = 6000
PIP_VALUE_USD = 10.0

def run_hybrid_audit():
    try:
        df_price = pd.read_csv('data/raw/eurusd_nexus_15m.csv', index_col=0, parse_dates=True)
        df_macro = pd.read_csv('data/raw/macro_yields.csv', index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("data files missing")
        return

    if df_price.index.tz is not None:
        df_price.index = df_price.index.tz_localize(None)
    if df_macro.index.tz is not None:
        df_macro.index = df_macro.index.tz_localize(None)

    df = df_price.join(df_macro, how='left')
    df[['DXY', 'US_10Y', 'DE_10Y', 'Yield_Spread']] = df[['DXY', 'US_10Y', 'DE_10Y', 'Yield_Spread']].ffill()
    df = df.dropna()

    try:
        features = np.load('models/processed_features.npy')
        if len(features) != len(df) - 130:
            raise FileNotFoundError
    except (FileNotFoundError, ValueError):
        engine = DataEngine(window_size=130)
        engine.load_scaler('models/data_scaler.pkl')
        features = engine.prepare_institutional_set(df, df['DXY'], df['US_10Y'], df['DE_10Y'], is_training=False)

    mask = (df.index >= START_DATE) & (df.index <= END_DATE)
    results = df.loc[mask].copy()
    if results.empty:
        print("no data in the specified range")
        return

    start_idx_df = np.searchsorted(df.index, pd.Timestamp(START_DATE))
    end_idx_df = np.searchsorted(df.index, pd.Timestamp(END_DATE), side='right')
    start_idx_feat = max(0, start_idx_df - 130)
    end_idx_feat = max(0, end_idx_df - 130)
    features_slice = features[start_idx_feat:end_idx_feat]

    if len(features_slice) != len(results):
        min_len = min(len(features_slice), len(results))
        features_slice = features_slice[:min_len]
        results = results.iloc[:min_len]

    vae = VAE(input_dim=features.shape[1], latent_dim=LATENT_DIM)
    vae.load_state_dict(torch.load('models/vae_model.pth', map_location='cpu', weights_only=True))
    vae.eval()

    regime_tool = RegimeEngine(n_components=N_REGIMES)
    regime_tool.load_engine('models/regime_hmm.pkl')

    meta_filter = MetaFilter()
    meta_filter.load_model('models/meta_filter.pkl')
    meta_filter.threshold = META_THRESHOLD

    scaler_macro = joblib.load('models/macro_scaler.pkl')
    risk_mgmt = RiskManager(account_balance=CAPITAL, risk_per_trade=0.02)

    with torch.no_grad():
        feat_tensor = torch.tensor(features_slice, dtype=torch.float32)
        _, mus, _, p_weights, p_sl, p_tp = vae(feat_tensor, deterministic=True)
        latent_features = mus.numpy()
        scaled_mus = regime_tool.scaler.transform(latent_features)
        states = regime_tool.model.predict(scaled_mus)
        probs = regime_tool.model.predict_proba(scaled_mus)

    results['state'] = states
    results['p_weight'] = p_weights.numpy().flatten()
    results['p_sl'], results['p_tp'] = p_sl.numpy().flatten(), p_tp.numpy().flatten()
    results['regime_conf'] = [probs[j][states[j]] for j in range(len(states))]
    results['atr'] = risk_mgmt.calculate_atr(df).loc[results.index]

    active_pos = 0
    entry_p, entry_t = 0, None
    sl, tp, ptp = 0, 0, 0
    trade_size = 0
    partial_hit = False
    current_equity = CAPITAL
    highest_price = 0
    lowest_price = 0

    equity_curve = [{'time': results.index[0], 'equity': current_equity}]
    transaction_log = []
    state_buffer = []

    spread_price_delta = FLOAT_SPREAD * 0.0001

    for i in range(len(results)):
        row, t = results.iloc[i], results.index[i]

        state_buffer.append(row['state'])
        if len(state_buffer) > MIN_STABILITY:
            state_buffer.pop(0)
        is_stable = all(s == ALPHA_STATE for s in state_buffer)

        if active_pos != 0:
            if t.hour == 22 and t.minute == 0:
                swap = SWAP_LONG_POINTS if active_pos == 1 else SWAP_SHORT_POINTS
                current_equity += (swap * 0.1) * trade_size

            current_atr = row['atr']
            if USE_TRAILING_STOP:
                if active_pos == 1:
                    if row['High'] > highest_price:
                        highest_price = row['High']
                    if (highest_price - entry_p) > (current_atr * TRAIL_TRIGGER):
                        new_sl = highest_price - (current_atr * TRAIL_DISTANCE)
                        if new_sl > sl:
                            sl = new_sl
                elif active_pos == -1:
                    if row['Low'] < lowest_price:
                        lowest_price = row['Low']
                    if (entry_p - lowest_price) > (current_atr * TRAIL_TRIGGER):
                        new_sl = lowest_price + (current_atr * TRAIL_DISTANCE)
                        if new_sl < sl:
                            sl = new_sl

            if active_pos == 1:
                is_partial = (row['High'] - spread_price_delta) >= ptp
            else:
                is_partial = (row['Low'] + spread_price_delta) <= ptp

            if is_partial and not partial_hit:
                banked_pips = (ptp - entry_p) / 0.0001 if active_pos == 1 else (entry_p - ptp) / 0.0001
                realized_usd = (banked_pips - (FLOAT_SPREAD / 2)) * (PIP_VALUE_USD * (trade_size * 0.5))
                current_equity += realized_usd

                be_level = entry_p + (0.00005 * active_pos)
                sl = max(sl, be_level) if active_pos == 1 else min(sl, be_level)
                partial_hit = True

                transaction_log.append({
                    "Entry Time": entry_t, "Exit Time": t,
                    "Type": "Long" if active_pos == 1 else "Short",
                    "Action": "Partial Bank (50%)",
                    "Price In": entry_p, "Price Out": ptp,
                    "Lots": trade_size * 0.5,
                    "PnL ($)": round(realized_usd, 2),
                    "Equity": round(current_equity, 2),
                })

            is_hard_cutoff = (t.hour == 21 and t.minute >= 45)
            if active_pos == 1:
                is_tp = (row['High'] - spread_price_delta) >= tp
                is_sl = (row['Low'] - spread_price_delta) <= sl
            else:
                is_tp = (row['Low'] + spread_price_delta) <= tp
                is_sl = (row['High'] + spread_price_delta) >= sl

            if is_tp or is_sl or is_hard_cutoff:
                exit_reason = "Time"
                if is_sl:
                    exit_p, exit_reason = sl, "Trailed SL" if USE_TRAILING_STOP else "SL"
                elif is_tp:
                    exit_p, exit_reason = tp, "TP"
                else:
                    exit_p = row['Close']

                pips = (exit_p - entry_p) / 0.0001 if active_pos == 1 else (entry_p - exit_p) / 0.0001
                mult = 0.5 if partial_hit else 1.0
                realized_usd = (pips - (FLOAT_SPREAD / 2)) * PIP_VALUE_USD * (trade_size * mult)
                current_equity += realized_usd

                transaction_log.append({
                    "Entry Time": entry_t, "Exit Time": t,
                    "Type": "Long" if active_pos == 1 else "Short",
                    "Action": f"Closed ({exit_reason})",
                    "Price In": entry_p, "Price Out": exit_p,
                    "Lots": trade_size * mult,
                    "PnL ($)": round(realized_usd, 2),
                    "Equity": round(current_equity, 2),
                })

                equity_curve.append({'time': t, 'equity': current_equity})
                active_pos, partial_hit = 0, False

        elif 7 <= t.hour <= 19:
            curr_risk = row['SP500'] / (row['GOLD'] + 1e-9)
            curr_yield = row['Yield_Spread']
            curr_rel = row['Close'] / (row['GBPUSD'] + 1e-9)

            macro_vec = np.array([[curr_risk, curr_yield, curr_rel]])
            scaled_macro = scaler_macro.transform(macro_vec)

            current_z = latent_features[i].reshape(1, -1)
            hybrid_input = np.hstack([current_z, scaled_macro])
            is_approved, success_prob = meta_filter.get_veto_decision(hybrid_input)

            if (row['state'] == ALPHA_STATE) and (row['regime_conf'] >= CONFIDENCE_REQ) and is_stable and is_approved:
                if i + 1 < len(results):
                    next_bar = results.iloc[i + 1]
                    fill_price = next_bar['Open']

                    if abs(row['p_weight']) < 0.00002:
                        continue
                    side = 1 if row['p_weight'] > 0 else -1

                    trade = risk_mgmt.get_trade_parameters(
                        fill_price, row['atr'], side,
                        sl_mult=row['p_sl'], pt_mult=row['p_tp'],
                    )
                    sizing = meta_filter.get_position_sizing(success_prob)
                    trade['lots'] *= sizing

                    current_equity -= (COMMISSION_PER_LOT * trade['lots'])

                    active_pos = side
                    entry_p = trade['entry']
                    entry_t = next_bar.name
                    sl, tp, ptp = trade['sl'], trade['tp'], trade['partial_tp']
                    trade_size = trade['lots']
                    highest_price = entry_p
                    lowest_price = entry_p

    if transaction_log:
        audit_df = pd.DataFrame(transaction_log)
        try:
            audit_df.to_excel("backtest_audit.xlsx", index=False)
        except ModuleNotFoundError:
            audit_df.to_csv("backtest_audit.csv", index=False)
        print(audit_df[['Exit Time', 'Action', 'PnL ($)']].tail(5))
    else:
        print("no trades executed")

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(16, 9))
    trades_df = pd.DataFrame(equity_curve)

    if not trades_df.empty:
        ax.step(trades_df['time'], trades_df['equity'], where='post', color='#00FF99', linewidth=2)
        final_eq = trades_df['equity'].iloc[-1]
        ret = ((final_eq - CAPITAL) / CAPITAL) * 100

        plt.text(0.02, 0.90,
                 f"{START_DATE} - {END_DATE}\n"
                 f"final equity: ${final_eq:,.2f} ({ret:+.2f}%)\n"
                 f"trades: {len(transaction_log)}\n"
                 f"alpha state: {ALPHA_STATE}\n"
                 f"trailing stop: {'on' if USE_TRAILING_STOP else 'off'}",
                 transform=ax.transAxes, fontsize=12, family='monospace',
                 bbox=dict(facecolor='black', alpha=0.8))

        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))
        ax.set_title("EUR/USD backtest audit", fontweight='bold')
        plt.show()
    else:
        print("no equity data to plot")

run_hybrid_audit()
