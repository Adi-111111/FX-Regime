import pandas as pd
import numpy as np
import torch
import joblib
import os
import random
from sklearn.preprocessing import StandardScaler
from data_engine import DataEngine
from vae_policy import VAE, institutional_sharpe_loss
from regime_engine import RegimeEngine
from analyse_states import correlate_states_to_returns
from risk_manager import RiskManager
from meta_filter import MetaFilter

TRAINING_CUTOFF = "2026-01-10"

def set_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def apply_dynamic_triple_barrier(df, atr_series, pt_mult=2.0, sl_mult=1.0, max_hold=35):
    labels = []
    t_close = 22
    for i in range(len(df) - max_hold):
        entry_price = df['Close'].iloc[i]
        vol = atr_series.iloc[i]
        upper_barrier = entry_price + (vol * pt_mult)
        lower_barrier = entry_price - (vol * sl_mult)

        current_time = df.index[i]
        bars_to_close = ((t_close - current_time.hour) * 4) - (current_time.minute // 15)
        vertical_barrier = min(max_hold, max(1, bars_to_close))

        outcome = 0
        for j in range(1, vertical_barrier):
            if df['High'].iloc[i + j] >= upper_barrier:
                outcome = 1
                break
            if df['Low'].iloc[i + j] <= lower_barrier:
                outcome = -1
                break
        labels.append(outcome)
    return np.array(labels)

def run_institutional_pipeline():
    set_seeds(42)
    os.makedirs('models', exist_ok=True)

    try:
        df_price = pd.read_csv('data/raw/eurusd_nexus_15m.csv', index_col=0, parse_dates=True)
        df_macro = pd.read_csv('data/raw/macro_yields.csv', index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("data files not found")
        return

    if df_price.index.tz is not None:
        df_price.index = df_price.index.tz_localize(None)
    if df_macro.index.tz is not None:
        df_macro.index = df_macro.index.tz_localize(None)

    df = df_price.join(df_macro, how='left')
    df[['DXY', 'US_10Y', 'DE_10Y', 'Yield_Spread']] = df[['DXY', 'US_10Y', 'DE_10Y', 'Yield_Spread']].ffill()
    df = df.dropna()

    window = 100
    engine = DataEngine(window_size=window)
    cutoff_dt = pd.Timestamp(TRAINING_CUTOFF)

    train_mask_strict = df.index < cutoff_dt
    df_train = df.loc[train_mask_strict].copy()
    engine.prepare_institutional_set(
        df_train,
        df['DXY'].loc[train_mask_strict],
        df['US_10Y'].loc[train_mask_strict],
        df['DE_10Y'].loc[train_mask_strict],
        is_training=True,
    )
    engine.save_scaler("models/data_scaler.pkl")

    features = engine.prepare_institutional_set(df, df['DXY'], df['US_10Y'], df['DE_10Y'], is_training=False)
    np.save('models/processed_features.npy', features)

    max_hold = 35
    risk_tool = RiskManager()
    atr = risk_tool.calculate_atr(df)
    full_labels = apply_dynamic_triple_barrier(df, atr, pt_mult=2.0, sl_mult=1.0, max_hold=max_hold)

    start_idx, end_idx = window, len(df) - max_hold
    aligned_features = features[:(end_idx - start_idx)]
    aligned_labels = full_labels[start_idx:]
    future_close = df['Close'].shift(-12)
    price_returns = (future_close / df['Close'] - 1).values[start_idx:end_idx]
    time_tensor = torch.tensor(df['hour_feat'].values[start_idx:end_idx], dtype=torch.float32).unsqueeze(1)

    aligned_index = df.index[start_idx:end_idx]
    train_mask = aligned_index < cutoff_dt

    X_train = aligned_features[train_mask]
    y_train = aligned_labels[train_mask]
    ret_train = price_returns[train_mask]
    time_train = time_tensor[train_mask]

    print(f"total samples {len(aligned_features)}, training samples {len(X_train)}")

    vae = VAE(input_dim=X_train.shape[1], latent_dim=6)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

    feat_tensor_train = torch.tensor(X_train, dtype=torch.float32)
    ret_tensor_train = torch.tensor(ret_train, dtype=torch.float32).unsqueeze(1)

    vae.train()
    best_loss, patience_counter = float('inf'), 0
    patience, min_delta = 100, 0.05

    for epoch in range(1000):
        optimizer.zero_grad()
        noise = torch.randn_like(feat_tensor_train) * 0.01
        noisy_input = feat_tensor_train + noise
        recon, mu, logvar, weights, _, _ = vae(noisy_input)

        loss = institutional_sharpe_loss(
            recon, feat_tensor_train, mu, logvar, weights, y_train, ret_tensor_train,
            temporal_features=time_train,
        )
        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        if current_loss < best_loss - min_delta:
            best_loss, patience_counter = current_loss, 0
        else:
            patience_counter += 1

        if epoch % 50 == 0:
            print(f"epoch {epoch}, loss {current_loss:.4f}, patience {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"stopped early at epoch {epoch}")
            break

    vae.eval()
    with torch.no_grad():
        _, mus, _, weights, _, _ = vae(feat_tensor_train)
        latent_train = mus.numpy()
        signals_train = weights.numpy().flatten()

    regime_tool = RegimeEngine(n_components=5)
    states_train = regime_tool.fit_regimes(latent_train)
    regime_tool.save_engine('models/regime_hmm.pkl')

    train_subset_df = df.iloc[start_idx:end_idx].loc[train_mask].copy()
    risk_ratio = (train_subset_df['SP500'] / (train_subset_df['GOLD'] + 1e-9)).values.reshape(-1, 1)
    yield_spread = train_subset_df['Yield_Spread'].values.reshape(-1, 1)
    rel_strength = (train_subset_df['Close'] / (train_subset_df['GBPUSD'] + 1e-9)).values.reshape(-1, 1)

    scaler_macro = StandardScaler()
    raw_macro_feats = scaler_macro.fit_transform(np.hstack([risk_ratio, yield_spread, rel_strength]))
    hybrid_features_train = np.hstack([latent_train, raw_macro_feats])

    hurdle = 0.00050
    direction = np.sign(signals_train)
    potential_return = ret_train * direction
    meta_labels_train = (potential_return > hurdle).astype(int)
    print(f"profitable setups: {meta_labels_train.sum()} / {len(meta_labels_train)}")

    meta_model = MetaFilter(n_estimators=500, max_depth=12)
    meta_model.train_model(hybrid_features_train, meta_labels_train)

    feat_names = ['z1', 'z2', 'z3', 'z4', 'Risk_Ratio', 'Yield_Spread', 'Rel_Strength']
    print(meta_model.get_feature_importance(feat_names))

    joblib.dump(scaler_macro, 'models/macro_scaler.pkl')
    meta_model.save_model()
    torch.save(vae.state_dict(), 'models/vae_model.pth')

    stats = correlate_states_to_returns(train_subset_df, states_train, forward_window=max_hold)
    best_state = stats['mean'].abs().idxmax()
    print(f"recommended ALPHA_STATE = {best_state}")

run_institutional_pipeline()
