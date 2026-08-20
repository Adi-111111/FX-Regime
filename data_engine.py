import pandas as pd
import numpy as np
import iisignature
import joblib
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

class DataEngine:
    def __init__(self, window_size=130):
        self.window_size = window_size
        self.scaler = StandardScaler()

    def calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))

    def calculate_adx(self, df, period=14):
        plus_dm = df['High'].diff().clip(lower=0)
        minus_dm = df['Low'].diff().clip(upper=0).abs()
        tr = pd.concat([
            df['High'] - df['Low'],
            (df['High'] - df['Close'].shift()).abs(),
            (df['Low'] - df['Close'].shift()).abs(),
        ], axis=1).max(axis=1)

        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-9))
        minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-9))
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
        return dx.rolling(period).mean() / 100.0

    def get_nexus_divergences(self, df):
        eur_gbp_ratio = df['Close'] / (df['GBPUSD'] + 1e-9)
        eur_gbp_mom = eur_gbp_ratio.pct_change(5)

        risk_appetite = df['SP500'] / (df['GOLD'] + 1e-9)
        risk_appetite_norm = risk_appetite.rolling(40).rank(pct=True)

        jpy_corr = df['Close'].rolling(40).corr(df['USDJPY']).fillna(0)

        return pd.concat(
            [eur_gbp_mom, risk_appetite_norm, jpy_corr],
            axis=1, keys=['EUR_Relative_Strength', 'Risk_Appetite', 'JPY_Nexus_Corr'],
        )

    def get_asian_range_features(self, df):
        is_asian = (df.index.hour >= 22) | (df.index.hour < 6)
        session_id = (df.index - pd.Timedelta(hours=22)).normalize()

        asian_high = df['High'].where(is_asian).groupby(session_id).cummax().ffill()
        asian_low = df['Low'].where(is_asian).groupby(session_id).cummin().ffill()

        high_dist = (df['Close'] - asian_high) / df['Close']
        low_dist = (df['Close'] - asian_low) / df['Close']

        return pd.concat(
            [high_dist, low_dist], axis=1, keys=['Asian_High_Dist', 'Asian_Low_Dist']
        ).fillna(0)

    def prepare_institutional_set(self, df, dxy, us_yield, de_yield, is_training=True):
        df['rsi'] = self.calculate_rsi(df['Close']) / 100.0
        df['adx'] = self.calculate_adx(df)

        nexus_feats = self.get_nexus_divergences(df)
        spread = (us_yield - de_yield).ffill().fillna(0)
        dxy_mom = dxy.pct_change(10).fillna(0)
        asian_feats = self.get_asian_range_features(df)
        df['hour_feat'] = df.index.hour / 23.0

        context_cols = [
            df['rsi'], df['adx'], spread, dxy_mom, df['hour_feat'],
            nexus_feats['EUR_Relative_Strength'],
            nexus_feats['Risk_Appetite'],
            nexus_feats['JPY_Nexus_Corr'],
            asian_feats['Asian_High_Dist'],
            asian_feats['Asian_Low_Dist'],
        ]
        combined_exog = pd.concat(context_cols, axis=1).reindex(df.index).ffill().fillna(0)

        price_values = df[['Open', 'High', 'Low', 'Close']].values
        final_feats = []

        for i in tqdm(range(self.window_size, len(price_values)), desc="building path signatures"):
            win = price_values[i - self.window_size:i]
            win_min, win_max = win.min(axis=0), win.max(axis=0)
            norm_w = (win - win_min) / (win_max - win_min + 1e-6)
            sig = iisignature.sig(norm_w, 2)
            final_feats.append(np.concatenate([sig, combined_exog.iloc[i].values]))

        feature_matrix = np.array(final_feats)

        if is_training:
            return self.scaler.fit_transform(feature_matrix)
        try:
            return self.scaler.transform(feature_matrix)
        except Exception:
            print("scaler not fitted, falling back to fit_transform")
            return self.scaler.fit_transform(feature_matrix)

    def save_scaler(self, path='models/data_scaler.pkl'):
        joblib.dump(self.scaler, path)

    def load_scaler(self, path='models/data_scaler.pkl'):
        try:
            self.scaler = joblib.load(path)
        except FileNotFoundError:
            print(f"no scaler found at {path}")
