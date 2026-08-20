import pandas as pd
import numpy as np

class RiskManager:
    def __init__(self, account_balance=6000, risk_per_trade=0.02, sl_mult=2.0, pt_mult=2.0):
        self.account_balance = account_balance
        self.risk_per_trade = risk_per_trade
        self.sl_mult = sl_mult
        self.pt_mult = pt_mult
        self.pip_value_std = 10.0

    def calculate_atr(self, df, window=14):
        high_low = df['High'] - df['Low']
        high_cp = (df['High'] - df['Close'].shift()).abs()
        low_cp = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        return tr.rolling(window).mean()

    def get_trade_parameters(self, current_price, current_atr, side, sl_mult=None, pt_mult=None):
        active_sl_mult = sl_mult if sl_mult is not None else self.sl_mult
        active_pt_mult = pt_mult if pt_mult is not None else self.pt_mult

        stop_distance = current_atr * active_sl_mult
        target_distance = current_atr * active_pt_mult
        partial_distance = current_atr * 1.5

        if side == 1:
            stop_loss = current_price - stop_distance
            take_profit = current_price + target_distance
            partial_tp = current_price + partial_distance
        else:
            stop_loss = current_price + stop_distance
            take_profit = current_price - target_distance
            partial_tp = current_price - partial_distance

        stop_pips = stop_distance / 0.0001
        risk_amount_usd = self.account_balance * self.risk_per_trade
        raw_lot_size = risk_amount_usd / (max(stop_pips, 1.0) * self.pip_value_std)
        final_lot_size = round(max(0.01, min(raw_lot_size, 5.0)), 2)

        return {
            'entry': current_price,
            'sl': stop_loss,
            'tp': take_profit,
            'partial_tp': partial_tp,
            'lots': final_lot_size,
            'risk_usd': risk_amount_usd,
            'sl_mult_used': active_sl_mult,
            'pt_mult_used': active_pt_mult,
        }

    def check_temporal_validity(self, current_time):
        hour = current_time.hour
        is_liquid = (7 <= hour <= 11) or (12 <= hour <= 16)
        return is_liquid and hour < 20

    def move_to_breakeven(self, entry_price, current_sl, side):
        buffer = 0.00005
        if side == 1:
            return max(current_sl, entry_price + buffer)
        return min(current_sl, entry_price - buffer)
