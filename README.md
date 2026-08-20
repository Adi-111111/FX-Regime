# FX-Regime
EUR/USD systematic strategy using a VAE-HMM latent regime model

Pipeline
master_ingest.py — pulls EUR/USD OHLC plus four correlated instruments (GBPUSD, USDJPY, an equity index proxy, gold) from MetaTrader 5 across four timeframes, and pulls US/German 10-year yields and the dollar index from FRED and Yahoo Finance.
data_engine.py — builds the feature set from raw prices: RSI, ADX, three cross-asset "nexus" features (EUR relative strength vs GBP, a risk-appetite proxy from equities vs gold, EUR/USD-USD/JPY rolling correlation), an Asian-session range feature, and a level-2 path signature over a rolling 130-bar window via iisignature. Everything is standardised with a scaler that's fit once on training data only and reused at inference.
vae_policy.py — a VAE that compresses the feature vector into a latent space, plus a policy head that reads the latent code and outputs a trade direction and adaptive stop-loss/take-profit multipliers. The loss function trains the latent space directly against trading utility: reconstruction, KL divergence, a cost-aware Sharpe ratio term, and an alignment penalty against triple-barrier labels, combined in one objective.
pipeline_fx.py — orchestrates training: fits the feature scaler on data before a fixed cutoff date only, trains the VAE, fits a 5-state Gaussian HMM (regime_engine.py) on the resulting latent means to identify regimes, and trains a Random Forest meta-filter (meta_filter.py) to veto low-confidence setups based on latent features plus macro context.
analyse_states.py — measures each regime's forward-return mean and Sharpe proxy on the training set, to identify which discovered state is actually worth trading. Selection is on the largest absolute mean return, not the largest positive one — the regime filter's job is to flag conditions with a distinguishable-from-noise drift of either sign, since the actual long/short direction comes from the VAE policy head's p_weight, not from the regime label itself. It's normal for the selected state to have a historically negative mean.
risk_manager.py — ATR-based position sizing, adaptive stop-loss/take-profit levels, and session/time-of-day validity checks.
fx_backtester.py — runs the full simulated audit over a chosen date range: loads the trained VAE, HMM, and meta-filter, generates regime and direction signals bar by bar, and simulates execution with spread, commission, overnight swap, partial take-profits, and a trailing stop. Logs every trade and plots the equity curve.
Running it

python master_ingest.py
python pipeline_fx.py
python fx_backtester.py

pipeline_fx.py prints a recommended ALPHA_STATE at the end — the regime with the strongest forward-return signal on the training set. Update the ALPHA_STATE constant at the top of fx_backtester.py with that value before running the audit.
