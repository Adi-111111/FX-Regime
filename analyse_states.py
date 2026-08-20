def correlate_states_to_returns(df_main, states, forward_window=130):
    df_analysis = df_main.iloc[-len(states):].copy()
    df_analysis['state'] = states
    df_analysis['fwd_return'] = df_analysis['Close'].shift(-forward_window) / df_analysis['Close'] - 1

    stats = df_analysis.groupby('state')['fwd_return'].agg(['mean', 'std', 'count'])
    stats['sharpe_proxy'] = stats['mean'] / (stats['std'] + 1e-6)

    print(stats)
    return stats
