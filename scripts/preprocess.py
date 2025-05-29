import pandas as pd

def clean_ufc_data(csv_path):
    import pandas as pd

    # Load the data
    df = pd.read_csv(csv_path)

    # Drop irrelevant metadata and rarely filled columns
    columns_to_drop = [
        'Date', 'Location', 'Country', 'Finish', 'FinishRound', 'FinishRoundTime',
        'TotalFightTimeSecs', 'TitleBout', 'Gender', 'NumberOfRounds', 'BetterRank', 'EmptyArena',
        'RedDecOdds', 'BlueDecOdds', 'RSubOdds', 'BSubOdds', 'RKOOdds', 'BKOOdds',

        # Fighter-specific rankings — usually sparse
        'RWFlyweightRank', 'RWFeatherweightRank', 'RWStrawweightRank', 'RWBantamweightRank',
        'RHeavyweightRank', 'RLightHeavyweightRank', 'RMiddleweightRank', 'RWelterweightRank',
        'RLightweightRank', 'RFeatherweightRank', 'RBantamweightRank', 'RFlyweightRank', 'RPFPRank',
        'BWFlyweightRank', 'BWFeatherweightRank', 'BWStrawweightRank', 'BWBantamweightRank',
        'BHeavyweightRank', 'BLightHeavyweightRank', 'BMiddleweightRank', 'BWelterweightRank',
        'BLightweightRank', 'BFeatherweightRank', 'BBantamweightRank', 'BFlyweightRank', 'BPFPRank'
    ]
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # Keep only rows where the winner is clearly Red or Blue
    df = df[df['Winner'].isin(['Red', 'Blue'])]

    # Convert winner to binary: Red = 1, Blue = 0
    df['Winner'] = df['Winner'].map({'Red': 1, 'Blue': 0})

    # Drop columns that are completely empty
    df.dropna(axis=1, how='all', inplace=True)

    # Fill remaining missing values with 0
    df.fillna(0, inplace=True)

    return df


def build_fighter_profiles(df):
    # --- RED CORNER ---
    red_df = df.copy()
    red_df['Fighter'] = red_df['RedFighter']
    red_df['Win'] = df['Winner']
    red_stats = [col for col in df.columns if col.startswith('Red') and df[col].dtype != 'O' and col != 'RedFighter']
    red_df = red_df[['Fighter', 'Win'] + red_stats]
    red_df = red_df.rename(columns={col: col.replace('Red', '') for col in red_stats})

    # --- BLUE CORNER ---
    blue_df = df.copy()
    blue_df['Fighter'] = blue_df['BlueFighter']
    blue_df['Win'] = 1 - df['Winner']
    blue_stats = [col for col in df.columns if col.startswith('Blue') and df[col].dtype != 'O' and col != 'BlueFighter']
    blue_df = blue_df[['Fighter', 'Win'] + blue_stats]
    blue_df = blue_df.rename(columns={col: col.replace('Blue', '') for col in blue_stats})

    # --- COMBINE ---
    all_fights = pd.concat([red_df, blue_df])
    all_fights['FightOrder'] = all_fights.groupby('Fighter').cumcount()

    # --- WIN RATE ---
    winrate = all_fights.groupby('Fighter')['Win'].mean().reset_index()
    winrate = winrate.rename(columns={'Win': 'WinRate'})

    # --- MOST RECENT STREAK VALUES ---
    streak_cols = ['CurrentWinStreak', 'CurrentLoseStreak']
    has_streaks = all(col in all_fights.columns for col in streak_cols)

    if has_streaks:
        latest_streaks = (
            all_fights
            .sort_values('FightOrder')
            .drop_duplicates('Fighter', keep='last')[['Fighter'] + streak_cols]
        )
    else:
        latest_streaks = pd.DataFrame(columns=['Fighter'] + streak_cols)

    # --- AVERAGE STATS (exclude streaks and target columns) ---
    cols_to_exclude = ['Win', 'FightOrder'] + streak_cols
    cols_to_drop = [col for col in cols_to_exclude if col in all_fights.columns]
    stats_only = all_fights.drop(columns=cols_to_drop)
    avg_stats = stats_only.groupby('Fighter').mean().reset_index()

    # --- MERGE FINAL PROFILE ---
    fighter_profiles = avg_stats.merge(winrate, on='Fighter', how='left')
    fighter_profiles = fighter_profiles.merge(latest_streaks, on='Fighter', how='left')

    return fighter_profiles





