import pandas as pd

def clean_ufc_data(csv_path):
    # Load the raw CSV file
    df = pd.read_csv(csv_path)

    # Drop irrelevant metadata (not useful for modeling)
    columns_to_drop = [
        'Date', 'Location', 'Country', 'TitleBout',
        'NumberOfRounds', 'FinishDetails', 'TotalFightTimeSecs', 'BetterRank', 'Gender', 'NumberOfRounds'
    ]
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # Keep only rows where the winner is clearly Red or Blue
    df = df[df['Winner'].isin(['Red', 'Blue'])]

    # Convert 'Winner' column to binary: Red = 1, Blue = 0
    df['Winner'] = df['Winner'].map({'Red': 1, 'Blue': 0})

    # Drop columns that are completely empty
    df.dropna(axis=1, how='all', inplace=True)

    # Fill all other missing values with 0
    df.fillna(0, inplace=True)

    return df


def build_fighter_profiles(df):
    # ----- RED CORNER PROCESSING -----

    red_df = df.copy()
    red_df['Fighter'] = red_df['RedFighter']
    red_df['Win'] = df['Winner']
    red_stats = [col for col in df.columns if col.startswith('Red') and df[col].dtype != 'O' and col != 'RedFighter']
    red_df = red_df[['Fighter', 'Win'] + red_stats]
    red_df = red_df.rename(columns={col: col.replace('Red', '') for col in red_stats})


    # ----- BLUE CORNER PROCESSING -----

    blue_df = df.copy()
    blue_df['Fighter'] = blue_df['BlueFighter']
    blue_df['Win'] = 1 - df['Winner']
    blue_stats = [col for col in df.columns if col.startswith('Blue') and df[col].dtype != 'O' and col != 'BlueFighter']
    blue_df = blue_df[['Fighter', 'Win'] + blue_stats]
    blue_df = blue_df.rename(columns={col: col.replace('Blue', '') for col in blue_stats})


    # ----- COMBINE AND STREAK HANDLING -----

    all_fights = pd.concat([red_df, blue_df])
    all_fights['FightOrder'] = all_fights.groupby('Fighter').cumcount()

    if 'WinStreak' in all_fights.columns and 'LoseStreak' in all_fights.columns:
        latest_streaks = (
            all_fights
            .sort_values('FightOrder')
            .drop_duplicates('Fighter', keep='last')[['Fighter', 'WinStreak', 'LoseStreak']]
        )

        mean_stats = all_fights.groupby('Fighter').mean().reset_index()
        mean_stats = mean_stats.drop(columns=['WinStreak', 'LoseStreak'], errors='ignore')
        fighter_profiles = mean_stats.merge(latest_streaks, on='Fighter', how='left')
    else:
        # If streaks are not in dataset, just return mean stats
        fighter_profiles = all_fights.groupby('Fighter').mean().reset_index()

    return fighter_profiles


