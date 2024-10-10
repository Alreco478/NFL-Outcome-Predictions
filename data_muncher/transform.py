import pandas as pd
import numpy as np




def filter_to_plays(df):
    """
    Filters the DataFrame to retain only entries that represent actual plays.

    Args:
        df (pd.DataFrame): The DataFrame containing play-by-play data.

    Raises:
        ValueError: If df is not a pandas DataFrame or if the required column 'desc' is not present.

    Returns:
        pd.DataFrame: A DataFrame containing only play entries.
    """
    # Validate input types
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    if 'desc' not in df.columns:
        raise ValueError("DataFrame must contain a 'desc' column")

    # Filter out entries that aren't plays
    filter_values = ["GAME", "END QUARTER 1", "END QUARTER 2", "END QUARTER 3", "END GAME"]
    df_filtered = df[~df['desc'].isin(filter_values)]
    df_filtered = df_filtered[~df_filtered['desc'].str.contains(r'Timeout', na=False)]

    return df_filtered




def aggregate_df(df, group_by_list, column_func_df):
    """
    Aggregates team statistics for each game in the DataFrame.

    Args:
        df (pd.DataFrame): A DataFrame containing game data with team statistics.
        group_by_list (list): A list of columns to group by.
        column_func_df (pd.DataFrame): A DataFrame defining the aggregation functions for specific columns.

    Raises:
        ValueError: If df is not a pandas DataFrame, if required columns are missing, or if inputs are of incorrect type.

    Returns:
        pd.DataFrame: A DataFrame with aggregated statistics for each game and team.
    """
    # Validate input types
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    if not isinstance(group_by_list, list):
        raise ValueError("group_by_list must be a list")

    if not isinstance(column_func_df, pd.DataFrame):
        raise ValueError("column_func_df must be a pandas DataFrame")

    required_columns = group_by_list + column_func_df['Column'].tolist()

    if not all(col in df.columns for col in required_columns):
        raise ValueError("DataFrame is missing one or more required columns")

    aggregated = df.groupby(group_by_list).agg(column_func_df.set_index('Column')['Function']).reset_index()

    return aggregated




def add_calc_stats(df, pre_ag_df):
    """
    Adds calculated statistics to a DataFrame containing game data.

    Args:
        df (pd.DataFrame): DataFrame containing aggregated game statistics for offensive plays.
        pre_ag_df (pd.DataFrame): DataFrame with play-by-play data to derive defensive statistics.

    Raises:
        ValueError: If inputs are not pandas DataFrames or if required columns are missing.

    Returns:
        pd.DataFrame: The original DataFrame with additional calculated statistics, including:
            - win: Indicator of whether the team won.
            - season: The season extracted from the game_id.
            - various calculated offensive and defensive statistics.
    """

    # Validate input types
    if not isinstance(df, pd.DataFrame) or not isinstance(pre_ag_df, pd.DataFrame):
        raise ValueError("Inputs must be a pandas DataFrame")

    aggregated = df

    # Extract the season from the game_id (first 4 characters)
    aggregated['season'] = aggregated['game_id'].str[:4]

    # Add a column 'win' to indicate whether the posteam won or lost the game
    aggregated['win'] = aggregated.apply(
        lambda row: 1 if (
            (row['posteam'] == row['home_team'] and row['total_home_score'] > row['total_away_score']) or 
            (row['posteam'] == row['away_team'] and row['total_away_score'] > row['total_home_score'])
        ) else 0, axis=1)

    # Calculate the total wins and games played by each team per season
    season_stats = aggregated.groupby(['season', 'posteam']).agg(
        total_wins=('win', 'sum'),
        total_games=('game_id', 'count')  # count the total games using game_id
    ).reset_index()

    # Calculate the win percentage for each team per season
    season_stats['win_percentage_season'] = season_stats['total_wins'] / season_stats['total_games']

    # Check for potential division by zero
    season_stats['win_percentage_season'] = season_stats['win_percentage_season'].fillna(0)  # Replace NaNs with 0

    # Merge the season stats back into the aggregated DataFrame (if needed)
    aggregated = aggregated.merge(season_stats[['season', 'posteam', 'win_percentage_season']], 
                                on=['season', 'posteam'], how='left')

    # Make a copy of the dataframe but swap the teams to calculate defensive stats
    df_defense = pre_ag_df.copy()

    # Swap posteam with the opposing team to get the defensive stats for the right team
    df_defense['posteam'] = df_defense.apply(
        lambda row: row['home_team'] if row['posteam'] == row['away_team'] else row['away_team'], axis=1)

    # Aggregate the data by game and posteam (now representing the defensive side)
    aggregated_defense = df_defense.groupby(['game_id', 'posteam']).agg({
        'yards_gained': 'sum',
        'first_down': 'sum',
        'touchdown': 'sum',
        'play': 'sum',
        'interception': 'sum',
        'fumble_forced': 'sum',
        'fumble_lost': 'sum',
        'total_home_score': 'max',
        'total_away_score': 'max',
    }).reset_index()

    # Rename columns to make it clear these are defensive stats
    aggregated_defense.rename(columns={
        'yards_gained': 'yards_allowed',
        'first_down': 'first_down_allowed',
        'touchdown': 'touchdowns_allowed',
        'interception': 'turnovers_gained',
        'fumble_forced': 'fumbles_forced',
        'fumble_lost': 'fumbles_recovered',
    }, inplace=True)

    # Now merge this defensive aggregated dataframe back with the original offensive one
    aggregated = aggregated.merge(aggregated_defense, on=['game_id', 'posteam'], suffixes=('', '_defense'))

    # Add offensive calculated stats: yards per play, points per play, first down rate, and number of turnovers (lost)
    aggregated['yards_per_play_offense'] = aggregated['yards_gained'] / aggregated['play']
    aggregated['points_per_play_offense'] = (aggregated['total_home_score'] + aggregated['total_away_score']) / aggregated['play']
    aggregated['first_down_rate_offense'] = aggregated['first_down'] / aggregated['play']
    aggregated['turnovers_lost'] = aggregated['interception'] + aggregated['fumble_lost']

    # Add defensive calculated stats: yards per play allowed, points per play allowed, first down rate allowed, and turnovers gained
    aggregated['yards_per_play_allowed'] = aggregated['yards_allowed'] / aggregated['play_defense']
    aggregated['points_per_play_allowed'] = (aggregated['total_home_score_defense'] + aggregated['total_away_score_defense']) / aggregated['play_defense']
    aggregated['first_down_rate_allowed'] = aggregated['first_down_allowed'] / aggregated['play_defense']

    # Turnovers gained by defense (interceptions + fumbles recovered)
    aggregated['turnovers_gained'] = aggregated['turnovers_gained'] + aggregated['fumbles_recovered']

    # add special teams calculated stats: fg percentage, xp percentage
    aggregated['fg_percentage'] = aggregated['field_goal_result_made'] / (
            aggregated['field_goal_result_made'] + aggregated['field_goal_result_missed'])
    aggregated['xp_percentage'] = aggregated['extra_point_result_good'] / (
            aggregated['extra_point_result_good'] + aggregated['extra_point_result_failed'])

    # Add turnover differential
    aggregated['turnover_differential'] = aggregated['turnovers_gained'] - aggregated['turnovers_lost']

    return aggregated
