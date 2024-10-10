import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import plot_tree
import joblib
import nfl_data_py as nfl



def summarize_df(df, display_rows = 0):
    """
    Summarizes a pandas DataFrame by displaying its structure and basic statistics.

    Args:
        df (pd.DataFrame): The DataFrame to summarize.
        display_rows (int): The number of rows to display from the top of the DataFrame. 
                            Defaults to 0 (no rows displayed).

    Raises:
        ValueError: If df is not a pandas DataFrame or if display_rows is not an integer.

    Returns:
        None
    """
    # Validate input types
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    
    if not isinstance(display_rows, int):
        raise ValueError("display_rows must be an int")
    
    # Display the rows if requested
    if display_rows > 0:
        print(f"\nFirst {display_rows} rows:")
        print(df.head(display_rows))

    # Get and print DataFrame dimensions
    num_rows, num_columns = df.shape
    print(f"Number of Rows: {num_rows}")
    print(f"Number of Columns: {num_columns}")

    # Count and print duplicate rows
    print("\nNumber of duplicate rows:", df.duplicated().sum())

    return 



def check_missing_values(df, display_rows = 0):
    """
    Check for missing values in the dataset.

    This function calculates the total number of missing values in each column 
    of the given DataFrame and returns a DataFrame containing the column names, 
    total missing values, and the percentage of missing values for each column.

    Parameters:
    df: (pd.DataFrame): The DataFrame to check for missing values.
    display_rows: (int): Number of rows to show of missing values dataframe

    Returns:
    pd.DataFrame: A DataFrame with columns 'Column', 'Total Missing', 
                  and 'Percentage Missing' indicating the missing values 
                  statistics for each column in the input DataFrame.
    """
    # input validation
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    
    if not isinstance(display_rows, int):
        raise ValueError("display_rows must be an int")

    # Check for missing values
    missing_values = df.isnull().sum().sort_values(ascending=False)
    missing_df = pd.DataFrame({'Column': missing_values.index, 'NA_Count': missing_values.values})
    
    if display_rows > 0:
        print(missing_df.head(display_rows))
    
    return missing_df



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




def linreg_evaluate(model, features_test, target_test):
    """
    Tests a linear regression model by calculating performance metrics 
    and displaying the coefficients for each feature.
    
    Args:
        model (LinearRegression): A trained LinearRegression model.
        X_test (pd.DataFrame): The test features used for prediction.
        y_test (pd.Series): The actual target values for the test set.
        
    Raises:
        ValueError: If the model has not been trained or predictions cannot be made.
    
    Returns:
        None: This function prints the summary and does not return any value.
    """
    y_pred = model.predict(features_test)

    mse = mean_squared_error(target_test, y_pred)
    r2 = r2_score(target_test, y_pred)
    
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'R² Score: {r2:.2f}')

    # Get coefficients and feature names
    coefficients = model.coef_
    feature_names = features_test.columns

    # Create a DataFrame to summarize coefficients
    coef_table = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

    # Sort by absolute coefficient value
    coef_table['Abs Coefficient'] = coef_table['Coefficient'].abs()
    coef_table = coef_table.sort_values(by='Abs Coefficient', ascending=False).drop(columns='Abs Coefficient')

    # Display the coefficients table
    print(coef_table)

    return




def optimize_alpha(mode='', features_train=None, target_train=None, features_test=None, target_test=None):
    """
    Shows info to optimize the alpha parameter for Lasso and Ridge regression models and plots the MSE and R² scores.
    
    Args:
        mode (str): The type of regression model to optimize ('ridge', 'lasso', or 'both').
        features_train (pd.DataFrame): The training features.
        target_train (pd.Series): The training target values.
        features_test (pd.DataFrame): The test features.
        target_test (pd.Series): The test target values.
        
    Raises:
        ValueError: If the mode is not recognized.
    
    Returns:
        None: This function plots the results and does not return any value.
    """
    
    # Validate mode
    if mode not in ['ridge', 'lasso', 'both']:
        raise ValueError("Mode must be one of 'ridge', 'lasso', or 'both'")
    
    df_list = [features_train, features_test]
    for i in df_list:
        if not isinstance(i, pd.DataFrame):
            raise ValueError("features_train and features_test must be a pandas DataFrame")
        
    series_list = [target_train, target_test]
    for i in series_list:
        if not isinstance(i, pd.Series):
            raise ValueError("features_train and features_test must be a pandas DataFrame")

    df_lasso = pd.DataFrame(columns=['mse', 'r^2'])
    df_ridge = pd.DataFrame(columns=['mse', 'r^2'])

    for i in np.arange(0.01, 0.5, 0.01):
        if mode in ['lasso', 'both']:
            lasso_model = Lasso(alpha=i)
            lasso_model.fit(features_train, target_train)
            y_pred = lasso_model.predict(features_test)
            df_lasso.loc[i*100] = [mean_squared_error(target_test, y_pred), r2_score(target_test, y_pred)]

        if mode in ['ridge', 'both']:
            ridge_model = Ridge(alpha=i)
            ridge_model.fit(features_train, target_train)
            y_pred = ridge_model.predict(features_test)
            df_ridge.loc[i*100] = [mean_squared_error(target_test, y_pred), r2_score(target_test, y_pred)]

    plt.figure(figsize=(10, 5))

    # Plot for Lasso
    if mode in ['lasso', 'both']:
        plt.plot(df_lasso.index / 100, df_lasso['mse'], label='Lasso MSE', color='blue', linestyle='--')
        plt.plot(df_lasso.index / 100, df_lasso['r^2'], label='Lasso R²', color='blue')

    # Plot for Ridge
    if mode in ['ridge', 'both']:
        plt.plot(df_ridge.index / 100, df_ridge['mse'], label='Ridge MSE', color='red', linestyle='--')
        plt.plot(df_ridge.index / 100, df_ridge['r^2'], label='Ridge R²', color='red')

    plt.xlabel('Alpha')
    plt.ylabel('Score')
    plt.title('Lasso and Ridge Regression Metrics vs Alpha')
    plt.legend()
    plt.grid(True)
    plt.show()

    return



def rf_model_evaluate(model, features_test, target_test, tree_plot=False, feature_importance=False):
    """
    Evaluates the performance of a Random Forest model.

    Args:
        model (RandomForestClassifier): The trained Random Forest model.
        features_test (pd.DataFrame): The features for the test set.
        target_test (pd.Series): The true labels for the test set.
        tree_plot (bool): Whether to plot a tree from the Random Forest.

    Raises:
        ValueError: If the model is not trained or if test data is invalid.
    
    Returns:
        None
    """
    # Generate predictions
    y_pred = model.predict(features_test)

    # Calculate metrics
    accuracy = accuracy_score(target_test, y_pred)
    conf_matrix = confusion_matrix(target_test, y_pred)
    class_report = classification_report(target_test, y_pred)

    # Print evaluation metrics
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Confusion Matrix:\n{conf_matrix}')
    print(f'Classification Report:\n{class_report}')

    # Plotting a decision tree from the random forest if requested
    if tree_plot:
        # Extract one tree from the random forest
        single_tree = model.estimators_[0]

        # Visualize the decision tree
        plt.figure(figsize=(20, 10))  # Adjust figure size to ensure the tree is readable
        plot_tree(single_tree, feature_names=features_test.columns, filled=True, rounded=True, class_names=['0', '1'])
        plt.title('Decision Tree from Random Forest')
        plt.show()

    if feature_importance:
        # Feature importance
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Plot the feature importances
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(features_test.shape[1]), importances[indices], align='center')
        plt.xticks(range(features_test.shape[1]), features_test.columns[indices], rotation=90)
        plt.xlim([-1, features_test.shape[1]])
        plt.tight_layout()
        plt.show()

    return
 


years = list(range(2003, 2024))
columns =   ['play_id', 'game_id', 'home_team', 'away_team', 'season_type', 'week', 'posteam', 'posteam_type',
             'side_of_field', 'yardline_100', 'game_date', 'game_seconds_remaining', 'down', 'ydsnet', 'desc',
             'play_type', 'yards_gained', 'pass_length', 'yards_after_catch', 'field_goal_result', 
             'kick_distance', 'extra_point_result', 'two_point_conv_result', 'total_home_score', 
             'total_away_score', 'ep', 'epa', 'total_home_epa', 'total_away_epa', 'total_home_rush_epa', 
             'total_away_rush_epa', 'total_home_pass_epa', 'total_away_pass_epa', 'wp', 'def_wp', 'home_wp', 
             'away_wp', 'wpa', 'total_home_rush_wpa', 'total_away_rush_wpa', 'total_home_pass_wpa', 
             'total_away_pass_wpa', 'punt_blocked', 'first_down_rush', 'first_down_pass', 
             'first_down_penalty', 'third_down_converted', 'third_down_failed', 'fourth_down_converted', 
             'fourth_down_failed', 'incomplete_pass', 'touchback', 'interception', 'fumble_forced', 
             'fumble_not_forced', 'fumble_out_of_bounds', 'safety', 'penalty', 'fumble_lost', 'rush_attempt', 
             'pass_attempt', 'sack', 'touchdown', 'pass_touchdown', 'rush_touchdown', 'return_touchdown', 
             'extra_point_attempt', 'two_point_attempt', 'field_goal_attempt', 'kickoff_attempt', 
             'punt_attempt', 'fumble', 'complete_pass', 'passing_yards', 'receiving_yards', 'rushing_yards', 
             'return_yards', 'penalty_team', 'penalty_yards', 'penalty_type', 'season', 'series_result', 
             'weather', 'play_type_nfl', 'special_teams_play', 'st_play_type', 'drive_first_downs', 
             'drive_inside20', 'drive_ended_with_score', 'away_score', 'home_score', 'location', 
             'result', 'total', 'spread_line', 'total_line', 'surface', 'temp', 'wind', 'pass', 
             'rush', 'first_down', 'special', 'play', 'qb_epa']
df = nfl.import_pbp_data(years, columns, downcast=True)

summarize_df(df, display_rows=10)

check_missing_values(df, display_rows=500)

filtered_df = filter_to_plays(df)

trimmed_df = filtered_df.drop(columns=['st_play_type'])

encoded_df = pd.get_dummies(df, columns=['field_goal_result',
                                         'extra_point_result',
                                         'two_point_conv_result'], 
                                         drop_first=True)

group_by = ['game_id', 'posteam', 'home_team', 'away_team']

column_functions = {'yards_gained': 'sum',
                    'receiving_yards': 'sum',
                    'rushing_yards': 'sum',
                    'return_yards': 'sum',
                    'incomplete_pass': 'sum',
                    'complete_pass': 'sum',
                    'interception': 'sum',
                    'pass_attempt': 'sum',
                    'rush_attempt': 'sum',
                    'touchdown': 'sum',
                    'pass_touchdown': 'sum',
                    'rush_touchdown': 'sum',
                    'return_touchdown': 'sum',
                    'first_down': 'sum',
                    'play': 'sum',
                    'sack': 'sum',
                    'fumble': 'sum',
                    'fumble_lost': 'sum',
                    'field_goal_result_made': 'sum',
                    'field_goal_result_missed': 'sum',
                    'extra_point_result_good': 'sum',
                    'extra_point_result_failed': 'sum',
                    'safety': 'sum',
                    'total_home_epa': 'max',
                    'total_away_epa': 'max',
                    'total_home_score': 'max',
                    'total_away_score': 'max',
                    'total': 'max'}

df_column_functions = pd.DataFrame(list(column_functions.items()), columns=['Column', 'Function'])

aggregated_df = aggregate_df(encoded_df, group_by, df_column_functions)

transformed_df = add_calc_stats(aggregated_df, encoded_df)

features = ['yards_per_play_offense', 'points_per_play_offense', 'first_down_rate_offense', 
            'turnovers_lost', 'yards_per_play_allowed', 'points_per_play_allowed', 
            'first_down_rate_allowed', 'turnovers_gained', 'turnover_differential']
target_variable = 'win'

X = transformed_df[features]
y = transformed_df[target_variable]
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

linreg_model = LinearRegression()
linreg_model.fit(X_train, y_train)
linreg_evaluate(linreg_model, features_test = X_test, target_test = y_test)

#optimize_alpha(mode='both', features_train=X_train, target_train=y_train, features_test=X_test, target_test=y_test)

ridge_model = Ridge(alpha=.25)
ridge_model.fit(X_train, y_train)
linreg_evaluate(ridge_model, features_test = X_test, target_test = y_test)

lasso_model = Lasso(alpha=.25)
lasso_model.fit(X_train, y_train)
linreg_evaluate(lasso_model, features_test = X_test, target_test = y_test)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_model_evaluate(rf_model, features_test=X_test, target_test=y_test, tree_plot=False, feature_importance=False)

i = input("Would you like to save the models to a folder? (y/n) (ex = 'C:\\Users\\alrec\\Desktop\\DATCAP Repo\\models'): ")
if i in {'Y','y', 'ex'}:
    if i == 'ex':
        filepath = r'C:\Users\alrec\Desktop\DATCAP Repo\models'
    else:
        filepath = input("Please enter the filepath for the folder to save to: ")
    joblib.dump(linreg_model, f'{filepath}\linreg_model.pkl')
    joblib.dump(ridge_model, f'{filepath}/ridge_model.pkl')
    joblib.dump(lasso_model, f'{filepath}\lasso_model.pkl')
    joblib.dump(rf_model, f'{filepath}/rf_model.pkl')
    print('models saved')
