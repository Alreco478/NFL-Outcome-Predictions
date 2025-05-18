import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import nfl_data_py as nfl
from transform import filter_to_plays, aggregate_df, add_calc_stats
from model import linreg_evaluate, rf_model_evaluate, logreg_model_evaluate
from sklearn.metrics import accuracy_score





def optimize_year(train_range, test_year, model, alpha = .05):
    """
    Optimizes a model by training it on NFL play-by-play data from a specified range of years, 
    and then evaluates its performance on a specific test year.

    Args:
        train_range (tuple): Tuple with the start and end year (inclusive) for training data (e.g., (2003, 2020)).
        test_year (int): The year to be used for testing the model.
        model (str): Model type to use ('linreg', 'ridge', 'lasso', 'rf', 'logreg').
        alpha (float): Regularization strength for Ridge and Lasso regression (default is 0.05).

    Raises:
        ValueError: If the input model is invalid or train_range is not a tuple.

    Returns:
        None
    """
    
    # Input validation
    if not isinstance(train_range, tuple) or len(train_range) != 2:
        raise ValueError("train_range must be a tuple with start and end year.")
    if model not in {'linreg', 'ridge', 'lasso', 'rf', 'logreg'}:
        raise ValueError("model must be one of {'linreg', 'ridge', 'lasso', 'rf', 'logreg'}")
    if not isinstance(test_year, int) or test_year < 1999 or test_year > 2025:
        raise ValueError("test_year must be an integer between 1999 and 2024.")


    # columns to pull from database
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
                'weather', 'play_type_nfl', 'special_teams_play', 'drive_first_downs', 
                'drive_inside20', 'drive_ended_with_score', 'away_score', 'home_score', 'location', 
                'result', 'total', 'spread_line', 'total_line', 'surface', 'temp', 'wind', 'pass', 
                'rush', 'first_down', 'special', 'play', 'qb_epa']

    # create df with combined data
    df = nfl.import_pbp_data(range(train_range[0], train_range[1] + 1), columns, downcast=True)

    # create dataset to test on
    df_test = nfl.import_pbp_data([test_year], columns, downcast=True)

    # create filtered_df_test
    filtered_df_test = filter_to_plays(df_test)

    # use one hot encoding on the listed columns
    columns_to_encode = ['field_goal_result',
                        'extra_point_result',
                        'two_point_conv_result']
    
    # create encoded_df_test
    encoded_df_test = pd.get_dummies(filtered_df_test, columns=columns_to_encode, drop_first=True)

    # columns to group by in aggregate_df
    group_by = ['game_id', 'posteam', 'home_team', 'away_team']

    # dict of columns to be included in aggregated_df and the corresponding functions to be applied
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

    # turn column_functions into a df
    df_column_functions = pd.DataFrame(list(column_functions.items()), columns=['Column', 'Function'])
    
    # create aggregated_df_test
    aggregated_df_test = aggregate_df(encoded_df_test, group_by, df_column_functions)

    # create transformed_df_test
    transformed_df_test = add_calc_stats(aggregated_df_test, encoded_df_test)
    
    # test year ranges
    for i in range(train_range[0], train_range[1] + 1):
        last_year = train_range[1]
        
        # remove one year from the training data
        filtered_year_df = df[df['season'] != i]
        filtered_df = filter_to_plays(filtered_year_df)

        # create filtered_df
        filtered_df = filter_to_plays(filtered_df)
        
        # create encoded_df
        encoded_df = pd.get_dummies(filtered_df, columns=columns_to_encode, drop_first=True)

        # create aggregated_df
        aggregated_df = aggregate_df(encoded_df, group_by, df_column_functions)

        # create transformed_df
        transformed_df = add_calc_stats(aggregated_df, encoded_df)

        # select features to be used in models
        features = ['yards_per_play_offense', 'first_down_rate_offense', 'yards_per_play_allowed', 
                    'first_down_rate_allowed', 'turnover_differential']
        # select target_variable
        target_variable = 'win'

        # create X_train and y_train
        X_train = transformed_df[features]
        y_train = transformed_df[target_variable]

        # create X_test and y_test
        X_test = transformed_df_test[features]
        y_test = transformed_df_test[target_variable]
            
        if model == 'linreg':
            # create and evaluate linear regression model
            linreg_model = LinearRegression()
            linreg_model.fit(X_train, y_train)
            print(f"{i} - {last_year}: ")
            linreg_evaluate(linreg_model, features_test = X_test, target_test = y_test)
        elif model == 'ridge':
            # create and evaluate ridge regression model
            ridge_model = Ridge(alpha=alpha)
            ridge_model.fit(X_train, y_train)
            print(f"{i} - {last_year}: ")
            linreg_evaluate(ridge_model, features_test = X_test, target_test = y_test)
        elif model == 'lasso':
            # create and evaluate lasso regression model
            lasso_model = Lasso(alpha=alpha)
            lasso_model.fit(X_train, y_train)
            print(f"{i} - {last_year}: ")
            linreg_evaluate(lasso_model, features_test = X_test, target_test = y_test)
        elif model == 'rf':
            # create and evaluate random forest model
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            print(f"{i} - {last_year}: ")
            y_pred = rf_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f'Accuracy: {accuracy:.4f}')
        elif model == 'logreg':
            # create and evaluate logistic regression model
            logreg_model = LogisticRegression()
            logreg_model.fit(X_train, y_train)
            print(f"{i} - {last_year}: ")
            y_pred = logreg_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f'Accuracy: {accuracy:.4f}')

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

    for i in np.arange(0.001, 0.5, 0.001):
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