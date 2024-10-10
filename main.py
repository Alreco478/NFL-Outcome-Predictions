import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import nfl_data_py as nfl
from explore import summarize_df, check_missing_values
from transform import filter_to_plays, aggregate_df, add_calc_stats
from model import linreg_evaluate, optimize_alpha, rf_model_evaluate, logreg_model_evaluate





# range of years to pull from
years = list(range(2003,2023))

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
df = nfl.import_pbp_data(years, columns, downcast=True)

# display first n rows, number of rows, number of columns, number of duplicates
summarize_df(df, display_rows=10)

# display how many nans are in each column (check_missing_values returns df of missing values)
check_missing_values(df, display_rows=500)

# filter df to only include plays (gets rid of timeouts, penalties, start of game, etc.)
filtered_df = filter_to_plays(df)

# use one hot encoding on the listed columns
columns_to_encode = ['field_goal_result',
                     'extra_point_result',
                     'two_point_conv_result']
encoded_df = pd.get_dummies(filtered_df, columns=columns_to_encode, drop_first=True)

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

# create aggregated_df
aggregated_df = aggregate_df(encoded_df, group_by, df_column_functions)

# add calculated stats columns (win, season, yards_per_play_offense, points_per_play_offense, 
#                               first_down_rate_offense, turnovers_lost, yards_per_play_allowed, 
#                               points_per_play_allowed, first_down_rate_allowed, turnovers_gained,
#                               turnover_differential, fg percentage, xp percentage)
transformed_df = add_calc_stats(aggregated_df, encoded_df)

# select features to be used in models
features = ['yards_per_play_offense', 'points_per_play_offense', 'first_down_rate_offense', 
            'turnovers_lost', 'yards_per_play_allowed', 'points_per_play_allowed', 
            'first_down_rate_allowed', 'turnovers_gained', 'turnover_differential']
# select target_variable
target_variable = 'win'

# create X (df of feature columns) and y (series for the target variable)
X = transformed_df[features]
y = transformed_df[target_variable]
    
# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create and evaluate linear regression model
linreg_model = LinearRegression()
linreg_model.fit(X_train, y_train)
linreg_evaluate(linreg_model, features_test = X_test, target_test = y_test)

# plot the relationship between alpha and r^2/mse for ridge and lasso models to determine alpha
#optimize_alpha(mode='both', features_train=X_train, target_train=y_train, features_test=X_test, target_test=y_test)

# create and evaluate ridge regression model
ridge_model = Ridge(alpha=.05)
ridge_model.fit(X_train, y_train)
linreg_evaluate(ridge_model, features_test = X_test, target_test = y_test)

# create and evaluate lasso regression model
lasso_model = Lasso(alpha=.01)
lasso_model.fit(X_train, y_train)
linreg_evaluate(lasso_model, features_test = X_test, target_test = y_test)

# create and evaluate random forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_model_evaluate(rf_model, features_test=X_test, target_test=y_test, tree_plot=False, feature_importance=False)

# create and evaluate logistic regression model
logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)
print(logreg_model_evaluate(logreg_model, X_test, y_test).head(10))

# download any models or datasets
print("Would you like to save any files from the project? (y/n)")
i = input("")
if i in {'Y','y', 'ex'}:
    print("Would you like to save the models to a folder? (y/n)")
    print("ex = 'C:\\Users\\alrec\\Desktop\\DATCAP Repo\\models'")
    i = input("")
    if i in {'Y','y', 'ex'}:
        if i == 'ex':
            filepath = r'C:\Users\alrec\Desktop\DATCAP Repo\models'
        else:
            filepath = input("Please enter the filepath for the folder to save to: ")
        os.makedirs(filepath, exist_ok=True)
        joblib.dump(linreg_model, f'{filepath}\linreg_model.pkl')
        joblib.dump(ridge_model, f'{filepath}/ridge_model.pkl')
        joblib.dump(lasso_model, f'{filepath}\lasso_model.pkl')
        joblib.dump(rf_model, f'{filepath}/rf_model.pkl')
        print('models saved')
    
    print("Would you like to save the data to a folder? (y/n)")
    i = input("").strip()

    if i in {'Y', 'y', 'ex'}:
        if i == 'ex':
            filepath = r'C:/Users/alrec/Desktop/DATCAP Repo/data'
        else:
            filepath = input("Please enter the filepath for the folder to save to: ").strip()
        
        os.makedirs(filepath, exist_ok=True)
        
        df.to_csv(os.path.join(filepath, 'df.csv'), index=False)
        filtered_df.to_csv(os.path.join(filepath, 'filtered_df.csv'), index=False)
        aggregated_df.to_csv(os.path.join(filepath, 'aggregated_df.csv'), index=False)
        transformed_df.to_csv(os.path.join(filepath, 'transformed_df.csv'), index=False)

        print('Data saved successfully')
