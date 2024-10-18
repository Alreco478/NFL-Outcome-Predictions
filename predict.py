import pandas as pd
import numpy as np
import os





def predict_winner(model, df, home_team, away_team):
    """
    Predicts the winner of an NFL game between two teams using a Logistic Regression model.

    Args:
        model (LogisticRegression): The model to use for the predictions.
        df (pd.DataFrame): The DataFrame containing scaled features for each team with a 'posteam' column.
        home_team (str): The name of the home team.
        away_team (str): The name of the away team.

    Raises:
        ValueError: If the input types are invalid or if the model is not trained.

    Returns:
        str: The predicted winner of the game and the probability of winning.
    """
    
    # Validate input types
    if not isinstance(home_team, str) or not isinstance(away_team, str):
        raise ValueError("Both home_team and away_team must be strings.")

    if not hasattr(model, 'predict_proba'):
        raise ValueError("The provided model is not a trained Logistic Regression model.")
    
    # Get the features for the home and away team from scaled_df
    home_row = df[df['posteam'] == home_team]
    away_row = df[df['posteam'] == away_team]
    
    if home_row.empty:
        raise ValueError(f"home team ({home_row}) must exist in the scaled features DataFrame.")
    
    if away_row.empty:
        raise ValueError(f"away team ({away_row}) must exist in the scaled features DataFrame.")

    # Combine the features into a single DataFrame for prediction
    features = pd.DataFrame([home_row.values[0][:-1], away_row.values[0][:-1]], 
                            index=['home', 'away'], 
                            columns=df.columns[:-1])
    
    # Make predictions
    probabilities = model.predict_proba(features)

    # Get the predicted winner based on probabilities
    predicted_winner = home_team if probabilities[0][1] > probabilities[1][1] else away_team
    winning_probability = max(probabilities[0][1], probabilities[1][1]) * 100  # Convert to percentage

    return predicted_winner, winning_probability





def predict_week(logreg_model, scaled_data, games):
    """
    Predicts the winner for all NFL games in the given week's schedule.

    Args:
    - logreg_model: Trained logistic regression model for prediction.
    - scaled_data: Scaled dataframe containing team stats for prediction.
    - games: List of tuples where each tuple contains ('Home', 'Away') teams.

    Returns:
    - predictions_df: DataFrame with columns ['Home', 'Away', 'Predicted Winner', 'Winning Probability']
    """
    # Create an empty list to store the predictions
    predictions = []

    # Iterate over each tuple in the games list
    for home_team, away_team in games:
        # Use the provided predict_winner function to make a prediction
        predicted_winner, winning_probability = predict_winner(logreg_model, scaled_data, home_team, away_team)

        # Append the result to the predictions list
        predictions.append({
            'Home': home_team,
            'Away': away_team,
            'Predicted Winner': predicted_winner,
            'Winning Probability (%)': round(winning_probability * 100, 2)
        })

    # Convert the list of predictions to a DataFrame
    predictions_df = pd.DataFrame(predictions)

    return predictions_df
