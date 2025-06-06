#  NFL Outcome Predictions
###  Predicting NFL game outcomes using machine learning and statistical analysis

##  Overview
The primary goal of this project is to identify the most significant indicators of future success for an NFL team, then use them to make predictions for the winner of matchups. The approach is as follows:
1. Obtain NFL play by play data through the nfl_data_py package
2. Group data by team and game
3. Select features [yards per play offense, first down rate offense, yards per play allowed, first down rate allowed, turnover differential] and target variable [win]
4. Split dataset into train and test sets
5. Create logistic regression model and determine feature importances
6. Aggregate data from the current season to find values for each feature for each team
7. Scale current season data to standard units
8. Plug the current season data into the logistic regression model to determine the probability of a team winning any given game
9. Calculate the projected winner's probability of winning using the following formula:  
    Predicted winner’s probability = max(home winning probability, away winning probability) / (home winning probability + away winning probability)


##  Project Structure
├── models/ # Saved models  
├── src/ # Source code  
├── visualizations/ # Visualizations from various steps in the process  
├── main.ipynb # Notebook to run predictions   
└── README.md # This file  

##  Data
This project uses the nfl_data_py package

##  Model Performance
- The logistic regression model correctly guesses the winner of a game about 82% of the time

##  Results
- The most significant variables when determining the winner of an nfl game are the following: Turnover differential, First down percentage (both offense and defense), and Yards per play (both offense and defence)
- The actual ability of the model to predict nfl outcomes is questionable, however it makes very reasonable predictions based on who has been the better team in the current season.

##  Potential Improvements
- Factor in home team advantage
- Test more variables to determine their significance, like penalty yards
