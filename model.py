import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import Lasso, Ridge
from sklearn.tree import plot_tree




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
    rmse = np.sqrt(mse)
    r2 = r2_score(target_test, y_pred)
    
    print(f'Mean Squared Error: {mse:.2f}')
    print(f"Root Mean Squared Error: {rmse:.2f}")
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




def logreg_model_evaluate(model, features_test, target_test):
    """
    Evaluates the performance of a Logistic Regression model.

    Args:
        model (LogisticRegression): The trained Logistic Regression model.
        features_test (pd.DataFrame): The features for the test set.
        target_test (pd.Series): The true labels for the test set.

    Raises:
        ValueError: If the model is not trained or if test data is invalid.
    
    Returns:
        pd.DataFrame: A DataFrame containing the actual labels and the predicted labels.
    """
    
    # Generate predictions
    y_pred = model.predict(features_test)

    # Calculate metrics
    accuracy = accuracy_score(target_test, y_pred)
    conf_matrix = confusion_matrix(target_test, y_pred)
    class_report = classification_report(target_test, y_pred)

    results_df = pd.DataFrame({'Actual': target_test, 'Predicted': y_pred})
    return results_df

    
