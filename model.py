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





def logreg_model_evaluate(model, features_test, target_test, feature_importance=False):
    """
    Evaluates the performance of a Logistic Regression model and returns a DataFrame with actual, predicted labels,
    and the probability of winning.

    Args:
        model (LogisticRegression): The trained Logistic Regression model.
        features_test (pd.DataFrame): The features for the test set.
        target_test (pd.Series): The true labels for the test set.
        feature_importance (bool): Whether to display feature importance values and plot. Default is False.

    Raises:
        ValueError: If the model is not trained or if test data is invalid.

    Returns:
        pd.DataFrame: A DataFrame containing the actual labels, predicted labels, and probability of winning.
    """

    # Generate predictions and predicted probabilities
    y_pred = model.predict(features_test)
    y_prob = model.predict_proba(features_test)[:, 1]  # Probability of the positive class (team winning)

    # Calculate metrics
    accuracy = accuracy_score(target_test, y_pred)
    conf_matrix = confusion_matrix(target_test, y_pred)
    class_report = classification_report(target_test, y_pred)

    # Print evaluation metrics
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Confusion Matrix:\n{conf_matrix}')
    print(f'Classification Report:\n{class_report}')

    # Create a DataFrame with actual, predicted, and probability columns
    results_df = pd.DataFrame({
        'Actual': target_test,
        'Predicted': y_pred,
        'Probability': y_prob
    })

    if feature_importance:
        # Get feature names and coefficients
        feature_names = features_test.columns
        coefficients = model.coef_[0]

        # Create a DataFrame for coefficients
        coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

        # Calculate the absolute value of coefficients for importance
        coef_df['Importance'] = np.abs(coef_df['Coefficient'])
        coef_df = coef_df.sort_values(by='Importance', ascending=False)

        # Print feature importance values
        print("\nFeature Importance:")
        print(coef_df[['Feature', 'Coefficient', 'Importance']])

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.barh(coef_df['Feature'], coef_df['Importance'], color='skyblue')
        plt.xlabel('Importance')
        plt.title('Feature Importance from Logistic Regression')
        plt.show()

    return results_df
