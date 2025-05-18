import pandas as pd




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




def check_missing_values(df, display_rows=0):
    """
    Check for missing values in the dataset.

    This function calculates the total number of missing values in each column 
    of the given DataFrame and returns a DataFrame containing the column names, 
    total missing values, and the percentage of missing values for each column.

    Parameters:
        df (pd.DataFrame): The DataFrame to check for missing values.
        display_rows (int): The number of rows to show from the missing values DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with columns 'Column', 'NA_Count' indicating the missing values 
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
