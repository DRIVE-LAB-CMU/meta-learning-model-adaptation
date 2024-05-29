import pandas as pd

def process_csv(input_file: str, output_file: str, factor: float):
    """
    Reads a CSV file, divides the first two columns by a given factor,
    and saves only the first two columns to a new CSV file.
    
    :param input_file: Path to the input CSV file
    :param output_file: Path to the output CSV file
    :param factor: The factor by which to divide the first two columns
    """
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Check if the CSV has at least two columns
    if df.shape[1] < 2:
        raise ValueError("The input CSV must have at least two columns")
    
    # Divide the first two columns by the given factor
    df.iloc[:, 0] = df.iloc[:, 0] / factor
    df.iloc[:, 1] = df.iloc[:, 1] / factor
    
    # Select only the first two columns
    df_first_two_columns = df.iloc[:, :2]
    
    # Save the modified data to a new CSV file
    df_first_two_columns.to_csv(output_file, index=False)

# Example usage
input_file = 'rounded_rectangle.csv'  # Path to the input CSV file
output_file = 'rounded_rectangle.csv'  # Path to the output CSV file
factor = 10.0  # Factor by which to divide the first two columns

process_csv(input_file, output_file, factor)
