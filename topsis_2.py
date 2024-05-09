# 1. Program Structure and Imports
import pandas as pd
from sklearn.preprocessing import normalize, LabelEncoder
import warnings
import sys

def main():
    warnings.filterwarnings('ignore')
    try:
        input_file, weights_str, impacts_str, result_file = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    except IndexError:
        print("Usage: python <program.py> <InputDataFile> <Weights> <Impacts> <ResultFileName>")
        sys.exit(1)

    # 2. Extracting data from command line arguments
    try:
        weights = list(map(float, weights_str.split(',')))
        impacts = impacts_str.split(',')
    except ValueError:
        print("Invalid weights format. Weights must be numeric values separated by commas.")
        sys.exit(1)
    except:
        print("Invalid impacts format.")
        sys.exit(1)

    # 3. Reading input file (assuming CSV format)
    try:
        df = pd.read_csv(input_file)  # Use the first column as the index
    except FileNotFoundError:
        print(f"File not found: {input_file}")
        sys.exit(1)

    # 4. Validate input data
    if len(df.columns) < 3:
        print("Input file must contain three or more columns.")
        sys.exit(1)

    # 5. Convert Categorical Columns to Numerical
    categorical_columns = [col for col in df.columns[1:] if df[col].dtype == 'O']
    if categorical_columns:
        label_encoder = LabelEncoder()
        for col in categorical_columns:
            df[col] = label_encoder.fit_transform(df[col])

    # 6. Normalization
    df_copy = df.copy()
    for i, col in enumerate(df.columns[1:], start=1):  # Start from the 2nd column
        df_copy[col] = normalize(X=df_copy[[col]], norm='l2', axis=0) * weights[i - 1]  # Adjust index

    # 7. Finding Ideal Best and Ideal Worst
    ideal_best = [df_copy[col].max() if imp == '+' else df_copy[col].min() for col, imp in zip(df_copy.columns, impacts)]
    ideal_worst = [df_copy[col].min() if imp == '+' else df_copy[col].max() for col, imp in zip(df_copy.columns, impacts)]

    # 8. Calculating Euclidean Distance
    df_copy['splus'] = df_copy.apply(lambda row: sum((ideal - val) ** 2 for ideal, val in zip(ideal_best, row)) ** 0.5, axis=1)
    df_copy['sminus'] = df_copy.apply(lambda row: sum((ideal - val) ** 2 for ideal, val in zip(ideal_worst, row)) ** 0.5, axis=1)

    # 9. Calculate Performance Score
    df_copy['performance_score'] = df_copy['sminus'] / (df_copy['sminus'] + df_copy['splus'])

    # 10. Assign Rank
    df_copy['rank_highest'] = df_copy['performance_score'].rank(ascending=False)

    # 11. Appending Back to Original 'df'
    df['Topsis_Score'] = df_copy['performance_score']
    df['Rank'] = df_copy['rank_highest']
    
    # 12. Save the result DataFrame to a CSV file
    df.to_csv(result_file, index=False)

if __name__ == "__main__":
    main()
