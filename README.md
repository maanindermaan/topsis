# TOPSIS Analysis Package

This Python package implements the Technique for Order of Preference by Similarity to Ideal Solution (TOPSIS) analysis. The package is designed to facilitate the execution of TOPSIS analysis on input data using a command-line interface.

## Usage

To utilize the TOPSIS analysis, follow these steps:

### 1. Program Structure and Imports:
   - The package structure is as follows:
     - **Imports:**
       - `pandas` for data manipulation.
       - `normalize` from `sklearn.preprocessing` for data normalization.
       - `LabelEncoder` from `sklearn.preprocessing` for encoding categorical columns.
       - `warnings` for handling warnings.
       - `sys` for command-line arguments.

### 2. Extracting Data from Command Line:
   - Command-line arguments are extracted, including the input file path, weights, impacts, and the desired result file name. An error message is displayed if the correct number of arguments is not provided.

### 3. Reading Input Data:
   - The input Excel file is read using pandas, with the 'Fund Name' column set as the index. If the file is not found, an error message is displayed, and the program exits.

### 4. Validate Input Data:
   - The script checks if the input data meets certain criteria:
     - The input file must contain three or more columns.
     - Columns, excluding the 'Fund Name' index, should contain only numeric values.
     - The number of weights, impacts, and columns (excluding the index) must be the same.
     - Impacts must be either '+' or '-'.

### 5. Convert Categorical Columns to Numerical:
   - If there are categorical columns in the input data, they are encoded using the `LabelEncoder` from scikit-learn.

### 6. Normalization:
   - The input data is normalized using the sklearn `normalize` function. Each column is normalized based on the specified weights.

### 7. Finding Ideal Best and Ideal Worst:
   - Ideal best and ideal worst values are determined for each column based on the specified impacts ('+' or '-'). Ideal best values are the maximum values for '+' impacts, and ideal worst values are the minimum values.

### 8. Calculating Euclidean Distance:
   - Euclidean distances (splus and sminus) are computed for each row, representing the similarity to ideal solutions (best and worst).

### 9. Calculate Performance Score:
   - The performance score is calculated for each row based on the formula: sminus / (sminus + splus).

### 10. Assign Rank:
   - Each row is assigned a rank based on its performance score, with lower scores receiving higher ranks.

### 11. Appending Results to Original DataFrame:
   - The Topsis Score and Rank columns are appended to the original DataFrame.

### 12. Save the Result DataFrame to a CSV File:
   - The resulting DataFrame, including Topsis Score and Rank columns, is saved to a CSV file specified in the command-line arguments.

## Execution

To execute the package, run the script from the command line, providing the required input parameters in the following order:
```bash
python program.py <InputDataFile> <Weights> <Impacts> <ResultFileName>
```

Ensure the correct number of parameters, handle file not found exceptions, and adhere to the specified input file format. The resulting DataFrame with Topsis Score and Rank columns will be saved in a CSV file.

Note: Make sure to install the required libraries before running the program using the following:
```bash
pip install pandas scikit-learn
```