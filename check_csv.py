import pandas as pd
try:
    df = pd.read_csv('/Users/davidhuang/Desktop/CoRefusion/data/test.csv', nrows=5, header=None)
    print("Columns:", df.columns.tolist())
    print("First row values (summarized):")
    for i, col in enumerate(df.columns):
        val = str(df.iloc[0, i])
        print(f"Col {i}: {val[:50]}...")
except Exception as e:
    print("Error:", e)
