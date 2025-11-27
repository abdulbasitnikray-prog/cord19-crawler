import pandas as pd
# Visualize the CSV Files
pd.set_option('display.max_columns',None)

df = pd.read_csv("D:/Cord19/cord/2022/metadata.csv",nrows=5)
print(df.T)