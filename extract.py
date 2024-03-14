import pandas as pd

# Load the dataset
df = pd.read_csv('articles1.csv', encoding='utf-8')

# Filter the columns and the first 1000 rows
df = df[['author', 'title', 'content']][:100]

df.to_csv('humanraw.csv',encoding='utf-8', index=False)

