import pandas as pd
data_file = 'cloud_data.csv'
df = pd.read_csv(data_file, encoding="utf-8")
df = df[df['Distance'] <= 100000]
df.to_csv('distance_cut.csv', index=False)