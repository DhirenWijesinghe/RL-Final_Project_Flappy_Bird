import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('scores_151.csv', names=['episode','score'])
df = df.iloc[21: , :]

# df.plot(x='episode', y='score')
plt.plot(df["episode"], df["score"])
plt.show()