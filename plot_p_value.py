import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("co_significance.csv")  
#df = pd.read_csv("vote_count_significance.csv")
#df = pd.read_csv("vote_average_significance.csv")
#df = pd.read_csv("revenue_significance.csv")

sns.set(style="whitegrid")
plt.figure(figsize=(8,5))
sns.histplot(df['p_value'], bins=40, kde=False, color='skyblue', edgecolor='black')
plt.axvline(x=0.05, color='red', linestyle='--', label='p = 0.05')
plt.xlabel('p-value')
plt.ylabel('Number of keywords')
plt.title('Distribution of p-values for Keyword Centrality')
plt.legend()
plt.tight_layout()
plt.show()
