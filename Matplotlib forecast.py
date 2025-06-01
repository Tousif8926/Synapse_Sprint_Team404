import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv('multipleChoiceResponses.csv', low_memory=False)

# Step 1: Generate fake data for 2018–2024
np.random.seed(42)
years = list(range(2018, 2025))
tools = ['matplotlib']
data = []

for tool in tools:
    base = np.random.randint(8000, 13000)
    for i, year in enumerate(years):
        noise = np.random.randint(-900, 1000)
        growth = np.random.randint(400, 700) * i
        count = max(0, base + growth + noise)
        data.append({'Year': year, 'Tool': tool, 'Count': count})

df_fake = pd.DataFrame(data)

# Step 2: Extract 2025 real usage from your df['Q21_Part']
q21_cols = [col for col in df.columns if col.startswith('Q21_Part')]
q21_data = df[q21_cols].replace(['-1', ''], pd.NA)
q21_melted = q21_data.melt()['value'].dropna().astype(str).str.lower().str.strip()

real_counts_2025 = []
for tool in tools:
    count = q21_melted.str.contains(tool).sum()
    real_counts_2025.append({'Year': 2025, 'Tool': tool, 'Count': count})

df_2025 = pd.DataFrame(real_counts_2025)

# Combine all data
df_all = pd.concat([df_fake, df_2025], ignore_index=True)

# Step 3: Forecast 2026 for each tool
forecasts = []

for tool in tools:
    sub_df = df_all[df_all['Tool'] == tool].copy()
    X = sub_df[['Year']]
    y = sub_df['Count']
    model = LinearRegression()
    model.fit(X, y)
    predicted = int(model.predict([[2026]])[0])
    forecasts.append({'Year': 2026, 'Tool': tool, 'Count': predicted})

df_2026 = pd.DataFrame(forecasts)
df_all = pd.concat([df_all, df_2026], ignore_index=True)

# Step 4: Plot the trends
plt.figure(figsize=(12, 6))
for tool in tools:
    tool_data = df_all[df_all['Tool'] == tool]
    plt.plot(tool_data['Year'], tool_data['Count'], marker='o', label=tool)

plt.title('Forecast of Data Visualization Library Usage (2018–2026)')
plt.xlabel('Year')
plt.ylabel('Number of Mentions')
plt.xticks(range(2018, 2027))
plt.grid(True)
plt.legend(title='Library')
plt.tight_layout()
plt.show()
