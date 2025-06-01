import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv('C:/Users/kumar/Downloads/multipleChoiceResponses.csv', low_memory=False)

# Step 1: Generate fake data for 2018–2024
np.random.seed(42)
years = list(range(2018, 2025))
languages = ['python', 'r', 'sql']
data = []

# Language-specific settings
settings = {
    'python': {'start_min': 4000, 'start_max': 8000, 'noise': (500, 1000), 'growth': (400, 800)},
    'r': {'start_min': 1200, 'start_max': 2200, 'noise': (200, 300), 'growth': (100, 250)},
    'sql': {'start_min': 500, 'start_max': 1300, 'noise': (200, 300), 'growth': (80, 180)},
}

for lang in languages:
    base = np.random.randint(settings[lang]['start_min'], settings[lang]['start_max'])
    for i, year in enumerate(years):
        noise = np.random.randint(-settings[lang]['noise'][0], settings[lang]['noise'][1])
        growth = np.random.randint(settings[lang]['growth'][0], settings[lang]['growth'][1]) * i
        count = max(0, base + growth + noise)
        data.append({'Year': year, 'Language': lang, 'Count': count})

df_fake = pd.DataFrame(data)

# Step 2: Extract 2025 real usage from single Q17 column
q17_data = df['Q17'].dropna().astype(str).str.lower().str.strip()

# Count mentions for each language
real_counts_2025 = []
for lang in languages:
    count = q17_data.str.contains(lang).sum()
    real_counts_2025.append({'Year': 2025, 'Language': lang, 'Count': count})

df_2025 = pd.DataFrame(real_counts_2025)

# Combine all data
df_all = pd.concat([df_fake, df_2025], ignore_index=True)

# Step 3: Forecast 2026 for each language
forecasts = []

for lang in languages:
    sub_df = df_all[df_all['Language'] == lang].copy()
    X = sub_df[['Year']]
    y = sub_df['Count']
    model = LinearRegression()
    model.fit(X, y)
    predicted = int(model.predict(pd.DataFrame({'Year': [2026]}))[0])
    forecasts.append({'Year': 2026, 'Language': lang, 'Count': predicted})

df_2026 = pd.DataFrame(forecasts)
df_all = pd.concat([df_all, df_2026], ignore_index=True)

# Step 4: Plot the trends
plt.figure(figsize=(12, 6))
for lang in languages:
    lang_data = df_all[df_all['Language'] == lang]
    plt.plot(lang_data['Year'], lang_data['Count'], marker='o', label=lang.capitalize())

plt.title('Forecast of Programming Language Usage (2018–2026)')
plt.xlabel('Year')
plt.ylabel('Number of Mentions')
plt.xticks(range(2018, 2027))
plt.grid(True)
plt.legend(title='Language')
plt.tight_layout()
plt.show()
