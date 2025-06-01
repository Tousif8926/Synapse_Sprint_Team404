import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv('multipleChoiceResponses.csv', low_memory=False)

# Step 1: Generate fake data for 2018–2024 for "Student"
np.random.seed(123)
years = list(range(2018, 2025))
roles = ['Student']
data = []

for role in roles:
    base = np.random.randint(2000, 3000)
    for i, year in enumerate(years):
        noise = np.random.randint(-1000, 1000)
        growth = np.random.randint(300, 600) * i
        count = max(0, base + growth + noise)
        data.append({'Year': year, 'Role': role, 'Count': count})

df_fake_roles = pd.DataFrame(data)

# Step 2: Extract actual "Student" count from 2025 data (Q6)
job_series = df['Q6'].dropna().astype(str).str.lower().str.strip()
real_counts_2025 = []

for role in roles:
    count = job_series.str.contains(role.lower()).sum()
    real_counts_2025.append({'Year': 2025, 'Role': role, 'Count': count})

df_2025_roles = pd.DataFrame(real_counts_2025)

# Combine all data
df_all_roles = pd.concat([df_fake_roles, df_2025_roles], ignore_index=True)

# Step 3: Forecast for 2026
forecasts_roles = []

for role in roles:
    sub_df = df_all_roles[df_all_roles['Role'] == role].copy()
    X = sub_df[['Year']]
    y = sub_df['Count']
    model = LinearRegression()
    model.fit(X, y)
    predicted = int(model.predict([[2026]])[0])
    forecasts_roles.append({'Year': 2026, 'Role': role, 'Count': predicted})

df_2026_roles = pd.DataFrame(forecasts_roles)
df_all_roles = pd.concat([df_all_roles, df_2026_roles], ignore_index=True)

# Step 4: Plot the trends
plt.figure(figsize=(12, 6))
for role in roles:
    role_data = df_all_roles[df_all_roles['Role'] == role]
    plt.plot(role_data['Year'], role_data['Count'], marker='o', label=role)

plt.title('Forecast of Student Role Mentions (2018–2026)')
plt.xlabel('Year')
plt.ylabel('Number of Mentions')
plt.xticks(range(2018, 2027))
plt.grid(True)
plt.legend(title='Job Role')
plt.tight_layout()
plt.show()
