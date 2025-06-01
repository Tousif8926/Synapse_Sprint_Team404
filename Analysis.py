import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('multipleChoiceResponses.csv', low_memory=False)

# Utility function to plot value counts
def plot_counts(series, title, xlabel, ylabel='Count', top_n=None):
    counts = series.value_counts().drop(['-1', '', None], errors='ignore').head(top_n)
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=counts.values,
        y=counts.index,
        hue=counts.index,       
        palette='crest',
        dodge=False,            
        legend=False            
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

# Q17 - Most Common Programming Language Used (Single-select)
plot_counts(df['Q17'],'Most Common Programming Languages Used','Number of Respondents',top_n=15)

# Q6 - Job Title
plot_counts(df['Q6'], 'Job Titles of Respondents', 'Number of Respondents', top_n=15)

# Q11 - Activities in Role
q11_cols = [col for col in df.columns if col.startswith('Q11_Part')]
q11_data = df[q11_cols].replace(['-1', ''], pd.NA)

# Melt and clean values (remove non-string or very long ones)
melted = q11_data.melt()['value'].dropna()
melted = melted[melted.apply(lambda x: isinstance(x, str) and len(x) < 80)]

plot_counts(melted, 'Important Activities in Current Role', 'Number of Mentions')

# Q21 - Data Visualization Libraries
q21_cols = [col for col in df.columns if col.startswith('Q21_Part')]
q21_data = df[q21_cols].replace(['-1', ''], pd.NA)
q21_melted = q21_data.melt()['value'].dropna()
q21_melted = q21_melted[q21_melted.apply(lambda x: isinstance(x, str) and len(x) < 80)]

plot_counts(q21_melted, 'Data Visualization Libraries Used (Past 5 Years)', 'Number of Mentions')

# Q49 - Reproducibility Tools
q49_cols = [col for col in df.columns if col.startswith('Q49_Part')]
q49_data = df[q49_cols].replace(['-1', ''], pd.NA)
q49_melted = q49_data.melt()['value'].dropna()
q49_melted = q49_melted[q49_melted.apply(lambda x: isinstance(x, str) and len(x) < 80)]

plot_counts(q49_melted, 'Tools for Reproducible Work', 'Number of Mentions')

