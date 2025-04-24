import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.weightstats import ztest

# Load data
df = pd.read_csv("Department_of_Licensing_Professional_License_Counts_20250406.csv")

# Clean data
df_cleaned = df.dropna(subset=['State'])
df_cleaned.columns = df_cleaned.columns.str.strip().str.lower().str.replace(" ", "_")

# --- EDA & Visualization Setup ---
sns.set(style="whitegrid")

# 1. Top Professional Areas by License Count
plt.figure(figsize=(12, 6))
top_areas = df_cleaned.groupby("professional_area")["count"].sum().sort_values(ascending=False).head(10)
sns.barplot(x=top_areas.values, y=top_areas.index, palette="viridis")
plt.title("Top 10 Professional Areas by License Count", fontsize=16)
plt.xlabel("Total Licenses")
plt.ylabel("Professional Area")
plt.tight_layout()
plt.show()

# 2. Most Common License Types
plt.figure(figsize=(12, 6))
top_licenses = df_cleaned.groupby("license_type")["count"].sum().sort_values(ascending=False).head(10)
sns.barplot(x=top_licenses.values, y=top_licenses.index, palette="coolwarm")
plt.title("Top 10 License Types by Count", fontsize=16)
plt.xlabel("Total Licenses")
plt.ylabel("License Type")
plt.tight_layout()
plt.show()

# 3. Violin Plot: Distribution of License Count by Top License Types
top_5_license_types = df_cleaned.groupby('license_type')['count'].sum().sort_values(ascending=False).head(5).index
df_top_license_types = df_cleaned[df_cleaned['license_type'].isin(top_5_license_types)]

plt.figure(figsize=(12, 6))
sns.violinplot(x='license_type', y='count', data=df_top_license_types, palette='Pastel1')
plt.title("Distribution of License Counts for Top 5 License Types")
plt.xlabel("License Type")
plt.ylabel("License Count")
plt.tight_layout()
plt.show()

# 4. Heatmap: Year vs Professional Area (Top 10)
top_10_areas = top_areas.index.tolist()
df_top_areas = df_cleaned[df_cleaned['professional_area'].isin(top_10_areas)]
pivot_table = df_top_areas.pivot_table(index='professional_area', columns='year', values='count', aggfunc='sum')
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title("License Count Heatmap (Top Professional Areas by Year)", fontsize=14)
plt.xlabel("Year")
plt.ylabel("Professional Area")
plt.tight_layout()
plt.show()

# 5. Boxplot: License Count by State (Top 10 States)
top_10_states = df_cleaned.groupby('state')['count'].sum().sort_values(ascending=False).head(10).index
df_top_states = df_cleaned[df_cleaned['state'].isin(top_10_states)]
plt.figure(figsize=(12, 6))
sns.boxplot(x='state', y='count', data=df_top_states, palette='Set3')
plt.title("License Count Distribution in Top 10 States")
plt.xlabel("State")
plt.ylabel("License Count")
plt.tight_layout()
plt.show()

# Group by county and get top 5 by total license count
top_counties = df_cleaned.groupby('county')['count'].sum().sort_values(ascending=False).head(5)
plt.figure(figsize=(8, 8))
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0']
plt.pie(top_counties.values, labels=top_counties.index, autopct='%1.1f%%', colors=colors, startangle=140)
plt.title('Top 5 Counties by License Count')
plt.axis('equal')  # Keeps it circular
plt.tight_layout()
plt.show()

# --- Pandas/Numpy Based Analysis ---

# 1. Total licenses issued overall
total_licenses = df_cleaned['count'].sum()
print("Total licenses issued:", total_licenses)

# 2. Average number of licenses per record
average_licenses = df_cleaned['count'].mean()
print("Average licenses per record:", average_licenses)

# 3. Standard deviation of license counts
std_dev_licenses = df_cleaned['count'].std()
print("Standard deviation of license counts:", std_dev_licenses)

# 4. Median licenses per record
median_licenses = df_cleaned['count'].median()
print("Median licenses per record:", median_licenses)

# 5. Total licenses by year
total_by_year = df_cleaned.groupby('year')['count'].sum()
print("\nTotal licenses by year:\n", total_by_year)

# 6. Total licenses by county (top 5)
top_counties = df_cleaned.groupby('county')['count'].sum().sort_values(ascending=False).head(5)
print("\nTop 5 counties by total licenses:\n", top_counties)

# 7. Most frequent license type
most_frequent_license = df_cleaned['license_type'].mode()[0]
print("\nMost frequent license type:", most_frequent_license)

# 8. Unique professional areas
unique_areas = df_cleaned['professional_area'].nunique()
print("\nNumber of unique professional areas:", unique_areas)

# 9. Correlation between year and license count (numerical trend)
year_counts = df_cleaned.groupby('year')['count'].sum()
print("\nCorrelation (year vs total count):\n", year_counts.corr(pd.Series(year_counts.index)))

# 10. Proportion of licenses issued in WA
wa_proportion = df_cleaned[df_cleaned['state'] == 'WA']['count'].sum() / total_licenses
print("\nProportion of licenses issued in WA:", wa_proportion)

# Hypothesis: License counts in WA are significantly different from the rest of the states
wa_counts = df_cleaned[df_cleaned['state'] == 'WA']['count']
other_counts = df_cleaned[df_cleaned['state'] != 'WA']['count']
t_stat, p_val = stats.ttest_ind(wa_counts, other_counts, equal_var=False)
print("\nHypothesis Test: License count difference (WA vs Others)")
print("T-statistic:", t_stat)
print("P-value:", p_val)

# Separate King County vs Other Counties
king_county = df_cleaned[df_cleaned['county'] == 'King']['count']
other_counties = df_cleaned[df_cleaned['county'] != 'King']['count']

# Run the Z-test
z_stat, p_value = ztest(king_county, other_counties, alternative='two-sided')

print("Z-statistic:", z_stat)
print("P-value:", p_value)

# Interpretation
if p_value < 0.05:
    print("Reject the null hypothesis: There's a statistically significant difference in license counts.")
else:
    print("Fail to reject the null hypothesis: No significant difference detected.")
