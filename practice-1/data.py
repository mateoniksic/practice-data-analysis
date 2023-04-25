import pandas as pd
import numpy as np
import scipy.stats as stats

"""
1.1. LOADING DATA
"""
df = pd.read_csv(
    'data.csv',
    na_values=['NA', np.nan],
    usecols=[
        'MainBranch',
        'Employment',
        'EdLevel',
        'YearsCodePro',
        'Country',
        'Gender',
        'ConvertedCompYearly'
    ])


print(f"\n{60*'='}\nLOADING DATA\n{60*'='}")
print(f'Initial data size: {len(df)} records.\n')


"""
1.2. CLEANING DATA
"""
# Drop all rows that contain NA and are duplicates.
df = df.dropna().drop_duplicates()

# Filter data - Only select rows that contain 'I am a developer by profession'.
df = df[(df['MainBranch'] == 'I am a developer by profession')]
df = df.drop('MainBranch', axis='columns')

# Replace 'Less than 1 year', 'More than 50 years' with numbers in column 'YearsCodePro'.
df = df.replace(['Less than 1 year', 'More than 50 years'], [0, 51])
df['YearsCodePro'] = pd.to_numeric(df['YearsCodePro'])

# Filter data - Use only data from 5th to 95th percentile to remove outliers - ConvertedCompYearly.
_qlow, _qhigh = np.percentile(df.ConvertedCompYearly, [5, 95])
df = df[(df['ConvertedCompYearly'] > _qlow) &
        (df['ConvertedCompYearly'] < _qhigh)]

# Filter data - Use only data up to 95th percentile to remove outliers - YearsCodePro.
_qhigh = np.percentile(df.YearsCodePro, 95)
df = df[(df['YearsCodePro'] < _qhigh)]

# Sort values from the lowest to the highest yearly compensation and reset index.
df = df.sort_values(by='ConvertedCompYearly').reset_index(drop=True)


print(f'\n{60*"="}\nCLEANING DATA\n{60*"-"}\nSelect only "Developers by profession" from 5th to 95th percentile based on gross yearly compensation in USD and up to 95th percentile based on "YearsCodePro".\n{60*"="}')
print(f'Cleaned data size: {len(df)} records.\n')


"""
2. DATA MANIPULATION
"""
# Group rows by country.
groupedby_country = df.groupby('Country')
groupedby_gender = df.groupby('Gender')

# Count total number of countries.
total_countries, _ = np.shape(groupedby_country.first())

# Find top 10 countries with minimum and maximum yearly compensation.
compensationby_country = groupedby_country['ConvertedCompYearly'].aggregate(['min', 'max'])
compensationby_country_top10_min = compensationby_country.sort_values(by='min').head(10)
compensationby_country_top10_max = compensationby_country.sort_values(by='max', ascending=False).head(10)

# Find top 5 countries with minimum and maximum years of experience.
yoeby_country = groupedby_country['YearsCodePro'].aggregate(['mean'])
yoeby_country_top5 = yoeby_country.sort_values(by='mean', ascending=False).head(5)
yoeby_country_last5 = yoeby_country.sort_values(by='mean').head(5)

# Mean compensation by gender
compensationby_gender_mean = []
for group_name, group_df in groupedby_gender:
    compensationby_gender_mean.append((group_name, np.around(group_df['ConvertedCompYearly'].mean(), 2)))

print(f'\n{60*"="}\nDATA MANIPULATION\n{60*"="}')
print(f'Number of countries: {total_countries}.\n')
print(f'Top 10 countries with the lowest gross yearly compensation:\n{compensationby_country_top10_min}.\n\n')
print(f'Top 10 countries with the highest gross yearly compensation:\n{compensationby_country_top10_max}.\n\n')
print(f'Top 5 countries with the highest median years of developer experience:\n\n{yoeby_country_top5}.\n\n')
print(f'Last 5 countries with the lowest median years of developer experience:\n\n{yoeby_country_last5}.\n\n')

print(f'Mean compensation per year per gender:\n')
for gender, compensation in compensationby_gender_mean:
    print(f'{gender}: ${format(compensation, ",")}.')

"""
3. DESCRIPTIVE STATISTICS
"""
# ConvertedCompYearly World Wide
ycomp_ww_std = np.around(np.std(df['ConvertedCompYearly'], axis=0))
ycomp_ww_var = np.around(np.var(df['ConvertedCompYearly'], axis=0))
ycomp_ww_mean = np.around(df['ConvertedCompYearly'].mean(), 2)
ycomp_ww_median = np.around(df['ConvertedCompYearly'].median(), 2)
ycomp_ww_min = np.around(df['ConvertedCompYearly'].min(), 2)
ycomp_ww_max = np.around(df['ConvertedCompYearly'].max(), 2)

# ConvertedCompYearly USA
ycomp_usa_mean = np.around(groupedby_country.get_group('United States of America')['ConvertedCompYearly'].mean(), 2)
ycomp_usa_median = np.around(groupedby_country.get_group('United States of America')['ConvertedCompYearly'].median(), 2)
ycomp_usa_min = np.around(groupedby_country.get_group('United States of America')['ConvertedCompYearly'].min(), 2)
ycomp_usa_max = np.around(groupedby_country.get_group('United States of America')['ConvertedCompYearly'].max(), 2)

# ConvertedCompYearly Croatia
ycomp_cro_mean = np.around(groupedby_country.get_group('Croatia')['ConvertedCompYearly'].mean(), 2)
ycomp_cro_median = np.around(groupedby_country.get_group('Croatia')['ConvertedCompYearly'].median(), 2)
ycomp_cro_min = np.around(groupedby_country.get_group('Croatia')['ConvertedCompYearly'].min(), 2)
ycomp_cro_max = np.around(groupedby_country.get_group('Croatia')['ConvertedCompYearly'].max(), 2)

# ConvertedCompYearly India
ycomp_in_mean = np.around(groupedby_country.get_group('India')['ConvertedCompYearly'].mean(), 2)
ycomp_in_median = np.around(groupedby_country.get_group('India')['ConvertedCompYearly'].median(), 2)
ycomp_in_min = np.around(groupedby_country.get_group('India')['ConvertedCompYearly'].min(), 2)
ycomp_in_max = np.around(groupedby_country.get_group('India')['ConvertedCompYearly'].max(), 2)

print(f'\n{60*"="}\nDESCRIPTIVE STATISTICS\n{60*"="}')
print(f'Standard deviation - gross yearly compensation: ${format(ycomp_ww_std, ",")}.\n')
print(f'Variance - gross yearly compensation: ${format(ycomp_ww_var, ",")}.\n')
print(f'Average gross yearly compensation - World Wide: ${format(ycomp_ww_mean, ",")}.\n')
print(f'Median gross yearly compensation - World Wide: ${format(ycomp_ww_median, ",")}.\n')
print(f'Minimum gross yearly compensation - World Wide: ${format(ycomp_ww_min, ",")}.\n')
print(f'Maximum gross yearly compensation - World Wide: ${format(ycomp_ww_max, ",")}.\n\n')

print(f'Average gross yearly compensation - USA: ${format(ycomp_usa_mean, ",")}.\n')
print(f'Median gross yearly compensation - USA: ${format(ycomp_usa_median, ",")}.\n')
print(f'Minimum gross yearly compensation - USA: ${format(ycomp_usa_min, ",")}.\n')
print(f'Maximum gross yearly compensation - USA: ${format(ycomp_usa_max, ",")}.\n\n')

print(f'Average gross yearly compensation - Croatia: ${format(ycomp_cro_mean, ",")}.\n')
print(f'Median gross yearly compensation - Croatia: ${format(ycomp_cro_median, ",")}.\n')
print(f'Minimum gross yearly compensation - Croatia: ${format(ycomp_cro_min, ",")}.\n')
print(f'Maximum gross yearly compensation - Croatia: ${format(ycomp_cro_max, ",")}.\n\n')

print(f'Average gross yearly compensation - India: ${format(ycomp_in_mean, ",")}.\n')
print(f'Median gross yearly compensation - India: ${format(ycomp_in_median, ",")}.\n')
print(f'Minimum gross yearly compensation - India: ${format(ycomp_in_min, ",")}.\n')
print(f'Maximum gross yearly compensation - India: ${format(ycomp_in_max, ",")}.\n')


"""
4. INFERENCIAL STATISTICS
"""
# Compensation yearly World Wide
data = df['ConvertedCompYearly'].values

# Calculate confidence interval.
confidence_level = 0.99

# Calculate significance.
alpha = 1 - confidence_level

# Sample size.
n = len(data)

# Sample mean. Can be anything like median.
mean = np.mean(data)

# Calculate standard error of the mean.
standard_error = np.std(data, ddof=1) / np.sqrt(n)

# Calculate critical value (z-score) based on alpha and sample size.
critical_value = abs(np.round(stats.norm.ppf(alpha/2), 2))

# Calculate confidence interval.
lower_bound = mean - critical_value * standard_error
upper_bound = mean + critical_value * standard_error

print(f'\n{60*"="}\nINFERENCIAL STATISTICS\n{60*"="}')
print(
    f"Average yearly compensation World Wide - confidence interval: ({confidence_level*100}%), [{lower_bound:.2f}, {upper_bound:.2f}]\n")


"""
5. STATIC ANALYSIS
"""
ycomp_cro = groupedby_country.get_group('Croatia')['ConvertedCompYearly'].values
ycomp_cro_size = len(ycomp_cro)

ycomp_ph = groupedby_country.get_group('Philippines')['ConvertedCompYearly'].values
ycomp_ph_size = len(ycomp_ph)

ycomp_cl = groupedby_country.get_group('Chile')['ConvertedCompYearly'].values[:95]
ycomp_cl_size = len(ycomp_cl)


corr_cro_ph, _ = np.around(stats.pearsonr(ycomp_cro, ycomp_ph), 4)
corr_cro_cl, _ = np.around(stats.pearsonr(ycomp_cro, ycomp_cl), 4)
corr_ph_cl, _ = np.around(stats.pearsonr(ycomp_ph, ycomp_cl), 4)

f_stat, p_value = stats.f_oneway(ycomp_cro, ycomp_ph, ycomp_cl)

print(f'\n{60*"="}\nSTATIC ANALYSIS\n{60*"="}')
print(f'Data set sizes: Croatia ({ycomp_cro_size}), Philippines ({ycomp_ph_size}), Chile ({ycomp_cl_size}).\n\n')

print(f"Correlation between Croatia and Philippines: {corr_cro_ph}.\n")
print(f"Correlation between Croatia and Chile: {corr_cro_cl}.\n")
print(f"Correlation between Philippines and Chile: {corr_ph_cl}.\n\n")

print(f'F-statistic: ({np.around(f_stat, 4)}), P-value: ({np.around(p_value, 4)}).\n')
print('Null hypothesis assuumes no statistical significant differences among the means of the groups, p-value < 0.05 is used to reject null hypothesis. There are statistically significant differences among the means of the groups (Croatia, Philippines, and Chile).\n')
