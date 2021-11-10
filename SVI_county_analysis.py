# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

Author: Prudvi Gaddam

"""


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

svi_2018 = pd.read_csv('G:\\My Drive\\School\\Side Project\\SVI side project\\SVI_side_project\\SVI2018_US_COUNTY.csv')
col_for_PCA = ['EP_POV','EP_UNEMP','EP_PCI','EP_NOHSDP','EP_AGE65','EP_AGE17','EP_DISABL','EP_SNGPNT','EP_MINRTY','EP_LIMENG','EP_MUNIT','EP_MOBILE','EP_CROWD','EP_NOVEH','EP_GROUPQ']

# Histogram of all the values in a given 

fig, axis = plt.subplots(3,5,figsize=(8, 8))
svi_2018.loc[:, col_for_PCA].hist(ax=axis)

#Remove the entry for  Rio Arriba county, NM because it shows a really weird POV and UNEMP rate
svi_2018 = svi_2018[svi_2018.FIPS != 35039] 

# Recheck histogram to make sure it's all fine
fig, axis = plt.subplots(3,5,figsize=(8, 8))
svi_2018.loc[:, col_for_PCA].hist(ax=axis)

# Performing a correlation analysis 

corrMatrix = svi_2018.loc[:, col_for_PCA].corr()
corrMatrix_sig = corrMatrix[np.abs(corrMatrix) >= 0.2]
sns.heatmap(corrMatrix_sig, annot=True)
plt.show()

# performing PCA on SVI by county dataset

x = svi_2018.loc[:, col_for_PCA].values
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=5)

principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
ax.scatter(principalComponents[:,0], principalComponents[:,1])
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.show()

pca.explained_variance_ratio_

# Trying out k-means on the dataset and pulling the code (reminder to try to automate the number of clusters, centroids etc. )

from sklearn.cluster import KMeans

km = KMeans(
    n_clusters=3, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
)

y_km = km.fit_predict(principalDf)

# %% plotting the data

plt.scatter(
    principalDf.iloc[y_km == 0, 0], principalDf.iloc[y_km == 0, 1],
    s=50, c='lightgreen',
    marker='s', edgecolor='black',
    label='cluster 1'
)

plt.scatter(
    principalDf.iloc[y_km == 1, 0], principalDf.iloc[y_km == 1, 1],
    s=50, c='orange',
    marker='o', edgecolor='black',
    label='cluster 2'
)

plt.scatter(
    principalDf.iloc[y_km == 2, 0], principalDf.iloc[y_km == 2, 1],
    s=50, c='lightblue',
    marker='v', edgecolor='black',
    label='cluster 3'
)

# plot the centroids
plt.scatter(
    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
    s=250, marker='*',
    c='red', edgecolor='black',
    label='centroids'
)
plt.legend(scatterpoints=1)
plt.grid()
plt.show()

# %% Checking the number of clusters needed for the SVI dataset

distortions = []
for i in range(1, 51):
    km = KMeans(
        n_clusters=i, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    km.fit(x)
    distortions.append(km.inertia_)


plt.plot(range(1, 51), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

# picked 8 clusters basedo n the data

# %% Visualizing the 8 clusters in PCA space

km =  KMeans(
        n_clusters=8, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
km.fit(x)

svi_2018['cluster'] = km.labels_

plt.figure()
plt.scatter(
    principalDf.iloc[:, 0], principalDf.iloc[:, 1], c = km.labels_
)
plt.show()

col_for_desc = col_for_PCA.copy()
col_for_desc.append('cluster')
cluster_desc = svi_2018.loc[:, col_for_desc].groupby('cluster').agg('mean')

# %% Cleaning up and doing some basic analysis on NC county health data

nc_county_health = pd.read_excel(r'G:\My Drive\School\Side Project\SVI side project\SVI_side_project\2021 County Health Rankings North Carolina Data - v1.xlsx', header = 1, sheet_name = 'Ranked Measure Data')
nc_county_health = nc_county_health.loc[1:len(nc_county_health)]

nc_county_health_addl_measures = pd.read_excel(r'G:\My Drive\School\Side Project\SVI side project\SVI_side_project\2021 County Health Rankings North Carolina Data - v1.xlsx', header = 1, sheet_name = 'Additional Measure Data')
nc_county_health_addl_measures = nc_county_health_addl_measures[1:len(nc_county_health_addl_measures)]

nc_county_health = nc_county_health.merge(nc_county_health_addl_measures, left_on = ['FIPS', 'State', 'County'], right_on = ['FIPS', 'State', 'County'])
nc_county_health = nc_county_health.loc[:,~nc_county_health.columns.str.contains('CI')]
nc_county_health = nc_county_health.loc[:,~nc_county_health.columns.str.contains('Z-Score')]

nc_county_health_cluster = nc_county_health.merge(svi_2018[['FIPS', 'cluster']], right_on=['FIPS'], left_on=['FIPS'])

summary_nc = nc_county_health_cluster[['Life Expectancy', 'Child Mortality Rate', 'Drug Overdose Mortality Rate', 'Median Household Income','Teen Birth Rate','cluster','Suicide Rate (Age-Adjusted)']].groupby('cluster').agg('mean')

# %% Performing the same data import and data cleaning on SC

sc_county_health = pd.read_excel(r'G:\My Drive\School\Side Project\SVI side project\SVI_side_project\2021 County Health Rankings South Carolina Data - v1.xlsx', header = 1, sheet_name = 'Ranked Measure Data')
sc_county_health = sc_county_health.loc[1:len(sc_county_health)]

sc_county_health_addl_measures = pd.read_excel(r'G:\My Drive\School\Side Project\SVI side project\SVI_side_project\2021 County Health Rankings South Carolina Data - v1.xlsx', header = 1, sheet_name = 'Additional Measure Data')
sc_county_health_addl_measures = sc_county_health_addl_measures[1:len(sc_county_health_addl_measures)]

sc_county_health = sc_county_health.merge(sc_county_health_addl_measures, left_on = ['FIPS', 'State', 'County'], right_on = ['FIPS', 'State', 'County'])
sc_county_health = sc_county_health.loc[:,~sc_county_health.columns.str.contains('CI')]
sc_county_health = sc_county_health.loc[:,~sc_county_health.columns.str.contains('Z-Score')]

sc_county_health_cluster = sc_county_health.merge(svi_2018[['FIPS', 'cluster']], right_on=['FIPS'], left_on=['FIPS'])

summary_sc = sc_county_health_cluster[['Life Expectancy', 'Child Mortality Rate', 'Drug Overdose Mortality Rate', 'Median Household Income','Teen Birth Rate','cluster','Suicide Rate (Age-Adjusted)']].groupby('cluster').agg('mean')

# %% Take a look at national data

# Load in the dataset related to county health measures and outcomes
national_county_health = pd.read_excel(r'G:\My Drive\School\Side Project\SVI side project\SVI_side_project\2021 County Health Rankings Data - v1.xlsx', header = 1, sheet_name = 'Ranked Measure Data')
national_county_health = national_county_health.loc[1:len(national_county_health)]

national_county_health_addl_measures = pd.read_excel(r'G:\My Drive\School\Side Project\SVI side project\SVI_side_project\2021 County Health Rankings Data - v1.xlsx', header = 1, sheet_name = 'Additional Measure Data')
national_county_health_addl_measures = national_county_health_addl_measures[1:len(national_county_health_addl_measures)]

# Merge the two excel sheets from teh datasets
national_county_health = national_county_health.merge(national_county_health_addl_measures, left_on = ['FIPS', 'State', 'County'], right_on = ['FIPS', 'State', 'County'])
national_county_health = national_county_health.loc[:,~national_county_health.columns.str.contains('CI')]
national_county_health = national_county_health.loc[:,~national_county_health.columns.str.contains('Z-score')]

national_county_health_copy = national_county_health.copy()

# Remove columns that have unreliable in them
national_county_health_copy = national_county_health_copy[national_county_health_copy['Unreliable'].isna() == True]

# Drop columns with over 50% Null values
perc = 50.0 # Like N %
min_count =  int(((100-perc)/100)*national_county_health_copy.shape[0] + 1)
national_county_health_copy = national_county_health_copy.dropna(axis=1, 
                thresh=min_count)

national_county_health_cluster = national_county_health_copy.merge(svi_2018[['FIPS', 'cluster']], right_on=['FIPS'], left_on=['FIPS'], how = 'inner')

summary_national = national_county_health_cluster[['Life Expectancy', 'Child Mortality Rate', 'Drug Overdose Mortality Rate', 'Median Household Income','Teen Birth Rate','cluster','Suicide Rate (Age-Adjusted)']].groupby('cluster').agg('mean')

print(national_county_health_cluster.groupby('cluster').size())

for i in range(0,8):
    plt.figure(i+1)
    sns.distplot(national_county_health_cluster[national_county_health_cluster['cluster'] == i]['Drug Overdose Mortality Rate'], bins = 20)
    
cluster_4 = national_county_health_cluster[national_county_health_cluster['cluster'] == 4]

sns.scatterplot(data = national_county_health_cluster, x = '% Insufficient Sleep', y = 'Suicide Rate (Age-Adjusted)')

# Filter values of suicide rate using whiskers of a box-whisker plot (median +/- 1.5*IQR). Note that there are interesting findings at the extremes, we are just limiting the range to find correlations that are interesting
q3, q1 = np.percentile(national_county_health_cluster['Suicide Rate (Age-Adjusted)'].dropna(), [75 ,25])
iqr = q3 - q1
upper_bound = np.median(national_county_health_cluster['Suicide Rate (Age-Adjusted)'].dropna()) + 1.5*iqr
lower_bound = np.median(national_county_health_cluster['Suicide Rate (Age-Adjusted)'].dropna()) - 1.5*iqr

suicide_filtered = national_county_health_cluster[(national_county_health_cluster['Suicide Rate (Age-Adjusted)'] > lower_bound) & (national_county_health_cluster['Suicide Rate (Age-Adjusted)'] < upper_bound)]

suicide_filtered_numeric = suicide_filtered._get_numeric_data()

suicide_corr = suicide_filtered_numeric.corr()['Suicide Rate (Age-Adjusted)'][:]

sns.scatterplot(data = suicide_filtered_numeric, x = 'Juvenile Arrest Rate', y = 'Suicide Rate (Age-Adjusted)')


