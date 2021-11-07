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
principalDf = pd.DataFrame(data = principalComponents, columns = ['PC1', 'principal component 2'])

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

# Trying out k-means on the dataset

from sklearn.cluster import KMeans

km = KMeans(
    n_clusters=3, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
)

y_km = km.fit_predict(x)