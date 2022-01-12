# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 14:15:03 2021

@author: Prudvi
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

#%% Defining useful functions here that will be used later

def simple_hist(df, list_of_vars):
    for var in list_of_vars:
        plt.figure()
        sns.displot(data = df, x = var)
        plt.show()

#%% Initial data exploration of SVI data
svi_2018 = pd.read_csv("https://test-bucket-prg.s3.amazonaws.com/SVI2018_US_COUNTY.csv")

# Taking only the variables that are important and can be used for modeling from the SVI dataset 
svi_cols = ['EP_POV','EP_UNEMP','EP_PCI','EP_NOHSDP','EP_AGE65','EP_AGE17','EP_DISABL','EP_SNGPNT','EP_MINRTY','EP_LIMENG','EP_MUNIT','EP_MOBILE','EP_CROWD','EP_NOVEH','EP_GROUPQ']

# ID cols from the SVI dataset
id_cols = ['ST', 'STATE', 'ST_ABBR','COUNTY','FIPS','LOCATION','AREA_SQMI','E_TOTPOP']

# Plot a histogram of all the variables from the svi_cols variable to do outlier analysis and initial data exploration
simple_hist(svi_2018, svi_cols)
    
#Remove the entry for  Rio Arriba county, NM because it shows a really weird POV and UNEMP rate
svi_2018 = svi_2018[svi_2018.FIPS != 35039] 

#Check out the histogram again with the offending entry removed
simple_hist(svi_2018, svi_cols)

# Check if there are any missing values in the dataset
svi_2018[svi_cols].isna().sum().reset_index(name="n").plot.bar(x='index', y='n', rot=45)

#%% Take a look at national county health data

# Load in the dataset related to county health measures and outcomes
national_county_health = pd.read_excel('https://test-bucket-prg.s3.amazonaws.com/2021+County+Health+Rankings+Data+-+v1.xlsx', header = 1, sheet_name = 'Ranked Measure Data')
national_county_health = national_county_health.loc[1:len(national_county_health)]

national_county_health_addl_measures = pd.read_excel(r'G:\My Drive\School\Side Project\SVI side project\SVI_side_project\2021 County Health Rankings Data - v1.xlsx', header = 1, sheet_name = 'Additional Measure Data')
national_county_health_addl_measures = national_county_health_addl_measures[1:len(national_county_health_addl_measures)]

# Merge the two excel sheets from the datasets and drop varaibles that are the confidence intervals and the z-scores of the counties
national_county_health = national_county_health.merge(national_county_health_addl_measures, left_on = ['FIPS', 'State', 'County'], right_on = ['FIPS', 'State', 'County'])
national_county_health = national_county_health.loc[:,~national_county_health.columns.str.contains('CI')]
national_county_health = national_county_health.loc[:,~national_county_health.columns.str.contains('Z-score')]

national_county_health_copy = national_county_health.copy()

# Remove columns that have unreliable in them
national_county_health_copy = national_county_health_copy[national_county_health_copy['Unreliable'].isna() == True].reset_index()

# Drop columns with over 50% Null values
perc = 50.0 # Like N %
min_count =  int(((100-perc)/100)*national_county_health_copy.shape[0] + 1)
national_county_health_copy = national_county_health_copy.dropna(axis=1, 
                thresh=min_count)

tar_var = ['Life Expectancy', 'Child Mortality Rate', 'Drug Overdose Mortality Rate', 'Teen Birth Rate','Suicide Rate (Age-Adjusted)', '% Driving Deaths with Alcohol Involvement']
nch_id_var = ['FIPS' , 'State', 'County']

#Check out the histogram again with the offending entry removed
simple_hist(national_county_health_copy, tar_var)

#%% This section does an ANOVA (or ANOVA like) analysis looking at the impact of racial makeup on suicide rates in a county

# First method for looking at this question - check out the counties where most people are of a certain race
def racial_makeup(col1, col2, col3, col4, col5):
    if col1 > 50:
        race = 'Majority Black'
    elif col2 > 50:
        race = 'Majority Asian'
    elif col3 > 50:
        race = 'Majority Hispanic'
    elif col4 > 50:
        race = 'Majority Native'
    elif col5 > 50:
        race = 'Majority White'
    else:
        race = 'Other/Mixed'
    return race

national_county_health_copy['Race'] = ''

for i in range(0,len(national_county_health_copy)):
    black = national_county_health_copy['% Black'][i]
    asian = national_county_health_copy['% Asian'][i]
    hispanic = national_county_health_copy['% Hispanic'][i]
    native = national_county_health_copy['% American Indian & Alaska Native'][i]
    white = national_county_health_copy['% Non-Hispanic White'][i]
    # print(racial_makeup(black, asian, hispanic, native))
    # print(i)
    national_county_health_copy['Race'][i] = racial_makeup(black, asian, hispanic, native, white)
    # print(national_county_health_copy['FIPS'][i])

# Check the impact of race using simple means 
suicide_rate_race_1 = national_county_health_copy.groupby('Race')['Suicide Rate (Age-Adjusted)'].mean()
overdose_death_rate_race_1 = national_county_health_copy.groupby('Race')['Drug Overdose Mortality Rate'].mean()
teen_birth_rate_race_1 = national_county_health_copy.groupby('Race')['Teen Birth Rate'].mean()
pct_drunk_driving_deaths_1 = national_county_health_copy.groupby('Race')['% Driving Deaths with Alcohol Involvement'].mean()
# Add in as section for ANOVA here for the different groups

#%% Perform the analysis in a slightly different manner looking at counties with top % black, hispanic, asian, white and NA counties, look at suicide, drug overdose and drunk driving deaths 

national_county_health_copy['black_pctile'] = national_county_health_copy['% Black'].rank(pct = True)
national_county_health_copy['asian_pctile'] = national_county_health_copy['% Asian'].rank(pct = True)
national_county_health_copy['hispanic_pctile'] = national_county_health_copy['% Hispanic'].rank(pct = True)
national_county_health_copy['native_pctile'] = national_county_health_copy['% American Indian & Alaska Native'].rank(pct = True)
national_county_health_copy['white_pctile'] = national_county_health_copy['% Non-Hispanic White'].rank(pct = True)

def racial_pctile(col1, col2, col3, col4, col5):
    if col1 >= 0.9:
        race = '10th pctile Native'
    elif col2 >= 0.9:
        race = '10th pctile Asian'
    elif col3 >= 0.9:
        race = '10th pctile Black'
    elif col4 >= 0.9:
        race = '10th pctile Hispanic'
    elif col5 >= 0.9:
        race = '10th pctile White'
    else:
        race = 'Other/Mixed'
    return race

national_county_health_copy['Race Percentile'] = ''

for i in range(0,len(national_county_health_copy)):
    black = national_county_health_copy['black_pctile'][i]
    asian = national_county_health_copy['asian_pctile'][i]
    hispanic = national_county_health_copy['hispanic_pctile'][i]
    native = national_county_health_copy['native_pctile'][i]
    white = national_county_health_copy['white_pctile'][i]
    # print(racial_makeup(black, asian, hispanic, native))
    # print(i)
    national_county_health_copy['Race Percentile'][i] = racial_pctile(native, asian, black, hispanic, white)

# Check the impact of race using simple means 
suicide_rate_race_2 = national_county_health_copy.groupby('Race Percentile')['Suicide Rate (Age-Adjusted)'].mean()
overdose_death_rate_race_2 = national_county_health_copy.groupby('Race Percentile')['Drug Overdose Mortality Rate'].mean()
teen_birth_rate_race_2 = national_county_health_copy.groupby('Race Percentile')['Teen Birth Rate'].mean()
pct_drunk_driving_deaths_2 = national_county_health_copy.groupby('Race Percentile')['% Driving Deaths with Alcohol Involvement'].mean()

#%% Create a simpler list of  variables from the national county health data (this is provisional, additional predictor variables will be added in the future)
nch_simple = national_county_health_copy[nch_id_var + tar_var]
nch_simple['suicide_rank'] = nch_simple['Suicide Rate (Age-Adjusted)'].rank(pct = True)
nch_simple['suicide_risk'] = (nch_simple['suicide_rank'] >= 0.9) * 1

# low_risk = nch_simple[nch_simple['suicide_risk'] == 1]

#%% Build the logistic regression model using SVI metrics for national county health data
# dataframe that contains the svi metrics and the national county health datasets
nch_simple_svi = nch_simple.merge(svi_2018[['FIPS'] + svi_cols], right_on=['FIPS'], left_on=['FIPS'], how = 'inner')

# separate out the x variables
X = nch_simple_svi[svi_cols]

#%%% Calculate the variance inflation factor
from statsmodels.stats.outliers_influence import variance_inflation_factor

#dropping columns iteratively after examining the VIF 
X = X.drop(columns = ['EP_POV', 'EP_PCI', 'EP_NOHSDP', 'EP_AGE65', 'EP_DISABL', 'EP_AGE17'])

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns

# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]
  
print(vif_data)

#%%% Check out the histograms for the two different distributions

xcols = list(X.columns)

for col in xcols:
    plt.figure()
    sns.kdeplot(x = col, data = nch_simple_svi[nch_simple_svi['suicide_risk'] == 1])
    sns.kdeplot(x = col, data = nch_simple_svi[nch_simple_svi['suicide_risk'] == 0])
    plt.legend(['High Risk','Low Risk'])
    plt.show()
    
#%%% Create dataframe with all pairwise interactions from X

from sklearn.preprocessing import PolynomialFeatures
#generating interaction terms
x_int_obj = PolynomialFeatures(2, interaction_only=True, include_bias=False).fit(X)
X_int = pd.DataFrame(x_int_obj.transform(X), columns = x_int_obj.get_feature_names(X.columns))

#%% Building dataset for logistic regression and modeling using policy implementation

#%%% Identify key variables 

nch_policy_var = ['% Fair or Poor Health', '% Adults with Obesity', '% Smokers','% Physically Inactive', '% With Access to Exercise Opportunities', 
                  '% Excessive Drinking', 'Chlamydia Rate','# Primary Care Physicians', 'Primary Care Physicians Ratio','# Mental Health Providers','Mental Health Provider Ratio', 'Food Environment Index',
                  '# Dentists','Dentist Ratio','Preventable Hospitalization Rate','% With Annual Mammogram','% Completed High School','% Some College', '% Unemployed',
                  '% Children in Poverty','Income Ratio','% Children in Single-Parent Households','Social Association Rate','Violent Crime Rate',
                  'Injury Death Rate','Average Daily PM2.5','% Severe Housing Problems','Severe Housing Cost Burden','Overcrowding','Inadequate Facilities',
                  '% Drive Alone to Work','% Long Commute - Drives Alone','Child Mortality Rate','% Frequent Physical Distress','% Frequent Mental Distress',
                  '% Adults with Diabetes', 'HIV Prevalence Rate', '% Food Insecure','% Limited Access to Healthy Foods','% Insufficient Sleep', '% Uninsured_y','% Uninsured.1',
                  'High School Graduation Rate', 'Average Grade Performance','Median Household Income','% Enrolled in Free or Reduced Lunch', 'Segregation index', 'Segregation Index',
                  'Juvenile Arrest Rate','Traffic Volume','% Broadband Access','% Not Proficient in English']

nch_policy = national_county_health_copy[nch_policy_var]
summary_table = nch_policy.describe()

nch_policy['Mental Health Provider Ratio'] = nch_policy['Mental Health Provider Ratio'].apply(lambda x: float(str(x).partition(':')[0]))
nch_policy['Primary Care Physicians Ratio'] = nch_policy['Primary Care Physicians Ratio'].apply(lambda x: float(str(x).partition(':')[0]))
nch_policy['Dentist Ratio'] = nch_policy['Dentist Ratio'].apply(lambda x: float(str(x).partition(':')[0]))

nch_policy.rename(columns = {'% Uninsured_y':'% Uninsured Adults', '% Uninsured.1':'% Uninsured Children', 'Segregation index':'Black/White Segregation Index', 'Segregation Index':'Non-White/White Segregation Index'}, inplace = True)

#%%% Impute missing values
import numpy as np
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors = 5, weights = "uniform")
nch_policy_imputed = pd.DataFrame(imputer.fit_transform(nch_policy), columns = nch_policy.columns)

#%%% Feature creation
svi_2018_geo = svi_2018[['STATE','ST_ABBR','COUNTY','FIPS','LOCATION','AREA_SQMI','E_TOTPOP']]

nch_policy_imputed['FIPS'] = national_county_health_copy['FIPS']
nch_policy_imputed = nch_policy_imputed.merge(svi_2018_geo, right_on=['FIPS'], left_on=['FIPS'], how = 'inner')
nch_policy_imputed['Obesity to Exercise Opportunity Ratio'] = nch_policy_imputed['% Adults with Obesity']/nch_policy_imputed['% With Access to Exercise Opportunities']
nch_policy_imputed['Physically Inactive to Exercise Opportunity Ratio'] = nch_policy_imputed['% Physically Inactive']/nch_policy_imputed['% With Access to Exercise Opportunities']
nch_policy_imputed['Smokers to Exercise Opportunity Ratio'] = nch_policy_imputed['% Smokers']/nch_policy_imputed['% With Access to Exercise Opportunities']
nch_policy_imputed['Obesity to Food Environment Index Ratio'] = nch_policy_imputed['% Adults with Obesity']/nch_policy_imputed['% With Access to Exercise Opportunities']
nch_policy_imputed['Healthy Lifestyle Access'] = nch_policy_imputed['% With Access to Exercise Opportunities']*(100 - nch_policy_imputed['% Limited Access to Healthy Foods'])
nch_policy_imputed['Mental Health Providers per sqm'] = nch_policy_imputed['# Mental Health Providers']/nch_policy_imputed['AREA_SQMI']
nch_policy_imputed['Primary Care Physicians per sqm'] = nch_policy_imputed['# Primary Care Physicians']/nch_policy_imputed['AREA_SQMI']
nch_policy_imputed['Uninsured Mental Health Provider Ratio'] = nch_policy_imputed['Mental Health Provider Ratio']*nch_policy_imputed['% Uninsured Adults']/100 
nch_policy_imputed['Uninsured Primary Care provider Ratio'] = nch_policy_imputed['Primary Care Physicians Ratio']*nch_policy_imputed['% Uninsured Adults']/100 
nch_policy_imputed['Mental Health Provider Rate to Mental Distress Days Ratio'] = nch_policy_imputed['Mental Health Provider Ratio']/nch_policy_imputed['% Frequent Mental Distress']

nch_policy_imputed.drop(['# Primary Care Physicians', '# Mental Health Providers'], axis = 1)
