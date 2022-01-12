Disclaimer: If you're reading this, this is my first readme for my github project so if it's a little messy, apologies! 
I've done my best to organize it in the little time I have during grad school. Once I get some time to learn how to do a proper readme,
expect something hopefully more useful! 

Project Title: 
Exploring the intersection of demographics, health infrastructure and mental health outcomes

Project Description:
I've been interested on the intersection of demographics, health infrastructure and mental health outcomes for some time now.  The key indicators I've used to define mental health outcomes are drug overdose death rate and suicide rate. This analysis has been done at a county level for the United States

Two datasets are used (which are uploaded to my AWS S3 bucket):
1. County Health Dataset: https://www.countyhealthrankings.org/
2. CDC's Social Vulnerability Index: https://www.atsdr.cdc.gov/placeandhealth/svi/index.html

This project has a few different aims that it looks to accomplish
1. Understand the disproportionate relationship (if any) of racial dominance in a county on key indicators of mental health
2. Use modeling approaches to both understand (explainable AI/ML) and predict whether or not a county will be at high risk for mental health events and what
health infrastructure indicators are correlated with this
3. Dashboarding to visualize the data from two datasets above

Coding Langauges Used:

I have used R and Python and linked them together using the R library reticulate. This has allowed me to do data preprocessing and exploration using Python, and model building in R. R has some useful functionality that isn't present in Python
(such as stepwise regression techniques as well as easy ways to incorporate pairwise interactions rapidly). In order to properly run the code, please first run the reticulate_package_install.Rmd code prior to running the svi_county_analysis_reticulate.Rmd

Files: 

There are a few different files relevant to this project
1. SVI_county_analysis_AWS.py - is the script I wrote in Python using the Spyder IDE for the initial analysis and exploration
2. reticulate_package_install.Rmd - this is the script used to install packages if you don't have reticulate library and the necessary tools to link Python and R together. More information about reticulate is here: https://rstudio.github.io/reticulate/
3. svi_county_analysis_reticulate.Rmd which has my combined R and Python code

Update as of 1/22/2022:
- Data has been cleaned and various features have been engineered from the original datasets
- Preliminary descriptive statistics examining the difference in suicide rates and drug overdose deaths for different racial groups has been done 
- Preliminary modeling has been done to identify counties at high or low risk and some of the relevant variables have been identified 
- Tableau dashboard development is in process (but not uploaded...yet)

