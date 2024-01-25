#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:


Data= pd.read_csv('bank_data.csv')
Data


# # Univariant analysis

# In[8]:


Data.info()


# In[9]:


Data.head()


# In[10]:


#Check the 'describe' function to get a better idea of the numerical values in the data.

Data.describe(percentiles=[0.00,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90])


# In[11]:


sns.histplot(Data['expected_recovery_amount'], kde=False, color='skyblue', edgecolor='black')


# In[12]:


sns.histplot(Data['actual_recovery_amount'], kde=False, color='skyblue', edgecolor='black')


# # 2. Graphical exploratory data analysis

# In[13]:


sns.scatterplot(data=Data, x='expected_recovery_amount', y='age')
plt.xlim(0, 2000)
plt.ylim(0, 60)


# # Statistical test: Age vs. expected recovery amount

# In[16]:


from scipy import stats

# Assuming 'Data' is your DataFrame
era_900_1100 = Data.loc[(Data['expected_recovery_amount'] < 1100) & (Data['expected_recovery_amount'] >= 900)]
by_recovery_strategy = era_900_1100.groupby(['recovery_strategy'])
summary_stats = by_recovery_strategy['age'].describe().unstack()

# Perform Kruskal-Wallis test
Level_0_age = era_900_1100.loc[era_900_1100['recovery_strategy'] == "Level 0 Recovery"]['age']
Level_1_age = era_900_1100.loc[era_900_1100['recovery_strategy'] == "Level 1 Recovery"]['age']

# Use stats.kruskal to perform the Kruskal-Wallis test2 
kruskal_result = stats.kruskal(Level_0_age, Level_1_age)


# # Statistical test: sex vs. expected recovery amount

# In[17]:


# Number of customers in each category
crosstab = pd.crosstab(Data.loc[(Data['expected_recovery_amount']<1100) & 
                              (Data['expected_recovery_amount']>=900)]['recovery_strategy'], 
                       Data['sex'])
print(crosstab)

# Chi-square test
chi2_stat, p_val, dof, ex = stats.chi2_contingency(crosstab)
print(p_val)


# # Exploratory graphical analysis: recovery amount¶
# 

# In[18]:


# Scatter plot of Actual Recovery Amount vs. Expected Recovery Amount 
plt.scatter(x=Data['expected_recovery_amount'], y=Data['actual_recovery_amount'], c="g", s=2)
plt.xlim(900, 1100)
plt.ylim(0, 2000)
plt.xlabel("Expected Recovery Amount")
plt.ylabel("Actual Recovery Amount")
plt.legend(loc=2)


# # Regression modeling: no threshold

# In[19]:


import statsmodels.api as sm

# Define X and y
X = era_900_1100['expected_recovery_amount']
y = era_900_1100['actual_recovery_amount']
X = sm.add_constant(X)

# Build linear regression model
model = sm.OLS(y, X).fit()
predictions = model.predict(X)

# Print out the model summary statistics
model.summary()


# # Regression modeling: adding true threshold

# In[22]:


#Create indicator (0 or 1) for expected recovery amount >= $1000
Data['indicator_1000'] = np.where(Data['expected_recovery_amount']<1000, 0, 1)
era_900_1100 = Data.loc[(Data['expected_recovery_amount']<1100) & 
                      (Data['expected_recovery_amount']>=900)]

# Define X and y
X = era_900_1100[['expected_recovery_amount','indicator_1000']]
y = era_900_1100['actual_recovery_amount']
X = sm.add_constant(X)

# Build linear regression model
model = sm.OLS(y,X).fit()

# Print the model summary
model.summary()


# In[ ]:





# In[ ]:




