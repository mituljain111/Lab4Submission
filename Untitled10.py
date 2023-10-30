#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

us_data= pd.read_csv("US_Communities.csv")

us_data


# In[2]:





# In[6]:


import numpy as np
import scipy.stats
column1 = us_data.medIncome
column2 = us_data.ViolentCrimesPerPop

scipy.stats.pearsonr(column1, column2)


# In[7]:


scipy.stats.spearmanr(column1, column2)


# In[8]:


import matplotlib.pyplot as plt
plt.scatter(column1, column2)


# In[9]:


heart_data= pd.read_csv("Heart_data.csv")
heart_data


# In[10]:


mean = heart_data.mean()
std_dev = heart_data.std()

print("Mean:")
print(mean)

print("\nStandard Deviation:")
print(std_dev)


# In[13]:


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].boxplot(heart_data.values, vert=False)
axes[0].set_title('Boxplots')
axes[0].set_xlabel('Value')
axes[0].set_yticklabels(heart_data.columns)  # Use column names as labels on the y-axis

for column in heart_data.columns:
    axes[1].hist(heart_data[column], bins=20, alpha=0.5, label=column)

axes[1].set_title('Histograms')
axes[1].set_xlabel('Value')
axes[1].legend()

axes[1].set_xlim(axes[0].get_xlim())


# In[14]:


from scipy.stats import ttest_ind

t_stat, p_value = ttest_ind(mean, std_dev)

alpha = 0.05
if p_value < alpha:
    print(f"p-value: {p_value} (Significant difference between means)")
else:
    print(f"p-value: {p_value} (No significant difference between means)")


# In[15]:


t_stat, _ = ttest_ind(column1, column2)

pooled_std = ((column1.std() ** 2 + column2.std() ** 2) / 2) ** 0.5

cohen_d = t_stat / pooled_std

print(f"Cohen's d: {cohen_d}")


# In[ ]:




