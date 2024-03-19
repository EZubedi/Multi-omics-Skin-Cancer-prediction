#........Importing libabaries...........#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from scipy import stats
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.statistics import multivariate_logrank_test
import lifelines
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from lifelines.utils import median_survival_times

#...........Read the dataset :...............#

data = pd.read_csv('cox.csv', encoding_errors= 'replace')

#printing number of missing values.......#
print(data.isnull().sum())

#filling missing values with mean() .......#

data.fillna(data.mean(), inplace=True)

#finding overall survival probability

sns.set_theme(font_scale = 1.5)

# Age of the patient
T = data.age
# SUrvival period of patient 
E = data.survival
#creating KM object
kmf = KaplanMeierFitter()

# We next use the KaplanMeierFitter method fit() to fit the model to the data. 
kmf.fit(durations = T, event_observed = E, label = 'overall survival rate', alpha=0.05)

#printing event tables

print(kmf.event_table)

kmf.plot_survival_function()

plt.title("Kaplan Meier Curve")
plt.xlabel("Age after SKCM Diagnosis")
plt.ylabel("Survival")
plt.ylim([0,1])
plt.show()
"""
# Organize our data :
# If status = 1 , then dead = 0
# If status = 2 , then dead = 1
data.loc[data.status == 0, 'dead'] = 0
data.loc[data.status == 1, 'dead'] = 1
print (data.head())
data.to_csv('cox.csv', index=False)
sns.set_theme(font_scale = 1.5)
# kmf_m for male data.
# kmf_f for female data.
kmf_m = KaplanMeierFitter() 
kmf_f = KaplanMeierFitter() 

# Dividing data into groups :
Male = data.query("sex == 0")
Female = data.query("sex == 1")

# The 1st arg accepts an array or pd.Series of individual survival survivals
# The 2nd arg accepts an array or pd.Series that indicates if the event 
# interest (or death) occured.

kmf_m.fit(durations = Male['age'],event_observed = Male["dead"] ,label="Male")
kmf_f.fit(durations = Female['age'],event_observed = Female["dead"], label="Female")

print (kmf_m.event_table)
print (kmf_f.event_table)

print (kmf_m.predict(11))
print (kmf_f.predict(11))

print (kmf_m.survival_function_)
print (kmf_f.survival_function_)

# Plot the survival_function data :

kmf_f.survival_function_.plot()
kmf_m.plot()
plt.xlabel("Age after SKCM diagnosis")
plt.ylabel("Survival probability")
plt.title("KMF")
plt.show()

#printing commulative density
print (kmf_m.cumulative_density_)
print (kmf_f.cumulative_density_)

kmf_m.plot_cumulative_density()
kmf_f.plot_cumulative_density()

plt.xlabel("Age after SKCM diagnosis")
plt.ylabel("Survival")
plt.title("Comulative density")
plt.show()"""

#Survival analyssi with age group
ax = plt.subplot(111)
X = data[data.age<=30].survival
Y = data[data.age<=30].sex
kmf.fit(X, event_observed = Y, label = 'age <=30')
kmf.plot(ax = ax)

X = data[(data.age>30)|(data.age<=45)].survival
Y = data[(data.age>30)|(data.age<=45)].sex
kmf.fit(X, event_observed = Y, label = 'age >30 and <=45')
kmf.plot(ax = ax)

X = data[data.age>45].survival
Y = data[data.age>45].sex
kmf.fit(X, event_observed = Y, label = 'age >45')
kmf.plot(ax = ax)

plt.title("Kaplan Meier estimates by Age Factor")
plt.xlabel("Days after SKCM diagnosis")
plt.ylabel("Survival")
plt.ylim([0,1])
plt.show()

# plotting the probability of patients with respect to their disease_status_last_followup and gender 

ax = plt.subplot(111)
X = data[data.sex==0].survival
Y = data[data.sex==0].disease_status_last_followup
kmf.fit(X, event_observed = Y, label = 'Progression')
kmf.plot(ax = ax)
X = data[(data.sex==1)].survival
Y = data[(data.sex==1)].disease_status_last_followup
kmf.fit(X, event_observed = Y, label = 'Complete Remission')
kmf.plot(ax = ax)
plt.title("Disease_status_last_followup")
plt.xlabel("Days after SKCM diagnosis")
plt.ylabel("Survival probabilities")
plt.ylim([0,1])
plt.show()

#Plot the graph
d_data = data.iloc[10:15,:]
kmf.predict_survival_function(d_data).plot()
plt.show()

