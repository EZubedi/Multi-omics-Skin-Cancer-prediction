import pandas as pd
import numpy as np
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

data = pd.read_csv('surv1.csv', encoding_errors= 'replace')

#printing number of missing values.......#
#finding overall survival probability

sns.set_theme(font_scale = 1.5)

# Create Model
cph = CoxPHFitter()
# Fit the data to train the model
cph.fit(data, 'sex', event_col='survival')
# Have a look at the significance of the features
cph.print_summary()
cph.plot()
plt.show()


## I want to see the Survival curve at the patient level.
## Random patients

d = [37659,37635]

rows_selected = data.loc[data['patients'].isin(d)]

#Plot the graph for random patients
cph.predict_survival_function(rows_selected).plot()
plt.show()