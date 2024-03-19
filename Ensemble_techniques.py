from xgboost import XGBClassifier
import os
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import confusion_matrix,mean_squared_error,precision_score,recall_score,f1_score
from sklearn.metrics import classification_report , roc_curve, f1_score, accuracy_score, recall_score , roc_auc_score,make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import make_pipeline
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
import re
import warnings
warnings.filterwarnings("ignore")
# loading training data and reading top
import re
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


df = pd.read_csv('features_selected.csv', encoding_errors= 'replace')

df.info()

# Let's scale the data first
scaler = StandardScaler()
feature_df_scaled = scaler.fit_transform(df)
#print(feature_df_scaled.shape)
#print(feature_df_scaled)

#number of clusters selected 
kmeans = KMeans(n_clusters=2)
kmeans.fit(feature_df_scaled)
labels = kmeans.labels_
#print(kmeans.cluster_centers_.shape)

#dataset visualization

X = feature_df_scaled
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=2)
skf = StratifiedKFold(n_splits=10)

bagging = BaggingClassifier(RandomForestClassifier(max_depth = 12, min_samples_leaf=1, random_state=42),
                           n_estimators=20,
                           max_samples=0.5,
                           max_features=0.5,
                           random_state=10)

bagging.fit(X_train, y_train)
y_pred_rf = bagging.predict(X_test)

print(classification_report(y_test, y_pred_rf))

print('\033[01m             Confusion_matrix \033[0m')
cf_matrix = confusion_matrix(y_test, y_pred_rf)
plot_ = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, annot_kws={'size': 20}, fmt= '0.2%')
plt.ylabel('Prediction',fontsize=17)
plt.xlabel('Actual',fontsize=17)
plt.title('Confusion Matrix for Bagging Method',fontsize=17)
plt.show()

#Boosting
ad = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=10, learning_rate=0.01)
ad.fit(X_train, y_train)
y_pred = ad.predict(X_test)
print(classification_report(y_test, y_pred))
print('\033[01m             Confusion_matrix \033[0m')
cf_matrix = confusion_matrix(y_test, y_pred)
plot_ = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, annot_kws={'size': 20}, fmt= '0.2%')
plt.ylabel('Prediction',fontsize=17)
plt.xlabel('Actual',fontsize=17)
plt.title('Confusion Matrix for Boosting Method',fontsize=17)
plt.show()


# Voting Classifier - Multiple model ensemble

lr = LogisticRegression()
tree = GradientBoostingClassifier()
NB = GaussianNB()

Voting = VotingClassifier(
	estimators=[('lr',lr),('tree',tree),('NB',NB)], voting = 'soft')

# training all the model on the train dataset
Voting.fit(X_train, y_train)
pred = Voting.predict(X_test)
accuracy_test=[]
acc = accuracy_score(pred, y_test)
accuracy_test.append(acc)
print('Test Accuracy :\033[32m \033[01m {:.2f}% \033[30m \033[0m'.format(acc*100))
print('\033[01m              Classification_report \033[0m')
print(classification_report(y_test, pred))
print('\033[01m             Confusion_matrix \033[0m')
cf_matrix = confusion_matrix(y_test, pred)
plot_ = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, annot_kws={'size': 20}, fmt= '0.2%')
plt.ylabel('Prediction',fontsize=17)
plt.xlabel('Actual',fontsize=17)
plt.title('Confusion Matrix for Voting Method',fontsize=17)
plt.show()
print('\033[31m###################- End -###################\033[0m')

#Stacking Technique

lg =     LGBMClassifier()
clf =    AdaBoostClassifier()
tree =   ExtraTreesClassifier()

stacking = StackingClassifier(
    [ 
        ('LGBMClassifier', lg),
        ('AdaBoostClassifier', clf),
        ('ExtraTreesClassifier', tree),
])

stacking.fit(X_train, y_train)
y_pred = stacking.predict(X_test)
print(classification_report(y_test, y_pred))

# compute the confusion matrix

cm = confusion_matrix(y_test,y_pred)
#Plot the confusion matrix.
plot_ = sns.heatmap(cm/np.sum(cm), annot=True, annot_kws={'size': 20}, fmt= '0.2%')
plt.ylabel('Prediction',fontsize=20)
plt.xlabel('Actual',fontsize=20)
plt.title('Confusion Matrix for Stacking Method',fontsize=17)
plt.show()
"""
# Add the models to the list that you want to view on the ROC plot
models = [
{
    'label': 'Bagging',
    'model': BaggingClassifier(),
},
{
    'label': 'Boosting',
    'model': AdaBoostClassifier(),
},
{
    'label': 'Stacking',
    'model': StackingClassifier([ 
        ('LGBMClassifier', lg),
        ('AdaBoostClassifier', clf),
        ('ExtraTreesClassifier', tree),
]),
},
{
    'label': 'Voting',
    'model': VotingClassifier([ 
        ('LogisticRegression', lr),
        ('GradientBoostingClassifier', tree),
        ('NB', NB)
])
}
]

# Below for loop iterates through your models listd
for m in models:
    model = m['model'] # select the model
    model.fit(X_train, y_train) # train the model
    y_pred=model.predict(X_test) # predict the test data
# Compute False postive rate, and True positive rate
    fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict_proba(X_test)[:,1])
# Calculate Area under the curve to display on the plot
    auc = metrics.roc_auc_score(y_test,model.predict(X_test))
# Now, plot the computed values
    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (m['label'], auc))
# Custom settings for the plot 
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('ROC for 4-Ensemble Techniques')
plt.legend(loc="lower right")
plt.show() # Display"""

