import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix,mean_squared_error,precision_score,recall_score,f1_score
from sklearn.metrics import classification_report , roc_curve, f1_score, accuracy_score, recall_score , roc_auc_score,make_scorer
import re
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
#loading dataset
df = pd.read_csv('features_selected.csv' , encoding_errors= 'replace')

data = df.values

# Let's scale the data first
scaler = StandardScaler()
feature_df_scaled = scaler.fit_transform(df)

#number of clusters selected 
kmeans = KMeans(n_clusters=2)
kmeans.fit(feature_df_scaled)
labels = kmeans.labels_

#dataset visualization

X = feature_df_scaled
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=2)

skf = StratifiedKFold(n_splits=10)

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=0)
X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)
print (X_train_smote.shape, y_train_smote.shape)
print (X_test.shape, y_test.shape)

#sns.set(font_scale=1.7)

# LogisticRegression Model
reg = LogisticRegression().fit(X_train_smote, y_train_smote)
y_pred = reg.predict(X_test)
print("Accuracy for LogisticRegression:",metrics.accuracy_score(y_test, (y_pred)))
print(classification_report(y_test,((y_pred))))

# compute the confusion matrix
cm = confusion_matrix(y_test,(y_pred))

#Plot the confusion matrix.
plot_ = sns.heatmap(cm/np.sum(cm), annot=True, annot_kws={'size': 20}, fmt= '0.2%')
plt.ylabel('Prediction',fontsize=17)
plt.xlabel('Actual',fontsize=17)
plt.title('Confusion Matrix for LR',fontsize=17)
plt.show()

# DecisionTreeClassifier

DT = DecisionTreeClassifier().fit(X_train_smote, y_train_smote)
y_pred = DT.predict(X_test)
print("Accuracy for DT",metrics.accuracy_score(y_test, (y_pred)))
print(classification_report(y_test,((y_pred))))

# compute the confusion matrix
cm = confusion_matrix(y_test,(y_pred))
#Plot the confusion matrix.
#Plot the confusion matrix.
plot_ = sns.heatmap(cm/np.sum(cm), annot=True, annot_kws={'size': 20}, fmt= '0.2%')
plt.ylabel('Prediction',fontsize=17)
plt.xlabel('Actual',fontsize=17)
plt.title('Confusion Matrix for DT',fontsize=17)
plt.show()

#GradientBoostingClassifier

GB = GradientBoostingClassifier().fit(X_train_smote, y_train_smote)
y_pred = GB.predict(X_test)
print("Accuracy for GB",metrics.accuracy_score(y_test, (y_pred)))
print(classification_report(y_test,((y_pred))))

# compute the confusion matrix
cm = confusion_matrix(y_test,(y_pred))
#Plot the confusion matrix.
#Plot the confusion matrix.
plot_ = sns.heatmap(cm/np.sum(cm), annot=True, annot_kws={'size': 20}, fmt= '0.2%')

plt.ylabel('Prediction',fontsize=17)
plt.xlabel('Actual',fontsize=17)
plt.title('Confusion Matrix for GB',fontsize=17)
plt.show()

#RandomForestClassifier
RF = RandomForestClassifier().fit(X_train_smote, y_train_smote)
y_pred = RF.predict(X_test)
print(confusion_matrix(y_test, (y_pred)))
acc_train_log = metrics.accuracy_score(y_test, (y_pred))
print("Accuracy for RF:",metrics.accuracy_score(y_test, (y_pred)))
print(classification_report(y_test,((y_pred))))

# compute the confusion matrix
cm = confusion_matrix(y_test,(y_pred))
#Plot the confusion matrix.
#Plot the confusion matrix.
plot_ = sns.heatmap(cm/np.sum(cm), annot=True, annot_kws={'size': 20}, fmt= '0.2%')

plt.ylabel('Prediction',fontsize=17)
plt.xlabel('Actual',fontsize=17)
plt.title('Confusion Matrix for RF',fontsize=17)
plt.show()


#ExtraTreesClassifier

extra = ExtraTreesClassifier().fit(X_train_smote, y_train_smote)
y_pred = extra.predict(X_test)
print(confusion_matrix(y_test, (y_pred)))
acc_train_log = metrics.accuracy_score(y_test, (y_pred))
print("Accuracy for ET:",metrics.accuracy_score(y_test, (y_pred)))
print(classification_report(y_test,((y_pred))))

# compute the confusion matrix
cm = confusion_matrix(y_test,(y_pred))
#Plot the confusion matrix.
#Plot the confusion matrix.
plot_ = sns.heatmap(cm/np.sum(cm), annot=True, annot_kws={'size': 20}, fmt= '0.2%')

plt.ylabel('Prediction',fontsize=17)
plt.xlabel('Actual',fontsize=17)
plt.title('Confusion Matrix for ET',fontsize=17)
plt.show()

#plotting all models ghraph in one ghraph
plt.figure()

#adaboost
abc = AdaBoostClassifier(n_estimators=50, learning_rate=1).fit(X_train_smote, y_train_smote)

# Train Adaboost Classifer
y_pred = abc.predict(X_test)
print(confusion_matrix(y_test, (y_pred)))
acc_train_log = metrics.accuracy_score(y_test, (y_pred))
print("Accuracy for AdaBoost:",metrics.accuracy_score(y_test, (y_pred)))
print(classification_report(y_test,((y_pred))))

# compute the confusion matrix
cm = confusion_matrix(y_test,(y_pred))
#Plot the confusion matrix.
#Plot the confusion matrix.
plot_ = sns.heatmap(cm/np.sum(cm), annot=True, annot_kws={'size': 20}, fmt= '0.2%')

plt.ylabel('Prediction',fontsize=17)
plt.xlabel('Actual',fontsize=17)
plt.title('Confusion Matrix for AdaBoost',fontsize=17)
plt.show()

from lightgbm import LGBMClassifier
#LGBMClassifier
lg = LGBMClassifier().fit(X_train_smote, y_train_smote)

#Predict the response for test dataset
y_pred = lg.predict(X_test)
acc_train_log = metrics.accuracy_score(y_test, (y_pred))
print("Accuracy for LGBM:",metrics.accuracy_score(y_test, (y_pred)))
print(classification_report(y_test,((y_pred))))

# compute the confusion matrix
cm = confusion_matrix(y_test,(y_pred))
#Plot the confusion matrix.
#Plot the confusion matrix.
plot_ = sns.heatmap(cm/np.sum(cm), annot=True, annot_kws={'size': 20}, fmt= '0.2%')
plt.ylabel('Prediction',fontsize=17)
plt.xlabel('Actual',fontsize=17)
plt.title('Confusion Matrix for LGBM',fontsize=17)
plt.show()

#NaiveBayes

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB().fit(X_train_smote, y_train_smote)
y_pred = nb.predict(X_test)
print(confusion_matrix(y_test, (y_pred)))
acc_train_log = metrics.accuracy_score(y_test, (y_pred))
print("Accuracy for GNB:",metrics.accuracy_score(y_test, (y_pred)))
print(classification_report(y_test,((y_pred))))

# compute the confusion matrix
cm = confusion_matrix(y_test,(y_pred))
#Plot the confusion matrix.
#Plot the confusion matrix.
plot_ = sns.heatmap(cm/np.sum(cm), annot=True, annot_kws={'size': 20}, fmt= '0.2%')

plt.ylabel('Prediction',fontsize=17)
plt.xlabel('Actual',fontsize=17)
plt.title('Confusion Matrix for GNB',fontsize=17)
plt.show()


# Add the models to the list that you want to view on the ROC plot
models = [
{
    'label': 'LR',
    'model': LogisticRegression(),
},
{
    'label': 'GB',
    'model': GradientBoostingClassifier(),
},
{
    'label': 'DT',
    'model': DecisionTreeClassifier(),
},
{
    'label': 'RF',
    'model': RandomForestClassifier(),
},
{
    'label': 'AdaBoost',
    'model': AdaBoostClassifier()
},
{
    'label': 'ET',
    'model':  ExtraTreesClassifier()
},
{
    'label': 'LGBM',
    'model':  LGBMClassifier()
},
{
    'label': 'GNB',
    'model':  GaussianNB()
}
]

# Below for loop iterates through your models listd
for m in models:
    model = m['model'] # select the model
    model.fit(X_train_smote, y_train_smote) # train the model
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
plt.xlabel('1-Specificity(False Positive Rate)', fontsize=12)
plt.ylabel('Sensitivity(True Positive Rate)', fontsize=12)
plt.title('ROC for 8-ML Models', fontsize=12)
plt.legend(loc="lower right", fontsize=12)
plt.show() # Display

