#http://archive.ics.uci.edu/ml/machine-learning-databases/00222/
import numpy as np
from scipy.stats import skew
import scipy.stats as stats
from scipy.stats import skew
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV, ElasticNetCV, LogisticRegression
from sklearn.metrics import mean_squared_error, make_scorer
from IPython.display import display
from sklearn import preprocessing


plt.rc("font", size=14)
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

#load the data and drop all the 
data= pd.read_csv('../input/bank-additional-full.csv', sep=";");
data=data.dropna(axis=0) 
data= data.replace({'education': {'basic.4y': 'basic', 'basic.6y': 'basic', 'basic.9y': 'basic'} } );
data_binary= data.replace({'y': {'yes': 1, 'no': 0} } );


#Figure 1: Unitary and binary data analysis
plt.figure( figsize= [7, 7])
plt.subplot(221)
sns.countplot(x='y',data=data, palette='hls')
plt.text(0.1, 20000, r'$Imbalanced\ classes$', fontsize=10,  color=[0, 0, 0]);

plt.subplot(222)
job_call= data_binary.groupby("job").count()["y"]
job_y= data_binary.groupby("job").sum()["y"]
job_rate= job_y/job_call
plt.bar(job_y.index, job_rate)
plt.xticks(rotation=90)
#pd.crosstab(data.job,data.y).plot(kind='bar')
#plt.title('Purchase Frequency')
plt.xlabel('Job')
plt.ylabel('Probability of Purchase')

plt.subplot(223)
education_call= data_binary.groupby("education").count()["y"]
education_y= data_binary.groupby("education").sum()["y"]
education_rate= education_y/education_call
plt.bar(education_y.index, education_rate)
plt.xticks(rotation=90)
#pd.crosstab(data.education,data.y).plot(kind='bar')
#plt.title('Purchase Frequency')
plt.xlabel('Education')
plt.ylabel('Probability of Purchase')

plt.subplot(224)
month_call= data_binary.groupby("month").count()["y"]
month_y= data_binary.groupby("month").sum()["y"]
month_rate= month_y/month_call
plt.bar(month_y.index, month_rate)
plt.xticks(rotation=90)
#pd.crosstab(data.month,data.y).plot(kind='bar')
#plt.title('Purchase Frequency')
plt.xlabel('Month')
plt.ylabel('Probability of Purchase')

plt.tight_layout()
plt.savefig("../output/deposit_data-analysis.pdf", format='pdf');
plt.show()



#Create dummy variables.
cat_vars=[feature for feature in data_binary.columns if data_binary[feature].dtype=="O" ]
data_cat=data_binary[ cat_vars].copy();
data_drop=data_binary.drop( cat_vars, axis=1);
data_cat=pd.get_dummies(data_cat);
data_dummy=pd.concat([data_drop, data_cat], axis=1);

#Balance the two categories: y=0; y=1;
from imblearn.over_sampling import SMOTE
X= data_dummy.loc[:, data_dummy.columns !='y'];
y= data_dummy.loc[:, 'y'];

os=SMOTE(random_state=0);
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=0);

os_data_X, os_data_y= os.fit_sample(X_train, y_train);
os_data_X= pd.DataFrame( data= os_data_X, columns= X_train.columns );
os_data_y= pd.DataFrame( data= os_data_y, columns=['y'] );


#Logistic Regression Model Fitting
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

X_train_rem, X_val_rem, y_train_rem, y_val_rem = train_test_split(os_data_X, os_data_y, test_size=0.3, random_state=0)
logreg = LogisticRegression(solver='lbfgs' )
logreg.fit(X_train_rem, y_train_rem)

y_pred = logreg.predict(X_val_rem)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_val_rem, y_val_rem)))

#Figure 2: preliminary prediction of logistic regression

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

plt.figure( figsize= [8, 4])
plt.subplot(121)
logit_roc_auc = roc_auc_score(y_val_rem, logreg.predict(X_val_rem) )
fpr, tpr, thresholds = roc_curve(y_val_rem, logreg.predict_proba(X_val_rem)[:,1])
#plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot(fpr, tpr)
plt.text(0, 0.3, r'$Logistic\ Regression:\ Area= {:.2f}$'.format(logit_roc_auc), fontsize=10,  color=[0, 0, 0]);

plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.subplot(122)
# Plot important coefficients
coefs = pd.Series( logreg.coef_.ravel(), index =X_train_rem.columns.values)
print("picked " + str( np.sum(coefs != 0)) + " features and eliminated the other " +        str(np.sum(coefs == 0)) + " features")
imp_coefs = pd.concat([coefs.sort_values().head(10), coefs.sort_values().tail(10)])
#imp_coefs= coefs.sort_values( ascending= False)
imp_coefs.plot(kind = "barh")
plt.title("Coefficients in the regression")
plt.tight_layout()


plt.savefig("../output/Regression_result.pdf", format='pdf');
plt.show()

from sklearn.metrics import classification_report
print(classification_report(y_val_rem, y_pred))

