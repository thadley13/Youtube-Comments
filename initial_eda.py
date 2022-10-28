# -*- coding: utf-8 -*-

# Imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Import data preprocessing functions from file
from vectorize_data import vectorize_data
from vec_data_no_norm import vectorize_data_no_norm

# Get X and Y data
X, Y = vectorize_data()

# Get train test split of data for initial analysis
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 13)

# Test Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, Y_train)
Y_hat = log_reg.predict(X_test)
cm_lg = confusion_matrix(Y_test, Y_hat)
cr_lg = classification_report(Y_test, Y_hat, zero_division = 0)
print(cr_lg)

# Test SVM
svm = SVC()
svm.fit(X_train, Y_train)
Y_hat = svm.predict(X_test)
cm_svm = confusion_matrix(Y_test, Y_hat)
cr_svm = classification_report(Y_test, Y_hat, zero_division = 0)
print(cr_svm)

# Test DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth = 5)
dt.fit(X_train, Y_train)
Y_hat = dt.predict(X_test)
cm_dt = confusion_matrix(Y_test, Y_hat)
cr_dt = classification_report(Y_test, Y_hat, zero_division = 0)
print(cr_dt)


# Test Random Forest
rf = RandomForestClassifier(max_features =10, max_depth = 15)
rf.fit(X_train, Y_train)
Y_hat = rf.predict(X_test)
cm_rf = confusion_matrix(Y_test, Y_hat)
cr_rf = classification_report(Y_test, Y_hat, zero_division = 0)
print(cr_rf)

# Test PCA
pca = PCA()
new_Xs = pca.fit_transform(X)
trans_df = pd.concat([pd.DataFrame(new_Xs), Y], axis = 1)
for num in reversed(range(2)):
    dat = trans_df[trans_df['CLASS'] == num]
    plt.scatter(list(dat.iloc[:,0]), list(dat.iloc[:,1]), label = '{}'.format(num))
plt.legend()
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.title('PCA Visualization Plot')
plt.show()

for num in reversed(range(2)):
    dat = trans_df[trans_df['CLASS'] == num]
    plt.scatter(list(dat.iloc[:,2]), list(dat.iloc[:,3]), label = '{}'.format(num))
plt.legend()
plt.xlabel('pc3')
plt.ylabel('pc4')
plt.title('PCA Visualization Plot')
plt.show()

pca = PCA()
new_Xs = pca.fit_transform(X)
trans_df = pd.concat([pd.DataFrame(new_Xs), Y], axis = 1)
dat0 = trans_df[trans_df['CLASS'] == 0]
dat1 = trans_df[trans_df['CLASS'] == 1]

plt.scatter(list(dat0.iloc[:,2]), list(dat0.iloc[:,3]), label = '0')
plt.legend()
plt.xlabel('pc3')
plt.ylabel('pc4')
plt.title('PCA Visualization Plot')
plt.show()

plt.scatter(list(dat1.iloc[:,2]), list(dat1.iloc[:,3]), label = '1')
plt.legend()
plt.xlabel('pc3')
plt.ylabel('pc4')
plt.title('PCA Visualization Plot')
plt.show()

# Note that the variance captured by pc1 and pc2 is not helpful for
# classification.

# Reduce to 10 features, skipping first two principal components and 
# Try fitting algorithms again.

new_X = new_Xs[:,2:12] 

# Get train test split of data for initial analysis
new_X_train, new_X_test, new_Y_train, new_Y_test = train_test_split(new_X, Y, test_size = 0.2, random_state = 13)

# Test Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(new_X_train, new_Y_train)
Y_hat = log_reg.predict(new_X_test)
cm_lg = confusion_matrix(new_Y_test, Y_hat)
cr_lg = classification_report(new_Y_test, Y_hat, zero_division = 0)
print(cr_lg)

# Test SVM
svm = SVC()
svm.fit(new_X_train, new_Y_train)
Y_hat = svm.predict(new_X_test)
cm_svm = confusion_matrix(new_Y_test, Y_hat)
cr_svm = classification_report(new_Y_test, Y_hat, zero_division = 0)
print(cr_svm)

# Test DecisionTreeClassifier
dt = DecisionTreeClassifier(max_leaf_nodes = 10)
dt.fit(new_X_train, new_Y_train)
Y_hat = dt.predict(new_X_test)
cm_dt = confusion_matrix(new_Y_test, Y_hat)
cr_dt = classification_report(new_Y_test, Y_hat, zero_division = 0)
print(cr_dt)


# Test Random Forest
rf = RandomForestClassifier(max_depth = 9, max_features = 4)
rf.fit(new_X_train, new_Y_train)
Y_hat = rf.predict(new_X_test)
cm_rf = confusion_matrix(new_Y_test, Y_hat)
cr_rf = classification_report(new_Y_test, Y_hat, zero_division = 0)
print(cr_rf)

# Don't normalize data, just text features
X_2, Y_2 = vectorize_data_no_norm(normalize = False)
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X_2, Y_2, test_size = 0.2, random_state = 13)

# Test Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X2_train, Y2_train)
Y_hat = log_reg.predict(X2_test)
cm_lg = confusion_matrix(Y2_test, Y_hat)
cr_lg = classification_report(Y2_test, Y_hat, output_dict = True, zero_division = 0)
ac_lg = cr_lg['accuracy']
print(cr_lg)

svm = SVC()
svm.fit(X2_train, Y2_train)
Y_hat = svm.predict(X2_test)
cm_svm = confusion_matrix(Y2_test, Y_hat)
cr_svm = classification_report(Y2_test, Y_hat, output_dict = True, zero_division = 0)
ac_svm = cr_svm['accuracy']
print(cr_svm)

dt = DecisionTreeClassifier(max_leaf_nodes = 10)
dt.fit(X2_train, Y2_train)
Y_hat = dt.predict(X2_test)
cm_dt = confusion_matrix(Y2_test, Y_hat)
cr_dt = classification_report(Y2_test, Y_hat, output_dict = True, zero_division = 0)
ac_rf = cr_rf['accuracy']
print(cr_dt)


# Test Random Forest
rf = RandomForestClassifier(max_features = 50)
rf.fit(X2_train, Y2_train)
Y_hat = rf.predict(X2_test)
cm_rf = confusion_matrix(Y2_test, Y_hat)
cr_rf = classification_report(Y2_test, Y_hat, output_dict = True, zero_division = 0)
ac_rf = cr_rf['accuracy']
print(cr_rf)

accuracies = [cr_lg['accuracy'],cr_svm['accuracy'],cr_dt['accuracy'],cr_rf['accuracy']]
plt.bar(['Log Reg','SVM','Dec Tree','Rand Forest'], accuracies)
plt.xlabel('ML Algorithm')
plt.ylabel('Accuracy')
plt.title('Initial Results')