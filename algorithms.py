# Load libraries
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.linear_model import LogisticRegression

# Load dataset
url = "data_folder/modified_HR_data.csv"
names = ['satisfaction_level','last_evaluation','number_project',
         'average_montly_hours','time_spend_company','Work_accident',
         'promotion_last_5years','salary','left']
dataset = pandas.read_csv(url)

array = dataset.values
X = array[:,0:8]
Y = array[:,8]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
seed = 8
scoring = 'accuracy'

# Spot Check Algorithms
models = []
#models.append(('LR', LogisticRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('NB', GaussianNB()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVM', SVC()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    #kfold = model_selection.KFold(n_splits=10, random_state=seed)
    kfold = model_selection.KFold(n_splits=10)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

"""
# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
"""
print()
# Make predictions on validation dataset
gnb = GaussianNB()
gnb.fit(X_train, Y_train)
predictions = gnb.predict(X_validation)
print('Naive Bayes: ' + str(100*round(accuracy_score(Y_validation, predictions), 2)) + "%")
print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))
print()

dtc = DecisionTreeClassifier()
dtc.fit(X_train, Y_train)
predictions = dtc.predict(X_validation)
print('Decision Tree(CART): ' + str(100*round(accuracy_score(Y_validation, predictions), 2)) + "%")
print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))
print()

knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print('K-Nearest Neighbors: ' + str(100*round(accuracy_score(Y_validation, predictions), 2)) + "%")
print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))
print()

svc = SVC()
svc.fit(X_train, Y_train)
predictions = svc.predict(X_validation)
print('Support Vector Machine: ' + str(100*round(accuracy_score(Y_validation, predictions), 2)) + "%")
print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))
