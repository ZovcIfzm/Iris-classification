# Tutorial source
# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

#
# Summarizing the data

print(dataset.shape)  # prints dimensions: (rows, columns)
print(dataset.head(20))  # prints first 20 rows
print(dataset.describe())  # prints statistical summary for each feature
print(dataset.groupby('class').size())  # prints size of each "class" group
# when rading from csv, one of the features was named "class".
# This function tells you how many instances had the same class, for each class
'''
#
# Visualizing the data
# subplots option causes box plots to be created for each feature.
# layout defines how the GUI displays these plots
# sharex sharey I think means each plot has the same scaling for x and y.
dataset.plot(kind='box', subplots=True, layout=(
    2, 2), sharex=False, sharey=False)  # box and whisker plots
pyplot.show()

dataset.hist()  # histogram
pyplot.show()

scatter_matrix(dataset)  # scatter plot matrix
pyplot.show()
'''
#
# Evaluate algorithms
# Split-out validation dataset
array = dataset.values
# splits into feature vector X, and labels y.
# random_state acts as the seed for a random number generator for splitting
X = array[:, 0:4]
y = array[:, 4]
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, y, test_size=0.20, random_state=1)

#
# Build models
# Spot Check Algorithms - k-fold cross validation technique
models = []
# solver defines which algorithm to use for logistic regression
# liblinear is good for small datasets
# sag and saga are faster for larger datasets
# liblinear is limited to one-versus-rest scheme for multiclass problems
# multi_class defines how to deal with multi_class probems. ovr is one vs rest scheme
# one versus reset means a "binary problem" is set over each label,
#   and it is binary loss that is minimzed for each label.
#   multinomial causes multinomial loss over the entire distribution to be minimized.
models.append(('LR', LogisticRegression(
    solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))

# SVC is support vectors classifier
# creates hyperplanes to seperate points
# gamma defines how close the program tries to fit the hyperplanes to the data
# think triple point model vs super close boundaries
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    # k-fold validation is where the dataset is split k times,
    #   for each subgroup, we have the model train on the other k-1,
    #   then validate on the particular one we're iterating through
    #   results and often the mean of the k model fit scores.

    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(
        model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

#
# Compare Algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

#
# Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

'''
# final example set
# make predictions
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)
# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
# Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
'''
