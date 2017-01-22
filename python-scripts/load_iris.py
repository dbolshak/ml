from sklearn.datasets import load_iris

iris_dataset = load_iris()

'''
The data itself is contained in the target and data fields. data contains the numeric measurements of sepal length, sepal width, petal length, and petal width in a NumPy array
'''
print("dir on iris_dataset: \n{}".format(dir(iris_dataset)))
print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))
print("Value of the DESCR key:\n{}".format(iris_dataset['DESCR'][:193] + "\n..."))
print("Target names: {}".format(iris_dataset['target_names']))
print("Feature names: \n{}".format(iris_dataset['feature_names']))
print("Type of data: {}".format(type(iris_dataset['data'])))
print("Type of target: {}".format(type(iris_dataset['target'])))

'''
The rows in the data array correspond to flowers, while the columns represent the four measurements that were taken for each flower:
'''
print("Shape of data: {}".format(iris_dataset['data'].shape))
print("Here are the feature values for the first five samples(five columns of data):\n{}".format(iris_dataset['data'][:5]))
print("Target is a one-dimensional array, with one entry per flower.Shape of target: {}".format(iris_dataset['target'].shape))
print("Target:\n{}".format(iris_dataset['target']))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'],
    iris_dataset['target'],
    random_state = 0)

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))

print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))

