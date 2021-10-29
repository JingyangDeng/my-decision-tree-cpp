#!/usr/bin/python3
import numpy as np
from sklearn.tree import DecisionTreeClassifier

np.random.seed(0)

clf = DecisionTreeClassifier(criterion='entropy')
train = np.loadtxt('./car-dataset/car.data.train', delimiter=' ')
test = np.loadtxt('./car-dataset/car.data.test', delimiter=' ')
test_data = test[:, :-1]
test_label = test[:, -1]
train_data = train[:, :-1]
train_label = train[:, -1]

print()
print('-------------- sklearn --------------')
print('training...')
clf.fit(train_data, train_label)
print('testing...')
pred = clf.predict(test_data)
acc = 100 * sum([1 if test_label[i] == pred[i] else 0 for i in range(len(test_label))]) / len(test_label)
print('accuracy = %.4f%%' % acc)
