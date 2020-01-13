from sklearn import datasets
from sklearn import svm
digits = datasets.load_digits()


clf = svm.SVC(gamma=0.001, C=100)


clf.fit(digits.data[:-1], digits.target[:-1])


clf.predict(digits.data[-1:])


print(digits.target[-1:])


%matplotlib inline
import numpy as np
from matplotlib import pyplot as plt
plt.figure(figsize=(0.4,0.4))
plt.imshow(digits.images[-1], interpolation='nearest', cmap=plt.cm.binary)