import numpy as np
import mahotas as mh
from mahotas.features import surf
import sys
import pylab as p
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn import cross_validation
from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

fileLocation = "../data/X_small_train.txt"
filenames = []
with open(fileLocation,'r') as f:
    filenames = (f.readlines())


labelfileLocation = "../data/y_small_train.txt"
labelnames = []
with open(labelfileLocation,'r') as f:
    labelnames = (f.readlines())

labels = []
features = []
alldescriptors = []
for i in range(len(filenames)):
    filenames[i] = filenames[i].replace('\n','')
    labels.append(labelnames[i])
    fileUrl = '../data/images/' + filenames[i] + '.png'
    im = mh.imread(fileUrl)
    imgrey = mh.colors.rgb2gray(im, dtype=np.uint8)
    features.append(np.concatenate([mh.features.haralick(im).ravel()]))
    surfim = imgrey
    surfim = surfim.astype(np.uint8)
    alldescriptors.append(surf.dense(surfim, spacing=1))

concatenated = np.concatenate(alldescriptors)
km = KMeans(10)
km.fit(concatenated)
print "creating surf features"
sfeatures = []
for d in alldescriptors:
    c = km.predict(d)
    sfeatures.append(np.array([np.sum(c == ci) for ci in range(15)]))


features = np.array(features) 
sfeatures = np.array(sfeatures, dtype=float)
features = sfeatures
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42, stratify=labels)
clf = Pipeline([('scaler', StandardScaler()),('classifier', OneVsRestClassifier(SVC()))])
#clf=Pipeline([('scaler', StandardScaler()),('classifier',GradientBoostingClassifier(n_estimators=10, learning_rate=1.0,max_depth=1, random_state=42))])

print "Building model"
clf.fit(X_train,y_train)
score = clf.score(X_test,y_test)
print 'Accuracy of model: %.2f%%' % (score*100.)


