import sys
import numpy as np
import pandas as pd
import sklearn.preprocessing
import sklearn.decomposition
import sklearn.cross_decomposition
import matplotlib.pyplot as plt
import seaborn as sns

## TOOLS
def encode_labels(train_y):
    print 'encode labels as one-vs-all vectors'
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(train_y)
    train_y = le.transform(train_y)
    
    lb = sklearn.preprocessing.LabelBinarizer()
    lb.fit(train_y)
    train_y = lb.transform(train_y)

    return le, lb, train_y

def transform_label(y, le, lb):
    return lb.transform(le.transform(y))

def invert_transform_label(y, le, lb):
    return le.inverse_transform(lb.inverse_transform(y))

def cca_analyze(train_X, train_y, test_X, nc):
    cca = sklearn.cross_decomposition.CCA(n_components=nc)
    cca.fit(train_X, train_y)
    trainX = cca.transform(train_X)
    testX = cca.transform(test_X)
    return cca, trainX, testX

def pca_analyze(train_X, test_X, nc):
    pca = sklearn.decomposition.PCA(n_components=nc)
    pca.fit(train_X)
    trainX = pca.transform(train_X)
    testX = pca.transform(test_X)
    return pca, trainX, testX

def plot_pca_analysis(setnumber, pca, nv):
    print 'plot component analysis'
    evals = pca.explained_variance_ratio_
    evals_cs = evals.cumsum()
    plt.plot(range(1, nc+1), evals, 'o', label='individual')
    plt.plot(range(1, nc+1), evals_cs, 'o', label='cumulative')
    plt.legend()
    plt.suptitle('PCA '+setnumber)
    plt.savefig('plots/PCA/component_ratio_'+setnumber+'.png')

def plot_new_components(setnumber, train, trainX, test, testX, mode, label='PCA'):
    print 'plot the 2 most important components with hue of label'
    X_train = pd.DataFrame(trainX, columns=['comp_'+str(x) for x in range(1, trainX.shape[1]+1)])
    X_train['status'] = train['status']
    X_test = pd.DataFrame(testX, columns=['comp_'+str(x) for x in range(1, trainX.shape[1]+1)])
    X_test['status'] = test['status']
    if mode=='overplot':
        sns.lmplot('comp_1', 'comp_2', hue='status', fit_reg=False, data=X_train, markers='o', scatter_kws={'alpha':0.5, 's':20})
        # if label=='PCA':
        #     plt.xlim(-200, 400)
        #     plt.ylim(-1.5, 1.5)
        # elif label=='CCA':
        #     plt.xlim(-5, 10)
        #     plt.ylim(-5, 5)            
        plt.savefig('plots/PCA/train_classes_'+label+'_overplot_'+setnumber+'.png')
        sns.lmplot('comp_1', 'comp_2', hue='status', fit_reg=False, data=X_test, markers='*', scatter_kws={'alpha':0.5})
        # if label=='PCA':
        #     plt.xlim(-200, 400)
        #     plt.ylim(-1.5, 1.5)
        # elif label=='CCA':
        #     plt.xlim(-5, 10)
        #     plt.ylim(-5, 5)
        plt.suptitle(label+' '+setnumber)            
        plt.savefig('plots/PCA/test_classes_'+label+'_overplot_'+setnumber+'.png')
    elif mode=='separate':
        fig, axs = plt.subplots(2,2)
        axs = axs.flatten()
        axs[0].plot(X_train.loc[X_train.status=='long', 'comp_1'], X_train.loc[X_train.status=='long', 'comp_2'], 'ob', alpha=0.2)
        axs[0].set_title('long')
        axs[1].plot(X_train.loc[X_train.status=='medium', 'comp_1'], X_train.loc[X_train.status=='medium', 'comp_2'], 'og', alpha=0.2)
        axs[1].set_title('medium')
        axs[2].plot(X_train.loc[X_train.status=='short', 'comp_1'], X_train.loc[X_train.status=='short', 'comp_2'], 'or', alpha=0.2)
        axs[2].set_title('short')
        axs[3].plot(X_train.loc[X_train.status=='urgent', 'comp_1'], X_train.loc[X_train.status=='urgent', 'comp_2'], 'om', alpha=0.2)
        axs[3].set_title('urgent')
        # if label=='PCA':
        #     for i in range(4):
        #         axs[i].set_xlim(-200, 400)
        #         axs[i].set_ylim(-1.5, 1.5)
        # elif label=='CCA':
        #     for i in range(4):
        #         axs[i].set_xlim(-5, 10)
        #         axs[i].set_ylim(-5, 5)
        plt.suptitle(label+' '+setnumber)
        plt.savefig('plots/PCA/train_classes_'+label+'_separate_'+setnumber+'.png')
        return X_train, X_test
    
#sn = str(sys.argv[1])
sn = 1
setnumber = 'FD00' + str(sn)

# read preprocessed data
print 'read data'
train = pd.read_csv('data/train_'+setnumber+'.csv')
train_RULs = np.array([train.loc[train.id==i, 'cycle'].max() for i in train.id.unique()])
test = pd.read_csv('data/test_'+setnumber+'.csv')
test_RULs = np.array([test.loc[test.id==i, 'RUL'].min() for i in train.id.unique()])
# drop the extra column
print 'drop index column'
train.drop('Unnamed: 0', 1, inplace=True)
test.drop('Unnamed: 0', 1, inplace=True)

print 'bin RUL values'
#bins = [0, 50, 125, 200, 500]
bins = [0, 100, 200, 300, 400]
status_labels = ['urgent', 'short', 'medium', 'long']
train['status'] = pd.cut(train['RUL'], bins, labels=status_labels)
test['status'] = pd.cut(test['RUL'], bins, labels=status_labels)

print 'normalize features (using MinMaxScaler)'
train_scalables = train.loc[:,train.columns.difference(['id', 'cycle', 'status'])].values
test_scalables = test.loc[:,test.columns.difference(['id', 'cycle', 'status'])].values

train_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1)).fit(train_scalables)
test_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1)).fit(test_scalables)
train_values = train_scaler.transform(train_scalables)
test_values = test_scaler.transform(test_scalables)

# scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1))
# scaler.fit(train_scalables)
# train_values = scaler.transform(train_scalables)
# test_values = scaler.transform(test_scalables)

train.loc[:,train.columns.difference(['id', 'cycle', 'status'])] = train_values
test.loc[:,test.columns.difference(['id', 'cycle', 'status'])] = test_values

train.dropna(inplace=True)
test.dropna(inplace=True)

print 'separate features and labels'
train_X = train.values[:,3:-2]
test_X = test.values[:,3:-2]
train_y = train.values[:,-1]
test_y = test.values[:,-1]

le, lb, train_y = encode_labels(train_y)
test_y = transform_label(test_y, le, lb)

print 'Component analysis'
nc = 10

pca, trainX, testX = pca_analyze(train_X, test_X, nc)
plot_pca_analysis(setnumber, pca, nc)
plot_new_components(setnumber, train, trainX, test, testX, 'overplot', label='PCA')
plot_new_components(setnumber, train, trainX, test, testX, 'separate', label='PCA')

cca, trainX, testX = cca_analyze(train_X, train_y, test_X, nc)
plot_new_components(setnumber, train, trainX, test, testX, 'overplot', label='CCA')
X_train, X_test = plot_new_components(setnumber, train, trainX, test, testX, 'separate', label='CCA')

from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import AdaBoostClassifier
classifier = OneVsRestClassifier(AdaBoostClassifier(learning_rate=0.1, n_estimators=100, random_state=1))
classifier.fit(trainX, train_y)

from sklearn import metrics
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, n_classes=5):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tick_marks = np.arange(n_classes)
    #plt.xticks(tick_marks, iris.target_names, rotation=45)
    #plt.yticks(tick_marks, iris.target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

y_pred = classifier.predict(testX)

for i in range(5):
    cm = confusion_matrix(test_y[:,i], y_pred[:,i])
    plot_confusion_matrix(cm)
    plt.title('class %d'%i)
    plt.show()

for i in range(5):    
    fpr, tpr, thresholds = metrics.roc_curve(test_y[:,i], np.array(classifier.predict_proba(testX))[:,i])
    plt.plot(fpr, tpr, label="%d features"%(i))
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    plt.show()


# print 'KNN classification'
# ns = range(1, 30, 3)
# scores = np.array([])
# for n_neighbors in ns:
#     neigh = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
#     neigh.fit(trainX, train_y)
#     print n_neighbors, neigh.score(testX, test_y)
#     scores = np.append(scores, neigh.score(testX, test_y))

# best_score = scores.max()
# print best_score
# print np.argmax(scores)
# n_neighbors = ns[np.argmax(scores)]
# neigh = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
# neigh.fit(trainX, train_y)
# pred_y = neigh.predict(testX)
# y_proba = neigh.predict_proba(testX)
# y_p = invert_transform_label(pred_y, le, lb)
# y_test = invert_transform_label(test_y, le, lb)
# X_test['KNN_prediction'] = y_p

# sns.lmplot('comp_1', 'comp_2', hue='KNN_prediction', col='status', fit_reg=False, data=X_test, markers='*', scatter_kws={'alpha':0.5})
# plt.suptitle('KNN (n_neighbors = %d)\nscore: %s'%(n_neighbors, neigh.score(testX, test_y)))
# plt.savefig('plots/PCA/KNN_'+setnumber+'.png')

# ####
# print 'GaussianNB'
# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB()
# from sklearn.neighbors import KNeighborsClassifier
# neigh = KNeighborsClassifier(n_neighbors=3)
# neigh.fit(train_X, train_y)
# pred_y = neigh.predict(test_X)

# test_y = lb.inverse_transform(test_y)
# predy = lb.inverse_transform(pred_y)
# test_y = le.inverse_transform(test_y)
# pred_y = le.inverse_transform(pred_y)
# ####

# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.preprocessing import LabelBinarizer
# from sklearn.cross_decomposition import CCA

# def plot_hyperplane(clf, min_x, max_x, linestyle, label):
#     # get the separating hyperplane
#     w = clf.coef_[0]
#     a = -w[0] / w[1]
#     xx = np.linspace(min_x - 5, max_x + 5)  # make sure the line is long enough
#     yy = a * xx - (clf.intercept_[0]) / w[1]
#     plt.plot(xx, yy, linestyle, label=label)
    
# def plot_subfigure(X, Y, subplot, title, transform):
#     if transform == "pca":
#         X = PCA(n_components=2).fit_transform(X)
#     elif transform == "cca":
#         X = CCA(n_components=2).fit(X, Y).transform(X)
#     else:
#         raise ValueError

#     min_x = np.min(X[:, 0])
#     max_x = np.max(X[:, 0])

#     min_y = np.min(X[:, 1])
#     max_y = np.max(X[:, 1])

#     classif = OneVsRestClassifier(SVC(kernel='linear'))
#     classif.fit(X, Y)

#     plt.subplot(2, 2, subplot)
#     plt.title(title)

#     zero_class = np.where(Y[:, 0])
#     one_class = np.where(Y[:, 1])
#     plt.scatter(X[:, 0], X[:, 1], s=40, c='gray', edgecolors=(0, 0, 0))
#     plt.scatter(X[zero_class, 0], X[zero_class, 1], s=160, edgecolors='b',
#                 facecolors='none', linewidths=2, label='Class 1')
#     plt.scatter(X[one_class, 0], X[one_class, 1], s=80, edgecolors='orange',
#                 facecolors='none', linewidths=2, label='Class 2')

#     plot_hyperplane(classif.estimators_[0], min_x, max_x, 'k--',
#                     'Boundary\nfor class 1')
#     plot_hyperplane(classif.estimators_[1], min_x, max_x, 'k-.',
#                     'Boundary\nfor class 2')
#     plt.xticks(())
#     plt.yticks(())

#     plt.xlim(min_x - .5 * max_x, max_x + .5 * max_x)
#     plt.ylim(min_y - .5 * max_y, max_y + .5 * max_y)
#     if subplot == 2:
#         plt.xlabel('First principal component')
#         plt.ylabel('Second principal component')
#         plt.legend(loc="upper left")
