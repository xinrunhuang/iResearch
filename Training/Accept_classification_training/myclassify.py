"""
  train linear classifier using CV and find the best model on dev set
"""


import sys, pickle,random
from collections import Counter
from sklearn import datasets,preprocessing,model_selection
from sklearn import linear_model,svm,neural_network,ensemble
import pickle


def get_data(features_if, scale=False, n_features = None):
  data = datasets.load_svmlight_file(features_if, n_features=n_features)
  if scale:
    new_x = preprocessing.scale(data[0].toarray())
    return new_x, data[1]
  else:
    return data[0], data[1]


def main(args, scale=False):
    if len(args) < 4:
        print("Usage:",args[0],"<train if> <dev if> <test if> <of>")
        return -1

    ###########################
    # data loading
    ###########################
    n_features = sum(1 for line in open(args[4])) #None #train_features.shape[1]
    train_features, train_labels = get_data(args[1], scale=scale,n_features=n_features)
    dev_features, dev_labels = get_data(args[2], scale=scale, n_features=n_features)
    #test_features, test_labels = get_data(args[3], scale=scale, n_features=n_features)



    ###########################
    # majority
    ###########################
    train_counter = Counter(train_labels)
    dev_counter = Counter(dev_labels)
    #test_counter = Counter(test_labels)
    print(train_counter, train_features.shape)
    print(dev_counter, dev_features.shape)
    print("Train majority: {}, Dev majority: {}".format(
      round(100.0*train_counter[0]/(train_counter[0]+train_counter[1]),3),
      round(100.0*dev_counter[0]/(dev_counter[0]+dev_counter[1]),3),))


    ###########################
    #classifiers
    ###########################
    clfs = []
    best_classifier = None
    best_v = 0
    for c in [.1, .25, .5, 1.0]:
      for clf in [
          linear_model.LogisticRegression(C=c, dual=True),
          linear_model.LogisticRegression(C=c, penalty='l1'),
          svm.SVC(kernel='rbf', C=c)]:
        clfs.append(clf)
    clfs += [
      neural_network.MLPClassifier(alpha=1),
      ensemble.AdaBoostClassifier()]
    random.shuffle(clfs)
    print('Total number of classifiers',len(clfs))

    ###########################
    # training (CV) and testing
    ###########################
    for cidx, clf in enumerate(clfs):
      scores = model_selection.cross_val_score(clf, train_features, train_labels, cv=5, n_jobs=8)
      v = sum(scores)*1.0/len(scores)
      if v > best_v:
        #print("New best v!",v*100.0,clf)
        best_classifier = clf
        best_v = v

    print("Best v:",best_v*100.0,", Best clf: ",best_classifier)
    best_classifier.fit(train_features, train_labels)
    clf=best_classifier.fit(train_features, train_labels)
    with open('D:\plp 项目\原\PeerRead-master\data\iclr_2017\\train\clf.pickle', 'wb') as f:  # python路径要用反斜杠
        pickle.dump(clf, f)  # 将模型dump进f里面


    # train
    train_y_hat = clf.predict(train_features)
    print(train_y_hat)
    train_score = 100.0 * sum(train_labels == train_y_hat) / len(train_y_hat)
    print('Train accuracy: %.2f in %d examples' %(round(train_score,3), len(train_labels)))
    # dev
    dev_y_hat = best_classifier.predict(dev_features)
    dev_score = 100.0 * sum(dev_labels == dev_y_hat) / len(dev_y_hat)
    print('Dev accuracy: %.2f in %d examples' %(round(dev_score,3), len(dev_labels)))



if __name__ == "__main__":
    test = []
    test.append('myclassify.py')
    test.append('../../data/ourdataset/train/dataset/features.svmlite_30000_w2v_True.txt')
    test.append('../../data/ourdataset/test/dataset/features.svmlite_30000_w2v_True.txt')
    test.append('../../data/ourdataset/train/dataset/best_classifier_30000_w2v_True.pkl')
    test.append('../../data/ourdataset/train/dataset/features_30000_w2v_True.dat')
    main(test)
# Best v: 78.385349963508 , Best clf:  AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
#                    n_estimators=50, random_state=None)
# [0. 0. 0. ... 0. 0. 0.]
# Train accuracy: 79.06 in 11025 examples
# Dev accuracy: 78.17 in 1269 examples
