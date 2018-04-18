from sklearn import svm
from sklearn.feature_extraction import DictVectorizer
from sklearn import metrics
from import_data import get_data
from timeit import default_timer as timer

def load_data():
    vec = DictVectorizer()
    file_pairs = []
    for i in range(2014, 2019):
        file_pairs.append([('data/players_' + str(i) + '.csv'), ('data/all_stars_' + str(i) + '.txt')])
    var, labels = get_data(file_pairs)
    vec = vec.fit_transform(var).toarray()
    return vec, labels

def train():
    features, labels = load_data()
    num_samples = len(labels)
    classifier = svm.SVC(kernel='linear')
    print("Training started")
    start = timer()
    # Train on the first 90% of the data
    classifier.fit(features[:9*(num_samples // 10)], labels[:9*(num_samples // 10)])
    end = timer()
    print("Training ended, took " + str(end-start) + " seconds")

    # Now predict the last 10%:
    expected = labels[9*(num_samples // 10):]
    predicted = classifier.predict(features[9*(num_samples // 10):])

    print("Classification report for classifier %s:\n%s\n"
        % (classifier, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

def main():
    train()

main()