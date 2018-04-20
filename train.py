from sklearn import ensemble
from sklearn.feature_extraction import DictVectorizer
from sklearn import metrics
from import_data import get_data, get_oversampled_data
from timeit import default_timer as timer
import tensorflow as tf
import numpy as np
from scipy import stats

def load_data():
    vec = DictVectorizer()
    file_pairs = []
    for i in range(2014, 2019):
        file_pairs.append([('data/players_' + str(i) + '.csv'), ('data/all_stars_' + str(i) + '.txt')])
    var, labels = get_data(file_pairs)
    vec = vec.fit_transform(var).toarray()
    labels = np.array(labels)
    vec = np.array(vec)
    return vec, labels

def load_oversampled_data():
    vec = DictVectorizer()
    file_pairs = []
    for i in range(2014, 2019):
        file_pairs.append([('data/players_' + str(i) + '.csv'), ('data/all_stars_' + str(i) + '.txt')])
    var, labels = get_oversampled_data(file_pairs)
    vec = vec.fit_transform(var).toarray()
    labels = np.array(labels)
    vec = np.array(vec)
    return vec, labels

def z_scorify(data):
    rectified_data = []
    for item in data.T:
        if(np.array_equal(item, item.astype(bool))):
            rectified_data.append(item)
            continue
        rectified_data.append(stats.zscore(item))
    return np.array(rectified_data).T

def train_sklearn():
    features, labels = load_data()
    num_samples = len(labels)
    classifier = ensemble.RandomForestClassifier()
    print("Training started")
    start = timer()
    # Train on the first 90% of the data
    classifier.fit(features[:9*(num_samples // 10)], labels[:9*(num_samples // 10)])
    end = timer()
    print("Training ended, took " + str(end-start) + " seconds")

    # Now predict the last 10%:
    expected = labels[9*(num_samples // 10):]
    predicted = classifier.predict(features[9*(num_samples // 10):])
    print_metrics(expected, predicted)

def train_tf_keras():
    features, labels = load_oversampled_data()
    features = z_scorify(features)
    num_samples = len(labels)
    test_labels = labels[9*(num_samples // 10):]
    test_features = features[9*(num_samples // 10):]
    train_labels = labels[:9*(num_samples // 10)]
    train_features = features[:9*(num_samples // 10)]
    batch_size = 512
    #Define model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(64, activation='relu', input_dim=features.shape[1]))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    sgd = tf.keras.optimizers.SGD(lr=0.01)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
    #Run model
    model.fit(x=train_features, y=train_labels, batch_size=batch_size, epochs=100)
    this_metrics = model.evaluate(test_features, test_labels)
    print(model.metrics_names)
    print(this_metrics)
    print(model.predict(test_features))
    preds = np.array(model.predict(test_features) > 0.5)
    print_metrics(test_labels, preds)

def train_tf():
    features, labels = load_oversampled_data()
    features = z_scorify(features)
    num_samples = len(labels)
    test_labels = labels[9*(num_samples // 10):]
    test_features = features[9*(num_samples // 10):]
    train_labels = labels[:9*(num_samples // 10)]
    train_features = features[:9*(num_samples // 10)]
    batch_size = 512
    #Define model
    W = tf.Variable(tf.random_uniform([features.shape[1], 1], dtype=tf.float32))
    x = tf.placeholder(tf.float32, shape=(None, features.shape[1]))
    m = tf.matmul(x, W)
    b = tf.Variable(tf.zeros([1]))
    a = tf.add(m, b)
    z = tf.sigmoid(a)
    predictions = tf.to_int32(z > 0.5)
    targets = tf.placeholder(tf.float32, shape=[None, 1])
    loss = tf.losses.log_loss(targets, z)
    train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
    init = tf.global_variables_initializer()
    num_epochs = len(train_labels)//batch_size
    with tf.Session() as sess:
        sess.run(init, feed_dict={x:train_features[0:batch_size]})
        for i in range(num_epochs):
            start = i*batch_size
            end = (i+1)*batch_size
            l = sess.run(loss, feed_dict={x:train_features[start:end], targets: np.expand_dims(train_labels[start:end], -1)})
            print(l)
            sess.run(train, feed_dict={x:train_features[start:end], targets: np.expand_dims(train_labels[start:end], -1)})
            if(num_epochs < 10 or i%10==0):
                print('Iteration ' + str(i) + ': ')
                values = sess.run(predictions, feed_dict={x: test_features})
                print(np.squeeze(values))
                print_metrics(test_labels, np.squeeze(values))
        sess.run(train, feed_dict={x: train_features[num_epochs*batch_size:], targets: np.expand_dims(train_labels[num_epochs*batch_size:], -1)})
        print('Final Iteration' + str(i) + ': ')
        values = sess.run(predictions, feed_dict={x: test_features})
        print(np.squeeze(values))
        print_metrics(test_labels, np.squeeze(values))
    sess.close()

def print_metrics(expected, predicted):
    print("Classification report: \n%s\n"
        % (metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
    print("Log loss:\n%s" % metrics.log_loss(expected, predicted))

def main():
    train_tf()

main()