import numpy as np
from scipy.stats import norm
import cv2
from sys import maxsize

class WeakClassifier:
    """ weak classifier - threshold on the features
    Args:
        X (numpy.array): data array of flattened images
                        (row:observations, col:features) (float).
        y (numpy.array): Labels array of shape (num observations, )
    """
    def __init__(self):
        self.Xtrain = None
        self.ytrain = None
        self.idx_0 = 0
        self.idx_1 = 0
        self.threshold = 0
        self.feature = 0
        self.sign = 0
        self.weights = None

    def init(self, X, y, weights, thresh=0, feat=0, sign=1):
        self.Xtrain = np.float32(X)
        self.ytrain = np.float32(y)
        self.idx_0 = self.ytrain == -1
        self.idx_1 = self.ytrain == 1
        self.threshold = thresh
        self.feature = feat
        self.sign = sign
        self.weights = weights

    def train(self):
        # save the threshold that leads to best prediction
        tmp_signs = []
        tmp_thresholds = []

        for f in range(self.Xtrain.shape[1]):
            m0 = self.Xtrain[self.idx_0, f].mean()
            m1 = self.Xtrain[self.idx_1, f].mean()
            tmp_signs.append(1 if m0 < m1 else -1)
            tmp_thresholds.append((m0+m1)/2.0)

        tmp_errors=[]
        for f in range(self.Xtrain.shape[1]):
            tmp_result = self.weights*(tmp_signs[f]*((self.Xtrain[:,f]>tmp_thresholds[f])*2-1) != self.ytrain)
            tmp_errors.append(sum(tmp_result))

        feat = tmp_errors.index(min(tmp_errors))

        self.feature = feat
        self.threshold = tmp_thresholds[feat]
        self.sign = tmp_signs[feat]
        # -- print self.feature, self.threshold

    def predict(self, x):
        return self.sign * ((x[self.feature] > self.threshold) * 2 - 1)

    def serialize(self):
        obj = dict()
        obj["sign"] = self.sign
        obj["feature"] = self.feature
        obj["threshold"] = self.threshold
        return obj

    def deserialize(self, obj):
        self.sign = obj["sign"]
        self.feature = obj["feature"]
        self.threshold = obj["threshold"]


class BoostedClassifier:
    """Boosting classifier.

    Args:
        X (numpy.array): Data array of flattened images
                         (row:observations, col:features) (float).
        y (numpy.array): Labels array of shape (observations, ).
        num_iterations (int): number of iterations
                              (ie number of weak classifiers).

    Attributes:
        Xtrain (numpy.array): Array of flattened images (float32).
        ytrain (numpy.array): Labels array (float32).
        num_iterations (int): Number of iterations for the boosting loop.
        weakClassifiers (list): List of weak classifiers appended in each
                               iteration.
        alphas (list): List of alpha values, one for each classifier.
        num_obs (int): Number of observations.
        weights (numpy.array): Array of normalized weights, one for each
                               observation.
        eps (float): Error threshold value to indicate whether to update
                     the current weights or stop training.
    """

    def __init__(self):
        self.Xtrain = None
        self.ytrain = None
        self.num_iterations = 0
        self.weakClassifiers = []
        self.alphas = []
        self.num_obs = 0
        self.weights = None
        self.eps = 0.0001

    def init(self, X, y, num_iterations):
        self.Xtrain = np.float32(X)
        self.ytrain = np.float32(y)
        self.num_iterations = num_iterations
        self.weakClassifiers = []
        self.alphas = []
        self.num_obs = X.shape[0]
        self.weights = np.array([1.0 / self.num_obs] * self.num_obs)  # uniform weights
        self.eps = 0.0001

    def train(self):
        """Implement the for loop shown in the problem set instructions."""
        for k in range(self.num_iterations):
            w_sum = np.sum(self.weights)
            self.weights /= w_sum
            classifier = WeakClassifier()
            classifier.init(self.Xtrain, self.ytrain, self.weights)
            classifier.train()
            predictions = []
            epsilon_weight = 0.
            index = 0
            for row in self.Xtrain:
                prediction = classifier.predict(row)
                predictions.append(prediction)
                if prediction != self.ytrain[index]:
                    epsilon_weight += self.weights[index]
                index += 1
            if epsilon_weight is 0.:
                break
            alpha = 0.5*np.log((1-epsilon_weight)/epsilon_weight)
            if epsilon_weight > self.eps:
                self.alphas.append(alpha)
                self.weakClassifiers.append(classifier)
                for i in range(len(predictions)):
                    self.weights[i] = self.weights[i]*np.exp(-1*self.ytrain[i]*alpha*predictions[i])
            else:
                break

    def evaluate(self):
        """Return the number of correct and incorrect predictions.

        Use the training data (self.Xtrain) to obtain predictions. Compare
        them with the training labels (self.ytrain) and return how many
        where correct and incorrect.

        Returns:
            tuple: two-element tuple containing:
                correct (int): Number of correct predictions.
                incorrect (int): Number of incorrect predictions.
        """
        predictions = self.predict(self.Xtrain)
        correct = np.sum(predictions == self.ytrain)
        return correct, len(self.ytrain) - correct

    def predict(self, X):
        """Return predictions for a given array of observations.

        Use the alpha values stored in self.aphas and the weak classifiers
        stored in self.weakClassifiers.

        Args:
            X (numpy.array): Array of flattened images (observations).

        Returns:
            numpy.array: Predictions, one for each row in X.
        """
        predictions = np.zeros(len(X))
        for j in range(len(self.weakClassifiers)):
            curr_predictions = np.multiply(self.alphas[j], [self.weakClassifiers[j].predict(row) for row in X])
            predictions += curr_predictions
        return np.sign(predictions)

    def compute(self, X):
        """Return predictions for a given array of observations.

        Use the alpha values stored in self.aphas and the weak classifiers
        stored in self.weakClassifiers.

        Args:
            X (numpy.array): Array of flattened images (observations).

        Returns:
            numpy.array: Predictions, one for each row in X.
        """
        predictions = np.zeros(len(X))
        for j in range(len(self.weakClassifiers)):
            curr_predictions = np.multiply(self.alphas[j], [self.weakClassifiers[j].predict(row) for row in X])
            predictions += curr_predictions
        return predictions

    def serialize(self):
        obj = dict()
        wk_clf = []
        for clf in self.weakClassifiers:
            wk_clf.append(clf.serialize())
        obj["weak_classifiers"] = wk_clf
        obj["alphas"] = self.alphas
        return obj

    def deserialize(self, obj):
        self.weakClassifiers = []
        self.alphas = []
        objs = obj["weak_classifiers"]
        for clf in objs:
            wk_clf = WeakClassifier()
            wk_clf.deserialize(clf)
            self.weakClassifiers.append(wk_clf)
        self.alphas = obj["alphas"]


class BoostedEnsemble:

    def __init__(self, num_iterations, label):
        self.num_iterations = num_iterations
        self.boostedClassifiers = []
        self.label = label

    def train(self, X, Y):
        clf = BoostedClassifier()
        clf.init(X, Y, self.num_iterations)
        clf.train()
        good, bad = clf.evaluate()
        boost_accuracy = 100 * float(good) / (good + bad)
        print self.label + "-" + str(len(self.boostedClassifiers)) + ': Training accuracy {0:.2f}%'.format(boost_accuracy)
        self.boostedClassifiers.append(clf)

    def predict(self, X):
        predictions = []
        count = 0
        for clf in self.boostedClassifiers:
            prediction = clf.compute(X[count])
            predictions.append(prediction)
            count += 1
        return np.sign(np.sum(predictions, axis=0))


class CascadingBoostedEnsemble:

    SIZES = [(8, 8), (16, 16), (32, 32), (64, 64), (128, 128)]

    def __init__(self, num_iterations, label):
        self.num_iterations = num_iterations
        self.boostedClassifiers = []
        self.histograms = []
        self.mean = None
        self.cov = None
        self.hue_hist = None
        self.sat_hist = None
        self.label = label

    def train(self, X, Y):
        clf = BoostedClassifier()
        clf.init(X, Y, self.num_iterations)
        clf.train()
        good, bad = clf.evaluate()
        boost_accuracy = 100 * float(good) / (good + bad)
        print self.label + "-" + str(len(self.boostedClassifiers)) + ': Training accuracy {0:.2f}%'.format(boost_accuracy)
        self.boostedClassifiers.append(clf)

    def predict(self, X, validation_set=None):
        predictions = []
        count = 0
        for clf in self.boostedClassifiers:
            prediction = clf.compute(X[count])
            predictions.append(prediction)
            count += 1
        results = np.sign(np.sum(predictions, axis=0))
        if validation_set is not None:
            indices = np.intersect1d(np.where(results == 1), np.where(results == validation_set))
            filtered_values = np.transpose(predictions)[indices].T
            self.mean = np.mean(filtered_values, axis=1).tolist()
            self.cov = np.cov(filtered_values).tolist()
        return results

    def compute(self, X):
        predictions = []
        count = 0
        for clf in self.boostedClassifiers:
            prediction = clf.compute(X[count])
            predictions.append(prediction)
            count += 1
        return np.sum(predictions, axis=0)

    def compute_probability(self, X):
        predictions = []
        count = 0
        for clf in self.boostedClassifiers:
            prediction = clf.compute(X[count])
            predictions.append(prediction)
            count += 1
        ret = np.sum(predictions, axis=0)
        if ret <= 0:
            return 0.
        else:
            p_t = np.transpose(predictions)
            return self.mvn_p_value(p_t)

    def mvn_p_value(self, x):
        prod = 1.
        count = 0
        for val in x[0]:
            prod *= norm.cdf(val, self.mean[count], self.cov[count][count])
            count += 1
        score = np.power(prod, 1./(count + 1))
        return score

    def predict_weighted(self, X, image_sizes):
        predictions = []
        weights = self.get_weights(image_sizes)
        count = 0
        for clf in self.boostedClassifiers:
            prediction = clf.compute(X[count])
            predictions.append(prediction)
            count += 1
        return np.sign(np.average(predictions, axis=0, weights=np.transpose(weights)))

    def get_weights(self, image_sizes):
        wt_arr = []
        for image_size in image_sizes:
            weights = []
            for size in self.SIZES:
                wt = np.sqrt((size[0] - image_size[0])**2 + (size[1] - image_size[1])**2)
                weights.append(wt)
            wt_sum = sum(weights)
            weights /= wt_sum
            wt_arr.append(weights)
        return np.array(wt_arr)

    def get_hue_hist_distance(self, hist):
        if self.hue_hist is None or len(self.hue_hist) == 0 or len(self.hue_hist) != len(hist):
            return 1.0
        sum = 0.
        for i in range(len(hist)):
            px = np.sqrt(self.hue_hist[i])
            qx = np.sqrt(hist[i])
            sum += (px - qx)**2
        return np.sqrt(sum/2.)

    def get_sat_hist_distance(self, hist):
        if self.sat_hist is None or len(self.sat_hist) == 0 or len(self.sat_hist) != len(hist):
            return 1.0
        sum = 0.
        for i in range(len(hist)):
            px = np.sqrt(self.sat_hist[i])
            qx = np.sqrt(hist[i])
            sum += (px - qx)**2
        return np.sqrt(sum)

    def serialize(self):
        obj = dict()
        clfs = []
        for clf in self.boostedClassifiers:
            clfs.append(clf.serialize())
        obj['classifiers'] = clfs
        obj['label'] = self.label
        obj['mean'] = self.mean
        obj['cov'] = self.cov
        obj['hue_hist'] = self.hue_hist
        obj['sat_hist'] = self.sat_hist
        return obj

    def deserialize(self, obj):
        self.boostedClassifiers = []
        clfs = obj['classifiers']
        for clf_obj in clfs:
            clf = BoostedClassifier()
            clf.deserialize(clf_obj)
            self.boostedClassifiers.append(clf)
        self.label = obj['label']
        self.mean = obj['mean']
        self.cov = obj['cov']
        self.hue_hist = obj['hue_hist']
        self.sat_hist = obj['sat_hist']


class TrafficLightClassifier:

    def __init__(self):
        self.sat_histograms = {}
        self.hue_histograms = {}

    def train(self, images, annotation):
        hue_hist = cv2.calcHist(images, [0], None, [30], [0, 180])
        h = sum(hue_hist)
        hue_hist /= h
        sat_hist = cv2.calcHist(images, [1], None, [32], [0, 255])
        s = sum(sat_hist)
        sat_hist /= s
        self.sat_histograms[annotation] = sat_hist.flatten().tolist()
        self.hue_histograms[annotation] = hue_hist.flatten().tolist()

    def get_distance(self, patch, annotation):
        hue_hist = cv2.calcHist([patch], [0], None, [30], [0, 180])
        h = sum(hue_hist)
        hue_hist /= h
        sat_hist = cv2.calcHist([patch], [1], None, [32], [0, 255])
        s = sum(sat_hist)
        sat_hist /= s
        sat = self.sat_histograms[annotation]
        hue = self.hue_histograms[annotation]
        sat_dist = self.get_hellinger_distance(sat, sat_hist)
        hue_dist = self.get_hellinger_distance(hue, hue_hist)
        return np.sqrt((sat_dist**2 + hue_dist**2)/2.)

    def get_hue_distance(self, patch, annotation):
        hue_hist = cv2.calcHist([patch], [0], None, [30], [0, 180])
        h = sum(hue_hist)
        hue_hist /= h
        hue = self.hue_histograms[annotation]
        hue_dist = self.get_hellinger_distance(hue, hue_hist)
        return hue_dist

    def get_hellinger_distance(self, hist1, hist2):
        if len(hist1) == 0 or len(hist1) != len(hist2):
            return 1.0
        sum = 0.
        for i in range(len(hist1)):
            px = np.sqrt(hist1[i])
            qx = np.sqrt(hist2[i])
            sum += (px - qx)**2
        return np.sqrt(sum/2)

    def serialize(self):
        obj = dict()
        obj['sat'] = self.sat_histograms
        obj['hue'] = self.hue_histograms
        return obj

    def deserialize(self, obj):
        self.sat_histograms = obj['sat']
        self.hue_histograms = obj['hue']


class CascadingBoostedEnsembleCombined:

    SIZES = [(8, 8), (16, 16), (32, 32), (64, 64), (128, 128)]

    def __init__(self, num_iterations, label):
        self.num_iterations = num_iterations
        self.boostedClassifiers = []
        self.histograms = []
        self.mean = None
        self.cov = None
        self.hue_hist = None
        self.sat_hist = None
        self.label = label

    def train(self, X, Y):
        clf = BoostedClassifier()
        clf.init(X, Y, self.num_iterations)
        clf.train()
        good, bad = clf.evaluate()
        boost_accuracy = 100 * float(good) / (good + bad)
        print self.label + "-" + str(len(self.boostedClassifiers)) + ': Training accuracy {0:.2f}%'.format(boost_accuracy)
        self.boostedClassifiers.append(clf)

    def predict(self, X, validation_set=None):
        predictions = []
        count = 0
        for clf in self.boostedClassifiers:
            prediction = clf.compute(X[count])
            predictions.append(np.sign(prediction))
            count += 1
        results = np.sign(np.sum(predictions, axis=0))
        # if validation_set is not None:
        #     indices = np.intersect1d(np.where(results == 1), np.where(results == validation_set))
        #     filtered_values = np.transpose(predictions)[indices].T
        #     self.mean = np.mean(filtered_values, axis=1).tolist()
        #     self.cov = np.cov(filtered_values).tolist()
        return results

    def compute(self, X):
        predictions = []
        count = 0
        for clf in self.boostedClassifiers:
            prediction = clf.compute(X[count])
            predictions.append(prediction)
            count += 1
        return np.sum(predictions, axis=0)

    def compute_probability(self, X):
        predictions = []
        count = 0
        for clf in self.boostedClassifiers:
            prediction = clf.compute(X[count])
            predictions.append(prediction)
            count += 1
        ret = np.sum(predictions, axis=0)
        if ret <= 0:
            return 0.
        else:
            p_t = np.transpose(predictions)
            return self.mvn_p_value(p_t)

    def mvn_p_value(self, x):
        prod = 1.
        count = 0
        for val in x[0]:
            prod *= norm.cdf(val, self.mean[count], self.cov[count][count])
            count += 1
        score = np.power(prod, 1./(count + 1))
        return score

    def get_hue_hist_distance(self, hist):
        if self.hue_hist is None or len(self.hue_hist) == 0 or len(self.hue_hist) != len(hist):
            return 1.0
        sum = 0.
        for i in range(len(hist)):
            px = np.sqrt(self.hue_hist[i])
            qx = np.sqrt(hist[i])
            sum += (px - qx)**2
        return np.sqrt(sum/2.)

    def get_sat_hist_distance(self, hist):
        if self.sat_hist is None or len(self.sat_hist) == 0 or len(self.sat_hist) != len(hist):
            return 1.0
        sum = 0.
        for i in range(len(hist)):
            px = np.sqrt(self.sat_hist[i])
            qx = np.sqrt(hist[i])
            sum += (px - qx)**2
        return np.sqrt(sum)

    def serialize(self):
        obj = dict()
        clfs = []
        for clf in self.boostedClassifiers:
            clfs.append(clf.serialize())
        obj['classifiers'] = clfs
        obj['label'] = self.label
        obj['mean'] = self.mean
        obj['cov'] = self.cov
        obj['hue_hist'] = self.hue_hist
        obj['sat_hist'] = self.sat_hist
        return obj

    def deserialize(self, obj):
        self.boostedClassifiers = []
        clfs = obj['classifiers']
        for clf_obj in clfs:
            clf = BoostedClassifier()
            clf.deserialize(clf_obj)
            self.boostedClassifiers.append(clf)
        self.label = obj['label']
        self.mean = obj['mean']
        self.cov = obj['cov']
        self.hue_hist = obj['hue_hist']
        self.sat_hist = obj['sat_hist']






