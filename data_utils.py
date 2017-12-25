import cv2
import numpy as np
import os
import random
from time import sleep
import json
from classifier import BoostedClassifier, BoostedEnsemble, CascadingBoostedEnsemble



DATA_DIRECTORY = './data/signDatabasePublicFramesOnly/'
MODEL_DIRECTORY = './models/'
ANNOTATIONS_FILE_NAME = 'allAnnotations.csv'
ANNOTATIONS_FILE_PATH = os.path.join(DATA_DIRECTORY, ANNOTATIONS_FILE_NAME)

##
# 'RAMPSPEEDADVISORY40', 'RAMPSPEEDADVISORY45', 'TURNLEFT', 'STOP', 'CURVELEFT',
# 'RAMPSPEEDADVISORY20', 'LANEENDS', 'SPEEDLIMIT25', 'SCHOOLSPEEDLIMIT25', 'SPEEDLIMIT65',
# 'SPEEDLIMIT40', 'ZONEAHEAD45', 'DIP', 'SPEEDLIMIT45', 'STOPAHEAD', 'SLOW', 'DONOTPASS',
# 'YIELDAHEAD', 'ZONEAHEAD25', 'DONOTENTER', 'KEEPRIGHT', 'TRUCKSPEEDLIMIT55', 'THRUTRAFFICMERGELEFT',
# 'RAMPSPEEDADVISORY50', 'RIGHTLANEMUSTTURN', 'THRUMERGERIGHT', 'PEDESTRIANCROSSING', 'NORIGHTTURN',
# 'SPEEDLIMIT15', 'YIELD', 'CURVERIGHT', 'TURNRIGHT', 'MERGE', 'SPEEDLIMIT30', 'ROUNDABOUT', 'SPEEDLIMIT35',
# 'INTERSECTION', 'SIGNALAHEAD', 'SPEEDLIMIT55', 'SCHOOL', 'NOLEFTTURN', 'SPEEDLIMIT50',
# 'RAMPSPEEDADVISORYURDBL', 'ADDEDLANE', 'RAMPSPEEDADVISORY35', 'SPEEDLIMITURDBL', 'THRUMERGELEFT'
##


class DataProvider:

    def __init__(self):
        self.labels = []
        self.data_map = {}

    def create_index(self):
        f = open(ANNOTATIONS_FILE_PATH)
        f.readline()
        labels = []
        for line in iter(f):
            t = TrafficSign()
            t.build_metadata(line)
            if t.annotation in self.data_map:
                self.data_map[t.annotation].append(t)
            else:
                self.data_map[t.annotation] = [t]
        for k,v in self.data_map.iteritems():
            if len(v) > 10:
                labels.append(k)
        self.labels = np.array(labels)
        f.close()

    def show_random_image(self, label):
        if label in self.data_map:
            index = random.randint(0, len(self.data_map[label]))
            t = self.data_map[label][index]
            t.show()

    def get_random_image(self, label):
        if label in self.data_map:
            index = random.randint(0, len(self.data_map[label]))
            t = self.data_map[label][index]
            return t.get_image()

    def split_data(self, p=0.5):
        training_data = {}
        testing_data = {}
        for label in self.labels:
            label_data = np.array(self.data_map[label])
            m = len(label_data)
            n = int(p * m)
            indices = np.random.permutation(m)
            train_indices = indices[0:n]
            test_indices = indices[n:]
            training_data[label] = np.array(label_data[train_indices])
            testing_data[label] = np.array(label_data[test_indices])
        return training_data, testing_data

    def get_feature_set(self, label, sizes, p=0.5):
        train, test = self.split_data(p)
        label_data_train = train[label]
        label_data_test = test[label]
        X_train_set = []
        Y_train_set = []
        train_img_sizes = []
        X_test_set = []
        Y_test_set = []
        test_img_sizes = []
        iter_count = 0
        for size in sizes:
            iter_count += 1
            hog = HOGFeatureExtractor(size=size)
            X_train = []
            X_test = []
            for sign in label_data_train:
                img = sign.get_image()
                f = hog.compute_features(img)
                X_train.append(f)
                if iter_count is 1:
                    Y_train_set.append(1)
                    train_img_sizes.append(sign.get_image_size())
            for sign in label_data_test:
                img = sign.get_image()
                f = hog.compute_features(img)
                X_test.append(f)
                if iter_count is 1:
                    Y_test_set.append(1)
                    test_img_sizes.append(sign.get_image_size())
            # randomly select data for non-label
            rem_indices = np.where([lbl != label for lbl in self.labels])
            rem_labels = self.labels[rem_indices]
            for i in range(len(label_data_train)):
                neg_label = random.choice(rem_labels)
                neg_label_sign = random.choice(self.data_map[neg_label])
                neg_img = neg_label_sign.get_image()
                f = hog.compute_features(neg_img)
                X_train.append(f)
                if iter_count is 1:
                    train_img_sizes.append(neg_label_sign.get_image_size())
                    Y_train_set.append(-1)
            for i in range(len(label_data_test)):
                neg_label = random.choice(rem_labels)
                neg_label_sign = random.choice(self.data_map[neg_label])
                neg_img = neg_label_sign.get_image()
                f = hog.compute_features(neg_img)
                X_test.append(f)
                if iter_count is 1:
                    test_img_sizes.append(neg_label_sign.get_image_size())
                    Y_test_set.append(-1)
            X_train_set.append(np.array(X_train))
            X_test_set.append(np.array(X_test))
        return X_train_set, np.array(Y_train_set), X_test_set, np.array(Y_test_set), train_img_sizes, test_img_sizes

    def get_features_for_label(self, label, size=(32, 32), p=0.5):
        train, test = self.split_data(p)
        label_data_train = train[label]
        label_data_test = test[label]
        X_train = []
        Y_train = []
        X_test = []
        Y_test = []
        hog = HOGFeatureExtractor(size=size)
        for sign in label_data_train:
            img = sign.get_image()
            f = hog.compute_features(img)
            X_train.append(f)
            Y_train.append(1)
        for sign in label_data_test:
            img = sign.get_image()
            f = hog.compute_features(img)
            X_test.append(f)
            Y_test.append(1)
        # randomly select data for non-label
        rem_indices = np.where([lbl != label for lbl in self.labels])
        rem_labels = self.labels[rem_indices]
        for i in range(len(label_data_train)):
            neg_label = random.choice(rem_labels)
            neg_label_sign = random.choice(self.data_map[neg_label])
            neg_img = neg_label_sign.get_image()
            f = hog.compute_features(neg_img)
            X_train.append(f)
            Y_train.append(-1)
        for i in range(len(label_data_test)):
            neg_label = random.choice(rem_labels)
            neg_label_sign = random.choice(self.data_map[neg_label])
            neg_img = neg_label_sign.get_image()
            f = hog.compute_features(neg_img)
            X_test.append(f)
            Y_test.append(-1)
        return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)

    def get_hue_histogram_for_label(self, label):
        images = []
        for sign in self.data_map[label]:
            if not sign.file_name.split('/')[3].startswith('vid'):
                continue
            img = sign.get_image(color=True, load_again=True)
            if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
                continue
            sign_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            images.append(sign_hsv)
        if len(images) is 0:
            return None, None
        hue_hist = cv2.calcHist(images, [0], None, [12], [0, 180])
        h = sum(hue_hist)
        hue_hist /= h
        sat_hist = cv2.calcHist(images, [1], None, [16], [0, 255])
        s = sum(sat_hist)
        sat_hist /= s
        return hue_hist, sat_hist


class TrafficSign:

    def __init__(self):
        self.annotation = None
        self.X1 = 0
        self.Y1 = 0
        self.X2 = 0
        self.Y2 = 0
        self.occluded = False
        self.another_road = False
        self.on_another_road = False
        self.image = None
        self.file_name = None

    def build_metadata(self, line):
        splits = line.split(';')
        self.file_name = os.path.join(DATA_DIRECTORY, splits[0])
        self.annotation = splits[1].upper()
        self.X1 = int(splits[3])
        self.Y1 = int(splits[2])
        self.X2 = int(splits[5])
        self.Y2 = int(splits[4])
        parts = splits[6].split(',')
        self.occluded = parts[0] == '1'
        self.on_another_road = parts[1] == '1'

    def __load_image__(self, color=False):
        img = np.array(cv2.imread(os.path.join(self.file_name)))
        if not color:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.image = np.array(img[self.X1:self.X2, self.Y1:self.Y2])

    def get_image(self, size=None, color=False, load_again=False):
        if self.image is None or load_again:
            self.__load_image__(color=color)
        if size is None:
            return self.image
        else:
            return cv2.resize(self.image, size)

    def get_image_size(self):
        return abs(self.X1 - self.X2), abs(self.Y1 - self.Y2)

    def show(self):
        self.__load_image__()
        cv2.imshow(self.annotation, self.image)  # Display the image
        cv2.waitKey()  # Wait for key stroke


class HOGFeatureExtractor:

    def __init__(self, size):
        self.feature_type = 'HOG'
        self.size = size

    def compute_features(self, image):
        img = np.copy(image)
        img = cv2.resize(img, self.size)
        block_size = (self.size[0] / 2, self.size[1] / 2)
        block_stride = (self.size[0] / 4, self.size[1] / 4)
        cell_size = block_stride
        num_bins = 9
        hog = cv2.HOGDescriptor(self.size, block_size, block_stride,
                                cell_size, num_bins)
        f = hog.compute(img)
        return np.transpose(f)[0]


def segment_image(filename):
    img = cv2.imread(filename)
    #cv2.imshow("Original image", img)  # Display image
    img_float = np.float32(img)  # Convert image from unsigned 8 bit to 32 bit float
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 1)
    # Defining the criteria ( type, max_iter, epsilon )
    # cv2.TERM_CRITERIA_EPS - stop the algorithm iteration if specified accuracy, epsilon, is reached.
    # cv2.TERM_CRITERIA_MAX_ITER - stop the algorithm after the specified number of iterations, max_iter.
    # cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER - stop the iteration when any of the above condition is met.
    # max_iter - An integer specifying maximum number of iterations.In this case it is 10
    # epsilon - Required accuracy.In this case it is 1
    # k = 50  # Number of clusters
    # ret, label, centers = cv2.kmeans(img_float, k, criteria, 50, cv2.KMEANS_RANDOM_CENTERS)
    # # apply kmeans algorithm with random centers approach
    # center = np.uint8(centers)
    # # Convert the image from float to unsigned integer
    # res = center[label.flatten()]
    # # This will flatten the label
    # res2 = res.reshape(img.shape)
    # # Reshape the image
    # cv2.imshow("K Means", res2)  # Display image
    # cv2.imwrite("1.jpg", res2)  # Write image onto disk
    # meanshift = cv2.pyrMeanShiftFiltering(img, sp=8, sr=16, maxLevel=3, termcrit=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 5, 1))
    # # Apply meanshift algorithm on to image
    # cv2.imshow("Output of meanshift", meanshift)
    # # Display image
    # cv2.imwrite("2.jpg", meanshift)
    # Write image onto disk
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Convert image from RGB to GRAY
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # # apply thresholding to convert the image to binary
    fg = cv2.erode(thresh, None, iterations=1)
    # # erode the image
    bgt = cv2.dilate(thresh, None, iterations=1)
    # # Dilate the image
    ret, bg = cv2.threshold(bgt, 1, 128, 1)
    # # Apply thresholding
    marker = cv2.add(fg, bg)
    # # Add foreground and background
    canny = cv2.Canny(gray, 110, 150)
    # Apply canny edge detector
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Finding the contors in the image using chain approximation
    marker32 = np.int32(marker)
    # converting the marker to float 32 bit
    cv2.watershed(img,marker32)
    # Apply watershed algorithm
    m = cv2.convertScaleAbs(marker32)
    ret, thresh = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Apply thresholding on the image to convert to binary image
    thresh_inv = cv2.bitwise_not(thresh)
    # Invert the thresh
    res = cv2.bitwise_and(img, img, mask=thresh)
    # Bitwise and with the image mask thresh
    res3 = cv2.bitwise_and(img, img, mask=thresh_inv)
    # Bitwise and the image with mask as threshold invert
    res4 = cv2.addWeighted(res, 1, res3, 1, 0)

    blank = np.zeros(res4.shape)
    # Take the weighted average

    final = cv2.drawContours(blank, contours, -1, (0, 255, 0), 1)
    # for contour in contours:
    #     x, y, w, h = cv2.boundingRect(contour)
    #     if 8 <= w <= 128 and 8 <= h <= 128 and 0.25 <= float(w)/float(h) <= 4.:
    #         sign = img[y:y+h, x:x+w]
    #         cv2.imshow("sign", sign)
    #         cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)

    # Draw the contours on the image with green color and pixel width is 1
    cv2.imshow("ROIs", canny)  # Display the image
    # cv2.imwrite("3.jpg", blank)  # Write the image
    cv2.waitKey()  # Wait for key stroke
    #sleep(1)
#
# # from os import listdir
# # # from os.path import isfile, join
# mypath = 'C:/Projects/CV/FinalProject/data/signDatabasePublicFramesOnly/vid0/frameAnnotations-vid_cmp2.avi_annotations/'
# # # onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
# # # img = "C:/Projects/CV/FinalProject/data/signDatabasePublicFramesOnly/vid0/frameAnnotations-vid_cmp2.avi_annotations/keepRight_1323802829.avi_image29.png"
# #
# import glob
# image_files = glob.glob(mypath + "*.png")
# for img in image_files:
#     segment_image(img)
# #

# for label in d.labels:
#     arr = d.get_hue_histogram_for_label(label)
#     print label
#     if arr is not None:
#         print ' '.join(map(str, arr))
#
# ## Boosted Ensemble with scoring
# # for label in d.labels:
# #     ensemble = BoostedEnsemble(15, label)
#     for i in range(5):
#         x_train, y_train, x_test, y_test = d.get_features_for_label(label, p=0.2)
#         ensemble.train(x_train, y_train)
#     x_train, y_train, x_test, y_test = d.get_features_for_label(label, p=0.5)
#     y_pred = ensemble.predict(x_test)
#     boost_correct = np.sum(y_pred == y_test)
#     boost_accuracy = 100. * boost_correct / len(y_test)
#     print label + ': Testing accuracy {0:.2f}%'.format(boost_accuracy)

## simple boosted classifier
# for label in d.labels:
#     x_train, y_train, x_test, y_test = d.get_features_for_label(label)
#     boost = BoostedClassifier(x_train, y_train, 15)
#     boost.train()
#     obj = boost.serialize()
#     json_str = json.dumps(obj)
#     with open(os.path.join(MODEL_DIRECTORY, label + ".json"), 'w') as f:
#         f.write(json_str)
#     good, bad = boost.evaluate()
#     boost_accuracy = 100 * float(good) / (good + bad)
#     print label + ': Training accuracy {0:.2f}%'.format(boost_accuracy)
#     y_pred = boost.predict(x_test)
#     boost_correct = np.sum(y_pred == y_test)
#     boost_accuracy = 100. * boost_correct / len(y_test)
#     print label + ': Testing accuracy {0:.2f}%'.format(boost_accuracy)


# cascading Boosted Ensemble with scoring
# for label in d.labels:
#     json_str = None
#     with open(os.path.join(MODEL_DIRECTORY, label + ".json"), 'r') as f:
#         json_str = f.read()
#     obj = json.loads(json_str)
#     ensemble = CascadingBoostedEnsemble(15, label)
#     ensemble.deserialize(obj)
#     x_train, y_train, x_test, y_test, train_img_sizes, test_image_sizes = d.get_feature_set(label, CascadingBoostedEnsemble.SIZES)
#     y_pred = ensemble.predict(x_test)
#     boost_correct = np.sum(y_pred == y_test)
#     boost_accuracy = 100. * boost_correct / len(y_test)
#     print label + ': Testing accuracy {0:.2f}%'.format(boost_accuracy)
# d = DataProvider()
# d.create_index()
# for label in d.labels:
#     ensemble = CascadingBoostedEnsemble(15, label)
#     x_train, y_train, x_test, y_test, train_img_sizes, test_image_sizes \
#         = d.get_feature_set(label, CascadingBoostedEnsemble.SIZES, p=0.8)
#     for i in range(len(x_train)):
#         ensemble.train(x_train[i], y_train)
#     # # y_pred = ensemble.predict_weighted(x_test, test_image_sizes)
#     y_pred = ensemble.predict(x_test, y_test)
#     hue_hist, sat_hist = d.get_hue_histogram_for_label(label)
#     if hue_hist is not None:
#         ensemble.hue_hist = hue_hist.tolist()
#     if sat_hist is not None:
#         ensemble.sat_hist = sat_hist.tolist()
#     obj = ensemble.serialize()
#     json_str = json.dumps(obj)
#     with open(os.path.join(MODEL_DIRECTORY, label + ".json"), 'w') as f:
#         f.write(json_str)
#     boost_correct = np.sum(y_pred == y_test)
#     boost_accuracy = 100. * boost_correct / len(y_test)
#     print label + ': Testing accuracy {0:.2f}%'.format(boost_accuracy)
#

# img = d.get_random_image('STOP')
# small_size = (32, 32)
# x = cv2.resize(img, small_size)
# block_size = (small_size[0] / 2, small_size[1] / 2)
# block_stride = (small_size[0] / 4, small_size[1] / 4)
# cell_size = block_stride
# num_bins = 9
# hog = cv2.HOGDescriptor(small_size, block_size, block_stride,
#                         cell_size, num_bins)
# val = hog.compute(x)
#
# print val
# print len(val)
# svm_params = dict( kernel_type = cv2.SVM_LINEAR,
#                     svm_type = cv2.SVM_C_SVC,
#                     C=2.67, gamma=5.383)
#
# d = DataProvider()
# d.create_index()
# for label in d.labels:
#     x_train, y_train, x_test, y_test = d.get_features_for_label(label)
#     svm = cv2.SVM()
#     svm.train(x_train, y_train, params=svm_params)
#     result = svm.predict_all(x_test)
#     correct = np.sum(result.flatten() == y_test)
#     # mask = result == y_test
#     # correct = np.count_nonzero(mask)
#     print label + ": " + str(correct * 100.0 / result.size)