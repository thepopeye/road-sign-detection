import cv2
import numpy as np
import os
import random
from time import sleep
import json
from classifier import BoostedClassifier, BoostedEnsemble, CascadingBoostedEnsembleCombined
import glob
from sklearn.feature_extraction import image
import pickle


DATA_DIRECTORY = './data/lara/'
NEGATIVE_DATA_DIRECTORY = './data/lara/99/'
MODEL_DIRECTORY = './models/lara/'
ANNOTATIONS_FILE_NAME = 'annotations.txt'
ANNOTATIONS_FILE_PATH = os.path.join(DATA_DIRECTORY, ANNOTATIONS_FILE_NAME)


class DataProvider:

    def __init__(self):
        self.labels = []
        self.data_map = {}
        self.validation_map = {}

    def create_index(self):
        f = open(ANNOTATIONS_FILE_PATH)
        f.readline()
        labels = []
        for line in iter(f):
            t = TrafficLight()
            t.build_metadata(line)
            if t.should_ignore():
                continue
            if t.annotation in self.data_map:
                self.data_map[t.annotation].append(t)
            else:
                self.data_map[t.annotation] = [t]
            # validation map for data validation
            if t.file_path in self.validation_map:
                self.validation_map[t.file_path].append(t)
            else:
                self.validation_map[t.file_path] = [t]
        self.labels = np.array(self.data_map.keys())
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

    def get_hsv_histogram_for_label(self, label):
        images = []
        for sign in self.data_map[label]:
            img = sign.get_image()
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


class TrafficLight:

    POSITIONS = {'GO': 2, 'WARNING': 1, 'STOP': 0}

    def __init__(self):
        self.annotation = None
        self.image = None
        self.file_path = None
        self.X1 = 0
        self.X2 = 0
        self.Y1 = 0
        self.Y2 = 0
        self.Id = None

    def build_metadata(self, line):
        parts = line.split('/')
        splits = parts[1].split(' ')
        if len(splits[1]) is 3:
            self.file_path = os.path.join(DATA_DIRECTORY, 'frame_000' + str(splits[1]) + '.jpg')
        elif len(splits[1]) is 4:
            self.file_path = os.path.join(DATA_DIRECTORY, 'frame_00' + str(splits[1]) + '.jpg')
        elif len(splits[1]) is 5:
            self.file_path = os.path.join(DATA_DIRECTORY, 'frame_0' + str(splits[1]) + '.jpg')
        self.annotation = splits[9].strip('\n').strip('\'').upper()
        self.X1 = int(splits[3])
        self.Y1 = int(splits[2])
        self.X2 = int(splits[5])
        self.Y2 = int(splits[4])
        self.Id = splits[6]

    def should_ignore(self):
        return self.X1 < 0 \
               or self.X2 < 0 \
               or self.Y1 < 0 \
               or self.Y2 < 0 \
               or self.annotation == 'AMBIGUOUS'

    def __load_image__(self):
        img = np.array(cv2.imread(self.file_path))
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        self.image = np.array(img_hsv[self.X1:self.X2, self.Y1:self.Y2])

    def get_images(self, size=None):
        if self.image is None:
            self.__load_image__()
        # divide into approx 3 parts
        h = abs(self.X1 - self.X2)/3
        w = abs(self.Y1 - self.Y2)
        imgs = []
        for i in range(3):
            imgs.append(self.image[i*h:(i+1)*h, 0:w])
        ret = dict()
        indices = [0, 1, 2]
        pos = self.POSITIONS[self.annotation]
        indices.remove(pos)
        ret[self.annotation] = imgs[pos]
        ret['BOX'] = []
        for i in indices:
            ret['BOX'].append(imgs[i])
        return ret

    def get_image_size(self):
        return abs(self.X1 - self.X2), abs(self.Y1 - self.Y2)

    def show(self):
        if self.image is None:
            self.__load_image__()
        cv2.imshow(self.annotation, self.image)  # Display the image
        cv2.waitKey()  # Wait for key stroke


def get_hsv_histogram(image):
    hue_hist = cv2.calcHist([image], [0], None, [180], [0, 180])
    h = sum(hue_hist)
    hue_hist /= h
    sat_hist = cv2.calcHist([image], [1], None, [256], [0, 255])
    s = sum(sat_hist)
    sat_hist /= s
    return hue_hist, sat_hist

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
#     #annotation = d.data_map[label][0].annotation
#     ensemble = CascadingBoostedEnsembleCombined(15, label)
#     x_train, y_train, x_test, y_test \
#         = d.get_feature_set(label, CascadingBoostedEnsembleCombined.SIZES, p=0.5)
#     for i in range(len(x_train)):
#         ensemble.train(x_train[i], y_train)
#     y_pred = ensemble.predict(x_test, y_test)
#     # hue_hist, sat_hist = d.get_hsv_histogram_for_label(label)
#     # if hue_hist is not None:
#     #     ensemble.hue_hist = hue_hist.tolist()
#     # if sat_hist is not None:
#     #     ensemble.sat_hist = sat_hist.tolist()
#     obj = ensemble.serialize()
#     json_str = json.dumps(obj)
#     with open(os.path.join(MODEL_DIRECTORY, label + ".json"), 'w') as f:
#         f.write(json_str)
#     boost_correct = np.sum(y_pred == y_test)
#     boost_accuracy = 100. * boost_correct / len(y_test)
#     print label + ': Testing accuracy {0:.2f}%'.format(boost_accuracy)
# #

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
# create_negatives()
# from classifier import TrafficLightClassifier
# d = DataProvider()
# d.create_index()
# # d.show_random_image('STOP')
# box_images = []
# t = TrafficLightClassifier()
# for label in d.labels:
#     imgs = []
#     for light in d.data_map[label]:
#         light_map = light.get_images()
#         imgs.append(light_map[label])
#         box_images.extend(light_map['BOX'])
#     t.train(np.array(imgs), label)
# t.train(np.array(box_images), 'BOX')
# obj = t.serialize()
# json_str = json.dumps(obj)
# with open(os.path.join(MODEL_DIRECTORY, "model.json"), 'w') as f:
#     f.write(json_str)


# from sklearn import svm
# X_train = []
# Y_train = []
# X_test = []
# Y_test = []
# for label in d.labels:
#     x_train, y_train, x_test, y_test = d.get_features(label, size=(32, 32), p=0.8)
#     X_train.extend(x_train) # = np.append(X_train, x_train)
#     Y_train.extend(y_train) # = np.append(Y_train, y_train)
#     X_test.extend(x_test)   # = np.append(X_test,  []test)
#     Y_test.extend(y_test) # = np.append(Y_test, y_test)
# clf = svm.LinearSVC()
# clf.fit(X_train, Y_train)
# s = pickle.dumps(clf)
# with open(os.path.join(MODEL_DIRECTORY, "svm.txt"), 'w') as f:
#          f.write(s)
# result = clf.predict(X_test)
# correct = np.sum(result.flatten() == Y_test)
# print "Accuracy: " + str(correct * 100.0 / result.size)
# probs = clf.predict_proba(X_test)
# prob_per_class_dictionary = dict(zip(clf.classes_, probs))
# #results_ordered_by_probability = map(lambda x: x[0], sorted(zip(clf.classes_, probs), key=lambda x: x[1], reverse=True))
# print prob_per_class_dictionary