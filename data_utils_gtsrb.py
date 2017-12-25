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

DATA_DIRECTORY = './data/gtsrb/'
NEGATIVE_DATA_DIRECTORY = './data/gtsrb/99/'
MODEL_DIRECTORY = './models/gtsrb/'
ANNOTATIONS_FILE_NAME = 'gt.txt'
ANNOTATIONS_FILE_PATH = os.path.join(DATA_DIRECTORY, ANNOTATIONS_FILE_NAME)

SIGNS = [
    ('00','speed_limit_20','prohibitory'),
    ('01','speed_limit_30','prohibitory'),
    ('02','speed_limit_50','prohibitory'),
    ('03','speed_limit_60','prohibitory'),
    ('04','speed_limit_70','prohibitory'),
    ('05','speed_limit_80','prohibitory'),
    ('06','restriction_ends_80','other'),
    ('07','speed_limit_100','prohibitory'),
    ('08','speed_limit_120','prohibitory'),
    ('09','no_overtaking','prohibitory'),
    ('10','no_overtaking_trucks','prohibitory'),
    ('11','priority_at_next_intersection','danger'),
    ('12','priority_road','other'),
    ('13','give_way','other'),
    ('14','stop','other'),
    ('15','no_traffic_both_ways','prohibitory'),
    ('16','no_trucks','prohibitory'),
    ('17','no_entry','other'),
    ('18','danger','danger'),
    ('19','bend_left','danger'),
    ('20','bend_right','danger'),
    ('21','bend','danger'),
    ('22','uneven_road','danger'),
    ('23','slippery_road','danger'),
    ('24','road_narrows','danger'),
    ('25','construction','danger'),
    ('26','traffic_signal','danger'),
    ('27','pedestrian_crossing','danger'),
    ('28','school_crossing','danger'),
    ('29','cycles_crossing','danger'),
    ('30','snow','danger'),
    ('31','animals','danger'),
    ('32','restriction_ends','other'),
    ('33','go_right','mandatory'),
    ('34','go_left','mandatory'),
    ('35','go_straight','mandatory'),
    ('36','go_right_or_straight','mandatory'),
    ('37','go_left_or_straight','mandatory'),
    ('38','keep_right','mandatory'),
    ('39','keep_left','mandatory'),
    ('40','roundabout','mandatory'),
    ('41','restriction_ends_overtaking','other'),
    ('42','restriction_ends_overtaking_trucks','other')]

CODE_SIGN_MAP = dict()


def get_sign_name(code):
    s, label, c = SIGNS[code]
    return label.upper().replace('_', ' ')


class DataProvider:

    def __init__(self):
        self.labels = []
        self.data_map = {}
        self.file_map = {}
        # for stats
        self.true_positives = []
        self.false_positives = []
        self.true_negatives = []
        self.load_gt_file()

    def accumulate_stats(self, filename, sign_dict):
        if filename not in self.file_map:
            return
        t_col = self.file_map[filename]
        found_signs = []
        for sign in t_col:
            if int(sign.code) not in sign_dict:
                self.true_negatives.append(sign.code)
        for code, center in sign_dict.iteritems():
            found = False
            for sign in t_col:
                if int(sign.code) == code and sign.is_close(center):
                    self.true_positives.append(code)
                    found = True
            if not found:
                self.false_positives.append(code)

    def load_gt_file(self):
        f = open(ANNOTATIONS_FILE_PATH)
        f.readline()
        labels = []
        for line in iter(f):
            t = GTSRBTrafficSign()
            t.build_metadata(line)
            if t.file_name in self.file_map:
                self.file_map[t.file_name].append(t)
            else:
                self.file_map[t.file_name] = [t]

    def write_stats_to_file(self, filename):
        tp = ','.join(map(str, self.true_positives))
        fp = ','.join(map(str, self.false_positives))
        tn = ','.join(map(str, self.true_negatives))
        f = open(filename,mode='w')
        f.write(tp)
        f.write('\n')
        f.write(fp)
        f.write('\n')
        f.write(tn)
        f.close()

    def create_index(self):
        for sign in SIGNS:
            code, label, category = sign
            data_path = os.path.join(DATA_DIRECTORY, code)
            image_files = glob.glob(data_path + "/*.ppm")
            if len(image_files) < 10:
                continue
            for file_path in image_files:
                t = GTSRBTrafficSign()
                t.annotation = label
                t.code = code
                t.file_path = file_path
                if t.code in self.data_map:
                    self.data_map[t.code].append(t)
                else:
                    self.data_map[t.code] = [t]
        image_files = glob.glob(NEGATIVE_DATA_DIRECTORY + "/*.png")
        for img_file in image_files:
            t = GTSRBTrafficSign()
            t.file_path = img_file
            t.code = '99'
            t.annotation = 'NEGATIVE'
            if t.code in self.data_map:
                self.data_map[t.code].append(t)
            else:
                self.data_map[t.code] = [t]
        self.labels = np.array(self.data_map.keys())

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
        X_test_set = []
        Y_test_set = []
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
            for sign in label_data_test:
                img = sign.get_image()
                f = hog.compute_features(img)
                X_test.append(f)
                if iter_count is 1:
                    Y_test_set.append(1)
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
                    Y_train_set.append(-1)
            for i in range(len(label_data_test)):
                neg_label = random.choice(rem_labels)
                neg_label_sign = random.choice(self.data_map[neg_label])
                neg_img = neg_label_sign.get_image()
                f = hog.compute_features(neg_img)
                X_test.append(f)
                if iter_count is 1:
                    Y_test_set.append(-1)
            X_train_set.append(np.array(X_train))
            X_test_set.append(np.array(X_test))

        # histograms for H and S
        X_train_f1 = []
        X_train_f2 = []
        X_test_f1 = []
        X_test_f2 = []
        for sign in label_data_train:
            img = sign.get_image()
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            f1, f2 = get_hsv_histogram(img_hsv)
            X_train_f1.append(f1.T[0])
            X_train_f2.append(f2.T[0])
        for sign in label_data_test:
            img = sign.get_image()
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            f1, f2 = get_hsv_histogram(img_hsv)
            X_test_f1.append(f1.T[0])
            X_test_f2.append(f2.T[0])
        # randomly select data for non-label
        rem_indices = np.where([lbl != label for lbl in self.labels])
        rem_labels = self.labels[rem_indices]
        for i in range(len(label_data_train)):
            neg_label = random.choice(rem_labels)
            neg_label_sign = random.choice(self.data_map[neg_label])
            neg_img = neg_label_sign.get_image()
            img_hsv = cv2.cvtColor(neg_img, cv2.COLOR_BGR2HSV)
            f1, f2 = get_hsv_histogram(img_hsv)
            X_train_f1.append(f1.T[0])
            X_train_f2.append(f2.T[0])
        for i in range(len(label_data_test)):
            neg_label = random.choice(rem_labels)
            neg_label_sign = random.choice(self.data_map[neg_label])
            neg_img = neg_label_sign.get_image()
            img_hsv = cv2.cvtColor(neg_img, cv2.COLOR_BGR2HSV)
            f1, f2 = get_hsv_histogram(img_hsv)
            X_test_f1.append(f1.T[0])
            X_test_f2.append(f2.T[0])
        X_train_set.append(np.array(X_train_f1))
        X_train_set.append(np.array(X_train_f2))
        X_test_set.append(np.array(X_test_f1))
        X_test_set.append(np.array(X_test_f2))
        return X_train_set, np.array(Y_train_set), X_test_set, np.array(Y_test_set)

    def get_features(self, label, size=(32, 32), p=0.5):
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
            # f1, f2 = get_hsv_histogram(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            # np.append(f, f1)
            # np.append(f, f2)
            X_train.append(f)
            Y_train.append(int(sign.code))
        for sign in label_data_test:
            img = sign.get_image()
            f = hog.compute_features(img)
            # f1, f2 = get_hsv_histogram(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            # np.append(f, f1)
            # np.append(f, f2)
            X_test.append(f)
            Y_test.append(int(sign.code))
        return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)

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
            f1, f2 = get_hsv_histogram(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            np.append(f, f1)
            np.append(f, f2)
            X_train.append(f)
            Y_train.append(1)
        for sign in label_data_test:
            img = sign.get_image()
            f = hog.compute_features(img)
            f1, f2 = get_hsv_histogram(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            np.append(f, f1)
            np.append(f, f2)
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
            f1, f2 = get_hsv_histogram(cv2.cvtColor(neg_img, cv2.COLOR_BGR2HSV))
            np.append(f, f1)
            np.append(f, f2)
            X_train.append(f)
            Y_train.append(-1)
        for i in range(len(label_data_test)):
            neg_label = random.choice(rem_labels)
            neg_label_sign = random.choice(self.data_map[neg_label])
            neg_img = neg_label_sign.get_image()
            f = hog.compute_features(neg_img)
            f1, f2 = get_hsv_histogram(cv2.cvtColor(neg_img, cv2.COLOR_BGR2HSV))
            np.append(f, f1)
            np.append(f, f2)
            X_test.append(f)
            Y_test.append(-1)
        return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)

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


class GTSRBTrafficSign:

    def __init__(self):
        self.annotation = None
        self.code = None
        self.image = None
        self.file_path = None
        self.file_name = None
        self.category = None
        self.X1 = 0
        self.Y1 = 0
        self.X2 = 0
        self.Y2 = 0

    # ImgNo#.ppm;#leftCol#;##topRow#;#rightCol#;#bottomRow#;#ClassID#
    def build_metadata(self, line):
        splits = line.split(';')
        self.file_name = splits[0].split('.')[0]
        self.code = splits[5].strip('\n')
        self.X1 = int(splits[2])
        self.Y1 = int(splits[1])
        self.X2 = int(splits[4])
        self.Y2 = int(splits[3])

    def is_close(self, center):
        return np.allclose(center, ((self.X1 + self.X2)/2, (self.Y1 + self.Y2)/2), rtol=0., atol=10.)

    def __load_image__(self):
        self.image = np.array(cv2.imread(self.file_path))

    def get_image(self, size=None):
        if self.image is None:
            self.__load_image__()
        if size is None:
            return self.image
        else:
            return cv2.resize(self.image, size)

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


def create_negatives():
    mypath = 'C:/Projects/CV/FinalProject/data/gtsrb/'
    # mypath = "C:/Projects/CV/FinalProject/data/self_captured/"
    image_files = glob.glob(mypath + "*.ppm")
    count = 0
    for img_file in image_files:
        img = cv2.imread(img_file)
        patch = image.extract_patches_2d(img, patch_size=(32, 32), max_patches=4)
        cv2.imwrite(os.path.join(NEGATIVE_DATA_DIRECTORY, str(count) + ".png"), patch[0])
        count += 1


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
d = DataProvider()
d.create_index()
for label in d.labels:
    #annotation = d.data_map[label][0].annotation
    ensemble = CascadingBoostedEnsembleCombined(15, label)
    x_train, y_train, x_test, y_test \
        = d.get_feature_set(label, CascadingBoostedEnsembleCombined.SIZES, p=0.8)
    for i in range(len(x_train)):
        ensemble.train(x_train[i], y_train)
    y_pred = ensemble.predict(x_test, y_test)
    # hue_hist, sat_hist = d.get_hsv_histogram_for_label(label)
    # if hue_hist is not None:
    #     ensemble.hue_hist = hue_hist.tolist()
    # if sat_hist is not None:
    #     ensemble.sat_hist = sat_hist.tolist()
    # obj = ensemble.serialize()
    # json_str = json.dumps(obj)
    # with open(os.path.join(MODEL_DIRECTORY, label + ".json"), 'w') as f:
    #     f.write(json_str)
    boost_correct = np.sum(y_pred == y_test)
    boost_accuracy = 100. * boost_correct / len(y_test)
    print label + ': Testing accuracy {0:.2f}%'.format(boost_accuracy)
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

# d = DataProvider()
# d.create_index()
# from sklearn import svm
# X_train = []
# Y_train = []
# X_test = []
# Y_test = []
# for label in d.labels:
#     x_train, y_train, x_test, y_test = d.get_features(label, size=(96, 96), p=0.8)
#     X_train.extend(x_train) # = np.append(X_train, x_train)
#     Y_train.extend(y_train) # = np.append(Y_train, y_train)
#     X_test.extend(x_test)   # = np.append(X_test,  []test)
#     Y_test.extend(y_test) # = np.append(Y_test, y_test)
# clf = svm.LinearSVC()
# clf.fit(X_train, Y_train)
# s = pickle.dumps(clf)
# with open(os.path.join(MODEL_DIRECTORY, "svm96.txt"), 'w') as f:
#          f.write(s)
# result = clf.predict(X_test)
# correct = np.sum(result.flatten() == Y_test)
# print "Accuracy: " + str(correct * 100.0 / result.size)