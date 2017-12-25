import cv2
import glob
import data_utils
import numpy as np
import os
import random
from time import sleep
import json
from classifier import BoostedClassifier, BoostedEnsemble, CascadingBoostedEnsemble


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


class TrafficSignClassifier:

    def __init__(self):
        self.ensembles = []
        self.shape_detector = ShapeDetector()

    def init_models(self):
        model_files = glob.glob(data_utils.MODEL_DIRECTORY + "*.json")
        for model_file in model_files:
            with open(model_file, 'r') as f:
                json_str = f.read()
                obj = json.loads(json_str)
                ensemble = CascadingBoostedEnsemble(15, None)
                ensemble.deserialize(obj)
                self.ensembles.append(ensemble)

    def predict(self, image):
        # First predict if sign or not
        test_set = []
        iter_count = 0
        scores = dict()
        for size in CascadingBoostedEnsemble.SIZES:
            iter_count += 1
            hog = data_utils.HOGFeatureExtractor(size=size)
            test_features = []
            f = hog.compute_features(image)
            test_features.append(f)
            test_set.append(np.array(test_features))
        for ensemble in self.ensembles:
            scores[ensemble.label] = ensemble.compute_probability(test_set)
        return scores

    def annotate_scene(self, filename):
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, 110, 150)
        contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rects = self.get_filtered_regions(img, contours)
        #rects = get_non_overlapping_rectangles(contours)

        for rect in rects:
            x, y, w, h = rect
            sign = gray[y:y + h, x:x + w]
            scores = self.predict(sign)
            for label, score in scores.iteritems():
                if score > 0.5:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    put_text(img, (x, y), label)
        cv2.imshow("ROIs", img)  # Display the image
        cv2.waitKey()  # Wait for key stroke

    def get_filtered_regions(self, img, contours):
        hue_scores = {}
        sat_scores = {}
        rects = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if not (16 <= w <= 256 and 16 <= h <= 256 \
                    and 0.5 <= float(w) / float(h) <= 2.0 \
                    and cv2.contourArea(contour) / len(contour) > 2.):
                continue
            sign = img[y:y + h, x:x + w]
            sign_hsv = cv2.cvtColor(sign, cv2.COLOR_BGR2HSV)
            hue_hist = cv2.calcHist([sign_hsv], [0], None, [12], [0, 180])
            h_sum = sum(hue_hist)
            hue_hist /= h_sum
            sat_hist = cv2.calcHist([sign_hsv], [1], None, [16], [0, 255])
            s = sum(sat_hist)
            sat_hist /= s
            add = False
            for ensemble in self.ensembles:
                hue_dist = ensemble.get_hue_hist_distance(hue_hist)
                # hue_scores[ensemble.label] = hue_dist
                sat_dist = ensemble.get_sat_hist_distance(sat_hist)
                # sat_scores[ensemble.label] = sat_dist
                if hue_dist < 0.7 and sat_dist < 0.7:
                    add = True
            if add:
                rects.append((x, y, w, h))
        return rects


svm_params = dict(kernel_type=cv2.SVM_LINEAR,
                    svm_type = cv2.SVM_C_SVC,
                    C=2.67, gamma=5.383)


class SVMTrafficSignClassifier:

    def __init__(self):
        self.svms = {}
        self.data_provider = data_utils.DataProvider()

    def init_models(self):
        self.data_provider.create_index()
        for label in self.data_provider.labels:
            x_train, y_train, x_test, y_test = self.data_provider.get_features_for_label(label)
            svm = cv2.SVM()
            svm.train(x_train, y_train, params=svm_params)
            result = svm.predict_all(x_test)
            correct = np.sum(result.flatten() == y_test)
            print label + ": " + str(correct * 100.0 / result.size)
            self.svms[label] = svm

    def predict(self, image):
        # First predict if sign or not
        scores = dict()
        hog = data_utils.HOGFeatureExtractor((32, 32))
        f = hog.compute_features(image)
        for label, svm in self.svms.iteritems():
            scores[label] = svm.predict(f), svm.predict(f, True)
        return scores

    def annotate_scene(self, filename):
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, 110, 150)
        contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if 16 <= w <= 128 and 16 <= h <= 128 and 0.66 <= float(w) / float(h) <= 1.5:
                sign = gray[y:y + h, x:x + w]
                scores = self.predict(sign)
                for label, score in scores.iteritems():
                    if score[1] < 0.:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.imshow("ROIs", img)  # Display the image
        cv2.waitKey()  # Wait for key stroke


class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"

        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        elif len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)

            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

        # if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"

        # otherwise, we assume the shape is a circle
        else:
            shape = "circle"

        # return the name of the shape
        return shape


def get_non_overlapping_rectangles(contours):
    rects = []
    for contour in contours:
        # if len(contour) < 16 or len(contour) > 96:
        #     continue
        x, y, w, h = cv2.boundingRect(contour)
        if 16 <= w <= 256 and 16 <= h <= 256 \
                and 0.5 <= float(w) / float(h) <= 2.0 \
                and cv2.contourArea(contour) / len(contour) > 2.:
            rects.append((x, y, w, h))
    rem_list = []
    for i in range(len(rects)):
        for j in range(len(rects)):
            if i != j:
                if is_contained_in(rects[i], rects[j]):
                    rem_list.append(i)
                if is_contained_in(rects[j], rects[i]):
                    rem_list.append(j)
    ret = []
    for i in range(len(rects)):
        if i not in rem_list:
            ret.append(rects[i])
    return ret


def is_contained_in(rect_a, rect_b):
    x1, y1, w1, h1 = rect_a
    x2, y2, w2, h2 = rect_b
    if x1 <= x2 and y1 <= y2 and ((x1 + w1 >= x2 + w2 and y1 + h1 >= y2 + h2)):
        return True
    return False

def put_text(image, position, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = position
    fontScale = 0.50
    fontColor = (128, 128, 255)
    lineType = 1
    cv2.putText(image, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)

tsc = TrafficSignClassifier()
tsc.init_models()
#mypath = 'C:/Projects/CV/FinalProject/data/signDatabasePublicFramesOnly/vid3/frameAnnotations-vid_cmp2.avi_annotations/'
mypath = 'C:/Projects/CV/FinalProject/data/signDatabasePublicFramesOnly/vid0/frameAnnotations-vid_cmp2.avi_annotations/'
#mypath = "C:/Projects/CV/FinalProject/data/self_captured/"
#mypath = 'C:/Projects/CV/FinalProject/data/signDatabasePublicFramesOnly/vid4/frameAnnotations-vid_cmp2.avi_annotations/'
image_files = glob.glob(mypath + "*.png")
for img in image_files:
    tsc.annotate_scene(img)





# font = cv2.FONT_HERSHEY_SIMPLEX
#                         bottomLeftCornerOfText = x, y
#                         fontScale = 0.6
#                         fontColor = (0, 0, 0)
#                         lineType = 1
#                         cv2.putText(img, self.shape_detector.detect(contour),
#                                     bottomLeftCornerOfText,
#                                     font,
#                                     fontScale,
#                                     fontColor,
#                                     lineType)