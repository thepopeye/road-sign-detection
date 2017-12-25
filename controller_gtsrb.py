import cv2
import glob
import data_utils_gtsrb
import numpy as np
import os
import random
from time import sleep
import json
import data_utils_lara
from classifier import BoostedClassifier, BoostedEnsemble, CascadingBoostedEnsembleCombined, TrafficLightClassifier
import pickle
from scipy import stats
import warnings


warnings.filterwarnings("ignore")


class TrafficSignClassifier:

    def __init__(self):
        self.ensembles = []

    def init_models(self):
        model_files = glob.glob(data_utils_gtsrb.MODEL_DIRECTORY + "*.json")
        for model_file in model_files:
            with open(model_file, 'r') as f:
                json_str = f.read()
                obj = json.loads(json_str)
                ensemble = CascadingBoostedEnsembleCombined(15, None)
                ensemble.deserialize(obj)
                self.ensembles.append(ensemble)

    def predict(self, image):
        # First predict if sign or not
        test_set = []
        iter_count = 0
        scores = dict()
        for size in CascadingBoostedEnsembleCombined.SIZES:
            iter_count += 1
            hog = data_utils_gtsrb.HOGFeatureExtractor(size=size)
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
            sign = img[y:y + h, x:x + w]
            scores = self.predict(sign)
            for label, score in scores.iteritems():
                if score > 0.25:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 128, 255), 2)
                    put_text(img, (x, y), label)
        cv2.imshow("ROIs", img)  # Display the image
        cv2.waitKey()  # Wait for key stroke

    def get_filtered_regions(self, img, contours):
        hue_scores = {}
        sat_scores = {}
        rects = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if not (16 <= w <= 128 and 16 <= h <= 128 \
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
                if hue_dist < 0.5 and sat_dist < 0.5:
                    add = True
            if add:
                rects.append((x, y, w, h))
        return rects


class SVMTrafficSignClassifier:
    SIZES = [(32, 32), (48, 48), (64, 64), (96, 96), (128, 128)]

    def __init__(self):
        self.svms = []
        self.hogs = []
        self.data_provider = data_utils_gtsrb.DataProvider()
        self.data_provider.create_index()
        self.tlc = TrafficLightClassifier()

    def init_models(self):
        with open(os.path.join(data_utils_lara.MODEL_DIRECTORY, "model.json"), 'r') as f:
            j_str = f.read()
            obj = json.loads(j_str)
            self.tlc.deserialize(obj)
        for size in self.SIZES:
            self.hogs.append(data_utils_gtsrb.HOGFeatureExtractor(size=size))
            filename = "svm" + str(size[0]) + ".txt"
            with open(os.path.join(data_utils_gtsrb.MODEL_DIRECTORY, filename), 'r') as f:
                pickle_str = f.read()
                self.svms.append(pickle.loads(pickle_str))

    def predict(self, image):
        signs = []
        for i in range(len(self.svms)):
            hog = self.hogs[i]
            svm = self.svms[i]
            f = hog.compute_features(image)
            sign = svm.predict(f)
            if sign[0] == 99:
                return 99
            signs.append(sign[0])
        if len(np.unique(signs)) == len(signs):
            return max(signs)
        else:
            return stats.mode(signs).mode[0]

    def annotate_scene(self, filename, output_file_name=None, display=False):
        image = cv2.imread(filename)
        signs = self.annotate_signs(image)
        #stop_lights = self.annotate_stop_traffic_lights(image)
        #warning_lights = self.annotate_warning_traffic_lights(image)
        #go_lights = self.annotate_go_traffic_lights(image)
        #signs["STOP_LIGHT"] = stop_lights
        #signs["WARNING_LIGHT"] = warning_lights
        #signs["GO_LIGHT"] = go_lights
        for k, v in signs.iteritems():
            if k not in ["STOP_LIGHT", "WARNING_LIGHT","GO_LIGHT"]:
                for rect in v:
                    x, y, w, h = rect
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 128, 255), 2)
                    put_text(image, (x, y - 10), k)
            else:
                for center in v:
                    x, y = center
                    cv2.rectangle(image, (x - 2, y - 2), (x + 7, y + 7), (0, 0, 0), 1)
                    put_text(image, (x, y - 10), k)
        if output_file_name is not None:
            cv2.imwrite(output_file_name, img=image)
        if display:
            cv2.namedWindow('ROIs', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('ROIs', 900,600)
            cv2.imshow("ROIs", image)  # Display the image
            cv2.waitKey()  # Wait for key stroke
        return signs

    def annotate_signs(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, 110, 150)
        contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rects = get_non_overlapping_rectangles(contours)
        ret = {}
        for rect in rects:
            x, y, w, h = rect
            sign = image[y:y + h, x:x + w]
            score = self.predict(sign)
            if score != 99:
                text = data_utils_gtsrb.get_sign_name(score)
                if text in ret:
                    ret[text].append((x, y, w, h))
                else:
                    ret[text] = [(x, y, w, h)]
        return ret

    def annotate_warning_traffic_lights(self, image):
        img_copy = np.copy(image)
        hsv = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)
        img_copy[(hsv[..., 0] > 40) | (hsv[..., 0] < 10) | (hsv[..., 1] < 180)] = 0
        img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(img_gray, 110, 150)
        contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        ret = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 3 or h < 3 or float(w) / float(h) > 1.3 or float(w) / float(h) < 0.8:
                continue
            ret.append((x, y))
            # patch1 = hsv[y - h:y, x:x + w]
            # patch2 = hsv[y + h:y + 2 * h, x:x + w]
            # dist1 = self.tlc.get_hue_distance(patch1, 'BOX')
            # dist2 = self.tlc.get_hue_distance(patch2, 'BOX')
            # if dist1 < 0.7 and dist2 < 0.7 and abs(dist1 - dist2) < 0.02:
            #     ret.append((x, y))
                # cv2.rectangle(image, (x - 2, y - 2), (x + 7, y + 7), (0, 0, 0), 1)
        return ret

    def annotate_go_traffic_lights(self, image):
        img_copy = np.copy(image)
        hsv = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)
        img_copy[(hsv[..., 0] > 110) | (hsv[..., 0] < 40) | (hsv[..., 1] < 210)] = 0
        img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(img_gray, 110, 150)
        contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        ret = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 3 or h < 3 or float(w) / float(h) > 1.3 or float(w) / float(h) < 0.8:
                continue
            ret.append((x, y))
            # patch1 = hsv[y - h:y, x:x + w]
            # patch2 = hsv[y - 2 * h:y - h, x:x + w]
            # dist1 = self.tlc.get_hue_distance(patch1, 'BOX')
            # dist2 = self.tlc.get_hue_distance(patch2, 'BOX')
            # if dist1 < 0.7 and dist2 < 0.7 and abs(dist1 - dist2) < 0.02:
            #     ret.append((x,y))
                # cv2.rectangle(image, (x, y), (x + 7, y + 7), (0, 0, 0), 1)
        return ret

    def annotate_stop_traffic_lights(self, image):
        img_copy = np.copy(image)
        hsv = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)
        img_copy[(hsv[..., 0] > 40) | (hsv[..., 0] < 10) | (hsv[..., 1] < 180)] = 0
        img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(img_gray, 110, 150)
        contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        ret = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 3 or h < 3 or float(w) / float(h) > 1.3 or float(w) / float(h) < 0.8:
                continue
            ret.append((x, y))
            # patch1 = hsv[y - h:y, x:x + w]
            # patch2 = hsv[y - 2 * h:y - h, x:x + w]
            # dist1 = self.tlc.get_hue_distance(patch1, 'BOX')
            # dist2 = self.tlc.get_hue_distance(patch2, 'BOX')
            # if dist1 < 0.7 and dist2 < 0.7 and abs(dist1 - dist2) < 0.02:
            #     ret.append((x, y))
                # cv2.rectangle(image, (x - 2, y - 2), (x + 7, y + 7), (0, 0, 0), 1)
        return ret


def get_non_overlapping_rectangles(contours):
    rects = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 16 <= w <= 256 and 16 <= h <= 256 \
                and 0.8 <= float(w) / float(h) <= 1.91:
                # and cv2.contourArea(contour) / len(contour) > 2.:
            rects.append((x, y, w, h))
    rem_list = []
    for i in range(len(rects)):
        for j in range(i, len(rects)):
            if i != j:
                if is_contained_in(rects[i], rects[j]):
                    rem_list.append(i)
                # if is_contained_in(rects[j], rects[i]):
                #     rem_list.append(j)
    ret = []
    for i in range(len(rects)):
        if i not in rem_list:
            ret.append(rects[i])
    return ret


def is_contained_in(rect_a, rect_b):
    x1, y1, w1, h1 = rect_a
    x2, y2, w2, h2 = rect_b
    if x1 < x2 and y1 < y2 and x1 + w1 > x2 + w2 and y1 + h1 > y2 + h2:
        return True
    return False

def put_text(image, position, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = position
    fontScale = 0.40
    fontColor = (0, 0, 0)
    lineType = 1
    cv2.putText(image, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness=1, lineType=1)


def annotate_warning_traffic_lights(image, tlc):
    img_copy = np.copy(image)
    hsv = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)
    img_copy[(hsv[..., 0] > 40) | (hsv[..., 0] < 10) | (hsv[..., 1] < 180)] = 0
    img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(img_gray, 110, 150)
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 3 or h < 3 or float(w) / float(h) > 1.3 or float(w) / float(h) < 0.8:
            continue
        patch1 = hsv[y - h:y, x:x + w]
        patch2 = hsv[y + h:y + 2 * h, x:x + w]
        dist1 = tlc.get_hue_distance(patch1, 'BOX')
        dist2 = tlc.get_hue_distance(patch2, 'BOX')
        if dist1 < 0.7 and dist2 < 0.7 and abs(dist1 - dist2) < 0.02:
            cv2.rectangle(image, (x - 2, y - 2), (x + 7, y + 7), (0, 0, 0), 1)
    cv2.imshow("ROIs", image)  # Display the image
    cv2.waitKey()


def annotate_go_traffic_lights(image, tlc):
    img_copy = np.copy(image)
    hsv = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)
    img_copy[(hsv[..., 0] > 110) | (hsv[..., 0] < 40) | (hsv[..., 1] < 210)] = 0
    img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(img_gray, 110, 150)
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 3 or h < 3 or float(w) / float(h) > 1.3 or float(w) / float(h) < 0.8:
            continue
        patch1 = hsv[y - h:y, x:x + w]
        patch2 = hsv[y-2*h:y-h, x:x + w]
        dist1 = tlc.get_hue_distance(patch1, 'BOX')
        dist2 = tlc.get_hue_distance(patch2, 'BOX')
        if dist1 < 0.7 and dist2 < 0.7 and abs(dist1 - dist2) < 0.02:
            cv2.rectangle(image, (x, y), (x + 7, y + 7), (0, 0, 0), 1)
    cv2.imshow("ROIs", image)  # Display the image
    cv2.waitKey()


def annotate_stop_traffic_lights(image, tlc):
    img_copy = np.copy(image)
    hsv = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)
    img_copy[(hsv[..., 0] > 40) | (hsv[..., 0] < 10) | (hsv[..., 1] < 180)] = 0
    img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(img_gray, 110, 150)
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 3 or h < 3 or float(w) / float(h) > 1.3 or float(w) / float(h) < 0.8:
            continue
        patch1 = hsv[y - h:y, x:x + w]
        patch2 = hsv[y - 2 * h:y - h, x:x + w]
        dist1 = tlc.get_hue_distance(patch1, 'BOX')
        dist2 = tlc.get_hue_distance(patch2, 'BOX')
        if dist1 < 0.7 and dist2 < 0.7 and abs(dist1 - dist2) < 0.02:
            cv2.rectangle(image, (x-2, y-2), (x + 7, y + 7), (0, 0, 0), 1)


def smart_histogram(hsv):
    img = np.zeros((hsv.shape[0], hsv.shape[1], 10))
    for i in range(len(hsv)):
        for j in range(len(hsv[0])):
            count_array = np.zeros(10)
            val = update_count_array(count_array, hsv[i][j][0])
            if j > 0:
                val += img[i][j - 1]
            if i > 0:
                val += img[i - 1][j]
            if i > 0 and j > 0:
                val -= img[i-1][j-1]
            img[i][j] = val
    return img


def update_count_array(count_array, val):
    ret = np.copy(count_array)
    ret[int(val/18)] += 1
    return ret


#*************************SAMPLE USAGE***********************************
tsc = SVMTrafficSignClassifier()
tsc.init_models()
tsc.annotate_scene('C:/Projects/CV/FinalProject/data/gtsrb/00067.ppm',
                   output_file_name='C:/Projects/CV/FinalProject/data/lara/output/frame_004851_out.png'
                    ,display=True)
#******************************************************************************************
# in_path = 'C:/Projects/CV/FinalProject/data/gtsrb/'
# out_path = 'C:/Projects/CV/FinalProject/data/gtsrb/output/'
# image_files = glob.glob(in_path + "*.ppm")
# for img in image_files:
#     out_num = img.split('/')[5].split('\\')[1].split('.')[0]
#     output_file_name = out_num + '.png'
#     output_file_path = os.path.join(out_path, output_file_name)
#     print 'processed: ' + output_file_name
#     ret = tsc.annotate_scene(img, output_file_name=output_file_path, display=False)
#     tsc.data_provider.accumulate_stats(out_num, ret)
# tsc.data_provider.write_stats_to_file(os.path.join(out_path, 'result.txt'))
#
#annotate_traffic_lights('C:/Projects/CV/FinalProject/data/lara/frame_001085.jpg')