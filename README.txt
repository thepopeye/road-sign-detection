To use the classifier, import file controller_gtsrb.py and create and instance of SVMTrafficSignClassifer.
Then initiate models by calling init_models. Then call annotate_scene with the path to the image file.
Optionally, set output_file_name to save results, and ste display to True to show results in a window.
classifier.py has classifier implementations, using some experimental work with Ensemble of AdaBoost classifiers
to make classification scale invariant.
data_utils_*.py has utilities for data processing and is used in some places by the classifier.

#*************************SAMPLE USAGE****************************************************
# tsc = SVMTrafficSignClassifier()
# tsc.init_models()
# tsc.annotate_scene('C:/Projects/CV/FinalProject/data/lara/frame_004851.jpg',
#                    output_file_name='C:/Projects/CV/FinalProject/data/lara/output/frame_004851_out.png'
#                     ,display=True)
#******************************************************************************************