import sys
from re import escape
from config import *
from svm import *
from ensembling import *
from utils import *
import warnings
import time
warnings.filterwarnings('ignore')
SCRIPT_MODE = "DEV" # change to anything else for the grading version of the script. During DEV score_scaler is always kept at 1

asli_time = 0

# Redirect stdout and stderr
# sys.stdout = open('stdout.txt', 'w')
sys.stderr = open('stderr.txt', 'w')

result_string = 'experiment,marks\n'
def grading_pipeline(model_type, config, config_name, train_dataset, val_dataset, test_dataset, total_marks = 2, f1_adj = 0.98):
    global result_string
    print(f"Grading model with config: {config_name}")

    score_scaler = 1
    try:
        val_score_reported = config[config_name]['val_score']
    except Exception as poo:
        print(f"Validation score not found for {config_name}. Assigning zero marks\Cause: {poo}", file=sys.stderr)

        result_string += f'{config_name},0\n'
        return

    del config[config_name]['val_score']

    if model_type == 'SoftMargin_QP':
        model = SoftMarginSVMQP(**config[config_name])
    elif model_type == 'RandomForest':
        model = RandomForestClassifier(**config[config_name])
    elif model_type == 'AdaBoost':
        model = AdaBoostClassifier(**config[config_name])
    else:
        print('Model Type Not Found', file=sys.stderr)
        return

    print("Attempting to fit the model")
    try:
        model.fit(train_dataset["X"], train_dataset["y"])
    except Exception as poo:
        print(f"Training Failed: {poo}", file=sys.stderr)
        result_string += f'{config_name},0\n'
        return

    ###### TESTING VALIDATION RESULTS ######
    print("Testing validation F1-score with tolerance of 0.05")
    try:
        val_results = model.predict(val_dataset['X'])
        val_score_model = val_score(val_dataset['y'], val_results)
        delta = abs(val_score_model - val_score_reported) - 0.05
        if delta >= 0:
            print(delta)
            print("Reported and validation mismatch")
            score_scaler = np.exp(-delta/0.2)
        if SCRIPT_MODE == "DEV":
            score_scaler = 1
    except Exception as poo:
        print(f"Validation result calculation failed: {poo}", file=sys.stderr)
        result_string += f'{config_name},0\n'
        return

    print("Calculating results on test dataset")
    try:
        test_score_model = val_score(test_dataset['y'], model.predict(test_dataset['X']))
        print(f"Test score on {config_name} : {test_score_model} | Validation Mismatch Penalty = {(1-score_scaler) * 100}%")
        test_score_model *= score_scaler
        marks_frac = min(1, test_score_model / f1_adj)
        result_string += f'{config_name},{total_marks * marks_frac}\n'
    except Exception as poo:
        print(f"Test result calculation failed: {poo}", file=sys.stderr)
        result_string += f'{config_name},0\n'


def run_experiment(model_type, config, pre_config, config_name, total_marks = 2, f1_adj = 0.98):
    global asli_time

    required_config_for_preprocessing = pre_config[config_name]
    train_processor = MNISTPreprocessor('./dataset/train', required_config_for_preprocessing)
    train_X, train_y = train_processor.get_all_data()
    train_X, train_y = filter_dataset(train_X, train_y, ENTRY_NUMBER_LAST_DIGIT)
    train_y = convert_labels_to_svm_labels(train_y, ENTRY_NUMBER_LAST_DIGIT)
    train_dict = {"X": train_X, "y": train_y}

    val_processor = MNISTPreprocessor('./dataset/val', required_config_for_preprocessing)
    val_X, val_y = val_processor.get_all_data()
    val_X, val_y = filter_dataset(val_X, val_y, ENTRY_NUMBER_LAST_DIGIT)
    val_y = convert_labels_to_svm_labels(val_y, ENTRY_NUMBER_LAST_DIGIT)
    val_dict = {"X": val_X, "y": val_y}

    test_processor = MNISTPreprocessor('./dataset/test', required_config_for_preprocessing)
    test_X, test_y = test_processor.get_all_data()
    test_X, test_y = filter_dataset(test_X, test_y, ENTRY_NUMBER_LAST_DIGIT)
    test_y = convert_labels_to_svm_labels(test_y, ENTRY_NUMBER_LAST_DIGIT)
    test_dict = {"X": test_X, "y": test_y}

    iska_time = time.time()
    grading_pipeline(model_type, config, config_name, train_dict, val_dict, test_dict, total_marks, f1_adj)
    asli_time += (time.time() - iska_time)

##### GRADING STARTS #####
 
# f1_adj is just a dummy value here
if __name__ == '__main__':
    # run_experiment('SoftMargin_QP', SVM_CONFIG, PRE_PROCESSING_CONFIG, "hard_margin_linear", 4, 0.98)
    # run_experiment('SoftMargin_QP', SVM_CONFIG, PRE_PROCESSING_CONFIG, "hard_margin_rbf", 6, 0.98)
    # run_experiment('SoftMargin_QP', SVM_CONFIG, PRE_PROCESSING_CONFIG, "soft_margin_linear", 8, 0.98)
    # run_experiment('SoftMargin_QP', SVM_CONFIG, PRE_PROCESSING_CONFIG, "soft_margin_rbf", 12, 0.98)
    # run_experiment('RandomForest', ENSEMBLING_CONFIG, PRE_PROCESSING_CONFIG, 'RandomForest', 10, 0.98)
    run_experiment('AdaBoost', ENSEMBLING_CONFIG, PRE_PROCESSING_CONFIG, 'AdaBoost',  10,0.98)

if asli_time >= 1200:
    result_string += "TLE,1"
else :
    result_string += "TLE,0"


print(result_string)


# Close the redirected files
# sys.stdout.close()
sys.stderr.close()



## to do:
# gridsearch and randomised search on validation data using f1 score, can use multiprocessing here as well. Read on net what parameters are best
# report & analysis.py
# well documented & clean code
# CHANGE UTILS CONSTRUCTOR TO ORIGINAL