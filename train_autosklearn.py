import os
import pickle
from datetime import datetime
from pathlib import Path
from shutil import copyfile
import numpy as np
import pandas as pd
import autosklearn.classification
import sklearn.metrics
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.externals import joblib
import argparse

cv_folds = 10  # CV is by default stratifiedKFold for auto-sklearn
random_state = 43
#max_time_secs = 162000  # how long to run auto-ml in seconds (default = 3600)
max_eval_time_secs = 300 # how many seconds for a single call to a model (fitting/eval) (default = 360)
population_size = 50 #200

DATA_PATH = Path('./data/')
MODELS_PATH = Path('./models')
MODELS_PATH.mkdir(exist_ok=True)
AUTOSKLEARN_PATH = Path(MODELS_PATH / 'autosklearn')
AUTOSKLEARN_PATH.mkdir(exist_ok=True)
AUTOSKLEARN_TOPMODELS = AUTOSKLEARN_PATH / 'best_models'
AUTOSKLEARN_TOPMODELS.mkdir(exist_ok=True)
AUTOSKLEARN_ENSEMBLES = AUTOSKLEARN_PATH / 'ensembles'
AUTOSKLEARN_ENSEMBLES.mkdir(exist_ok=True)
runscripts_dir = AUTOSKLEARN_PATH / 'run_scripts'
runscripts_dir.mkdir(exist_ok=True)
model_logs_path = AUTOSKLEARN_PATH / 'model_logs'
model_logs_path.mkdir(exist_ok=True)
RESULTS_PATH = Path('./results')
RESULTS_PATH.mkdir(exist_ok=True)


granular_label_dict = {'currentMildDistress': 'amber', 'followupOk': 'amber', 'pastDistress': 'amber', 'underserved': 'amber',
                  'crisis': 'crisis', 'allClear': 'green', 'followupBye': 'green', 'supporting': 'green', 'angryWithForumMember': 'red',
                  'angryWithReachout': 'red', 'currentAcuteDistress': 'red', 'followupWorse': 'red'}

numerical_granular_label_dict = {0: 2, 1:3, 2:3, 3:1, 4:3, 5:0, 6:2, 7:0, 8:3, 9:0, 10:2, 11:0}


# need to modify scoring function to take into account labels converted to ints with labelencoder
def get_macroF1(y_true, y_pred):
    return sklearn.metrics.f1_score(y_true, y_pred, labels=[0, 1, 3], average="macro")


def get_macroF1_from_granular(y_true, y_pred, label_dict=numerical_granular_label_dict):
    # labels 0, 6, 10 are equal to green label and thus left out from macroF1
    y_true = [label_dict[x] for x in y_true]
    y_pred = [label_dict[x] for x in y_pred]
    return sklearn.metrics.f1_score(y_true, y_pred, labels=[0,1,3], average="macro")

macroF1MinusGreen = autosklearn.metrics.make_scorer(name='macroF1MinusGreen', score_func=get_macroF1)
macroF1FromGranular = autosklearn.metrics.make_scorer(name='macroF1FromGranular', score_func=get_macroF1_from_granular)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--features', help='define which features will be used for the model', type=str)
    parser.add_argument('-l', '--labels', help='define whether to use standard label (label) or granular label (granular)', type=str)
    parser.add_argument('-t', '--time', help='amount of time to run autosklearn in days', type=float)
    args = parser.parse_args()
    feats = args.features.rstrip('.csv')
    label = args.labels
    time_days = args.time
    print(f'feats are: {feats}.csv')
    print(f'label is: {label}')

    max_time_secs = (time_days * 24 * 60 * 60) - 3600 # subtract an  hour to make sure that final wrap up of script can still finish with given time alloted on SCC
    max_time_secs = int(max_time_secs)

    START_TIME = datetime.now().isoformat()
    print(f'Start time is: {START_TIME}')

    autosklearn_name = 'autosklearn_exported_pipeline.feats.' + str(feats) + '.label.' + str(label) + '.time.' + str(START_TIME) + '.py'
    run_log = 'run_' + autosklearn_name

    print("autosklearn runscript written to:", runscripts_dir / run_log)
    copyfile(os.path.realpath(__file__), runscripts_dir / run_log)

    # Read in features
    try:
        features = pd.read_csv(DATA_PATH / 'features' / (feats + '.csv'))
    except FileNotFoundError:
        print('You did not specify the correct features file to use, or those features have not yet been created')
        print(f'You specified: {feats}.csv')

    # define correct scoring function and load up correct labels column depending on label argument
    if label == 'label':
        score_function = macroF1MinusGreen
        post_info = pd.read_csv(DATA_PATH / 'processed' / 'all_posts_data.csv', usecols=['post_id', 'label', 'predict_me'])
        test_labels = pd.read_csv(DATA_PATH/'raw'/'clpsych17-test-labels.tsv', sep='\t',
                                header=None, usecols=[0, 1], names=['post_id', 'label'])

    elif label == 'granular':
        score_function = macroF1FromGranular
        post_info = pd.read_csv(DATA_PATH / 'processed' / 'all_posts_data.csv',
                                usecols=['post_id', 'granular_label', 'predict_me'])
        post_info.rename(columns={'granular_label': 'label'}, inplace=True)
        test_labels = pd.read_csv(DATA_PATH/'raw'/'clpsych17-test-labels.tsv', sep='\t',
                                header=None, usecols=[0, 2], names=['post_id', 'granular_label'])
        test_labels.rename(columns={'granular_label': 'label'}, inplace=True)

    else:
            print('labels argument should be either "label" or "granular"')

    # merge in the correct labels/predict_me to features for train and test sets
    features = features.merge(post_info, on='post_id',
                              how='left').set_index('post_id')

    train_df = features[features.predict_me == False].drop('predict_me', axis=1)
    test_df = features[features.predict_me == True].drop('predict_me', axis=1)

    y_train = train_df.pop('label')
    X_train = train_df

    # need to encode labels as ints for autosklearn
    le = preprocessing.LabelEncoder()
    y_train = le.fit_transform(y_train)

    test_df = test_df.drop('label', axis=1).merge(test_labels, on='post_id').set_index('post_id')
    y_test = test_df.pop('label')
    # make sure to encode test labels using same transformation as train labels
    y_test = le.transform(y_test)
    X_test = test_df

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=max_time_secs,
        per_run_time_limit=max_eval_time_secs,
        ensemble_size = population_size,
        tmp_folder= model_logs_path / f'autosklearn_{feats}_{label}_{START_TIME}', # folder to store configuration output and log files
        output_folder = AUTOSKLEARN_TOPMODELS / f'autosklearn_out_{feats}_{label}_{START_TIME}', # folder to store predictions for optional test set
        delete_tmp_folder_after_terminate=False,
        resampling_strategy=model_selection.RepeatedStratifiedKFold,
    resampling_strategy_arguments={'folds': 10, 'n_repeats': 5, 'random_state': random_state})


    # fit() changes the data in place, but refit needs the original data. We
    # therefore copy the data. In practice, one should reload the data
    automl.fit(X_train.copy(), y_train.copy(), metric=score_function)
    # During fit(), models are fit on individual cross-validation folds. To use
    # all available data, we call refit() which trains all models in the
    # final ensemble on the whole dataset.
    automl.refit(X_train.copy(), y_train.copy())
    print(automl.sprint_statistics())
    model_filename = AUTOSKLEARN_TOPMODELS / f'autosklearn_model-f.{feats}-l.{label}-{START_TIME}.joblib'
    joblib.dump(automl, model_filename)
    # dump out the ensemble model as a pickled dictionary
    x = automl.show_models()
    results = {'ensemble': x}
    pickle.dump(results, open(AUTOSKLEARN_ENSEMBLES / f'ensemble_model_description-f.{feats}-l.{label}-{START_TIME}.pickle', 'wb'))

    predictions = automl.predict(X_test)
    #print(get_macroF1(y_test, predictions))
    labelled_preds = le.inverse_transform(predictions)
    final_result = np.column_stack([X_test.index, labelled_preds])
    final_result = pd.DataFrame(final_result, columns=['post_id', 'predictions'])
    if label == 'granular':
        final_result['preds_final'] = final_result.predictions.map(granular_label_dict)
    print(f'MacroF1: {get_macroF1(y_test, predictions)}')
    final_results_path = RESULTS_PATH / f'autosklearn-f.{feats}-l.{label}-{START_TIME}.csv'
    final_result.to_csv(final_results_path, index=None)
    print("Model file at:" + str(model_filename))
    print(f'Final results at: {final_results_path}')