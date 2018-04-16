import os
import re
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from collections import Counter
from nltk.stem.porter import PorterStemmer
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight

def preprocComments(content):
    # Lower case
    content = content.lower()

    # Handle numbers
    content = re.sub('[0-9]+', "number", content)

    # Handle URLs
    content = re.sub('(http|https)://[^\s]*', 'httpaddr', content)

    # Handle Email address
    content = re.sub('[^\s]+@[^\s]+', 'emailaddr', content)

    # Handle dollar
    content = re.sub('[$]+', 'dollar', content)

    # Handle separator
    content = re.sub('[\s\r\n]+', " ", content)
    return content


def calc_vocab_frequency(content, vocab):
    content = preprocComments(content)
    porterStemmer = PorterStemmer()
    exclude_list = ['number', "httpaddr", "emailaddr", "dollar"]
    while content:
        sep = re.search('[ |@|$|/|#|.|-|:|&|*|+|=|\[|\]|\?|!|\(|\)||{|}|,|\'|\"|>|_|<|;|%]', content)
        try:
            str = re.sub("[^a-zA-Z0-9]", "", content[:sep.span()[0]])
            str = porterStemmer.stem(str.strip())
        except AttributeError:
            str = content
            content = ''
        except RecursionError:
            break
        else:
            content = content[sep.span()[1]:]

        if len(str) < 1 or str in exclude_list:
            continue
        for dict in vocab.values():
            dict[str] += 1
    return vocab


def sumForToxicType(df, remove_N_general=1000, min_toxic_fraction=0.01, regen=False):
    toxic_vocab = defaultdict(lambda: defaultdict(int))
    if not regen:
        dir_name = os.path.join(os.path.dirname(__file__), 'dict', str(remove_N_general))
        for file in os.listdir(dir_name):
            toxic_vocab[os.path.splitext(file)[0]] = gen_toxic_vocab(os.path.join(dir_name, file))
        return toxic_vocab

    general_vocab = defaultdict(int)
    comment_type_cnt = defaultdict(int)

    df['comment_text'] = df['comment_text'].str.strip()
    for idx, row in df.iterrows():
        vocal_for_processing = {}

        for col in df.columns:
            if row[col] == 1:
                comment_type_cnt[col] += 1
                vocal_for_processing[col] = toxic_vocab[col]

        if len(vocal_for_processing) == 0:
            general_vocab = calc_vocab_frequency(row['comment_text'], {'general': general_vocab})['general']
        else:
            for type, dict in calc_vocab_frequency(row['comment_text'], vocal_for_processing).items():
                toxic_vocab[type] = dict

    toxic_vocab = remove_general_words(remove_N_general, general_vocab, toxic_vocab)
    thresh = {key: min_toxic_fraction * val for key, val in comment_type_cnt.items()}
    toxic_vocab = keep_N_max(thresh, toxic_vocab)
    write_toxic_file(toxic_vocab, str(remove_N_general))
    return toxic_vocab


def remove_general_words(remove_N_general, general_vocab, toxic_vocab):
    for item, cnt in Counter(general_vocab).most_common(min(len(general_vocab), remove_N_general)):
        for dict_name in toxic_vocab:
            if item in toxic_vocab[dict_name]:
                toxic_vocab[dict_name].pop(item)
    return toxic_vocab


def keep_N_max(threshold, toxic_vocab):
    for name, dict in toxic_vocab.items():
        for item, cnt in Counter(dict).most_common():
            if cnt < threshold[name]:
                toxic_vocab[name].pop(item)
    return toxic_vocab


def gen_toxic_vocab(file_name):
    d = defaultdict(int)
    with open(file_name, 'r') as f:
        for line in f.readlines():
            key = line.split(',')[0]
            val = int(line.split(',')[1])
            d[key] = val
    return d


def write_toxic_file(toxic_vocab, dir):
    dir_name = os.path.join(os.path.dirname(__file__), 'dict', dir)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    for key, dict in toxic_vocab.items():
        file_name = os.path.join(dir_name, key + ".txt")
        if os.path.exists(file_name):
            os.remove(file_name)
        dict = Counter(dict).most_common()
        with open(file_name, 'w') as f:
            for word, cnt in dict:
                f.write("%s, %s\n" % (word, cnt))


def calc_toxic_word_index(content, vocab):
    content = preprocComments(content)
    porterStemmer = PorterStemmer()

    word_cnt = defaultdict(int)
    while content:
        sep = re.search('[ |@|$|/|#|.|-|:|&|*|+|=|\[|\]|\?|!|\(|\)||{|}|,|\'|\"|>|_|<|;|%]', content)
        try:
            str = re.sub("[^a-zA-Z0-9]", "", content[:sep.span()[0]])
            str = porterStemmer.stem(str.strip())
        except AttributeError:
            str = content
            content = ''
        except RecursionError:
            break
        else:
            content = content[sep.span()[1]:]

        if len(str) < 1:
            continue
        if str in vocab:
            word_cnt[str] += 1
    return word_cnt


def feature_engineering(original_data, vocab, type, is_test=False, csv=None):
    score_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    if not csv:
        original_data['comment_text'] = original_data['comment_text'].str.strip()
        modified_data = original_data.drop(score_cols, axis=1).join(pd.DataFrame(data=np.zeros((len(original_data), len(vocab[type]))),
                                                  columns=vocab[type].keys(), dtype=np.int))

        for index, row in modified_data.iterrows():
            word_cnt = calc_toxic_word_index(row['comment_text'], vocab[type])
            for key, val in word_cnt.items():
                modified_data.loc[index, key] = val

        if not is_test:
            modified_data['score'] = original_data[type]
        csv_path = os.path.join(os.path.dirname(__file__), 'dataset', "%s_%s.csv" % (str(type), 'test' if is_test else 'train'))
        if os.path.exists(csv_path):
            os.remove(csv_path)
        modified_data.to_csv(csv_path, index=False)
    else:
        modified_data = pd.read_csv(csv)
    return modified_data


def modelfit(estimator, train_set, target_set, performCV=True, cv_folds=5):
    # Fit the estimatororithm on the data
    estimator.fit(train_set, target_set)

    # Predict training set:
    train_predictions = estimator.predict(train_set)
    train_predprob = estimator.predict_proba(train_set)[:, 1]

    # Perform cross-validation:
    if performCV:
        cv_score = cross_val_score(estimator, train_set, target_set, cv=cv_folds, scoring='roc_auc')

    # Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(target_set, train_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(target_set, train_predprob))

    if performCV:
        print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" %
              (np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)))


def run_gridsearch(X, y, estimator, param_grid, **params):
    logger = init_logger(params['method'])
    try:
        logger.info("begin to perform GridSearch on %s..." % str(param_grid))
        gs = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=params['cv'], scoring=params['scoring'], n_jobs=params['n_jobs'], verbose=1)
        sample_weight = compute_sample_weight(class_weight='balanced', y=y) if params['sample_weight'] is True else None
        gs.fit(X, y, sample_weight=sample_weight)
        logger.info("best score (roc_auc) is %s" % gs.best_score_)
        logger.info("best params are %s" % gs.best_params_)
        for item in gs.grid_scores_:
            logger.info("mean: %s, %s\n" % (item[1], str(item[0])))
    except Exception:
        logger.error("Error found! Break down...\n")


def init_logger(model):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    logging_dir = os.path.join(os.path.dirname(__file__), "log")
    if not os.path.exists(logging_dir):
        os.mkdir(logging_dir)
    logging_file_name = "log_%s.log" % model
    file_handler = logging.FileHandler(os.path.join(logging_dir, logging_file_name))
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

