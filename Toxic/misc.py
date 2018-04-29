import os
import re
import copy
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from collections import Counter
from nltk.stem.porter import PorterStemmer
from sklearn import metrics
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight
import xml.etree.ElementTree as ET

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

        # remove duplicated substrings
        regx = re.match(r'\b(\w+)\1+\b', str, re.IGNORECASE)
        if regx:
            str = regx.group(1)
        if len(str) <= 2 or str in exclude_list:
            continue

        for dict in vocab.values():
            dict[str] += 1
    return vocab


def sumForToxicType(df, remove_N_general=1000, min_toxic_fraction=0.02, regen=False):
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


def feature_engineering(original_data, vocab, type, is_test=False):
    score_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    csv_path = os.path.join(os.path.dirname(__file__), 'dataset',
                            "%s_%s.csv" % (str(type), 'test' if is_test else 'train'))
    if not os.path.exists(csv_path):
        original_data['comment_text'] = original_data['comment_text'].str.strip()
        modified_data = original_data.drop(score_cols, axis=1).join(pd.DataFrame(data=np.zeros((len(original_data), len(vocab[type]))),
                                                  columns=vocab[type].keys(), dtype=np.int))

        for index, row in modified_data.iterrows():
            word_cnt = calc_toxic_word_index(row['comment_text'], vocab[type])
            for key, val in word_cnt.items():
                modified_data.loc[index, key] = val

        if not is_test:
            modified_data['score'] = original_data[type]
        modified_data.to_csv(csv_path, index=False)
    else:
        modified_data = pd.read_csv(csv_path)
    return modified_data


def modelfit(estimator, train_set, target_set, method, cv_folds=5, n_jobs=4):
    logger = init_logger(method)
    # Fit the estimator on the data
    estimator.fit(train_set, target_set)

    # Perform cross-validation
    logger.info("\nThe current parameters of %s are\n %s" % (method, str(estimator.get_params())))
    score_auc = cross_val_score(estimator, train_set, target_set, cv=cv_folds, scoring='roc_auc', verbose=1, n_jobs=n_jobs).mean()
    score_acc = cross_val_score(estimator, train_set, target_set, cv=cv_folds, scoring='accuracy', verbose=1, n_jobs=n_jobs).mean()
    logger.info("Average roc_auc score : %.10f, average accuracy score: %.10f" % (score_auc, score_acc))
    return score_auc, score_acc


def modelfit_xgboost(estimator, train_set, target_set, method, cv_folds=5, early_stopping_rounds=50, n_jobs=4):
    logger = init_logger(method)

    # Fit the algorithm on the data
    xgb_param = estimator.get_xgb_params()
    xgtrain = xgb.DMatrix(train_set, label=target_set)
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=estimator.get_params()['n_estimators'], nfold=cv_folds,
                        metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
    best_iter = cvresult.shape[0]
    logger.info("Best iteration: %s" % str(best_iter))
    estimator.set_params(n_estimators=best_iter)

    logger.info("\nThe current parameters of %s are\n %s" % (method, str(estimator.get_params())))
    estimator.fit(train_set, target_set, eval_metric='auc')
    score_auc = cross_val_score(estimator, train_set, target_set, cv=cv_folds, scoring='roc_auc', verbose=1, n_jobs=n_jobs).mean()
    score_acc = cross_val_score(estimator, train_set, target_set, cv=cv_folds, scoring='accuracy', verbose=1, n_jobs=n_jobs).mean()
    logger.info("Average roc_auc score : %.10f, average accuracy score: %.10f" % (score_auc, score_acc))
    return score_auc, score_acc, best_iter


def run_gridsearch(X, y, estimator, param_grid, **params):
    logger = init_logger(params['method'])
    try:
        logger.info("begin to perform GridSearch on %s..." % str(param_grid))
        logger.info("\nThe current parameters of %s are\n %s" % (params['method'], str(estimator.get_params())))
        gs = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=params['cv'], scoring=params['scoring'], n_jobs=params['n_jobs'], verbose=1)
        sample_weight = compute_sample_weight(class_weight='balanced', y=y) if params['sample_weight'] is True else None
        gs.fit(X, y, sample_weight=sample_weight)
        logger.info("best score (roc_auc) is %s" % gs.best_score_)
        logger.info("best params are %s" % gs.best_params_)
        for item in gs.grid_scores_:
            logger.info("mean: %s, %s\n" % (item[1], str(item[0])))
        return gs.best_params_, gs.best_score_
    except Exception as errinfo:
        logger.error("%s\nError found! Break down...\n" % errinfo)


def init_logger(model):
    logger = logging.getLogger(__name__ + '.' + model)
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
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def update_params_toXML(estimator, method, xmlPath):
    logger = init_logger(method)
    params = estimator.get_params()

    if not os.path.exists(xmlPath):
        top = ET.Element('root')
        doc = ET.SubElement(top, 'Model', name=method)
        for name, value in params.items():
            if isinstance(value, str):
                type = 'string'
            elif isinstance(value, bool):
                type = 'bool'
            elif isinstance(value, int):
                type = 'int'
            elif isinstance(value, float):
                type = 'float'
            else:
                type = 'None'
            sub = ET.SubElement(doc, 'param', name=name, type=type)
            sub.text = str(value)
            sub.tail = '\n'
        tree = ET.ElementTree(top)
    else:
        # The xml already exists
        tree = ET.parse(xmlPath)
        for model in tree.getroot().findall('Model'):
            if model.attrib['name'] == method:
                for elem in model:
                    try:
                        elem.text = str(params[elem.attrib['name']])
                    except Exception:
                        logger.info('Error found while updating parameters to %s!' % xmlPath)
                        return
                break
    tree.write(xmlPath)
    logger.info("Successfully Updating params to %s" % xmlPath)


def load_params_fromXML(estimator, method, xmlPath):
    logger = init_logger(method)

    origin_estimator = copy.deepcopy(estimator)
    tree = ET.parse(xmlPath)
    for model in tree.getroot().findall('Model'):
        if model.attrib['name'] == method:
            for child in model:
                name = child.attrib['name']
                type = child.attrib['type']
                try:
                    if type == 'string':
                        param_map = {name: str(child.text)}
                    elif type == 'bool':
                        param_map = {name: bool(child.text)}
                    elif type == 'int':
                        param_map = {name: int(child.text)}
                    elif type == 'float':
                        param_map = {name: float(child.text)}
                    else:
                        param_map = {name: None}
                    estimator.set_params(**param_map)
                except Exception:
                    logger.info('Error found while loading parameters from %s' % xmlPath)
                    return origin_estimator
            logger.info('Successfully loading parameters from %s' % xmlPath)
            logger.info('The load-in parameters are...')
            logger.info(estimator.get_params(()))
            break
    return estimator

