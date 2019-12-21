import pickle
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
from joblib import dump, load
from sklearn.metrics import f1_score, accuracy_score


BASE_DIR = '../'
GLOVE_DIR = os.path.join(BASE_DIR, 'glove')
TEXT_DIR = os.path.join(BASE_DIR, 'dataset')


def load_embedding_dict(embe_name='glove.6B.100d.txt'):
    embedding_index = {}
    with open(os.path.join(GLOVE_DIR, embe_name), encoding='utf-8') as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)             # 只对第一个出现的空格分割
            # coefs = np.fromstring(coefs, 'f', sep='')        # fromstring将字符串按分割符解码成矩阵
            coefs = np.asarray(coefs.split(), dtype='float32')
            embedding_index[word] = coefs
    print('Found %s word vectors' % len(embedding_index))
    return embedding_index


def ave_sentence_embedding(train_text, em_dict):
    train_list = []
    for each_sentence in train_text:
        sentence_em = 0
        each_sentence = each_sentence.replace('<e>', '').replace('</e>', '')
        each_sentence_list = each_sentence.split()
        for each_word in each_sentence_list:
            if each_word.lower() in em_dict:
                sentence_em += em_dict[each_word.lower()]
        ave_sentence_em = sentence_em / len(each_sentence)
        train_list.append(ave_sentence_em)

    return train_list


if __name__ == '__main__':
    with open(os.path.join(TEXT_DIR, 'SemEval2010.pickle'), 'rb') as f:
        train_sem, test_sem = pickle.load(f)

    embd_dict = load_embedding_dict()

    train_x = train_sem['text'].tolist()
    text_x = ave_sentence_embedding(train_x, embd_dict)
    # text_x = [x.strip().split() for x in train_sem['text']]
    label_y = train_sem['relation'].tolist()

    # 用tfidf表示向量效果不好
    # tf_vectorizer = TfidfVectorizer()
    # x = tf_vectorizer.fit_transform(train_x)
    # print(x)

    # 尝试了线性，不可分
    # lin_clf = svm.LinearSVC(max_iter=10000)
    # lin_clf.fit(text_x, label_y)

    clf = svm.SVC(kernel='rbf', decision_function_shape='ovr', gamma='scale', class_weight='balanced')
    clf.fit(text_x, label_y)

    # 测试一下
    input = "The most common <e1>audits</e1> were about <e2>waste</e2> and recycling."
    input_embe = ave_sentence_embedding([input.strip()], embd_dict)
    pre_label = clf.predict(input_embe)
    print(pre_label)
    # print(clf.predict_log_proba(input_embe))
    # print(clf.predict_proba(input_embe))

    # 两种保存方式，pickle和joblib
    dump(clf, 'utils/glove_svm.joblib')

    # 计算测试集
    test_x = test_sem['text'].tolist()
    test_text_x = ave_sentence_embedding(test_x, embd_dict)
    test_prelabel = clf.predict(test_text_x)
    print(test_prelabel)
    test_reallabel = test_sem['relation'].tolist()
    macro_f1 = f1_score(test_reallabel, test_prelabel, average='macro')
    ac_score = accuracy_score(test_reallabel, test_prelabel)
    print(macro_f1)
    print(ac_score)
