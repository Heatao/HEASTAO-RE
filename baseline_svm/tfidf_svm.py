import pickle
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import joblib
from sklearn.metrics import f1_score, accuracy_score


BASE_DIR = '../'
GLOVE_DIR = os.path.join(BASE_DIR, 'glove')
TEXT_DIR = os.path.join(BASE_DIR, 'dataset')


def get_stopwords():
    with open(os.path.join(TEXT_DIR, 'stop_words_en.txt'), 'r') as f_stopwords:
        stop_words = f_stopwords.read()
    stop_words_list = stop_words.split() + ['!', ',', '.', '?', '-s', '-ly', '</s>', 's']
    return stop_words_list


def remove_stopwords(text_list, stopwords):
    aim_list = []
    for each_sentence in text_list:
        each_sentence = each_sentence.replace('<e>', '').replace('</e>', '')
        each_sentence_list = each_sentence.split()
        temp_list = []
        for each_word in each_sentence_list:
            if each_word not in stopwords:
                temp_list.append(each_word)
        temp_sentence = ' '.join(temp_list)
        aim_list.append(temp_sentence)

    return aim_list


if __name__ == '__main__':
    with open(os.path.join(TEXT_DIR, 'SemEval2010.pickle'), 'rb') as f:
        train_sem, test_sem = pickle.load(f)

    # 数据处理
    train_x = train_sem['text'].tolist()
    label_y = train_sem['relation'].tolist()

    test_x = test_sem['text'].tolist()

    # 是否去停用词，去了效果下降
    # stopwords = get_stopwords()
    # train_x = remove_stopwords(train_x, stopwords)
    # test_x = remove_stopwords(test_x, stopwords)
    corpus = train_x + test_x

    # svm训练
    # 这里可以用max_df或者min_df过滤调一些不重要的词
    tf_vectorizer = TfidfVectorizer(smooth_idf=True, use_idf=True, stop_words=['<e>', '</e>'], decode_error='replace')
    cor_vec = tf_vectorizer.fit_transform(corpus)
    print(cor_vec.shape)

    x = tf_vectorizer.transform(train_x)
    clf = svm.SVC(kernel='rbf', decision_function_shape='ovr', gamma='scale', class_weight='balanced')
    clf.fit(x, label_y)

    # 测试一下
    ainput = "The most common <e1>audits</e1> were about <e2>waste</e2> and recycling."
    input_vec = tf_vectorizer.transform([ainput])
    pre_label = clf.predict(input_vec)
    print(pre_label)

    # 两种保存方式，pickle和joblib
    joblib.dump(clf, 'utils/tfidf_svm.joblib')

    '''
    feature_path = 'utils/feature.pkl'
    with open(feature_path, 'wb') as fw:
        pickle.dump(tf_vectorizer.vocabulary_, fw)
    '''
    joblib.dump(tf_vectorizer, 'utils/feature.joblib')

    # 计算测试集
    test_text_x = tf_vectorizer.transform(test_x)
    test_prelabel = clf.predict(test_text_x)

    # 计算f1值
    print(test_prelabel)
    test_reallabel = test_sem['relation'].tolist()
    macro_f1 = f1_score(test_reallabel, test_prelabel, average='macro')
    ac_score = accuracy_score(test_reallabel, test_prelabel)
    print(macro_f1)
    print(ac_score)
