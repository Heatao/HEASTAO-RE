import joblib
import pickle
import os


def svm_pre(ainput, utils_path='utils/'):

    # 用pickle读语法有问题
    # feature_path = 'utils/feature.pkl'
    # tf_vectorizer = TfidfVectorizer(decode_error='replace', vocabulary=pickle.load(open(feature_path, 'rb')))
    print(os.getcwd())
    print(os.path.join(utils_path, 'feature.joblib'))
    tf_vectorizer = joblib.load(os.path.join(utils_path, 'feature.joblib'))

    model_path = os.path.join(utils_path, 'tfidf_svm.joblib')
    clf = joblib.load(model_path)

    input_vec = tf_vectorizer.transform([ainput])
    pre_label = clf.predict(input_vec)
    return pre_label


if __name__ == '__main__':

    test = 'The most common <e1>audits</e1> were about <e2>waste</e2> and recycling.'
    print(svm_pre(test))
