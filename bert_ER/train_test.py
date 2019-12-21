import pickle
import tensorflow as tf
from bert_ER import run_on_dfs
from bert_ER import pretty_print


flags = tf.flags
FLAGS = flags.FLAGS


def predict_single_relation(text):
    text_list = [[text, 'Other'], [text, 'Other']]   # Adding "other" here will not affect the result, just the reuse of code
    pre_text = pd.DataFrame(text_list)
    pre_text.rename(columns={0: 'text', 1: 'relation'}, inplace=True)

    bert_model_hub = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
    tokenizer = create_tokenizer_from_hub_module(bert_model_hub)
    pre_features = make_features(pre_text, label_list, 128, tokenizer, "text", "relation")

    MAX_SEQ_LENGTH = 128
    pre_input_fn = run_classifier.input_fn_builder(
    features=pre_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=False,
    drop_remainder=False)

    pre_result = estimator.predict(pre_input_fn)            # return a generator
    label_id = next(pre_result)['labels']
    # return pre_result

    pre_relation = label_list[label_id]
    return pre_relation


def serving_input_fn():
    input_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='input_ids')
    input_mask = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='input_mask')
    segment_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='segment_ids')
    label_ids = tf.placeholder(tf.int32, [None], name='label_ids')
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        'input_ids': input_ids,
        'input_mask': input_mask,
        'segment_ids': segment_ids,
        'label_ids': label_ids,
    })()
    return input_fn


if __name__ == '__main__':
    myparam = {
        "DATA_COLUMN": "text",
        "LABEL_COLUMN": "relation",
        "LEARNING_RATE": 2e-5,
        "NUM_TRAIN_EPOCHS": 3
    }

    with open("../dataset/SemEval2010.pickle", 'rb') as f:
        train_sem, test_sem = pickle.load(f)

    result, estimator = run_on_dfs(train_sem, test_sem, **myparam)


    # predict
    test = "The <e1>company</e1> fabricates plastic <e2>chairs</e2>."
    print(predict_single_relation(test))


    # see the result
    pretty_print(result)

    # save model
    # estimator._export_to_tpu = False
    # estimator.export_savedmodel('/bert/my_model', serving_input_fn)
