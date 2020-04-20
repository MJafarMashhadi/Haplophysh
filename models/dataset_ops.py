import pandas as pd
import tensorflow as tf
import tensorflow.keras as k
from sklearn.model_selection import train_test_split


def postpad_to(sequence, to):
    return k.preprocessing.sequence.pad_sequences(sequence, to, padding='post', truncating='post')


def map_text(X):
    if X[-1] == '/':
        X = X[:-1]
    return X


def load_and_clean(*, save_to=None, random_state=42, malware_url_samples=50000):
    data = pd.concat([
        pd.read_csv('../data/mixed.csv', index_col=False),
        pd.DataFrame({
            'url': ['http://' + map_text(X) for X in open('../data/MalwareURLExport.csv', 'r').readlines()],
            'type': 1
        }).reset_index(drop=True).sample(
            n=malware_url_samples,
            replace=False,
            random_state=random_state
        ),
        pd.read_csv('../data/kaggle_data_clean.csv', index_col=0).reset_index(drop=True)
    ]).drop_duplicates(subset=['url'])
    if save_to is not None:
        data.to_csv(save_to, index=False)

    return data


def load_data(*, file_name='../data/cleaned-and-combined.csv', split_ratio=None, random_state=42):
    data = pd.read_csv(file_name, index_col=False)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    if split_ratio is None:
        data_train = data_validation = None
    else:
        data_train, data_validation = train_test_split(
            data,
            test_size=split_ratio,
            stratify=data['type'],
            shuffle=True,
            random_state=42
        )

    return data_train, data_validation, data


def create_dataset_preloaded(word_vectorizer, char_vectorizer, data: pd.DataFrame, vec_length=200):
    word_tokenizer = word_vectorizer.build_tokenizer()

    # wv = tf.constant(postpad_to(word_vectorizer.texts_to_sequences(data['url']), vec_length), name='word')
    wv = tf.constant(postpad_to(
        data['url'].map(lambda url: [word_vectorizer.vocabulary_.get(a, -1)+2 for a in word_tokenizer(url)])  # 0 = padding, 1 = OOV
        , vec_length), name='word')
    cv = tf.constant(postpad_to(char_vectorizer.texts_to_sequences(data['url']), vec_length), name='char')
    targets = tf.one_hot(data['type'], depth=2)

    ds = tf.data.Dataset.from_tensor_slices(((wv, cv), targets))
    return ds


def create_dataset_generator(word_vectorizer, char_vectorizer, data: pd.DataFrame, vec_length=200):
    word_tokenizer = word_vectorizer.build_tokenizer()

    def gen():
        for row in data.iterrows():
            url = row[1].url
            _type = row[1].type

            wv = tf.constant(postpad_to(
                [[word_vectorizer.vocabulary_.get(a, -1) + 2 for a in word_tokenizer(url)]]  # 0 = padding, 1 = OOV
                , vec_length), name='word')
            cv = tf.constant(postpad_to(char_vectorizer.texts_to_sequences([url]), vec_length), name='char')
            target = _type  # tf.one_hot([_type], depth=2)

            yield {'word': tf.squeeze(wv), 'char': tf.squeeze(cv)}, tf.squeeze(target)

    ds = tf.data.Dataset.from_generator(gen, output_types=(
        {'word': tf.float64, 'char': tf.float64}, tf.int32
    ), output_shapes=(
        {'word': tf.TensorShape([vec_length]), 'char': tf.TensorShape([vec_length])}, tf.TensorShape([])
    ))

    return ds

