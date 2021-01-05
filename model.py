import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf


def predict(json_obj):

    cols = ['make', 'model', 'reg_year', 'mileage', 'fuel_type',
            'transmission', 'drive_type', 'prev_owners']

    X = pd.DataFrame(json_obj, index=[0])
    X.columns = cols

    make_tokenizer_file = open(r'.\models\make_tokenizer_file.obj', 'rb')
    make_tokenizer = pickle.load(make_tokenizer_file)
    make_tokenizer_file.close()

    model_tokenizer_file = open(r'.\models\model_tokenizer_file.obj', 'rb')
    model_tokenizer = pickle.load(model_tokenizer_file)
    model_tokenizer_file.close()

    X['make'] = X['make'].apply(lambda x: x.replace(' ', ''))
    X['model'] = X['model'].apply(lambda x: x.replace(' ', ''))

    X['make'] = tf.squeeze(make_tokenizer.texts_to_sequences(X['make'])).numpy()
    X['model'] = tf.squeeze(model_tokenizer.texts_to_sequences(X['model'])).numpy()

    def func1(x):
        try:
            return x.lower()
        except:
            return x
        
    def func2(x):
        try:
            return pd.to_numeric(x)
        except:
            return x
        
    for col in X.columns:
        X[col] = X[col].apply(lambda x: func1(x))
        X[col] = X[col].apply(lambda x: func2(x))

    onehot_vars = ['fuel_type', 'transmission', 'drive_type']

    onehot_encoder_file = open(r'.\models\onehot_encoder_file.obj', 'rb')
    onehot_encoder = pickle.load(onehot_encoder_file)
    onehot_encoder_file.close()

    X = X.join(pd.DataFrame(onehot_encoder.transform(X[onehot_vars]).toarray(),
                            columns=onehot_encoder.get_feature_names(onehot_vars))).drop(
        onehot_vars, axis=1)

    X = np.expand_dims(X.iloc[0].to_numpy(), axis=1)
    X = np.expand_dims(X, axis=1)

    model = tf.keras.models.load_model('./models/tf_model.h5')

    prediction = model.predict((X[0], X[1], X[2:].T))[0][0][0]

    return int(prediction)


if __name__ == '__main__':
    pass
