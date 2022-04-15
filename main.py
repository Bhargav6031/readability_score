import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


import tensorflow as tf
from tensorflow.keras import Model, Input, backend as K
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Dense, Embedding, Bidirectional, LSTM, Dropout
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from transformers import TFBertModel, BertConfig, BertTokenizerFast


model_name = 'bert_v13'

data_dir = Path('/commonlitreadabilityprize')
train_file = data_dir / 'train.csv'
test_file = data_dir / 'test.csv'
sample_file = data_dir / 'sample_submission.csv'

build_dir = Path('./build/')
output_dir = build_dir / model_name
trn_encoded_file = output_dir / 'trn.enc.joblib'
val_predict_file = output_dir / f'{model_name}.val.txt'
submission_file = 'submission.csv'

pretrained_dir = '/tfbert-large-uncased'

id_col = 'id'
target_col = 'target'
text_col = 'excerpt'

max_len = 205
n_fold = 5
n_est = 9
n_stop = 2
batch_size = 8
seed = 42


trn = pd.read_csv(train_file, index_col=id_col)
tst = pd.read_csv(test_file, index_col=id_col)
y = trn[target_col].values
print(trn.shape, y.shape, tst.shape)
trn.head()



def load_tokenizer():
    if not os.path.exists(pretrained_dir + '/vocab.txt'):
        Path(pretrained_dir).mkdir(parents=True, exist_ok=True)
        tokenizer = BertTokenizerFast.from_pretrained("bert-large-uncased")
        tokenizer.save_pretrained(pretrained_dir)
    else:
        print('loading the saved pretrained tokenizer')
        tokenizer = BertTokenizerFast.from_pretrained(pretrained_dir)
        
    model_config = BertConfig.from_pretrained(pretrained_dir)
    model_config.output_hidden_states = True
    return tokenizer, model_config

def load_bert(config):
    if not os.path.exists(pretrained_dir + '/tf_model.h5'):
        Path(pretrained_dir).mkdir(parents=True, exist_ok=True)
        bert_model = TFBertModel.from_pretrained("bert-large-uncased", config=config)
        bert_model.save_pretrained(pretrained_dir)
    else:
        print('loading the saved pretrained model')
        bert_model = TFBertModel.from_pretrained(pretrained_dir, config=config)
    return bert_model



def bert_encode(texts, tokenizer, max_len=max_len):
    input_ids = []
    token_type_ids = []
    attention_mask = []
    
    for text in texts:
        token = tokenizer(text, max_length=max_len, truncation=True, padding='max_length',
                         add_special_tokens=True)
        input_ids.append(token['input_ids'])
        token_type_ids.append(token['token_type_ids'])
        attention_mask.append(token['attention_mask'])
    
    return np.array(input_ids), np.array(token_type_ids), np.array(attention_mask)



tokenizer, bert_config = load_tokenizer()

X = bert_encode(trn[text_col].values, tokenizer, max_len=max_len)
X_tst = bert_encode(tst[text_col].values, tokenizer, max_len=max_len)
y = trn[target_col].values
print(X[0].shape, X_tst[0].shape, y.shape)



def build_model(bert_model, max_len=max_len):    
    input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    token_type_ids = Input(shape=(max_len,), dtype=tf.int32, name="token_type_ids")
    attention_mask = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")

    sequence_output = bert_model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
    clf_output = sequence_output[:, 0, :]
    clf_output = Dropout(.1)(clf_output)
    out = Dense(1, activation='linear')(clf_output)
    
    model = Model(inputs=[input_ids, token_type_ids, attention_mask], outputs=out)
    model.compile(Adam(lr=1e-5), loss='mean_squared_error', metrics=[RootMeanSquaredError()])
    
    return model



def scheduler(epoch, lr, warmup=5, decay_start=10):
    if epoch <= warmup:
        return lr / (warmup - epoch + 1)
    elif warmup < epoch <= decay_start:
        return lr
    else:
        return lr * tf.math.exp(-.1)

ls = LearningRateScheduler(scheduler)
es = EarlyStopping(patience=n_stop, restore_best_weights=True)

cv = KFold(n_splits=n_fold, shuffle=True, random_state=seed)

p = np.zeros_like(y, dtype=float)
p_tst = np.zeros((X_tst[0].shape[0], ), dtype=float)
for i, (i_trn, i_val) in enumerate(cv.split(X[0]), 1):
    print(f'training CV #{i}:')
    tf.random.set_seed(seed + i)

    bert_model = load_bert(bert_config)
    clf = build_model(bert_model, max_len=max_len)
    if i == 1:
        print(clf.summary())
    history = clf.fit([x[i_trn] for x in X], y[i_trn],
                      validation_data=([x[i_val] for x in X], y[i_val]),
                      epochs=n_est,
                      batch_size=batch_size,
                      callbacks=[ls])
    clf.save_weights(f'{model_name}_cv{i}.h5')

    p[i_val] = clf.predict([x[i_val] for x in X]).flatten()
    p_tst += clf.predict(X_tst).flatten() / n_fold
    
    K.clear_session()
    del clf, bert_model
    gc.collect()



print(f'CV RMSE: {mean_squared_error(y, p, squared=False):.6f}')
np.savetxt(val_predict_file, p, fmt='%.6f')



sub = pd.read_csv(sample_file, index_col=id_col)
sub[target_col] = p_tst
sub.to_csv(submission_file)
sub.head()
