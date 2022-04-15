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

data_dir = Path('../input/commonlitreadabilityprize')
train_file = data_dir / 'train.csv'
test_file = data_dir / 'test.csv'
sample_file = data_dir / 'sample_submission.csv'

build_dir = Path('./build/')
output_dir = build_dir / model_name
trn_encoded_file = output_dir / 'trn.enc.joblib'
val_predict_file = output_dir / f'{model_name}.val.txt'
submission_file = 'submission.csv'

pretrained_dir = '../input/tfbert-large-uncased'

id_col = 'id'
target_col = 'target'
text_col = 'excerpt'

max_len = 205
n_fold = 5
n_est = 9
n_stop = 2
batch_size = 8
seed = 42