 from google.colab import drive
drive.mount('/content/drive')
import pandas as pd

data = pd.read_csv("/content/drive/MyDrive/colab/2021_2학기_노바투스_사전교육/data999.csv")
data.head()
data.isnull().sum()
data.shape
data.dropna(subset=['loc_info_x'],axis=0,inplace=True)
data.loc[data['spt_arvl_dsp_diff'] >= 0, 'y'] = '5분이하'
data.loc[data['spt_arvl_dsp_diff'] >= 5, 'y'] = '5분이상'
data.loc[data['spt_arvl_dsp_diff'] >= 10, 'y'] = '10분이상'
data.loc[data['spt_arvl_dsp_diff'] >= 15, 'y'] = '15분이상'
data.loc[data['spt_arvl_dsp_diff'] >= 20, 'y'] = '20분이상'
data['y'].value_counts()
data_1 = data[['acdnt_cause', 'season_se_nm','daywk', 'cty_frmvl_se_nm', 'spt_frstt_dist', 'acdnt_place_nm',
               'time_unit_tmprt', 'loc_info_x', 'loc_info_y','hour', 'sigungu_nm','dong_se',
             'time_unit_ws', 'time_unit_humidity',  'spt_arvl_dsp_diff','y']]
data_1.head()
data_1['h'] = 0

data_1.loc[data_1['hour'] >= 0, 'h'] = '0'
data_1.loc[data_1['hour'] >= 6, 'h'] = '1'
data_1.loc[data_1['hour'] >= 12, 'h'] = '2'
data_1.loc[data_1['hour'] >= 18, 'h'] = '3'
data_one_hot = pd.get_dummies(data_1[['acdnt_cause','season_se_nm','daywk','acdnt_place_nm','h', 'sigungu_nm','dong_se']])
data_one_hot
pd.set_option('display.max_columns', 100)
data_2 = pd.concat([data_1[['cty_frmvl_se_nm', 'spt_frstt_dist', 'time_unit_tmprt', 'time_unit_ws', 'time_unit_humidity', 'spt_arvl_dsp_diff'
,'loc_info_x', 'loc_info_y','y']], data_one_hot], axis=1)
data_2.head()
import numpy as np
y_total = data_2[['y']]
X_total =  data_2.drop(columns=['spt_arvl_dsp_diff'], axis=1)
X_total =  X_total.drop(columns=['y'], axis=1)

# X_total = data_2[['spt_frstt_dist']]

# y_total = data_2[['spt_arvl_dsp_diff']]
X_total
y_total
# normalization_X_total = X_total[['cty_frmvl_se', 'spt_frstt_dist', 'time_unit_tmprt', 'time_unit_rainqty',\
#              'time_unit_ws', 'time_unit_humidity', 'time_unit_msnf']]
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

transformer = RobustScaler()
transformer.fit(X_total)

X_total = transformer.transform(X_total) 
X_total
# transformer = RobustScaler()
# transformer.fit(y_total)

# y_total = transformer.transform(y_total) 
y_total[y_total['y'] == '5분이하'] = 0
y_total[y_total['y'] == '5분이상'] = 1
y_total[y_total['y'] == '10분이상'] = 2
y_total[y_total['y'] == '15분이상'] = 3
y_total[y_total['y'] == '20분이상'] = 4
y_total
# normalization_X_total = (normalization_X_total - normalization_X_total.mean())/normalization_X_total.std()

# normalization_X_total = pd.concat([normalization_X_total, data_one_hot], axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_total, y_total.to_numpy().astype(float), test_size=0.3)
X_train.shape
y_train.T[0].T
np.expand_dims(y_train.T[0],axis=0).T.shape
y_train
np.unique(y_train)
# gpu 설정
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Masking, GRU, Conv1D, MaxPooling1D, Flatten, BatchNormalization ,Activation
from tensorflow.keras.models import Model, Sequential

with tf.device('/device:GPU:0'):
  input = tf.keras.layers.Input(shape=(99,))
  # net = Dense(units=128)(input)
  # net = BatchNormalization()(net)
  # net = Activation(activation='relu')(net)
  # net = Dense(units=256)(net)
  # net = BatchNormalization()(net)
  # net = Activation(activation='relu')(net)
  # net = Dense(units=256)(net)
  # net = BatchNormalization()(net)
  # net = Activation(activation='relu')(net)
  # net = Dense(units=64)(net)
  # net = BatchNormalization()(net)
  # net = Activation(activation='relu')(net)
  net = Dense(units=5)(input)
  net = Activation(activation='softmax')(net)
  model = tf.keras.models.Model(input, net)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history=model.fit(X_train,   y_train.astype('int64'), epochs=500, validation_data=(X_test,   y_test.T), batch_size = 17920)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=17)
rf.fit(X_train, y_train)
forest_predictions = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, forest_predictions)
history.history['val_accuracy']
history.history['accuracy']
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='darkgrid')
sns.lineplot(x=range(80), y=np.array(history.history['accuracy'][:80]))
sns.lineplot(x=range(80), y=np.array(history.history['val_accuracy'][:80]))
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(['Train', 'Test'], loc='upper left')
model.evaluate(X_test, , '6~10분')
predict_labels = model.predict(X_test)
len(np.argmax(predict_labels,axis=-1))
from sklearn.metrics import classification_report

print(classification_report(y_test, np.argmax(predict_labels,axis=-1), target_names=['5분이하', '6~10분', '11~15분', '15~20분', '20분이상']))
