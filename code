import pandas as pd

data1 = pd.read_csv('data999.csv')
data1.head()
data1.columns
data1.info()
print(data1['sido_nm'].unique(),'\n')
print(data1['sido_nm'].nunique(), '\n')
print(data1['sido_nm'].value_counts())
data2 = pd.read_csv('raw_data2.csv')
data2.head()
#print(data1.isnull().sum())
#print(data2.isnull().sum())
In [4]:
data1.dropna(axis=0, subset=['loc_info_x', 'loc_info_y', 'time_unit_tmprt', 
                            'time_unit_ws', 'time_unit_humidity'], inplace=True)

data1.isnull().sum()
data2.dropna(axis=0, subset=['loc_info_x', 'loc_info_y', 'time_unit_tmprt', 
                            'time_unit_ws', 'time_unit_humidity'], inplace=True)

data2.isnull().sum()
#data['time_unit_rainqty'].fillna(0, inplace=True)
#data['time_unit_msnf'].fillna(0, inplace=True)
#data['lfdau_nm'].fillna('', inplace=True)
#data.dropna(axis=0, subset=['time_unit_tmprt'], inplace=True)
#data.dropna(axis=0, subset=['time_unit_ws'], inplace=True)
#data.dropna(axis=0, subset=['time_unit_humidity'], inplace=True)
#data.dropna(axis=0, subset=['time_unit_rainqty'], inplace=True)
#data.drop(['time_unit_msnf', 'lfdau_nm'], axis=1, inplace=True)
#print(data1['emd_nm'].unique(),'\n')
#print(data1['emd_nm'].nunique(), '\n')
#print(data1['emd_nm'].value_counts())
In [8]:
#print(data.sort_values('spt_arvl_dsp_diff'))
In [9]:
#data1.columns
data1.drop(['resc_reprt_no', 'prcs_result_se_nm', 'sido_nm', 
            'dclr_mm', 'dclr_dd', 'dclr_mi', 'dsp_yr', 'dsp_mm', 
            'dsp_dd', 'dsp_hh', 'dsp_mi', 'spt_arvl_yr', 'spt_arvl_mm', 
            'spt_arvl_dd', 'spt_arvl_hh','spt_arvl_mi','resc_cmptn_ymd', 
            'resc_cmptn_tm', 'resc_cmptn_yr','resc_cmptn_mm', 
            'resc_cmptn_dd', 'resc_cmptn_hh', 'resc_cmptn_mi', 'hmg_ymd', 
            'hmg_tm', 'hmg_yr', 'hmg_mm', 'hmg_dd', 'hmg_hh', 'hmg_mi'], axis=1, inplace=True)
data1.head()
#data2.columns
In [12]:
data2.drop(['msfrtn_resc_reprt_no', 'prcs_result_se_nm', 'sido_nm', 
            'dclr_mnth', 'dclr_day', 'dclr_min', 'dsp_yr', 'dsp_mnth', 
            'dsp_day', 'dsp_hour', 'dsp_min', 'spt_arvl_yr', 'spt_arvl_mnth', 
            'spt_arvl_day', 'spt_arvl_hour','spt_arvl_min','resc_cmptn_ymd', 
            'resc_cmptn_tm','resc_cmptn_yr', 'resc_cmptn_mnth', 'resc_cmptn_day', 
            'resc_cmptn_hour','resc_cmptn_min', 'hmg_ymd', 'hmg_tm', 'hmg_yr', 
            'hmg_mnth', 'hmg_day', 'hmg_hour', 'hmg_min'], axis=1, inplace=True)
data2.head()
data1.rename(columns={'dclr_yr':'year', 'dclr_hh':'hour', 
                      'cty_frmvl_se':'cty_frmvl_se_nm'}, inplace=True)
data2.rename(columns={'dclr_yr':'year', 'dclr_hour':'hour'}, inplace=True)
def time_format(x):
    
    if len(x) == 6:
        return x
    elif len(x) == 5:
        return '0'+x
    elif len(x) == 4:
        return '00'+x
    elif len(x) == 3:
        return '000' + x
    elif len(x) == 2:
        return '0000' + x
    elif len(x) == 1:
        return '00000' + x

#data['dclr_ymd'] = data['dclr_ymd'].astype(str)
#data['dclr_tm'] = data['dclr_tm'].astype(str)
#data['dclr_tm'] = data['dclr_tm'].apply(time_format)

data1['dsp_ymd'] = data1['dsp_ymd'].astype(str)
data1['dsp_tm'] = data1['dsp_tm'].astype(str)
data1['dsp_tm'] = data1['dsp_tm'].apply(time_format)

data1['spt_arvl_ymd'] = data1['spt_arvl_ymd'].astype(str)
data1['spt_arvl_tm'] = data1['spt_arvl_tm'].astype(str)
data1['spt_arvl_tm'] = data1['spt_arvl_tm'].apply(time_format)

data2['dsp_ymd'] = data2['dsp_ymd'].astype(str)
data2['dsp_tm'] = data2['dsp_tm'].astype(str)
data2['dsp_tm'] = data2['dsp_tm'].apply(time_format)

data2['spt_arvl_ymd'] = data2['spt_arvl_ymd'].astype(str)
data2['spt_arvl_tm'] = data2['spt_arvl_tm'].astype(str)
data2['spt_arvl_tm'] = data2['spt_arvl_tm'].apply(time_format)
In [15]:
for i in data1.index:
    #data.at[i, 'dclr_time'] = data.at[i, 'dclr_ymd'] + ' ' + data.at[i, 'dclr_tm']
    data1.at[i, 'dsp_time'] = data1.at[i, 'dsp_ymd'] + ' ' + data1.at[i, 'dsp_tm']
    data1.at[i, 'spt_arvl_time'] = data1.at[i, 'spt_arvl_ymd'] + ' ' + data1.at[i, 'spt_arvl_tm']
    
In [16]:
for i in data2.index:
    #data.at[i, 'dclr_time'] = data.at[i, 'dclr_ymd'] + ' ' + data.at[i, 'dclr_tm']
    data2.at[i, 'dsp_time'] = data2.at[i, 'dsp_ymd'] + ' ' + data2.at[i, 'dsp_tm']
    data2.at[i, 'spt_arvl_time'] = data2.at[i, 'spt_arvl_ymd'] + ' ' + data2.at[i, 'spt_arvl_tm']
    
In [17]:
#data['dclr_time'] = pd.to_datetime(data['dclr_time'], format='%Y-%m-%d %H:%M:%S')
data1['dsp_time'] = pd.to_datetime(data1['dsp_time'], format='%Y-%m-%d %H:%M:%S')
data1['spt_arvl_time'] = pd.to_datetime(data1['spt_arvl_time'], format='%Y-%m-%d %H:%M:%S')

data2['dsp_time'] = pd.to_datetime(data2['dsp_time'], format='%Y-%m-%d %H:%M:%S')
data2['spt_arvl_time'] = pd.to_datetime(data2['spt_arvl_time'], format='%Y-%m-%d %H:%M:%S')
for i in data1.index:
    data1.at[i,'spt_arvl_dsp_diff'] = data1.at[i,'spt_arvl_time'] - data1.at[i, 'dsp_time']
In [19]:
for i in data2.index:
    data2.at[i,'spt_arvl_dsp_diff'] = data2.at[i,'spt_arvl_time'] - data2.at[i, 'dsp_time']
In [20]:
data = pd.concat([data1, data2])
data
data['spt_arvl_dsp_diff'] = data['spt_arvl_dsp_diff'].dt.seconds
data['spt_arvl_dsp_diff'] = data['spt_arvl_dsp_diff']/60
In [22]:
data.drop(['dclr_ymd', 'dclr_tm', 'dsp_ymd', 'dsp_tm', 'spt_arvl_ymd', 'spt_arvl_tm', 'dsp_time', 'spt_arvl_time'], axis=1, inplace=True)
In [23]:
import numpy as np

class_mapping = {label:idx for idx, label in enumerate(np.unique(data1['cty_frmvl_se_nm']))}

data['cty_frmvl_se_nm'] = data['cty_frmvl_se_nm'].map(class_mapping)
In [24]:
data.loc[data['spt_arvl_dsp_diff'] >= 0, 'y'] = '5분이하'
data.loc[data['spt_arvl_dsp_diff'] >= 5, 'y'] = '5분이상'
data.loc[data['spt_arvl_dsp_diff'] >= 10, 'y'] = '10분이상'
data.loc[data['spt_arvl_dsp_diff'] >= 15, 'y'] = '15분이상'
data.loc[data['spt_arvl_dsp_diff'] >= 20, 'y'] = '20분이상'
In [25]:
data['y'].value_counts()
data.to_csv("data999.csv", index=False)
In [3]:
df = pd.read_csv('data999.csv')
df.head()
df.shape
df.columns
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix

cols = ['spt_frstt_dist', 'time_unit_tmprt', 'time_unit_rainqty', 'time_unit_msnf', 'time_unit_ws', 'time_unit_humidity', 'spt_arvl_dsp_diff']

scatterplotmatrix(data[cols].values, figsize=(20, 15), 
                  names=cols, alpha=0.5)
plt.tight_layout()
plt.show()
from mlxtend.plotting import heatmap

cm = np.corrcoef(data[cols].values.T)
hm = heatmap(cm, row_names=cols, column_names=cols)

plt.show()
data.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, test_size=0.1, random_state=0)
In [103]:
from sklearn.linear_model import LinearRegression

mlr = LinearRegression()
mlr.fit(X_train, y_train) 
y_predict = mlr.predict(X_test)
In [105]:
import matplotlib.pyplot as plt
%matplotlib inline

plt.scatter(y_test, y_predict, alpha=0.4)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("MULTIPLE LINEAR REGRESSION")
plt.show()
print(mlr.coef_)
print(mlr.score(X_train, y_train))
