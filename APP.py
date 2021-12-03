import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image
import pickle
from sklearn.preprocessing import MinMaxScaler
import numpy as nm

st.set_page_config(page_title='Package Prediction')
st.header('Package Predictor')

excel_file = 'AI_Project_Data_1.xlsx'
sheet_name = 'data'

data  = pd.read_excel(excel_file,
                    sheet_name = sheet_name,
                    usecols = 'A:N',
                    header = 0)

"""Missing Values Treatment"""

data.isnull().sum()

m_ip = data['IP'].mean()
m_psp = data['PSP'].mean()
m_elcslab = data['ELCS LAB'].mean()
m_eng1 = data['ENGLISH 1'].mean()
m_dbms = data['DBMS'].mean()
m_cn  = data['CN'].mean()
m_wt = data['WT'].mean()

data['IP'].fillna(value = m_ip, inplace=True) 
data['PSP'].fillna(value = m_psp, inplace=True)
data['ELCS LAB'].fillna(value = m_elcslab, inplace=True)
data['ENGLISH 1'].fillna(value = m_eng1, inplace=True)
data['DBMS'].fillna(value = m_dbms, inplace=True)
data['CN'].fillna(value = m_cn, inplace=True)
data['WT'].fillna(value = m_wt, inplace=True)

"""Min and Max Values"""

ssc_min = data['SSC'].min()
ssc_max = data['SSC'].max()
inter_min = data['Inter'].min()
inter_max = data['Inter'].max()
btech_min = data['B.Tech 3-1'].min()
btech_max = data['B.Tech 3-1'].max()
ip_min = data['IP'].min()
ip_max = data['IP'].max()
psp_min = data['PSP'].min()
psp_max = data['PSP'].max()
elcs_min = data['ELCS LAB'].min()
elcs_max = data['ELCS LAB'].max()
eng1_min = data['ENGLISH 1'].min()
eng1_max = data['ENGLISH 1'].max()
ds_min = data['DS'].min()
ds_max = data['DS'].max()
os_min = data['OS'].min()
os_max = data['OS'].max()
dbms_min = data['DBMS'].min()
dbms_max = data['DBMS'].max()
oopc_min = data['OOPC'].min()
oopc_max = data['OOPC'].max()
cn_min = data['CN'].min()
cn_max = data['CN'].max()
wt_min = data['WT'].min()
wt_max = data['WT'].max()
package_min = data['Package'].min()
package_max = data['Package'].max()

"""Normalisation"""

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# Applying scaler() to all the columns except the 'yes-no' and 'dummy' variables
num_vars = ['SSC', 'Inter', 'B.Tech 3-1', 'IP', 'PSP', 'ELCS LAB',
            'ENGLISH 1', 'DS', 'OS', 'DBMS', 'OOPC', 'CN', 'WT', 'Package']
data[num_vars] = scaler.fit_transform(data[num_vars])


"""Data Split"""

from sklearn.model_selection import train_test_split
feature_cols = ['SSC', 'Inter', 'B.Tech 3-1', 'IP', 'PSP', 'ELCS LAB',
                'ENGLISH 1', 'DS', 'OS', 'DBMS', 'OOPC', 'CN', 'WT']
X = data[feature_cols] 
y = data.Package
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state = 1)

"""Model Training"""

#Fitting Decision Tree classifier to the training set  
from sklearn.ensemble import RandomForestRegressor
regressor= RandomForestRegressor(n_estimators= 3)  
regressor.fit(X_train, y_train)


list_of_columns = data.columns
input_data=pd.DataFrame(columns=list_of_columns)
input_data.drop(['Package'], axis='columns', inplace=True)


input_data.at[0, 'SSC'] = st.slider('SSC Percentage : ',
                                        min_value = 0,
                                        max_value = 100,
                                        )
input_data.at[0, 'Inter'] = st.slider('Inter Percenatge : ',
                                        min_value = 0,
                                        max_value = 100,
                                        )
input_data.at[0, 'B.Tech 3-1'] = st.slider('B.tech Percenatge : ',
                                        min_value = 0,
                                        max_value = 100,
                                        )
input_data.at[0, 'IP'] = st.slider('IP Marks : ',
                                        min_value = 0,
                                        max_value = 100,
                                        )
input_data.at[0, 'PSP'] = st.slider('PSP Marks : ',
                                        min_value = 0,
                                        max_value = 100,
                                        )
input_data.at[0, 'ELCS LAB'] = st.slider('ELCS Lab Marks : ',
                                        min_value = 0,
                                        max_value = 100,
                                        )
input_data.at[0, 'ENGLISH 1'] = st.slider('English 1 : ',
                                        min_value = 0,
                                        max_value = 100,
                                        )
input_data.at[0, 'DS'] = st.slider('DS Marks : ',
                                        min_value = 0,
                                        max_value = 100,
                                        )
input_data.at[0, 'OS'] = st.slider('OS Marks : ',
                                        min_value = 0,
                                        max_value = 100,
                                        )
input_data.at[0, 'DBMS'] = st.slider('DBMS Marks : ',
                                        min_value = 0,
                                        max_value = 100,
                                        )
input_data.at[0, 'OOPC'] = st.slider('OOPC Marks : ',
                                        min_value = 0,
                                        max_value = 100,
                                        )
input_data.at[0, 'CN'] = st.slider('CN Marks : ',
                                        min_value = 0,
                                        max_value = 100,
                                        )
input_data.at[0, 'WT'] = st.slider('WT Marks : ',
                                        min_value = 0,
                                        max_value = 100,
                                        )

if st.button("Predict the Package"):
    result =  regressor.predict(input_data)
    Package=result*(package_max-package_min)+package_min
    st.text('Predicted  Package = ')
    st.text(Package)
