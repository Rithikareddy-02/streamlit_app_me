import numpy as np
import pickle
import streamlit as st
import pandas as pd


# loading the saved model
model = pickle.load(open('D:/Projects/Mini Project/Application/diabetes_prediction.pkl', 'rb'))

bmi_min = 12
bmi_max = 98
mhealth_min = 0
mhealth_max = 30
phealth_min = 0
phealth_max = 30
age_min = 1
age_max = 13


st.title('Diabetes Prediction')

list_of_columns=['Diabetes_binary', 'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
                    'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                    'HvyAlcoholConsump', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age']
input_data=pd.DataFrame(columns=list_of_columns)
input_data.drop(['Diabetes_binary'], axis='columns',inplace=True)


input_data.at[0,'HighBP'] = st.slider('Enter if you have High BP (1 or 0) : ', min_value=0, max_value=1)
input_data.at[0,'HighChol'] = st.slider('Enter if you have High Cholestrol (1 or 0) : ', min_value=0, max_value=1)

input_data.at[0,'CholCheck'] = st.slider('Enter if you have cholestrol check in past 5 years (1 or 0)', min_value=0, max_value=1)
input_data.at[0,'BMI'] = st.slider('Enter your BMI', min_value=1, max_value=100)

input_data.at[0,'Smoker'] = st.slider('Enter if you are a Smoker or not (1 or 0): ', min_value=0, max_value=1)
input_data.at[0,'Stroke'] = st.slider('Enter if you had a Heart Stroke or not (1 or 0): ', min_value=0, max_value=1)

input_data.at[0,'HeartDiseaseorAttack'] = st.slider('Enter if you have any HeartDiseaseorAttack (1 or 0): ', min_value=0, max_value=1)
input_data.at[0,'PhysActivity']=st.slider('Enter if you do any PhysActivity(job is excluded)(1 or 0): ', min_value=0, max_value=1)

input_data.at[0,'Fruits'] = st.slider('Enter if you eat any Fruits daily or not (1 or 0)  : ', min_value=0, max_value=1)
input_data.at[0,'Veggies'] = st.slider('Enter if you eat any Veggies daily or not (1 or 0)  : ', min_value=0, max_value=1)

input_data.at[0,'HvyAlcoholConsump']=st.slider('Enter if you are a Heavy Alcohol Consumer or not (1 or 0) : ', min_value=0, max_value=1)
input_data.at[0,'MentHlth'] = st.slider('Enter no of days of poor Mental health in past 30 days: ', min_value=0, max_value=30)

input_data.at[0,'PhysHlth'] = st.slider('Enter no of days of poor Physical health in past 30 days: ', min_value=0, max_value=30)
input_data.at[0,'DiffWalk']=st.slider('Enter if you have any difficulty in Walking or not  (1 or 0): ', min_value=0, max_value=1)

input_data.at[0,'Sex']=st.slider('Enter your Gender (0- female, 1 - male) : ',min_value=0, max_value=1)

st.text('Please refer the below age category while uploading your age ')
st.text('1 (18 to 24) , 2 (25 to 29), 3 (30 to 34), 4 (35 to 39), 5 (40 to 44), 6(45 to 49), 7(50 to 54)')
st.text('8 (55 to 59), 9 (60 to 64), 10 (65 to 69), 11 (70 to 74), 12 (75 to 79), 13 (80 or older)')

input_data.at[0,'Age'] = st.slider('Enter your Age : ', min_value=1, max_value=13)

input_data['BMI']=(input_data['BMI']-bmi_min)/(bmi_max-bmi_min)
input_data['MentHlth']=(input_data['MentHlth']-mhealth_min)/(mhealth_max-mhealth_min)
input_data['PhysHlth']=(input_data['PhysHlth']-phealth_min)/(phealth_max-phealth_min)
input_data['Age']=(input_data['Age']-age_min)/(age_max-age_min)

if st.button('Result'):

    st.text('Predicted Result = ')

    y_pred=model.predict(input_data)
    if(y_pred[0]==1):
        st.text("You may have Diabetes")
    else:
        st.text('You are not having Diabetes')