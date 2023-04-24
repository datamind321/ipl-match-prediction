import streamlit as st
import pickle 
import pandas as pd 
import streamlit as st
from PIL import Image 
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



image = Image.open('ipl3.png')

st.image(image, caption='IPL 2023 Match Prediction')

teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Punjab Kings',
    'Rajasthan Royals',
    'Delhi Capitals',
    'Gujarat Titans',
    'Lucknow Super Giants',
    'Chennai Super Kings'
] 

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Kolkata', 'Delhi', 'Chennai',
       'Jaipur', 'Cape Town', 'Port Elizabeth', 'Durban', 'Centurion',
       'East London', 'Johannesburg', 'Kimberley', 'Bloemfontein',
       'Ahmedabad', 'Cuttack', 'Nagpur', 'Visakhapatnam', 'Pune',
       'Raipur', 'Ranchi', 'Abu Dhabi', 'Sharjah',  'Bengaluru']

pipe = pickle.load(open('pipe.pkl','rb'))
st.title('Tata IPL Win Predictor Powered by DataMind Predictions')

col1,col2= st.columns(2)

with col1:
    batting_team=st.selectbox('Select the batting team',sorted(teams))
with col2:
    bowling_team=st.selectbox('Select the bowling team',sorted(teams))
    
selected_city = st.selectbox('Select host city',sorted(cities))

target=st.number_input('Target')

col3,col4,col5=st.columns(3)

with col3:
    score=st.number_input('Score')
with col4:
    overs=st.number_input('Overs Completed')
with col5:
    wickets=st.number_input('Wickets Out')
if st.button('Predict Probability'):
    runs_left = target-score
    balls_left=120-(overs*6)
    wickets=10 - wickets
    crr = score/overs
    rrr = (runs_left*6)/balls_left
    
    input_df=pd.DataFrame({'team1':[batting_team],'team2':[bowling_team],'city':[selected_city],'runs_left':[runs_left],'balls_left':[balls_left],'wickets':[wickets],'total_runs_x':[target],'crr':[crr],'rrr':[rrr]})
    
    result=pipe.predict_proba(input_df)
    loss=result[0][0]
    win=result[0][1] 

   
    st.header(batting_team +'- '+str(round(win*100)) + "%") 
    
    st.header(bowling_team +'- '+str(round(loss*100)) + "%")  

    
    
    
    
    
    
