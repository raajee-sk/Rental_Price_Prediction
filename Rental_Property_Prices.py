#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from sklearn.preprocessing import StandardScaler
# ============================================       /     STREAMLIT DASHBOARD      /       ================================================= #
col1,col2,col3=st.columns(3)
with col1:
   st.write("")
with col2:   
    st.markdown(
    f"<h1 style='color:#F24C3D; font-size: 38px;'>Rental Price Prediction</h1>",
    unsafe_allow_html=True,)
with col3:
   st.write("")    
df=pd.read_csv(r"C:\Users\SKAN\Desktop\Raajee\Rent_Prediction\Rental_Property_Prices.csv")   

col1,col2,col3=st.columns(3)
with col1:
 
 locality=st.selectbox(':blue[Select a Locality]',df.locality.unique())
 type=st.text_input(':blue[Type Note:"BHK1":0,"BHK2":1,"BHK3":2,"BHK4":3,"BHK4PLUS":4,"RK1":5]')
 activation_date=st.text_input(':blue[Activation_date]')  
 lease_type=st.text_input(':blue[Lease_type: Note:"ANYONE":0,"BACHELOR":1,"FAMILY":2,"COMPANY":3]')
 gym=st.text_input(':blue[Gym]')
 lift=st.text_input(':blue[Lift]')
 
with col2:
 
 swimming_pool=st.text_input(':blue[Swimming_pool]')
 negotiable=st.text_input(':blue[Negotiable]')
 furnishing=st.text_input(':blue[Furnishing: Note:"NOT_FURNISHED":0,"SEMI_FURNISHED":1,"FULLY_FURNISHED":2]')
 parking=st.text_input(':blue[Parking:"NONE":0,"TWO_WHEELER":1,"FOUR_WHEELER":2,"BOTH":3]')
 property_size=st.text_input(':blue[Property_size]')
 property_age=st.text_input(':blue[Property_age]')
 

with col3:
 
 bathroom=st.text_input(':blue[Bathroom]')
 floor=st.text_input(':blue[Floor]')
 total_floor=st.text_input(':blue[Total_floor]')
 water_supply=st.text_input(':blue[Water_supply: Note:"CORPORATION":0,"BOREWELL":1,"CORP_BORE":2]')
 building_type=st.text_input(':blue[Building_type: Note:"IF":0,"IH":1,"AP":2,"GC":3]')
 balconies=st.text_input(':blue[Balconies]')
 
predict=st.button("Predict")

if predict:

    df1=pd.read_csv(r"C:\Users\SKAN\Desktop\Raajee\Rent_Prediction\Rental_Property_Prices2.csv")
    X=df1.drop("rent",axis=1)
    
    def predict_rent(locality,type,activation_date,lease_type,gym,lift,swimming_pool,negotiable,furnishing,parking,property_size,
        property_age,bathroom,floor,total_floor,water_supply,building_type,balconies):
        x=np.zeros(len(X.columns))
        
        loc_index=np.where(X.columns==locality)[0][0]
        if loc_index>=0:
            x[loc_index]=1
        x[0]=type
        x[1]=activation_date
        x[2]=lease_type
        x[3]=gym
        x[4]=lift
        x[5]=swimming_pool
        x[6]=negotiable
        x[7]=furnishing
        x[8]=parking
        x[9]=property_size
        x[10]=property_age
        x[11]=bathroom
        x[12]=floor
        x[13]=total_floor
        x[14]=water_supply
        x[15]=building_type
        x[16]=balconies
        return x
    
    
    import pickle
    sc=StandardScaler()
    with open(r'C:\Users\SKAN\Desktop\Raajee\Rent_Prediction\model.pkl', 'rb') as file:
        plr= pickle.load(file)
    x=predict_rent(str(locality),type,activation_date,lease_type,gym,lift,swimming_pool,negotiable,furnishing,parking,property_size,
       property_age,bathroom,floor,total_floor,water_supply,building_type,balconies)
    x=sc.fit_transform([x])[0]
    result=plr.predict([x])[0]

    st.write('## :green[Predicted Rental Price:] RS.',"{:.2f}".format(result) )  

    
 
 
 
 

 





 

