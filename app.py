import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
import plotly_express as px

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import haversine as hs
from haversine import Unit
import pymongo
import streamlit as st

from utils import dataframe_mongo,calculate_distance,save_model,open_model,recommend_user
st.title("Nearby Friends")
latitude=st.number_input('Latitude: ')
longitude=st.number_input('Longitude')
nearby_=st.slider('How many nearby friends you want to see:',min_value=2,max_value=100,step=10)
button=st.button('Findfriends')
if button==True:
    dataframe,kmeans=dataframe_mongo('mongodb://localhost:27017','Location_based_friend_recommendation','Location_data')
    save_model(kmeans)
    model=open_model()
    nearby_friends=recommend_user(model,dataframe,latitude,longitude,nearby_)
    st.dataframe(nearby_friends)
    st.map(nearby_friends,latitude='lat',longitude='lng',size=20)


