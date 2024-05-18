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
import pickle 


def calculate_distance(lat1,lng1,lat2,lng2):
    loc1=(lat1,lng1)
    loc2=(lat2,lng2)
    result=hs.haversine(loc1,loc2,unit=Unit.KILOMETERS)
    return result

def dataframe_mongo(link,db_name,collection_name):
    client=pymongo.MongoClient(link)
    db=client[db_name]
    collection=db[collection_name]
    data=[]
    for doc in collection.find():
        _id=doc['_id']
        Username=doc['UserName']
        City=doc['city']
        lat=doc['lat']
        lng=doc['lng']
        data.append([_id,Username,City,lat,lng])
    data_array=np.array(data)
    client.close()
    dataframe=pd.DataFrame(data_array,columns=['_id','UserName','City','lat','lng'])
    coordinates=dataframe[['lat','lng']]
    kmeans=KMeans(n_clusters=10,init='k-means++')
    kmeans.fit(coordinates)
    dataframe['cluster']=kmeans.labels_
    return dataframe,kmeans

def save_model(model):
    with open("model_pkl",'wb') as f:
        pickle.dump(model,f)

def open_model():
    with open('model_pkl' , 'rb') as f:
        model = pickle.load(f)
        return model


def recommend_user(model,df,lat,lon,n):
    cluster=model.predict(np.array([lat,lon]).reshape(1,-1))[0]
    print(cluster)    
    df2=df[df['cluster']==cluster][['_id','UserName','City','lat','lng']]
    df2['distance']=df2.apply(lambda row: calculate_distance(lat,lon,row['lat'],row['lng']),axis=1)
    df2.sort_values(by='distance',inplace=True)
    return df2[['_id','UserName','City','distance','lat','lng']].iloc[0:n]
    

