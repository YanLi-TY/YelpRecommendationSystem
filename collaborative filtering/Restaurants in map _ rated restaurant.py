# find the restaurants in map
# top rated restaurants

import gc 
import numpy as np 
from collections import Counter 
import pandas as pd 
import matplotlib.pyplot as plt 
plt.style.use('fivethirtyeight') 
import seaborn as sns 

import re 
import string 
import nltk 
import plotly
import plotly.offline as py 
py.init_notebook_mode(connected=True)
import plotly.tools as tls
import plotly.graph_objs as go
from plotly.graph_objs import *
import plotly.tools as tls
import plotly.figure_factory as fig_fact
plotly.tools.set_config_file(world_readable=True, sharing='public')

import warnings
warnings.filterwarnings('ignore')



# data preparing
df_yelp_business = pd.read_csv('./input/yelp_business.csv')
df_category_split = df_yelp_business['categories'].str.split(';', expand=True)[[0,1,2]]
df_category_split.columns = ['category_1', 'category_2', 'category_3']
df_yelp_business = pd.concat([df_yelp_business, df_category_split], axis=1)
df_yelp_business = df_yelp_business.drop(['categories'], axis=1)

# find nearby restaurants in map
df_yelp_business_restaurants = df_yelp_business.loc[(df_yelp_business['category_1'] == 'Restaurants') | 
                                                    (df_yelp_business['category_2'] == 'Restaurants') |
                                                     (df_yelp_business['category_3'] == 'Restaurants')]

descrip = df_yelp_business_restaurants[['name', 'stars']].astype(str).apply(lambda x: '. Rating: '.join(x), axis=1).tolist()

# mapbox access token. Go to Mapbox.com and sign up and get an access_token. 
mapbox_access_token = 'pk.eyJ1IjoiYWxtYXNzbml5YW1hdCIsImEiOiJjamQ3NGY2Zms0emhmMnFuMjJ5OGNvOWoxIn0.DKGZcDRHVFRYNiQBe1D_zw'

# define the data for ploting on mapbox
data = Data([
    Scattermapbox(
        lat=df_yelp_business_restaurants.latitude.tolist(),
        lon=df_yelp_business_restaurants.longitude.tolist(),
        mode='markers',
        marker=Marker(
            symbol='bar',
            size=9
        ),
        text=descrip,
    )
])
# define the map layout
layout = Layout(
    title='Zoom to your location and find your desire restaurants',
    autosize=True,
    hovermode='closest',
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=dict(
            lat=39.4440302947,
            lon=-98.9565517008
        ),
        style='light',
        pitch=0,
        zoom=3
    ),
)
fig = dict(data=data, layout=layout)
py.iplot(fig, filename='Multiple Mapbox')

# top 20 rated restaurants
df_yelp_review = pd.read_csv('./input/yelp_review.csv')
df_yelp_tip = pd.read_csv('./input/yelp_tip.csv')

df_yelp_review['name'] = df_yelp_review['business_id'].map(df_yelp_business_restaurants.set_index('business_id')['name'])
top_restaurants = df_yelp_review.name.value_counts().index[:20].tolist()
df_review_top = df_yelp_review.loc[df_yelp_review['name'].isin(top_restaurants)]
df_review_top.groupby(df_review_top.name)['stars'].mean().sort_values(ascending=True).plot(kind='barh',figsize=(12, 10))
plt.yticks(fontsize=18)
plt.title('Top rated restaurants on Yelp',fontsize=20)
plt.ylabel('Restaurants names', fontsize=18)
plt.xlabel('Ratings', fontsize=18)
plt.show()