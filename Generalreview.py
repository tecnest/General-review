#!/usr/bin/env python
# coding: utf-8

# In[1]:


#------------------------------------loading data--------------------------------------------------#
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

df= pd.read_excel("C:\Users\dell\Downloads\Generalreview\General-review\ITEMS&COLOR(1).xlsx", header=None)

new_header = df.iloc[2]  # Grab the third row for the header, remember it's index 2 because of zero-indexing
df = df[3:]  # Take the data less the header row
df.columns = new_header  # Set the header row as the DataFrame header

# Reset the index of the DataFrame if necessary
df.reset_index(drop=True, inplace=True)

numeric_columns = ['TY.Sales', 'oh', 'A_SellThru', 'TY.Qty', 'A_Days']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

#------------------------------------cleaning data--------------------------------------------------#
import numpy as np

# Replace 'inf' with 'NaN'
df.replace([np.inf, -np.inf], np.nan, inplace=True)
pd.DataFrame(df)

# Drop rows where 'TY.sales' is NaN
df.dropna(subset=['TY.Sales'], inplace=True)

# Drop rows with 'NaN'
df.dropna(inplace=True)
#pd.DataFrame(df)


# <div style="text-align: center; color: blue;"><h1>General Review</h1></div>
# <div style="text-align: left; color: orange;"><h2>Boy & Men Review</h2></div>
# 

# In[2]:


#-----------------------------------drop columns  &  group_bys  -----------------------------------------------------#

import locale
locale.setlocale(locale.LC_ALL, '')

st.title('General review')
st.header('Boy & Men Review')


df1=df.copy()

columns_to_drop = ['itemid', 'EailiestDate', 'TY.Disc%', 'A_AvgQSoldD-MinDate']  # List of columns to drop
df1.drop(columns_to_drop, axis=1, inplace=True)

agg_funcs = {
    'A_Days': 'mean',
    'TY.Qty': 'sum',
    'TY.Sales': 'sum',
    'oh':'sum',
    'A_SellThru': 'mean',
}

# Group by parameter and apply custom aggregation functions
gender = df1.groupby('Gender').agg(agg_funcs)
Cat=df1.groupby('Cat2').agg(agg_funcs)
Color=df1.groupby('Color').agg(agg_funcs)
men = df1[df1['Gender'] == 'Men']
boys = df1[df1['Gender'] == 'Boys']

gender1=gender.copy()
Cat1=Cat.copy()
Color1=Color.copy()
men1=men.copy()
boys1=boys.copy()

gender.drop('A_Days', axis=1, inplace=True)
total_OH = gender['oh'].sum()
total_sales=gender['TY.Sales'].sum()
gender['Sales%']=(gender['TY.Sales']/ total_sales) * 100
gender['Stock%'] = (gender['oh'] / total_OH) * 100

desired_order = ['TY.Sales', 'Sales%', 'oh', 'Stock%', 'TY.Qty', 'A_SellThru']
# Reindex the DataFrame with the desired order of columns
gender = gender.reindex(columns=desired_order)

for col in gender:
    gender[col] = gender[col].apply(lambda x: locale.format_string("%.2f", x, grouping=True) if not pd.isna(x) else np.nan)
#gender.reset_index(drop=True, inplace=True)
#gender.columns.values[0] = 'Category'
gender


# <div style="text-align: left; color: orange;"><h2>Category Review</h2></div>
# 

# In[3]:


Cat.drop('A_Days', axis=1, inplace=True)
total_OH = Cat['oh'].sum()
total_sales=Cat['TY.Sales'].sum()
Cat['Sales%']=(Cat['TY.Sales']/ total_sales) * 100
Cat['Stock%'] = (Cat['oh'] / total_OH) * 100

Cat1.drop('A_Days', axis=1, inplace=True)
total_OH = Cat1['oh'].sum()
total_sales=Cat1['TY.Sales'].sum()
Cat1['Sales%']=(Cat1['TY.Sales']/ total_sales) * 100
Cat1['Stock%'] = (Cat1['oh'] / total_OH) * 100

desired_order = ['TY.Sales', 'Sales%', 'oh', 'Stock%', 'TY.Qty', 'A_SellThru']
# Reindex the DataFrame with the desired order of columns
Cat = Cat.reindex(columns=desired_order)
Cat = Cat.sort_values(by='TY.Sales', ascending=False)
Cat1 = Cat.sort_values(by='TY.Sales', ascending=False)

for col in Cat:
    Cat[col] = Cat[col].apply(lambda x: locale.format_string("%.2f", x, grouping=True) if not pd.isna(x) else np.nan)
    
st.header('Category Review')
pd.DataFrame(Cat)
Cat

# <div style="text-align: center; color: blue;"><h1>Visualisation</h1></div>
# 

# In[7]:


import streamlit as st
import plotly.graph_objs as go
import pandas as pd

# Example of how you might define and use a function in Streamlit to create and display plots
fig = go.Figure()
fig.add_trace(go.Bar(x=Cat1.index, y=Cat1['TY.Sales'], hoverinfo='y', marker_color='royalblue'))
fig.update_layout(title='Sales by Category', width=1100, height=550, yaxis_title='TY.Sales', template='plotly_white')
st.plotly_chart(fig)  # Display the figure in Streamlit

# Calling the function with data
#create_bar_plot(Cat1, index=Cat1.index, value='TY.Sales', title='Sales Chart')



# In[6]:


fig1 = go.Figure()
fig1.add_trace(go.Pie(labels=Cat1.index, values=Cat1['Sales%'], hole=0.3))
fig1.update_layout(title='Sales Percentage by Category', width=1000, height=1000,legend=dict(
        x=1.5,  # Position the legend outside the pie chart on the right
        y=1,  # Align the legend with the top of the pie chart
        xanchor='left',  # Anchor the legend to the left of the x position
        bgcolor='rgba(255, 255, 255, 0.5)',  # Optional: background color with transparency
        bordercolor='Black',  # Optional: border color
        borderwidth=1  # Optional: border width
    ), template='plotly_white')
#    return fig

# Example dataset

# Bar plot for TY.Sales
#bar_fig = create_bar_plot(x=Cat1.index, y=Cat1['TY.Sales'], title='TY.Sales by Category')

# Pie chart for Sales%
#pie_fig = create_pie_chart(labels=Cat1.index, values=Cat1['Sales%'], title='Sales Percentage by Category')

# Display the interactive plots
st.plotly_chart(fig1)

