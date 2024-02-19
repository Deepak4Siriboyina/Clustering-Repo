# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:10.925165Z","iopub.execute_input":"2024-02-10T10:50:10.925768Z","iopub.status.idle":"2024-02-10T10:50:10.939308Z","shell.execute_reply.started":"2024-02-10T10:50:10.925711Z","shell.execute_reply":"2024-02-10T10:50:10.938111Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

'''
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
'''

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [markdown]
# # 1. Introduction

# %% [markdown]
# - The dataset under consideration represents a transnational record of transactions spanning from December 1, 2010, to December 9, 2011, for a UK-based online retail company specializing in unique all-occasion gifts.
# - As a registered non-store online retailer, the company caters to a diverse customer base, with a significant portion comprising wholesalers. 
# - With an extensive range of products and a broad customer demographic, understanding and effectively segmenting customers is crucial for optimizing marketing strategies, improving customer satisfaction, and ultimately driving profitability.

# %% [markdown]
# # 2. Problem Statement

# %% [markdown]
# ![image.png](attachment:f9ee1e1f-f038-4a92-a595-bb59be81ee51.png)

# %% [markdown]
# The objective of this clustering project is to **segment** the company's customer base based on their purchasing behavior, with the aim of gaining insights into distinct customer segments and tailoring marketing strategies accordingly. 
# - By identifying homogeneous groups of customers with similar purchasing patterns, we seek to address the following key questions:
# 
#     - What are the different customer segments within the dataset, and how do they differ in terms of purchasing behavior, transaction frequency, and monetary value?
#     - How can we effectively categorize customers into meaningful clusters to facilitate targeted marketing efforts and personalized customer experiences?
#     - What actionable insights can be derived from the identified customer segments to optimize marketing strategies, improve customer retention, and drive revenue growth?
# 
# 
# We aim to uncover hidden patterns within the data and provide actionable insights that can inform strategic decision-making and enhance the company's competitive advantage in the online retail landscape.

# %% [markdown]
# # 3. Importing the Libraries

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:10.941554Z","iopub.execute_input":"2024-02-10T10:50:10.942005Z","iopub.status.idle":"2024-02-10T10:50:10.948573Z","shell.execute_reply.started":"2024-02-10T10:50:10.941964Z","shell.execute_reply":"2024-02-10T10:50:10.947386Z"}}
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import datetime
#import warnings
import plotly.graph_objs as go  
import plotly.express as px
#warnings.simplefilter('ignore')
#pip install --upgrade pandas
#pip install --upgrade jinja2
 
# %% [markdown]
# # 4. Data Acquisition

# %% [markdown]
#  - This is a transnational data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail.

# %% [markdown]
# #### Data Author: Daqing Chen
# - chend@lsbu.ac.uk
# - School of Engineering, London South Bank University

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:10.949800Z","iopub.execute_input":"2024-02-10T10:50:10.950164Z","iopub.status.idle":"2024-02-10T10:50:12.035371Z","shell.execute_reply.started":"2024-02-10T10:50:10.950137Z","shell.execute_reply":"2024-02-10T10:50:12.033915Z"}}
data =  pd.read_csv('Data sets/data_source.csv', encoding='unicode_escape')
print('Shape of the data:', data.shape)
data.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:12.037415Z","iopub.execute_input":"2024-02-10T10:50:12.037760Z","iopub.status.idle":"2024-02-10T10:50:12.201132Z","shell.execute_reply.started":"2024-02-10T10:50:12.037733Z","shell.execute_reply":"2024-02-10T10:50:12.199488Z"}}
data.info()

# %% [markdown]
# ### Observations:
#     
# The dataset comprises **5,41,909** rows and **8** columns. Here is a concise overview of the column information.
#     
# - **InvoiceNo**: This column holds unique invoice numbers (object data type), each representing transactions that may involve multiple purchased items.
#    
#     
# - **StockCode**: An object data type column representing the product code for each item. 
# 
#     
# - **Description**: This object column, describing products, has 540,455 non-null entries out of 541,909, with some missing values.
# 
# - **Quantity**: This integer column represents the quantity of products bought in each transaction.
#     
# - **InvoiceDate**: This object column indicates the date of each transaction.
#     
# - **UnitPrice**: This float column denotes the price per unit for each product.
#     
# - **CustomerID**: This float column represents the unique ID for each customer.
#     
# - **Country**: This object column denotes the country of purchase for each product.
#     
# From initial observations, we can see there are missing values in the fields of **Description** and **CustomerID**.

# %% [markdown]
# # 5. Data Pre-profiling

# %% [markdown]
# - Summary statistics for all the Continous variables.

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:12.202192Z","iopub.execute_input":"2024-02-10T10:50:12.202552Z","iopub.status.idle":"2024-02-10T10:50:12.290135Z","shell.execute_reply.started":"2024-02-10T10:50:12.202501Z","shell.execute_reply":"2024-02-10T10:50:12.289005Z"}}
data.describe()

# %% [markdown]
# - Summary statistics for all the Categorical variables.

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:12.291766Z","iopub.execute_input":"2024-02-10T10:50:12.292441Z","iopub.status.idle":"2024-02-10T10:50:12.690158Z","shell.execute_reply.started":"2024-02-10T10:50:12.292401Z","shell.execute_reply":"2024-02-10T10:50:12.688995Z"}}
data.describe(include='object').T

# %% [markdown] {"execution":{"iopub.execute_input":"2024-01-17T08:37:46.660591Z","iopub.status.busy":"2024-01-17T08:37:46.660156Z","iopub.status.idle":"2024-01-17T08:37:46.667591Z","shell.execute_reply":"2024-01-17T08:37:46.666061Z","shell.execute_reply.started":"2024-01-17T08:37:46.660556Z"}}
# - Check for any Missing values.

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:12.691685Z","iopub.execute_input":"2024-02-10T10:50:12.692338Z","iopub.status.idle":"2024-02-10T10:50:12.829430Z","shell.execute_reply.started":"2024-02-10T10:50:12.692300Z","shell.execute_reply":"2024-02-10T10:50:12.828124Z"}}
100 *(data.isnull().sum()/data.shape[0])

# %% [markdown]
# - We have **25%** missing values in the field **CustomerID** and **0.26%** missing values in the field of **Description**

# %% [markdown]
# - Check for any Duplicate records.

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:12.830931Z","iopub.execute_input":"2024-02-10T10:50:12.831253Z","iopub.status.idle":"2024-02-10T10:50:13.216743Z","shell.execute_reply.started":"2024-02-10T10:50:12.831228Z","shell.execute_reply":"2024-02-10T10:50:13.215535Z"}}
duplicate_rec = data[data.duplicated(keep=False)]

d_r_sorted = duplicate_rec.sort_values(by=['InvoiceNo', 'StockCode', 'Description', 'CustomerID', 'Quantity'])
print(d_r_sorted.shape)
d_r_sorted.head(15)

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:13.220085Z","iopub.execute_input":"2024-02-10T10:50:13.220443Z","iopub.status.idle":"2024-02-10T10:50:13.227252Z","shell.execute_reply.started":"2024-02-10T10:50:13.220412Z","shell.execute_reply":"2024-02-10T10:50:13.226106Z"}}
100 * (d_r_sorted.shape[0]/data.shape[0])

# %% [markdown]
# - We've identified nearly **2%** duplicate records in the dataset.
# - Addressing these duplicates is crucial before proceeding with clustering, as they can introduce substantial noise to the data and hinder the effectiveness of the clustering process.

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:13.228653Z","iopub.execute_input":"2024-02-10T10:50:13.229050Z","iopub.status.idle":"2024-02-10T10:50:13.905980Z","shell.execute_reply.started":"2024-02-10T10:50:13.229017Z","shell.execute_reply":"2024-02-10T10:50:13.904592Z"}}
print(data[data.duplicated()].shape)
data[data.duplicated()]

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:13.907197Z","iopub.execute_input":"2024-02-10T10:50:13.907503Z","iopub.status.idle":"2024-02-10T10:50:14.241781Z","shell.execute_reply.started":"2024-02-10T10:50:13.907478Z","shell.execute_reply":"2024-02-10T10:50:14.240578Z"}}
100*(data[data.duplicated()].shape[0]/data.shape[0])

# %% [markdown]
# - We've identified **5268** unique values repeating in the dataset, contributing to a total of **10147** duplicate entries. 
# - To ensure data integrity, we should remove one instance of each duplicated record.

# %% [markdown]
# # 6. Data Pre-processing

# %% [markdown]
# - Removing rows with missing values in '**CustomerID**' and '**Description**' columns.

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:14.243003Z","iopub.execute_input":"2024-02-10T10:50:14.243311Z","iopub.status.idle":"2024-02-10T10:50:14.349823Z","shell.execute_reply.started":"2024-02-10T10:50:14.243286Z","shell.execute_reply":"2024-02-10T10:50:14.348594Z"}}
data.dropna(subset=['CustomerID', 'Description'], inplace=True)
data = data.reset_index(drop=True)

# %% [markdown]
# - Eliminating observed duplicate instances identified during the pre-profiling phase.

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:14.352543Z","iopub.execute_input":"2024-02-10T10:50:14.353086Z","iopub.status.idle":"2024-02-10T10:50:14.664770Z","shell.execute_reply.started":"2024-02-10T10:50:14.353041Z","shell.execute_reply":"2024-02-10T10:50:14.663472Z"}}
data.drop_duplicates(inplace=True)
data = data.reset_index(drop=True)

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:14.666621Z","iopub.execute_input":"2024-02-10T10:50:14.667020Z","iopub.status.idle":"2024-02-10T10:50:14.673692Z","shell.execute_reply.started":"2024-02-10T10:50:14.666980Z","shell.execute_reply":"2024-02-10T10:50:14.672820Z"}}
data.shape

# %% [markdown]
# # 7. Data Post-profiling

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:14.674802Z","iopub.execute_input":"2024-02-10T10:50:14.675520Z","iopub.status.idle":"2024-02-10T10:50:14.916182Z","shell.execute_reply.started":"2024-02-10T10:50:14.675477Z","shell.execute_reply":"2024-02-10T10:50:14.914834Z"}}
data[data.duplicated()].shape

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:14.917828Z","iopub.execute_input":"2024-02-10T10:50:14.918255Z","iopub.status.idle":"2024-02-10T10:50:15.024105Z","shell.execute_reply.started":"2024-02-10T10:50:14.918218Z","shell.execute_reply":"2024-02-10T10:50:15.022945Z"}}
100 *(data.isnull().sum()/data.shape[0])

# %% [markdown]
# - All the duplicated and missing values have been removed.

# %% [markdown]
# #### Dealing with cancelled orders.

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:15.025434Z","iopub.execute_input":"2024-02-10T10:50:15.025850Z","iopub.status.idle":"2024-02-10T10:50:15.177027Z","shell.execute_reply.started":"2024-02-10T10:50:15.025814Z","shell.execute_reply":"2024-02-10T10:50:15.175860Z"}}
Cancelled_orders = data[data['InvoiceNo'].str.contains('C')]
print('Shape:', Cancelled_orders.shape)
Cancelled_orders.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:15.178371Z","iopub.execute_input":"2024-02-10T10:50:15.178680Z","iopub.status.idle":"2024-02-10T10:50:15.185733Z","shell.execute_reply.started":"2024-02-10T10:50:15.178655Z","shell.execute_reply":"2024-02-10T10:50:15.184701Z"}}
100*(Cancelled_orders.shape[0]/data.shape[0])

# %% [markdown]
# - Cancelled orders are characterized by quantities denoted with a negative sign, constituting approximately **2.21%** of the total orders. 
# - To enhance the analysis, we propose removing these cancelled observations as they have minimal contribution.

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:15.187290Z","iopub.execute_input":"2024-02-10T10:50:15.187613Z","iopub.status.idle":"2024-02-10T10:50:15.212131Z","shell.execute_reply.started":"2024-02-10T10:50:15.187587Z","shell.execute_reply":"2024-02-10T10:50:15.210613Z"}}
Cancelled_orders.describe()

# %% [markdown]
# #####  Observations:
# - All quantities in cancelled transactions are negative, confirming their nature. 
# - The UnitPrice column exhibits a significant spread, reflecting a diverse range of products involved in cancelled transactions, spanning from low to high values.

# %% [markdown]
# # 8. Feature Engineering

# %% [markdown]
# - RFM is a method used for analyzing customer value and segmenting customers which is commonly used in the retail and professional services industries.
# - Its acronym stands for **Recency**, **Frequency** and **Monetary Value** of a customer.
# 
# - **Recency**: Recency assesses how recently a customer made a purchase. Customers with recent purchases, usually within the last few weeks, are more likely to have the product and brand at the forefront of their minds, increasing the likelihood of a repeat purchase.
# 
# - **Frequency**: Frequency indicates how often a customer makes purchases, offering insights into repeat buying patterns. Identifying customers with frequent repeat purchases is crucial, as it helps in recognizing individuals likely to continue shopping with your brand beyond their initial purchase.
# 
# - **Monetary value**: Monetary value signifies the total amount a customer spends within a specific period. This metric is crucial for understanding consumer behavior. For instance, customers with the highest monetary value may not make frequent purchases but tend to invest in more expensive products when they do.

# %% [markdown]
# - Data after removing the Cancelled orders.

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:15.214013Z","iopub.execute_input":"2024-02-10T10:50:15.214820Z","iopub.status.idle":"2024-02-10T10:50:15.391658Z","shell.execute_reply.started":"2024-02-10T10:50:15.214787Z","shell.execute_reply":"2024-02-10T10:50:15.390576Z"}}
new_df = data[~(data['InvoiceNo'].str.contains('C'))]
print('Shape of the final data:', new_df.shape)
new_df.head()

# %% [markdown]
# - **Recency**: Calculating the number of days since the purchase.

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:15.392626Z","iopub.execute_input":"2024-02-10T10:50:15.392936Z","iopub.status.idle":"2024-02-10T10:50:15.399708Z","shell.execute_reply.started":"2024-02-10T10:50:15.392911Z","shell.execute_reply":"2024-02-10T10:50:15.398425Z"}}
df = new_df
df.shape

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:15.401016Z","iopub.execute_input":"2024-02-10T10:50:15.401331Z","iopub.status.idle":"2024-02-10T10:50:15.529222Z","shell.execute_reply.started":"2024-02-10T10:50:15.401306Z","shell.execute_reply":"2024-02-10T10:50:15.528075Z"}}
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df.head(2)

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:15.530701Z","iopub.execute_input":"2024-02-10T10:50:15.531675Z","iopub.status.idle":"2024-02-10T10:50:15.649735Z","shell.execute_reply.started":"2024-02-10T10:50:15.531634Z","shell.execute_reply":"2024-02-10T10:50:15.648559Z"}}
df['DayofPurchase'] = df['InvoiceDate'].dt.date
df.head(2)

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:15.657260Z","iopub.execute_input":"2024-02-10T10:50:15.658181Z","iopub.status.idle":"2024-02-10T10:50:16.017830Z","shell.execute_reply.started":"2024-02-10T10:50:15.658144Z","shell.execute_reply":"2024-02-10T10:50:16.017020Z"}}
Customer_df = df.groupby('CustomerID')['DayofPurchase'].max().reset_index()
Customer_df.head(2)

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:16.018793Z","iopub.execute_input":"2024-02-10T10:50:16.019670Z","iopub.status.idle":"2024-02-10T10:50:16.126073Z","shell.execute_reply.started":"2024-02-10T10:50:16.019639Z","shell.execute_reply":"2024-02-10T10:50:16.124944Z"}}
df.info() 

# %% [markdown]
# - We observe that the 'DayofPurchase' column is currently of object type; it needs to be converted to a Datetime data type.

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:16.127459Z","iopub.execute_input":"2024-02-10T10:50:16.127953Z","iopub.status.idle":"2024-02-10T10:50:16.173125Z","shell.execute_reply.started":"2024-02-10T10:50:16.127916Z","shell.execute_reply":"2024-02-10T10:50:16.171676Z"}}
mrdate = df['DayofPurchase'].max() #mrdate = Identifying the most recent purchase date
Customer_df['DayofPurchase'] = pd.to_datetime(Customer_df['DayofPurchase'])
mrdate = pd.to_datetime(mrdate)

Customer_df['Days since last Purchase'] = (mrdate - Customer_df['DayofPurchase']).dt.days
Customer_df.head(2)

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:16.174463Z","iopub.execute_input":"2024-02-10T10:50:16.175548Z","iopub.status.idle":"2024-02-10T10:50:16.187257Z","shell.execute_reply.started":"2024-02-10T10:50:16.175486Z","shell.execute_reply":"2024-02-10T10:50:16.186100Z"}}
Customer_df.drop(columns=['DayofPurchase'], inplace=True) #We no longer need the column 'DayofPurchase'
Customer_df.head()

# %% [markdown]
# - **Frequency**: Calculating the customer's order count.

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:16.188338Z","iopub.execute_input":"2024-02-10T10:50:16.188679Z","iopub.status.idle":"2024-02-10T10:50:16.251675Z","shell.execute_reply.started":"2024-02-10T10:50:16.188626Z","shell.execute_reply":"2024-02-10T10:50:16.250781Z"}}
Total_Orders = df.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
Total_Orders.rename(columns={'InvoiceNo':'No.of.Orders'}, inplace = True)

Total_Orders.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:16.252652Z","iopub.execute_input":"2024-02-10T10:50:16.253150Z","iopub.status.idle":"2024-02-10T10:50:16.271082Z","shell.execute_reply.started":"2024-02-10T10:50:16.253124Z","shell.execute_reply":"2024-02-10T10:50:16.269719Z"}}
Customer_df = pd.merge(Customer_df, Total_Orders, on='CustomerID')
Customer_df.head()

# %% [markdown]
# - **Monetary Value**: Calculating the amount spent per transaction.

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:16.272383Z","iopub.execute_input":"2024-02-10T10:50:16.272719Z","iopub.status.idle":"2024-02-10T10:50:16.309847Z","shell.execute_reply.started":"2024-02-10T10:50:16.272690Z","shell.execute_reply":"2024-02-10T10:50:16.308580Z"}}
df['Amount'] = df['Quantity']*df['UnitPrice']
Amount = df.groupby('CustomerID')['Amount'].sum().reset_index()
Amount.head(2)

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:16.312484Z","iopub.execute_input":"2024-02-10T10:50:16.312859Z","iopub.status.idle":"2024-02-10T10:50:16.326386Z","shell.execute_reply.started":"2024-02-10T10:50:16.312828Z","shell.execute_reply":"2024-02-10T10:50:16.324994Z"}}
Customer_df = pd.merge(Customer_df, Amount, on='CustomerID')
Customer_df.head(2)

# %% [markdown]
# - Calculating for favorite day of the week for a customer to purchase

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:16.328468Z","iopub.execute_input":"2024-02-10T10:50:16.328949Z","iopub.status.idle":"2024-02-10T10:50:16.359761Z","shell.execute_reply.started":"2024-02-10T10:50:16.328921Z","shell.execute_reply":"2024-02-10T10:50:16.358582Z"}}
df['Day.of.the.Week'] = df['InvoiceDate'].dt.dayofweek
df.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:16.361545Z","iopub.execute_input":"2024-02-10T10:50:16.361970Z","iopub.status.idle":"2024-02-10T10:50:16.401555Z","shell.execute_reply.started":"2024-02-10T10:50:16.361932Z","shell.execute_reply":"2024-02-10T10:50:16.400732Z"}}
PreferredDay = df.groupby(['CustomerID', 'Day.of.the.Week']).size().reset_index(name='Count')
PreferredDay.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:16.402685Z","iopub.execute_input":"2024-02-10T10:50:16.403566Z","iopub.status.idle":"2024-02-10T10:50:16.421420Z","shell.execute_reply.started":"2024-02-10T10:50:16.403504Z","shell.execute_reply":"2024-02-10T10:50:16.420590Z"}}
PreferredDay = PreferredDay.loc[PreferredDay.groupby('CustomerID')['Count'].idxmax()][['CustomerID', 'Day.of.the.Week']]
PreferredDay.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:16.422914Z","iopub.execute_input":"2024-02-10T10:50:16.423844Z","iopub.status.idle":"2024-02-10T10:50:16.437099Z","shell.execute_reply.started":"2024-02-10T10:50:16.423809Z","shell.execute_reply":"2024-02-10T10:50:16.436238Z"}}
Customer_df =pd.merge(Customer_df, PreferredDay, on='CustomerID')
Customer_df.head()

# %% [markdown]
# - **Reference**: In Pandas, The day of the week works as Monday=0, Sunday=6.

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:16.438074Z","iopub.execute_input":"2024-02-10T10:50:16.439144Z","iopub.status.idle":"2024-02-10T10:50:16.503880Z","shell.execute_reply.started":"2024-02-10T10:50:16.439113Z","shell.execute_reply":"2024-02-10T10:50:16.502667Z"}}
Origin_Country = df.groupby(['CustomerID', 'Country']).size().reset_index(name='No.of.Orders')
Origin_Country.drop(columns = 'No.of.Orders', inplace=True)
Origin_Country.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:16.505160Z","iopub.execute_input":"2024-02-10T10:50:16.505482Z","iopub.status.idle":"2024-02-10T10:50:16.524648Z","shell.execute_reply.started":"2024-02-10T10:50:16.505453Z","shell.execute_reply":"2024-02-10T10:50:16.523420Z"}}
Customer_df = pd.merge(Customer_df, Origin_Country, on='CustomerID' )
Customer_df.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:16.525847Z","iopub.execute_input":"2024-02-10T10:50:16.526162Z","iopub.status.idle":"2024-02-10T10:50:16.532763Z","shell.execute_reply.started":"2024-02-10T10:50:16.526136Z","shell.execute_reply":"2024-02-10T10:50:16.531539Z"}}
Customer_df.shape

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:16.534142Z","iopub.execute_input":"2024-02-10T10:50:16.534452Z","iopub.status.idle":"2024-02-10T10:50:16.549627Z","shell.execute_reply.started":"2024-02-10T10:50:16.534425Z","shell.execute_reply":"2024-02-10T10:50:16.548492Z"}}
Customer_df.info()

# %% [markdown]
# - This represents the final customer data that will be used for segmentation.

# %% [markdown]
# # 9. Exploratory Data Analysis

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:16.552700Z","iopub.execute_input":"2024-02-10T10:50:16.553059Z","iopub.status.idle":"2024-02-10T10:50:16.579657Z","shell.execute_reply.started":"2024-02-10T10:50:16.553031Z","shell.execute_reply":"2024-02-10T10:50:16.578453Z"}}
Customer_df.describe()

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:16.581077Z","iopub.execute_input":"2024-02-10T10:50:16.581505Z","iopub.status.idle":"2024-02-10T10:50:16.594428Z","shell.execute_reply.started":"2024-02-10T10:50:16.581463Z","shell.execute_reply":"2024-02-10T10:50:16.593323Z"}}
Customer_df.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:16.596114Z","iopub.execute_input":"2024-02-10T10:50:16.596655Z","iopub.status.idle":"2024-02-10T10:50:16.606257Z","shell.execute_reply.started":"2024-02-10T10:50:16.596615Z","shell.execute_reply":"2024-02-10T10:50:16.604742Z"}}
Customer_df['Country'].nunique()

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:16.607260Z","iopub.execute_input":"2024-02-10T10:50:16.607623Z","iopub.status.idle":"2024-02-10T10:50:16.704761Z","shell.execute_reply.started":"2024-02-10T10:50:16.607592Z","shell.execute_reply":"2024-02-10T10:50:16.703581Z"}}
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(Customer_df['Country'])

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:16.705880Z","iopub.execute_input":"2024-02-10T10:50:16.706181Z","iopub.status.idle":"2024-02-10T10:50:16.714041Z","shell.execute_reply.started":"2024-02-10T10:50:16.706155Z","shell.execute_reply":"2024-02-10T10:50:16.712915Z"}}
Customer_df['Country_enc'] = pd.DataFrame(le.transform(Customer_df['Country']))

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:16.715613Z","iopub.execute_input":"2024-02-10T10:50:16.716028Z","iopub.status.idle":"2024-02-10T10:50:16.736250Z","shell.execute_reply.started":"2024-02-10T10:50:16.715999Z","shell.execute_reply":"2024-02-10T10:50:16.735225Z"}}
Customer_df.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:16.737700Z","iopub.execute_input":"2024-02-10T10:50:16.738022Z","iopub.status.idle":"2024-02-10T10:50:16.754170Z","shell.execute_reply.started":"2024-02-10T10:50:16.737995Z","shell.execute_reply":"2024-02-10T10:50:16.752955Z"}}
Customer_df_new = Customer_df.drop(columns= 'Country')
Customer_df_new.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:16.755383Z","iopub.execute_input":"2024-02-10T10:50:16.755789Z","iopub.status.idle":"2024-02-10T10:50:16.769278Z","shell.execute_reply.started":"2024-02-10T10:50:16.755748Z","shell.execute_reply":"2024-02-10T10:50:16.768110Z"}}
Customer_df_new['Day.of.the.Week'].value_counts()

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:16.770440Z","iopub.execute_input":"2024-02-10T10:50:16.771377Z","iopub.status.idle":"2024-02-10T10:50:17.087847Z","shell.execute_reply.started":"2024-02-10T10:50:16.771345Z","shell.execute_reply":"2024-02-10T10:50:17.086633Z"}}
plt.figure(figsize=(10,6))
sns.histplot(Customer_df_new['Day.of.the.Week'], color='olive')

plt.xticks(size=12)
plt.yticks(size=12)
plt.xlabel(xlabel='Day of the Week', size=14)
plt.ylabel(ylabel='Number of Customers Preferring that Day', size=14)
plt.title(label='Preference in a Week', size=16)
plt.show()

# %% [markdown]
# - Suprisingly, there are no orders on **Saturday**. 
# - **Thursday** emerges as the preferred shopping day for the majority of customers compared to other days of the week.

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:17.089325Z","iopub.execute_input":"2024-02-10T10:50:17.089808Z","iopub.status.idle":"2024-02-10T10:50:17.103241Z","shell.execute_reply.started":"2024-02-10T10:50:17.089760Z","shell.execute_reply":"2024-02-10T10:50:17.102192Z"}}
Customer_df_new['Days since last Purchase'].describe()

# %% [markdown]
# - On average, customers have made a purchase in the past **92 days**.
# - Half of the customers made a purchase in the past **50 days**.
# - **25%** of customers made a purchase in less than **3** weeks.

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:17.105981Z","iopub.execute_input":"2024-02-10T10:50:17.106318Z","iopub.status.idle":"2024-02-10T10:50:17.470796Z","shell.execute_reply.started":"2024-02-10T10:50:17.106290Z","shell.execute_reply":"2024-02-10T10:50:17.469389Z"}}
plt.figure(figsize=(10,6))
sns.histplot(Customer_df_new['Days since last Purchase'], kde=True, color='olivedrab')

plt.xticks(size=12)
plt.yticks(size=12)
plt.xlabel(xlabel='Days since last Purchase', size=14)
plt.ylabel(ylabel='No. of Customers', size=14)
plt.title(label='Distibution of Recency', size=16)
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:17.472325Z","iopub.execute_input":"2024-02-10T10:50:17.472784Z","iopub.status.idle":"2024-02-10T10:50:17.484238Z","shell.execute_reply.started":"2024-02-10T10:50:17.472745Z","shell.execute_reply":"2024-02-10T10:50:17.482974Z"}}
Customer_df_new['Amount'].describe()

# %% [markdown]
# - On average, each customer has a monetary value of **$2000**.

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:17.485457Z","iopub.execute_input":"2024-02-10T10:50:17.485793Z","iopub.status.idle":"2024-02-10T10:50:17.510267Z","shell.execute_reply.started":"2024-02-10T10:50:17.485767Z","shell.execute_reply":"2024-02-10T10:50:17.509077Z"}}
Customer_df_new.loc[Customer_df_new['Amount'].nlargest(10).index]

# %% [markdown]
# - Top **10** Customers by Transactional Monetary Value.
# - And we can see that majority of them are from Label **35** ie., the **United Kingdom.**

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:17.511712Z","iopub.execute_input":"2024-02-10T10:50:17.512029Z","iopub.status.idle":"2024-02-10T10:50:17.528888Z","shell.execute_reply.started":"2024-02-10T10:50:17.512003Z","shell.execute_reply":"2024-02-10T10:50:17.527950Z"}}
Customer_df_new.loc[Customer_df_new['No.of.Orders'].nlargest(10).index]

# %% [markdown]
# - Top **10** Customers by Transactional Order Volume – Lot Size.
# - We can observe that majority of them are ordering on **Wednesday**.
# - And also, we can see that majority of them are from again Label **35** ie., the **United Kingdom.**

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:17.530308Z","iopub.execute_input":"2024-02-10T10:50:17.530613Z","iopub.status.idle":"2024-02-10T10:50:17.541899Z","shell.execute_reply.started":"2024-02-10T10:50:17.530588Z","shell.execute_reply":"2024-02-10T10:50:17.540775Z"}}
Avg_orders = Customer_df.groupby('Country')['No.of.Orders'].mean().reset_index()
Avg_orders.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:17.543303Z","iopub.execute_input":"2024-02-10T10:50:17.543652Z","iopub.status.idle":"2024-02-10T10:50:17.558408Z","shell.execute_reply.started":"2024-02-10T10:50:17.543623Z","shell.execute_reply":"2024-02-10T10:50:17.556864Z"}}
Avg_amount = Customer_df.groupby('Country')['Amount'].mean().reset_index()
Avg_amount.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:17.560327Z","iopub.execute_input":"2024-02-10T10:50:17.561304Z","iopub.status.idle":"2024-02-10T10:50:17.575558Z","shell.execute_reply.started":"2024-02-10T10:50:17.561269Z","shell.execute_reply":"2024-02-10T10:50:17.574301Z"}}
MVC = pd.merge(Avg_orders, Avg_amount, on='Country')
MVC.head()

# %% [markdown]
# - Engineering a new field, **ATV** - Average Transaction Value.

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:17.577021Z","iopub.execute_input":"2024-02-10T10:50:17.577872Z","iopub.status.idle":"2024-02-10T10:50:17.594881Z","shell.execute_reply.started":"2024-02-10T10:50:17.577838Z","shell.execute_reply":"2024-02-10T10:50:17.593620Z"}}
MVC['ATV'] = MVC['Amount']/MVC['No.of.Orders']
MVC.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:17.596185Z","iopub.execute_input":"2024-02-10T10:50:17.596509Z","iopub.status.idle":"2024-02-10T10:50:17.615090Z","shell.execute_reply.started":"2024-02-10T10:50:17.596480Z","shell.execute_reply":"2024-02-10T10:50:17.614210Z"}}
MVC.loc[MVC['ATV'].nlargest(10).index]

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:17.616154Z","iopub.execute_input":"2024-02-10T10:50:17.617038Z","iopub.status.idle":"2024-02-10T10:50:17.983243Z","shell.execute_reply.started":"2024-02-10T10:50:17.617010Z","shell.execute_reply":"2024-02-10T10:50:17.982141Z"}}
plt.figure(figsize=(10,6))
MVC.groupby('Country')['ATV'].sum().sort_values(ascending=False)[:10].plot.barh(color='darkseagreen')

plt.xticks(size=12)
plt.yticks(size=12)
plt.xlabel(xlabel='Average Transaction Value($)', size=14)
plt.ylabel(ylabel='Country', size=14)
plt.title(label='Top 10 Countries with Highest ATV', size=16)
plt.show()

# %% [markdown]
# - Top **10** Countries with Highest Average Transactional Value – **High Ticket Sales**.

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:17.984479Z","iopub.execute_input":"2024-02-10T10:50:17.984820Z","iopub.status.idle":"2024-02-10T10:50:18.003681Z","shell.execute_reply.started":"2024-02-10T10:50:17.984793Z","shell.execute_reply":"2024-02-10T10:50:18.002539Z"}}
MVC.loc[MVC['ATV'].sort_values(ascending=False).index]

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:18.005026Z","iopub.execute_input":"2024-02-10T10:50:18.005641Z","iopub.status.idle":"2024-02-10T10:50:18.525081Z","shell.execute_reply.started":"2024-02-10T10:50:18.005609Z","shell.execute_reply":"2024-02-10T10:50:18.523891Z"}}
plt.figure(figsize=(15,8))
MVC.groupby('Country')['ATV'].sum().sort_values(ascending=True).plot.barh(color='lightseagreen')

plt.xticks(size=12)
plt.yticks(size=10)
plt.xlabel(xlabel='Average Transaction Value($)', size=14)
plt.ylabel(ylabel='Country', rotation=90, size=14)
plt.title(label='Countries interms of High ATV', size=16)
plt.show()

# %% [markdown]
# - Despite the **United Kingdom** having the **highest number of customers**, **maximum order count**, and **highest overall spending**, the average transactional value per customer (**ATV**) in the UK is notably lower compared to other countries. 
# - Leading in this aspect are **Singapore** and **Netherlands**, followed by **Australia** and **Japan**.

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:18.526301Z","iopub.execute_input":"2024-02-10T10:50:18.526739Z","iopub.status.idle":"2024-02-10T10:50:18.544957Z","shell.execute_reply.started":"2024-02-10T10:50:18.526697Z","shell.execute_reply":"2024-02-10T10:50:18.543894Z"}}
Customer_df_new.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T11:32:18.054186Z","iopub.execute_input":"2024-02-10T11:32:18.054653Z","iopub.status.idle":"2024-02-10T11:32:18.323418Z","shell.execute_reply.started":"2024-02-10T11:32:18.054622Z","shell.execute_reply":"2024-02-10T11:32:18.321886Z"}}
plt.figure(figsize=(15,8))
Customer_df['Country'].value_counts()[1:20].plot(kind='pie', fontsize=8, startangle=0, cmap='Reds')

plt.title(label='Top 20 Countries by Customer Count (Excluding UK)', size=16)
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:18.820310Z","iopub.execute_input":"2024-02-10T10:50:18.821045Z","iopub.status.idle":"2024-02-10T10:50:19.379984Z","shell.execute_reply.started":"2024-02-10T10:50:18.821008Z","shell.execute_reply":"2024-02-10T10:50:19.378699Z"}}
plt.figure(figsize=(10,6))
sns.regplot(y='Amount', x='No.of.Orders', data=Customer_df_new, color='darkcyan')

plt.xticks(size=12)
plt.yticks(size=12)
plt.xlabel(xlabel='No. of Orders', size=14)
plt.ylabel(ylabel='Amount Spent', size=14)
plt.title('Amount Vs No. Of Orders')
plt.grid()
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:19.381419Z","iopub.execute_input":"2024-02-10T10:50:19.381870Z","iopub.status.idle":"2024-02-10T10:50:19.399539Z","shell.execute_reply.started":"2024-02-10T10:50:19.381827Z","shell.execute_reply":"2024-02-10T10:50:19.398364Z"}}
Customer_df_new.corr()

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:19.400884Z","iopub.execute_input":"2024-02-10T10:50:19.401299Z","iopub.status.idle":"2024-02-10T10:50:19.719471Z","shell.execute_reply.started":"2024-02-10T10:50:19.401259Z","shell.execute_reply":"2024-02-10T10:50:19.718624Z"}}
sns.heatmap(Customer_df_new.corr())

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:19.720577Z","iopub.execute_input":"2024-02-10T10:50:19.721475Z","iopub.status.idle":"2024-02-10T10:50:19.748684Z","shell.execute_reply.started":"2024-02-10T10:50:19.721444Z","shell.execute_reply":"2024-02-10T10:50:19.747554Z"}}
Customer_df_new.describe()

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:19.750331Z","iopub.execute_input":"2024-02-10T10:50:19.750694Z","iopub.status.idle":"2024-02-10T10:50:20.726459Z","shell.execute_reply.started":"2024-02-10T10:50:19.750663Z","shell.execute_reply":"2024-02-10T10:50:20.725140Z"}}
fig, ax = plt.subplots(3,1,  figsize=(10, 6))
plt.subplots_adjust(top = 2)

sns.histplot(Customer_df_new['Days since last Purchase'], kde=True, color='g', bins=50, ax=ax[0]);
sns.histplot(Customer_df_new['No.of.Orders'], color='b', kde=True, bins=50, ax=ax[1]);
sns.histplot(Customer_df_new['Amount'], color='r', kde=True, bins=50, ax=ax[2]);

plt.show()

# %% [markdown]
# - We observe **skewness** in the columns 'Days since last Purchase,' 'No. of Orders,' and 'Amount.' 

# %% [markdown]
# # 10. Feature Scaling

# %% [markdown]
# - **Scaling** the data ensures equal weighting and regularization.

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:20.736839Z","iopub.execute_input":"2024-02-10T10:50:20.737232Z","iopub.status.idle":"2024-02-10T10:50:20.753131Z","shell.execute_reply.started":"2024-02-10T10:50:20.737204Z","shell.execute_reply":"2024-02-10T10:50:20.751625Z"}}
Customer_df_new['ATV'] = Customer_df_new['Amount']/Customer_df_new['No.of.Orders']
Customer_df_new.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:20.754810Z","iopub.execute_input":"2024-02-10T10:50:20.755146Z","iopub.status.idle":"2024-02-10T10:50:20.769840Z","shell.execute_reply.started":"2024-02-10T10:50:20.755118Z","shell.execute_reply":"2024-02-10T10:50:20.768609Z"}}
df_to_be_scaled = Customer_df_new.drop(columns=['CustomerID', 'Day.of.the.Week', 'Country_enc'], axis=1)
df_to_be_scaled.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:20.771288Z","iopub.execute_input":"2024-02-10T10:50:20.771634Z","iopub.status.idle":"2024-02-10T10:50:20.791510Z","shell.execute_reply.started":"2024-02-10T10:50:20.771605Z","shell.execute_reply":"2024-02-10T10:50:20.790454Z"}}
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df_to_be_scaled)

scaled_data = scaler.transform(df_to_be_scaled)
scaled_df = pd.DataFrame(data=scaled_data, columns=df_to_be_scaled.columns)
scaled_df.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:20.793047Z","iopub.execute_input":"2024-02-10T10:50:20.793479Z","iopub.status.idle":"2024-02-10T10:50:20.808508Z","shell.execute_reply.started":"2024-02-10T10:50:20.793440Z","shell.execute_reply":"2024-02-10T10:50:20.807261Z"}}
scaled_df['CustomerID'] = Customer_df_new['CustomerID']
scaled_df.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:20.810085Z","iopub.execute_input":"2024-02-10T10:50:20.810425Z","iopub.status.idle":"2024-02-10T10:50:20.830014Z","shell.execute_reply.started":"2024-02-10T10:50:20.810395Z","shell.execute_reply":"2024-02-10T10:50:20.828827Z"}}
scaled_df['Day.of.the.Week'] = Customer_df_new['Day.of.the.Week']
scaled_df['Country_enc'] = Customer_df_new['Country_enc']
scaled_df.head()

# %% [markdown]
# # 11. Dimensionality Reduction

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:20.831243Z","iopub.execute_input":"2024-02-10T10:50:20.831586Z","iopub.status.idle":"2024-02-10T10:50:21.126951Z","shell.execute_reply.started":"2024-02-10T10:50:20.831553Z","shell.execute_reply":"2024-02-10T10:50:21.125493Z"}}
from sklearn.decomposition import PCA
scaled_df.set_index('CustomerID', inplace=True)

pca = PCA().fit(scaled_df)

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)
print(explained_variance_ratio)
print(cumulative_explained_variance)

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:21.128507Z","iopub.execute_input":"2024-02-10T10:50:21.133461Z","iopub.status.idle":"2024-02-10T10:50:21.141947Z","shell.execute_reply.started":"2024-02-10T10:50:21.133402Z","shell.execute_reply":"2024-02-10T10:50:21.140607Z"}}
max_variance_features = sorted(range(len(explained_variance_ratio)), key=lambda i: explained_variance_ratio[i], reverse=True)
max_variance_features

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:21.143253Z","iopub.execute_input":"2024-02-10T10:50:21.143634Z","iopub.status.idle":"2024-02-10T10:50:21.156168Z","shell.execute_reply.started":"2024-02-10T10:50:21.143602Z","shell.execute_reply":"2024-02-10T10:50:21.155033Z"}}
feature_names = scaled_df.columns

print("Top features contributing to maximum variance:")
for i in range(len(max_variance_features)):
    feature_index = max_variance_features[i]
    feature_name = feature_names[feature_index]
    print(f"\n'{feature_name}': Explained Variance Ratio - \n{(explained_variance_ratio[feature_index])*100}")

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:21.158213Z","iopub.execute_input":"2024-02-10T10:50:21.158648Z","iopub.status.idle":"2024-02-10T10:50:21.195270Z","shell.execute_reply.started":"2024-02-10T10:50:21.158605Z","shell.execute_reply":"2024-02-10T10:50:21.192172Z"}}
pca = PCA(n_components=4)

Customer_df_new_pca = pca.fit_transform(scaled_df)

Customer_df_new_pca = pd.DataFrame(Customer_df_new_pca, columns=['PC'+str(i+1) for i in range(pca.n_components_)])

Customer_df_new_pca.index = scaled_df.index
Customer_df_new_pca.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:21.196974Z","iopub.execute_input":"2024-02-10T10:50:21.197705Z","iopub.status.idle":"2024-02-10T10:50:21.339043Z","shell.execute_reply.started":"2024-02-10T10:50:21.197662Z","shell.execute_reply":"2024-02-10T10:50:21.337900Z"}}
# Define a function to highlight the top 3 absolute values in each column of a dataframe
def top3(column):
    top3 = column.abs().nlargest(3).index
    return ['background-color:  #F1F7B4' if i in top3 else '' for i in column.index]

# Create the PCA component DataFrame and apply the highlighting function
pc_df = pd.DataFrame(pca.components_.T, columns=['PC{}'.format(i+1) for i in range(pca.n_components_)],  
                     index=scaled_df.columns)

pc_df.style.apply(top3, axis=0)

# %% [markdown]
# - Extracted coefficients corresponding to each principle component.

# %% [markdown]
# # 12. Model Development

# %% [markdown]
# ####  **Elbow Analysis** - To find the optimum K value.

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:21.340503Z","iopub.execute_input":"2024-02-10T10:50:21.341188Z","iopub.status.idle":"2024-02-10T10:50:38.068691Z","shell.execute_reply.started":"2024-02-10T10:50:21.341150Z","shell.execute_reply":"2024-02-10T10:50:38.067464Z"}}
inertia_vals = []
k_vals = [x for x in range(1,16)]

from sklearn.cluster import KMeans
for k in k_vals:
    print('k_value is:', k)
    km = KMeans(n_clusters=k, max_iter=500, random_state=4)
    km.fit(scaled_df)
    inertia_vals.append(km.inertia_)

# %% [markdown]
# #####  Visualizing K_value vs Inertia

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:38.074008Z","iopub.execute_input":"2024-02-10T10:50:38.074404Z","iopub.status.idle":"2024-02-10T10:50:38.389281Z","shell.execute_reply.started":"2024-02-10T10:50:38.074376Z","shell.execute_reply":"2024-02-10T10:50:38.388161Z"}}
fig = plt.figure(figsize=(15, 7))
plt.plot([x for x in range(1,16)], inertia_vals, marker='o', 
         linestyle='dashed', markersize=10, markerfacecolor='red', c='green')

plt.xlabel(xlabel='K - (Number of Clusters)', fontsize=14)
plt.ylabel(ylabel='Inertia', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title(label='K_Value vs Inertia', fontsize=16)

plt.grid()
plt.show()

# %% [markdown]
# - Utilizing the interactive plot with **Plotly**.

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:38.390798Z","iopub.execute_input":"2024-02-10T10:50:38.391203Z","iopub.status.idle":"2024-02-10T10:50:38.922755Z","shell.execute_reply.started":"2024-02-10T10:50:38.391176Z","shell.execute_reply":"2024-02-10T10:50:38.921638Z"}}
fig = go.Figure()
fig.add_trace(go.Scatter(x=k_vals, y=inertia_vals, fill='toself', mode='lines+markers'))

fig.update_layout(xaxis= dict(tickmode='linear', tick0=1, dtick=1), 
                  title_text = 'K_values vs Inertia', 
                  title_x=0.5, xaxis_title='K - (No. of Clusters)',
                  yaxis_title='Inertia')
fig.show()

# %% [markdown]
# ##### From the above Elbow plot, we can consider the no. of clusters as 3.

# %% [markdown]
# ####  **Final Model** 

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:38.925582Z","iopub.execute_input":"2024-02-10T10:50:38.926317Z","iopub.status.idle":"2024-02-10T10:50:39.935017Z","shell.execute_reply.started":"2024-02-10T10:50:38.926283Z","shell.execute_reply":"2024-02-10T10:50:39.933840Z"}}
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=500, random_state=4)
kmeans.fit(Customer_df_new_pca)

from collections import Counter
cluster_frequencies = Counter(kmeans.labels_)

label_mapping = {label: new_label for new_label, (label, _) in 
                 enumerate(cluster_frequencies.most_common())}

label_mapping = {v: k for k, v in {2: 1, 1: 0, 0: 2}.items()}

new_labels = np.array([label_mapping[label] for label in kmeans.labels_])

Customer_df_new['cluster'] = new_labels

# Append the new cluster labels to the PCA version of the dataset
Customer_df_new_pca['cluster'] = new_labels


# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:39.936377Z","iopub.execute_input":"2024-02-10T10:50:39.936738Z","iopub.status.idle":"2024-02-10T10:50:39.951050Z","shell.execute_reply.started":"2024-02-10T10:50:39.936708Z","shell.execute_reply":"2024-02-10T10:50:39.949849Z"}}
Customer_df_new.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:39.952706Z","iopub.execute_input":"2024-02-10T10:50:39.953156Z","iopub.status.idle":"2024-02-10T10:50:39.963116Z","shell.execute_reply.started":"2024-02-10T10:50:39.953115Z","shell.execute_reply":"2024-02-10T10:50:39.962031Z"}}
centers = kmeans.cluster_centers_
centers

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:39.964915Z","iopub.execute_input":"2024-02-10T10:50:39.966042Z","iopub.status.idle":"2024-02-10T10:50:39.978845Z","shell.execute_reply.started":"2024-02-10T10:50:39.965995Z","shell.execute_reply":"2024-02-10T10:50:39.977732Z"}}
Customer_df_new_pca['cluster'].value_counts()

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:39.980804Z","iopub.execute_input":"2024-02-10T10:50:39.981578Z","iopub.status.idle":"2024-02-10T10:50:39.998810Z","shell.execute_reply.started":"2024-02-10T10:50:39.981520Z","shell.execute_reply":"2024-02-10T10:50:39.997624Z"}}
Customer_df_new_pca.head()

# %% [markdown]
# #### Visualizing the Clusters

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:40.000292Z","iopub.execute_input":"2024-02-10T10:50:40.001403Z","iopub.status.idle":"2024-02-10T10:50:40.010209Z","shell.execute_reply.started":"2024-02-10T10:50:40.001361Z","shell.execute_reply":"2024-02-10T10:50:40.009074Z"}}
colors = ['#FF0D00', '#0D00FF', '#00FF3A']

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:40.012034Z","iopub.execute_input":"2024-02-10T10:50:40.012581Z","iopub.status.idle":"2024-02-10T10:50:40.194113Z","shell.execute_reply.started":"2024-02-10T10:50:40.012549Z","shell.execute_reply":"2024-02-10T10:50:40.192837Z"}}
cluster_0 = Customer_df_new_pca[Customer_df_new_pca['cluster'] == 0]
cluster_1 = Customer_df_new_pca[Customer_df_new_pca['cluster'] == 1]
cluster_2 = Customer_df_new_pca[Customer_df_new_pca['cluster'] == 2]

# Create a 3D scatter plot
fig = go.Figure()

# Add data points for each cluster separately and specify the color
fig.add_trace(go.Scatter3d(x=cluster_0['PC1'], y=cluster_0['PC2'], z=cluster_0['PC3'], 
                           mode='markers', marker=dict(color=colors[0], size=5, opacity=0.4), name='Cluster 0'))
fig.add_trace(go.Scatter3d(x=cluster_1['PC1'], y=cluster_1['PC2'], z=cluster_1['PC3'], 
                           mode='markers', marker=dict(color=colors[1], size=5, opacity=0.4), name='Cluster 1'))
fig.add_trace(go.Scatter3d(x=cluster_2['PC1'], y=cluster_2['PC2'], z=cluster_2['PC3'], 
                           mode='markers', marker=dict(color=colors[2], size=5, opacity=0.4), name='Cluster 2'))

# Set the title and layout details
fig.update_layout(
    title=dict(text='3D Visualization of Customer Clusters in PCA Space', x=0.5),
    scene=dict(
        xaxis=dict(backgroundcolor="#E9FFA5", gridcolor='white', title='PC1'),
        yaxis=dict(backgroundcolor="#E9FFA5", gridcolor='white', title='PC2'),
        zaxis=dict(backgroundcolor="#E9FFA5", gridcolor='white', title='PC3'),
    ),
    width=900,
    height=800
)

# Show the plot
fig.show()

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:40.195387Z","iopub.execute_input":"2024-02-10T10:50:40.195763Z","iopub.status.idle":"2024-02-10T10:50:40.251779Z","shell.execute_reply.started":"2024-02-10T10:50:40.195730Z","shell.execute_reply":"2024-02-10T10:50:40.250619Z"}}
labels = Customer_df_new_pca['cluster'].value_counts().index[0:10]
values = Customer_df_new_pca['cluster'].value_counts().values[0:10]

# Initiate an empty figure
fig = go.Figure()

# Add a trace of bar to the figure
fig.add_trace(trace=go.Bar(x=values, 
                           y=labels, 
                           orientation='h',
                           marker=dict(color='rgba(69, 214, 179, 1.0)',
                                       line=dict(color='rgba(66, 5, 84, 1.0)', 
                                                 width=3))))

# Update the layout with some cosmetics
fig.update_layout(height=500, 
                  width=1000, 
                  title_text='Distribution of Customers', 
                  title_x=0.5, 
                  xaxis_title='No. of Customers', 
                  yaxis_title='Clusters')

# Display the figure
fig.show()

# %% [markdown]
# # 13. Clustering Evaluation

# %% [markdown]
# - **Silhouette score**: The silhouette score is a metric used to evaluate the quality of clusters in clustering algorithms. It measures how similar an object is to its own cluster (cohesion) compared to other clusters (separation). The silhouette score ranges from -1 to 1, where a higher score indicates that the object is well-matched to its own cluster and poorly matched to neighboring clusters.
# - **Calinski-Harabasz score**: The Calinski-Harabasz score, also known as the Variance Ratio Criterion, is another metric used to evaluate the quality of clusters in clustering algorithms. It measures the ratio of between-cluster dispersion to within-cluster dispersion. A higher Calinski-Harabasz score indicates better-defined clusters.
# - **Davies Bouldin score**: The Davies-Bouldin score is yet another metric used for evaluating clustering algorithms. It quantifies the "compactness" and "separation" of clusters, where lower values indicate better clustering.

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:40.252952Z","iopub.execute_input":"2024-02-10T10:50:40.253286Z","iopub.status.idle":"2024-02-10T10:50:40.729133Z","shell.execute_reply.started":"2024-02-10T10:50:40.253250Z","shell.execute_reply":"2024-02-10T10:50:40.727619Z"}}
no_of_obs = len(Customer_df_new_pca)

# Separate the features and the cluster labels
X = Customer_df_new_pca.drop('cluster', axis=1)
clusters = Customer_df_new_pca['cluster']

# Compute the metrics
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from tabulate import tabulate


Sil_score = silhouette_score(X, clusters)
Calinski_score = calinski_harabasz_score(X, clusters)
Davies_score = davies_bouldin_score(X, clusters)

table_data = [
    ["Number of Customers", no_of_obs],
    ["Silhouette Score", Sil_score],
    ["Calinski Harabasz Score", Calinski_score],
    ["Davies Bouldin Score", Davies_score]
]

print(tabulate(table_data, headers=["Evaluation Metric", "Score"], tablefmt='pretty'))


# Write scores to a file
with open("metrics.txt", 'w') as outfile:
        outfile.write("Silhouette Score: %2.1f%%\n" % Sil_score)
        outfile.write("Calinski Harabasz Score: %2.1f%%\n" % Calinski_score)
        outfile.write("Davies Bouldin Score: %2.1f%%\n" % Davies_score)

# %% [markdown]
# - The Silhouette score is **0.4360** which is considered average, but still suggests that the clustering result has relatively well-separated clusters.
# - The Calinski-Harabasz score of **11725** implies that the clustering result has very well-defined clusters with minimal overlap and high compactness. It indicates a strong clustering solution, with clear separation between the clusters.
# - The Davies Bouldin score of **0.7410** indicates that the clusters are reasonably well-separated, although there may be some degree of overlap or ambiguity between clusters. It suggests that the clustering result is generally good, but there may still be room for improvement.

# %% [markdown]
# ### Radar Chart Analysis

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:40.730775Z","iopub.execute_input":"2024-02-10T10:50:40.731096Z","iopub.status.idle":"2024-02-10T10:50:41.600115Z","shell.execute_reply.started":"2024-02-10T10:50:40.731070Z","shell.execute_reply":"2024-02-10T10:50:41.598825Z"}}
df_customer = Customer_df_new.set_index('CustomerID')

# Standardize the data (excluding the cluster column)
scaler = StandardScaler()
df_customer_standardized = scaler.fit_transform(df_customer.drop(columns=['cluster'], axis=1))

# Create a new dataframe with standardized values and add the cluster column back
df_customer_standardized = pd.DataFrame(df_customer_standardized, columns=df_customer.columns[:-1], index=df_customer.index)
df_customer_standardized['cluster'] = df_customer['cluster']

# Calculate the centroids of each cluster
cluster_centroids = df_customer_standardized.groupby('cluster').mean()

# Function to create a radar chart
def create_radar_chart(ax, angles, data, color, cluster):
    # Plot the data and fill the area
    ax.fill(angles, data, color=color, alpha=0.4)
    ax.plot(angles, data, color=color, linewidth=2, linestyle='solid')
    
    # Add a title
    ax.set_title(f'Cluster {cluster}', size=20, color=color, y=1.1)

# Set data
labels=np.array(cluster_centroids.columns)
num_vars = len(labels)

# Compute angle of each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# The plot is circular, so we need to "complete the loop" and append the start to the end
labels = np.concatenate((labels, [labels[0]]))
angles += angles[:1]

# Initialize the figure
fig, ax = plt.subplots(figsize=(20, 10), subplot_kw=dict(polar=True), nrows=1, ncols=3)

# Create radar chart for each cluster
for i, color in enumerate(colors):
    data = cluster_centroids.loc[i].tolist()
    data += data[:1]  # Complete the loop
    create_radar_chart(ax[i], angles, data, color, i)

# Add input data
ax[0].set_xticks(angles[:-1])
ax[0].set_xticklabels(labels[:-1])

ax[1].set_xticks(angles[:-1])
ax[1].set_xticklabels(labels[:-1])

ax[2].set_xticks(angles[:-1])
ax[2].set_xticklabels(labels[:-1])

# Add a grid
ax[0].grid(color='grey', linewidth=0.5)

# Display the plot
plt.tight_layout()
plt.show()
plt.savefig("Radar_chart.png", dpi=120)



# %% [markdown]
# # 14. Customer Categorization based on Clustering Analysis

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:41.601442Z","iopub.execute_input":"2024-02-10T10:50:41.601818Z","iopub.status.idle":"2024-02-10T10:50:41.618654Z","shell.execute_reply.started":"2024-02-10T10:50:41.601788Z","shell.execute_reply":"2024-02-10T10:50:41.617477Z"}}
Customer_df_new['Country'] = Customer_df['Country']
Customer_df_new.head()

# %% [markdown]
# #### The Cust_set1 represents customers belonging to the 'Red' segment of the Radar chart analysis:
# 
#  - These customers exhibit a moderate level of spending, but their transactions are not very frequent.
#  - They prefer to shop on Wednesdays and Tuesdays.
#  - Their average transaction value is relatively high, indicating substantial purchases when they shop.
#  - The majority of them are based in the UK.
# 
# Here is the list of customers in this cluster:

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:41.620190Z","iopub.execute_input":"2024-02-10T10:50:41.620736Z","iopub.status.idle":"2024-02-10T10:50:41.641997Z","shell.execute_reply.started":"2024-02-10T10:50:41.620688Z","shell.execute_reply":"2024-02-10T10:50:41.640907Z"}}
Cust_set1 = Customer_df_new[Customer_df_new['cluster'] == 0]
print(Cust_set1.shape)
Cust_set1.head()

# %% [markdown]
# #### The Cust_set2 represents customers belonging to the 'Blue' segment of the Radar chart analysis:
# 
# - These customers tend to spend less, with fewer transactions and products purchased.
# - Their average transaction value is the lowest among all customer groups, indicating less shopping activity compared to other segments.
# - The majority of them prefer to shop on Thursdays and Fridays, closer to the weekends, as indicated by the high Day_of_Week value.
# 
# Here is the list of customers in this cluster:

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:41.643928Z","iopub.execute_input":"2024-02-10T10:50:41.644965Z","iopub.status.idle":"2024-02-10T10:50:41.668665Z","shell.execute_reply.started":"2024-02-10T10:50:41.644921Z","shell.execute_reply":"2024-02-10T10:50:41.667487Z"}}
Cust_set2 = Customer_df_new[Customer_df_new['cluster'] == 1]
print(Cust_set2.shape)
Cust_set2.head()

# %% [markdown]
# #### The Cust_set3 represents customers belonging to the 'Green' segment of the Radar chart analysis:
# 
# - The majority of these customers are residing in Germany, France, and Belgium.
# - Their average transaction value is higher than the other two groups of customers, suggesting they may be considered as original high-ticket buyers.
# - The majority of them prefer to shop on Thursdays and Mondays.
# 
# Here is the list of customers in this cluster:

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:41.670046Z","iopub.execute_input":"2024-02-10T10:50:41.670461Z","iopub.status.idle":"2024-02-10T10:50:41.689569Z","shell.execute_reply.started":"2024-02-10T10:50:41.670430Z","shell.execute_reply":"2024-02-10T10:50:41.688260Z"}}
Cust_set3 = Customer_df_new[Customer_df_new['cluster'] == 2]
print(Cust_set3.shape)
Cust_set3.head()

# %% [markdown]
#  ### Pie chart illustrating the distribution of customers across clusters.

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T10:50:41.690897Z","iopub.execute_input":"2024-02-10T10:50:41.691214Z","iopub.status.idle":"2024-02-10T10:50:41.937456Z","shell.execute_reply.started":"2024-02-10T10:50:41.691187Z","shell.execute_reply":"2024-02-10T10:50:41.936026Z"}}
plt.figure(figsize=(15,8))
Customer_df_new['cluster'].value_counts().plot(kind='pie', fontsize=14, autopct='%1.2f%%', 
                                               textprops={'color':"brown"}, wedgeprops=dict(width=0.5), 
                                               shadow=True, startangle=180, cmap='viridis_r', legend=True)
plt.ylabel(ylabel='Count', size=14)
plt.title(label='Proportion of Clusters by No. of Customers', size=16)
plt.show()
plt.savefig("Clusters.png", dpi=120)

# %% [markdown]
# #### Here's a concise conclusion for each of the three customer sets:
# 
# **Cust_set1 (Yellow Segment):**
# 
# - These customers display moderate spending behavior, with fewer but substantial transactions.
# - They prefer shopping on Wednesdays and Tuesdays.
# - Majority of them are located in the UK.
# 
# 
# **Cust_set2 (Cyan Segment):**
# 
# - Customers in this segment tend to spend less, with fewer transactions and products purchased.
# - They show a preference for shopping on Thursdays and Fridays.
# - Their average transaction value is the lowest among all segments.
# 
# 
# **Cust_set3 (Purple Segment):**
# 
# - The majority of these customers are located in Germany, France, and Belgium.
# - They exhibit higher spending behavior compared to other segments, possibly representing high-ticket buyers.
# - Prefer shopping on Thursdays and Mondays.
# 
# 
# These conclusions provide valuable insights into the distinct characteristics and preferences of each customer segment, aiding in **targeted marketing strategies** and **personalized customer experiences**.

# %% [markdown]
# # 15. Conclusion

# %% [markdown]
# 
# 
# - Our data analysis journey encompassed a comprehensive approach, beginning with exploratory data analysis (**EDA**) to gain an understanding of the dataset's characteristics, trends, and patterns. 
# - Through statistical modeling techniques, we delved deeper into the relationships between variables, identifying **key factors** that influence customer behavior and purchasing patterns.
# 
# --------------------------------------------------------------------------
# 
# - Uncovering hidden patterns within the data was made possible through the application of **unsupervised machine learning**, particularly the **KMeans clustering** algorithm. 
# - By segmenting customers based on their purchasing behavior, we gained insights into distinct **customer groups** and their preferences.
# 
# --------------------------------------------------------------------------
# 
# - To further enhance our analysis, we employed dimensionality reduction techniques, notably **Principal Component Analysis (PCA)**, to effectively manage high-dimensional data and identify the **most important features** driving customer behavior.
# 
# --------------------------------------------------------------------------
# 
# - The visualization of our findings played a pivotal role in conveying complex insights in a clear and intuitive manner.
# - Leveraging powerful visualization libraries such as Seaborn, Matplotlib, and Plotly, we created compelling visualizations that provided valuable insights into **customer segments, purchasing trends**, and **geographical distributions**.
# 
# --------------------------------------------------------------------------
# 
# - Through our comprehensive analysis, we unearthed actionable insights with the potential to significantly impact the **profitability** of the online store. 
# - By understanding customer preferences, **optimizing marketing strategies**, and **tailoring product offerings**, we can drive **revenue growth** and enhance **customer satisfaction**.
# 
# --------------------------------------------------------------------------
# 
# Our journey exemplifies the transformative power of data science in driving informed decision-making and unlocking opportunities for business success. By harnessing the wealth of information contained within data, we empower organizations to make strategic decisions that drive growth and competitive advantage in today's data-driven world."

# %% [code]
