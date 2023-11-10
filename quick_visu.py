# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 17:55:03 2023

@author: thoma
"""

import pandas as pd 
import numpy as np
import plotly.express as px
import geopandas as gpd
from dash import Dash, dcc, html, Input, Output
from shapely.geometry import Polygon
import random
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns



for folder in  ['mali2012','car2013','burundi2015','ssudan2013']:
    df_true = pd.read_csv('Input/'+folder+'/Observed_events.csv',index_col=(0))
    df_true[df_true>1]=1
    df_pred = pd.read_csv('Input/'+folder+'/conflicts.csv',index_col=(0,1))
    
    plt.figure(figsize=(15,10))
    plt.plot(df_true.sum(axis=1),label='Obs')
    plt.plot(df_pred.sum(axis=1).reset_index(drop=True),label='Pred')
    plt.title(folder)
    plt.xticks([])
    plt.legend()
    plt.show()

data=[]
for folder in  ['mali2012','car2013','burundi2015','ssudan2013']:
    df_true = pd.read_csv('Input/'+folder+'/Observed_events.csv',index_col=(0))
    df_true[df_true>1]=1
    data.append(df_true.stack().value_counts())
    
df = pd.DataFrame(data)

# Calculate the percentage of zeros
df['Zero_Percentage'] = (df.iloc[:,0] / (df.iloc[:,0] + df.iloc[:,1])) * 100

df=df.append(df.mean(),ignore_index=True)

# Set the figure size
plt.figure(figsize=(8, 6))
plt.bar(['Mali2012','CAR2013','Burundi2015','SSudan2013','Mean'], df['Zero_Percentage'], color='lightblue')
plt.xlabel('Country-Year')
plt.ylabel('Percentage of Zeros')
plt.title('Zero-Inflated Data Percentage')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


confusion_matrix_1 = [[2977,  582], [  29,   12]]
confusion_matrix_2 = [[19496, 3116], [200, 148]]
confusion_matrix_3 = [[1955, 686], [25, 106]]
confusion_matrix_4 = [[14103, 584], [391, 22]]

names=['Mali-2012','CAR-2013','Burundi-2015','SSudan-2013','Mean']
# Create a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(10,10))
# Define a colormap
cmap = "Blues"
display_labels = ['Peace', 'Conflict']
# Plot each confusion matrix with the same colormap
for i, cm in enumerate([confusion_matrix_1, confusion_matrix_2, confusion_matrix_3, confusion_matrix_4]):
    row, col = divmod(i, 2)  # Calculate the row and column index
    total_samples = np.sum(cm)  # Calculate the total number of samples
    percent_cm = (cm / total_samples) * 100  # Convert counts to percentages

    sns.heatmap(percent_cm, annot=True, fmt=".2f", cmap=cmap, ax=axes[row, col],cbar=False,xticklabels=display_labels, yticklabels=display_labels)
    axes[row, col].set_title(names[i])

# Adjust the layout
plt.tight_layout()

# Display the plot
plt.show()