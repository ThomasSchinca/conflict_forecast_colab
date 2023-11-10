# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 19:46:15 2022

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
from sklearn.metrics import classification_report,accuracy_score,recall_score
from sklearn import metrics
import matplotlib.pyplot as plt

# =============================================================================
# Selection of the case 
# =============================================================================
acc={'mali2012':[],'car2013':[],'burundi2015':[],'ssudan2013':[]}
re={'mali2012':[],'car2013':[],'burundi2015':[],'ssudan2013':[]}
for cut in [19.5,19.75,20,20.25,20.5]:
    for folder in  ['mali2012','car2013','burundi2015','ssudan2013']:
    
        # =============================================================================
        # Conflict data + Location data 
        # =============================================================================
        # Import of the acled data 
        df = pd.read_csv('Input/2007-01-01-2020-12-31-Burundi-Central_African_Republic-Mali-South_Sudan.csv')
        df['event_date'] =  pd.to_datetime(df['event_date'])
        
        # Extraction of the localities we are interested in (found in the Github)
        # We extract with a buffer of 0.1° around the locality 
        df_loc_tot = pd.read_csv('Input/'+folder+'/locations.csv')
        df_loc = df_loc_tot[(df_loc_tot.location_type=='conflict_zone')]   
        df_loc = df_loc.reset_index(drop=True)
        df_period = pd.read_csv('Input/'+folder+'/conflict_period.csv')
        
        fat_loc = pd.DataFrame(index=pd.date_range(start='01/01/2007',end='12/31/2020',freq='D'))
        conf_loc = pd.DataFrame(index=pd.date_range(start='01/01/2007',end='12/31/2020',freq='D'))
        for i in range(len(df_loc.index)):
            df_latlong = df[(df.latitude>df_loc.latitude[i]-0.05) & (df.latitude<df_loc.latitude[i]+0.05) &(df.longitude<df_loc.longitude[i]+0.05) &(df.longitude<df_loc.longitude[i]+0.05)]
            fat_loc[str(df_loc.iloc[i,0])] = df_latlong.groupby(['event_date'],as_index=True)['fatalities'].sum()  
            conf_loc[str(df_loc.iloc[i,0])] =  df_latlong.groupby(['event_date'],as_index=True)['event_id_cnty'].count()  
        
        fat_loc = fat_loc.fillna(0)
        conf_loc = conf_loc.fillna(0)
        
        fat_loc = fat_loc.loc[:pd.to_datetime(df_period.columns[1])+pd.Timedelta(df_period.iloc[0,1], "d")]
        fat_loc = fat_loc.iloc[-(365*5+df_period.iloc[0,1]):,:]
        conf_loc = conf_loc.loc[:pd.to_datetime(df_period.columns[1])+pd.Timedelta(df_period.iloc[0,1], "d")]
        conf_loc = conf_loc.iloc[-(365*5+df_period.iloc[0,1]):,:]
        
        # Creation of the real data 
        real_data_fat = fat_loc.loc[pd.date_range(start=df_period.columns[1],periods=df_period.iloc[0,1]+1,freq='D'),:]
        real_data_fat=real_data_fat.reset_index(drop=True)
        real_data_fat = real_data_fat.iloc[1:,:]
        real_data_conf = conf_loc.loc[pd.date_range(start=df_period.columns[1],periods=df_period.iloc[0,1]+1,freq='D'),:]
        real_data_conf=real_data_conf.reset_index(drop=True)
        real_data_conf = real_data_conf.iloc[1:,:]
            
        # =============================================================================
        # Prio - Grid data    
        # =============================================================================
        
        # import the prio grid (yearly data +static) 
        static = pd.read_csv('Input/PRIO-GRID Static Variables - 2022-11-11.csv')
        yearly = pd.read_csv('Input/PRIO-GRID Yearly Variables for 2007-2014 - 2023-03-30.csv')
        
        # Extract only the grid realted to our localities
        df_stat=[]
        n_grid=[]
        for i in range(len(df_loc['#name'])):
            df_stat.append(static[(static.xcoord-0.25<df_loc.iloc[i,4]) & (static.xcoord+0.25>df_loc.iloc[i,4]) & (static.ycoord-0.25<df_loc.iloc[i,3]) & (static.ycoord+0.25>df_loc.iloc[i,3])])
            n_grid.append([static[(static.xcoord-0.25<df_loc.iloc[i,4]) & (static.xcoord+0.25>df_loc.iloc[i,4]) & (static.ycoord-0.25<df_loc.iloc[i,3]) & (static.ycoord+0.25>df_loc.iloc[i,3])].iloc[0,0],df_loc.iloc[i,0]])
        df_stat=df_stat*14
        df_stat=np.array(df_stat)
        df_stat = pd.DataFrame(df_stat[:,0,:])
        df_stat.columns = static.columns
        n_grid=pd.DataFrame(n_grid)
        
        l=[]
        for i in range(2007,2021):
            l=l+[i]*len(df_loc)
        df_stat['year'] = l
        
        # Gather the yearly and static in the same dataset
        yearly = yearly[yearly['gid'].isin(list(df_stat.gid.unique()))]
        yearly = yearly.merge(df_stat,how='inner',on=['gid','year'])
        yearly = yearly.dropna(axis=1)
        for g in yearly.gid.unique():
            for ye in range(2015,2021):
                yearly.loc[len(yearly),:]=yearly[(yearly.year==2014) & (yearly.gid == g)].iloc[0,:]
                yearly.loc[len(yearly)-1,'year']=ye
        
        # =============================================================================
        # Input creation 
        # =============================================================================
        
        number_s = 31    # number of t-n included in the model
        n_best = 365*5   # 5 yr of training 
        
        conf_loc[conf_loc>1]=1  # we are forecasting (events/no - events)
        
        ts_seq=[]   # auto regressive extract
        w_l=[]      # spatial extract
        w_prio=[]   # prio grid extract 
        for col in range(len(conf_loc.columns)):
            for i in range(number_s-1,len(conf_loc)):
                ts_seq.append(conf_loc.iloc[i-number_s+1:i+1,col])
                w_l.append([conf_loc.drop(conf_loc.columns[col],axis=1).iloc[i-number_s+1:i].sum().sum(),conf_loc.drop(conf_loc.columns[col],axis=1).iloc[i-7:i].sum().sum()])
                w_prio.append(yearly[(yearly.year==conf_loc.index[i].year) & (yearly.gid==n_grid[n_grid[1]==conf_loc.columns[col]].iloc[0,0])].iloc[0,:])
        
        # auto regressive part
        ts_seq=np.array(ts_seq)
        ts_seq_l= ts_seq.reshape(len(conf_loc.columns),len(conf_loc.index)-number_s+1,number_s)
        ts_seq_learn=ts_seq_l[:,:n_best-number_s+1,:]
        ts_seq_learn=ts_seq_learn.reshape(ts_seq_learn.shape[0]*ts_seq_learn.shape[1],number_s)
        
        # spatial part
        w_l=np.array(w_l)
        w_l_l= w_l.reshape(len(conf_loc.columns),len(conf_loc.index)-number_s+1,2)
        w_l_test=w_l_l[:,:n_best-number_s+1:n_best-number_s+2,:]
        w_l_test=w_l_test.reshape(w_l_test.shape[0]*w_l_test.shape[1],2)
        w_l_l=w_l_l[:,:n_best-number_s+1,:]
        w_l_l=w_l_l.reshape(w_l_l.shape[0]*w_l_l.shape[1],2)
        
        # prio grid part 
        w_prio=np.array(w_prio)
        w_prio_l= w_prio.reshape(len(conf_loc.columns),len(conf_loc.index)-number_s+1,len(yearly.columns))
        w_prio_test=w_prio_l[:,n_best-number_s+1:n_best-number_s+2,:]
        w_prio_test=w_prio_test.reshape(w_prio_test.shape[0]*w_prio_test.shape[1],len(yearly.columns))
        w_prio_l=w_prio_l[:,:n_best-number_s+1,:]
        w_prio_l=w_prio_l.reshape(w_prio_l.shape[0]*w_prio_l.shape[1],len(yearly.columns))
        
        # input/output
        ts_seq_learn_x = ts_seq_learn[:,:-1]
        ts_seq_learn_y = ts_seq_learn[:,-1]
        
        # concatenate of the three parts
        ts_seq_learn_x =  np.concatenate([ts_seq_learn_x,w_l_l,w_prio_l],axis=1)  
        
        # =============================================================================
        # Forecast
        # =============================================================================
        '''
        ### Low events predicted
        
        n_predi = df_period.iloc[0,1]   # number of days predicted 
        ts_seq_test=ts_seq_l[:,n_best-number_s+1:n_best-number_s+2,:] 
        ts_seq_test=ts_seq_test.reshape(ts_seq_test.shape[0],ts_seq_test.shape[2])
        ts_seq_test_x = ts_seq_test[:,:-1]
        ts_seq_test_y = ts_seq_test[:,-1]
        
        # random forest model creation + fit 
        rf = RandomForestClassifier(random_state=42,class_weight='balanced_subsample')
        rf.fit(ts_seq_learn_x,ts_seq_learn_y)
        df_pred_l = conf_loc.iloc[:n_best,:]  #forecast results 
        
        # Creation of the first day we want to forecast
        ts_seq_test_x = np.concatenate([ts_seq_test_x,w_l_test,w_prio_test],axis=1) 
        res = pd.DataFrame(rf.predict(ts_seq_test_x)).T
        res.columns =df_pred_l.columns
        df_pred_l = df_pred_l.append(res)
        
        # Forecast of all the other days 
        for i in range(1,n_predi):
            w_l_test=[]
            w_prio_test=[]
            ts_seq_test_x = np.array(df_pred_l.iloc[-number_s+1:,:].T)
            # update of the input
            for col in range(len(conf_loc.columns)):
                w_l_test.append([df_pred_l.drop(conf_loc.columns[col],axis=1).iloc[-number_s+1:].sum().sum(),df_pred_l.drop(conf_loc.columns[col],axis=1).iloc[-7:].sum().sum()])
                w_prio_test.append(yearly[(yearly.year==conf_loc.index[n_best+i].year) & (yearly.gid==n_grid[n_grid[1]==conf_loc.columns[col]].iloc[0,0])].iloc[0,:])
            w_prio_test=np.array(w_prio_test)
            #w_prio_test = w_prio_test[:,0,:]
            #concatenate of the input
            ts_seq_test_x = np.concatenate([ts_seq_test_x,np.array(w_l_test),w_prio_test],axis=1)
            res = pd.DataFrame(rf.predict(ts_seq_test_x)).T  #prediction
            res.columns =df_pred_l.columns    #store results
            df_pred_l = df_pred_l.append(res)
        
        df_pred_l.index = conf_loc.index[:n_best+n_predi]
        # classification report 
        print(classification_report(np.array(conf_loc.iloc[n_best:n_best+n_predi,:]).reshape((df_period.iloc[0,1]*len(df_loc),)),np.array(df_pred_l.iloc[n_best:n_best+n_predi,:]).reshape((df_period.iloc[0,1]*len(df_loc),))))
        confusion_matrix = metrics.confusion_matrix(np.array(conf_loc.iloc[n_best:n_best+n_predi,:]).reshape((df_period.iloc[0,1]*len(df_loc),)),np.array(df_pred_l.iloc[n_best:n_best+n_predi,:]).reshape((df_period.iloc[0,1]*len(df_loc),)))
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])
        cm_display.plot()
        plt.title(folder+' low')
        plt.show()
        
        df_pred_l_ext = df_pred_l.iloc[n_best:n_best+n_predi,:]
        df_pred_l_ext = df_pred_l_ext.reset_index(drop=True)
        '''
        
        ### High events predicted
        
        k=int(pd.DataFrame(ts_seq_learn_y==0).value_counts()[1]/cut)   # number of observation with no events in the train
        
        n_predi = df_period.iloc[0,1]  
        ts_seq_test=ts_seq_l[:,n_best-number_s+1:n_best-number_s+2,:]
        ts_seq_test=ts_seq_test.reshape(ts_seq_test.shape[0],ts_seq_test.shape[2])
        ts_seq_test_x = ts_seq_test[:,:-1]
        ts_seq_test_y = ts_seq_test[:,-1]
        
        random.seed(0)
        ts_seq_learn_x_sub=ts_seq_learn_x[ts_seq_learn_y==1]
        ts_seq_learn_y_sub = ts_seq_learn_y[ts_seq_learn_y==1]
        pick=random.sample(range((len(ts_seq_learn_x)-len(ts_seq_learn_x_sub))),k)
        ts_seq_learn_x_sub = np.concatenate([ts_seq_learn_x_sub,ts_seq_learn_x[ts_seq_learn_y==0][pick,:]])
        ts_seq_learn_y_sub = np.concatenate([ts_seq_learn_y_sub,ts_seq_learn_y[ts_seq_learn_y==0][pick]])
        
        rf = RandomForestClassifier(random_state=42)
        rf.fit(ts_seq_learn_x_sub,ts_seq_learn_y_sub)
        df_pred_h = conf_loc.iloc[:n_best,:]
        
        ts_seq_test_x = np.concatenate([ts_seq_test_x,w_l_test,w_prio_test],axis=1) 
        res = pd.DataFrame(rf.predict(ts_seq_test_x)).T
        res.columns =df_pred_h.columns
        df_pred_h = df_pred_h.append(res)
        
        for i in range(1,n_predi):
            w_l_test=[]
            w_prio_test=[]
            ts_seq_test_x = np.array(df_pred_h.iloc[-number_s+1:,:].T)
            for col in range(len(conf_loc.columns)):
                w_l_test.append([df_pred_h.drop(conf_loc.columns[col],axis=1).iloc[-number_s+1:].sum().sum(),df_pred_h.drop(conf_loc.columns[col],axis=1).iloc[-7:].sum().sum()])
                w_prio_test.append(yearly[(yearly.year==conf_loc.index[n_best+i].year) & (yearly.gid==n_grid[n_grid[1]==conf_loc.columns[col]].iloc[0,0])].iloc[0,:])
            w_prio_test=np.array(w_prio_test)
            #w_prio_test = w_prio_test[:,0,:]
            ts_seq_test_x = np.concatenate([ts_seq_test_x,np.array(w_l_test),w_prio_test],axis=1)
            res = pd.DataFrame(rf.predict(ts_seq_test_x)).T
            res.columns =df_pred_h.columns
            df_pred_h = df_pred_h.append(res)
        
        df_pred_h.index = conf_loc.index[:n_best+n_predi]
        # classification report 
        print(classification_report(np.array(conf_loc.iloc[n_best:n_best+n_predi,:]).reshape((df_period.iloc[0,1]*len(df_loc),)),np.array(df_pred_h.iloc[n_best:n_best+n_predi,:]).reshape((df_period.iloc[0,1]*len(df_loc),))))
        confusion_matrix = metrics.confusion_matrix(np.array(conf_loc.iloc[n_best:n_best+n_predi,:]).reshape((df_period.iloc[0,1]*len(df_loc),)),np.array(df_pred_h.iloc[n_best:n_best+n_predi,:]).reshape((df_period.iloc[0,1]*len(df_loc),)))
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Peace', 'Conflict'])
        cm_display.plot(colorbar=False)
        plt.title(folder+' high ')
        plt.show()
        
        df_pred_h_ext = df_pred_h.iloc[n_best:n_best+n_predi,:]
        df_pred_h_ext = df_pred_h_ext.reset_index(drop=True)
        
        df_other = pd.DataFrame(index = range(len(df_pred_h_ext)),columns = df_loc_tot[(df_loc_tot.location_type!='conflict_zone') & (df_loc_tot.location_type!='town')]['#name'])
        df_other.iloc[:,:] = int(0)
        
        df_pred_h_ext = pd.concat([pd.Series(range(len(df_pred_h_ext)),name='#Day',index=range(len(df_pred_h_ext))),df_pred_h_ext,df_other],axis=1)
        
        acc[folder].append(accuracy_score(np.array(conf_loc.iloc[n_best:n_best+n_predi,:]).reshape((df_period.iloc[0,1]*len(df_loc),)),np.array(df_pred_h.iloc[n_best:n_best+n_predi,:]).reshape((df_period.iloc[0,1]*len(df_loc),))))
        re[folder].append(recall_score(np.array(conf_loc.iloc[n_best:n_best+n_predi,:]).reshape((df_period.iloc[0,1]*len(df_loc),)),np.array(df_pred_h.iloc[n_best:n_best+n_predi,:]).reshape((df_period.iloc[0,1]*len(df_loc),))))
        
        data = real_data_conf.to_numpy().flatten()
        zero_values = data[data == 0]
        non_zero_values = data[data != 0]

        plt.figure(figsize=(7,7))
        plt.hist([zero_values, non_zero_values], bins=[0, 1],color=['#99ccff','#336699'], edgecolor='black', label=['Zeros', 'Non-Zeros'])
        plt.ylabel('Frequency')
        plt.xticks([0.3,0.7],['Peace', 'Conflict'])
        plt.title(folder)
        plt.show()
        # =============================================================================
        # Dataset export
        # =============================================================================
        
        # Observed values
        # real_data_fat.to_csv('Input/'+folder+'/Observed_fatalities.csv')
        # real_data_conf.to_csv('Input/'+folder+'/Observed_events.csv')
        
        # # Forecasts
        # #df_pred_l_ext.to_csv('Input/'+folder+'/Low_forecast.csv')
        # df_pred_h_ext.to_csv('Input/'+folder+'/conflicts.csv')
        
        # df_true = pd.read_csv('Input/'+folder+'/Observed_events.csv',index_col=(0))
        # df_true[df_true>1]=1
        # df_pred = pd.read_csv('Input/'+folder+'/conflicts.csv',index_col=(0,1))
        
        # plt.figure(figsize=(15,10))
        # plt.plot(df_true.sum(axis=1),label='Obs')
        # plt.plot(df_pred.sum(axis=1).reset_index(drop=True),label='Pred')
        # plt.title(folder)
        # plt.xticks([])
        # plt.legend()
        # plt.show()

colors = ['b', 'g', 'r', 'c']
fig, ax = plt.subplots()
h=[]
for key in acc.keys():
    acc_values = acc[key]
    re_values = re[key]
    multiplication = [a * b for a, b in zip(acc_values, re_values)]
    x_values = range(len(acc_values))
    ax.plot(x_values, multiplication, label=key, color=colors.pop(0),marker='o')
    h.append(multiplication)

ax.plot(x_values, pd.DataFrame(h).mean(), label='Mean', color='k', linestyle='--')
ax.set_xlabel('k value for training')
ax.set_ylabel('Accuracy * Recall ')
ax.set_xticks([0,1,2,3,4],[19.5,19.75,20,20.25,20.5])
ax.legend()
plt.show()


for folder in  ['mali2012','car2013','burundi2015','ssudan2013']:

    # =============================================================================
    # Conflict data + Location data 
    # =============================================================================
    # Import of the acled data 
    df = pd.read_csv('Input/2007-01-01-2020-12-31-Burundi-Central_African_Republic-Mali-South_Sudan.csv')
    df['event_date'] =  pd.to_datetime(df['event_date'])
    
    # Extraction of the localities we are interested in (found in the Github)
    # We extract with a buffer of 0.1° around the locality 
    df_loc_tot = pd.read_csv('Input/'+folder+'/locations.csv')
    df_loc = df_loc_tot[(df_loc_tot.location_type=='conflict_zone')]   
    df_loc = df_loc.reset_index(drop=True)
    df_period = pd.read_csv('Input/'+folder+'/conflict_period.csv')
    
    fat_loc = pd.DataFrame(index=pd.date_range(start='01/01/2007',end='12/31/2020',freq='D'))
    conf_loc = pd.DataFrame(index=pd.date_range(start='01/01/2007',end='12/31/2020',freq='D'))
    for i in range(len(df_loc.index)):
        df_latlong = df[(df.latitude>df_loc.latitude[i]-0.05) & (df.latitude<df_loc.latitude[i]+0.05) &(df.longitude<df_loc.longitude[i]+0.05) &(df.longitude<df_loc.longitude[i]+0.05)]
        fat_loc[str(df_loc.iloc[i,0])] = df_latlong.groupby(['event_date'],as_index=True)['fatalities'].sum()  
        conf_loc[str(df_loc.iloc[i,0])] =  df_latlong.groupby(['event_date'],as_index=True)['event_id_cnty'].count()  
    
    fat_loc = fat_loc.fillna(0)
    conf_loc = conf_loc.fillna(0)
    
    fat_loc = fat_loc.loc[:pd.to_datetime(df_period.columns[1])+pd.Timedelta(df_period.iloc[0,1], "d")]
    fat_loc = fat_loc.iloc[-(365*5+df_period.iloc[0,1]):,:]
    conf_loc = conf_loc.loc[:pd.to_datetime(df_period.columns[1])+pd.Timedelta(df_period.iloc[0,1], "d")]
    conf_loc = conf_loc.iloc[-(365*5+df_period.iloc[0,1]):,:]
    
    # Creation of the real data 
    real_data_fat = fat_loc.loc[pd.date_range(start=df_period.columns[1],periods=df_period.iloc[0,1]+1,freq='D'),:]
    real_data_fat=real_data_fat.reset_index(drop=True)
    real_data_fat = real_data_fat.iloc[1:,:]
    real_data_conf = conf_loc.loc[pd.date_range(start=df_period.columns[1],periods=df_period.iloc[0,1]+1,freq='D'),:]
    real_data_conf=real_data_conf.reset_index(drop=True)
    real_data_conf = real_data_conf.iloc[1:,:]
        
    # =============================================================================
    # Prio - Grid data    
    # =============================================================================
    
    # import the prio grid (yearly data +static) 
    static = pd.read_csv('Input/PRIO-GRID Static Variables - 2022-11-11.csv')
    yearly = pd.read_csv('Input/PRIO-GRID Yearly Variables for 2007-2014 - 2023-03-30.csv')
    
    # Extract only the grid realted to our localities
    df_stat=[]
    n_grid=[]
    for i in range(len(df_loc['#name'])):
        df_stat.append(static[(static.xcoord-0.25<df_loc.iloc[i,4]) & (static.xcoord+0.25>df_loc.iloc[i,4]) & (static.ycoord-0.25<df_loc.iloc[i,3]) & (static.ycoord+0.25>df_loc.iloc[i,3])])
        n_grid.append([static[(static.xcoord-0.25<df_loc.iloc[i,4]) & (static.xcoord+0.25>df_loc.iloc[i,4]) & (static.ycoord-0.25<df_loc.iloc[i,3]) & (static.ycoord+0.25>df_loc.iloc[i,3])].iloc[0,0],df_loc.iloc[i,0]])
    df_stat=df_stat*14
    df_stat=np.array(df_stat)
    df_stat = pd.DataFrame(df_stat[:,0,:])
    df_stat.columns = static.columns
    n_grid=pd.DataFrame(n_grid)
    
    l=[]
    for i in range(2007,2021):
        l=l+[i]*len(df_loc)
    df_stat['year'] = l
    
    # Gather the yearly and static in the same dataset
    yearly = yearly[yearly['gid'].isin(list(df_stat.gid.unique()))]
    yearly = yearly.merge(df_stat,how='inner',on=['gid','year'])
    yearly = yearly.dropna(axis=1)
    for g in yearly.gid.unique():
        for ye in range(2015,2021):
            yearly.loc[len(yearly),:]=yearly[(yearly.year==2014) & (yearly.gid == g)].iloc[0,:]
            yearly.loc[len(yearly)-1,'year']=ye
    
    # =============================================================================
    # Input creation 
    # =============================================================================
    
    number_s = 31    # number of t-n included in the model
    n_best = 365*5   # 5 yr of training 
    
    conf_loc[conf_loc>1]=1  # we are forecasting (events/no - events)
    
    ts_seq=[]   # auto regressive extract
    w_l=[]      # spatial extract
    w_prio=[]   # prio grid extract 
    for col in range(len(conf_loc.columns)):
        for i in range(number_s-1,len(conf_loc)):
            ts_seq.append(conf_loc.iloc[i-number_s+1:i+1,col])
            w_l.append([conf_loc.drop(conf_loc.columns[col],axis=1).iloc[i-number_s+1:i].sum().sum(),conf_loc.drop(conf_loc.columns[col],axis=1).iloc[i-7:i].sum().sum()])
            w_prio.append(yearly[(yearly.year==conf_loc.index[i].year) & (yearly.gid==n_grid[n_grid[1]==conf_loc.columns[col]].iloc[0,0])].iloc[0,:])
    
    # auto regressive part
    ts_seq=np.array(ts_seq)
    ts_seq_l= ts_seq.reshape(len(conf_loc.columns),len(conf_loc.index)-number_s+1,number_s)
    ts_seq_learn=ts_seq_l[:,:n_best-number_s+1,:]
    ts_seq_learn=ts_seq_learn.reshape(ts_seq_learn.shape[0]*ts_seq_learn.shape[1],number_s)
    
    # spatial part
    w_l=np.array(w_l)
    w_l_l= w_l.reshape(len(conf_loc.columns),len(conf_loc.index)-number_s+1,2)
    w_l_test=w_l_l[:,:n_best-number_s+1:n_best-number_s+2,:]
    w_l_test=w_l_test.reshape(w_l_test.shape[0]*w_l_test.shape[1],2)
    w_l_l=w_l_l[:,:n_best-number_s+1,:]
    w_l_l=w_l_l.reshape(w_l_l.shape[0]*w_l_l.shape[1],2)
    
    # prio grid part 
    w_prio=np.array(w_prio)
    w_prio_l= w_prio.reshape(len(conf_loc.columns),len(conf_loc.index)-number_s+1,len(yearly.columns))
    w_prio_test=w_prio_l[:,n_best-number_s+1:n_best-number_s+2,:]
    w_prio_test=w_prio_test.reshape(w_prio_test.shape[0]*w_prio_test.shape[1],len(yearly.columns))
    w_prio_l=w_prio_l[:,:n_best-number_s+1,:]
    w_prio_l=w_prio_l.reshape(w_prio_l.shape[0]*w_prio_l.shape[1],len(yearly.columns))
    
    # input/output
    ts_seq_learn_x = ts_seq_learn[:,:-1]
    ts_seq_learn_y = ts_seq_learn[:,-1]
    
    # concatenate of the three parts
    ts_seq_learn_x =  np.concatenate([ts_seq_learn_x,w_l_l,w_prio_l],axis=1)  
    
    # =============================================================================
    # Forecast
    # =============================================================================
    
    k=int(pd.DataFrame(ts_seq_learn_y==0).value_counts()[1]/5.4)   # number of observation with no events in the train
    
    n_predi = df_period.iloc[0,1]  
    ts_seq_test=ts_seq_l[:,n_best-number_s+1:n_best-number_s+2,:]
    ts_seq_test=ts_seq_test.reshape(ts_seq_test.shape[0],ts_seq_test.shape[2])
    ts_seq_test_x = ts_seq_test[:,:-1]
    ts_seq_test_y = ts_seq_test[:,-1]
    
    random.seed(0)
    ts_seq_learn_x_sub=ts_seq_learn_x[ts_seq_learn_y==1]
    ts_seq_learn_y_sub = ts_seq_learn_y[ts_seq_learn_y==1]
    pick=random.sample(range((len(ts_seq_learn_x)-len(ts_seq_learn_x_sub))),k)
    ts_seq_learn_x_sub = np.concatenate([ts_seq_learn_x_sub,ts_seq_learn_x[ts_seq_learn_y==0][pick,:]])
    ts_seq_learn_y_sub = np.concatenate([ts_seq_learn_y_sub,ts_seq_learn_y[ts_seq_learn_y==0][pick]])
    
    rf = RandomForestClassifier(random_state=42)
    rf.fit(ts_seq_learn_x_sub,ts_seq_learn_y_sub)
    df_pred_h = conf_loc.iloc[:n_best,:]
    
    ts_seq_test_x = np.concatenate([ts_seq_test_x,w_l_test,w_prio_test],axis=1) 
    res = pd.DataFrame(rf.predict(ts_seq_test_x)).T
    res.columns =df_pred_h.columns
    df_pred_h = df_pred_h.append(res)
    
    for i in range(1,n_predi):
        w_l_test=[]
        w_prio_test=[]
        ts_seq_test_x = np.array(df_pred_h.iloc[-number_s+1:,:].T)
        for col in range(len(conf_loc.columns)):
            w_l_test.append([df_pred_h.drop(conf_loc.columns[col],axis=1).iloc[-number_s+1:].sum().sum(),df_pred_h.drop(conf_loc.columns[col],axis=1).iloc[-7:].sum().sum()])
            w_prio_test.append(yearly[(yearly.year==conf_loc.index[n_best+i].year) & (yearly.gid==n_grid[n_grid[1]==conf_loc.columns[col]].iloc[0,0])].iloc[0,:])
        w_prio_test=np.array(w_prio_test)
        #w_prio_test = w_prio_test[:,0,:]
        ts_seq_test_x = np.concatenate([ts_seq_test_x,np.array(w_l_test),w_prio_test],axis=1)
        res = pd.DataFrame(rf.predict(ts_seq_test_x)).T
        res.columns =df_pred_h.columns
        df_pred_h = df_pred_h.append(res)
    
    df_pred_h.index = conf_loc.index[:n_best+n_predi]
    # classification report 
    print(classification_report(np.array(conf_loc.iloc[n_best:n_best+n_predi,:]).reshape((df_period.iloc[0,1]*len(df_loc),)),np.array(df_pred_h.iloc[n_best:n_best+n_predi,:]).reshape((df_period.iloc[0,1]*len(df_loc),))))
    confusion_matrix = metrics.confusion_matrix(np.array(conf_loc.iloc[n_best:n_best+n_predi,:]).reshape((df_period.iloc[0,1]*len(df_loc),)),np.array(df_pred_h.iloc[n_best:n_best+n_predi,:]).reshape((df_period.iloc[0,1]*len(df_loc),)))
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Peace', 'Conflict'])
    cm_display.plot(colorbar=False)
    plt.title(folder+' low ')
    plt.show()
    
    df_pred_l_ext = df_pred_h.iloc[n_best:n_best+n_predi,:]
    df_pred_l_ext = df_pred_l_ext.reset_index(drop=True)
    
    df_other = pd.DataFrame(index = range(len(df_pred_l_ext)),columns = df_loc_tot[(df_loc_tot.location_type!='conflict_zone') & (df_loc_tot.location_type!='town')]['#name'])
    df_other.iloc[:,:] = int(0)
    
    df_pred_l_ext = pd.concat([pd.Series(range(len(df_pred_l_ext)),name='#Day',index=range(len(df_pred_l_ext))),df_pred_l_ext,df_other],axis=1)

    ### High events predicted
    
    k=int(pd.DataFrame(ts_seq_learn_y==0).value_counts()[1]/20)   # number of observation with no events in the train
    
    n_predi = df_period.iloc[0,1]  
    ts_seq_test=ts_seq_l[:,n_best-number_s+1:n_best-number_s+2,:]
    ts_seq_test=ts_seq_test.reshape(ts_seq_test.shape[0],ts_seq_test.shape[2])
    ts_seq_test_x = ts_seq_test[:,:-1]
    ts_seq_test_y = ts_seq_test[:,-1]
    
    random.seed(0)
    ts_seq_learn_x_sub=ts_seq_learn_x[ts_seq_learn_y==1]
    ts_seq_learn_y_sub = ts_seq_learn_y[ts_seq_learn_y==1]
    pick=random.sample(range((len(ts_seq_learn_x)-len(ts_seq_learn_x_sub))),k)
    ts_seq_learn_x_sub = np.concatenate([ts_seq_learn_x_sub,ts_seq_learn_x[ts_seq_learn_y==0][pick,:]])
    ts_seq_learn_y_sub = np.concatenate([ts_seq_learn_y_sub,ts_seq_learn_y[ts_seq_learn_y==0][pick]])
    
    rf = RandomForestClassifier(random_state=42)
    rf.fit(ts_seq_learn_x_sub,ts_seq_learn_y_sub)
    df_pred_h = conf_loc.iloc[:n_best,:]
    
    ts_seq_test_x = np.concatenate([ts_seq_test_x,w_l_test,w_prio_test],axis=1) 
    res = pd.DataFrame(rf.predict(ts_seq_test_x)).T
    res.columns =df_pred_h.columns
    df_pred_h = df_pred_h.append(res)
    
    for i in range(1,n_predi):
        w_l_test=[]
        w_prio_test=[]
        ts_seq_test_x = np.array(df_pred_h.iloc[-number_s+1:,:].T)
        for col in range(len(conf_loc.columns)):
            w_l_test.append([df_pred_h.drop(conf_loc.columns[col],axis=1).iloc[-number_s+1:].sum().sum(),df_pred_h.drop(conf_loc.columns[col],axis=1).iloc[-7:].sum().sum()])
            w_prio_test.append(yearly[(yearly.year==conf_loc.index[n_best+i].year) & (yearly.gid==n_grid[n_grid[1]==conf_loc.columns[col]].iloc[0,0])].iloc[0,:])
        w_prio_test=np.array(w_prio_test)
        #w_prio_test = w_prio_test[:,0,:]
        ts_seq_test_x = np.concatenate([ts_seq_test_x,np.array(w_l_test),w_prio_test],axis=1)
        res = pd.DataFrame(rf.predict(ts_seq_test_x)).T
        res.columns =df_pred_h.columns
        df_pred_h = df_pred_h.append(res)
    
    df_pred_h.index = conf_loc.index[:n_best+n_predi]
    # classification report 
    print(classification_report(np.array(conf_loc.iloc[n_best:n_best+n_predi,:]).reshape((df_period.iloc[0,1]*len(df_loc),)),np.array(df_pred_h.iloc[n_best:n_best+n_predi,:]).reshape((df_period.iloc[0,1]*len(df_loc),))))
    confusion_matrix = metrics.confusion_matrix(np.array(conf_loc.iloc[n_best:n_best+n_predi,:]).reshape((df_period.iloc[0,1]*len(df_loc),)),np.array(df_pred_h.iloc[n_best:n_best+n_predi,:]).reshape((df_period.iloc[0,1]*len(df_loc),)))
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Peace', 'Conflict'])
    cm_display.plot(colorbar=False)
    plt.title(folder+' high ')
    plt.show()
    
    df_pred_h_ext = df_pred_h.iloc[n_best:n_best+n_predi,:]
    df_pred_h_ext = df_pred_h_ext.reset_index(drop=True)
    
    df_other = pd.DataFrame(index = range(len(df_pred_h_ext)),columns = df_loc_tot[(df_loc_tot.location_type!='conflict_zone') & (df_loc_tot.location_type!='town')]['#name'])
    df_other.iloc[:,:] = int(0)
    
    df_pred_h_ext = pd.concat([pd.Series(range(len(df_pred_h_ext)),name='#Day',index=range(len(df_pred_h_ext))),df_pred_h_ext,df_other],axis=1)

    # =============================================================================
    # Dataset export
    # =============================================================================
    
    #Observed values
    real_data_fat.to_csv('Input/'+folder+'/Observed_fatalities.csv')
    real_data_conf.to_csv('Input/'+folder+'/Observed_events.csv')
    
    # Forecasts
    df_pred_l_ext.to_csv('Input/'+folder+'/Low_forecast.csv')
    df_pred_h_ext.to_csv('Input/'+folder+'/conflicts.csv')
    
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
    
    
