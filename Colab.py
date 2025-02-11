# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 19:46:15 2022

@author: thoma
"""

import pandas as pd 
import numpy as np
import random
from datetime import datetime,timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,recall_score,roc_curve, roc_auc_score
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import bernoulli
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import ttest_1samp

# =============================================================================
# Functions   
# =============================================================================
    
def coin_flip_experiment(num_flips, p):
    results = []
    for _ in range(num_flips):
        flips_needed = 0
        while random.random() > p:
            flips_needed += 1
        results.append(flips_needed)
    return results


# =============================================================================
#
# MODEL 1 : Conflict/No conflict classification
#
# =============================================================================


l_fpr=[]
l_tpr=[]
l_thresholds=[]
l_roc_auc=[]
data_obs=[]
data_pred=[]
df_plot_loc=[]
names_beaut=['Mali 2012', 'CAR 2013','Burundi 2015','S. Sudan 2013']
c=0

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
    df_pred_h_proba=df_pred_h.copy()
    
    ts_seq_test_x = np.concatenate([ts_seq_test_x,w_l_test,w_prio_test],axis=1) 
    res = pd.DataFrame(rf.predict(ts_seq_test_x)).T
    res.columns =df_pred_h.columns
    df_pred_h = df_pred_h.append(res)
    
    res_prob = pd.DataFrame(rf.predict_proba(ts_seq_test_x)[:,1]).T
    res_prob.columns =df_pred_h.columns
    df_pred_h_proba = df_pred_h_proba.append(res_prob)
    
    
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
        
        res_prob = pd.DataFrame(rf.predict_proba(ts_seq_test_x)[:,1]).T
        res_prob.columns =df_pred_h.columns
        df_pred_h_proba = df_pred_h_proba.append(res_prob)
    
    df_pred_h.index = conf_loc.index[:n_best+n_predi]
    df_pred_h_proba.index = conf_loc.index[:n_best+n_predi]
    
    data_obs.append(np.array(conf_loc.iloc[n_best:n_best+n_predi,:]).reshape((df_period.iloc[0,1]*len(df_loc),)))
    data_pred.append(np.array(df_pred_h_proba.iloc[n_best:n_best+n_predi,:]).reshape((df_period.iloc[0,1]*len(df_loc),)))
    fpr, tpr, thresholds = roc_curve(np.array(conf_loc.iloc[n_best:n_best+n_predi,:]).reshape((df_period.iloc[0,1]*len(df_loc),)),np.array(df_pred_h_proba.iloc[n_best:n_best+n_predi,:]).reshape((df_period.iloc[0,1]*len(df_loc),)), pos_label=1)
    roc_auc = roc_auc_score(np.array(conf_loc.iloc[n_best:n_best+n_predi,:]).reshape((df_period.iloc[0,1]*len(df_loc),)),np.array(df_pred_h_proba.iloc[n_best:n_best+n_predi,:]).reshape((df_period.iloc[0,1]*len(df_loc),))) 
    l_fpr.append(fpr)
    l_tpr.append(tpr)
    l_thresholds.append(thresholds)
    l_roc_auc.append(roc_auc)
    df_pred_h_ext = df_pred_h.iloc[n_best:n_best+n_predi,:]
    df_pred_h_ext = df_pred_h_ext.reset_index(drop=True)

    df_other = pd.DataFrame(index = range(len(df_pred_h_ext)),columns = df_loc_tot[(df_loc_tot.location_type!='conflict_zone') & (df_loc_tot.location_type!='town')]['#name'])
    df_other.iloc[:,:] = int(0)
    
    df_pred_h_ext = pd.concat([pd.Series(range(len(df_pred_h_ext)),name='#Day',index=range(len(df_pred_h_ext))),df_pred_h_ext,df_other],axis=1)

    # =============================================================================
    # Dataset export
    # =============================================================================
    
    df_true = pd.read_csv('Input/'+folder+'/Observed_events.csv',index_col=(0))
    df_true[df_true>1]=1
    df_pred = pd.read_csv('Input/'+folder+'/conflicts.csv',index_col=(0,1))
    
    l_pred=[]
    l_obs=[]
    for i in range(len(df_true.columns)):
        l_pred.append(df_pred.iloc[:,i][df_pred.iloc[:,i]==1].count())
        l_obs.append(df_true.iloc[:,i][df_true.iloc[:,i]==1].count())
    df_loc['Pred']=l_pred
    df_loc['Obs']=l_obs
    df_plot_loc.append(df_loc)
    
    c=c+1
    

accu_r=[]
accu_p=[]
accu_b=[]
accu_r_tot=[]
accu_p_tot=[]
accu_b_tot=[]
conf_mat=[]
for folder in  ['mali2012','car2013','burundi2015','ssudan2013']:
    df_loc_tot = pd.read_csv('Input/'+folder+'/locations.csv')
    df_period = pd.read_csv('Input/'+folder+'/conflict_period.csv')
    df_true = pd.read_csv('Input/'+folder+'/Observed_events.csv',index_col=(0))
    df_true[df_true>1]=1
    df_pred = pd.read_csv('Input/'+folder+'/conflicts.csv',index_col=(0,1))
    df_pred=df_pred.iloc[:,:len(df_true.columns)]
    df_pred=df_pred.reset_index(drop=True)
    b_dist = bernoulli(df_true.mean(axis=1).mean())
    random_sample = b_dist.rvs(size=df_pred.shape)
    df_pred_ber = pd.DataFrame(random_sample, columns=df_pred.columns)

    df_loc_tot.to_csv('Input_B/'+folder+'/locations.csv')
    df_loc_tot.to_csv('Input_R/'+folder+'/locations.csv')
    
    df_period.to_csv('Input_B/'+folder+'/conflict_period.csv')
    df_period.to_csv('Input_R/'+folder+'/conflict_period.csv')

    
    b_dist = bernoulli(0.5)
    random_sample = b_dist.rvs(size=df_pred.shape)
    df_pred_rand= pd.DataFrame(random_sample, columns=df_pred.columns)
    
    true_labels = pd.Series(df_true.values.ravel()).dropna()
    predicted_probabilities = pd.Series(df_pred.values.ravel()).dropna()
    predicted_b = pd.Series(df_pred_ber.values.ravel()).dropna()
    predicted_r = pd.Series(df_pred_rand.values.ravel()).dropna()
    
    #ber_roc.append(roc_auc_score(true_labels, [df_true.mean(axis=1).mean()]*len(true_labels)))
    accu_p_tot.append(accuracy_score(true_labels,predicted_probabilities))
    accu_r_tot.append(accuracy_score(true_labels,predicted_r))
    accu_b_tot.append(accuracy_score(true_labels,predicted_b))
    
    accu_p.append(recall_score(true_labels,predicted_probabilities))
    accu_r.append(recall_score(true_labels,predicted_r))
    accu_b.append(recall_score(true_labels,predicted_b))
    
    df_pred_ind = pd.read_csv('Input/'+folder+'/conflicts.csv',index_col=(0,1))
    df_pred_ber.index = df_pred_ind.index
    df_pred_ber = pd.concat([df_pred_ber,df_pred_ind.iloc[:,len(df_pred_ber.columns):]])
    df_pred_ber.to_csv('Input_B/'+folder+'/conflict.csv')
    
    df_pred_rand.index = df_pred_ind.index
    df_pred_rand = pd.concat([df_pred_rand,df_pred_ind.iloc[:,len(df_pred_rand.columns):]])
    df_pred_rand.to_csv('Input_R/'+folder+'/conflict.csv')
    
    conf_mat.append(metrics.confusion_matrix(true_labels,predicted_probabilities))
    

plt.figure(figsize=(14, 9))
constants = [0.6, 0.3]
for constant in constants:
    x_values = np.linspace(0, 1, 100)
    y_values = constant / x_values
    plt.plot(x_values, y_values, 'k--', lw=1.0)
plt.scatter(accu_p, accu_p_tot, label='RF', marker='o', c=l_roc_auc, cmap='Reds', vmin=0.5, vmax=0.8,
            alpha=0.5, edgecolors='black', s=150)  # Doubled size
plt.scatter(accu_r, accu_r_tot, label='Random', marker='s', c=[0.5] * 4, cmap='Reds', vmin=0.5, vmax=0.8,
            alpha=0.5, edgecolors='black', s=150)  # Doubled size
plt.scatter(accu_b, accu_b_tot, label='Bernoulli', marker='d', c=[0.5] * 4, cmap='Reds', vmin=0.5, vmax=0.8,
            alpha=0.5, edgecolors='black', s=150)  # Doubled size
plt.scatter(pd.Series(accu_p).mean(), pd.Series(accu_p_tot).mean(), marker='o', c=pd.Series(l_roc_auc).mean(),
            cmap='Reds', vmin=0.5, vmax=0.8, edgecolors='black', s=300)  # Doubled size
plt.scatter(pd.Series(accu_r).mean(), pd.Series(accu_r_tot).mean(), marker='s', c=[0.5], cmap='Reds', vmin=0.5,
            vmax=0.8, edgecolors='black', s=300)  # Doubled size
plt.scatter(pd.Series(accu_b).mean(), pd.Series(accu_b_tot).mean(), marker='d', c=[0.5], cmap='Reds', vmin=0.5,
            vmax=0.8, edgecolors='black', s=300)  # Doubled size
plt.colorbar(label='ROC-AUC score')
plt.legend()
plt.xlabel('Recall', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.xlim(-0.05, 0.85)
plt.ylim(0.45, 1.02)
plt.show()

names=['Mali-2012','CAR-2013','Burundi-2015','SSudan-2013']
fig, axes = plt.subplots(2, 2, figsize=(10,10))
plt.xlabel('Predicted')
plt.ylabel('Actual')
cmap = "Blues"
display_labels = ['Peace', 'Conflict']
for i, cm in enumerate(conf_mat):
    row, col = divmod(i, 2)  # Calculate the row and column index
    total_samples = np.sum(cm)  # Calculate the total number of samples
    percent_cm = (cm / total_samples) * 100  # Convert counts to percentages
    sns.set(font_scale=1.5)
    sns.heatmap(percent_cm, annot=True, fmt=".2f", cmap=cmap, ax=axes[row, col],cbar=False,xticklabels=display_labels, yticklabels=display_labels)
    axes[row, col].set_title(names[i])
plt.tight_layout()
plt.show()




# =============================================================================
#
# MODEL 2 : First onset forecast
#
# =============================================================================


c=0
max_l= 120
tot_res=[]
tot_real=[]
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
    
    result_df = pd.DataFrame(index=conf_loc.index,columns=conf_loc.columns)
    for column in conf_loc.columns:
        next_1_index = conf_loc[conf_loc[column] == 1].index
        for date in range(len(next_1_index)):
            dateu = next_1_index[date]
            if date!=0:
                next_1_timedelta = abs((conf_loc.loc[next_1_index[date-1]:dateu,column].index - dateu).days)
                result_df.loc[next_1_index[date-1]:dateu,column] = next_1_timedelta
            elif dateu> datetime.strptime(df_period.columns[1], "%Y-%m-%d"):
                result_df.loc[next_1_index[date-1]:dateu,column] = float('NaN')
            else:
                next_1_timedelta = abs((conf_loc.loc[:dateu,column].index - dateu).days)
                result_df.loc[:dateu,column] = next_1_timedelta
    
    ts_seq=[]   # auto regressive extract
    ts_seq_out=[]
    w_l=[]      # spatial extract
    w_prio=[]   # prio grid extract 
    for col in range(len(conf_loc.columns)):
        for i in range(number_s-1,len(conf_loc)):
            ts_seq.append(conf_loc.iloc[i-number_s+1:i+1,col])
            ts_seq_out.append(result_df.iloc[i-number_s+1:i+1,col])
            w_l.append([conf_loc.drop(conf_loc.columns[col],axis=1).iloc[i-number_s+1:i].sum().sum(),conf_loc.drop(conf_loc.columns[col],axis=1).iloc[i-7:i].sum().sum()])
            w_prio.append(yearly[(yearly.year==conf_loc.index[i].year) & (yearly.gid==n_grid[n_grid[1]==conf_loc.columns[col]].iloc[0,0])].iloc[0,:])
    
    # auto regressive part
    ts_seq_out=np.array(ts_seq_out)
    ts_seq_out_l= ts_seq_out.reshape(len(conf_loc.columns),len(conf_loc.index)-number_s+1,number_s)
    ts_seq_out_learn=ts_seq_out_l[:,:n_best-number_s+1,:]
    ts_seq_out_learn=ts_seq_out_learn.reshape(ts_seq_out_learn.shape[0]*ts_seq_out_learn.shape[1],number_s)
    
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
    ts_seq_learn_y = ts_seq_out_learn[:,-1]
    
    # concatenate of the three parts
    ts_seq_learn_x =  np.concatenate([ts_seq_learn_x,w_l_l,w_prio_l],axis=1)  
    
    
    # =============================================================================
    # Forecast
    # =============================================================================

    ts_seq_learn_x_sub=ts_seq_learn_x[~pd.Series(ts_seq_learn_y).isna()]
    ts_seq_learn_y_sub = ts_seq_learn_y[~pd.Series(ts_seq_learn_y).isna()]
    
    ts_seq_learn_x_sub=ts_seq_learn_x_sub[pd.Series(ts_seq_learn_y_sub)<max_l]
    ts_seq_learn_y_sub = ts_seq_learn_y_sub[pd.Series(ts_seq_learn_y_sub)<max_l]

    ts_seq_test=ts_seq_l[:,n_best-number_s+1:n_best-number_s+2,:]
    ts_seq_test=ts_seq_test.reshape(ts_seq_test.shape[0],ts_seq_test.shape[2])
    ts_seq_test_x = ts_seq_test[:,:-1]
    
    result_df = pd.DataFrame(index=conf_loc.index,columns=conf_loc.columns)
    for column in conf_loc.columns:
        next_1_index = conf_loc[conf_loc[column] == 1].index
        for date in range(len(next_1_index)):
            dateu = next_1_index[date]
            if date!=0:
                next_1_timedelta = abs((conf_loc.loc[next_1_index[date-1]:dateu,column].index - dateu).days)
                result_df.loc[next_1_index[date-1]:dateu,column] = next_1_timedelta
            else:
                next_1_timedelta = abs((conf_loc.loc[:dateu,column].index - dateu).days)
                result_df.loc[:dateu,column] = next_1_timedelta

    rf = RandomForestRegressor(random_state=42)
    rf.fit(ts_seq_learn_x_sub,ts_seq_learn_y_sub)
    df_pred_h = conf_loc.iloc[:n_best,:]
    df_pred_h_proba=df_pred_h.copy()
    
    ts_seq_test_x = np.concatenate([ts_seq_test_x,w_l_test,w_prio_test],axis=1) 
    res_2 = pd.DataFrame(rf.predict(ts_seq_test_x)).T
    
    real = result_df.loc[datetime.strptime(df_period.columns[1], "%Y-%m-%d")+timedelta(days=1),:]
    
    tot_res.append(res_2)
    tot_real.append(real)
    
    
# =============================================================================
# Export output files    
# =============================================================================
c=0
for folder in  ['mali2012','car2013','burundi2015','ssudan2013']:
    df_pred = pd.read_csv('Input_v2/'+folder+'/conflicts.csv',index_col=(0,1))
    df_pred_new = pd.DataFrame(columns=df_pred.columns,index=df_pred.index)
    col=0
    for j in tot_res[c].iloc[0,:]:
        df_pred_new.iloc[int(j):,col]=1
        col+=1
    df_pred_new = df_pred_new.fillna(0)
    df_pred_new.to_csv('Input_v2/'+folder+'/conflicts.csv')
    c+=1


l_tot=[]
l_tot_coin=[]
max_d=[300,820,396,604]
for i in range(len(tot_res)):
    tot_real[i] = tot_real[i].fillna(max_d[i]+1)
    l_tot=l_tot+abs((tot_res[i])-tot_real[i].T.reset_index(drop=True)).values.tolist()[0]
    l_tot_coin = l_tot_coin+abs((coin_flip_experiment(len(tot_real[i].T),0.5))-tot_real[i].T.reset_index(drop=True)).values.tolist()
l_tot=pd.Series(l_tot)
l_tot_coin=pd.Series(l_tot_coin)
   
b_dist=[]
for folder in  ['mali2012','car2013','burundi2015','ssudan2013']:
    df_loc_tot = pd.read_csv('Input/'+folder+'/locations.csv')
    df_period = pd.read_csv('Input/'+folder+'/conflict_period.csv')
    df_true = pd.read_csv('Input/'+folder+'/Observed_events.csv',index_col=(0))
    df_true[df_true>1]=1
    df_pred = pd.read_csv('Input/'+folder+'/conflicts.csv',index_col=(0,1))
    df_pred=df_pred.iloc[:,:len(df_true.columns)]
    df_pred=df_pred.reset_index(drop=True)
    b_dist.append(df_true.mean(axis=1).mean())
    
l_tot_ber=[]
max_d=[300,820,396,604]
for i in range(len(tot_res)):
    tot_real[i] = tot_real[i].fillna(max_d[i]+1)
    l_tot_ber = l_tot_ber+abs((coin_flip_experiment(len(tot_real[i].T),b_dist[i]))-tot_real[i].T.reset_index(drop=True)).values.tolist()
l_tot_ber=pd.Series(l_tot_ber)

data = [l_tot, l_tot_ber, l_tot_coin]
labels = ['RF', 'Bernouilli', 'Random']
df_data = pd.DataFrame(data).T
df_data.columns=['RF', 'Bernouilli', 'Random']
df_data['min_column'] = df_data.idxmin(axis=1)

# =============================================================================
# Boxplot 
# =============================================================================
df_data['Bernouilli'] = df_data['Bernouilli'].replace(0, 1)
df_data['Random'] = df_data['Random'].replace(0, 1)

df_data['ratio_1']=np.log(df_data['Bernouilli']/df_data['RF'])
df_data['ratio_2']=np.log(df_data['Random']/df_data['RF'])
ttest_1 = ttest_1samp(df_data['ratio_1'],0)
ttest_2 = ttest_1samp(df_data['ratio_2'],0)

data_to_plot = [df_data['ratio_1'], df_data['ratio_2']]
labels = ['Bernouilli', 'Random']
melted_df = df_data[['ratio_1','ratio_2']].melt(var_name='Model', value_name='Log ratio')

sns.set(style="ticks", rc={"figure.figsize": (7, 8)})
b = sns.boxplot(data=melted_df, x="Model", y="Log ratio", width=0.4, color="white", linewidth=2, showfliers=False)
b = sns.stripplot(data=melted_df, x="Model", y="Log ratio", color="darkgrey", linewidth=1, alpha=0.4)
b.set_ylabel("Log Ratio", fontsize=20)
b.set_xlabel("Model", fontsize=20)
b.set_xticklabels(['Bernouilli', 'Random'])
b.tick_params(axis='both', which='both', labelsize=20)
b.text(x=0, y=4, s=f"P-val: {ttest_1.pvalue:.3f}", ha='center', va='center', fontsize=20, color='black')
b.text(x=1, y=4, s=f"P-val: {ttest_2.pvalue:.3f}", ha='center', va='center', fontsize=20, color='black')
b.axhline(y=0, linestyle='--', color='black', linewidth=1)
b.scatter(0,df_data['ratio_1'].mean(),marker='^',color='red')
b.scatter(1,df_data['ratio_2'].mean(),marker='^',color='red')
sns.despine(offset=5, trim=True)
plt.show()
