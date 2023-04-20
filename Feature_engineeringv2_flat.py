# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 12:16:28 2023
Feature Engineering version 2 - smarter upgrades.
@author: Jian Wang
"""
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

feature_mapping = {'damageStats_totalDamageTaken': ('p1','p2','p5'),
                  'damageStats_totalDamageDoneToChampions':('p1','p2','p3','p4','p5'),
                  'championStats_movementSpeed':('p2','p3','p5'),
                  'championStats_healthMax':('p1','p5'),
                  'championStats_attackSpeed':('p4'),
                  'championStats_attackDamage':('p4'),
                  'championStats_armor':('p1','p2','p5'),
                  'championStats_abilityPower':('p3','p5')}

charfeatures = ['damageStats_totalDamageTaken','damageStats_totalDamageDoneToChampions','championStats_movementSpeed',
                'championStats_magicResist','championStats_magicPenPercent','championStats_healthMax','championStats_ccReduction',
                'championStats_attackSpeed','championStats_attackDamage','championStats_armor','championStats_abilityPower']

categorical_features =['matchId','sampleTimestamp','TURRET_PLATE_KILL_BOT_LANE','TURRET_PLATE_KILL_MID_LANE','TURRET_PLATE_KILL_TOP_LANE','dragon_kill','OBJECTIVE_RIFTHERALD']


# simpler time filter
def early_game_filter(df, time_cutoff=1200000):
    valid_matches = df[df['sampleTimestamp']>=time_cutoff]['matchId'].unique()
    return df[df['matchId'].isin(valid_matches)]

def gold_handler(df):
    id_cols = ['matchId','sampleTimestamp']
    feature_cols = [x for x in df.columns.values if x.startswith('totalGold')]
    t1_cols = [x for x in feature_cols if 'TEAM1' in x]
    t2_cols = [x for x in feature_cols if 'TEAM2' in x]

    df1 = df[id_cols + t1_cols].set_index(id_cols)
    df2 = df[id_cols + t2_cols].set_index(id_cols)
    diff = df1 - df2
    diff.columns = [x + '_diff' for x in diff.columns.values]
    df_gold = df1.merge(diff)
    return df_gold

def xp_handler(df):
    id_cols = ['matchId','sampleTimestamp']
    feature_cols = [x for x in df.columns.values if x.startswith('xp')]
    t1_cols = [x for x in feature_cols if 'TEAM1' in x]
    t2_cols = [x for x in feature_cols if 'TEAM2' in x]
    df_xp = df[id_cols + feature_cols]
    
    df_xp['xpMax_TEAM1'] = df_xp[t1_cols].max(axis=1)
    df_xp['xpAvg_TEAM1'] = df_xp[t1_cols].mean(axis=1)
    df_xp['xpAvg_TEAM2'] = df_xp[t2_cols].mean(axis=1)
    df_xp['xpLead'] = df_xp['xpMax_TEAM1'] - df_xp['xpAvg_TEAM2']
    df_xp['xpGapAdcMid'] = df_xp['xp_TEAM1_JUNGLE'] - df_xp['xp_TEAM2_MIDDLE']
    df_xp['xpAvgGap'] = df_xp['xpAvg_TEAM1'] - df_xp['xpAvg_TEAM2']
    return df_xp[id_cols + ['xpAvg_TEAM1','xpAvgGap','xpGapAdcMid','xpLead']]

def damage_handler(df):
    id_cols = ['matchId','sampleTimestamp']
    feature_cols = [x for x in df.columns.values if x.startswith('damageStats_totalDamageDoneToChampions') and not x.endswith('UTILITY')]
    return df[id_cols + feature_cols]


## gold features handler
def gold_handler(df):
    process_df = df[['matchId','sampleTimestamp','participantId','totalGold']]
    process_df = process_df.groupby(['matchId','sampleTimestamp','participantId']).first().unstack(level=2).reset_index()
    renamedict = {}
    renamedict[0] = 'matchId'
    renamedict[1] = 'sampleTimestamp'
    for i in range(2,7):
        renamedict[i] ='totalGold_p'+str(i-1)
    return pd.DataFrame(np.array(process_df)).rename(columns = renamedict).set_index(['matchId','sampleTimestamp'])


def champ_char_handler(df,char):
    process_df = df[['matchId','sampleTimestamp','participantId',char]]
    process_df = process_df.groupby(['matchId','sampleTimestamp','participantId']).first().unstack(level=2).reset_index()
    renamedict = {}
    renamedict[0] = 'matchId'
    renamedict[1] = 'sampleTimestamp'
    for i in range(2,7):
        renamedict[i] =str(char)+'_p'+str(i-1)
    return pd.DataFrame(np.array(process_df)).rename(columns = renamedict).set_index(['matchId','sampleTimestamp'])


def champ_char_diff_handler(df1,df2):
    diff = df1-df2
    colrename = {}
    for cn in diff.columns:
        colrename[cn] = cn+'_diff'
    diff.rename(columns = colrename,inplace =True)
    return diff

def player_filtering(dicts,features,player_list,feature_storage):
    test = list(feature_storage[features].columns)
    masks = [any(c in L for c in player_list) for L in test]
    newcol = feature_storage[features].columns[[masks]]
    return feature_storage[features][newcol]

def standardization(featuredf):
    featuredf = featuredf.set_index(['matchId','sampleTimestamp'])
    featuredf = featuredf-featuredf.mean()
    featuredf = featuredf/featuredf.var()
    featuredf = featuredf.groupby(['matchId'])[list(featuredf.columns)].ffill()
    return featuredf
    
def feature_engineering(pklfile):
    df = pd.read_pickle(pklfile)
    dfearly = early_game_filter(df,60000*20,3)
    ## team 1 and team2
    T1_dfearly = dfearly[dfearly['teamIdStr']=='TEAM1']
    T2_dfearly = dfearly[dfearly['teamIdStr']=='TEAM2']
    
    T1_gold = gold_handler(T1_dfearly)
    T2_gold = gold_handler(T2_dfearly)
    gold_diff = T1_gold - T2_gold

    colrename = {}
    for cn in gold_diff.columns:
        colrename[cn] = cn+'_diff'
    gold_diff.rename(columns = colrename,inplace =True)
    T1_gold = T1_gold.merge(gold_diff,on =['matchId','sampleTimestamp'],how='left')
    
    ## xp feature engineering
    xp = champ_char_handler(T1_dfearly,'xp').merge(champ_char_handler(T2_dfearly,'xp').reset_index(),on =['matchId','sampleTimestamp'],how='left')
    xp['T1maxxp'] = xp[['xp_p1_x','xp_p2_x','xp_p3_x','xp_p4_x','xp_p5_x']].max(axis=1)
    xp['T1avg'] = xp[['xp_p1_x','xp_p2_x','xp_p3_x','xp_p4_x']].mean(axis=1)
    xp['T2avg'] = xp[['xp_p1_y','xp_p2_y','xp_p3_y','xp_p4_y']].mean(axis=1)
    xp['xplead'] = xp['T1maxxp']-xp['T2avg']
    xp['adc_xpgap_mid'] = xp['xp_p4_x'] - xp['xp_p3_y']
    xp['xp_avggap'] = xp['T1avg']-xp['T2avg']
    xp = xp[['matchId','sampleTimestamp','T1avg','xp_avggap','adc_xpgap_mid','xplead']]
    ## xp standardize, panel standardization (I am actually not sure if panel average makes sense here. but our goal is just to standardize the 
    ## feature, so why not?)
    xp =standardization(xp)
    
    ## total damage feature engineering
    total_dmg = champ_char_handler(T1_dfearly,'damageStats_totalDamageDoneToChampions').merge(champ_char_handler(T2_dfearly,'damageStats_totalDamageDoneToChampions').reset_index(),on =['matchId','sampleTimestamp'],how='left')
    total_dmg = total_dmg.loc[:,(total_dmg.columns!='damageStats_totalDamageDoneToChampions_p5_x')\
    &(total_dmg.columns!='damageStats_totalDamageDoneToChampions_p5_y')]
    ## standardization
    total_dmg = standardization(total_dmg)
    
    ## categorical features (Plate infomration)
    categorical_feature_T1 = T1_dfearly[categorical_features]
    categorical_feature_T1 = categorical_feature_T1.groupby(['matchId','sampleTimestamp']).first()
    ## These categorical features might only matter by differences from my perpsective
    categorical_feature_T2 = T2_dfearly[categorical_features]
    categorical_feature_T2 = categorical_feature_T2.groupby(['matchId','sampleTimestamp']).first()
    categorical_feature = categorical_feature_T1.merge(categorical_feature_T2,on =['matchId','sampleTimestamp'],how='left')
    
    ## Creating full timestamp data Frame
    sampleTS = pd.DataFrame({'sampleTimestamp':list(set(T1_dfearly['sampleTimestamp']))}).sort_values(['sampleTimestamp'])
    matchId = pd.DataFrame({'matchId':list(set(T1_dfearly['matchId']))}).sort_values(['matchId'])
    fullXdf = matchId.merge(sampleTS,how='cross')
    
    ## Creating Value_Feature, Linear Interpolation
    fullXdf = fullXdf.merge(T1_gold.reset_index(),on =['matchId','sampleTimestamp'],how='left')
    fullXdf = fullXdf.merge(xp.reset_index(),on =['matchId','sampleTimestamp'],how='left')
    fullXdf = fullXdf.merge(total_dmg.reset_index(),on =['matchId','sampleTimestamp'],how='left')
    fullXdf = fullXdf.set_index(['matchId','sampleTimestamp'])
    for col in fullXdf:
        fullXdf[col] = pd.to_numeric(fullXdf[col], errors='coerce')
    
    gb = fullXdf.groupby('matchId')
    Value_Feature = np.array([gb.get_group(x).interpolate(method ='linear', limit_direction ='forward').to_numpy() for x in gb.groups])
    
    
    Categorical_feature = fullXdf.merge(categorical_feature.reset_index(),on =['matchId','sampleTimestamp'],how='left').set_index(['matchId','sampleTimestamp'])
    Categorical_feature = Categorical_feature.groupby(['matchId'])[list(categorical_feature.columns)].ffill()
    gb = Categorical_feature.groupby('matchId')
    Categorical_feature = np.array([gb.get_group(x).to_numpy() for x in gb.groups])
    
    ## create X variable
    X = np.dstack((Value_Feature,Categorical_feature)).astype(np.float32)
    
    ## create Y variable
    YVariable = fullXdf.reset_index().groupby('matchId').first().reset_index()[['matchId']]
    YVariable = YVariable.merge(T1_dfearly[['matchId','win']].groupby('matchId').first(),on='matchId',how='left')
    Y = np.array(YVariable['win']).astype(np.float32)
    return X,Y

if __name__ == '__main__':
    pklfile = 'data_final/df_final_merge.pkl'
    X,Y = feature_engineering(pklfile) 
    with open('data_training/X.npy', 'wb') as f:
        np.save(f,X)
    with open('data_training/Y.npy', 'wb') as f:
        np.save(f,Y)
        