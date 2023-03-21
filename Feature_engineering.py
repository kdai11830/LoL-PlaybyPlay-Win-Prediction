# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 12:16:28 2023
Feature Engineering py file
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

categorical_features =['matchId','sampleTimestamp','TURRET_PLATE_KILL_BOT_LANE','TURRET_PLATE_KILL_MID_LANE','TURRET_PLATE_KILL_TOP_LANE','dragon_kill']


## early game selection filter
def early_game_filter(df,time_cutoff,dragon_cutoff):
    df['dragon_kill'] = df['OBJECTIVE_AIR_DRAGON']+df['OBJECTIVE_CHEMTECH_DRAGON']+df['OBJECTIVE_EARTH_DRAGON']+df['OBJECTIVE_FIRE_DRAGON']+df['OBJECTIVE_HEXTECH_DRAGON']
    Dragon_Kill = pd.DataFrame(df[['dragon_kill','matchId','sampleTimestamp','teamIdStr']].groupby(['matchId','sampleTimestamp'])['dragon_kill'].sum())
    Dragon_Kill = Dragon_Kill.rename(columns={'dragon_kill':'Total_dragon_kill'})
    df = df.merge(Dragon_Kill,on = (['matchId','sampleTimestamp']),how='left')
    df['Total_dragon_kill'] = df['Total_dragon_kill']/5
    
    time_cutoff = 60000*20
    dragon_cutoff = 3
    mask1 = df['sampleTimestamp'] < time_cutoff
    mask2 = df['Total_dragon_kill']<dragon_cutoff
    return df[(mask1)&(mask2)]

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
    
    charfeatures = ['damageStats_totalDamageTaken','damageStats_totalDamageDoneToChampions','championStats_movementSpeed',
                    'championStats_magicResist','championStats_magicPenPercent','championStats_healthMax','championStats_ccReduction',
                    'championStats_attackSpeed','championStats_attackDamage','championStats_armor','championStats_abilityPower']
    feature_storage = {}
    for datacolumn in charfeatures:
        T1_df = champ_char_handler(T1_dfearly,datacolumn)
        T2_df = champ_char_handler(T2_dfearly,datacolumn)
        data_diff = champ_char_diff_handler(T1_df,T2_df)
        feature_storage[datacolumn] = T1_df.merge(data_diff,on =['matchId','sampleTimestamp'],how='left')
    
    for features in feature_mapping:
        feature_storage[features] =player_filtering(feature_storage,features,feature_mapping[features],feature_storage)
    
    categorical_feature = df[categorical_features]
    categorical_feature = categorical_feature.groupby(['matchId','sampleTimestamp']).first()
    
    ## Creating full timestamp data Frame
    sampleTS = pd.DataFrame({'sampleTimestamp':list(set(T1_dfearly['sampleTimestamp']))}).sort_values(['sampleTimestamp'])
    matchId = pd.DataFrame({'matchId':list(set(T1_dfearly['matchId']))}).sort_values(['matchId'])
    fullXdf = matchId.merge(sampleTS,how='cross')
    
    ## Creating Value_Feature, Linear Interpolation
    for features in feature_mapping:
        Xdf = feature_storage[features]
        fullXdf = fullXdf.merge(Xdf.reset_index(),on =['matchId','sampleTimestamp'],how='left').set_index(['matchId','sampleTimestamp'])
    fullXdf = fullXdf.merge(T1_gold,on =['matchId','sampleTimestamp'],how='left')
    
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
    pklfile = 'F:////League of legends Game Prediction//LoL-PlaybyPlay-Win-Prediction//data_merged//data_merged//df_final_merge.pkl'
    X,Y = feature_engineering(pklfile) 
    with open('F:////League of legends Game Prediction//LoL-PlaybyPlay-Win-Prediction//TrainingData//X.npy', 'wb') as f:
        np.save(f,X)
    with open('F:////League of legends Game Prediction//LoL-PlaybyPlay-Win-Prediction//TrainingData//Y.npy', 'wb') as f:
        np.save(f,Y)
        