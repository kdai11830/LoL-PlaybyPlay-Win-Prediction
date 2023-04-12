import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import sqlalchemy as db
import json
from collections import defaultdict
import pickle
import os
import gc

# load match reference data for participants and overall match
def load_match_data(dir):
    df_match_participant_info = pd.read_pickle(f'{dir}/match_participant_info.pkl')
    df_match_info = pd.read_pickle(f'{dir}/match_team_info.pkl')

    df_match_info['teamId'] = df_match_info['teamId'].astype(int)
    df_match_info['teamIdStr'] = df_match_info['teamId'].map({100:'TEAM1', 200:'TEAM2'})
    df_match_participant_info['teamId'] = df_match_participant_info['teamId'].astype(int)
    df_match_participant_info['teamIdStr'] = df_match_participant_info['teamId'].map({100:'TEAM1', 200:'TEAM2'})
    
    return df_match_participant_info, df_match_info

# return both 2D row and flat row for each timestamp
def load_stats(dir, df_match_participant_info):
    df_stats = pd.read_pickle(f'{dir}/participant_stats.pkl')
    cols = [
        'matchId','participantId','puuid','teamId','teamIdStr','win',
        'championId','championName','lane','role','teamPosition'
    ]
    df_stats['participantId'] = df_stats['participantId'].astype(int)
    df_final_merge = df_stats.merge(df_match_participant_info[cols], how='left', on=['matchId','participantId'])
    df_final_merge = df_final_merge.dropna(subset=['teamId','win'])

    df_final_merge_flat = df_final_merge.reset_index().pivot_table(index=['matchId','sampleTimestamp'], columns=['teamIdStr','teamPosition'])
    # df_final_merge_flat = df_final_merge_flat.drop('', axis=1, level=1)
    df_final_merge_flat.columns = ['_'.join(col) for col in df_final_merge_flat.columns.values]
    df_final_merge_flat = df_final_merge_flat.reset_index()

    team1_drop = df_final_merge_flat[df_final_merge_flat['championId_TEAM1_'].notna()]['matchId'].unique()
    team2_drop = df_final_merge_flat[df_final_merge_flat['championId_TEAM2_'].notna()]['matchId'].unique()
    df_final_merge_flat = df_final_merge_flat[(~df_final_merge_flat['matchId'].isin(team1_drop)) & (~df_final_merge_flat['matchId'].isin(team2_drop))]
    
    cols_to_drop = [col for col in df_final_merge_flat.columns if col.endswith('_')]
    df_final_merge_flat = df_final_merge_flat.drop(columns=cols_to_drop)

    return df_final_merge, df_final_merge_flat

# return both final datasets with building kill data merged in
def load_building_kill(dir, df_final_merge, df_final_merge_flat, df_match_participant_info):
    df_building_kill = pd.read_pickle(f'{dir}/building_kill.pkl')

    df_building_kill.loc[df_building_kill['buildingType']=='INHIBITOR_BUILDING', 'towerType'] = 'INHIBITOR'
    df_building_kill['teamId'] = df_building_kill['teamId'].map({100: 200, 200: 100})
    df_building_kill_agg = df_building_kill.groupby([
        'matchId','sampleTimestamp','teamId','killerId','buildingType','towerType'
        ]).agg({
        'timestamp':'count'
        }).reset_index().rename(columns={'killerId':'participantId','timestamp':'buildingKillCount'})
    
    # multidimension individual building kills
    df_building_kill_pivot = df_building_kill_agg.reset_index().pivot_table(index=['matchId','sampleTimestamp','participantId'],columns=['buildingType','towerType'],values='buildingKillCount')
    df_building_kill_pivot.columns = [col[1]+'_FINAL_HIT' for col in df_building_kill_pivot.columns.values]
    running_cols = ['RUNNING_'+col for col in df_building_kill_pivot.columns.values]
    df_building_kill_pivot[running_cols] = df_building_kill_pivot.groupby(['matchId','participantId'])[df_building_kill_pivot.columns.values].transform(pd.Series.cumsum)
    df_building_kill_pivot = df_building_kill_pivot.reset_index()
    
    df_final_merge = df_final_merge.merge(df_building_kill_pivot, how='left',on=['matchId','sampleTimestamp','participantId'])


    # flat individual building kills
    df_building_kill_pivot = df_building_kill_pivot.merge(df_match_participant_info[['matchId','participantId','teamId','teamPosition']],how='left',on=['matchId','participantId'])
    df_building_kill_pivot['teamIdStr'] = df_building_kill_pivot['teamId'].map({100:'TEAM1', 200:'TEAM2'})
    df_building_kill_pivot = df_building_kill_pivot.pivot_table(index=['matchId','sampleTimestamp'],columns=['teamIdStr','teamPosition'])
    df_building_kill_pivot = df_building_kill_pivot.drop('', axis=1, level=2)
    df_building_kill_pivot.columns = ['_'.join(col) for col in df_building_kill_pivot.columns.values]
    df_building_kill_pivot = df_building_kill_pivot.reset_index()
    
    df_final_merge_flat = df_final_merge_flat.merge(df_building_kill_pivot, how='left', on=['matchId','sampleTimestamp'])

    # multidimension team building kills
    df_building_kill_pivot = df_building_kill_agg.reset_index().pivot_table(index=['matchId','sampleTimestamp','teamId'],columns=['buildingType','towerType'],values='buildingKillCount').fillna(0)
    df_building_kill_pivot.columns = [col[1]+'_TEAM_KILLS' for col in df_building_kill_pivot.columns.values]
    running_cols = ['RUNNING_'+col for col in df_building_kill_pivot.columns.values]
    df_building_kill_pivot[running_cols] = df_building_kill_pivot.groupby(['matchId','teamId'])[df_building_kill_pivot.columns.values].transform(pd.Series.cumsum)
    df_building_kill_pivot = df_building_kill_pivot.reset_index()

    df_final_merge = df_final_merge.merge(df_building_kill_pivot, how='left',on=['matchId','sampleTimestamp','teamId']).fillna(0)

    # flat team building kills
    # TODO fix this shit
    df_building_kill_pivot = df_building_kill_pivot.merge(df_match_participant_info[['matchId','teamId','teamPosition']],how='left',on=['matchId','teamId'])
    df_building_kill_pivot['teamIdStr'] = df_building_kill_pivot['teamId'].map({100:'TEAM1', 200:'TEAM2'})
    df_building_kill_pivot = df_building_kill_pivot.pivot_table(index=['matchId','sampleTimestamp'],columns=['teamIdStr'])
    # df_building_kill_pivot = df_building_kill_pivot.drop('', axis=1, level=2)
    df_building_kill_pivot.columns = ['_'.join(col) for col in df_building_kill_pivot.columns.values]
    df_building_kill_pivot = df_building_kill_pivot.reset_index()
    
    df_final_merge_flat = df_final_merge_flat.merge(df_building_kill_pivot, how='left', on=['matchId','sampleTimestamp'])
    
    # do ffills for running cols
    running_cols = [col for col in df_final_merge.columns.values if col.startswith('RUNNING')]
    df_final_merge[running_cols] = df_final_merge.groupby(['matchId'])[running_cols].ffill()
    running_cols = [col for col in df_final_merge_flat.columns.values if col.startswith('RUNNING')]
    df_final_merge_flat[running_cols] = df_final_merge_flat.groupby(['matchId'])[running_cols].ffill()
    df_final_merge = df_final_merge.fillna(0)
    df_final_merge_flat = df_final_merge_flat.fillna(0)

    return df_final_merge, df_final_merge_flat

# return both final datasets with champion kill data merged in
# NOTE: not including special kills (first blood, ace, multikills)
def load_champion_kill(dir, df_final_merge, df_final_merge_flat, df_match_participant_info):
    df_champion_kill = pd.read_pickle(f'{dir}/champion_kill.pkl')
    df_champion_kill_merge = df_champion_kill.merge(
        df_match_participant_info[['matchId','participantId','teamPosition','teamIdStr']].rename(
        columns={'participantId':'killerId','teamIdStr':'killerTeamIdStr','teamPosition':'killerTeamPosition'}
        ), how='left', on=['matchId','killerId'])
    df_champion_kill_merge = df_champion_kill_merge.merge(
        df_match_participant_info[['matchId','participantId','teamPosition','teamIdStr']].rename(
        columns={'participantId':'victimId','teamIdStr':'victimTeamIdStr','teamPosition':'victimTeamPosition'}
        ), how='left', on=['matchId','victimId'])
    df_champion_kill_agg = df_champion_kill_merge.groupby(
        ['matchId','sampleTimestamp','killerTeamIdStr','killerTeamPosition','killerId']
        ).agg({'timestamp':'count'}).reset_index().rename(columns={'timestamp':'KILL_COUNT'})
    df_champion_death_agg = df_champion_kill_merge.groupby(
        ['matchId','sampleTimestamp','victimTeamIdStr','victimTeamPosition','victimId']
        ).agg({'timestamp':'count'}).reset_index().rename(columns={'timestamp':'DEATH_COUNT'})
    
    df_champion_kill_agg['RUNNING_KILL_COUNT'] = df_champion_kill_agg.groupby(['matchId','killerId'])['KILL_COUNT'].transform(pd.Series.cumsum)
    df_champion_death_agg['RUNNING_DEATH_COUNT'] = df_champion_death_agg.groupby(['matchId','victimId'])['DEATH_COUNT'].transform(pd.Series.cumsum)

    # merge to multidimension df
    df_final_merge = df_final_merge.merge(df_champion_kill_agg[['matchId','sampleTimestamp','killerId','KILL_COUNT','RUNNING_KILL_COUNT']].rename(columns={'killerId':'participantId'}), how='left', on=['matchId','sampleTimestamp','participantId'])
    df_final_merge = df_final_merge.merge(df_champion_death_agg[['matchId','sampleTimestamp','victimId','DEATH_COUNT','RUNNING_DEATH_COUNT']].rename(columns={'victimId':'participantId'}), how='left', on=['matchId','sampleTimestamp','participantId'])
    df_final_merge['RUNNING_KILL_COUNT'] = df_final_merge.groupby(['matchId','participantId'])['RUNNING_KILL_COUNT'].ffill()
    df_final_merge['RUNNING_DEATH_COUNT'] = df_final_merge.groupby(['matchId','participantId'])['RUNNING_DEATH_COUNT'].ffill()
    df_final_merge = df_final_merge.fillna(0)

    # merge to flat df
    df_champion_kill_agg_pivot = df_champion_kill_agg.drop(columns=['killerId']).pivot_table(index=['matchId','sampleTimestamp'], columns=['killerTeamIdStr','killerTeamPosition'])
    df_champion_kill_agg_pivot.columns = ['_'.join(col) for col in df_champion_kill_agg_pivot.columns.values]
    cols = list(df_champion_kill_agg_pivot.columns)
    df_champion_kill_agg_pivot = df_champion_kill_agg_pivot.reset_index()

    df_champion_death_agg_pivot = df_champion_death_agg.drop(columns=['victimId']).pivot_table(index=['matchId','sampleTimestamp'], columns=['victimTeamIdStr','victimTeamPosition'])
    df_champion_death_agg_pivot.columns = ['_'.join(col) for col in df_champion_death_agg_pivot.columns.values]
    cols.extend(list(df_champion_death_agg_pivot.columns))
    df_champion_death_agg_pivot = df_champion_death_agg_pivot.reset_index()
    
    df_final_merge_flat = df_final_merge_flat.merge(df_champion_kill_agg_pivot, how='left', on=['matchId','sampleTimestamp'])
    df_final_merge_flat = df_final_merge_flat.merge(df_champion_death_agg_pivot, how='left', on=['matchId','sampleTimestamp'])
    cols = [col for col in cols if col.startswith('RUNNING')]
    df_final_merge_flat[cols] = df_final_merge_flat.groupby('matchId')[cols].ffill().fillna(0)
    df_final_merge_flat = df_final_merge_flat.fillna(0)

    return df_final_merge, df_final_merge_flat

# merge dragon kill data
# dragon kill only by timestamp
# NOTE: NOT USED, USE load_objective_kill INSTEAD
def load_dragon(dir, df_final_merge, df_final_merge_flat):
    df_dragon = pd.read_pickle(f'{dir}/dragon_soul_given.pkl')
    df_dragon['name'] = df_dragon['name'].str.upper()
    df_dragon['teamIdStr'] = df_dragon['teamId'].map({100:'TEAM1', 200:'TEAM2'})

    # multidim
    tmp = df_dragon.groupby([
        'matchId','sampleTimestamp','teamId','name'
        ]).agg({'timestamp':'count'}).rename(columns={'timestamp':'DRAGON_KILL'}).reset_index().pivot_table(
        index=['matchId','sampleTimestamp','teamId'], columns=['name']
        )
    tmp.columns = ['_'.join(col) for col in tmp.columns.values]
    df_final_merge = df_final_merge.merge(tmp.reset_index(), how='left', on=['matchId','sampleTimestamp','teamId'])
    df_final_merge[tmp.columns.values] = df_final_merge.groupby(['matchId','teamId'])[tmp.columns.values].ffill()
    df_final_merge = df_final_merge.fillna(0)

    # flat
    tmp = df_dragon.groupby([
    'matchId','sampleTimestamp','teamIdStr','name'
    ]).agg({'timestamp':'count'}).rename(columns={'timestamp':'DRAGON_KILL'}).reset_index().pivot_table(
    index=['matchId','sampleTimestamp'], columns=['name','teamIdStr']
    )
    tmp.columns = ['_'.join(col) for col in tmp.columns.values]
    df_final_merge_flat = df_final_merge_flat.merge(tmp.reset_index(), how='left', on=['matchId','sampleTimestamp'])
    df_final_merge_flat[tmp.columns.values] = df_final_merge_flat.groupby(['matchId'])[tmp.columns.values].ffill()
    df_final_merge_flat = df_final_merge_flat.fillna(0)

    return df_final_merge, df_final_merge_flat


def load_objective_kill(dir, df_final_merge, df_final_merge_flat):
    df_elite_kill = pd.read_pickle(f'{dir}/elite_monster_kill.pkl')
    df_elite_kill = df_elite_kill.rename(columns={'killerTeamId':'teamId','monsterSubType':'monster'})
    df_elite_kill['teamIdStr'] = df_elite_kill['teamId'].map({100:'TEAM1', 200:'TEAM2'})
    df_elite_kill['monster'] = df_elite_kill['monster'].fillna(df_elite_kill['monsterType'])

    tmp = df_elite_kill.groupby(
        ['matchId','sampleTimestamp','teamId','monster']
        ).agg({'timestamp':'count'}).rename(columns={'timestamp':'OBJECTIVE'}).reset_index().pivot_table(
        index=['matchId','sampleTimestamp','teamId'], columns=['monster']
        )
    tmp.columns = ['_'.join(col) for col in tmp.columns.values]
    cols = tmp.columns.values
    tmp = tmp.reset_index()
    tmp[cols] = tmp.groupby(['matchId','teamId'])[cols].transform(pd.Series.cumsum)

    df_final_merge = df_final_merge.merge(tmp, how='left', on=['matchId','sampleTimestamp','teamId'])
    df_final_merge[cols] = df_final_merge.groupby(['matchId','teamId'])[cols].ffill().fillna(0)

    tmp = df_elite_kill.groupby(
        ['matchId','sampleTimestamp','teamIdStr','monster']
        ).agg({'timestamp':'count'}).rename(columns={'timestamp':'OBJECTIVE'}).reset_index().pivot_table(
        index=['matchId','sampleTimestamp'], columns=['monster','teamIdStr']
        )
    tmp.columns = ['_'.join(col) for col in tmp.columns.values]
    cols = tmp.columns.values
    tmp = tmp.reset_index()
    tmp[cols] = tmp.groupby(['matchId'])[cols].transform(pd.Series.cumsum)

    df_final_merge_flat = df_final_merge_flat.merge(tmp, how='left', on=['matchId','sampleTimestamp'])
    df_final_merge_flat[cols] = df_final_merge_flat.groupby(['matchId'])[cols].ffill().fillna(0)

    return df_final_merge, df_final_merge_flat



def load_items(dir, df_final_merge, df_final_merge_flat, df_match_participant_info):
    df_items_purchased = pd.read_pickle(f'{dir}/item_purchased.pkl')

    tmp = df_items_purchased.groupby(
        ['matchId','sampleTimestamp','participantId']
        ).agg({'itemId':'count'}).rename(columns={'itemId':'ITEMS_PURCHASED'}).reset_index()
    tmp['ITEMS_PURCHASED'] = tmp.groupby(['matchId','participantId'])['ITEMS_PURCHASED'].transform(pd.Series.cumsum)

    df_final_merge = df_final_merge.merge(tmp, how='left', on=['matchId','sampleTimestamp','participantId'])
    df_final_merge['ITEMS_PURCHASED'] = df_final_merge.groupby(['matchId','participantId'])['ITEMS_PURCHASED'].ffill().fillna(0)

    tmp = df_items_purchased.groupby(
        ['matchId','sampleTimestamp','participantId']
        ).agg({'itemId':'count'}).rename(columns={'itemId':'ITEMS_PURCHASED'}).reset_index().merge(
        df_match_participant_info[['matchId','participantId','teamIdStr','teamPosition']],how='left',on=['matchId','participantId']
        ).pivot_table(index=['matchId','sampleTimestamp'], columns=['teamIdStr','teamPosition'], values=['ITEMS_PURCHASED']).drop('', axis=1, level=2)

    cols = tmp.columns = ['_'.join(col) for col in tmp.columns.values]
    tmp = tmp.reset_index()
    tmp[cols] = tmp.groupby('matchId')[cols].transform(pd.Series.cumsum)

    df_final_merge_flat = df_final_merge_flat.merge(tmp, how='left', on=['matchId','sampleTimestamp'])
    df_final_merge_flat[cols] = df_final_merge_flat.groupby('matchId')[cols].ffill().fillna(0)

    return df_final_merge, df_final_merge_flat


def load_turretplate(dir, df_final_merge, df_final_merge_flat):
    df_turretplate = pd.read_pickle(f'{dir}/turret_plate_destroyed.pkl')
    df_turretplate['teamId'] = df_turretplate['teamId'].map({100:200,200:100})
    df_turretplate['teamIdStr'] = df_turretplate['teamId'].map({100:'TEAM1', 200:'TEAM2'})
    df_turretplate = df_turretplate.rename(columns={'killerId':'participantId'})
    tmp = df_turretplate.groupby(
        ['matchId','sampleTimestamp','teamId','laneType']
        ).agg({'timestamp':'count'}).rename(columns={'timestamp':'TURRET_PLATE_KILL'}).reset_index().pivot_table(
        index=['matchId','sampleTimestamp','teamId'], columns=['laneType']
        )
    tmp.columns = ['_'.join(col) for col in tmp.columns.values]
    cols = tmp.columns.values
    tmp = tmp.reset_index()
    tmp[cols] = tmp.groupby(['matchId','teamId'])[cols].transform(pd.Series.cumsum)

    df_final_merge = df_final_merge.merge(tmp, how='left', on=['matchId','sampleTimestamp','teamId'])
    df_final_merge[cols] = df_final_merge.groupby(['matchId','teamId'])[cols].ffill().fillna(0)

    tmp = df_turretplate.groupby(
        ['matchId','sampleTimestamp','teamIdStr','laneType']
        ).agg({'timestamp':'count'}).rename(columns={'timestamp':'TURRET_PLATE_KILL'}).reset_index().pivot_table(
        index=['matchId','sampleTimestamp'], columns=['laneType','teamIdStr']
        )
    tmp.columns = ['_'.join(col) for col in tmp.columns.values]
    cols = tmp.columns.values
    tmp = tmp.reset_index()
    tmp[cols] = tmp.groupby(['matchId'])[cols].transform(pd.Series.cumsum)

    df_final_merge_flat = df_final_merge_flat.merge(tmp, how='left', on=['matchId','sampleTimestamp'])
    df_final_merge_flat[cols] = df_final_merge_flat.groupby('matchId')[cols].ffill().fillna(0)

    return df_final_merge, df_final_merge_flat


def load_wardplaced(dir, df_final_merge, df_final_merge_flat, df_match_participant_info):
    df_ward = pd.read_pickle(f'{dir}/ward_placed.pkl')
    df_ward = df_ward.rename(columns={'creatorId':'participantId'})
    df_ward = df_ward.merge(df_match_participant_info[['matchId','teamId','participantId','teamPosition']],how='left', on=['matchId','participantId'])
    df_ward['teamIdStr'] = df_ward['teamId'].map({100:'TEAM1', 200:'TEAM2'})

    tmp = df_ward.groupby(['matchId','sampleTimestamp','teamId']).agg({'timestamp':'count'}).rename(columns={'timestamp':'WARDS_PLACED'}).reset_index()
    tmp['WARDS_PLACED'] = tmp.groupby(['matchId','teamId'])['WARDS_PLACED'].transform(pd.Series.cumsum)

    df_final_merge = df_final_merge.merge(tmp, how='left', on=['matchId','sampleTimestamp','teamId'])
    df_final_merge['WARDS_PLACED'] = df_final_merge.groupby(['matchId','teamId'])['WARDS_PLACED'].ffill().fillna(0)

    tmp = df_ward.groupby(
        ['matchId','sampleTimestamp','teamIdStr']
        ).agg({'timestamp':'count'}).rename(columns={'timestamp':'WARDS_PLACED'}).reset_index().pivot_table(
        index=['matchId','sampleTimestamp'], columns=['teamIdStr']
        )
    tmp.columns = ['_'.join(col) for col in tmp.columns.values]
    cols = tmp.columns.values
    tmp = tmp.reset_index()
    tmp[cols] = tmp.groupby(['matchId'])[cols].transform(pd.Series.cumsum)

    df_final_merge_flat = df_final_merge_flat.merge(tmp, how='left', on=['matchId','sampleTimestamp'])
    df_final_merge_flat[cols] = df_final_merge_flat.groupby('matchId')[cols].ffill().fillna(0)

    return df_final_merge, df_final_merge_flat

def load_summonerstats(dir, df_final_merge, df_final_merge_flat,df_match_participant_info):
    df_summonerstats = pd.read_pickle(f'{dir}/summoner_stats.pkl')
    df_participant_map = pd.read_pickle(f'{dir}/participant_puuid_map.pkl')
    df_summonerstats = df_summonerstats.rename(columns={
        'total_kills':'playerStatsKills',
        'total_deaths':'playerStatsDeaths',
        'num_wins':'playerStatsWins',
        'num_matches':'playerStatsMatches',
        'kd_ratio':'playerStatsKd',
        'win_ratio':'playerStatsWinratio'
    })

    df_participant_map = df_participant_map.merge(df_summonerstats, how='left', on='puuid')
    df_participant_map = df_participant_map.merge(df_match_participant_info[['matchId','teamId','participantId','teamPosition']],how='left', on=['matchId','participantId'])
    df_participant_map['teamIdStr'] = df_participant_map['teamId'].map({100:'TEAM1', 200:'TEAM2'})

    df_final_merge = df_final_merge.merge(df_summonerstats, how='left', on='puuid')
    
    tmp = df_participant_map.pivot_table(index=['matchId'], columns=['teamPosition','teamIdStr'], values=['playerStatsKills','playerStatsDeaths','playerStatsWins','playerStatsMatches','playerStatsKd','playerStatsWinratio'])
    tmp = tmp.loc[:,tmp.columns[tmp.columns.get_level_values(1) != '']].fillna(0)
    tmp.columns = ['_'.join(x) for x in tmp.columns.values]

    df_final_merge_flat = df_final_merge_flat.merge(tmp, how='left', on=['matchId'])

    return df_final_merge, df_final_merge_flat




def load_pipeline(dir):
    print("loading match data...")
    df_match_participant_info, df_match_info = load_match_data(dir)

    print("loading participant stats...")
    df_final_merge, df_final_merge_flat = load_stats(dir, df_match_participant_info)

    print("loading building kill data...")
    df_final_merge, df_final_merge_flat = load_building_kill(dir, df_final_merge, df_final_merge_flat, df_match_participant_info)

    print("loading champion kill data...")
    df_final_merge, df_final_merge_flat = load_champion_kill(dir, df_final_merge, df_final_merge_flat, df_match_participant_info)

    print("loading objective kill data...")
    df_final_merge, df_final_merge_flat = load_objective_kill(dir, df_final_merge, df_final_merge_flat)

    print("loading item purchase data...")
    df_final_merge, df_final_merge_flat = load_items(dir, df_final_merge, df_final_merge_flat, df_match_participant_info)

    print("loading turret plate data...")
    df_final_merge, df_final_merge_flat = load_turretplate(dir, df_final_merge, df_final_merge_flat)

    print("loading ward placed data...")
    df_final_merge, df_final_merge_flat = load_wardplaced(dir, df_final_merge, df_final_merge_flat, df_match_participant_info)

    print("loading summoner stats data...")
    df_final_merge, df_final_merge_flat = load_summonerstats(dir, df_final_merge, df_final_merge_flat,df_match_participant_info)

    print('data loading complete!')

    return df_final_merge, df_final_merge_flat


if __name__ == '__main__':
    with open('config.json','r') as f:
        cfg = json.load(f)
    dir = cfg['data_dir']
    outdir = cfg['output_dir']

    df_final_merge, df_final_merge_flat = load_pipeline(dir)
    
    df_final_merge.to_pickle(f'{outdir}/df_final_merge.pkl')
    df_final_merge_flat.to_pickle(f'{outdir}/df_final_merge_flat.pkl')



