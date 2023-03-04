import numpy as np
import pandas as pd
import json
import requests
import time
import base64
import os
from collections import defaultdict
import pickle
import random

# get current list of master, gm, or challenger summoner names
# assume we only want ranked solo 5x5 and not ranked flex
def get_high_tier_summoner_names(api_key, tiers=['MASTER','GM','CHALLENGER']):
    tiers = [tier.upper() for tier in tiers]
    summoner_list = []
    headers = {
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Charset": "application/x-www-form-urlencoded; charset=UTF-8",
        "Origin": "https://developer.riotgames.com",
        "X-Riot-Token": api_key
    }
    url_dict = {
        'MASTER': 'https://na1.api.riotgames.com/lol/league/v4/masterleagues/by-queue/RANKED_SOLO_5x5',
        'GM': 'https://na1.api.riotgames.com/lol/league/v4/grandmasterleagues/by-queue/RANKED_SOLO_5x5',
        'CHALLENGER': 'https://na1.api.riotgames.com/lol/league/v4/challengerleagues/by-queue/RANKED_SOLO_5x5'
    }
    for tier in tiers:
        url = url_dict[tier]
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            res = json.loads(response.content.decode('utf-8'))
            for summoner in res['entries']:
                summoner_list.append(summoner['summonerName'])
        except:
            raise
    return summoner_list

# get summoner names for diamond and below
# need to specify tier and divisions
def get_lower_tier_summoner_names(api_key, tiers=['DIAMOND'], divisions=['I','II','III']):
    tiers = [tier.upper() for tier in tiers]
    divisions = [division.upper() for division in divisions]
    summoner_list = []
    headers = {
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Charset": "application/x-www-form-urlencoded; charset=UTF-8",
        "Origin": "https://developer.riotgames.com",
        "X-Riot-Token": api_key
    }
    for tier in tiers:
        for division in divisions:
            url = f'https://na1.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/{tier}/{division}?page=1'
            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                res = json.loads(response.content.decode('utf-8'))
                for summoner in res:
                    summoner_list.append(summoner['summonerName'])
            except:
                raise
    return summoner_list


# get current list of players in specified divisions from diamond III up
# assume we only want ranked solo 5x5 and not ranked flex
# does rank matter for summoners?
def get_summoner_names(api_key, high_tiers=['MASTER','GM','CHALLENGER'],lower_tiers=['DIAMOND'], divisions=['I','II','III']):
    lower_tiers = [tier.upper() for tier in lower_tiers]
    high_tiers = [tier.upper() for tier in high_tiers]
    divisions = [division.upper() for division in divisions]

    high_tier_summoner_list = get_high_tier_summoner_names(api_key, tiers=high_tiers)
    lower_tier_summoner_list = get_lower_tier_summoner_names(api_key, tiers=lower_tiers, divisions=divisions)

    return list(set(high_tier_summoner_list + lower_tier_summoner_list))


# get puuid (global player ID) from summoner names
def get_riot_puuid(summoner_list, api_key):
    puuid_list = []
    headers = {
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Charset": "application/x-www-form-urlencoded; charset=UTF-8",
        "Origin": "https://developer.riotgames.com",
        "X-Riot-Token": api_key
    }


    for i,summoner in enumerate(summoner_list):
        if i % 20 == 0:
            print(f'puuid {i}/{len(summoner_list)}')
        # use americas regional routing value
        url = f'https://na1.api.riotgames.com/lol/summoner/v4/summoners/by-name/{summoner}'
        try:
            response = requests.get(url, headers=headers)
            # response.raise_for_status()
            res = json.loads(response.content.decode('utf-8'))
            puuid_list.append(res['puuid'])
        except:
            continue
        

        # sleep included to not exceed rate limit
        # rate limit = 100 requests every 2 minutes = 1 request per 1.2 seconds max
        # 1.5 seconds to be safe
        time.sleep(1.2)

    return puuid_list

# get list of ranked match IDs from list of puuids
# set maximum number of games to pull from each player
def get_riot_match_ids(puuid_list, api_key, max_players=100, max_games=20):
    match_ids = []
    headers = {
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Charset": "application/x-www-form-urlencoded; charset=UTF-8",
        "Origin": "https://developer.riotgames.com",
        "X-Riot-Token": api_key
    }

    for i,puuid in enumerate(puuid_list):
        # if limit set, return once we hit limit
        if i >= max_players:
            return match_ids
        
        if i % 20 == 0:
            print(f'puuid {i}/{len(puuid_list)}')

        # use americas regional routing value
        url = f'https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?type=ranked&start=0&count={max_games}'

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            res = response.content.decode('utf-8')
            res = "".join(res).replace('"','').replace('[','').replace(']','').split(',')
            # check for empty strings
            res = [x for x in res if len(x) > 0]
            match_ids.extend(res)
        except:
            raise
        
        
        # sleep to bypass request limit
        time.sleep(1.2)

    return list(set(match_ids))


# helper functions to put timeline json into dataframe format
def round_timestamp(x, base=60000):
    return base * round(x/base)

def decode_timeline_json(timeline_json,frame_interval=None):
    events_reframe = defaultdict(list)
    participants_reframe = pd.DataFrame()

    match_id = timeline_json['metadata']['matchId']
    if frame_interval is None:
        frame_interval = timeline_json['info']['frameInterval']

    for frame in timeline_json['info']['frames']:
        timestamp = round_timestamp(frame['timestamp'], base=frame_interval)
        for event in frame['events']:
            event['sampleTimestamp'] = timestamp
            events_reframe[event['type']].append(event)
        for participant_id, participant_frame in frame['participantFrames'].items():
            tmp = pd.json_normalize(participant_frame, sep='_')
            tmp['participantId'] = participant_id
            tmp['sampleTimestamp'] = timestamp
            participants_reframe = pd.concat([participants_reframe, tmp], ignore_index=True)
    
    participants_reframe['matchId'] = match_id
    participants_reframe = participants_reframe.set_index(['matchId','participantId','sampleTimestamp']).reset_index()

    events_reframe_dfs = {}
    for event, event_dict in events_reframe.items():
        events_reframe_dfs[event] = pd.DataFrame.from_dict(event_dict)
        events_reframe_dfs[event]['matchId'] = match_id
        events_reframe_dfs[event] = events_reframe_dfs[event].set_index(['matchId','sampleTimestamp','type']).reset_index()

    participant_mapping_df = pd.DataFrame.from_dict(timeline_json['info']['participants'])
    participant_mapping_df['matchId'] = match_id
    participant_mapping_df = participant_mapping_df.set_index('matchId').reset_index()

    return participants_reframe, events_reframe_dfs, participant_mapping_df



# get jsons from match timelines api
def get_match_timeline_data(match_id_list, api_key):
    participants_reframe_all = pd.DataFrame()
    event_reframe_all = defaultdict(pd.DataFrame)
    participant_mapping_all = pd.DataFrame()
    headers = {
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Charset": "application/x-www-form-urlencoded; charset=UTF-8",
        "Origin": "https://developer.riotgames.com",
        "X-Riot-Token": api_key
    }

    for i,match_id in enumerate(match_id_list):
        if i % 20 == 0:
            print(f'match {i}/{len(match_id_list)}')
        url = f'https://americas.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline'
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            res = json.loads(response.content.decode('utf-8'))
            participants_reframe, events_reframe_dfs, participant_mapping_df = decode_timeline_json(res,frame_interval=10000)
            participants_reframe_all = pd.concat([participants_reframe_all, participants_reframe])   
            for event_type, event_df in events_reframe_dfs.items():
                event_reframe_all[event_type] = pd.concat([event_reframe_all[event_type], event_df])
            participant_mapping_all = pd.concat([participant_mapping_all,participant_mapping_df])
            
        except:
            continue
    
    return participants_reframe_all, event_reframe_all, participant_mapping_all

def write_timeline_data(participants_reframe_all, event_reframe_all, participant_mapping_all, dir):
    participants_reframe_all.to_pickle(f'{dir}participant_stats.pkl')
    for event in event_reframe_all:
        event_reframe_all[event].to_pickle(f'{dir}{event.lower()}.pkl')
    participant_mapping_all.to_pickle(f'{dir}participant_puuid_map.pkl')


# script to download replays from client API
# NOTE: only works if replay can be downloaded manually from inside client
# replay will be available in replay path found in league client
def download_replay(gameId, port, token):
    tok = base64.b64encode(f"riot:{token}".encode("utf-8"))
    tok = str(tok, encoding="utf-8")
    url = f"https://127.0.0.1:{port}/lol-replays/v1/rofls/{gameId}/download"
    print('tok:', tok, gameId)
    req = requests.post(
        url=url,
        headers={
            "Authorization": f"Basic {tok}",
            "Content-Type": "application/json"
        },
        data=json.dumps({
            "gameId": gameId
        }),
        verify=False
    )
    print('downloading:', url, gameId)
    time.sleep(0.25)
    print("CODE:", req.status_code, req.content)
    return req

if __name__ == '__main__':
    f = open('keys.json','r')
    keys = json.load(f)
    f.close()

    api_key = keys['riot_api_key']

    # summoner_list = get_summoner_names(api_key)
    # summoner_list = get_challenger_summoner_names(api_key)
    # summoner_list = []
    # with open('summoner_list.pkl','wb') as f:
    #     pickle.dump(summoner_list, f)

    # with open('summoner_list.pkl','rb') as f:
    #     summoner_list = pickle.load(f)

    # # shuffle summoner list for now, can change later
    # random.shuffle(summoner_list)
    # summoner_list = summoner_list[:600]

    # puuid_list = get_riot_puuid(summoner_list, api_key)
    # with open('summoner_puuid_list.pkl','wb') as f:
    #     pickle.dump(puuid_list, f)

    # with open('summoner_puuid_list.pkl','rb') as f:
    #     puuid_list = pickle.load(f)

    # match_id_list = get_riot_match_ids(puuid_list, api_key, max_players=len(puuid_list), max_games=20)
    # with open('match_id_list.pkl','wb') as f:
    #     pickle.dump(match_id_list, f)

    with open('match_id_list.pkl','rb') as f:
        match_id_list = pickle.load(f)

    # print(len(summoner_list))
    print(len(match_id_list))

    match_id_list = match_id_list[:5200]

    
    participants_reframe_all, event_reframe_all, participant_mapping_all = get_match_timeline_data(match_id_list, api_key)
    # print(event_reframe_all)

    write_timeline_data(participants_reframe_all, event_reframe_all, participant_mapping_all, 'data/')

    # port = keys['app_port']
    # token = keys['remoting_auth_token']

    # test_game_id = match_id_list[0].split('_')[1]
    # download_replay(test_game_id, port, token)


"""
forget about replay scraping for now
use match timeline api
dataframe index -> one aggregate timestamp every x minutes
pivot for all ten champs, objectives, etc.
early, mid, late game, timestamp as feature

"""