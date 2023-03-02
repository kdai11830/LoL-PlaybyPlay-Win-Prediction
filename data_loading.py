import numpy
import pandas
import json
import requests
import time
import base64
import os

# get current list of challenger summoner names
# assume we only want ranked solo 5x5 and not ranked flex
def get_challenger_summoner_names(api_key):
    summoner_list = []
    headers = {
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Charset": "application/x-www-form-urlencoded; charset=UTF-8",
        "Origin": "https://developer.riotgames.com",
        "X-Riot-Token": api_key
    }
    url = 'https://na1.api.riotgames.com/lol/league/v4/challengerleagues/by-queue/RANKED_SOLO_5x5'
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        res = json.loads(response.content.decode('utf-8'))
        for summoner in res['entries']:
            summoner_list.append(summoner['summonerName'])
        return summoner_list
    else:
        print('SOMETHING WENT WRONG')
        print(response.status_code)


# get puuid (global player ID) from summoner names
def get_riot_puuid(summoner_list, api_key):
    puuid_list = []
    headers = {
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Charset": "application/x-www-form-urlencoded; charset=UTF-8",
        "Origin": "https://developer.riotgames.com",
        "X-Riot-Token": api_key
    }


    for summoner in summoner_list:
        # use americas regional routing value
        url = f'https://na1.api.riotgames.com/lol/summoner/v4/summoners/by-name/{summoner}'
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            res = json.loads(response.content.decode('utf-8'))
            puuid_list.append(res['puuid'])
        else:
            print('SOMETHING WENT WRONG')
            print(response.status_code)

        # sleep included to not exceed rate limit
        # rate limit = 100 requests every 2 minutes = 1 request per 1.2 seconds max
        # 1.5 seconds to be safe
        time.sleep(1.5)

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

        # use americas regional routing value
        url = f'https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?type=ranked&start=0&count={max_games}'
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            res = response.content.decode('utf-8')
            res = "".join(res).replace('"','').replace('[','').replace(']','').split(',')
            # check for empty strings
            res = [x for x in res if len(x) > 0]
            match_ids.extend(res)
        else:
            print('SOMETHING WENT WRONG')
            print(response.status_code)

    return match_ids

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

    summoner_list = get_challenger_summoner_names(api_key)
    puuid_list = get_riot_puuid(summoner_list, api_key)
    match_id_list = get_riot_match_ids(puuid_list, api_key, max_games=20)

    print(len(summoner_list))
    print(len(match_id_list))

    port = keys['app_port']
    token = keys['remoting_auth_token']

    test_game_id = match_id_list[0].split('_')[1]
    download_replay(test_game_id, port, token)


