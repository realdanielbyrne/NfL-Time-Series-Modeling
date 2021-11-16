import pandas as pd
import numpy as np
import requests
from datetime import date


baseurl = 'https://www.pro-football-reference.com/teams/{team}/{year}.htm'
teams = ['crd', 'atl', 'rav', 'buf', 'car', 'chi', 'cin', 'cle', 'dal',
         'den', 'det', 'gnb', 'htx', 'clt', 'jax', 'kan', 'rai', 'sdg',
         'ram', 'mia', 'min', 'nwe', 'nor', 'nyj', 'phi', 'pit', 'sfo',
         'sea', 'tam', 'oti', 'was']
team='dal'
timestamp_index=True
start=2017
stop=2022
to_csv=True
year =2021


def get_teams_stats(start=2017, stop=2022, to_csv=True):
    team_stats = {}
    for team in teams:
        stats = get_team_game_stats(team, start, stop, to_csv)
        team_stats[team] = stats

    return team_stats


def get_team_game_stats(team='dal', timestamp_index=True, start=2017, stop=2022, to_csv=True):
    years = np.arange(start, stop)
    stats = pd.DataFrame()

    for year in years:
        url = baseurl.format(team=team, year=str(year))
        html = requests.get(url).content
        df_list = pd.read_html(html)
        df = df_list[1]
        df.columns = ["_".join(a) for a in df.columns.to_flat_index()]

        df.columns = ['Week', 'Day', 'Date', 'Time', 'BoxScore',
                      'W/L', 'OT', 'Record', '@', 'Opponent', 'TeamScore',
                      'OpponentScore', 'Off_1stDn', 'Off_Totyd', 'Off_PassYd',
                      'Off_RushYd', 'Off_TO', 'Def_1stDn', 'Def_Totyd',
                      'Def_PassYd', 'Def_RushYd', 'Def_TO', 'ExpOff',
                      'ExpDef', 'ExpSpTeams']
        df.dropna(subset=['W/L'], inplace=True)    
        df['year'] = str(year)
        df['OT'] = np.where((df['OT'] == 'OT'), 1, 0)
        df['@'] = np.where((df['@'] == '@'), 1, 0)
        df['W/L'] = np.where((df['W/L'] == 'W'), 1, 0)
        
        stats = stats.append(df)
    
    stats['Def_TO'] = stats['Def_TO'].fillna(0)
    stats['Off_TO'] = stats['Off_TO'].fillna(0)
    stats['Team'] = team

    stats[['month', 'day']] = stats.Date.str.split(expand=True)
    stats['Timestamp_str'] = stats['year'].str.cat(stats['month'], sep="/").str.cat(stats['day'], sep="/")
    stats['Timestamp'] = pd.to_datetime(stats['Timestamp_str'], format="%Y/%B/%d")

    if timestamp_index:
        stats.set_index(stats['Timestamp'])
    else:
        stats.index = pd.RangeIndex(len(stats.index))
        stats.index = range(len(stats.index))

    if to_csv:
        stats.to_csv('./' + team + str(start) + '_' + str(stop) + '.csv')

    return stats


if __name__ == "__main__":

    # Get Data
    year = date.today().year
    team_stats = get_teams_stats(year - 3, year + 1)

