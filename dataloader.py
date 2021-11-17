import pandas as pd
import numpy as np
import requests
from datetime import date
import argparse


baseurl = 'https://www.pro-football-reference.com/teams/{team}/{year}.htm'
teams = ['crd', 'atl', 'rav', 'buf', 'car', 'chi', 'cin', 'cle', 'dal',
         'den', 'det', 'gnb', 'htx', 'clt', 'jax', 'kan', 'rai', 'sdg',
         'ram', 'mia', 'min', 'nwe', 'nor', 'nyj', 'phi', 'pit', 'sfo',
         'sea', 'tam', 'oti', 'was']

# team='dal'
# timestamp_index=True
# start=2018
# stop=2022
# to_csv=True
# year = 2021


def get_teams_stats(years, to_csv=True, timeindex = False):
    
    team_stats = {}
    for team in teams:
        stats = get_team_game_stats(team, years, to_csv, timeindex)
        team_stats[team] = stats.copy()

    return team_stats

def win_pct(c):
    splits = c.split('-')
    
    if len(splits) > 2:
        w,l,d = int(splits[0]),int(splits[1]),int(splits[2])
        wpct=w/(w+l+d)
    else:
        w,l = int(splits[0]),int(splits[1])
        wpct=w/(w+l)
    return wpct

def get_team_game_stats(team, years, to_csv=True, timeindex=False  ):
    
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
        df['year'] = year
        df['OT'] = np.where((df['OT'] == 'OT'), 1, 0)
        df['@'] = np.where((df['@'] == '@'), 1, 0)
        df['W/L'] = np.where((df['W/L'] == 'W'), 1, 0)

        df['win_pct']=df['Record'].map(win_pct)
        stats = stats.append(df)
    
    stats['Def_TO'] = stats['Def_TO'].fillna(0)
    stats['Off_TO'] = stats['Off_TO'].fillna(0)
    stats['Team'] = team

    stats[['month', 'day']] = stats.Date.str.split(expand=True)
    stats['Timestamp_str'] = stats['year'].str.cat(stats['month'], sep="/").str.cat(stats['day'], sep="/")
    stats['Timestamp'] = pd.to_datetime(stats['Timestamp_str'], format="%Y/%B/%d")
    
    # drop columns
    stats.drop(columns=['BoxScore','Timestamp_str','month','day','Date'],inplace=True)

    if timeindex:
        stats.set_index(stats['Timestamp'])
    else:
        stats.index = pd.RangeIndex(len(stats.index))
        stats.index = range(len(stats.index))

    if to_csv:
        filename = './stats/{team}_{start}_{stop}.csv'.format(team=team,start=years[0],stop=years[-1])
        print('Saving '+team+' stats to '+ filename)
        stats.to_csv(filename,index=False)

    return stats


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='NFL spread Predictor.')
    parser.add_argument('-b','--start',
                        action='store',
                        type=int,
                        default="2017",
                        required=False,
                        help="The first years stats are pulled from.")

    parser.add_argument('-e','--end',
                        type=int,
                        required=False,
                        help="The last year stats are pulled.")


    parser.add_argument('--timeindex',
                        default=False,
                        type=bool,
                        required=False,
                        help="The target stat to predict.")

    parser.add_argument('-s','--to_csv',
                        default=True,
                        type=bool,
                        required=False,
                        help="The target stat to predict.")

    args = parser.parse_args("")


    # Get Data
    if args.end is None:
        args.end = int(date.today().year)
    if args.start is None:
        args.start = args.end - 5
    
    years = [str(i) for i in range(args.start,args.end + 1)]
    team_stats = get_teams_stats(years, args.to_csv,args.timeindex)

