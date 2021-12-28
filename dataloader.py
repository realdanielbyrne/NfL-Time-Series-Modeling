#%%
import pandas as pd
import numpy as np
import requests
from datetime import date
import argparse


gamelog_url = 'https://www.pro-football-reference.com/teams/{team}/{year}/gamelog/'
game_results_url = 'https://www.pro-football-reference.com/teams/{team}/{year}.htm'

teams = ['crd', 'atl', 'rav', 'buf', 'car', 'chi', 'cin', 'cle', 'dal',
         'den', 'det', 'gnb', 'htx', 'clt', 'jax', 'kan', 'rai', 'sdg',
         'ram', 'mia', 'min', 'nwe', 'nor', 'nyj', 'phi', 'pit', 'sfo',
         'sea', 'tam', 'oti', 'was']

team='dal'
timestamp_index=True
start=2018
stop=2022
to_csv=True
year = '2021'

#%%
def get_teams_stats(years, to_csv=True, timeindex = False):
    
    team_stats = {}
    for team in teams:
        stats = get_team_game_stats(team, years, to_csv, timeindex)
        team_stats[team] = stats.copy()

    return team_stats

#%%
def win_pct(c):
    splits = c.split('-')
    
    if len(splits) > 2:
        w,l,d = int(splits[0]),int(splits[1]),int(splits[2])
        wpct=w/(w+l+d)
    else:
        w,l = int(splits[0]),int(splits[1])
        wpct=w/(w+l)
    return wpct

#%%
def get_gamelog(team,year):
    url = gamelog_url.format(team=team, year=str(year))
    html = requests.get(url).content
    gl_list = pd.read_html(html)
    gl = gl_list[0].copy()
    gl.columns = ["_".join(a) for a in gl.columns.to_flat_index()]
    gl.columns = ['Week', 'Day', 'Date', 'Boxscore', 'W/L', 'OT', '@','Opp',
                    'TmScore', 'OppScore',
                    'PassCmp', 'PassAtt', 'PassYds','PassTd', 'PassInt',
                    'Sacks', 'SackYds', 'PassY/A', 'PassNY/A', 'Cmp%', 'Qbr',
                    'RushAtt','RushYds','RushY/A','RushTd',
                    'FGM','FGA','XPM','XPA',
                    'Pnt','PntYds',
                    '3DConv','3DAtt','4DConv','4DAtt','ToP']

    gl.dropna(subset=['W/L'], inplace=True)    
    gl['Year'] = year
    gl['OT'] = np.where((gl['OT'] == 'OT'), 1, 0)
    gl['@'] = np.where((gl['@'] == '@'), 1, 0)
    gl['W/L'] = np.where((gl['W/L'] == 'W'), 1, 0)
    gl['Team'] = team
    gl[['month','day']] = gl.Date.str.split(expand=True)
    gl['Timestamp_str'] = gl['Year'].str.cat(gl['month'], sep ="/").str.cat(gl['day'], sep ="/")
    gl['Timestamp'] = pd.to_datetime(gl['Timestamp_str'], format="%Y/%B/%d")
    gl.drop(columns=['Boxscore','Timestamp_str','month','day','Week','Day','Date'],inplace=True)
    gl.index = pd.RangeIndex(len(gl.index))
    
    return gl

#%%
def get_game_results(team,year):
    # scrape data from pro-football reference
    url = game_results_url.format(team=team, year=year)
    html = requests.get(url).content
    gr_list = pd.read_html(html)
    gr = gr_list[1].copy()
    
    # flatten index and rename columns
    gr.columns = ["_".join(a) for a in gr.columns.to_flat_index()]
    gr.columns = ['Week', 'Day', 'Date', 'Time', 'Boxscore',
                    'W/L', 'OT', 'Record', '@', 'Opp', 'TmScore',
                    'OppScore', 'Off_1stDn', 'Off_Totyd', 'Off_PassYd',
                    'Off_RushYd', 'Off_TO', 'Def_1stDn', 'Def_Totyd',
                    'Def_PassYd', 'Def_RushYd', 'Def_TO', 'ExpOff',
                    'ExpDef', 'ExpSpTeams']
    
    # cleanup
    gr = gr.dropna(subset=['W/L'])
    gr['OT'] = np.where((gr['OT'] == 'OT'), 1, 0)
    gr['@'] = np.where((gr['@'] == '@'), 1, 0)
    gr['W/L'] = np.where((gr['W/L'] == 'W'), 1, 0)
    gr['Def_TO'] = gr['Def_TO'].fillna(0)
    gr['Off_TO'] = gr['Off_TO'].fillna(0)
    gr['Team'] = team
    gr['Year'] = year
    gr[['month','day']] = gr.Date.str.split(expand=True)
    gr['Timestamp_str'] = gr['Year'].str.cat(gr['month'], sep ="/").str.cat(gr['day'], sep ="/")
    gr['Timestamp'] = pd.to_datetime(gr['Timestamp_str'], format="%Y/%B/%d")
    gr['month'] = gr['Timestamp'].dt.month
    gr['year'] = gr['Timestamp'].dt.year
    gr['Avg3Game'] = gr['TmScore'].rolling(3).mean()
    gr['Avg6Game'] = gr['TmScore'].rolling(6).mean()
    gr['Win%']=gr['Record'].map(win_pct)
    
    gr.index = pd.RangeIndex(len(gr.index))
    gr.drop(columns=['Boxscore','Timestamp_str','Date','day'],inplace=True)
    return gr

#%%
def get_team_game_stats(team, years, to_csv=True, timeindex=False  ):
    
    stats = pd.DataFrame()
    for year in years:
        gl = get_gamelog(team,year)
        gl = gl[['PassCmp','PassAtt','PassYds','PassTd','PassInt','Sacks',
                 'SackYds','PassY/A','Cmp%','Qbr','RushAtt','RushYds',
                 'RushY/A','RushTd','FGM','FGA','XPM','XPA','Pnt','PntYds',
                 '3DConv','3DAtt','4DConv','4DAtt','ToP','Timestamp']]
        gr = get_game_results(team,year)     

        
        gl = pd.merge(gl,gr,on='Timestamp')
        stats = stats.append(gl)
    
    if timeindex:
        stats.set_index(stats['Timestamp'])
    else:
        stats.index = pd.RangeIndex(len(stats.index))

    if to_csv:
        filename = './stats/combinedstats/{team}_{start}_{stop}.csv'.format(team=team,start=years[0],stop=years[-1])
        print('Saving '+team+' stats to '+ filename)
        stats.to_csv(filename,index=False)

    return stats

#%%
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='NFL spread Predictor.')
    parser.add_argument('-b','--start',
                        action='store',
                        type=int,
                        default=int(date.today().year)-3,
                        required=False,
                        help="The first years stats are pulled from.")

    parser.add_argument('-e','--end',
                        type=int,
                        default=int(date.today().year),
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
        args.start = args.end - 4
    
    years = [str(i) for i in range(args.start,args.end + 1)]
    team_stats = get_teams_stats(years, args.to_csv,args.timeindex)


# %%
