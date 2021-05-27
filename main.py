#**********************************************STEP 1: IMPORTATION OF PYTHON LIBRARIES##################################################################
# Step 1: Importation of Python libraries

# libraries for data gathering, analysis and visualisation:
import pandas as pd
import requests
import json
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import re

# libraries for machine learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve,roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier, BaggingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# option to supress warnings
#import warnings
#warnings.filterwarnings("ignore")

#**********************************************SETP 2: ACCESS THE DATA##################################################################
# This Step is commented-out in its entirety using triple-quoted strings
'''# Step 2: Access the data

df_list = [] # initialise a list to store the dataframes that will be built up during this step

# First API Call: obtain majority of FPL data for this season (present season is 2020/21)
master_link = 'https://fantasy.premierleague.com/api/bootstrap-static/' #endpoint where majority of data of interest is stored on the FPL API
request = requests.get(master_link)  #use requests to fetch the content from the FPL API endpoint of interest
print(request.status_code) # code return of 200 implies success
master_data = request.json() #use json to obtain a a json object of the result from the FPL API

# Create (for this season) a team information dataframe (via the key "team" data obtainable from the return of First API Call)
team_info_2021_df = pd.DataFrame.from_dict(master_data['teams']) #create a dataframe of team information from the 'teams' section of the data returned from the FPL API
team_info_2021_df['season']='2020/21' #add column noting the season
team_info_2021_df.df_name = 'team_info_2021_df' #give the dataframe a name for easier management going forward
df_list.append(team_info_2021_df) #append it to the list of dataframes as initialised

# Second API Call: obtain FPL fixture data for this season and convert to pandas dataframe. 
fixtures_link = 'https://fantasy.premierleague.com/api/fixtures/' #endpoint where fixture data of interest is stored on the FPL API
request = requests.get(fixtures_link) #use requests to fetch the content from the FPL API endpoint of interest
print(request.status_code) #code return of 200 implies success
fixtures_data = request.json() #use json to obtain a a json object of the result from the FPL API
fixtures_2021 = pd.DataFrame.from_dict(fixtures_data) #convert the json object/dictionary as returned to a dataframe
fixtures_2021['season']='2020/21' add column noting the season
fixtures_2021.df_name = 'fixtures_2021' give the dataframe a name for easier management going forward
df_list.append(fixtures_2021) append the dataframe to the list of dataframes as initialised


# Obtain prior data for 2019-20 and 2018-19

# 2019-20 team info
team_info_1920_df = pd.read_csv('https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2019-20/teams.csv') # use pandas read_csv to access csv data hosted on github
team_info_1920_df['season']='2019/20' # add column noting the season
team_info_1920_df.df_name = 'team_info_1920_df' # give the dataframe a name for easier management going forward
df_list.append(team_info_1920_df) # append it to the list of dataframes as initialised
# 2019-20 fixtures info
fixtures_1920 = pd.read_csv('https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2019-20/fixtures.csv') # use pandas read_csv to access csv data hosted on github
fixtures_1920['season']='2019/20' # add column noting the season
fixtures_1920.df_name = 'fixtures_1920' # give the dataframe a name for easier management going forward
df_list.append(fixtures_1920) # append it to the list of dataframes as initialised
# 2018-19 team info
info_1819 = requests.get('https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2018-19/raw.json') #use requests to access data hosted on github
team_info_1819 = info_1819.json() #use json to gather the data hosted on github
team_info_1819_df = pd.DataFrame(team_info_1819['teams']) 
team_info_1819_df['season']='2018/19'
team_info_1819_df.df_name='team_info_1819_df'
df_list.append(team_info_1819_df)
# 2018-19 fixtures info
fixtures_1819=pd.read_csv('https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2018-19/fixtures.csv')
fixtures_1819['season']='2018/19'
fixtures_1819.df_name='fixtures_1819'
df_list.append(fixtures_1819)'''

'''# to convert the above to csv files and save to local directory:

names = [df.df_name for df in df_list]
for df in df_list:
    df.to_csv(df.df_name + '.csv')
'''

'''# to load in the saved csv files

for df in df_list:
    df = pd.read_csv(df.df_name + '.csv')'''
'''
#Can otherwise be undertaken with:
team_info_2021_df=pd.read_csv('team_info_2021_df.csv')
fixtures_2021=pd.read_csv('fixtures_2021.csv')
team_info_1920_df=pd.read_csv('team_info_1920_df.csv')
fixtures_1920=pd.read_csv('fixtures_1920.csv')
team_info_1819_df=pd.read_csv('team_info_1819_df.csv')
fixtures_1819=pd.read_csv('fixtures_1819.csv')
'''

#**********************************************STEP 3: EXPLOTATORY DATA ANALYSIS (EDA)##################################################################
# Step 3 - Exploratory Data Analysis (EDA)

# This Step is commented-out until EDA Round 2 using triple-quoted strings

#EDA Round 1
'''# initial analysis with .head() and .tail()
for df in df_list:
    print('Looking at .head()') # provides snapshot of first 5 rows of dataframe
    print(df.df_name,df.head())
    print('*'*50)
    print('Looking at .tail()') # provides snapshot of last 5 rows of dataframe
    print(df.df_name, df.tail())
    print('*'*50)'''

# further analysis on dataframes
'''for df in df_list:
    print('Looking at .describe()') # provides summary statistics
    print(df.describe())
    print('*'*50)
    print('Looking at .info()') # provides information regarding dataframe (whether float, integer, bool or string) and also the number of non-null values
    print(df.info())
    print('*'*50)
    print('Looking at .shape') # provides dimensions of dataframe (rows x columns)
    print(df.shape)
    print('*'*50)
    print('Looking at .columns') # provides list of the columns of each dataframe
    print(df.columns)
    print('*'*50)

#Data manipulation Round 1
# Here we want to combine the information of the fixtures and team information for each season

# first, we update the columns in each dataframe for consistency
# A number of redundant columns are dropped at this point, mainly relating to the time of the matches/team submission deadlines, length in minutes etc.
team_skill_2021_df = team_info_2021_df[['id', 'name','strength_overall_home','strength_overall_away',
       'strength_attack_home', 'strength_attack_away', 'strength_defence_home',
       'strength_defence_away','season']]
fixtures_2021 = fixtures_2021[['code','event','finished','id', 'started', 'team_h', 'team_h_score', 'team_h_difficulty', 'team_a', 'team_a_score','team_a_difficulty','stats','season']]

fixtures_1819=fixtures_1819[['code', 'event', 'finished', 'id', 'started',
       'team_a', 'team_a_score', 'team_h', 'team_h_score',
       'team_h_difficulty', 'team_a_difficulty','stats','season']]

team_skill_1819_df = team_info_1819_df[['id', 'name','strength_overall_home','strength_overall_away',
       'strength_attack_home', 'strength_attack_away', 'strength_defence_home',
       'strength_defence_away','season']]

fixtures_1920=fixtures_1920[['code', 'event', 'finished', 'id', 'started',
       'team_a', 'team_a_score', 'team_h', 'team_h_score', 
       'team_h_difficulty', 'team_a_difficulty','stats','season']]

team_skill_1920_df = team_info_1920_df[['id', 'name','strength_overall_home','strength_overall_away',
       'strength_attack_home', 'strength_attack_away', 'strength_defence_home',
       'strength_defence_away','season']]

# Second, we prepare create team home and away skill dataframes for eventual mapping to the fixtures
home_skill_2021_df = team_skill_2021_df[['id', 'name','strength_overall_home','strength_attack_home', 'strength_defence_home',]]
home_skill_2021_df = home_skill_2021_df.rename(columns = {'id':'team_h','name':'h_team_name'})
away_skill_2021_df = team_skill_2021_df[['id', 'name','strength_overall_away','strength_attack_away', 'strength_defence_away',]]
away_skill_2021_df = away_skill_2021_df.rename(columns = {'id':'team_a','name':'a_team_name'})

home_skill_1920_df = team_skill_1920_df[['id', 'name','strength_overall_home','strength_attack_home', 'strength_defence_home',]]
home_skill_1920_df = home_skill_1920_df.rename(columns = {'id':'team_h','name':'h_team_name'})
away_skill_1920_df = team_skill_1920_df[['id', 'name','strength_overall_away','strength_attack_away', 'strength_defence_away',]]
away_skill_1920_df = away_skill_1920_df.rename(columns = {'id':'team_a','name':'a_team_name'})

home_skill_1819_df = team_skill_1819_df[['id', 'name','strength_overall_home','strength_attack_home', 'strength_defence_home',]]
home_skill_1819_df = home_skill_1819_df.rename(columns = {'id':'team_h','name':'h_team_name'})
away_skill_1819_df = team_skill_1819_df[['id', 'name','strength_overall_away','strength_attack_away', 'strength_defence_away',]]
away_skill_1819_df = away_skill_1819_df.rename(columns = {'id':'team_a','name':'a_team_name'})

# Third, we merge the fixtures dataframe for each season the the home skill and away skill dataframes
fixtures_2021 = fixtures_2021.merge(home_skill_2021_df, on = 'team_h').merge(away_skill_2021_df, on = 'team_a')
fixtures_1920 = fixtures_1920.merge(home_skill_1920_df, on = 'team_h').merge(away_skill_1920_df, on = 'team_a')
fixtures_1819 = fixtures_1819.merge(home_skill_1819_df, on = 'team_h').merge(away_skill_1819_df, on = 'team_a')

# Fourth, we concatenated the three years of fixtures to make one dataframe with all fixtures and their associated skill attributes
fixtures_3_years = pd.concat([fixtures_2021,fixtures_1920,fixtures_1819])

# Fifth, we create a column 'result' using numpy select that will eventually represent our target variable

conditions= [fixtures_3_years.team_h_score > fixtures_3_years.team_a_score,
fixtures_3_years.team_h_score == fixtures_3_years.team_a_score,
fixtures_3_years.team_h_score < fixtures_3_years.team_a_score,
fixtures_3_years.finished == False] # define the conditions
values = ['H','D','A','match pending'] # set out the values where the conditions are met
fixtures_3_years['result'] = np.select(conditions,values)

# Now we create further feature columns for eventual analysis and exploration
# add in covid as a feature (this will indicate matches that were played since COVID-19 was declared a pandemic).
# covid is considered important to model for as it has resulted in matches being played without fans in attendance (or the number of fans being significantly reduced)
fixtures_3_years['covid']= np.where((fixtures_3_years['season']== '2020/21') | (fixtures_3_years['event'] >38), 1, 0)

# We add further features to illustrate the difference in attributes between teams playing against eachother in a fixture
fixtures_3_years['diffrank'] = fixtures_3_years.team_h_difficulty - fixtures_3_years.team_a_difficulty
fixtures_3_years['diffstrength_overall'] = fixtures_3_years.strength_overall_home - fixtures_3_years.strength_overall_away
fixtures_3_years['diff_def_v_att'] = fixtures_3_years.strength_defence_home - fixtures_3_years.strength_attack_away
fixtures_3_years['diff_att_v_def'] = fixtures_3_years.strength_attack_home - fixtures_3_years.strength_defence_away
fixtures_3_years.reset_index(inplace = True)

# Sixth, save the dataframe "fixtures_3_years" as a csv for future use
fixtures_3_years.to_csv('fixtures_3_years.csv') # data from the FPL API saved on 20 May 2021
'''

# EDA Round 2
# Exploring our combined dataframe:
# First, read in the saved data (csv file) here:
fixtures_3_years = pd.read_csv('fixtures_3_years.csv') # reading in the saved FPL data from 20 May 2021 previously saved as csv

# Second, undertake EDA on data
#fixtures_3_years.head()
#fixtures_3_years.info()
#fixtures_3_years.describe()
#fixtures_3_years.shape

#Third, deal with missing values
# From EDA Round 2, we note there are a number of missing values
#print(fixtures_3_years.isnull().sum())

# As the missing value relate to where the fixture score values are missing, it is likely that this relates to where the dataframe includes matches that have not yet been played
#print(fixtures_3_years.loc[fixtures_3_years['result'] == 'match pending', 'code'].isnull().count())

# Create heatmap of missing values
sns.heatmap(fixtures_3_years.isnull(),yticklabels=False,cbar=False,cmap='viridis').set(ylabel ="Null values denoted in yellow", xlabel = "Feature",title="Check for null values (1)")
plt.savefig("check_for_nulls_1.png",bbox_inches="tight")
#plt.show()
plt.close()

# It is best to drop the rows containing null values in this instance as we cannot admit unplayed matches to our analysis. This can be done in a number of ways:
#fixtures_3_years = fixtures_3_years[fixtures_3_years.result!='match pending']
fixtures_3_years=fixtures_3_years.dropna()

# Create another heatmap to confirm that there are no missing values
sns.heatmap(fixtures_3_years.isnull(),yticklabels=False,cbar=False,cmap='viridis').set(ylabel ="Null values denoted in yellow", xlabel = "Feature",title="Check for null values (2)")
plt.savefig("check_for_nulls_2.png",bbox_inches="tight")
#plt.show()
plt.close()

# Fourth, check for duplicates
duplicates = fixtures_3_years[fixtures_3_years.drop('stats',axis=1).duplicated()]
#print(duplicates) # as an empty dataframe is returned, no duplicates are observed

# Visualizations

# Visualizing the data

# Countplot of fixtures that ended as Home win, Draw, or Away win across each season
sns.set_style('whitegrid')
sns.countplot(x='season',hue = 'result',data=fixtures_3_years,palette='RdBu_r').set(title="Countplot of results by season")
plt.savefig("countplot_of_results_by_season.png",bbox_inches="tight")
#plt.show()
plt.close()

# Chart - stacked barchart showing percentage of fixtures that ended as Home win, Draw, or Away win across each season
counter = fixtures_3_years.groupby('season')['result'].value_counts().unstack()
percentage_dist = 100 * counter.divide(counter.sum(axis = 1), axis = 0)
ax = percentage_dist.plot.bar(stacked=True)
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    ax.text(x+width/2,
            y+height/2,
            '{:.3f} %'.format(height),
            horizontalalignment='center',
            verticalalignment='center')
plt.title('Stacked barchart of results by season')
plt.xlabel('Season')
plt.ylabel('%')
plt.savefig("stacked_barchart_of_results_by_season.png",bbox_inches="tight")
#plt.show()
plt.close()

# Chart - stacked barchart showing percentage of fixtures that ended as Home win, Draw, or Away win over the past 3 seasons
counter = fixtures_3_years.groupby('season')['result'].value_counts().unstack()
percentage_dist = 100 * counter.divide(counter.sum(axis = 1), axis = 0)
ax = percentage_dist.plot.bar(stacked=True)
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    ax.text(x+width/2,
            y+height/2,
            '{:.3f} %'.format(height),
            horizontalalignment='center',
            verticalalignment='center')
plt.title('Stacked barchart of results by season')
plt.xlabel('Season')
plt.ylabel('%')
plt.savefig("stacked_barchart_of_results_by_season.png",bbox_inches="tight")
#plt.show()
plt.close()

# Has covid-19 and empty stadia impacted the distribution of results?
# charting showing split of results in covid and not covid times
counter = fixtures_3_years.groupby('covid')['result'].value_counts().unstack()
percentage_dist = 100 * counter.divide(counter.sum(axis = 1), axis = 0)
ax = percentage_dist.plot.bar(stacked=True)
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    ax.text(x+width/2,
            y+height/2,
            '{:.3f} %'.format(height),
            horizontalalignment='center',
            verticalalignment='center')
plt.title('Has covid-19 impacted the distribution of results?')
plt.ylabel('%')
plt.xlabel('')
plt.xticks(ticks = [0,1],labels=['Pre-pandemic','During pandemic'])
plt.savefig('Illustrating_impact_of_covid_on_results.png',bbox_inches = 'tight')
#plt.show()
plt.close()

# Is FPL's FDR a good indicator of result?
# Countplot showing over the past three seasons the distribution of results over differences in team ranking
sns.countplot(x='diffrank',hue='result',data=fixtures_3_years, palette='RdBu_r')
plt.title("FPL's FDR data is a good indicator: \n The stronger the team, the more likely a victory")
plt.xlabel('Difference in FDR (Home team FDR minus Away team FDR)')
plt.savefig('Illustrating_diffrank_impact_on_fixtures.png',bbox_inches = 'tight')
#plt.show()
plt.close()

# Is FPL's overall strength rating a good indicator of result?
# Stripplot showing how results are distributed on the basis of differences in overall strength
fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
sns.stripplot(x=fixtures_3_years['diffstrength_overall'],y=fixtures_3_years['result'],jitter=0.25, size=8, ax=ax, linewidth=.5)
plt.title("FPL's overall strength ratings are good indicator's: \n The stronger the team, the more likely a victory")
plt.xlabel("Difference in FPL's rating for overall strength (Home team rating minus Away team rating)")
plt.savefig('Illustrating_diffstrengthoverall_impact_on_fixtures.png',bbox_inches = 'tight')
#plt.show()
plt.close()

# EPL table
def get_gw_table(gw=38):
    master_link = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    request = requests.get(master_link)
    master_data = request.json()

    team_info_df = pd.DataFrame.from_dict(master_data['teams'])
    team_id_df = team_info_df[['id', 'name']]
    fixtures_link = 'https://fantasy.premierleague.com/api/fixtures/'
    request = requests.get(fixtures_link)
    fixtures_data = request.json()
        # initialize PL table
    table_df = pd.DataFrame(columns=['Position','Played','Win','Draw','Loss','GF','GA','GD','Points'])
    table_df = pd.DataFrame.join(team_id_df,table_df)
    table_df = table_df[['Position','name','id','Played','Win','Draw','Loss','GF','GA','GD','Points']]
    team_ids = table_df.id.values
    team_info = table_df.drop(['id'],axis=1).values.tolist()
    table_dict = {key: value for key, value in zip(team_ids, team_info)}

    fixtures_df = pd.DataFrame.from_dict(fixtures_data)
    fixtures_df = fixtures_df[['finished','event','id', 'team_h', 'team_h_score', 'team_a', 'team_a_score']]
    fixtures_df = fixtures_df[fixtures_df.finished == True]
    games_played = [0 for i in range(21)]
    home_games_won = [0 for i in range(21)]
    home_games_lost=[0 for i in range(21)]
    games_drawn =[0 for i in range(21)]
    away_games_lost = [0 for i in range(21)]
    away_games_won = [0 for i in range(21)]
    home_goals_scored= [0 for i in range(21)]
    home_goals_conceded= [0 for i in range(21)]
    away_goals_scored= [0 for i in range(21)]
    away_goals_conceded= [0 for i in range(21)]
    df_bygw = fixtures_df['event']<=gw
    filtered_df = fixtures_df[df_bygw]
    for index,row in filtered_df.iterrows():
        #to get number of games
        games_played[row['team_h']]+=1
        games_played[row['team_a']]+= 1
        # to get games wond
        if row['team_h_score'] > row['team_a_score']:
            home_games_won[row['team_h']]+= 1
            away_games_lost[row['team_a']]+=1
        elif row['team_h_score'] < row['team_a_score']:
            home_games_lost[row['team_h']]+= 1
            away_games_won[row['team_a']]+=1
        else:
            games_drawn[row['team_h']]+= 1
            games_drawn[row['team_a']] += 1
        home_goals_scored[row['team_h']]+=row['team_h_score']
        home_goals_conceded[row['team_h']] += row['team_a_score']
        away_goals_scored[row['team_a']]+=row['team_a_score']
        away_goals_conceded[row['team_a']] += row['team_h_score']
    table_df.Played = games_played[1:]
    table_df.Win = [home_games_won[i] + away_games_won[i] for i in range(1,len(games_played))]
    table_df.Draw = games_drawn[1:]
    table_df.Loss = [home_games_lost[i] + away_games_lost[i] for i in range(1,len(games_played))]
    table_df.GF = [int(home_goals_scored[i] + away_goals_scored[i]) for i in range(1,len(games_played))]
    table_df.GA = [int(home_goals_conceded[i] + away_goals_conceded[i]) for i in range(1,len(games_played))]
    table_df.GD = table_df.GF - table_df.GA
    table_df.Points = table_df.Draw + 3*(table_df.Win.values)
    table_df.sort_values(by=["Points",'GD','GF'],ascending=False,inplace=True)
    table_df.Position = [i for i in range(1,21)]
    table_df.drop(columns=['id'],inplace = True)
    table_df.rename(columns = {'name':'Team'}, inplace = True)
    table_df.set_index('Position', inplace=True)
    return table_df
# get_gw_table() #to call function

def get_gw_table_movements(gw=38):
    gw_played = [i for i in range(1, gw + 1)]
    dataframe_collection = {}

    for gw in gw_played:
        new_data = get_gw_table(gw)
        dataframe_collection[gw] = pd.DataFrame(new_data,
                                                columns=['Team', 'Played', 'Win', 'Draw', 'Loss', 'GF', 'GA', 'GD',
                                                         'Points'])
        dataframe_collection[gw]['gw'] = int(gw)

    plt.style.use('Solarize_Light2')
    order_df = pd.concat(dataframe_collection).drop(['Played', 'Win', 'Draw', 'Loss', 'GF', 'GA', 'GD', 'Points'],
                                                    axis=1).reset_index()

    fig, ax1 = plt.subplots()

    order_df.groupby(['gw', 'Team']).sum().unstack().plot(ax=ax1, secondary_y=False, kind='line', y='Position',
                                                          marker=9,
                                                          color=['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w', '0.8'])
    plt.yticks(ticks=[i for i in range(1, 21)])
    plt.gca().invert_yaxis()
    order_df.groupby(['gw', 'Team']).sum().unstack().plot(secondary_y=True, ax=ax1, kind='line', y='Position', marker=9,
                                                          color=['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w', '0.8'])
    plt.yticks(ticks=[i for i in range(1, 21)], labels=[team for team in order_df.Team.values[-20:]])
    plt.gca().invert_yaxis()
    ax1.get_legend().remove()
    plt.title('2020-21 Premier League positions by week')

    plt.show()
    fig.savefig('pl.png')
    plt.close()
# get_gw_table_movements() #to call function


# Regex

# creating a function that calls FPL API and gathers information regarding player injuries using Regex
# This returns (i) number of presently injured players (ii) pie charts showing teams affected and locations of injuries (iii) dataframe in order of player popularity of injuried players
def get_fpl_injuries():
    # Part 1 - calls to FPL API
    master_link = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    request = requests.get(master_link)
    master_data = request.json()
    player_data = master_data['elements']
    # Part 2 - gather team information
    team_id_list = []
    team_name_list = []
    team_info = master_data['teams']
    team_id_dict = {}
    for team in team_info:
        team_id_dict[team['id']]=team['name']
    # Part 3 - gather player information
    full_name_list = []
    news_list = []
    team_list = []
    tsb_list = []
    for i in player_data:
        full_name = i['first_name']+' '+i['second_name']
        full_name_list.append(full_name)
        news_list.append(i['news'])
        tsb_list.append(i['selected_by_percent'])
        team_list.append(i['team'])
    # Part 4 - create dictionary and dataframe
    player_news_dict = dict((z[0],list(z[1:])) for z in zip(full_name_list,team_list,tsb_list,news_list))
    news_df = pd.DataFrame.from_dict(player_news_dict, orient='index',columns=['team','tsb','news'])
    news_df.loc[news_df['news'] == '','news'] = np.NaN
    news_df.team = news_df.team.map(team_id_dict)
    # Part 5 - print statements
    print("The number of FPL players as of today is",news_df.shape[0])
    print("The number of injured FPL players today is",int(news_df.news.str.count('injur',flags=re.IGNORECASE).sum()))
    # Part 6 - create charts with matplotlib
    injured_players_df = news_df[news_df['news'].str.contains('[Ii]njur')==True]
    q = news_df[news_df['news'].str.contains('injur',flags=re.IGNORECASE)==True]
    q['injury_location']=q['news'].str.split(' ').str[0]
    labels = q.team.unique()
    injuries_by_team = q.team.value_counts()
    injuries_by_team.plot.pie(title='FPL player injuries by team today',autopct='%1.1f%%',ylabel='')
    plt.show()
    plt.savefig('injuries_by_team_today.png')
    plt.close()
    injuries_by_location = q.injury_location.value_counts()
    injuries_by_location.plot.pie(title='FPL player injuries by location today',autopct='%1.1f%%',ylabel='')
    plt.show()
    plt.savefig('injuries_by_location_today.png')
    plt.close()
    # Part 7 - create dataframe detailing injured FPL players
    injury_df = news_df[news_df['news'].str.contains('injur',flags=re.IGNORECASE)==True].sort_values(by=['tsb'],ascending = False)
    return injury_df
    
#return_injury_df = get_fpl_injuries() #to call function
#print(return_injury_df) #to print resulting dataframe noting injured players



#************************************************************************STEP 4: MACHINE LEARNING*************************************************************

# Step 4: Machine Learning

# we convert the "problem" a binary classification one and make the result binary with two outcomes: 1. Away win or 2. not Away win
fixtures_3_years['result_binary'] = fixtures_3_years.result.map({'H':0,'D':0,'A':1})

# we create a new barchart to consider how fixtures are distributed on this binary basis by season
counter = fixtures_3_years.groupby('season')['result_binary'].value_counts().unstack()
percentage_dist = 100 * counter.divide(counter.sum(axis = 1), axis = 0)
ax = percentage_dist.plot.bar(stacked=True)
ax.legend(['not Away win','Away win'])
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    ax.text(x+width/2,
            y+height/2,
            '{:.3f} %'.format(height),
            horizontalalignment='center',
            verticalalignment='center')
plt.title('Stacked barchart of results (either Away win or not) by season')
plt.xlabel('Season')
plt.ylabel('%')
#plt.savefig("stacked_barchart_of_results_by_season_awaywinornot.png",bbox_inches="tight")
#plt.show()
plt.close()

# we create a new piechart to consider how fixtures were distributed on this binary basis by season over past 3 seasons
labels = fixtures_3_years.result_binary.unique()
results = fixtures_3_years.result_binary.value_counts()
results.plot.pie(title='Allocation of Away wins v non-Away wins in past 3 EPL seasons',autopct='%1.1f%%',ylabel='', labels = ['not Away win','Away win'])
plt.legend()
#plt.show()
#plt.savefig('pie_chart_binary_result.png')
plt.close()

# We define features and labels for machine learning

# we create an initial list of features
list_of_features = ['team_h_difficulty',
       'team_a_difficulty',
       'strength_overall_home', 'strength_attack_home',
       'strength_defence_home', 'strength_overall_away',
       'strength_attack_away', 'strength_defence_away', 'covid',
       'diffrank', 'diffstrength_overall', 'diff_def_v_att', 'diff_att_v_def','covid']

# list_of_features_filtered = ['diffrank','diffstrength_overall', 'diff_def_v_att', 'diff_att_v_def'] #filtered list eventually evaluating classifiers on primary features

# pass the list of features to note them as the feature rows
X_all = fixtures_3_years[list_of_features]
#print(X_all.info())

# create the labels
y_all = fixtures_3_years.result_binary

# Heatmap showing correlation between features and labels
train_data=pd.concat([X_all,y_all],axis=1)
colormap = plt.cm.RdBu
plt.figure(figsize=(21,18))
plt.title('Pearson Correlation of Features and Target', y=1.05, size=15)
sns.heatmap(train_data.astype(float).corr(),linewidths=0.1,vmax=1.0,
            square=True, cmap=colormap, linecolor='white', annot=True)
#plt.savefig('pearson_correlation_heatmap.png',bbox_inches="tight")
#plt.show()
plt.close()

#and Heatmap ranking the top features
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features with Target - top features', y=1.05, size=15)
k = 11 # number of variables for heatmap
cols = abs(train_data.astype(float).corr()).nlargest(k, 'result_binary')['result_binary'].index
cm = np.corrcoef(train_data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 11}, yticklabels=cols.values, xticklabels=cols.values)
#plt.savefig('pearson_correlation_heatmap_top_features.png',bbox_inches="tight")
#plt.show()
plt.close()



# split training and test data (we stratify given that the proportion of Away-win and not are unequal)
X_train, X_test,y_train,y_test = train_test_split(X_all, y_all, test_size=.8,stratify=y_all,random_state=4)

# we create a pre-defined list of the classifiers that we wish to fit to the data
classifiers = [('Logistic Regression', LogisticRegression()), ('K Nearest Neighbours', KNeighborsClassifier()), ('Decision Tree Classifier', DecisionTreeClassifier()), ('Random Forest',  RandomForestClassifier()), ('SVC',SVC())]

# Ceate a function that takes in a pre-defined list of classifiers that fits and predicts.
# Function to return Confusion Matrix, Classification report and accuracy scores for each classifier
# A pipeline is used to pre-scale the data
def fit_predict_classifiers(classifiers):
    fit_predict_results_dict = {}
    # Iterate over the pre-defined list of classifiers
    for clf_name, clf in classifiers:
        # we scale the data with StandardScaler
        steps = [('scaler', StandardScaler()),
        (clf_name, clf)]
        pipeline = Pipeline(steps)
        # Fit the pipeline to the training set: scaled
        clf_scaled = pipeline.fit(X_train,y_train)

        # Predict y_pred
        y_pred = clf_scaled.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Evaluate clf's accuracy on the test set
        print("*" * 25)
        print(f"Overall performance of the {clf_name} classifier:")
        print('Accuracy: {:.2f}'.format(accuracy))
        fit_predict_results_dict[clf_name]=accuracy
        # obtain clf's confusion matrix
        print("Confusion matrix: \n",confusion_matrix(y_test, y_pred))

        # obtain clf's classification report
        print("Classification report: \n",classification_report(y_test, y_pred))
        print("*" * 25)
        fit_predict_results_df = pd.DataFrame.from_dict(fit_predict_results_dict,orient='index',columns=['Accuracy']).sort_values(by='Accuracy', ascending=False)
    return fit_predict_results_df


# Call the function fit_predict_classifiers() passing in the list of pre-defined classifiers
# Label the resulting dataframe and print it to observe the accuracy from each classifier

#print("*" * 25)
#fit_predict_results_df = fit_predict_classifiers(classifiers)
#print("*" * 25)
#print('Ordered accuracy results of classifiers:')
#print(fit_predict_results_df)
#print("*" * 25)

# Undertaking k-fold cross validation
# Create a function that takes in the pre-defined list of classifiers that provides cross validation of the results observed eariler
# A pipeline is used to pre-scale the data
def get_kfold_scores_with_scaling(classifiers,k):
    k_fold_scoring_dict = {}
    for clf_name,classifier in classifiers:
        pipe = Pipeline([('scalar',StandardScaler()),(clf_name,classifier)])
        k_fold_score = cross_val_score(estimator=pipe,X=X_all,y=y_all,cv=k)
        k_fold_scoring_dict[clf_name]=k_fold_score,k_fold_score.mean(), k_fold_score.std()
    k_fold_with_scaling_scoring_df = pd.DataFrame.from_dict(k_fold_scoring_dict,orient='index',columns = ['scores','mean score','score standard deviation']).sort_values(by='mean score',ascending=False)
    k_fold_with_scaling_scoring_df = k_fold_with_scaling_scoring_df[['mean score','score standard deviation','scores']]
    return k_fold_with_scaling_scoring_df
    
# Call the function with k = 10 fold cross validation
k = 10
#k_fold_with_scaling_scoring_df = get_kfold_scores_with_scaling(classifiers,k)
#print("*" * 25)
#print(f'Ordered accuracy results of classifiers with k-fold ({k}) cross validation:')
#print(k_fold_with_scaling_scoring_df)
#print("*" * 25)


# Plot ROC curves and determine AUC
def get_roc_curves(classifiers):
    fig = plt.figure(figsize=(8,4))
    # ROC curve and AUC
    fpr_list = []
    tpr_list = []
    clf_list = []
    y_pred_prob_list = []
    for clf_name, classifier in classifiers:

        if clf_name == 'SVC':
            ROC_SVC = SVC(probability=True).fit(X_train,y_train) # ROC can only be plotted on SVC with probability = True. In our fit of SVC no delta in accuracy was observed when prbability was changed fro False to True
            y_pred = ROC_SVC.predict(X_test)
            y_pred_prob = ROC_SVC.predict_proba(X_test)[:,1]
            fpr,tpr,thresholds = roc_curve(y_test,y_pred_prob)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            clf_list.append(clf_name)
            y_pred_prob_list.append(y_pred_prob)
        else:
            y_pred_prob = classifier.predict_proba(X_test)[:,1]
            fpr,tpr,thresholds = roc_curve(y_test,y_pred_prob)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            clf_list.append(clf_name)
            y_pred_prob_list.append(y_pred_prob)
    for i in range(len(fpr_list)):
        plt.plot(fpr_list[i], tpr_list[i],label = clf_list[i])
        #title = str({clf_list[i]}+"AUC:"+{roc_auc_score(y_test, y_pred_prob)})
    plt.legend()
    plt.title('Plotting ROC of each classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.show()
    #plt.savefig('ROC curves of classifiers.png',bbox_inches="tight")
    plt.close()
    print('*' * 25)
    print("AUC summary of classifiers:")
    for i in range(len(fpr_list)):
        print(f"AUC of {clf_list[i]}: ",format(roc_auc_score(y_test, y_pred_prob_list[i])))
    print('*' * 25)

#get_roc_curves(classifiers) #to call the function


# Hyperparameter tuning

# detail the classifiers to hypertune
classifiers_for_hypertuning = [('Logistic Regression', LogisticRegression()), ('K Nearest Neighbours', KNeighborsClassifier()), ('SVC', SVC())]
# set out the hyperparameters for each classifier
params_for_hypertuning = ({"C":np.logspace(-4,4,50),"penalty":['l1','l2','elasticnet'],"solver":['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],"class_weight":[None,'balanced']},{ 'n_neighbors' : [1,2,3,5,7,9],'weights' : ['uniform','distance'],'metric' : ['minkowski','euclidean','manhattan'], 'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']},{'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.5, 0.2, 0.1, 0.05, 0.01, 0.001, 0.0001],'kernel': ['linear', 'poly', 'rbf', 'sigmoid']})

def hyperparam_tuning(classifiers_for_hypertuning,params_for_hypertuning):
    hyperparam_tuning_dict = {}
    temp = list(zip(classifiers_for_hypertuning,params_for_hypertuning))
    for a,b in temp:
        pipe = Pipeline([('scaler',StandardScaler()),(a[0],a[1])])
        pipeline_params = {str(a[0]+'__'+k):v for k,v in b.items()}
        grid = GridSearchCV(pipe,param_grid=pipeline_params,cv=5)
        # fitting the model for grid search
        grid.fit(X_train, y_train)

        # print best parameter after tuning
        print('*' * 25)
        print(a[0],"'s best score: ",grid.best_score_)
        print('*' * 25)

        best_params = grid.best_params_
        best_params = {k.split('__')[1]:v for k,v in best_params.items()}
        print(a[0],"'s best parameters: ",best_params)
        print('*' * 25)
        hyperparam_tuning_dict[a[0]]=grid.best_score_,best_params
        
        # print how our model looks after hyper-parameter tuning with the best parameters
        best_model = grid.best_estimator_
        grid_predictions = grid.predict(X_test)

        #print confusion matrix
        print('*' * 25)
        print(a[0],"'s confusion matrix: \n",confusion_matrix(y_test, grid_predictions))
        print('*' * 25)
        # print classification report
        print(a[0],"'s classification report: \n",classification_report(y_test, grid_predictions))
        print('*' * 25)
        hyperparam_tuning_results_df = pd.DataFrame.from_dict(hyperparam_tuning_dict,orient='index',columns = ['best score','best params']).sort_values(by='best score',ascending=False)
    return hyperparam_tuning_results_df

#hyperparam_tuning_df = hyperparam_tuning(classifiers_for_hypertuning, params_for_hypertuning) #to call the function
#hyperparam_tuning_df.to_csv('hyperparam_tuning_df.csv') #to save results as a csv

#summarize impacts of hyperparameter tuning
def summarize_hyperarameter_impacts():
# Hyperparameter tuning - summary of impact
    summary_df = hyperparam_tuning_df.join(fit_predict_results_df)
    summary_df['hyperparam_uplift']=summary_df['best score']-summary_df['Accuracy']
    return summary_df

#summary_df = summarize_hyperarameter_impacts() # to call the function
#summary_df.to_csv('summary_hyperparamtuning_df.csv') # to save the summary as a csv

# to view the summary:
#print('*'*25)
#print("Ordered table of classifier accuracy with hyperparameter tuning and its impact (noting best hyperparameters):")
#print(summary_df)
#print('*'*25)

#Boosting
# detail list of classifiers for boosting
classifiers_to_boost = [('Decision Tree Classifier', DecisionTreeClassifier()), ('Random Forest',  RandomForestClassifier())]
# ada boost
def ada_boost(classifiers_to_boost):
    for clf_name, classifier in classifiers_to_boost:
        # Ada Boosting
        ada = AdaBoostClassifier(base_estimator = classifier,n_estimators=500)
        ada.fit(X_train,y_train)
        y_pred = ada.predict(X_test)
        print(clf_name,'score after ada boosting: ',ada.score(X_test,y_test))

#print('*'*25)
#print("Boosting accuracy scores:")
#ada_boost(classifiers_to_boost) #to call the function

# Gradient Boosting
def gradient_boost():
    gb = GradientBoostingClassifier()
    gb.fit(X_train,y_train)
    y_pred = gb.predict(X_test)
    print('Gradient Boosting score:',gb.score(X_test,y_test))

#gradient_boost() #to call the function
#print('*'*25)

#Feature importance with Random Forest
# instantiate and fit Random Forest
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
# Create a pd.Series of features importances
importances = pd.Series(data=rf.feature_importances_)
# Draw a horizontal barplot of importances_sorted
plt.barh(list_of_features,rf.feature_importances_)
plt.title('Features Importances determined by Random Forest')
plt.xlabel('Relative Importance')
#plt.savefig('rf_feature_importances.png',bbox_inches="tight")
#plt.show()
plt.close()

#KMeans clustering
def get_KMeans(n_clusters=2):
    kmeans = KMeans(n_clusters)
    y = kmeans.fit_predict(X_all)
    fixtures_3_years['cluster'] = y
    fixtures_3_years['KMeans_test'] = fixtures_3_years['result_binary'].apply(lambda x: x)
    labels = kmeans.labels_
    correct_labels = sum(fixtures_3_years['KMeans_test'] == labels)
    print(f'KMeans correct labels summary: KMeans correctly clustered {correct_labels} out of {len(fixtures_3_years)}, for overall accuracy of {100 * float(correct_labels / (len(fixtures_3_years))):.3f}%')
# get_KMeans(2) #to call the function
