# Step 1: Importation of Python packages to be used in data gathering, analysis, visualization and machine learning

# packages for data gathering, analysis and visualisation:
import pandas as pd
import requests
import json
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

# packages for machine learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier, BaggingClassifier

'''# Step 2: Access the data
df_list = []
# First API Call: obtain majority of FPL data for this season
master_link = 'https://fantasy.premierleague.com/api/bootstrap-static/'
request = requests.get(master_link)
master_data = request.json()
print(request.status_code)

# Create (for this season) a team information dataframe (via the key "team" data obtainable from the return of First API Call)
team_info_2021_df = pd.DataFrame.from_dict(master_data['teams'])
team_info_2021_df['season']='2020/21'
team_info_2021_df.df_name = 'team_info_2021_df'
df_list.append(team_info_2021_df)

# Second API Call: obtain FPL fixture data for this season and convert to pandas dataframe
fixtures_link = 'https://fantasy.premierleague.com/api/fixtures/'
request = requests.get(fixtures_link)
fixtures_data = request.json()
fixtures_2021 = pd.DataFrame.from_dict(fixtures_data)
fixtures_2021['season']='2020/21'
fixtures_2021.df_name = 'fixtures_2021'
df_list.append(fixtures_2021)
print(request.status_code)

# Obtain prior data for 2019-20 and 2018-19

# 2019-20 team info
team_info_1920_df = pd.read_csv('https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2019-20/teams.csv')
team_info_1920_df['season']='2019/20'
team_info_1920_df.df_name = 'team_info_1920_df'
df_list.append(team_info_1920_df)
# 2019-20 fixtures info
fixtures_1920 = pd.read_csv('https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2019-20/fixtures.csv')
fixtures_1920['season']='2019/20'
fixtures_1920.df_name = 'fixtures_1920'
df_list.append(fixtures_1920)
# 2018-19 team info
info_1819 = requests.get('https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2018-19/raw.json')
team_info_1819 = info_1819.json()
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

# Step 3 - Exploratory Data Analysis (EDA)

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
    print('Looking at .info()') # provides infromation regarding dataframe
    print(df.info())
    print('*'*50)
    print('Looking at .shape') # provides dimensions of dataframe (rows x columns)
    print(df.shape)
    print('*'*50)
    print('Looking at .columns') # provides list of the columns of each dataframe
    print(df.columns)
    print('*'*50)'''

#Data manipulation Round 1
'''# Here we want to combine the information of the fixtures and team information for each season

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

# Third, we merge the fixtures dataframe for each season the the home skill and awa skill dataframes
fixtures_2021 = fixtures_2021.merge(home_skill_2021_df, on = 'team_h').merge(away_skill_2021_df, on = 'team_a')
fixtures_1920 = fixtures_1920.merge(home_skill_1920_df, on = 'team_h').merge(away_skill_1920_df, on = 'team_a')
fixtures_1819 = fixtures_1819.merge(home_skill_1819_df, on = 'team_h').merge(away_skill_1819_df, on = 'team_a')

# Fourth, we concatenate the three years of fixtures to make one dataframe with all fixtures and their associated skill attributes
fixtures_3_years = pd.concat([fixtures_2021,fixtures_1920,fixtures_1819])

# Fifth, we create a column 'result' using numpy select that will eventually represent our target variable

conditions= [fixtures_3_years.team_h_score > fixtures_3_years.team_a_score,
fixtures_3_years.team_h_score == fixtures_3_years.team_a_score,
fixtures_3_years.team_h_score < fixtures_3_years.team_a_score,
fixtures_3_years.finished == False] # define the conditions

values = ['H','D','A','match pending'] # set out the values where the conditions are met

fixtures_3_years['result'] = np.select(conditions,values)

fixtures_3_years.to_csv('fixtures_3_years.csv')''' # data from the FPL API saved on 20 May 2021

# read in the data here:
fixtures_3_years = pd.read_csv('fixtures_3_years.csv') # reading in the saved FPL data from 20 May 2021
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

'''
# EDA Round 2 - exploring our combined dataframe:
fixtures_3_years.head()
fixtures_3_years.info()
fixtures_3_years.describe()
fixtures_3_years.shape

# From EDA Round 2, we note there are a number of missing values
print(fixtures_3_years.isnull().sum())

# As the missing value relate to where the fixture score values are missing, it is likely that this relates to where the dataframe includes matches that have not yet been played
print(fixtures_3_years.loc[fixtures_3_years['result'] == 'match pending', 'code'].isnull().count())
# this is confirmed by the above

# heatmap of missing values
sns.heatmap(fixtures_3_years.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()

# It is best to drop the rows containing null values in this instance as we cannot admit unplayed matches to our analysis. This can be done in a number of ways:
#fixtures_3_years = fixtures_3_years[fixtures_3_years.result!='match pending']
fixtures_3_years=fixtures_3_years.dropna()

# Here we create a heatmap to confirm that there are no missing values
sns.heatmap(fixtures_3_years.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()

# check for duplicates
duplicates = fixtures_3_years[fixtures_3_years.drop('stats',axis=1).duplicated()]
print(duplicates) # as None is printed, no duplicates are observed'''