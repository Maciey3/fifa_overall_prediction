import pandas as pd
import sqlite3

con = sqlite3.connect('../database.sqlite')
df = pd.read_sql_query("select * from Player", con)
df.to_csv('../data/Players.csv')

df = pd.read_sql_query("select * from Player_Attributes", con)
df.to_csv('../data/Players_Attributes.csv')