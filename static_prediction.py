from model import Model
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

attributes = pd.read_csv('data/customAttributes.csv')
attributes = attributes.drop('Unnamed: 0', axis=1)
attributes = attributes.drop([
    'id',
    'overall_rating',
    'player_api_id',
    'player_fifa_api_id',
    'date',
    'preferred_foot',
    'attacking_work_rate',
    'defensive_work_rate'
], axis=1)

model = Model()
model.fit(LinearRegression())
print(f'LinearRegression: {round(model.staticPredict(attributes)[0], 2)}')

model.fit(KNeighborsRegressor(n_neighbors=20))
print(f'KNeighborRegressor: {round(model.staticPredict(attributes)[0], 2)}')

model.fit(MLPRegressor(max_iter=500))
print(f'MLPRegressor: {round(model.staticPredict(attributes)[0], 2)}')
