import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class Model:
    def __init__(self, goalkeepers=True, standard=False, pca=0, playersSrc='data/Players.csv', attributesSrc='data/Players_Attributes.csv'):
        self.playersSrc = playersSrc
        self.attributesSrc = attributesSrc
        self.trainingSet = 0

        self.initPlayers()
        self.initAttributes()
        self.concatenatePlayersAttributes()
        if goalkeepers:
            self.subsGoalkeepers()
        self.subsOverall()
        if standard:
            self.standardScale()
        if pca:
            self.pca(pca)

    def initPlayers(self):
        self.players = pd.read_csv(self.playersSrc)
        self.players = self.players.drop('Unnamed: 0', axis=1)
        self.players = self.players.drop([
            'player_api_id',
            'player_fifa_api_id',
            'birthday',
            'height',
            'weight'
        ], axis=1)

    def initAttributes(self):
        self.attributes = pd.read_csv(self.attributesSrc)
        self.attributes = self.attributes.drop('Unnamed: 0', axis=1)
        self.attributes = self.attributes.drop([
            'player_api_id',
            'player_fifa_api_id',
            'date',
            'preferred_foot',
            'attacking_work_rate',
            'defensive_work_rate'
        ], axis=1)

    def concatenatePlayersAttributes(self):
        self.playersWithAttributes = pd.merge(self.players, self.attributes, how='left', left_on='id', right_on='id').dropna(subset=['overall_rating', 'volleys'])
        self.playersWithAttributes = self.playersWithAttributes.drop('player_name', axis=1)
        self.playersWithAttributes = self.playersWithAttributes.drop('id', axis=1)

    def subsGoalkeepers(self):
        self.playersWithAttributes = self.playersWithAttributes.drop(self.playersWithAttributes[self.playersWithAttributes.gk_positioning > 50].index)

    def subsOverall(self):
        self.overall = self.playersWithAttributes['overall_rating']
        self.playersWithAttributes = self.playersWithAttributes.drop('overall_rating', axis=1)

    def trainingAndTestSplit(self):
        if not self.trainingSet:
            self.players_train,\
            self.players_test,\
            self.overall_train,\
            self.overall_test = train_test_split(self.playersWithAttributes, self.overall, test_size=0.2)
            self.trainingSet = 1

    def standardScale(self):
        self.playersWithAttributes = StandardScaler().fit_transform(self.playersWithAttributes)

    def pca(self, n):
        pca = PCA(n_components=n)
        self.playersWithAttributes = pca.fit_transform(self.playersWithAttributes)

    def fit(self, regressor):
        self.trainingAndTestSplit()
        self.regressor = regressor
        self.regressor.fit(self.players_train, self.overall_train)

    def fitAndPredict(self, regressor):

        self.trainingAndTestSplit()
        self.regressor = regressor
        self.regressor.fit(self.players_train, self.overall_train)
        self.prediction = self.regressor.predict(self.players_test)

    def staticPredict(self, playerStats):
        return self.regressor.predict(playerStats)

    def stats(self):
        prediction = self.prediction
        print(f'{self.regressor.__class__.__name__} mse: {round(self.mse(), 2)}')
        print(f'{self.regressor.__class__.__name__} mae: {round(self.mae(), 2)}')
        print(f'{self.regressor.__class__.__name__} r2 score: {round(self.r2(), 2)}')

    def compare(self):
        quant = 30
        plt.plot()
        plt.title(f'Przykładowe porównanie predykcji\n z danymi rzeczywistymi ({quant} elementów)\n'
                  f'[{self.regressor.__class__.__name__}]')
        plt.scatter(range(quant), self.overall_test[:quant], color='blue', label='Dane rzeczywiste')
        plt.scatter(range(quant), self.prediction[:quant], color='red', alpha=0.5, label='Predykcja')
        plt.legend()
        plt.ylabel('Ocena zawodnika')
        plt.tight_layout()
        plt.show()

        if self.playersWithAttributes.shape[1] == 1:
            plt.plot()
            plt.title(f'Regresja dla danych jednowymiarowych otrzymanych przez PCA\n'
                      f'[{self.regressor.__class__.__name__}]')
            plt.scatter(self.players_test, self.overall_test, color='blue', label='Dane rzeczywiste')
            plt.scatter(self.players_test, self.prediction, color='red', alpha=0.5, label='Predykcja')
            plt.legend()
            plt.ylabel('Ocena zawodnika')
            plt.tight_layout()
            plt.show()

    def mse(self):
        return mean_squared_error(self.overall_test, self.prediction)

    def mae(self):
        return mean_absolute_error(self.overall_test, self.prediction)

    def r2(self):
        return r2_score(self.overall_test, self.prediction)

