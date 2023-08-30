from model import Model
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

model1 = Model()
model2 = Model(goalkeepers=False, standard=True)
model3 = Model(pca=5, standard=True)
model4 = Model(pca=1)

A = [LinearRegression(), KNeighborsRegressor(n_neighbors=20), MLPRegressor(max_iter=500)]

for regressor in A:
    model1.fitAndPredict(regressor)
    model1.stats()
    model1.compare()
    print()

for regressor in A:
    model2.fitAndPredict(regressor)
    model2.stats()
    model2.compare()
    print()

for regressor in A:
    model3.fitAndPredict(regressor)
    model3.stats()
    model3.compare()
    print()

for regressor in A:
    model4.fitAndPredict(regressor)
    model4.stats()
    model4.compare()
    print()
