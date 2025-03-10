using PyCall
using ScikitLearn
pyimport("numpy")

println(PyCall.python)

ensemble = pyimport("sklearn.ensemble")
RandomForestRegressor = ensemble.RandomForestRegressor