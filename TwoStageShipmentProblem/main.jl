include("Simulation.jl")
using .SimModule

sim = SimModule.Simulator(100, true)
X_train, Y_train = sim.X_train, sim.Y_train

# function get_model in Simulator is not implemented, try another approach


