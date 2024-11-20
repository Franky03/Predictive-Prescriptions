include("Simulation.jl")
include("Shipment.jl")
using .ShipModule
using .SimModule

sim = SimModule.Simulator(100, true)
X_train, Y_train = sim.X_train, sim.Y_train

shipmet =  get_shipment_model(sim, 
    fill(1.0 / length(sim.Y_train[1, :]), length(sim.Y_train[1, :]))
)

@show shipmet
