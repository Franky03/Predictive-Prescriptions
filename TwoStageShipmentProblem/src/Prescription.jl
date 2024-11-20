module PrescModule

include("Shipment.jl")
include("MLUtils.jl")
include("Simulation.jl")

using LinearAlgebra
using StatsBase

const Simulator = SimModule.Simulator
const get_shipment_model = SimModule.get_shipment_model

const Shipment = ShipModule.Shipment
const cost_function = ShipModule.cost_function
const risk_functional = ShipModule.risk_functional
const solve = ShipModule.solve

const MinMaxScaler = MLUtils.MinMaxScaler
const min_max_fit_transform! = MLUtils.min_max_fit_transform!


mutable struct Prescriptor
    shipment::Shipment
    scaler::Union{MinMaxScaler, Nothing}
    costFunction::Union{Function, Nothing}
    riskFunctional::Union{Function, Nothing}
    size::Union{Int, Nothing}

    function Prescriptor(shipment::Shipment, scaler::Union{MinMaxScaler, Nothing}, size::Int = nothing)
        new(shipment, scaler, cost_function, risk_functional, size)
    end
end


function get_risks_alt_and_opt(p::Prescriptor, weights::Vector{Float64}, z_alt::Any, z_opt::Any, Y_train::Matrix{Float64})
    altRisk = p.riskFunctional(p.shipment, z_alt, Y_train, weights)
    optRisk = p.riskFunctional(p.shipment, z_opt, Y_train, weights)
    return altRisk, optRisk
end

function _fit_scaler(prescriptor::Prescriptor, X_train::Matrix{Float64})
    prescriptor.scaler = MinMaxScaler()
    x_train_scaled = min_max_fit_transform!(prescriptor.scaler, X_train)
    return x_train_scaled
end

function _distance(x1::Vector, x2::Vector, objectiveNorm::Int = 1)
    @assert objectiveNorm == 1 "Only L1 norm is supported"
    return sum(abs.(x1 .- x2))
end

function cost_difference_vector(prescriptor::Prescriptor, z_alt, z_opt, sim::Simulator)
    """
    Calculate the vector of sample costs:
    {δ^i(z_alt, z_opt)}_i for i ∈ {1,…,n} .
    """

    return [
        prescriptor.costFunction(z_alt, sim.Y_train[s, :], s) -
        prescriptor.costFunction(z_opt, sim.Y_train[s, :], s)
        for s in 1:sim.size
    ]
end

function solve_cso_problem(sim::Simulator, x, prescriptor_weights::Function)
    """
    Solve Contextual Stochastic Optimization (CSO) problem
    in context x.
    """
    # build the weights
    weights = prescriptor_weights(x)
    # get the shipment model
    shipment = get_shipment_model(sim, weights)
    z_star = solve(shipment)
    return z_star
end

function solve_saa_problem(sim::Simulator)
    """
    Solve the Sample Average Approximation (SAA) problem.
    """
    shipment = get_shipment_model(sim, nothing)
    z_star = solve(shipment)
    return z_star
end

end