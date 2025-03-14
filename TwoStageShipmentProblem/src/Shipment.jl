module ShipModule

using JuMP
import HiGHS
using GLPK
using DelimitedFiles
using LinearAlgebra

export Shipment, setup_model, solve, cost_function, risk_functional

mutable struct Shipment 
    demands::Matrix{Float64} # demands for each location
    dy::Int # number of locations
    dz::Int # number of warehouses
    ship_cost::Float64 # cost of shipping (per unit)
    prod_cost::Float64 # cost of production (per unit)
    last_minute_cost::Float64 # cost of production in the last minute (per unit)
    distance_matrix::Array{Float64, 2} # distance matrix between warehouses and locations
    model::Model 
    weights::Vector{Float64} # adaptive weights for the objective function
end

function Shipment(
    demands::Matrix{Float64}, 
    dy::Int,
    dz::Int = 4, 
    weights::Union{Nothing, Vector{Float64}} = nothing,
    verbose::Bool = false,
    size::Int = 100
)
    # parameters
    ship_cost = 10.0
    prod_cost = 5.0
    last_minute_cost = 100.0
    
    # create the distnce matrix with base on the circle 

    """
    We take locations spaced evenly on the 2-dimensional unit circle and warehouses spaced
    evenly on the circle of radius 0.85.
    """

    warehouses = [
        [0.85, 0], [0, 0.85], [-0.85, 0], [0, -0.85], [0.42, 0.42]
    ]

    distance_matrix = zeros(Float64, dz, dy)
    for i in 1:dy 
        location = [cos(i * π / (dy / 2)), sin(i * π / (dy / 2))] # symmetric locations on the unit circle
        for j in 1:dz
            distance_matrix[j, i] = sqrt(sum((warehouses[j] .- location).^2))
        end
    end

    # adaptive weights
    if weights === nothing
        weights = fill(1.0 / length(demands), length(demands)) # equal weights
    else
        @assert length(weights) ==  size "Weights size is incorrect, expected $(size), got $(length(weights))"
    end

    # create model 
    model = Model(HiGHS.Optimizer)
    #set_optimizer_attribute(model, "msg_lev", verbose ? GLPK.GLP_MSG_ALL : GLPK.GLP_MSG_OFF)

    Shipment(demands, dy, dz, ship_cost, prod_cost, last_minute_cost, distance_matrix, model, weights)
end

function setup_model(shipment::Shipment)
    model = shipment.model
    dz, dy = shipment.dz, shipment.dy
    demands = shipment.demands
    ship_cost, prod_cost, spot_cost = shipment.ship_cost, shipment.prod_cost, shipment.last_minute_cost
    distance_matrix = shipment.distance_matrix
    weights = shipment.weights

    # variables

    # quantity shipped from warehouse i to location j
    @variable(model, s[1:dz, 1:dy, 1:size(demands)[1]] >= 0) 
    # production in the last hour in each warehouse i
    @variable(model, t[1:dz, 1:size(demands)[1]] >= 0) 
    # quantity produced in advance in each warehouse i
    @variable(model, z[1:dz] >= 0) 

    # constraints
    
    for sample in 1:size(demands)[1]
        # satisfy demand for each location j
        for j in 1:dy
            @constraint(model, sum(s[i, j, sample] for i in 1:dz) >= demands[sample, j])
        end

        # flow conservation: production and spot orders cover shipments
        for i in 1:dz
            @constraint(model, sum(s[i, j, sample] for j in 1:dy) <= z[i] + t[i, sample])
        end
    end

    # objective function

    @objective(
        model, Min,
        # first stage costs
        prod_cost * sum(z[i] for i in 1:dz) +
        # second stage costs 
        sum(    
            weights[sample] * (
                spot_cost * sum(t[i, sample] for i in 1:dz) +
                ship_cost * sum(distance_matrix[i, j] * s[i, j, sample] for i in 1:dz, j in 1:dy)
            ) for sample in 1:size(demands)[1]
        )
    )

    shipment
end

function solve(shipment)
    optimize!(shipment.model)
    
    prod_obj = Dict()
    last_minute_obj = Dict()
    ship_obj = Dict()

    dz, dy = shipment.dz, shipment.dy
    for i in 1:dz
        prod_obj[i] = value(shipment.model[:z][i])
        for s in 1:size(shipment.demands)[1]
            last_minute_obj[(i, s)] = value(shipment.model[:t][i, s])
            for j in 1:dy
                ship_obj[(i, j, s)] = value(shipment.model[:s][i, j, s])
            end
        end
    end

    return (prod_obj, last_minute_obj, ship_obj)
end

function _array_recourse_decision(shipment::Shipment, spot, ship)
    spot_array = zeros(Float64, shipment.dz, size(shipment.demands)[1])
    ship_array = zeros(Float64, shipment.dz, shipment.dy, size(shipment.demands)[1])

    for i in 1:shipment.dz
        for s in 1:size(shipment.demands)[1]
            spot_array[i, s] = spot[(i, s)] # spot order at warehouse i for sample s 
            for j in 1:shipment.dy # shipment from warehouse i to location j for sample s
                ship_array[i, j, s] = ship[(i, j, s)] 
            end
        end
    end

    return spot_array, ship_array
end

function cost_function(shipment::Shipment, z, spot=nothing, ship=nothing, s=nothing)
    "if spot and ship are null, then the cost is calculated for the first stage"
    if spot === nothing || ship === nothing
        prod, spot, ship = z
    else 
        prod, spot, ship = z
        spot_array, ship_array = _array_recourse_decision(shipment, spot, ship)
        spot = spot_array[:, s]
        ship = ship_array[:, :, s]
    end

    # calculate the costs
    prod_costs = sum(shipment.prod_cost * prod[i] for i in 1:shipment.dz)
    spot_costs = sum(shipment.last_minute_cost * spot[i] for i in 1:shipment.dz)
    ship_costs = sum(
        sum(shipment.ship_cost * shipment.distance_matrix[i, j] * ship[i, j] for j in 1:shipment.dy) 
        for i in 1:shipment.dz
    )

    return prod_costs + spot_costs + ship_costs

end

function risk_functional(shipment::Shipment, z, demand, weights)
    # calculate the expected cost
    index_ = 1:size(demands)[1]
    prod, spot, ship = z
    spot_array, ship_array = _array_recourse_decision(shipment, spot, ship)

    expected_costs = sum(
        cost_function(shipment, (prod, spot_array, ship_array), s) * weights[s] for s in index_
    )

    return expected_costs
end

end