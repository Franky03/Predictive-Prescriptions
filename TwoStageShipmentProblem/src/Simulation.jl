module SimModule

using LinearAlgebra
using Random
using Distributions

include("Shipment.jl")
using .ShipModule 

const Shipment = ShipModule.Shipment
const setup_model = ShipModule.setup_model

export Simulator, get_shipment_model

mutable struct Simulator
    size::Int
    verbose::Bool
    d_x::Int
    d_y::Int
    uCovar::Matrix{Float64}
    A::Matrix{Float64}
    B::Matrix{Float64}
    Theta1::Matrix{Float64}
    Theta2::Matrix{Float64}
    Phi1::Matrix{Float64}
    Phi2::Matrix{Float64}
    u::Matrix{Float64}
    X_train::Matrix{Float64}
    Y_train::Matrix{Float64}
end

function Simulator(size::Int, verbose::Bool)
    d_x = 3
    d_y = 12

    uCovar = [1.0 0.5 0.0; 0.5 1.2 0.5; 0.0 0.5 0.8]
    A = 2.5 * repeat([0.8 0.1 0.1; 0.1 0.8 0.1; 0.1 0.1 0.8], inner=(4, 1))
    B = 7.5 * [0.0 -1.0 -1.0; -1.0 0.0 -1.0; -1.0 -1.0 0.0; 0.0 -1.0 1.0;
               -1.0 0.0 1.0; -1.0 1.0 0.0; 0.0 1.0 -1.0; 1.0 0.0 -1.0;
               1.0 -1.0 0.0; 0.0 1.0 1.0; 1.0 0.0 1.0; 1.0 1.0 0.0]
    Theta1 = [0.4 0.8 0.0; -1.1 -0.3 0.0; 0.0 0.0 0.0]
    Theta2 = [0.0 0.8 0.0; -1.1 0.0 0.0; 0.0 0.0 0.0]
    Phi1 = [0.5 -0.9 0.0; 1.1 -0.7 0.0; 0.0 0.0 0.5]
    Phi2 = [0.0 -0.5 0.0; -0.5 0.0 0.0; 0.0 0.0 0.0]

    sim = Simulator(size, verbose, d_x, d_y, uCovar, A, B, Theta1, Theta2, Phi1, Phi2, zeros(size, d_x), zeros(size, d_x), zeros(size, d_y))
    sim.uCovar, sim.X_train, sim.Y_train = get_shipment_data(sim)
    return sim
end


function sample_shipment_innovations(sim::Simulator)
    return rand(MvNormal(zeros(sim.d_x), sim.uCovar))
end

function get_shipment_model(sim, weights::Vector{Float64})
    ship = Shipment(sim.Y_train, sim.d_y, 4, weights, sim.verbose, sim.size)
    ship = setup_model(ship)
    return ship
end

function shipment_covariate(u::Vector{Float64}, 
    Theta1::Matrix{Float64}, Theta2::Matrix{Float64}, 
    Phi1::Matrix{Float64}, Phi2::Matrix{Float64}, 
    um1::Vector{Float64} = zeros(3), 
    um2::Vector{Float64} = zeros(3), 
    xm1::Vector{Float64} = zeros(3), 
    xm2::Vector{Float64} = zeros(3))
    
x = u + (Theta1 * um1) + (Theta2 * um2) + (Phi1 * xm1) + (Phi2 * xm2)

return x
end

function sample_shipment_demand(sim::Simulator, x::Vector{Float64})
    demand = zeros(sim.d_y)
    for i in 1:sim.d_y
        innovation = x .+ (randn() / 4)
        demand[i] = (dot(sim.A[i, :], innovation) +
                     dot(sim.B[i, :], x) * randn())
    end
    return max.(zeros(sim.d_y), demand)
end

function get_shipment_data(sim::Simulator)
    u = zeros(sim.size, sim.d_x)
    x = zeros(sim.size, sim.d_x)
    y = zeros(sim.size, sim.d_y)

    println("Generating shipment data...")

    @assert sim.size >= 100 "Simulator size is too small for this loop"
    for i in 1:sim.size
        u[i, :] .= sample_shipment_innovations(sim)
        if i == 1
            x[i, :] .= shipment_covariate(u[i, :], sim.Theta1, sim.Theta2, sim.Phi1, sim.Phi2)
        elseif i == 2
            @assert i > 1 "Index out of bounds for i = 2"
            x[i, :] .= shipment_covariate(u[i, :], sim.Theta1, sim.Theta2, sim.Phi1, sim.Phi2, u[i-1, :], zeros(sim.d_x), x[i-1, :], zeros(sim.d_x))
        else
            @assert i > 2 "Index out of bounds for i > 2"
            x[i, :] .= shipment_covariate(u[i, :], sim.Theta1, sim.Theta2, sim.Phi1, sim.Phi2, u[i-1, :], u[i-2, :], x[i-1, :], x[i-2, :])
        end        
        y[i, :] .= sample_shipment_demand(sim, x[i, :])
    end
    
    return u, x, y
end

end