module RfPrescModule

include("Shipment.jl")
include("MLUtils.jl")
include("Simulation.jl")

using LinearAlgebra
using ScikitLearn
using ScikitLearn: fit!
using Random
using DecisionTree
using PyCall

@sk_import ensemble: RandomForestRegressor

const RandomizedSearchCV = ScikitLearn.GridSearch.RandomizedSearchCV

const Simulator = SimModule.Simulator
const Shipment = ShipModule.Shipment
const MinMaxScaler = MLUtils.MinMaxScaler

const get_shipment_model = SimModule.get_shipment_model
const cost_function = ShipModule.cost_function
const risk_functional = ShipModule.risk_functional
const solve = ShipModule.solve

const min_max_fit_transform! = MLUtils.min_max_fit_transform!
const min_max_transform! = MLUtils.min_max_transform!

export RfPrescriptor, solve_cso_problem, solve_saa_problem

mutable struct RfPrescriptor
    randomForest::PyObject
    nbTrees::Int
    nbSamples::Int
    Y_train::Matrix{Float64}
    x_train_scaled::Union{Matrix{Float64}, Nothing}
    trainingLeaves::Union{Matrix{Int}, Nothing}
    nbSamplesInLeaf::Union{Dict{Int, Dict{Int, Int}}, Nothing}
    isSampleInTreeLeaf::Union{Dict{Int, Dict{Int, Vector{Int}}}, Nothing}
    scaler::Union{Any, Nothing}
    cvarOrder::Int
    W_MAX::Float64
end

function RfPrescriptor(
    X_train, Y_train, nbTrees=100, cvarOrder=2, random_state=nothing, isScaled=false
)   
    if isnothing(random_state)
        random_state = MersenneTwister(12)
    end
    prescriptor = RfPrescriptor(
        nothing, nbTrees, size(X_train, 1), Y_train, nothing, nothing, nothing, nothing, nothing, cvarOrder, 0.0
    )

    if !isScaled
        scaler = MinMaxScaler()
        X_train_scaled = min_max_fit_transform!(scaler, X_train)
        prescriptor.scaler = scaler
    else
        X_train_scaled = X_train
        prescriptor.scaler = nothing 
    end

    prescriptor.x_train_scaled = X_train_scaled

    param_dist = Dict(
        "n_estimators" => 100:200:1000,
        "max_depth" => 1:10,
        "min_samples_split" => 2:10
    )

    rf = RandomForestRegressor()
    model = RandomizedSearchCV(rf, param_dist, n_iter=10, cv=5, random_state=random_state)

    search = fit!(model, X_train_scaled, Y_train)

    nbTrees = search.best_params_[:n_estimators]
    prescriptor.nbTrees = nbTrees

    prescriptor.randomForest = model.best_estimator_

    # read the structure of the random forest 

    trainingLeaves = apply_random_forest(prescriptor.randomForest, X_train_scaled)

    prescriptor.trainingLeaves = trainingLeaves

    prescriptor.nbSamplesInLeaf = count_samples_in_leaves(prescriptor.randomForest, trainingLeaves)

    prescriptor.isSampleInTreeLeaf = get_matrix_of_sample_leaf_tree(prescriptor.randomForest, trainingLeaves)

    return prescriptor
end

function predict_leaf(tree, x)
    node = tree.root
    while !node.is_leaf
        if x[node.feature] <= node.threshold
            node = node.left
        else
            node = node.right
        end
    end
    return node.id
end


function apply_random_forest(model, X::Matrix{Float64})
    return PyCall.PyObject.(model.apply(X))
end

function count_samples_in_leaves(rf, trainingLeaves)
    "Count the number of samples in each leaf of each tree of the random forest."
    nbTrees = length(rf.estimators_)
    nbSamplesInLeaf = Dict{Int, Dict{Int, Int}}()
    for t in 1:nbTrees
        nbSamplesInLeaf[t] = Dict{Int, Int}()
        tree = rf.estimators_[t].tree_
        for v in 1:tree.node_count
            isLeaf = (tree.children_left[v] == -1)
            if isLeaf
                nbSamplesInLeaf[t][v] = sum(trainingLeaves[:, t] .== v)
            end
        end
    end 
    return nbSamplesInLeaf
end

function get_matrix_of_sample_leaf_tree(rf,trainingLeaves)
    "Return a matrix of size (n_samples, n_trees) where each element is the leaf index of the sample in the tree."
    nbTrees = length(rf.estimators_)
    isSampleInTreeLeaf = Dict{Int, Dict{Int, Vector{Int}}}()
    for t in 1:nbTrees
        tree = rf.estimators_[t].tree_
        isSampleInTreeLeaf[t] = Dict{Int, Vector{Int}}()
        for v in 1:tree.node_count
            isLeaf = (tree.children_left[v] == -1)
            if isLeaf
                isSampleInTreeLeaf[t][v] = findall(trainingLeaves[:, t] .== v)
            end
        end
    end
    return isSampleInTreeLeaf
end

function prescriptor_weights(prescriptor::RfPrescriptor, context::Matrix{Float64})
    "Calculate the prescription weights for a given context."
    nbSamples = prescriptor.nbSamples
    context = min_max_transform!(prescriptor.scaler, context) # scale the context
    # get the leaves of the trees
    leaves = apply_random_forest(prescriptor.randomForest, context)
    weights = zeros(Float64, nbSamples)

    for i in 1:nbSamples
        treeWeights = zeros(Float64, prescriptor.nbTrees)
        for t in 1:prescriptor.nbTrees
            v = leaves[t]
            if haskey(prescriptor.isSampleInTreeLeaf[t], v)
                I = (i in prescriptor.isSampleInTreeLeaf[t][v]) ? 1.0 : 0.0
                S = prescriptor.nbSamplesInLeaf[t][v]
                treeWeights[t] = I / S
            end
        end

        weights[i] = sum(treeWeights) / prescriptor.nbTrees
    end

    # validate sum of weights
    if (abs(sum(weights)) - 1) > 1e-12
        error("Erro ao calcular pesos: soma dos pesos não é 1.")
    end
    
    return weights
end

function get_risks_alt_and_opt(shipment::Shipment, weights::Vector{Float64}, z_alt::Any, z_opt::Any, Y_train::Matrix{Float64})
    altRisk = risk_functional(shipment, z_alt, Y_train, weights)
    optRisk = risk_functional(shipment, z_opt, Y_train, weights)
    return altRisk, optRisk
end

function _distance(x1::Vector, x2::Vector, objectiveNorm::Int = 1)
    @assert objectiveNorm == 1 "Only L1 norm is supported"
    return sum(abs.(x1 .- x2))
end

function cost_difference_vector(z_alt, z_opt, sim::Simulator)
    """
    Calculate the vector of sample costs:
    {δ^i(z_alt, z_opt)}_i for i ∈ {1,…,n} .
    """

    return [
        cost_function(z_alt, sim.Y_train[s, :], s) -
        cost_function(z_opt, sim.Y_train[s, :], s)
        for s in 1:sim.size
    ]
end

function solve_cso_problem(sim, prescriptor::RfPrescriptor, x::Matrix{Float64})
    """
    Solve Contextual Stochastic Optimization (CSO) problem
    in context x.
    """
    # build the weights
    weights = prescriptor_weights(prescriptor, x)
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