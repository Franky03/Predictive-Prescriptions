using LightGraphs
using DataStructures
const DiGraph = LightGraphs.DiGraph

using JuMP
using Gurobi
using Crayons.Box

using CSV
using DataFrames
using Random

# helper functions

const EPS = 1e-6

function print_graph(G)
    for e in edges(G)
        println("$e")
    end

    for v in vertices(G)
        println("$v")
    end
end

function relu(x)
    return max(0, x)
end

function hard_sigmoid(x)
    if x < -2.5
        return 0
    elseif x > 2.5
        return 1
    else
        return 0.2 * x + 0.5
    end
end

function heaviside(x)
    return x > 0 ? 1 : 0
end

function softmax(x)
    exp_x = exp.(x .- maximum(x)) # to avoid overflow
    return exp_x ./ sum(exp_x)
end

function reverse_mapping(mapping)
    return Dict(value => key for (key, value) in mapping)
end


activation_functions = Dict(
    "relu" => relu,
    "softmax" => softmax,
    "identity" => x -> x,
    "heaviside" => heaviside,
    "hard_sigmoid" => hard_sigmoid
)

# main functions

function get_neural_network(layer_sizes, activations)
    if length(layer_sizes) != length(activations)
        println("Error: number of layers and layer activations should match")
        return nothing
    end

    G = DiGraph() # directed graph
    layers = []
    bias_nodes = []

    nodes_attributes = OrderedDict() # dictionary to store attributes of each node
    edge_weights = OrderedDict() # dictionary to store weights of each edge
    node_mapping = OrderedDict() # dictionary to store mapping of nodes to their indices

    current_index = 1

    for(layer_idx, (size, activation)) in enumerate(zip(layer_sizes, activations))

        layer = [Symbol("l$(layer_idx)_$i") for i in 1:size] # create nodes for each layer

        push!(layers, layer)

        for node in layer
            # mapping node to its index
            node_mapping[node] = current_index
            #println("Node: $node, Index: $current_index")
            # ading node to the graph
            add_vertex!(G)
            # adding node to the dictionary
            nodes_attributes[node] = Dict(:activation => activation)
            current_index += 1
        end

        # adding bias node for the hidden layers
        if 1 < layer_idx <= length(layer_sizes)
            bias_node = Symbol("b_$(layer_idx)")
            add_vertex!(G)
            push!(bias_nodes, bias_node)
            node_mapping[bias_node] = current_index
            nodes_attributes[bias_node] = Dict(:activation => "identity")

            current_index += 1

            # connecting bias node to all the nodes in the current layer
            for node in layer
                add_edge!(G, node_mapping[bias_node], node_mapping[node])
                edge_weights[(bias_node, node)] = 1.0
            end
        end
    end

    # connecting nodes in the adjacent layers
    for (l1, l2) in zip(layers[1:end], layers[2:end])
        for node1 in l1
            for node2 in l2
                add_edge!(G, node_mapping[node1], node_mapping[node2])
                edge_weights[(node1, node2)] = 1.0
            end
        end
    end

    return (G, layers, bias_nodes, node_mapping, edge_weights, nodes_attributes)
end

function foward_propagation(G, layers, bias_nodes, x, node_mapping, edge_weights, node_attributes)
    # setting the input layer
    h_values = Dict(
        node => x[i] for (i, node) in enumerate(layers[1])
    )

    # setting the bias nodes
    for node in bias_nodes
        h_values[node] = 1.0
    end

    for layer_index in 2:length(layers)
        next_h_values = Dict()

        for node in layers[layer_index]
            weighted_sum = 0.0

            # summing up the weighted inputs
            for neighbor in inneighbors(G, node_mapping[node])
                # getting the label of the neighbor node
                neighbor_label = reverse_mapping(node_mapping)[neighbor]
                weight = edge_weights[(node_mapping[neighbor_label], node_mapping[node])]
                weighted_sum += weight * h_values[neighbor_label] # h_values[neighbor_label] is the output of the neighbor node
            end

            # applying the activation function
            activation_type = node_attributes[node][:activation]
            activation_fn = activation_functions[activation_type]
            next_h_values[node] = activation_fn(weighted_sum)
        end

        # updating the h_values
        for (node, value) in  next_h_values
            h_values[node] = value
        end
    end

    output_layer = layers[end]
    y = [h_values[node] for node in output_layer]
    
    return y
end

function create_model(G, layers, bias_nodes, X, Y)
    
    if typeof(X) != Matrix
        X = hcat(X...)' # transpose the input matrix
    end

    if typeof(Y) != Matrix
        Y = hcat(Y...)' # transpose the target matrix
    end

    M = 1000.0

    n, p = size(X) # number of samples and features
    q = size(Y, 2) # number of classes

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)
    
    @variable(model, z[1:n] >= 0) # sum of v for each sample 
    @variable(model, v[1:n, 1:q] >= 0) # diffeence between the output and the target value
    @variable(model, h[1:n, vertices(G)], lower_bound=-M, upper_bound=M) # output of each node
    @variable(model, π[1:n, vertices(G)], Bin)

    @variable(model, s1[1:n, vertices(G)], Bin)
    @variable(model, s2[1:n, vertices(G)], Bin)
    @variable(model, s3[1:n, vertices(G)], Bin)
    

    # create edge variables
    edge_list = [(src(e), dst(e)) for e in edges(G)]

    @variable(model, θ[1:n, edge_list], lower_bound=-M, upper_bound=M)

    @variable(model, w[edge_list], lower_bound=-1.0, upper_bound=1.0)

    # constraints
    for k in 1:n
        for d in 1:length(layers[end])
            j = node_mapping[layers[end][d]]
            # constraint : v[k,d] need to be absolute
            @constraint(model, Y[k, d] - h[k, j] <= v[k, d])
            @constraint(model, h[k, j] - Y[k, d] <= v[k, d])
        end
        @constraint(model, z[k] == sum(v[k, d] for d in 1:q))
    end

    # constraint for the input layer
    for k in 1:n
        for d in 1:length(layers[1])
            j = node_mapping[layers[1][d]]
            @constraint(model, h[k, j] == X[k, d])
        end
    end

    # constraint for the bias nodes
    for k in 1:n
        for d in 1:length(bias_nodes)
            j = node_mapping[bias_nodes[d]]
            @constraint(model, h[k, j] == 1.0)
        end
    end

    # constraint for the hidden layers (activation functions and aggregation)
    for k in 1:n
        for layer in layers[2:end]
            for d in 1:length(layer)
                j = node_mapping[layer[d]]
                activation_type = node_attributes[layer[d]][:activation]

                inneigh = inneighbors(G, j)  # Nós predecessores de `j`

                if activation_type == "heaviside"

                    @constraint(model, M * π[k, j] >= sum(θ[k, (i, j)] for i in inneigh))
                    @constraint(model, -M * (1 - π[k, j]) <= sum(θ[k, (i, j)] for i in inneigh))
                    @constraint(model, h[k, j] == π[k, j])
                    @constraint(model, sum(θ[k, (i, j)] for i in inneigh) >= EPS - M * (1 - π[k, j])) 
                    @constraint(model, sum(θ[k, (i, j)] for i in inneigh) <= M * π[k, j] - EPS)

                elseif activation_type == "relu"
                    
                    @constraint(model, sum(θ[k, (i, j)] for i in inneigh) <= h[k, j])
                    @constraint(model, sum(θ[k, (i, j)] for i in inneigh) + M * (1 - π[k, j]) >= h[k, j])
                    @constraint(model, h[k, j] <= M * π[k, j])
                    @constraint(model, h[k, j] >= 0)
                    
                elseif activation_type == "identity"

                    @constraint(model, sum(θ[k, (i, j)] for i in inneigh) == h[k, j])
                elseif activation_type == "hard_sigmoid"
                    # exclusive constraints (s1+s2+s3 = 1)
                    @constraint(model, s1[k, j] + s2[k, j] + s3[k, j] == 1)

                    # case 1: x < -2.5 → h[k, j] = 0
                    @constraint(model, sum(θ[k, (i, j)] for i in inneigh) <= -2.5 + M * (1 - s1[k, j]))
                    @constraint(model, h[k, j] <= M * (1 - s1[k, j]))
                    @constraint(model, h[k, j] >= -M * (1 - s1[k, j]))
                    
                    # case 2: -2.5 <= x <= 2.5 → h[k, j] = 0.2 * x + 0.5
                    @constraint(model, sum(θ[k, (i, j)] for i in inneigh) >= -2.5 - M * (1 - s2[k, j]))
                    @constraint(model, sum(θ[k, (i, j)] for i in inneigh) <= 2.5 + M * (1 - s2[k, j]))
                    @constraint(model, h[k, j] >= 0.2 * sum(θ[k, (i, j)] for i in inneigh) + 0.5 - M * (1 - s2[k, j]))
                    @constraint(model, h[k, j] <= 0.2 * sum(θ[k, (i, j)] for i in inneigh) + 0.5 + M * (1 - s2[k, j]))

                    # case 3: x > 2.5 → h[k, j] = 1
                    @constraint(model, sum(θ[k, (i, j)] for i in inneigh) >= 2.5 - M * (1 - s3[k, j]))
                    @constraint(model, h[k, j] >= 1 - M * (1 - s3[k, j]))
                    @constraint(model, h[k, j] <= 1 + M * (1 - s3[k, j]))
                end
            end
        end
    end

    # bilinear constraint

    for k in 1:n
        for (i, j) in edge_list
            # h_U -> activation function upper bound
            # h_L -> activation function lower bound

            h_U = activation_functions[node_attributes[reverse_mapping(node_mapping)[i]][:activation]](M)
            h_L = activation_functions[node_attributes[reverse_mapping(node_mapping)[i]][:activation]](-M)
            
            if reverse_mapping(node_mapping)[i] in layers[1]
                d = findfirst(x -> x == reverse_mapping(node_mapping)[i], layers[1])
                h_U = X[k, d]
                h_L = X[k, d]
            elseif reverse_mapping(node_mapping)[i] in bias_nodes
                h_U = 1.0
                h_L = 1.0
            end

            @constraint(model, θ[k, (i, j)] >= -h[k, i] + w[(i, j)] * h_L + h_L)
            @constraint(model, θ[k, (i, j)] <= h[k, i] + w[(i, j)] * h_L - h_L)
            @constraint(model, θ[k, (i, j)] >= h[k, i] + w[(i, j)] * h_U - h_U)
            @constraint(model, θ[k, (i, j)] <= -h[k, i] + w[(i, j)] * h_U + h_U)
            
        end
    end
            
    @objective(model, Min, sum(z[k] for k in 1:n)) # minimize the sum of z[k]

    optimize!(model)

    if termination_status(model) == MOI.OPTIMAL
        edge_weights = Dict()
        for (i, j) in edge_list
            edge_weights[(i, j)] = value(w[(i, j)])
        end
        return edge_weights
    else
        println("Model is not optimal :(")
        # exit program
        exit(1)
    end
end

function one_hot_encode(Y::Vector{String15}, binary_classification::Bool = true)
    classes = unique(Y)  # ["Iris-virginica", "Iris-setosa", "Iris-versicolor"]
    if !binary_classification
        class_to_onehot = Dict(c => [i == j ? 1.0 : 0.0 for j in 1:length(classes)] for (i, c) in enumerate(classes))
    else
        class_to_onehot = Dict(c => (i == 1 ? [1.0] : [0.0]) for (i, c) in enumerate(classes))
    end

    one_hot_Y = [class_to_onehot[y] for y in Y]
    return one_hot_Y, length(classes)
end

function read_iris_data(binary_classification::Bool = true)
    file_path = "iris.csv"
    iris_data = CSV.read(file_path, DataFrame)
    if binary_classification
        iris_data = filter(row -> row.Species in ["Iris-setosa", "Iris-versicolor"], iris_data)
    end

    Random.seed!(42)
    shuffle!(iris_data)

    X = [[row.SepalLengthCm, row.SepalWidthCm, row.PetalLengthCm, row.PetalWidthCm] for row in eachrow(iris_data)]
    Y = [row.Species for row in eachrow(iris_data)]

    Y_onehot, num_classes = one_hot_encode(Y, binary_classification)

    if binary_classification
        num_classes = 1
    end

    return X, Y_onehot, num_classes
end

function split_data(X, Y, train_ratio::Float64 = 0.8)
    n_samples = length(X)
    train_size = Int(floor(train_ratio * n_samples))

    X_train = X[1:train_size]
    Y_train = Y[1:train_size]

    X_test = X[train_size+1:end]
    Y_test = Y[train_size+1:end]

    return X_train, Y_train, X_test, Y_test
end

function confusion_matrix(y_test, y_preds)
    tp = sum((y_test .== 1.0) .& (y_preds .== 1.0))
    tn = sum((y_test .== 0.0) .& (y_preds .== 0.0))
    fp = sum((y_test .== 0.0) .& (y_preds .== 1.0))
    fn = sum((y_test .== 1.0) .& (y_preds .== 0.0))
    return [tp fn; fp tn]  # formato [[TP, FN], [FP, TN]]
end

function classification_metrics(conf_matrix)
    tp, fn = conf_matrix[1, 1], conf_matrix[1, 2]
    fp, tn = conf_matrix[2, 1], conf_matrix[2, 2]
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)  # Evitar divisão por zero
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    return Dict(
        "Accuracy" => accuracy,
        "Precision" => precision,
        "Recall" => recall,
        "F1 Score" => f1_score
    )
end

X, Y, n_out = read_iris_data()
X_train, Y_train, X_test, Y_test = split_data(X, Y)

n_in = length(X[1])

layer_sizes = [n_in, n_out] # 1 because it is perceptron XD
activations = ["identity", "hard_sigmoid"]

println("Layer sizes: $layer_sizes")

started_at = time()

params = get_neural_network(layer_sizes, activations)

if params !== nothing
    (G, layers, bias_nodes, node_mapping, edge_weights, node_attributes) = params
    
    edge_weights = create_model(G, layers, bias_nodes, X_train, Y_train)

    end_train = time()

    println("Training time: $(end_train - started_at)")

    started_at = time()

    y_preds = []
    for (k, x) in enumerate(X_test)   
        y_pred = foward_propagation(G, layers, bias_nodes, x, node_mapping, edge_weights, node_attributes)
        y_pred = y_pred[1] > 0.5 ? 1.0 : 0.0
        push!(y_preds, y_pred)
        println("real: $(Y_test[k]), predicted: $y_pred")
    end

    end_test = time()
    println("Testing time: $(end_test - started_at)")

    y_preds = [y[1] for y in y_preds]
    y_test = [y[1] for y in Y_test]

    cm = confusion_matrix(y_test, y_preds)
    println(cm)

    metrics = classification_metrics(cm)
    println("Metrics:")
    for (key, value) in metrics
        println("$key: $value")
    end
end
