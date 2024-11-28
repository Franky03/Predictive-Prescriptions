using LightGraphs
using DataStructures
const DiGraph = LightGraphs.DiGraph

# helper functions

function print_graph(G)
    for e in edges(G)
        println("$e")
    end
end

function relu(x)
    return max(0, x)
end

function sigmoid(x)
    return 1 / (1 + exp(-x))
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
    "sigmoid" => sigmoid,
    "softmax" => softmax,
    "identity" => x -> x
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

        println("Layer $layer_idx: $layer")

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
        if 1 <= layer_idx < length(layer_sizes)
            bias_node = Symbol("b_$(layer_idx)")
            add_vertex!(G)
            push!(bias_nodes, bias_node)
            #println("Bias Node: $bias_node, Index: $current_index")
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
    for (l1, l2) in zip(layers[1:end-1], layers[2:end])
        for node1 in l1
            for node2 in l2
                add_edge!(G, node_mapping[node1], node_mapping[node2])
                edge_weights[(node1, node2)] = 1.0
            end
        end
    end

    #print_graph(G)

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
                weight = edge_weights[(neighbor_label, node)]
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

layer_sizes = [3, 4, 2]
activations = ["relu", "sigmoid", "softmax"]
params = get_neural_network(layer_sizes, activations)

if params !== nothing
    (G, layers, bias_nodes, node_mapping, edge_weights, node_attributes) = params
    x = [4.0, 2.0, 3.0]
    y = foward_propagation(G, layers, bias_nodes, x, node_mapping, edge_weights, node_attributes)
    println("Output: $y")
end
