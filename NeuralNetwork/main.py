import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import mip
from mip import Model, xsum, maximize, BINARY, OptimizationStatus

# activation functions 

EPS = 1e-6

def heaviside(x):
    return 1 if x > 0 else 0

def relu(x):
    return max(0, x)

# This workaround is acceptable since inputs are assumed 
# to be bounded within the unit boX
def identity(x):
    return x

activation_functions = {
    "heaviside": heaviside,
    "relu": relu,
    "identity": identity,
}

def construct_layered_graph(layer_sizes, activations):
    if len(layer_sizes) != len(activations):
        print("Error: number of layers and layer activations should match")
        exit(0)

    G = nx.DiGraph()

    layers = []
    bias_nodes = []

    for layer_idx, (size, activation) in enumerate(zip(layer_sizes, activations)):
        layer = [f"L{layer_idx}_{i}" for i in range(1, size + 1)]
        layers.append(layer)
        G.add_nodes_from(layer, activation=activation)  

        if 0 < layer_idx < len(layer_sizes):  
            bias_node = f"B_{layer_idx}"
            bias_nodes.append(bias_node)
            G.add_node(bias_node, activation="identity")  
            for node in layer:
                G.add_edge(bias_node, node, weight=1.0)  

    # connecting two consecutive layers
    for l1, l2 in zip(layers, layers[1:]):
        for node1 in l1:
            for node2 in l2:
                G.add_edge(node1, node2, weight=1.0)

    return G, layers, bias_nodes

def draw_layered_graph(G, layers, bias_nodes):
    # node horizontal positions by layer
    pos = {}
    for i, layer in enumerate(layers):
        for j, node in enumerate(layer):
            pos[node] = (i, -j)  

        # push bias nodes away a bit
        if 0 < i < len(layers):  
            bias_node = f"B_{i}"
            pos[bias_node] = (i - 0.5, 1)

    # labels = {node: G.nodes[node]['activation'] for node in G.nodes}
    labels = {node: node for node in G.nodes}

    node_colors = ["red" if node in bias_nodes else "lightblue" for node in G.nodes]
    nx.draw(
            G, pos, with_labels=True, labels=labels, node_color=node_colors, 
            font_size=8, node_size=1000, edge_color="gray", connectionstyle='arc3,rad=0.1'
            )

    edge_labels = {
            (i, j): f"{data['weight']:.2f}" for i, j, data in G.edges(data=True)
            }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    plt.show()

def forward_pass(G, layers, bias_nodes, x):
    h_values = {f"L0_{i+1}": x[i] for i in range(len(x))}

    next_h_values = {}

    for node in bias_nodes:
        next_h_values[node] = 1

    h_values.update(next_h_values)

    for layer_idx, layer in enumerate(layers):
        if layer_idx == 0:
            continue

        next_h_values = {}

        for node in layer:
            weighted_sum = 0
            for prev_node in G.predecessors(node):
                edge_weight = G.get_edge_data(prev_node, node).get('weight')
                weighted_sum += h_values[prev_node] * edge_weight

            next_h_values[node] = activation_functions[G.nodes[node]['activation']](weighted_sum)

        h_values.update(next_h_values)

    output_layer = layers[-1]
    y = np.array([h_values[node] for node in output_layer])

    return y

def gen_model(G, layers, bias_nodes, X, Y):
    M = 1000.0

    model = mip.Model(sense=mip.MINIMIZE, solver_name=mip.CBC)

    n = len(X) # Number of samples
    p = len(X[0]) # Number of features
    q = len(Y[0]) # Number of output classes

    print(f"Number of samples: {n}")
    print(f"Number of features: {p}")
    print(f"Number of output classes: {q}")

    z = [model.add_var(var_type=mip.CONTINUOUS, lb=0.0, name=f"z_{k}") for k in range(n)] 
    v = [[model.add_var(var_type=mip.CONTINUOUS, lb=0.0, name=f"v_{k}_{d}") for d in range(q)] for k in range(n)]

    h = {
            k: {j: model.add_var(var_type=mip.CONTINUOUS, lb=-M, ub=M, name=f"h_{k}_{j}") for j in G.nodes}
            for k in range(n)
            }

    pi = {
            k: {j: model.add_var(var_type=mip.BINARY, name=f"pi_{k}_{j}") for j in G.nodes} 
            for k in range(n)
            }

    theta = {
            k: {(i, j): model.add_var(var_type=mip.CONTINUOUS, lb=-M, ub=M, name=f"theta_{k}_{i}_{j}")
                for (i, j) in G.edges}
            for k in range(n)
            }

    w = {(i, j): model.add_var(var_type=mip.CONTINUOUS, lb=-1.0, ub=1.0, name=f"w_{i}_{j}") for i, j in G.edges}

    # for var in model.vars:
    #     print(f"Variable name: {var.name}")


    # Linearizing the objective function (1-norm)
    model.objective = xsum(z[k] for k in range(n))

    for k in range(n):
        for d, j in enumerate(layers[-1]):
            # model += mip.xsum(h[k][d] for k in range(n)) <= v[k][d]#, "sum_constraint"
            model += float(Y[k, d]) - h[k][j] <= v[k][d]
            model += h[k][j] - float(Y[k, d]) <= v[k][d]
        model += z[k] == xsum(v[k][d] for d in range(q))

    # Input nodes
    for k in range(n):
        for d, j in enumerate(layers[0]):
            model += h[k][j] == X[k][d]

    # Bias nodes
    for k in range(n):
        for j in bias_nodes:
            model += h[k][j] == 1

    # Linearizing the activation and aggregation functions
    for k in range(n):
        for layer in layers[1:]:
            for j in layer:
                if G.nodes[j]['activation'] == "heaviside":
                    model += M * pi[k][j] >= xsum(theta[k][(i, j)] for i in G.predecessors(j))#, f"constraint_pi_{k}_{j}"
                    model += -M * (1 - pi[k][j]) <= xsum(theta[k][(i, j)] for i in G.predecessors(j))#, f"constraint_pi_{k}_{j}"
                    model += h[k][j] == pi[k][j] 
                    
                    model +=  xsum(theta[k][(i, j)] for i in G.predecessors(j)) >=  EPS - M * (1 - pi[k][j])
                    model += xsum(theta[k][(i, j)] for i in G.predecessors(j))  <= -EPS + M * (pi[k][j])

                    # x >=  epsilon - M * (1 - y)
                    # x <= -epsilon + M * y

                if G.nodes[j]['activation'] == "relu":
                    model += h[k][j] >= xsum(theta[k][(i, j)] for i in G.predecessors(j))
                    model += h[k][j] <= xsum(theta[k][(i, j)] for i in G.predecessors(j)) + M * (1 - pi[k][j]) 
                    model += h[k][j] <= M * pi[k][j]
                    model += h[k][j] >= 0

                if G.nodes[j]['activation'] == "identity":
                    model += h[k][j] == xsum(theta[k][(i, j)] for i in G.predecessors(j))

    # Bilinear constraints
    for k in range(n):
        for i, j in G.edges:
            h_U = activation_functions[G.nodes[i]['activation']](M)
            h_L = activation_functions[G.nodes[i]['activation']](-M)

            if i in layers[0]:
                d = layers[0].index(i)
                h_U = X[k][d]
                h_L = X[k][d]

            if i in bias_nodes:
                h_U = 1.0
                h_L = 1.0

            # print("a", i,h_U, h_L,activation_functions[G.nodes[i]['activation']] )

            model += theta[k][(i, j)] >= -h[k][i] + w[(i, j)] * h_L + h_L
            model += theta[k][(i, j)] >= h[k][i] + w[(i, j)] * h_U - h_U
            model += theta[k][(i, j)] <= h[k][i] + w[(i, j)] * h_L - h_L
            model += theta[k][(i, j)] <= w[(i, j)] * h_U - h[k][i] + h_U

                
    status = model.optimize()
    
    if status == mip.OptimizationStatus.OPTIMAL:
        #print(f"Optimal objective value: {model.objective_value}")

        w_vec = []

        for (i, j), var in w.items():
            #print(f"w[{i}, {j}] = {var.x}")
            G[i][j]['weight'] = var.x

            w_vec.append(var.x)

        return w_vec

        # for k in h:
        #     for j in h[k]:
        #         value = h[k][j].x
        #         print(f"h[{k}][{j}] = {value}")

        # print()

        # # for k in range(n):
        # #     for (i,j) in G.edges():
        # #         print(f"t[{k}][{i} {j}] = {theta[k][(i,j)].x}, {w[(i,j)].x}, {h[k][i].x}")

        # for k in range(n):
        #     print('----')
        #     for j in layers[1]:
        #         acc = 0.0
        #         acc1 = 0.0
        #         for i in G.predecessors(j):
        #             acc += h[k][i].x * w[(i,j)].x
        #             acc1 += theta[k][(i,j)].x

        #         print(j, acc, acc1, h[k][j].x)

    else:
        print("No solution found.")
        model.write("model.lp")  # Writes IIS to a file

    # exit(0)

# iris dataset test
""" import pandas as pd

file_path = "iris.csv"

df = pd.read_csv(file_path)

X = df.iloc[:, 1:5].values  

# Extract the last column and encode it as binary vectors
species = df['Species'].unique()
species_dict = {species[i]: i for i in range(len(species))}
labels = df['Species'].map(species_dict).values
Y = np.eye(len(species))[labels]
X_min = X.min(axis=0)  # Minimum of each column
X_max = X.max(axis=0)  # Maximum of each column
X = (X - X_min) / (X_max - X_min)
# X = 2 * X - 1
#print(X)

layer_sizes = [4,3]
activations = ["identity", "heaviside"]

n_samples = 20
random_indices = np.random.choice(X.shape[0], size=n_samples, replace=False)

X = X[random_indices]
Y = Y[random_indices]

G, layers, bias_nodes = construct_layered_graph(layer_sizes, activations)

#gen_model(G, layers, bias_nodes, X, Y)

for k, x in enumerate(X):
    y = forward_pass(G, layers, bias_nodes, x) """
    #print(f"y_{k}: {y}")

#exit(0)

# XOR Test

layer_sizes = [2,2,2]
activations = ["identity", "heaviside", "identity"]

G, layers, bias_nodes = construct_layered_graph(layer_sizes, activations)

X = np.array([[4.0, 2.0, 5.0], [2.0, 1.0, 3.0]])
Y = np.array([[1.0, 0.0], [0.0, 1.0]])

gen_model(G, layers, bias_nodes, X, Y)

for k, x in enumerate(X):
    y = forward_pass(G, layers, bias_nodes, x)
    print(f"y_{k}: {y}")

exit(0)
# 
# layer_sizes = [1, 1, 1]
# # activations = ["constant", "heaviside", "relu"]
# activations = ["identity", "identity", "identity"]
# 
# G, layers, bias_nodes = construct_layered_graph(layer_sizes, activations)
# 
# X = np.array([[0.0], [0.25], [0.5], [0.75]])
# Y = np.array([[0.0], [0.125], [0.25], [0.375]])
# 
# gen_model(G, layers, bias_nodes, X, Y)
# 
# for k, x in enumerate(X):
#     y = forward_pass(G, layers, bias_nodes, x)
#     print(f"y_{k}: {y}")
# 
# exit(0)


# Separation test
import matplotlib.pyplot as plt

random_vector = np.random.normal(size=2)  # Random vector in 2D
unit_vector = random_vector / np.linalg.norm(random_vector)  # Normalize it
X = np.random.uniform(low=-1, high=1, size=(40, 2))
Y = np.array([[1.0] if np.dot(point, unit_vector) > 0 else [0.0] for point in X])

layer_sizes = [len(X[0]), 1]
# activations = ["constant", "heaviside", "relu"]
activations = ["identity", "heaviside"]

G, layers, bias_nodes = construct_layered_graph(layer_sizes, activations)

w_vec = gen_model(G, layers, bias_nodes, X, Y)

slope = -w_vec[0] / w_vec[1]
intercept = -w_vec[2] / w_vec[1]



for k, x in enumerate(X):
    y = forward_pass(G, layers, bias_nodes, x)
    #print(f"y_{k}: {y}")


# plotting 

# Define the line parameters
# The line equation is determined by the vector's normal form
a, b = unit_vector  # Coefficients of the normal vector
c = 0  # The line passes through the origin (dot product = 0)

# Generate the line
x_vals = np.linspace(-1.0, 1.0, 500)  # x-values for plotting
y_vals = -(a / b) * x_vals  # Corresponding y-values

# Plot the points
X_above = X[np.dot(X, unit_vector) > 0]  # Points above the hyperplane
X_below = X[np.dot(X, unit_vector) <= 0]  # Points below the hyperplane

plt.figure(figsize=(8, 8))
plt.scatter(X_above[:, 0], X_above[:, 1], color='blue', label='Label: 1', alpha=0.7)
plt.scatter(X_below[:, 0], X_below[:, 1], color='red', label='Label: 0', alpha=0.7)

# Plot the line
plt.plot(x_vals, y_vals, color='black', label='Hyperplane', linewidth=2)
plt.plot(x_vals, slope * x_vals + intercept, color='orange', label='Hyperplane', linewidth=2)

# Set plot limits and labels
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
plt.title('Hyperplane and Labeled Points')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.legend()
plt.grid(alpha=0.3)
plt.plot()
