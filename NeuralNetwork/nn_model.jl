using JuMP
using GLPK
using LinearAlgebra

n_features = 5    # Número de entradas (features)
n_samples = 10    # Número de exemplos no conjunto de dados
n_output_neurons = 2  # Número de neurônios na camada de saída
x = rand(n_samples, n_features)  # Matriz de entradas (amostras x features)
y = rand(n_samples)  # Vetor de valores esperados (alvo)
M = 1e3  # Constante grande para restrições de ativação

model = Model(GLPK.Optimizer)

# Variáveis
@variable(model, W1[1:n_features])  # Pesos da primeira camada (neuron 1)
@variable(model, W2[1:n_features])  # Pesos da primeira camada (neuron 2)

@variable(model, b1)                # Bias da primeira camada (neuron 1)
@variable(model, b2)                # Bias da primeira camada (neuron 2)

@variable(model, W_out[1:n_output_neurons])  # Pesos da camada de saída
@variable(model, b_out)             # Bias da camada de saída

@variable(model, v1[1:n_samples] >= 0)  # Variável de ativação neuron 1
@variable(model, v2[1:n_samples] >= 0)  # Variável de ativação neuron 2

@variable(model, π1[1:n_samples], Bin)  # Variável binária neuron 1
@variable(model, π2[1:n_samples], Bin)  # Variável binária neuron 2

@variable(model, ŷ[1:n_samples])  # Predição da rede para cada amostra

# Definir a saída como função das ativações
@constraint(model, [i=1:n_samples], ŷ[i] == W_out[1] * v1[i] + W_out[2] * v2[i] + b_out)

# Restrições de ativação para o neuron 1
@constraint(model, [i=1:n_samples], v1[i] >= dot(W1, x[i, :]) + b1)
@constraint(model, [i=1:n_samples], v1[i] <= dot(W1, x[i, :]) + b1 + M * (1 - π1[i]))
@constraint(model, [i=1:n_samples], v1[i] <= M * π1[i])

# Restrições de ativação para o neuron 2
@constraint(model, [i=1:n_samples], v2[i] >= dot(W2, x[i, :]) + b2)
@constraint(model, [i=1:n_samples], v2[i] <= dot(W2, x[i, :]) + b2 + M * (1 - π2[i]))
@constraint(model, [i=1:n_samples], v2[i] <= M * π2[i])

# Objetivo: minimizar a soma dos erros quadráticos
@objective(model, Min, sum((y[i] - ŷ[i])^2 for i in 1:n_samples))

optimize!(model)

# Resultados
if termination_status(model) == MOI.OPTIMAL
    println("Valor ótimo: ", objective_value(model))
    println("Pesos primeira camada: ", value.(W1), value.(W2))
    println("Bias primeira camada: ", value(b1), value(b2))
    println("Pesos saída: ", value.(W_out), value(b_out))
else
    println("Solução não encontrada!")
end