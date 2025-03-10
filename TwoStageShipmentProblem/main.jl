include("src/RFPrescription.jl")
include("src/Simulation.jl")
include("src/Shipment.jl")

using JSON

const RfPrescriptor = RfPrescModule.RfPrescriptor
const Simulator = SimModule.Simulator

const solve_cso_problem = RfPrescModule.solve_cso_problem
const solve_saa_problem = RfPrescModule.solve_saa_problem

function print_results(prod_obj, last_minute_obj, ship_obj, io)
    println(io,"===== Resultados da Otimização =====\n")
    
    println(io,"Produção Antecipada:")
    for (warehouse, qty) in sort(collect(prod_obj))
        println(io,"  Armazém $(warehouse): $(qty) unidades")
    end
    
    println(io,"\nProdução de Última Hora (por cenário):")
    for ((warehouse, scenario), qty) in sort(collect(last_minute_obj))
        println(io,"  Armazém $(warehouse), Cenário $(scenario): $(qty) unidades")
    end
    
    println(io,"\nEnvios (por cenário):")
    for ((warehouse, location, scenario), qty) in sort(collect(ship_obj))
        println(io,"  Armazém $(warehouse) -> Local $(location), Cenário $(scenario): $(qty) unidades")
    end
end

function results_to_json(prod_obj, last_minute_obj, ship_obj)
    # Converte o dicionário de produção para ter chaves em string
    prod_json = Dict{String, Any}()
    for (warehouse, qty) in prod_obj
        prod_json[string(warehouse)] = qty
    end

    # Converte o dicionário de produção de última hora, criando chaves "warehouse-scenario"
    last_minute_json = Dict{String, Any}()
    for ((warehouse, scenario), qty) in last_minute_obj
        key = "$(warehouse)-$(scenario)"
        last_minute_json[key] = qty
    end

    # Converte o dicionário de envios, criando chaves "warehouse-location-scenario"
    shipping_json = Dict{String, Any}()
    for ((warehouse, location, scenario), qty) in ship_obj
        key = "$(warehouse)-$(location)-$(scenario)"
        shipping_json[key] = qty
    end

    # Monta o objeto final com as três seções
    results = Dict(
        "production" => prod_json,
        "last_minute" => last_minute_json,
        "shipping" => shipping_json
    )
    
    return JSON.json(results)
end

function _run_model(
    simulator::Simulator,
    x_init, x_alt
)
    X_train, Y_train = simulator.X_train, simulator.Y_train
    prescriptor = RfPrescriptor(X_train, Y_train, 100, 2, nothing, false)

    @assert typeof(simulator) == Simulator "Simulator type is incorrect, expected $(Simulator), got $(typeof(simulator))"

    z_opt = solve_cso_problem(
        simulator, prescriptor, x_init
    )

    open("z_opt.txt", "w") do io
        println(io, z_opt)
    end

    open("result_final.txt", "w") do io
        print_results(z_opt..., io)        
    end

    open("result_final.json", "w") do io
        println(io, results_to_json(z_opt...))
    end

end

function _start_simulator()
    args = Dict(ARGS[i] => ARGS[i+1] for i in 1:2:length(ARGS))
    size = parse(Int, get(args, "--size", "100")) 
    verbose = parse(Bool, get(args, "--verbose", "false"))
    simulator = Simulator(size, verbose)
    
    _run_model(simulator, simulator.X_train, simulator.X_train)
    
end


_start_simulator()

    