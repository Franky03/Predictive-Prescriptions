include("src/RFPrescription.jl")
include("src/Simulation.jl")
include("src/Shipment.jl")

const RfPrescriptor = RfPrescModule.RfPrescriptor
const Simulator = SimModule.Simulator

const solve_cso_problem = RfPrescModule.solve_cso_problem
const solve_saa_problem = RfPrescModule.solve_saa_problem

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

    z_alt = solve_cso_problem(
        simulator, prescriptor, x_alt
    )

    open("z_opt.txt", "w") do io
        println(io, z_opt)
    end

    open("z_alt.txt", "w") do io
        println(io, z_alt)
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

    