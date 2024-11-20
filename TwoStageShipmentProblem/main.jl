include("RFPrescription.jl")
include("Simulation.jl")

const Prescriptor = RfPrescModule.RfPrescriptor
const Simulator = SimModule.Simulator

const solve_cso_problem = RfPrescModule.solve_cso_problem
const solve_saa_problem = RfPrescModule.solve_saa_problem

function _run_model(
    simulator::Simulator,
    x_init, x_alt
)
    prescriptor = Prescriptor(simulator.X_train, sim.Y_train, 100, 2, nothing, false)

    z_opt = solve_cso_problem(
        simulator, x_init, prescriptor
    )

    z_alt = solve_cso_problem(
        simulator, x_alt, prescriptor
    )

    println("Optimal cost: ", z_opt)
    println("Alternative cost: ", z_alt)

end

function _start_simulator()
    
end
