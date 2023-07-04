# --------------------------------------------------
#  Example 1: Simple Cournout Model
#  2023/06
#  mail@juliabarbosa.net
# -------------------------------------------------- 

using JuMP
using Plots
using PATHSolver

include("pathlic.jl")


# -- Data Structures --
struct EX1Input
    
    # Producer parameters   
    number_producers::Int
    marginal_cost::Float64

    # Demand parameters
    D0::Float64
    a::Float64

    # Constructor with default values
    EX1Input() = new(2, 20, 100.0, 1.0)

    # Constructor with all parameters
    EX1Input(number_producers::Int, marginal_cost::Number, D0::Number, a::Number) =
         new(number_producers, marginal_cost, D0, a)
end

struct EX1Output
    x::Array{Float64,1}
    D::Float64
    πel::Float64
    number_producers::Int
    marginal_cost::Float64
    D0::Float64
    a::Float64

    # Constructor from dictionary
    EX1Output(res::Dict, input::EX1Input) =
         new(res["x"], res["D"], res["πel"], input.number_producers, input.marginal_cost, input.D0, input.a)
end


# -- Model --
function run_model(input::EX1Input)

    number_producers = input.number_producers
    marginal_cost = input.marginal_cost
    D0 = input.D0
    a = input.a

    # Set
    P = 1:number_producers

    # Model
    m = Model(PATHSolver.Optimizer)

    @variable(m, x[P])
    @variable(m, πel)
    @variable(m, D)

    # Expressions for the complementarity conditions
    @expression(m, dLdx[p in P], πel - a*x[p] - marginal_cost)
    @expression(m, InverseDemandEq, a*(D0 - D) - πel)
    @expression(m, PowerBalanceEq, D - sum(x[_p] for _p in P))

    # Complementarity conditions:
    @constraint(m, dLdx ⟂ x)
    @constraint(m, InverseDemandEq ⟂ D)
    @constraint(m, PowerBalanceEq ⟂ πel)

    print(m)
    status = optimize!(m)

    res = Dict("x" => value.(x).data, "D" => value(D), "πel" => value(πel))
    
    return EX1Output(res, input)
end


# -- Plotting --
function plot_energy(res::EX1Output)

    p = bar(res.x, title = "Production per Agent", label = "x",  xticks = (1:res.number_producers, 1:res.number_producers), yaxis = "Production")
    
    hline!([res.D0], label = "D0")
    hline!([res.D], label = "D")

    return p
end

function plot_prices(res::EX1Output)
    data = [res.marginal_cost, res.πel]
    p = bar(data, title = "Prices", legend=false, xticks = (1:2, ["marginal cost", "πel"]), yaxis = "Price")

    return p
end