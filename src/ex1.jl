# --------------------------------------------------
#  Example 1: Simple Cournout Model
#  2023/06
#  mail@juliabarbosa.net
# -------------------------------------------------- 

using JuMP, Complementarity
using Plots

struct EX1Input
    """
    Input parameters for the model 
    """
    number_producers::Int
    marginal_cost::Float64
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
    d::Float64
    πel::Float64
    number_producers::Int
    marginal_cost::Float64
    D0::Float64
    a::Float64

    # Constructor from dictionary
    EX1Output(res::Dict, input::EX1Input) =
         new(res["x"], res["d"], res["πel"], input.number_producers, input.marginal_cost, input.D0, input.a)
end

function run_model(input::EX1Input)

    number_producers = input.number_producers
    marginal_cost = input.marginal_cost
    D0 = input.D0
    a = input.a

    # Set
    P = 1:number_producers

    # Model
    m = MCPModel()

    @variable(m, x[P])
    @variable(m, πel)
    @variable(m, d)

    # Expressions for the complementarity conditions
    @mapping(m, F1[p in P], a*(D0 - d) - a*x[p] - marginal_cost)
    @mapping(m, F2, a*(D0 - d) - πel)
    @mapping(m, F3, d - sum(x[_p] for _p in P))

    # Complementarity conditions, e.g. 0 <= F1 ⟂ x >= 0
    @complementarity(m, F1, x)
    @complementarity(m, F2, d)
    @complementarity(m, F3, πel)


    print(m)
    # Solve
    status = solveMCP(m, output="no")

    # Return dictionary with results
    res = Dict()
    merge!(res, Dict("x" => collect(result_value.(x))))
    merge!(res, Dict("d" => result_value(d)))
    merge!(res, Dict("πel" => result_value(πel)))

    return EX1Output(res, input)
end

function plot_energy(res::EX1Output)

    p = bar(res.x, title = "Production per Agent", label = "x",  xticks = (1:res.number_producers, 1:res.number_producers), yaxis = "Production")
    
    hline!([res.D0], label = "D0")
    hline!([res.d], label = "d")

    return p
end

function plot_prices(res::EX1Output)
    data = [res.marginal_cost, res.πel]
    p = bar(data, title = "Prices", legend=false, xticks = (1:2, ["marginal cost", "πel"]), yaxis = "Price")

    return p
end



