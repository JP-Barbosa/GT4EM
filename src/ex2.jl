# --------------------------------------------------
#  Example 2: Cournot Model with emissions Trading
#  2023/06
#  mail@juliabarbosa.net
# -------------------------------------------------- 

using JuMP, Complementarity
using Plots

struct EX2Input

   number_producers::Int
   marginal_cost::Array{Float64,1}
   emission_intensity::Array{Float64,1}
   emission_budget::Float64
   D0::Float64
   a::Float64
   producers::Array{String,1}


   # Constructor with sanity checks
   function EX2Input(number_producers::Int, marginal_cost::Array{Float64,1}, emission_intensity::Array{Float64,1}, emission_budget::Float64, D0::Float64, a::Float64, producers::Array{String,1})
      
      if length(producers) != number_producers
           throw(ArgumentError("producers must have length equal to number_producers"))
       end

      if length(marginal_cost) != number_producers
           throw(ArgumentError("marginal_cost must have length equal to number_producers"))
         end
      if length(emission_intensity) != number_producers
           throw(ArgumentError("emission_intensity must have length equal to number_producers"))
       end
      new(number_producers, marginal_cost, emission_intensity,emission_budget, D0, a, producers)
   end

   # Constructor for numbers only 
   function EX2Input(number_producers::Int, marginal_cost::Array{Float64,1}, emission_intensity::Array{Float64,1}, emission_budget:: Float64, D0::Float64, a::Float64)
      producers = ["Producer_$i" for i in 1:number_producers]
      return EX2Input(number_producers, marginal_cost, emission_intensity, emission_budget, D0, a, producers)
   end 

   # Construction with producer names
   function EX2Input(producers::Array{String,1}, marginal_cost::Array{Float64,1}, emission_intensity::Array{Float64,1}, emission_budget::Float64,  D0::Float64, a::Float64)
      number_producers = length(producers)
      return EX2Input(number_producers, marginal_cost, emission_intensity,emission_budget, D0, a, producers)
   end

 end


struct EX2Output
   production :: Array{Float64,1}
   allowance_price::Number
   electricity_price::Number
   demand::Number
   input::EX2Input

   # Constructor from dictionary
   EX2Output(res::Dict, input::EX2Input) =
        new(res["production"], res["allowance_price"], res["electricity_price"], res["demand"], input)
 end

function run_model(data::EX2Input)

   a = data.a
   D0 = data.D0
   emission_budget = data.emission_budget
   emission_intensity = data.emission_intensity
   marginal_cost = data.marginal_cost


   # Sets
   P = 1:data.number_producers

   # Model
   m = MCPModel()

   # Declare variables ..
   @variable(m, x[P]) # Production
   @variable(m, πel)  # Electricity Price
   @variable(m, d)    # Demand

   # Positive variables ( Dual variables of inequality constraints must be positive)
   @variable(m, Φ >= 0) # Allowance Price 

   # Declare Expressions for the complementarity conditions ..
   @mapping(m, dLdx[p ∈ P], -a*(D0-d) + a*x[p] + marginal_cost[p] + emission_intensity[p]*Φ)
   @mapping(m, EmissionsLimit, -sum(emission_intensity[p_]*x[p_] for p_ ∈ P) + emission_budget)
   
   #@mapping(m, dLdx[p ∈ P], a*(D0-d) - a*x[p] - marginal_cost[p])
   @mapping(m, InverseDemandFun, a*(D0-d) - πel)
   @mapping(m, PowerBalance, d - sum(x[p_] for p_ ∈ P))

   # Declare the complementarity conditions ..
   @complementarity(m, dLdx, x)
   @complementarity(m, PowerBalance, πel)
   @complementarity(m, InverseDemandFun, d)
   @complementarity(m, EmissionsLimit, Φ)


   print(m)
   # Solve the model ..
   status = solveMCP(m, output="no")

   res = Dict()
   merge!(res, Dict("production" => collect(result_value.(x))))
   merge!(res, Dict("allowance_price" => result_value(Φ)))
   merge!(res, Dict("electricity_price" => result_value(πel)))
   merge!(res, Dict("demand" => result_value(d)))
   

   return EX2Output(res, data)

   end



   
function plot_energy(res::EX2Output)
   x = res.production
   p = bar(res.input.producers, res.production)
   hline!([res.demand], label = "D")
   hline!([res.input.D0], label = "D0")
   return p
   end

function plot_prices(res::EX2Output)
   p = bar(res.input.producers, res.input.marginal_cost, label = "Marginal Cost")
   hline!([res.electricity_price], label = "Electricity Price")
   hline!([res.allowance_price], label = "Allowance Price")
   return p
   end