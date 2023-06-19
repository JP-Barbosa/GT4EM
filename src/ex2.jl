# --------------------------------------------------
#  Example 2: Cournot Model with emissions Trading
#  2023/06
#  mail@juliabarbosa.net
# -------------------------------------------------- 

using JuMP
using Plots

using PATHSolver
include("pathlic.jl")

# -- Data Structures --
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

# -- Model --
function run_model(data::EX2Input)

   a = data.a
   D0 = data.D0
   emission_budget = data.emission_budget
   emission_intensity = data.emission_intensity
   marginal_cost = data.marginal_cost


   # Sets
   P = 1:data.number_producers

   # Model
   m = Model(PATHSolver.Optimizer)

   
   @variable(m, x[P]) # Production
   @variable(m, πel)  # Electricity Price
   @variable(m, D)    # Demand


   @variable(m, Φ >= 0) # Allowance Price --> D.V of inequality constraint > 0

   # Declare Expressions for the complementarity conditions ..
   @expression(m, dLdx[p ∈ P], πel - a*x[p] - marginal_cost[p] - emission_intensity[p]*Φ)
   @expression(m, EmissionsLimit, -sum(emission_intensity[p_]*x[p_] for p_ ∈ P) + emission_budget)
   @expression(m, InverseDemandFun, a*(D0-D) - πel)
   @expression(m, PowerBalance, D - sum(x[p_] for p_ ∈ P))

   # Declare the complementarity conditions ..
   @constraint(m, dLdx ⟂ x)
   @constraint(m, PowerBalance ⟂ πel)
   @constraint(m, InverseDemandFun  ⟂ D)
   @constraint(m, EmissionsLimit ⟂ Φ)


   print(m)
   # Solve the model ..
   status = optimize!(m)

   res = Dict(
      "production" => value.(x).data,
      "allowance_price" => value(Φ),
      "electricity_price" => value(πel),
      "demand" => value(D)
      )
   

   return EX2Output(res, data)

end


# -- Plotting --  
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