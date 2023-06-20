# --------------------------------------------------
#  Example 3: Stackelberg Model 
#  2023/06
#  mail@juliabarbosa.net
# -------------------------------------------------- 

# -- Packages --
using JuMP, Ipopt, Complementarity

using PATHSolver
PATHSolver.c_api_License_SetString("2830898829&Courtesy&&&USR&45321&5_1_2021&1000&PATH&GEN&31_12_2025&0_0_0&6000&0_0<LICENSE STRING>")

# -- Data Structures --
struct EX3Input

   # Producer parameters
   producers::Array{String,1}             # Producer names
   number_producers::Int                  # Number of producers
   leader:: Int                           # Index of the leader
   marginal_cost::Array{Float64,1}        # Marginal cost of each producer
   max_capacity::AbstractArray{Float64,1}         # Maximum capacity of each producer
   
   # Consumer parameters
   D0::Float64
   a::Float64

   # Constructor with sanity checks
   function EX3Input(producers::Array{String,1}, number_producers:: Int, leader::Int, marginal_cost::Array{Float64,1}, max_capacity::AbstractArray{Float64, 1} , D0::Float64, a::Float64)

      # Check consistency producer and number_producers
      if length(producers) != number_producers
         throw(ArgumentError("producers must have length equal to number_producers"))
      end
      # Check consistency leader index valid
      if leader > number_producers
         throw(ArgumentError("leader index must be less than number_producers"))
      end
      # Check consistency marginal cost for each producer
      if length(marginal_cost) != number_producers
         throw(ArgumentError("marginal_cost must have length equal to number_producers"))
      end
      # Check consistency max capacity for each producer
      if length(max_capacity) != number_producers
         throw(ArgumentError("max_capacity must have length equal to number_producers"))
      end

      new(producers, number_producers, leader, marginal_cost, max_capacity, D0, a)

   end

   # Constructor for numbers only
   function EX3Input(number_producers:: Int, leader::Int, marginal_cost::AbstractArray{Float64,1}, max_capacity::AbstractArray{Float64, 1} ,  D0::Float64, a::Float64)
      producers = ["Producer_$i" for i in 1:number_producers]
      return EX3Input(producers, number_producers, leader, marginal_cost, max_capacity, D0, a)
   end

end 

struct EX3Output
   x::Array{Float64,1}
   D::Float64
   πel::Float64
   input:: EX3Input

   function EX3Output(x::Array{Float64,1}, D::Float64, πel::Float64, input:: EX3Input)
      new(x, D, πel, input)
   end

   # Constructor from Dict
   function EX3Output(dict::Dict, input:: EX3Input)
      new(dict["x"], dict["D"], dict["πel"], input)
   end
   
end


# -- Functions --
function run_model(data::EX3Input)
   """ 
   Run Stackelberg model.
   This Constitutes an Mathematical Programming Problem with Complementarity Constraints (MPCC)
   also kwon as Mathematical Program with Equilibrium Constraints (MPEC). 
   In this case equality constrains do not have to matched to a varaible, only inequality constraints 
   of the followers are matched to a complementarity condition.
   """

   # Sets 
   P = 1:data.number_producers
   l = data.leader
   F = [p for p in P if p!=l] # followers
   # Model 
   m = Model(Ipopt.Optimizer)
   max_cap = data.max_capacity

   @variable(m, x[P])
   @variable(m,πel)
   @variable(m,D)

   # Dual variables 
   @variable(m, δ[F] >= 0)  # capacity constraint

   # Leader's objective
   @objective(m, Max, πel*x[l]- data.marginal_cost[l]*x[l])
   @constraint(m, x[l] - max_cap[l] <= 0)

   # Followers Cournout model
   @constraint(m, dLdx[p ∈ F], πel - data.a*x[p] - data.marginal_cost[p] - δ[p] == 0)
   
   for p ∈ F
      @complements(m, 0 <= max_cap[p] - x[p], δ[p] >= 0)
   end
  
   # Systems constraints
   @expression(m, InverseDemandEq, data.a*(data.D0 - D) - πel)
   @constraint(m, InverseDemandEq ==0)

   @expression(m, PowerBalanceEq, D - sum(x[_p] for _p in P))
   @constraint(m, PowerBalanceEq ==0)
   

   print(m)

   # Solve
   status = optimize!(m)

   # Return results
   return EX3Output(Dict("x" => value.(x).data, "D" => value(D), "πel" => value(πel)), data)
   

end

#  -- Plotting --
function plot_production(data::EX3Output)
   """ 
   Plot production of each producer.
   """

   bar(data.x, label = "Production",  ylabel = "Production")
   
   # Change xticks to producer names and mark leader
   mark_leader(data)

   scatter!(data.input.max_capacity, label = "Capacity")
   hline!([data.input.D0], label = "D0")
   hline!([data.D], label = "D")

end


function plot_prices(data::EX3Output)

   bar(data.input.marginal_cost, label = "Marginal Cost")
   mark_leader(data)
   hline!([data.πel], label = "πel")

end

function mark_leader(data::EX3Output)
   """ 
   Mark leader producer in plot.
   """
   producers = data.input.producers
   producers[data.input.leader] = "$(producers[data.input.leader])(L)"
   xticks!(1:data.input.number_producers, producers)
end




