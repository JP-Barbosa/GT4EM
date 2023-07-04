# --------------------------------------------------
#  Example 4 - Multileader EPEC
#  2023/06
#  julia.barbosa@eins.tu-darmstadt.de
# -------------------------------------------------- 

# -- Data Structures --
using JuMP, Complementarity, Ipopt
using Plots, StatsPlots

using PATHSolver 
include("pathlic.jl")


""" 
Assume two producers A and B, with zero marginal cost.
Producer A is located at node 1 and producer B at node 2.
The demand at each node n is Dₙ.
The inverse demand function is πₙ = Dₙ - aₙ*Dₙ, where aₙ is the price πₙ elasticity.

Both regions are connected by a transmission line with a capacity of C.
The TSO maximize profit, by buying sₙ energy from node n and selling sₙ'  to node n'.

This is a multileader problem, where the producers are the leaders and the TSO and price-taker consumers are followers.

Adapted from: Complementarity Modeling in Energy Markets, by Gabriel A., Conejo A., Fuller J., Hobbs B. and Ruiz C. (2013). Chapter 7.4.2
"""

# -- Data Structures --

struct EX4Input
   N::Array{Int64,1}
   a:: Array{Float64,1}
   D0::Array{Float64,1}
   capacity::Float64
end

struct LeadersInput
   N::Array{Int64,1}
   production_belif::Array{Float64,1} 
   
   # Consumer Params
   D0::Array{Float64,1}
   a::Array{Float64,1}

   # Transmission Params
   capacity::Float64

   function LeadersInput(N, production_belif, D0, a, capacity)
      #println("Creating LeadersInput with belief $production_belif")
      return new(N, production_belif, D0, a, capacity)
   end

end

mutable struct LeadersOutput
   p::Array{Float64,1}
   πel::Array{Float64,1}
   s::Array{Float64,1}
   D::Array{Float64,1}
   γ::Float64
   λ::Array{Float64,1}
   
   input::LeadersInput

   status::Symbol
   
   function LeadersOutput(p, πel, s, D, γ, λ, input::LeadersInput)
      return new(p, πel, s, D, γ, λ, input, :not_solved)
   end

   # from dict 
   function LeadersOutput(inputs:: Dict, input::LeadersInput)
      return new(inputs[:p], inputs[:πel], inputs[:s], inputs[:D], inputs[:γ], inputs[:λ], input)
   end

   function print(self::LeadersOutput) 
      println("LeadersOutput with p = $(self.p) \n πel = $(self.πel) \n s = $(self.s) \n D = $(self.D) \n γ = $(self.γ) \n λ = $(self.λ)") 
   end
end


# -- Model --
function solver_leader(data::LeadersInput, l::Int)
    mpcc = Model(Ipopt.Optimizer)

    N = data.N
    a = data.a
    D0= data.D0
    cap = data.capacity
    belief =  data.production_belif
    PTDF = [1 0; -1 0]
   
    _L = [i for i ∈ N if i != l]
    direction = [1, 2]
   # Leader Variable
   @variable(mpcc, p[N] >= 0) 

   
   # Follower Variables 
   @variable(mpcc, s[N])
   @variable(mpcc, D[N])
   @variable(mpcc, γ)
   @variable(mpcc, πel[N])
   @variable(mpcc, λ[N] >= 0)
   
   # Model
   @objective(mpcc, Max, πel[l]*p[l])

   # fix every other leader production to the belief
   @constraint(mpcc, p[_L] .== belief[_L])
   

   # Expressions  
   @expression(mpcc, invDemand[i in N], a[i]*(D0[i] - D[i]) - πel[i]) 
   @expression(mpcc, transmissionBalanceEq, s[1] + s[2])

   @expression(mpcc, dLds[i ∈ N], πel[i] - γ - sum(λ[i_]*PTDF[i_, i] for i_ ∈ N))

   @expression(mpcc, maxCapacity[d ∈ direction], sum(s[i]*PTDF[d,i] for i ∈ N) - cap)

   @expression(mpcc, nodePowerBalance[i ∈ N], D[i] - s[i] - p[i])

   @constraint(mpcc, invDemand .== 0 )
   @constraint(mpcc, transmissionBalanceEq == 0)
   @constraint(mpcc, dLds .== 0)

   for i ∈ N
      @complements(mpcc, 0 ≥ maxCapacity[i], λ[i] ≥ 0)
   end

   @constraint(mpcc, nodePowerBalance .== 0)

   # Solve
   #print(mpcc)
   set_silent(mpcc)
   optimize!(mpcc)

   # Generate output
   res = Dict(
      :p => value.(p).data,
      :πel => value.(πel).data,
      :s => value.(s).data,
      :D => value.(D).data,
      :γ => value(γ),
      :λ => value.(λ).data
      )


   output = LeadersOutput(res, data)

   return output
end


function norm(x::AbstractArray, y::AbstractArray)
   return sqrt(sum((x[i] - y[i])^2 for i ∈ 1:length(x)))
end


function run_model(input::EX4Input; ϵ = 1e-1, max_iter = 20, p0 = [0.0, 0])
   """Solves the EPEC problem using Diagonalization."""
   
   # Extract data
   N = input.N
   D0 = input.D0
   a = input.a
   cap = input.capacity
   
   # Initialize counter ...
   global iter = 0
   
   # Main Loop
   print("Staring Diagonalization...  \n")
   p = p0

   sol = :nothing

   while iter < max_iter
      global iter += 1
      
      p_old = p[:]
      
      println("   $iter  - Production: $(round.(p, digits=2))")
      
      # Calculate new production for each leader ..
      for leader ∈ N
         input = LeadersInput(N, p, D0, a, cap)
         sol = solver_leader(input, leader)
         p[leader] = sol.p[leader]
      end

      # Check convergence...
      if norm(p_old, p) < ϵ
         println("Converged in $iter iterations")
         sol.status = :converged
         return sol
      end
   end
   
   # raise Warning if not converged
   @warn "Model DID NOT converge in $iter iterations"

   #println("Model DID NOT converge in $iter iterations")
   sol.status = :not_converged
   return sol
   
end 



# -- Plotting functions -- 
function plot_best_response(input:: EX4Input)
   """Plots best response curves, considering leaders and market simetric"""


   N = input.N
   D0 = input.D0
   a = input.a
   cap = input.capacity

   # Compute best response
   best_response = []
   l_productions = [250.0:5:800.0;]
   for l_production ∈ l_productions
      ii = LeadersInput(N, [0.0, l_production], D0, a, cap)
      sol = solver_leader(ii, 1)
      push!(best_response, sol.p[1])
      #println("Leader 1 production: $sol \n")
   end
   
   # Make plot
   scatter(l_productions, best_response; markersize=1.5, label="P₁(P₂)", xlabel="Leader 2 Production (P₂)[MW]", ylabel="Leader 1 Production (P₁) [MW]", markercolor=:black, legend=:topleft)
   xaxis!([300, 800])
   yaxis!([300, 800])
   display(scatter!(best_response, l_productions, markershape=:hline, markersize = 2, label="P₂(P₁)"))

end 

function plot_production(out::LeadersOutput)
   # Check status 
   plotnote = ""
   if out.status != :converged
      println("Model did not converge!! Results are not NE.")
      plotnote = "Model did not converge!! Results are not NE."
   end

   # Plot production
   print(out)
   p = out.p
   s = out.s
   d = out.D
   
   nam = ["Node 1", "Node 2"]
   
   groupedbar(nam, [p s d], bar_position = :dodge, 
   bar_width = 0.5, label = ["Production" "Transmission" "Demand"]
   )
   ylabel!("Power [MW]")
   # add plot note 
   if out.status != :converged
      @warn "Model did not converge!! Results are not NE."
      plotnote = "Model did not converge!! Results are not NE."
      annotate!(0.5, 0.5, text(plotnote, 7), linecolor = :red)
   end

end

function plot_prices(out::LeadersOutput)
   # Plot price
   ticks = ["Node 1", "Node 2"]
   bar(out.πel,  lw = 2,  ylabel = "Electrity Price [€/MWh]", label=false)
   xticks!(1:2, ticks)
end





















