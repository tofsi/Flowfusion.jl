


using Pkg

Pkg.activate(".")
using Revise
#Pkg.develop(path="../../ForwardBackward/")
#Pkg.develop(path="../")
using ForwardBackward, Flowfusion, NNlib, Flux, RandomFeatureMaps, Optimisers, Plots, ProgressMeter
#include("../src/counting.jl")   # ← your updated process file defining CountingFlow, floss_R, etc.
# ------------------------------
# Model: predicts residual R̂ = X̂₁ - Xₜ
# ------------------------------
struct RModel{A}
    layers::A
end
Flux.@layer RModel

function RModel(; embeddim=64, spacedim=2, layers=3)
    embed_time = Chain(RandomFourierFeatures(1 => embeddim, 1f0),
        Dense(embeddim => embeddim, swish))
    embed_state = Chain(Dense(spacedim => embeddim, swish))
    ffs = [Dense(embeddim => embeddim, swish) for _ in 1:layers]
    decode = Dense(embeddim => spacedim)
    layers = (; embed_time, embed_state, ffs, decode)
    RModel(layers)
end

function (f::RModel)(t, Xt)
    l = f.layers
    tXt = tensor(Xt)
    tv = zero(tXt[1:1, :]) .+ expand(t, ndims(tXt))
    x = l.embed_time(tv) .+ l.embed_state(tXt)
    for ff in l.ffs
        x = x .+ ff(x)
    end
    l.decode(x)  # returns predicted residual R̂
end

# ------------------------------
# Data and Process
# ------------------------------
# Parameters
T = Float32
n_samples = 400
k = 500  # minimum offset to keep X1 > X0 >= 0

sampleX0(n) = rand(0:k, 2, n)
sampleX1(n) = (Flowfusion.random_discrete_cat(n; d=32, lo=-2.5, hi=2.5) .+ k)

# Hazard CDF F(t): monotone [0,1]
F(t) = t^2
P = CountingFlow(F)

# ------------------------------
# Model / Optimizer
# ------------------------------
model = RModel(embeddim=128, spacedim=2, layers=3)
eta = 0.001
opt_state = Flux.setup(AdamW(eta=eta), model)

# ------------------------------
# Training loop
# ------------------------------
iters = 10000
@showprogress for i in 1:iters
    X0 = sampleX0(n_samples)
    X1 = sampleX1(n_samples)
    t = rand(T, n_samples)
    Xt = bridge(P, X0, X1, t)

    l, g = Flux.withgradient(model) do m
        R̂ₜ = m(t, Xt)
        floss(P, R̂ₜ, X1, Xt, 1.0)
    end

    Flux.update!(opt_state, model, g[1])
    if i % 50 == 0
        println("Iter: $i, Loss: $l")
    end
end

# Sampling
n_inference_samples = 1000
X0 = sampleX0(n_inference_samples)
steps = 0f0:0.05f0:1f0   # smaller step for smoother lines (optional)

using ProgressMeter
p = Progress(length(steps) - 1; desc="Sampling", dt=0.2)

# Collector that stores times and states
mutable struct TrajCollector
    t::Vector{Float32}
    xt::Vector{Matrix{Int}}
end
collector() = TrajCollector(Float32[], Matrix{Int}[])

function collect!(C::TrajCollector, t, Xt, _Rhat)
    push!(C.t, Float32(t))
    push!(C.xt, Int.(Xt))
end

C = collector()
tracker = (t, Xt, Rhat) -> begin
    collect!(C, t, Xt, Rhat)
    next!(p; showvalues=[(:t, t)])
end

@time samp = gen(P, X0, model, steps; tracker=tracker)

# Visualization
using Plots
pl = scatter(X0[1, :], X0[2, :], color=:blue, alpha=0.5, label="Initial", size=(400, 400))
scatter!(pl, samp[1, :], samp[2, :], color=:green, alpha=0.4, label="Sampled")

# Draw a few trajectories
for j in 1:min(5, size(X0, 2))  # plot up to 5 paths
    xs = [C.xt[k][1, j] for k in 1:length(C.t)]
    ys = [C.xt[k][2, j] for k in 1:length(C.t)]
    plot!(pl, xs, ys, color=:red, alpha=0.3, label=nothing)
end

plot!(pl, [-10], [-10], color=:red, label="Trajectory")  # legend handle
display(pl)
savefig(pl, "countingflow.svg")
