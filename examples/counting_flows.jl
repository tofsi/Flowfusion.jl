


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
k = 20  # minimum offset to keep X1 > X0 >= 0

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
iters = 200
@showprogress for i in 1:iters
    X0 = DiscreteState(32, round.(Int, sampleX0(n_samples)))
    X1 = DiscreteState(32, round.(Int, sampleX1(n_samples)))
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

# ------------------------------
# Sampling
# ------------------------------
n_inference_samples = 1000
X0 = DiscreteState(32, round.(Int, sampleX0(n_inference_samples)))
steps = 0f0:0.01f0:1f0
paths = Tracker()

@time samp = gen(P, X0, model, steps; tracker=paths)

# ------------------------------
# Visualization
# ------------------------------
pl = scatter(tensor(X0)[1, :], tensor(X0)[2, :],
    color="blue", alpha=0.5, label="Initial", size=(400, 400))
scatter!(tensor(samp)[1, :], tensor(samp)[2, :],
    color="green", alpha=0.2, label="Sampled")

tvec = stack_tracker(paths, :t)
xttraj = stack_tracker(paths, :xt)
for i in 1:100:n_inference_samples
    plot!(xttraj[1, i, :], xttraj[2, i, :], color="red", alpha=0.1, label=:none)
end
plot!(pl, [-10], [-10], color="red", label="Trajectory", alpha=0.4)
pl
savefig("countingflow.svg")
















































