# BirthDeathFlow Example
using Pkg

Pkg.activate(".")
using Revise
using ForwardBackward
using Flowfusion
using Flux
using RandomFeatureMaps
using Optimisers
using NNlib
using ProgressMeter
using Plots
using Statistics

# --------------------------------------------------------------------
# State wrapper for counting process
# --------------------------------------------------------------------

# --------------------------------------------------------------------
# Model: predicts residual R̂ = X̂₁ - Xₜ  (in X-space, 2D)
# --------------------------------------------------------------------

struct RModel{A}
    layers::A
    state_dim::Int
    out_dim::Int
end

Flux.@layer RModel

function RModel(; embeddim=64, state_dim=4, out_dim=4, layers=5)
    embed_time = Chain(
        RandomFourierFeatures(1 => embeddim, 1f0),
        Dense(embeddim => embeddim, swish),
    )
    embed_state = Chain(Dense(state_dim => embeddim, swish))
    ffs = [Dense(embeddim => embeddim, swish) for _ in 1:layers]
    decode = Dense(embeddim => out_dim, softplus)
    layers = (; embed_time, embed_state, ffs, decode)
    RModel(layers, state_dim, out_dim)
end

# Core call: works on flat state tensors (state_dim × batch)
function (f::RModel)(t, Xt::AbstractArray)
    l = f.layers
    tXt = tensor(Xt)                          # (state_dim, batch)
    tv = zero(tXt[1:1, :]) .+ expand(t, ndims(tXt))  # (1, batch)
    x = l.embed_time(tv) .+ l.embed_state(tXt)
    for ff in l.ffs
        x = x .+ ff(x)
    end
    l.decode(x)  # returns predicted residual R̂ (out_dim × batch)
end

# Convenience adapter: model over CountingState
(model::RModel)(t, Zt::CountingState) = begin
    Z = Zt.state                       # 2×2×N
    batch_size = size(Z, 3)
    Zflat = reshape(Z, 4, batch_size)  # state_dim = 4
    reshape(model(t, Zflat), 2, 2, batch_size)                   # returns R̂ₜ :: 2×N
end

# --------------------------------------------------------------------
# Data and Process
# --------------------------------------------------------------------
# One incrementing and one decrementing process over Z ∈ ℕ^{2×2×N}
# X ∈ ℕ^{2×N} is the observed sum: X = Z¹ + Z²

T = Float32
n_samples = 400
k = 25  # offset to keep X1 reasonably above Z0 (and X0)

# Sampling scheme: Z₀ uniform over a small hypercube
function sampleZ0(n, k)
    Z0_arr = rand(1:k, 2, 2, n)    # 2 components × 2 dims × batch
    CountingState(Z0_arr)
end

# Target distribution: 2D categorical, shifted by k to ensure X1 > Z0[1,:,:]
sampleX1(n) = Flowfusion.random_discrete_cat(n; d=32, lo=-2.5, hi=2.5) .+ k

# Map latent Z to observed X
Z_to_X(Z) = Z[1, :, :] .+ Z[2, :, :]   # 2×N

# X₁ → Z₁: inc component (index 1) goes to X₁, aux component (index 2) goes to 0
function X1_to_Z1(X1::AbstractArray{<:Integer})
    d, n = size(X1)
    return vcat(reshape(X1, 1, d, n), zeros(Int, 1, d, n))
end

# Whether Z components are incrementing or decrementing
# component 1 increments, component 2 decrements
incrementing = [true, false]

# Hazard distribution F(t): Uniform over [0, 1]
F(t) = t

# Birth–death flow in Z-space; array-based bridge/step are defined in counting.jl
P = BirthDeathFlow(F, Z_to_X, X1_to_Z1, incrementing)


# --------------------------------------------------------------------
# Model / Optimizer
# --------------------------------------------------------------------
model = RModel(embeddim=128, state_dim=4, out_dim=4, layers=5)

eta = 0.001
opt_state = Flux.setup(AdamW(eta=eta), model)

# --------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------
iters = 1000
@showprogress for i in 1:iters
    Z0 = sampleZ0(n_samples, k)         # CountingState
    X1 = sampleX1(n_samples)           # 2×N
    t = rand(T, n_samples)            # length-N
    Zt = bridge(P, Z0, X1, t)          # CountingState

    l, g = Flux.withgradient(model) do m
        R̂ₜ = m(t, Zt)                  # uses (model::RModel)(t, ::CountingState)
        floss(P, R̂ₜ, X1, Zt, 1.0f0)
    end

    Flux.update!(opt_state, model, g[1])
    if i % 50 == 0
        println("Iter: $i, Loss: $l")
    end
end

# --------------------------------------------------------------------
# Sampling with generic gen + CountingState
# --------------------------------------------------------------------
n_inference_samples = 1000
Z0 = sampleZ0(n_inference_samples, k)   # CountingState
X0 = Z_to_X(Z0.state)                   # initial observed X

steps = 0f0:0.05f0:1f0   # smaller step for smoother lines

p = Progress(length(steps) - 1; desc="Sampling", dt=0.2)

# Collector that stores times and states in X-space
mutable struct TrajCollector
    t::Vector{Float32}
    xt::Vector{Matrix{Int}}          # Xₜ is still 2×N
    Rt::Vector{Array{Float32,3}}     # R̂ₜ is 2×2×N now
end

collector() = TrajCollector(Float32[], Matrix{Int}[], Matrix{Float32}[])

function collect!(C::TrajCollector, t, Xt, Rhat)
    push!(C.t, Float32(t))
    push!(C.xt, Int.(Xt))      # 2×N
    push!(C.Rt, Rhat)          # 2×2×N
end

C = collector()

tracker = (t, Xt, Rhat) -> begin
    # Xt and Rhat come in as 1-element tuples from gen
    Xt_state = Xt isa Tuple ? Xt[1] : Xt
    Rhat_mat = Rhat isa Tuple ? Rhat[1] : Rhat

    # Xt_state is CountingState{Array{Int,3}}
    Zt = Xt_state.state             # 2×2×N
    Xt_mat = Z_to_X(Zt)             # 2×N

    collect!(C, t, Xt_mat, Rhat_mat)
end

# IMPORTANT: resolveprediction(R̂ₜ, ::CountingState) = R̂ₜ, so `hat` is just R̂ₜ.
# gen will call step(P, ::CountingState, ::AbstractArray, ...), which we defined above.

@time samp_state = gen(P, Z0, model, steps; tracker=tracker)
ZT = samp_state.state           # final Z
samp = Z_to_X(ZT)               # final X samples 2×N

# --------------------------------------------------------------------
# Visualization
# --------------------------------------------------------------------

# Sample from the target X₁ distribution (for comparison)
sample = sampleX1(1000)
pl = scatter(sample[1, :], sample[2, :],
    msw=0, alpha=0.4, color=:gray, label="Target X₁",
    size=(400, 400))
savefig(pl, "x1_dist.svg")

# Initial vs learned terminal distribution
pl = scatter(
    X0[1, :], X0[2, :],
    color=:blue, alpha=0.1, label="Initial", size=(400, 400)
)
scatter!(pl,
    samp[1, :], samp[2, :],
    color=:green, alpha=0.1, label="Learned Distribution")
#scatter!(pl, sample[1, :], sample[2, :], color=:red, alpha=0.2, label="Data Distribution")

# Draw a few trajectories in X-space
for j in 1:min(50, size(X0, 2))
    xs = [C.xt[k][1, j] for k in 1:length(C.t)]
    ys = [C.xt[k][2, j] for k in 1:length(C.t)]
    plot!(pl, xs, ys, color=:red, alpha=0.3, label=nothing)
end

plot!(pl, [-10], [-10], color=:red, label="Trajectory")  # legend handle
display(pl)
savefig(pl, "countingflow.svg")

# Plot mean predicted residual over time
mean_r_hat = [mean(Rt, dims=(2, 3)) for Rt in C.Rt]  # each 2×1×1
mean_r_hat = hcat(mean_r_hat...)                    # 2×T
mean_r_hat = reshape(mean_r_hat, 2, :)
pl = plot(C.t, mean_r_hat[1, :], label="process 1")
plot!(pl, C.t, mean_r_hat[2, :], label="process 2")
savefig(pl, "mean_r_hat.svg")
