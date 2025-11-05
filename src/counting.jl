struct CountingFlow <: Process
    F::Function # Hazard function
end

# TODO: DiscreteState really doesn't make sense here lol I should use a straight tensor of ints instead?
function bridge(P::CountingFlow, X₀::DiscreteState, X₁::DiscreteState, t::AbstractVector{<:Real})
    x₀ = Int.(tensor(X₀))
    x₁ = Int.(tensor(X₁))
    R₀ = x₁ .- x₀
    @assert all(R₀ .>= 0) "Counts must be nondecreasing"
    Xₜ = similar(R₀)
    N = size(x₀, 2)
    @assert length(t) == N "length(t) must match batch size"
    @inbounds for j in 1:N
        p = P.F(t[j])
        @inbounds for i in eachindex(R₀)
            Xₜ[i, j] = x₀[i, j] + rand(Binomial(R₀[i, j], Float64(p)))
        end
    end
    DiscreteState(X0.K, Xₜ)
end

function step(P::CountingFlow, Xₜ::DiscreteState, R̂ₜ, s₁::Real, s₂::Real)
    xₜ = Int.(tensor(Xₜ))
    r̂ₜ = Float32.(tensor(R̂ₜ))
    # Conditional hazard cdf at s₂ given survival until s₁:
    p = clamp((P.F(s₂) - P.F(s₁)) / max(1f0 - P.F(s₁), eps(Float32)), 0f0, 1f0)
    X_next = similar(xₜ)
    @inbounds for j in eachindex(r̂ₜ)
        X_next[j] = xₜ[j] + rand(Binomial(round(r̂ₜ[j]), Float64(p)))
    end
    DiscreteState(Xₜ.K, X_next)
end

function floss(P::CountingFlow, R̂ₜ, X₁::DiscreteState, Xₜ::DiscreteState, scale)
    R = tensor(X₁) .- tensor(Xₜ)
    return scale * abs2.(R̂ₜ .- R)
end

function gen(P::CountingFlow,
    X₀::DiscreteState,
    model,
    steps::AbstractVector;
    tracker::Function=Returns(nothing),
    midpoint::Bool=false)
    Xₜ = copy(X₀)
    for (s₁, s₂) in zip(steps, steps[begin+1:end])
        t = midpoint ? (s₁ + s₂) / 2 : s₁
        R̂ₜ = model(t, Xₜ)
        Xₜ = mask(step(P, Xₜ, R̂ₜ, s₁, s₂), X₀)
        tracker(t, Xₜ, R̂ₜ)
    end
    return Xₜ
end







