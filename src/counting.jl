struct CountingFlow <: Process
    F::Function # Hazard Distribution cdf
end

function bridge(P::CountingFlow, X₀::AbstractMatrix{<:Integer}, X₁::AbstractMatrix{<:Integer}, t::AbstractVector{<:Real})
    R₀ = X₁ .- X₀
    @assert all(R₀ .>= 0) "Counts must be nondecreasing"
    Xₜ = similar(X₀)
    d, N = size(X₀)
    @assert length(t) == N "length(t) must match batch size"
    @inbounds for j in 1:N
        p = P.F(t[j])
        @inbounds for i in 1:d
            Xₜ[i, j] = X₀[i, j] + rand(Binomial(R₀[i, j], Float64(p)))
        end
    end
    return Xₜ
end

function step(P::CountingFlow, Xₜ::AbstractArray{<:Integer}, R̂ₜ::AbstractArray, s₁::Real, s₂::Real)
    d, N = size(Xₜ)
    X_next = similar(Xₜ)
    # Conditional hazard cdf at s₂ given survival until s₁:
    p = clamp((P.F(s₂) - P.F(s₁)) / max(1f0 - P.F(s₁), eps(Float32)), 0f0, 1f0)
    n = max.(round.(R̂ₜ), 0)
    @inbounds for j in 1:N, i in 1:d
        X_next[i, j] = Xₜ[i, j] + rand(Binomial(n[i, j], Float64(p)))
    end
    return X_next
end

function floss(P::CountingFlow, R̂ₜ::AbstractMatrix, X₁::AbstractMatrix{<:Integer}, Xₜ::AbstractMatrix{<:Integer}, c)
    return scaledmaskedmean(abs2.(R̂ₜ .- (X₁ .- Xₜ)), c, nothing)
end

function gen(P::CountingFlow,
    X₀::AbstractMatrix{<:Integer},
    model,
    steps::AbstractVector;
    tracker::Function=Returns(nothing),
    midpoint::Bool=false)
    Xₜ = copy(X₀)
    for (s₁, s₂) in zip(steps, steps[begin+1:end])
        t = midpoint ? (s₁ + s₂) / 2 : s₁
        R̂ₜ = model(t, Xₜ)
        Xₜ = step(P, Xₜ, R̂ₜ, s₁, s₂)
        tracker(t, Xₜ, R̂ₜ)
    end
    return Xₜ
end







