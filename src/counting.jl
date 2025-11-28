struct CountingFlow <: Process
    F::Function # Hazard Distribution cdf
end

struct BirthDeathFlow <: Process # More general setup than CountingFlow, allows for processes that increment and decrement
    F::Function # Hazard Distribution cdf
    Z_to_X::Function # -> AbstractMatrix{<:Integer} Obtain "observed" process values from total process values
    X1_to_Z1::Function # Data X to target tuple of Z
    incrementing::Vector{Bool} # Indicates if component is decrementing or incrementing
end

abstract type CountingAbstractState <: State end
struct CountingState{A} <: CountingAbstractState
    state::A   # e.g. Z :: Array{Int,3} with size (2,2,batch)
end
Base.copy(cs::CountingState) = CountingState(Base.copy(cs.state))

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

function bridge(P::CountingFlow,
    X₀::CountingState,
    X₁::AbstractArray{<:Integer},
    t::AbstractVector{<:Real})
    Xₜ = bridge(P, X₀.state, X₁, t)
    return CountingState(Xₜ)
end

function bridge(P::BirthDeathFlow, Z₀::AbstractArray{<:Integer}, X₁::AbstractArray{<:Integer}, t::AbstractVector{<:Real})
    incrementing_coefficient = ifelse.(P.incrementing, 1, -1)
    # Get remaining increment fields for all components
    R₀ = (P.X1_to_Z1(X₁) .- Z₀) .* reshape(incrementing_coefficient, :, 1, 1)
    @assert all(R₀ .>= 0) "Counts must be nondecreasing"
    Zₜ = similar(Z₀)
    n_processes, n_dim, batch_size = size(Z₀)
    @assert length(t) == batch_size "length(t) must match batch size"
    @inbounds for k in 1:batch_size
        p = P.F(t[k])
        @inbounds for i in 1:n_processes, j in 1:n_dim
            Zₜ[i, j, k] = Z₀[i, j, k] + incrementing_coefficient[i] * rand(Binomial(R₀[i, j, k], Float64(p)))
        end
    end
    return Zₜ
end

function bridge(P::BirthDeathFlow,
    Z₀::CountingState,
    X₁::AbstractArray{<:Integer},
    t::AbstractVector{<:Real})
    Zₜ = bridge(P, Z₀.state, X₁, t)
    return CountingState(Zₜ)
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

function step(P::CountingFlow,
    Xₜ::CountingState,
    R̂ₜ::AbstractArray,
    s₁::Real,
    s₂::Real)
    X_next = step(P, Xₜ.state, R̂ₜ, s₁, s₂)
    return CountingState(X_next)
end

function step(P::BirthDeathFlow, Zₜ::AbstractArray{<:Integer}, R̂ₜ::AbstractArray, s₁::Real, s₂::Real)
    incrementing_coefficient = ifelse.(P.incrementing, 1, -1)
    n_processes, n_dim, batch_size = size(Zₜ)
    Z_next = similar(Zₜ)
    # Conditional hazard cdf at s₂ given survival until s₁:
    p = clamp((P.F(s₂) - P.F(s₁)) / max(1f0 - P.F(s₁), eps(Float32)), 0f0, 1f0)
    R̂_int = round.(Int, R̂ₜ)
    n = max.(0, R̂_int)
    @inbounds for i in 1:n_processes, j in 1:n_dim, k in 1:batch_size
        Z_next[i, j, k] = Zₜ[i, j, k] + incrementing_coefficient[i] * rand(Binomial(n[i, j, k], Float64(p)))
    end
    return Z_next
end

function step(P::BirthDeathFlow,
    Zₜ::CountingState,
    R̂ₜ::AbstractArray,
    s₁::Real,
    s₂::Real)
    Z_next = step(P, Zₜ.state, R̂ₜ, s₁, s₂)  # array version
    return CountingState(Z_next)
end

function floss(P::CountingFlow, R̂ₜ::AbstractMatrix, X₁::AbstractMatrix{<:Integer}, Xₜ::AbstractMatrix{<:Integer}, c)
    return scaledmaskedmean(abs2.(R̂ₜ .- (X₁ .- Xₜ)), c, nothing)
end

function floss(P::BirthDeathFlow, R̂ₜ::AbstractArray, X₁::AbstractMatrix{<:Integer}, Zₜ::AbstractMatrix{<:Integer}, c)
    return scaledmaskedmean(abs2.(R̂ₜ .- (P.X1_to_Z1(X₁) .- Zₜ)), c, nothing)
end

function floss(P::BirthDeathFlow, R̂ₜ::AbstractArray, X₁::AbstractMatrix{<:Integer}, Zₜ::CountingState, c)
    return scaledmaskedmean(abs2.(R̂ₜ .- (P.X1_to_Z1(X₁) .- Zₜ.state)), c, nothing)
end

resolveprediction(R̂ₜ::AbstractArray, Xₜ::CountingState) = R̂ₜ
#= function gen(P::CountingFlow,
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
end =#







