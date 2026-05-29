# --------------------------------------------------------------------------
# Minimal reverse-mode automatic differentiation (a Wengert-list / tape engine),
# pure Julia, no external dependencies. `ADVar <: Real` flows through the same
# generic math used by the soft rasterizer, so a scalar loss over high-dimensional
# parameters (vertex positions, per-face colours) gets its full gradient in a
# single backward pass — O(1) in output dimension, unlike ForwardDiff's O(n).
#
# This is the engine's own reverse mode; it intentionally avoids heavy external
# AD packages (Enzyme/Zygote) that cannot be installed under the §14 disk
# constraint. Correctness is validated against ForwardDiff in the test suite.
# --------------------------------------------------------------------------

mutable struct ADVar <: Real
    val::Float64
    adj::Float64                 # accumulated adjoint (∂output/∂this)
    args::Tuple                  # parent ADVars
    partials::Tuple              # ∂this/∂parent for each parent (Float64)
end

# The active tape (operations recorded in creation = topological order).
const _AD_TAPE = ADVar[]

function _ad_record(val::Float64, args::Tuple, partials::Tuple)
    v = ADVar(val, 0.0, args, partials)
    push!(_AD_TAPE, v)
    return v
end

ADVar(x::Real) = _ad_record(Float64(x), (), ())     # leaf / constant

Base.convert(::Type{ADVar}, x::Real) = x isa ADVar ? x : ADVar(x)
Base.convert(::Type{ADVar}, x::ADVar) = x
Base.promote_rule(::Type{ADVar}, ::Type{<:Real}) = ADVar

Base.Float64(x::ADVar) = x.val
Base.float(x::ADVar) = x.val
(::Type{T})(x::ADVar) where {T<:Integer} = T(x.val)

# ---- arithmetic ----
Base.:+(a::ADVar, b::ADVar) = _ad_record(a.val + b.val, (a, b), (1.0, 1.0))
Base.:-(a::ADVar, b::ADVar) = _ad_record(a.val - b.val, (a, b), (1.0, -1.0))
Base.:-(a::ADVar)           = _ad_record(-a.val, (a,), (-1.0,))
Base.:*(a::ADVar, b::ADVar) = _ad_record(a.val * b.val, (a, b), (b.val, a.val))
function Base.:/(a::ADVar, b::ADVar)
    _ad_record(a.val / b.val, (a, b), (1.0 / b.val, -a.val / (b.val * b.val)))
end

# ---- powers ----
function Base.:^(a::ADVar, p::Integer)
    _ad_record(a.val^p, (a,), (Float64(p) * a.val^(p - 1),))
end
function Base.:^(a::ADVar, p::Real)
    v = a.val^p
    d = a.val == 0 ? 0.0 : p * a.val^(p - 1)
    _ad_record(v, (a,), (d,))
end
Base.literal_pow(::typeof(^), a::ADVar, ::Val{p}) where {p} = a^p

# ---- elementary functions ----
Base.exp(a::ADVar)  = (e = exp(a.val); _ad_record(e, (a,), (e,)))
Base.log(a::ADVar)  = _ad_record(log(a.val), (a,), (1.0 / a.val,))
Base.sqrt(a::ADVar) = (s = sqrt(a.val); _ad_record(s, (a,), (s == 0 ? 0.0 : 0.5 / s,)))
Base.abs(a::ADVar)  = _ad_record(abs(a.val), (a,), (a.val < 0 ? -1.0 : 1.0,))
Base.sin(a::ADVar)  = _ad_record(sin(a.val), (a,), (cos(a.val),))
Base.cos(a::ADVar)  = _ad_record(cos(a.val), (a,), (-sin(a.val),))

# ---- min/max (gradient flows to the selected argument) ----
Base.max(a::ADVar, b::ADVar) = a.val >= b.val ? _ad_record(a.val, (a, b), (1.0, 0.0)) :
                                                _ad_record(b.val, (a, b), (0.0, 1.0))
Base.min(a::ADVar, b::ADVar) = a.val <= b.val ? _ad_record(a.val, (a, b), (1.0, 0.0)) :
                                                _ad_record(b.val, (a, b), (0.0, 1.0))

# ---- comparisons (decided by the value; no gradient) ----
Base.:<(a::ADVar, b::ADVar)  = a.val < b.val
Base.:<=(a::ADVar, b::ADVar) = a.val <= b.val
Base.:>(a::ADVar, b::ADVar)  = a.val > b.val
Base.:>=(a::ADVar, b::ADVar) = a.val >= b.val
Base.:(==)(a::ADVar, b::ADVar) = a.val == b.val
Base.isless(a::ADVar, b::ADVar) = a.val < b.val
Base.isfinite(a::ADVar) = isfinite(a.val)
Base.isnan(a::ADVar) = isnan(a.val)

# ---- identities / rounding (discrete results carry no gradient) ----
Base.zero(::Type{ADVar}) = ADVar(0.0)
Base.one(::Type{ADVar})  = ADVar(1.0)
Base.zero(::ADVar) = ADVar(0.0)
Base.one(::ADVar)  = ADVar(1.0)
Base.floor(::Type{T}, a::ADVar) where {T<:Integer} = floor(T, a.val)
Base.ceil(::Type{T}, a::ADVar) where {T<:Integer}  = ceil(T, a.val)
Base.round(::Type{T}, a::ADVar) where {T<:Integer} = round(T, a.val)
Base.trunc(::Type{T}, a::ADVar) where {T<:Integer} = trunc(T, a.val)

"""
    reverse_gradient(f, x::Vector{Float64}) -> Vector{Float64}

Gradient of scalar `f(x)` via one reverse-mode pass. `f` must accept a vector of
`ADVar` and return a single `ADVar`. Validated against ForwardDiff in tests.
"""
function reverse_gradient(f, x::AbstractVector{<:Real})
    empty!(_AD_TAPE)
    inputs = ADVar[]
    for xi in x
        inp = ADVar(Float64(xi))          # leaf (recorded on tape)
        push!(inputs, inp)
    end
    y = f(inputs)
    y isa ADVar || error("reverse_gradient: f must return a scalar ADVar")
    # Backward pass: tape is in topological order, so iterate in reverse.
    for v in _AD_TAPE
        v.adj = 0.0
    end
    y.adj = 1.0
    @inbounds for k in length(_AD_TAPE):-1:1
        v = _AD_TAPE[k]
        a = v.adj
        a == 0.0 && continue
        for i in 1:length(v.args)
            v.args[i].adj += a * v.partials[i]
        end
    end
    g = [inp.adj for inp in inputs]
    empty!(_AD_TAPE)                       # release the graph
    return g
end

"""Value and gradient of `f` at `x` in one reverse pass."""
function reverse_value_gradient(f, x::AbstractVector{<:Real})
    g = reverse_gradient(f, x)
    return (f(Float64.(x)), g)
end
