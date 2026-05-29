# --------------------------------------------------------------------------
# Light types mirroring three.js light hierarchy.
# --------------------------------------------------------------------------

abstract type AbstractLight <: AbstractObject3D end

# ========================== AmbientLight ==========================

mutable struct AmbientLight <: AbstractLight
    position::Vec3{Float64}
    rotation::Euler{Float64}
    scale::Vec3{Float64}
    parent::Union{Nothing, AbstractObject3D}
    children::Vector{AbstractObject3D}
    visible::Bool
    name::String
    id::Int
    color::Color3{Float64}
    intensity::Float64
end

function AmbientLight(; color=Color3(1.0, 1.0, 1.0), intensity=1.0, name="AmbientLight")
    AmbientLight(Vec3(), Euler(), Vec3(1.0,1.0,1.0),
                 nothing, AbstractObject3D[], true, name, _next_id(),
                 color, intensity)
end

get_position(o::AmbientLight) = o.position
get_rotation(o::AmbientLight) = o.rotation
get_scale(o::AmbientLight) = o.scale
get_children(o::AmbientLight) = o.children
get_parent(o::AmbientLight) = o.parent
is_visible(o::AmbientLight) = o.visible
set_parent!(o::AmbientLight, p) = (o.parent = p)

# ========================== DirectionalLight ==========================

mutable struct DirectionalLight <: AbstractLight
    position::Vec3{Float64}
    rotation::Euler{Float64}
    scale::Vec3{Float64}
    parent::Union{Nothing, AbstractObject3D}
    children::Vector{AbstractObject3D}
    visible::Bool
    name::String
    id::Int
    color::Color3{Float64}
    intensity::Float64
    target::Vec3{Float64}
    cast_shadow::Bool
end

function DirectionalLight(; color=Color3(1.0, 1.0, 1.0), intensity=1.0,
                           position=Vec3(0.0, 1.0, 0.0), name="DirectionalLight")
    DirectionalLight(position, Euler(), Vec3(1.0,1.0,1.0),
                     nothing, AbstractObject3D[], true, name, _next_id(),
                     color, intensity, Vec3(), false)
end

get_position(o::DirectionalLight) = o.position
get_rotation(o::DirectionalLight) = o.rotation
get_scale(o::DirectionalLight) = o.scale
get_children(o::DirectionalLight) = o.children
get_parent(o::DirectionalLight) = o.parent
is_visible(o::DirectionalLight) = o.visible
set_parent!(o::DirectionalLight, p) = (o.parent = p)

# ========================== PointLight ==========================

mutable struct PointLight <: AbstractLight
    position::Vec3{Float64}
    rotation::Euler{Float64}
    scale::Vec3{Float64}
    parent::Union{Nothing, AbstractObject3D}
    children::Vector{AbstractObject3D}
    visible::Bool
    name::String
    id::Int
    color::Color3{Float64}
    intensity::Float64
    distance::Float64
    decay::Float64
    cast_shadow::Bool
end

function PointLight(; color=Color3(1.0, 1.0, 1.0), intensity=1.0,
                     distance=0.0, decay=2.0, position=Vec3(),
                     name="PointLight")
    PointLight(position, Euler(), Vec3(1.0,1.0,1.0),
               nothing, AbstractObject3D[], true, name, _next_id(),
               color, intensity, distance, decay, false)
end

get_position(o::PointLight) = o.position
get_rotation(o::PointLight) = o.rotation
get_scale(o::PointLight) = o.scale
get_children(o::PointLight) = o.children
get_parent(o::PointLight) = o.parent
is_visible(o::PointLight) = o.visible
set_parent!(o::PointLight, p) = (o.parent = p)

# ========================== SpotLight ==========================

mutable struct SpotLight <: AbstractLight
    position::Vec3{Float64}
    rotation::Euler{Float64}
    scale::Vec3{Float64}
    parent::Union{Nothing, AbstractObject3D}
    children::Vector{AbstractObject3D}
    visible::Bool
    name::String
    id::Int
    color::Color3{Float64}
    intensity::Float64
    distance::Float64
    angle::Float64
    penumbra::Float64
    decay::Float64
    target::Vec3{Float64}
    cast_shadow::Bool
end

function SpotLight(; color=Color3(1.0, 1.0, 1.0), intensity=1.0,
                    distance=0.0, angle=π/3, penumbra=0.0, decay=2.0,
                    position=Vec3(0.0, 1.0, 0.0), name="SpotLight")
    SpotLight(position, Euler(), Vec3(1.0,1.0,1.0),
              nothing, AbstractObject3D[], true, name, _next_id(),
              color, intensity, distance, angle, penumbra, decay,
              Vec3(), false)
end

get_position(o::SpotLight) = o.position
get_rotation(o::SpotLight) = o.rotation
get_scale(o::SpotLight) = o.scale
get_children(o::SpotLight) = o.children
get_parent(o::SpotLight) = o.parent
is_visible(o::SpotLight) = o.visible
set_parent!(o::SpotLight, p) = (o.parent = p)

# ========================== HemisphereLight ==========================

mutable struct HemisphereLight <: AbstractLight
    position::Vec3{Float64}
    rotation::Euler{Float64}
    scale::Vec3{Float64}
    parent::Union{Nothing, AbstractObject3D}
    children::Vector{AbstractObject3D}
    visible::Bool
    name::String
    id::Int
    color::Color3{Float64}        # sky color
    ground_color::Color3{Float64}  # ground color
    intensity::Float64
end

function HemisphereLight(; color=Color3(1.0, 1.0, 1.0),
                          ground_color=Color3(0.0, 0.0, 0.0),
                          intensity=1.0, name="HemisphereLight")
    HemisphereLight(Vec3(), Euler(), Vec3(1.0,1.0,1.0),
                    nothing, AbstractObject3D[], true, name, _next_id(),
                    color, ground_color, intensity)
end

get_position(o::HemisphereLight) = o.position
get_rotation(o::HemisphereLight) = o.rotation
get_scale(o::HemisphereLight) = o.scale
get_children(o::HemisphereLight) = o.children
get_parent(o::HemisphereLight) = o.parent
is_visible(o::HemisphereLight) = o.visible
set_parent!(o::HemisphereLight, p) = (o.parent = p)

# ========================== RectAreaLight ==========================
# A rectangular emitter. Shaded as a directional source from its centre with an
# emission plane defined by `target` (no distance falloff, single-sided).

mutable struct RectAreaLight <: AbstractLight
    position::Vec3{Float64}
    rotation::Euler{Float64}
    scale::Vec3{Float64}
    parent::Union{Nothing, AbstractObject3D}
    children::Vector{AbstractObject3D}
    visible::Bool
    name::String
    id::Int
    color::Color3{Float64}
    intensity::Float64
    width::Float64
    height::Float64
    target::Vec3{Float64}
end

function RectAreaLight(; color=Color3(1.0,1.0,1.0), intensity=1.0, width=1.0, height=1.0,
                        position=Vec3(0.0,1.0,0.0), name="RectAreaLight")
    RectAreaLight(position, Euler(), Vec3(1.0,1.0,1.0), nothing, AbstractObject3D[],
                  true, name, _next_id(), color, intensity, width, height, Vec3())
end

get_position(o::RectAreaLight) = o.position
get_rotation(o::RectAreaLight) = o.rotation
get_scale(o::RectAreaLight) = o.scale
get_children(o::RectAreaLight) = o.children
get_parent(o::RectAreaLight) = o.parent
is_visible(o::RectAreaLight) = o.visible
set_parent!(o::RectAreaLight, p) = (o.parent = p)

# ========================== LightProbe ==========================
# Order-1 spherical-harmonics ambient probe: irradiance varies linearly with the
# surface normal. `coeffs` = (DC, x-grad, y-grad, z-grad).

mutable struct LightProbe <: AbstractLight
    position::Vec3{Float64}
    rotation::Euler{Float64}
    scale::Vec3{Float64}
    parent::Union{Nothing, AbstractObject3D}
    children::Vector{AbstractObject3D}
    visible::Bool
    name::String
    id::Int
    coeffs::NTuple{4, Color3{Float64}}
    intensity::Float64
end

function LightProbe(; coeffs=(Color3(0.0,0.0,0.0), Color3(0.0,0.0,0.0),
                              Color3(0.0,0.0,0.0), Color3(0.0,0.0,0.0)),
                     intensity=1.0, name="LightProbe")
    LightProbe(Vec3(), Euler(), Vec3(1.0,1.0,1.0), nothing, AbstractObject3D[],
               true, name, _next_id(), coeffs, intensity)
end

"""Build a uniform (DC-only) light probe from an ambient colour."""
LightProbe(ambient::Color3; intensity=1.0) =
    LightProbe(coeffs=(ambient, Color3(0.0,0.0,0.0), Color3(0.0,0.0,0.0), Color3(0.0,0.0,0.0)),
               intensity=intensity)

get_position(o::LightProbe) = o.position
get_rotation(o::LightProbe) = o.rotation
get_scale(o::LightProbe) = o.scale
get_children(o::LightProbe) = o.children
get_parent(o::LightProbe) = o.parent
is_visible(o::LightProbe) = o.visible
set_parent!(o::LightProbe, p) = (o.parent = p)

# ========================== Light collection ==========================

function collect_lights(scene::AbstractObject3D)
    lights = AbstractLight[]
    traverse(scene, obj -> obj isa AbstractLight && push!(lights, obj))
    return lights
end
