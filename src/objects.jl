# --------------------------------------------------------------------------
# Additional scene-graph objects mirroring three.js: InstancedMesh,
# LineSegments, Sprite, LOD, Bone/Skeleton/SkinnedMesh, plus the Layers
# bitmask and a one-pass world-matrix cache (matrixWorldAutoUpdate analogue).
# --------------------------------------------------------------------------

# ========================== Layers ==========================
# three.js channel bitmask. Default: channel 0 enabled (mask = 1).

mutable struct Layers
    mask::UInt32
end
Layers() = Layers(UInt32(1))

layers_set!(l::Layers, channel::Int)    = (l.mask = UInt32(1) << (channel & 31); l)
layers_enable!(l::Layers, channel::Int) = (l.mask |= (UInt32(1) << (channel & 31)); l)
layers_disable!(l::Layers, channel::Int)= (l.mask &= ~(UInt32(1) << (channel & 31)); l)
layers_toggle!(l::Layers, channel::Int) = (l.mask ⊻= (UInt32(1) << (channel & 31)); l)
layers_enable_all!(l::Layers)  = (l.mask = 0xffffffff; l)
layers_disable_all!(l::Layers) = (l.mask = UInt32(0); l)
"""True if two layer masks share any enabled channel (three.js `Layers.test`)."""
layers_test(a::Layers, b::Layers) = (a.mask & b.mask) != 0

# ========================== InstancedMesh ==========================
# One geometry/material drawn at many transforms with bounded extra memory.

mutable struct InstancedMesh <: AbstractObject3D
    position::Vec3{Float64}
    rotation::Euler{Float64}
    scale::Vec3{Float64}
    parent::Union{Nothing, AbstractObject3D}
    children::Vector{AbstractObject3D}
    visible::Bool
    name::String
    id::Int
    geometry::Any
    material::Any
    instance_matrices::Vector{Mat4{Float64}}
end

function InstancedMesh(geometry, material, count::Int; name="InstancedMesh")
    mats = [Mat4{Float64}() for _ in 1:count]
    InstancedMesh(Vec3(), Euler(), Vec3(1.0,1.0,1.0), nothing, AbstractObject3D[],
                  true, name, _next_id(), geometry, material, mats)
end

get_position(o::InstancedMesh) = o.position
get_rotation(o::InstancedMesh) = o.rotation
get_scale(o::InstancedMesh) = o.scale
get_children(o::InstancedMesh) = o.children
get_parent(o::InstancedMesh) = o.parent
is_visible(o::InstancedMesh) = o.visible
set_parent!(o::InstancedMesh, p) = (o.parent = p)

instanced_count(o::InstancedMesh) = length(o.instance_matrices)
set_instance_matrix!(o::InstancedMesh, i::Int, m::Mat4) = (o.instance_matrices[i] = m)
get_instance_matrix(o::InstancedMesh, i::Int) = o.instance_matrices[i]

"""Collect every `InstancedMesh` under `root` (used by the rasterizer)."""
function collect_instanced(root::AbstractObject3D)
    out = InstancedMesh[]
    traverse(root, o -> o isa InstancedMesh && push!(out, o))
    return out
end

# ========================== LineSegments ==========================
# Geometry vertices interpreted as disjoint segment pairs (v1-v2, v3-v4, ...).

mutable struct LineSegments <: AbstractObject3D
    position::Vec3{Float64}
    rotation::Euler{Float64}
    scale::Vec3{Float64}
    parent::Union{Nothing, AbstractObject3D}
    children::Vector{AbstractObject3D}
    visible::Bool
    name::String
    id::Int
    geometry::Any
    material::Any
end

function LineSegments(geometry, material; name="LineSegments")
    LineSegments(Vec3(), Euler(), Vec3(1.0,1.0,1.0), nothing, AbstractObject3D[],
                 true, name, _next_id(), geometry, material)
end

get_position(o::LineSegments) = o.position
get_rotation(o::LineSegments) = o.rotation
get_scale(o::LineSegments) = o.scale
get_children(o::LineSegments) = o.children
get_parent(o::LineSegments) = o.parent
is_visible(o::LineSegments) = o.visible
set_parent!(o::LineSegments, p) = (o.parent = p)

# ========================== Sprite ==========================
# A camera-facing billboard.

mutable struct Sprite <: AbstractObject3D
    position::Vec3{Float64}
    rotation::Euler{Float64}
    scale::Vec3{Float64}
    parent::Union{Nothing, AbstractObject3D}
    children::Vector{AbstractObject3D}
    visible::Bool
    name::String
    id::Int
    material::Any
end

function Sprite(material; name="Sprite")
    Sprite(Vec3(), Euler(), Vec3(1.0,1.0,1.0), nothing, AbstractObject3D[],
           true, name, _next_id(), material)
end

get_position(o::Sprite) = o.position
get_rotation(o::Sprite) = o.rotation
get_scale(o::Sprite) = o.scale
get_children(o::Sprite) = o.children
get_parent(o::Sprite) = o.parent
is_visible(o::Sprite) = o.visible
set_parent!(o::Sprite, p) = (o.parent = p)

"""
World matrix for a sprite: positioned at its world location, oriented so its
local axes coincide with the camera's right/up/forward axes (screen-facing).
"""
function sprite_world_matrix(sprite::Sprite, camera::AbstractCamera)
    V = view_matrix(camera)
    # Rows 1..3 of the view matrix are the camera right/up/forward axes in world space.
    right = Vec3(mat4_get(V,1,1), mat4_get(V,1,2), mat4_get(V,1,3))
    up    = Vec3(mat4_get(V,2,1), mat4_get(V,2,2), mat4_get(V,2,3))
    fwd   = Vec3(mat4_get(V,3,1), mat4_get(V,3,2), mat4_get(V,3,3))
    p = get_position(sprite); s = get_scale(sprite)
    Mat4((right.x*s.x, right.y*s.x, right.z*s.x, 0.0,
          up.x*s.y,    up.y*s.y,    up.z*s.y,    0.0,
          fwd.x*s.z,   fwd.y*s.z,   fwd.z*s.z,   0.0,
          p.x,         p.y,         p.z,         1.0))
end

# ========================== LOD ==========================
# Level-of-detail container: child objects keyed by minimum camera distance.

mutable struct LOD <: AbstractObject3D
    position::Vec3{Float64}
    rotation::Euler{Float64}
    scale::Vec3{Float64}
    parent::Union{Nothing, AbstractObject3D}
    children::Vector{AbstractObject3D}
    visible::Bool
    name::String
    id::Int
    levels::Vector{Tuple{Float64, AbstractObject3D}}   # (min distance, object), ascending
end

function LOD(; name="LOD")
    LOD(Vec3(), Euler(), Vec3(1.0,1.0,1.0), nothing, AbstractObject3D[],
        true, name, _next_id(), Tuple{Float64, AbstractObject3D}[])
end

get_position(o::LOD) = o.position
get_rotation(o::LOD) = o.rotation
get_scale(o::LOD) = o.scale
get_children(o::LOD) = o.children
get_parent(o::LOD) = o.parent
is_visible(o::LOD) = o.visible
set_parent!(o::LOD, p) = (o.parent = p)

function add_lod_level!(lod::LOD, distance::Real, obj::AbstractObject3D)
    push!(lod.levels, (Float64(distance), obj))
    add!(lod, obj)
    sort!(lod.levels, by = first)
    return lod
end

"""Highest-distance LOD level whose threshold ≤ `distance` (three.js `getObjectForDistance`)."""
function lod_select(lod::LOD, distance::Real)
    isempty(lod.levels) && return nothing
    chosen = lod.levels[1][2]
    for (d, obj) in lod.levels
        distance >= d ? (chosen = obj) : break
    end
    return chosen
end

# ========================== Bone / Skeleton / SkinnedMesh ==========================

mutable struct Bone <: AbstractObject3D
    position::Vec3{Float64}
    rotation::Euler{Float64}
    scale::Vec3{Float64}
    parent::Union{Nothing, AbstractObject3D}
    children::Vector{AbstractObject3D}
    visible::Bool
    name::String
    id::Int
end

function Bone(; name="Bone")
    Bone(Vec3(), Euler(), Vec3(1.0,1.0,1.0), nothing, AbstractObject3D[],
         true, name, _next_id())
end

get_position(o::Bone) = o.position
get_rotation(o::Bone) = o.rotation
get_scale(o::Bone) = o.scale
get_children(o::Bone) = o.children
get_parent(o::Bone) = o.parent
is_visible(o::Bone) = o.visible
set_parent!(o::Bone, p) = (o.parent = p)

mutable struct Skeleton
    bones::Vector{Bone}
    bind_inverses::Vector{Mat4{Float64}}   # inverse of each bone's bind-pose world matrix
end

"""Build a skeleton, capturing the current bone world matrices as the bind pose."""
function Skeleton(bones::Vector{Bone})
    binv = [mat4_inverse(compute_world_matrix(b)) for b in bones]
    Skeleton(bones, binv)
end

"""Per-bone skinning matrix = current world × inverse bind (identity in bind pose)."""
skeleton_matrices(s::Skeleton) =
    [compute_world_matrix(s.bones[i]) * s.bind_inverses[i] for i in eachindex(s.bones)]

mutable struct SkinnedMesh <: AbstractObject3D
    position::Vec3{Float64}
    rotation::Euler{Float64}
    scale::Vec3{Float64}
    parent::Union{Nothing, AbstractObject3D}
    children::Vector{AbstractObject3D}
    visible::Bool
    name::String
    id::Int
    geometry::Any
    material::Any
    skeleton::Skeleton
    skin_indices::Vector{NTuple{4,Int}}     # bone indices per vertex (1-based)
    skin_weights::Vector{NTuple{4,Float64}} # blend weights per vertex
end

function SkinnedMesh(geometry, material, skeleton::Skeleton,
                     skin_indices, skin_weights; name="SkinnedMesh")
    SkinnedMesh(Vec3(), Euler(), Vec3(1.0,1.0,1.0), nothing, AbstractObject3D[],
                true, name, _next_id(), geometry, material, skeleton,
                skin_indices, skin_weights)
end

get_position(o::SkinnedMesh) = o.position
get_rotation(o::SkinnedMesh) = o.rotation
get_scale(o::SkinnedMesh) = o.scale
get_children(o::SkinnedMesh) = o.children
get_parent(o::SkinnedMesh) = o.parent
is_visible(o::SkinnedMesh) = o.visible
set_parent!(o::SkinnedMesh, p) = (o.parent = p)

"""
Linear blend skinning: deform each geometry vertex by the weighted sum of its
bones' skinning matrices. Returns `Vector{Vec3}` of deformed positions.
"""
function apply_skinning(sm::SkinnedMesh)
    mats = skeleton_matrices(sm.skeleton)
    geo = sm.geometry
    out = Vector{Vec3{Float64}}(undef, geo.n_vertices)
    @inbounds for vi in 1:geo.n_vertices
        p = get_vertex(geo, vi)
        idx = sm.skin_indices[vi]; w = sm.skin_weights[vi]
        acc = Vec3(0.0, 0.0, 0.0)
        for k in 1:4
            wk = w[k]
            wk == 0 && continue
            acc = acc + mat4_transform_point(mats[idx[k]], p) * wk
        end
        out[vi] = acc
    end
    return out
end

# ========================== World-matrix cache ==========================
# One traversal pass computing every object's world matrix (each child reuses
# its parent's already-computed matrix), instead of walking to the root per
# object. Mirrors three.js `updateMatrixWorld` done once per frame.

function compute_world_matrices(root::AbstractObject3D,
                                parent_world::Mat4,
                                cache::IdDict{AbstractObject3D, Mat4})
    world = parent_world * compute_local_matrix(root)
    cache[root] = world
    for child in get_children(root)
        compute_world_matrices(child, world, cache)
    end
    return cache
end

function compute_world_matrices(root::AbstractObject3D)
    cache = IdDict{AbstractObject3D, Mat4}()
    compute_world_matrices(root, Mat4{Float64}(), cache)
    return cache
end
