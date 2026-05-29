# --------------------------------------------------------------------------
# Renderer extras: tone mapping + sRGB output encoding, supersample anti-
# aliasing, line/point rasterization, a small post-processing EffectComposer,
# and a tiled (optionally threaded) rasterizer.
# --------------------------------------------------------------------------

# ========================== Tone mapping / sRGB ==========================

"""Reinhard tone map `c/(1+c)`, mapping HDR radiance into [0,1)."""
function tone_map_reinhard(img::AbstractArray)
    out = Array{Float64}(undef, size(img))
    @inbounds for i in eachindex(img)
        c = Float64(img[i]); out[i] = c / (1 + c)
    end
    return out
end

"""ACES filmic tone map (Narkowicz approximation)."""
function tone_map_aces(img::AbstractArray)
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    out = Array{Float64}(undef, size(img))
    @inbounds for i in eachindex(img)
        x = Float64(img[i])
        out[i] = clamp((x * (a*x + b)) / (x * (c*x + d) + e), 0.0, 1.0)
    end
    return out
end

linear_to_srgb(c) = c <= 0.0031308 ? 12.92*c : 1.055*c^(1/2.4) - 0.055
srgb_to_linear(c) = c <= 0.04045 ? c/12.92 : ((c + 0.055)/1.055)^2.4

"""Encode a linear-light image to sRGB for display/output."""
function srgb_encode(img::AbstractArray)
    out = Array{Float64}(undef, size(img))
    @inbounds for i in eachindex(img)
        out[i] = clamp(linear_to_srgb(clamp(Float64(img[i]), 0.0, 1.0)), 0.0, 1.0)
    end
    return out
end

# ========================== Supersample (MSAA) ==========================

"""Box-average downsample an H·ss × W·ss image to H × W."""
function downsample(img::AbstractArray, ss::Int)
    Hb, Wb, _ = size(img)
    H, W = Hb ÷ ss, Wb ÷ ss
    out = zeros(Float64, H, W, 3)
    inv = 1.0 / (ss * ss)
    @inbounds for c in 1:3, i in 1:H, j in 1:W
        s = 0.0
        for di in 0:ss-1, dj in 0:ss-1
            s += img[(i-1)*ss + di + 1, (j-1)*ss + dj + 1, c]
        end
        out[i, j, c] = s * inv
    end
    return out
end

"""Render at `ss`× resolution and box-downsample — supersampled anti-aliasing."""
function render_aa(scene::Scene, camera::AbstractCamera, width::Int, height::Int;
                   ss::Int=2, shading::Symbol=:flat, shadows::Bool=false)
    rt = RenderTarget(width*ss, height*ss)
    render!(rt, scene, camera; shading=shading, shadows=shadows)
    return downsample(rt.color, ss)
end

"""
    render_msaa!(rt, scene, camera; samples=4, shading=:flat, shadows=false)

In-renderer multisample anti-aliasing: render an internal ⌈√samples⌉× target and
box-downsample into `rt.color`. Unlike `render_aa` (which returns an array), this
fills the supplied `RenderTarget`, so the renderer itself yields an AA frame.
"""
function render_msaa!(rt::RenderTarget, scene::Scene, camera::AbstractCamera;
                      samples::Int=4, shading::Symbol=:flat, shadows::Bool=false)
    ss = max(round(Int, sqrt(samples)), 1)
    big = RenderTarget(rt.width*ss, rt.height*ss)
    render!(big, scene, camera; shading=shading, shadows=shadows)
    rt.color .= downsample(big.color, ss)
    return rt
end

# ========================== Pooled rendering (bounded allocation) ==========================

"""
Reusable scratch buffers for the flat opaque path, so repeated frames allocate a
bounded amount (independent of frame count). Mesh/light lists and the per-face
colour buffer are reused via [`render_pooled!`].
"""
mutable struct RenderCache
    meshes::Vector{Mesh}
    lights::Vector{AbstractLight}
    instanced::Vector{InstancedMesh}
    tri::Vector{Vec4{Float64}}
    clipped::Vector{Vec4{Float64}}
    sx::Vector{Float64}
    sy::Vector{Float64}
    sz::Vector{Float64}
    colors::Vector{Color3{Float64}}
end
function RenderCache()
    cl = Vector{Vec4{Float64}}(undef, 0); sizehint!(cl, 6)
    RenderCache(Mesh[], AbstractLight[], InstancedMesh[],
                Vector{Vec4{Float64}}(undef, 3), cl,
                Vector{Float64}(undef, 8), Vector{Float64}(undef, 8), Vector{Float64}(undef, 8),
                Color3{Float64}[])
end

# Collect into a reused vector (clears then refills), avoiding per-frame allocation.
function _collect_into!(out::Vector, root, pred)
    empty!(out)
    traverse(root, o -> pred(o) && push!(out, o))
    return out
end

"""
    render_pooled!(rt, scene, camera, cache; shading=:flat)

Flat-rasterize opaque meshes/instances reusing `cache`'s buffers — the same
image as `render!` for opaque flat scenes, but with bounded per-frame allocation
across repeated calls. Transparent meshes, lines and points are skipped here.
"""
function render_pooled!(rt::RenderTarget, scene::Scene, camera::AbstractCamera,
                        cache::RenderCache; shading::Symbol=:flat)
    clear!(rt, scene.background)
    proj = projection_matrix(camera); view = view_matrix(camera); near = _camera_near(camera)
    _collect_into!(cache.meshes, scene, m -> m isa Mesh)
    _collect_into!(cache.lights, scene, l -> l isa AbstractLight)
    _collect_into!(cache.instanced, scene, o -> o isa InstancedMesh)
    for mesh in cache.meshes
        (is_visible(mesh) && !is_transparent_material(mesh.material)) || continue
        _rasterize_geo_flat!(rt, mesh.geometry, compute_world_matrix(mesh), mesh.material,
                             cache.lights, proj, view, near, camera.position,
                             cache.tri, cache.clipped, cache.sx, cache.sy, cache.sz; colorbuf=cache.colors)
    end
    for im in cache.instanced
        is_visible(im) || continue
        base = compute_world_matrix(im)
        for M in im.instance_matrices
            _rasterize_geo_flat!(rt, im.geometry, base * M, im.material,
                                 cache.lights, proj, view, near, camera.position,
                                 cache.tri, cache.clipped, cache.sx, cache.sy, cache.sz; colorbuf=cache.colors)
        end
    end
    return rt
end

# ========================== Line / point rasterization ==========================

@inline function _put_pixel!(rt::RenderTarget, x::Int, y::Int, z, col::Color3)
    (1 <= x <= rt.width && 1 <= y <= rt.height) || return
    @inbounds if z < rt.depth[y, x]
        rt.depth[y, x] = z
        rt.color[y, x, 1] = col.r; rt.color[y, x, 2] = col.g; rt.color[y, x, 3] = col.b
    end
end

# DDA line with depth interpolation and z-test.
function _draw_line!(rt::RenderTarget, x0, y0, z0, x1, y1, z1, col::Color3)
    dx = x1 - x0; dy = y1 - y0
    steps = max(abs(dx), abs(dy))
    steps < 1 && (steps = 1.0)
    @inbounds for s in 0:Int(ceil(steps))
        t = s / steps
        _put_pixel!(rt, round(Int, x0 + dx*t), round(Int, y0 + dy*t), z0 + (z1 - z0)*t, col)
    end
    return nothing
end

# Project a world point; returns (sx, sy, ndcz, ok) — ok=false if behind the camera.
@inline function _project(vp::Mat4, p::Vec3, W, H)
    c = mat4_transform_vec4(vp, Vec4(p.x, p.y, p.z, 1.0))
    c.w <= 1e-6 && return (0.0, 0.0, 0.0, false)
    iw = 1.0 / c.w
    ((c.x*iw + 1)*0.5*W, (1 - c.y*iw)*0.5*H, c.z*iw, true)
end

"""Rasterize `LineObject` (polyline strips) and `LineSegments` (vertex pairs)."""
function render_lines!(rt::RenderTarget, scene::AbstractObject3D, camera::AbstractCamera)
    vp = projection_matrix(camera) * view_matrix(camera)
    W, H = rt.width, rt.height
    traverse(scene, function(obj)
        is_visible(obj) || return
        if obj isa LineObject || obj isa LineSegments
            wm = compute_world_matrix(obj); geo = obj.geometry
            col = hasfield(typeof(obj.material), :color) ? obj.material.color : Color3(1.0,1.0,1.0)
            stride = obj isa LineSegments ? 2 : 1
            nv = geo.n_vertices
            i = 1
            while i + 1 <= nv
                a = mat4_transform_point(wm, get_vertex(geo, i))
                b = mat4_transform_point(wm, get_vertex(geo, i+1))
                (ax, ay, az, oka) = _project(vp, a, W, H)
                (bx, by, bz, okb) = _project(vp, b, W, H)
                (oka && okb) && _draw_line!(rt, ax, ay, az, bx, by, bz, col)
                i += stride
            end
        end
    end)
    return rt
end

"""Rasterize `PointsObject` vertices as small squares sized by the material."""
function render_points!(rt::RenderTarget, scene::AbstractObject3D, camera::AbstractCamera)
    vp = projection_matrix(camera) * view_matrix(camera)
    W, H = rt.width, rt.height
    traverse(scene, function(obj)
        (is_visible(obj) && obj isa PointsObject) || return
        wm = compute_world_matrix(obj); geo = obj.geometry
        col = hasfield(typeof(obj.material), :color) ? obj.material.color : Color3(1.0,1.0,1.0)
        r = max(Int(round(hasfield(typeof(obj.material), :size) ? obj.material.size : 1.0)) ÷ 2, 0)
        for vi in 1:geo.n_vertices
            (px, py, pz, ok) = _project(vp, mat4_transform_point(wm, get_vertex(geo, vi)), W, H)
            ok || continue
            cx = round(Int, px); cy = round(Int, py)
            for dy in -r:r, dx in -r:r
                _put_pixel!(rt, cx+dx, cy+dy, pz, col)
            end
        end
    end)
    return rt
end

# ========================== EffectComposer (post-processing) ==========================

mutable struct EffectComposer
    passes::Vector{Function}     # each maps an H×W×3 image to an H×W×3 image
end
EffectComposer() = EffectComposer(Function[])
add_pass!(c::EffectComposer, f::Function) = (push!(c.passes, f); c)

"""Run the image through every pass in order."""
function compose(c::EffectComposer, img::AbstractArray)
    out = img
    for p in c.passes
        out = p(out)
    end
    return out
end

# Built-in passes.
function grayscale_pass(img::AbstractArray)
    H, W, _ = size(img)
    out = Array{Float64}(undef, H, W, 3)
    @inbounds for i in 1:H, j in 1:W
        g = 0.299*img[i,j,1] + 0.587*img[i,j,2] + 0.114*img[i,j,3]
        out[i,j,1] = g; out[i,j,2] = g; out[i,j,3] = g
    end
    return out
end
reinhard_pass(img) = tone_map_reinhard(img)
aces_pass(img) = tone_map_aces(img)
srgb_pass(img) = srgb_encode(img)

# ========================== Tiled / parallel rasterization ==========================

"""
    render_tiled!(rt, scene, camera; tiles=Threads.nthreads(), shading=:flat)

Flat-rasterize the scene in horizontal row bands. Bands write disjoint rows, so
they can run on separate threads (used when Julia is started with > 1 thread).
Produces the same image as [`render!`] for opaque flat scenes.
"""
function render_tiled!(rt::RenderTarget, scene::Scene, camera::AbstractCamera;
                       tiles::Int=max(Threads.nthreads(), 1), shading::Symbol=:flat)
    clear!(rt, scene.background)
    proj = projection_matrix(camera)
    view = view_matrix(camera)
    near = _camera_near(camera)
    H = rt.height
    meshes = collect_meshes(scene)
    lights = collect_lights(scene)
    band = cld(H, tiles)
    Threads.@threads for t in 1:tiles
        ylo = (t-1)*band + 1
        yhi = min(t*band, H)
        ylo > yhi && continue
        tri = Vector{Vec4{Float64}}(undef, 3)
        clipped = Vector{Vec4{Float64}}(undef, 0); sizehint!(clipped, 6)
        sx = Vector{Float64}(undef, 8); sy = Vector{Float64}(undef, 8); sz = Vector{Float64}(undef, 8)
        for mesh in meshes
            is_visible(mesh) || continue
            is_transparent_material(mesh.material) && continue
            _rasterize_geo_flat!(rt, mesh.geometry, compute_world_matrix(mesh), mesh.material,
                                 lights, proj, view, near, camera.position, tri, clipped, sx, sy, sz;
                                 ylo=ylo, yhi=yhi)
        end
    end
    return rt
end
