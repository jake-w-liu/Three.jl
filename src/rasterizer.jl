# --------------------------------------------------------------------------
# CPU software rasterizer with z-buffer.
# Produces an H×W×3 Float64 image array (RGB, values in [0,1]).
# --------------------------------------------------------------------------

struct RenderTarget{T<:Real}
    width::Int
    height::Int
    color::Array{T, 3}    # H × W × 3 (RGB)
    depth::Matrix{T}      # H × W
end

function RenderTarget(width::Int, height::Int; T=Float64)
    color = zeros(T, height, width, 3)
    depth = fill(T(Inf), height, width)
    RenderTarget{T}(width, height, color, depth)
end

function clear!(rt::RenderTarget, bg::Color3)
    rt.color[:, :, 1] .= bg.r
    rt.color[:, :, 2] .= bg.g
    rt.color[:, :, 3] .= bg.b
    rt.depth .= eltype(rt.depth)(Inf)
end

@inline _camera_near(c::AbstractCamera) = c.near

# Intersect view-space edge a→b with the near plane (view-space z = -near).
@inline function _clip_intersect_near(a::Vec4, b::Vec4, near)
    t = (-near - a.z) / (b.z - a.z)
    Vec4(a.x + t*(b.x - a.x), a.y + t*(b.y - a.y),
         a.z + t*(b.z - a.z), a.w + t*(b.w - a.w))
end

# Sutherland–Hodgman clip of a convex view-space polygon against the near plane,
# keeping the half-space z ≤ -near (in front of the camera's near plane). Writes
# the clipped vertices into `out` (reused buffer) and returns its length.
function _clip_near!(out::Vector{Vec4{T}}, verts::Vector{Vec4{T}}, n::Int, near) where T
    empty!(out)
    @inbounds for i in 1:n
        cur = verts[i]
        prv = verts[i == 1 ? n : i - 1]
        cur_in = cur.z <= -near
        prv_in = prv.z <= -near
        if cur_in
            prv_in || push!(out, _clip_intersect_near(prv, cur, near))
            push!(out, cur)
        elseif prv_in
            push!(out, _clip_intersect_near(prv, cur, near))
        end
    end
    return length(out)
end

# Rasterize one screen-space triangle (flat color) with z-buffer. `ylo`/`yhi`
# clamp the scanline range so a tiled renderer can restrict output to a band.
@inline function _rasterize_tri!(rt::RenderTarget, s1x, s1y, z1, s2x, s2y, z2,
                                 s3x, s3y, z3, fc::Color3, ylo::Int=1, yhi::Int=typemax(Int))
    W, H = rt.width, rt.height
    area = edge_function(s1x, s1y, s2x, s2y, s3x, s3y)
    abs(area) < 1e-10 && return nothing
    inv_area = 1.0 / area
    min_x = max(floor(Int, min(s1x, s2x, s3x)), 1)
    max_x = min(ceil(Int, max(s1x, s2x, s3x)), W)
    min_y = max(floor(Int, min(s1y, s2y, s3y)), 1, ylo)
    max_y = min(ceil(Int, max(s1y, s2y, s3y)), H, yhi)
    @inbounds for py in min_y:max_y
        for px in min_x:max_x
            cx = px - 0.5
            cy = py - 0.5
            w0 = edge_function(s2x, s2y, s3x, s3y, cx, cy) * inv_area
            w1 = edge_function(s3x, s3y, s1x, s1y, cx, cy) * inv_area
            w2 = edge_function(s1x, s1y, s2x, s2y, cx, cy) * inv_area
            if w0 >= 0 && w1 >= 0 && w2 >= 0
                z = w0 * z1 + w1 * z2 + w2 * z3
                if z < rt.depth[py, px]
                    rt.depth[py, px] = z
                    rt.color[py, px, 1] = fc.r
                    rt.color[py, px, 2] = fc.g
                    rt.color[py, px, 3] = fc.b
                end
            end
        end
    end
    return nothing
end

# ==========================================================================
# Smooth (per-pixel) shading path.
# Per-vertex world position and normal are interpolated perspective-correctly
# and the shading model is evaluated at every covered pixel, matching three.js
# smooth shading. Faces whose vertices share a normal (sphere, cylinder, …)
# render smoothly; faces with per-face normals (box) stay flat-faceted.
# ==========================================================================

struct ShadeVtx
    vp::Vec4{Float64}   # view-space position (for near-plane clipping)
    wp::Vec3{Float64}   # world-space position
    wn::Vec3{Float64}   # world-space normal (unnormalised; normalised per pixel)
end

@inline function _lerp_shadevtx(a::ShadeVtx, b::ShadeVtx, t)
    ShadeVtx(Vec4(a.vp.x + t*(b.vp.x - a.vp.x), a.vp.y + t*(b.vp.y - a.vp.y),
                  a.vp.z + t*(b.vp.z - a.vp.z), a.vp.w + t*(b.vp.w - a.vp.w)),
             Vec3(a.wp.x + t*(b.wp.x - a.wp.x), a.wp.y + t*(b.wp.y - a.wp.y), a.wp.z + t*(b.wp.z - a.wp.z)),
             Vec3(a.wn.x + t*(b.wn.x - a.wn.x), a.wn.y + t*(b.wn.y - a.wn.y), a.wn.z + t*(b.wn.z - a.wn.z)))
end

function _clip_near_attr!(out::Vector{ShadeVtx}, verts::Vector{ShadeVtx}, n::Int, near)
    empty!(out)
    @inbounds for i in 1:n
        cur = verts[i]
        prv = verts[i == 1 ? n : i - 1]
        cur_in = cur.vp.z <= -near
        prv_in = prv.vp.z <= -near
        if cur_in
            if !prv_in
                t = (-near - prv.vp.z) / (cur.vp.z - prv.vp.z)
                push!(out, _lerp_shadevtx(prv, cur, t))
            end
            push!(out, cur)
        elseif prv_in
            t = (-near - prv.vp.z) / (cur.vp.z - prv.vp.z)
            push!(out, _lerp_shadevtx(prv, cur, t))
        end
    end
    return length(out)
end

@inline function _rasterize_tri_smooth!(rt::RenderTarget,
        s1x, s1y, z1, iw1, wp1::Vec3, wn1::Vec3,
        s2x, s2y, z2, iw2, wp2::Vec3, wn2::Vec3,
        s3x, s3y, z3, iw3, wp3::Vec3, wn3::Vec3,
        material::AbstractMaterial, lights, cam_pos::Vec3, shadow_fn)
    W, H = rt.width, rt.height
    area = edge_function(s1x, s1y, s2x, s2y, s3x, s3y)
    abs(area) < 1e-10 && return nothing
    inv_area = 1.0 / area
    min_x = max(floor(Int, min(s1x, s2x, s3x)), 1)
    max_x = min(ceil(Int, max(s1x, s2x, s3x)), W)
    min_y = max(floor(Int, min(s1y, s2y, s3y)), 1)
    max_y = min(ceil(Int, max(s1y, s2y, s3y)), H)
    @inbounds for py in min_y:max_y
        for px in min_x:max_x
            cx = px - 0.5
            cy = py - 0.5
            b0 = edge_function(s2x, s2y, s3x, s3y, cx, cy) * inv_area
            b1 = edge_function(s3x, s3y, s1x, s1y, cx, cy) * inv_area
            b2 = edge_function(s1x, s1y, s2x, s2y, cx, cy) * inv_area
            (b0 >= 0 && b1 >= 0 && b2 >= 0) || continue
            z = b0 * z1 + b1 * z2 + b2 * z3
            z < rt.depth[py, px] || continue
            # Perspective-correct interpolation: weight by 1/w.
            iw = b0 * iw1 + b1 * iw2 + b2 * iw3
            a0 = b0 * iw1 / iw; a1 = b1 * iw2 / iw; a2 = b2 * iw3 / iw
            wp = Vec3(a0*wp1.x + a1*wp2.x + a2*wp3.x,
                      a0*wp1.y + a1*wp2.y + a2*wp3.y,
                      a0*wp1.z + a1*wp2.z + a2*wp3.z)
            wn = normalize(Vec3(a0*wn1.x + a1*wn2.x + a2*wn3.x,
                                a0*wn1.y + a1*wn2.y + a2*wn3.y,
                                a0*wn1.z + a1*wn2.z + a2*wn3.z))
            vd = normalize(cam_pos - wp)
            col = clamp_color(shade_face(wn, vd, wp, material, lights; shadow_fn=shadow_fn))
            rt.depth[py, px] = z
            rt.color[py, px, 1] = col.r
            rt.color[py, px, 2] = col.g
            rt.color[py, px, 3] = col.b
        end
    end
    return nothing
end

function _render_smooth!(rt::RenderTarget, meshes, lights, proj, view, near, cam_pos, shadow_fn=nothing)
    W, H = rt.width, rt.height
    tri = Vector{ShadeVtx}(undef, 3)
    clipped = ShadeVtx[]; sizehint!(clipped, 6)
    sx = Vector{Float64}(undef, 8); sy = Vector{Float64}(undef, 8)
    sz = Vector{Float64}(undef, 8); iw = Vector{Float64}(undef, 8)
    for mesh in meshes
        !is_visible(mesh) && continue
        world_mat = compute_world_matrix(mesh)
        modelview = view * world_mat
        normal_mat = mat4_transpose(mat4_inverse(world_mat))
        geo = mesh.geometry
        mat = mesh.material
        for fi in 1:geo.n_faces
            i1, i2, i3 = get_face(geo, fi)
            @inbounds for (slot, vi) in ((1, i1), (2, i2), (3, i3))
                v = get_vertex(geo, vi)
                nrm = get_normal(geo, vi)
                wp = mat4_transform_point(world_mat, v)
                wn = mat4_transform_direction(normal_mat, nrm)
                vp = mat4_transform_vec4(modelview, Vec4(v.x, v.y, v.z, 1.0))
                tri[slot] = ShadeVtx(vp, wp, wn)
            end
            m = _clip_near_attr!(clipped, tri, 3, near)
            m < 3 && continue
            @inbounds for k in 1:m
                cv = mat4_transform_vec4(proj, clipped[k].vp)
                invw = 1.0 / cv.w
                sx[k] = (cv.x * invw + 1) * 0.5 * W
                sy[k] = (1 - cv.y * invw) * 0.5 * H
                sz[k] = cv.z * invw
                iw[k] = invw
            end
            @inbounds for k in 2:(m - 1)
                _rasterize_tri_smooth!(rt,
                    sx[1], sy[1], sz[1], iw[1], clipped[1].wp, clipped[1].wn,
                    sx[k], sy[k], sz[k], iw[k], clipped[k].wp, clipped[k].wn,
                    sx[k+1], sy[k+1], sz[k+1], iw[k+1], clipped[k+1].wp, clipped[k+1].wn,
                    mat, lights, cam_pos, shadow_fn)
            end
        end
    end
    return rt
end

"""
    render!(rt, scene, camera; shading=:flat)

Render a scene with a camera into a RenderTarget using CPU rasterization.

Triangles are transformed to view space, clipped against the camera near plane
(so geometry straddling the camera renders its visible portion instead of
vanishing), then projected and rasterized with a z-buffer. Scratch buffers are
reused across faces to keep per-frame allocation bounded.

`shading=:flat` (default) evaluates one colour per face (fast). `shading=:smooth`
interpolates per-vertex world position and normal perspective-correctly and
shades each pixel, matching three.js smooth shading.
"""
function render!(rt::RenderTarget, scene::Scene, camera::AbstractCamera;
                 shading::Symbol=:flat, shadows::Bool=false, shadow_resolution::Int=512)
    clear!(rt, scene.background)

    proj = projection_matrix(camera)
    view = view_matrix(camera)
    near = _camera_near(camera)
    W, H = rt.width, rt.height

    (shading === :flat || shading === :smooth) ||
        throw(ArgumentError("shading must be :flat or :smooth, got :$shading"))

    meshes = collect_meshes(scene)
    lights = collect_lights(scene)
    shadow_fn = shadows ? _build_shadow_query(scene, lights; resolution=shadow_resolution) : nothing

    # Reused scratch buffers (bounded allocation per frame).
    tri = Vector{Vec4{Float64}}(undef, 3)
    clipped = Vector{Vec4{Float64}}(undef, 0)
    sizehint!(clipped, 6)
    sx = Vector{Float64}(undef, 8)
    sy = Vector{Float64}(undef, 8)
    sz = Vector{Float64}(undef, 8)

    # Opaque pass first (writes the depth buffer). Per-mesh shading mode honours
    # the mesh's `flat_shading` override, else the renderer default.
    transparent = Mesh[]
    smooth_meshes = Mesh[]
    for mesh in meshes
        !is_visible(mesh) && continue
        if is_transparent_material(mesh.material)
            push!(transparent, mesh)
        elseif _mesh_is_flat(mesh, shading)
            _rasterize_geo_flat!(rt, mesh.geometry, compute_world_matrix(mesh), mesh.material,
                                 lights, proj, view, near, camera.position, tri, clipped, sx, sy, sz;
                                 shadow_fn=shadow_fn)
        else
            push!(smooth_meshes, mesh)
        end
    end

    # InstancedMesh: same geometry/material drawn at each instance transform (flat).
    for im in collect_instanced(scene)
        !is_visible(im) && continue
        base = compute_world_matrix(im)
        for M in im.instance_matrices
            _rasterize_geo_flat!(rt, im.geometry, base * M, im.material,
                                 lights, proj, view, near, camera.position, tri, clipped, sx, sy, sz;
                                 shadow_fn=shadow_fn)
        end
    end

    # Smooth (per-pixel) opaque meshes share the same depth buffer.
    isempty(smooth_meshes) ||
        _render_smooth!(rt, smooth_meshes, lights, proj, view, near, camera.position, shadow_fn)

    # Transparent pass: back-to-front, z-tested against the opaque depth but not
    # writing depth, alpha-blended over the existing colour.
    if !isempty(transparent)
        sort!(transparent, by = m -> _mesh_view_depth(m, view))   # farthest first
        stamp = zeros(Int, H, W)          # ensures each pixel blends ≤ once per mesh
        sid = 0
        for mesh in transparent
            sid += 1
            α = material_opacity(mesh.material)
            _rasterize_geo_flat!(rt, mesh.geometry, compute_world_matrix(mesh), mesh.material,
                                 lights, proj, view, near, camera.position, tri, clipped, sx, sy, sz;
                                 alpha=α, stamp=stamp, stamp_id=sid, shadow_fn=shadow_fn)
        end
    end

    # Line and point primitives (depth-tested against the mesh passes).
    render_lines!(rt, scene, camera)
    render_points!(rt, scene, camera)

    return rt
end

# Effective shading for a mesh: its override if set, else the renderer default.
_mesh_is_flat(mesh::Mesh, default_shading::Symbol) =
    mesh.flat_shading === nothing ? (default_shading === :flat) : mesh.flat_shading

# Material face side (defaults to front-facing/back-culled, matching three.js).
material_side(m::AbstractMaterial) = hasfield(typeof(m), :side) ? getfield(m, :side) : :front

# View-space z of a mesh's world origin (more negative = farther from camera).
function _mesh_view_depth(mesh, view::Mat4)
    w = compute_world_matrix(mesh)
    o = mat4_transform_point(w, Vec3(0.0, 0.0, 0.0))
    v = mat4_transform_vec4(view, Vec4(o.x, o.y, o.z, 1.0))
    return v.z
end

# Material transparency helpers (some materials lack these fields).
material_opacity(m::AbstractMaterial) = hasfield(typeof(m), :opacity) ? getfield(m, :opacity) : 1.0
material_transparent(m::AbstractMaterial) = hasfield(typeof(m), :transparent) ? getfield(m, :transparent) : false
is_transparent_material(m::AbstractMaterial) = material_transparent(m) && material_opacity(m) < 1.0

# Rasterize one geometry (flat shading) at a given world transform, reusing the
# caller's scratch buffers. Shared by the mesh loop and the InstancedMesh loop.
function _rasterize_geo_flat!(rt::RenderTarget, geo, world_mat::Mat4, mat,
                              lights, proj::Mat4, view::Mat4, near, cam_pos::Vec3,
                              tri, clipped, sx, sy, sz;
                              alpha::Real=1.0, stamp=nothing, stamp_id::Int=0, shadow_fn=nothing,
                              ylo::Int=1, yhi::Int=typemax(Int), colorbuf=nothing)
    W, H = rt.width, rt.height
    modelview = view * world_mat
    face_colors = colorbuf === nothing ?
        shade_mesh_faces(geo, world_mat, mat, lights, cam_pos; shadow_fn=shadow_fn) :
        shade_mesh_faces!(colorbuf, geo, world_mat, mat, lights, cam_pos; shadow_fn=shadow_fn)
    blend = alpha < 1.0
    side = material_side(mat)
    normal_mat = side === :double ? world_mat : mat4_transpose(mat4_inverse(world_mat))
    has_normals = length(geo.normals) >= geo.n_vertices * 3
    for fi in 1:geo.n_faces
        i1, i2, i3 = get_face(geo, fi)
        v1 = get_vertex(geo, i1); v2 = get_vertex(geo, i2); v3 = get_vertex(geo, i3)
        # Back-face culling (skipped for double-sided materials).
        if side !== :double
            wc = mat4_transform_point(world_mat, Vec3((v1.x+v2.x+v3.x)/3, (v1.y+v2.y+v3.y)/3, (v1.z+v2.z+v3.z)/3))
            fn = _flat_face_normal(geo, i1, i2, i3,
                                   mat4_transform_point(world_mat, v1),
                                   mat4_transform_point(world_mat, v2),
                                   mat4_transform_point(world_mat, v3), normal_mat, has_normals)
            facing = dot(fn, cam_pos - wc)
            (side === :front ? facing <= 0 : facing > 0) && continue
        end
        tri[1] = mat4_transform_vec4(modelview, Vec4(v1.x, v1.y, v1.z, 1.0))
        tri[2] = mat4_transform_vec4(modelview, Vec4(v2.x, v2.y, v2.z, 1.0))
        tri[3] = mat4_transform_vec4(modelview, Vec4(v3.x, v3.y, v3.z, 1.0))

        m = _clip_near!(clipped, tri, 3, near)
        m < 3 && continue

        @inbounds for k in 1:m
            cv = mat4_transform_vec4(proj, clipped[k])
            invw = 1.0 / cv.w
            ndcx = cv.x * invw; ndcy = cv.y * invw; ndcz = cv.z * invw
            sx[k] = (ndcx + 1) * 0.5 * W
            sy[k] = (1 - ndcy) * 0.5 * H
            sz[k] = ndcz
        end

        fc = face_colors[fi]
        @inbounds for k in 2:(m - 1)        # fan-triangulate the clipped polygon
            if blend
                _rasterize_tri_blend!(rt, sx[1], sy[1], sz[1],
                                      sx[k], sy[k], sz[k],
                                      sx[k+1], sy[k+1], sz[k+1], fc, alpha, stamp, stamp_id)
            else
                _rasterize_tri!(rt, sx[1], sy[1], sz[1],
                                sx[k], sy[k], sz[k],
                                sx[k+1], sy[k+1], sz[k+1], fc, ylo, yhi)
            end
        end
    end
    return nothing
end

# Alpha-blend a triangle over the existing colour, z-tested but without writing
# depth (so transparent fragments don't occlude one another).
@inline function _rasterize_tri_blend!(rt::RenderTarget, s1x, s1y, z1, s2x, s2y, z2,
                                       s3x, s3y, z3, fc::Color3, alpha, stamp, stamp_id::Int)
    W, H = rt.width, rt.height
    area = edge_function(s1x, s1y, s2x, s2y, s3x, s3y)
    abs(area) < 1e-10 && return nothing
    inv_area = 1.0 / area
    min_x = max(floor(Int, min(s1x, s2x, s3x)), 1)
    max_x = min(ceil(Int, max(s1x, s2x, s3x)), W)
    min_y = max(floor(Int, min(s1y, s2y, s3y)), 1)
    max_y = min(ceil(Int, max(s1y, s2y, s3y)), H)
    ia = 1.0 - alpha
    use_stamp = stamp !== nothing
    @inbounds for py in min_y:max_y
        for px in min_x:max_x
            cx = px - 0.5; cy = py - 0.5
            w0 = edge_function(s2x, s2y, s3x, s3y, cx, cy) * inv_area
            w1 = edge_function(s3x, s3y, s1x, s1y, cx, cy) * inv_area
            w2 = edge_function(s1x, s1y, s2x, s2y, cx, cy) * inv_area
            if w0 >= 0 && w1 >= 0 && w2 >= 0
                # Skip pixels already blended by this same mesh (shared edges).
                use_stamp && stamp[py, px] == stamp_id && continue
                z = w0 * z1 + w1 * z2 + w2 * z3
                if z < rt.depth[py, px]
                    rt.color[py, px, 1] = fc.r * alpha + rt.color[py, px, 1] * ia
                    rt.color[py, px, 2] = fc.g * alpha + rt.color[py, px, 2] * ia
                    rt.color[py, px, 3] = fc.b * alpha + rt.color[py, px, 3] * ia
                    use_stamp && (stamp[py, px] = stamp_id)
                end
            end
        end
    end
    return nothing
end

"""
Edge function for barycentric coordinate computation.
Positive if (px,py) is on the left of line from (ax,ay) to (bx,by).
"""
@inline function edge_function(ax, ay, bx, by, px, py)
    (px - ax) * (by - ay) - (py - ay) * (bx - ax)
end

"""
Convert RenderTarget color buffer to a flat RGB array suitable for image export.
Returns Matrix{UInt8} of size (H, W*3) or Array{UInt8, 3} of size (H, W, 3).
"""
function render_to_rgb8(rt::RenderTarget)
    H, W = rt.height, rt.width
    img = Array{UInt8}(undef, H, W, 3)
    for j in 1:W
        for i in 1:H
            img[i, j, 1] = round(UInt8, clamp(rt.color[i, j, 1], 0, 1) * 255)
            img[i, j, 2] = round(UInt8, clamp(rt.color[i, j, 2], 0, 1) * 255)
            img[i, j, 3] = round(UInt8, clamp(rt.color[i, j, 3], 0, 1) * 255)
        end
    end
    return img
end
