# --------------------------------------------------------------------------
# Differentiable soft rasterizer (Liu et al., ICCV 2019 inspired).
#
# Key idea: replace hard z-buffer with soft aggregation.
#   - Soft coverage: sigmoid of signed distance to triangle edge
#   - Soft depth: softmax over face depths
#   - Result: fully differentiable image w.r.t. vertex positions,
#     material parameters, light parameters, and camera parameters.
#
# All operations are pure Julia scalar math — ForwardDiff compatible.
# --------------------------------------------------------------------------

"""Configuration for the soft rasterizer."""
struct SoftRasterizerConfig{T<:Real}
    sigma::T       # Edge softness in pixel units (larger = softer)
    gamma::T       # Depth aggregation temperature in NDC units
    bg_color::Color3{T}
    eps::T
end

function SoftRasterizerConfig(; sigma=1e-2, gamma=1.0,
                               bg_color=Color3(0.0, 0.0, 0.0),
                               eps=1e-8)
    T = promote_type(typeof(sigma), typeof(gamma), typeof(bg_color.r), typeof(eps))
    SoftRasterizerConfig{T}(T(sigma), T(gamma),
                            Color3(T(bg_color.r), T(bg_color.g), T(bg_color.b)), T(eps))
end

"""
Differentiable soft rendering.
Takes explicit arrays rather than scene-graph objects for AD compatibility.

Arguments:
- vertices: Vector{Vec3{T}} — world-space vertex positions
- faces: Vector{NTuple{3,Int}} — triangle face indices (1-based)
- face_colors: Vector{Color3{T}} — one color per face
- view_proj: Mat4{T} — combined view-projection matrix
- width, height: image dimensions
- config: SoftRasterizerConfig

Returns: Array{T, 3} of size (height, width, 3) — RGB image.
"""
function soft_render(vertices::Vector{Vec3{T}},
                     faces::Vector{NTuple{3,Int}},
                     face_colors::Vector{Color3{T}},
                     view_proj::Mat4,
                     width::Int, height::Int,
                     config::SoftRasterizerConfig = SoftRasterizerConfig()
                     ) where T

    σ = config.sigma
    γ = config.gamma
    bg = config.bg_color
    eps = config.eps

    n_faces = length(faces)
    W, H = width, height

    # Project all vertices to screen space
    screen_verts = Vector{Vec3{T}}(undef, length(vertices))
    ndc_verts = Vector{Vec3{T}}(undef, length(vertices))
    for (vi, v) in enumerate(vertices)
        clip = mat4_transform_vec4(view_proj, Vec4(v.x, v.y, v.z, one(T)))
        if clip.w > eps
            ndc = Vec3(clip.x / clip.w, clip.y / clip.w, clip.z / clip.w)
            ndc_verts[vi] = ndc
            sx = (ndc.x + 1) * T(0.5) * W
            sy = (1 - ndc.y) * T(0.5) * H
            screen_verts[vi] = Vec3(sx, sy, ndc.z)
        else
            ndc_verts[vi] = Vec3(zero(T), zero(T), T(1000))
            screen_verts[vi] = Vec3(-T(1000), -T(1000), T(1000))
        end
    end

    # Precompute screen-space triangles as NamedTuples
    screen_tris = Vector{NamedTuple{(:s1,:s2,:s3,:color,:min_x,:max_x,:min_y,:max_y,:area,:valid),
                                     Tuple{Vec3{T},Vec3{T},Vec3{T},Color3{T},Int,Int,Int,Int,T,Bool}}}(undef, n_faces)

    for fi in 1:n_faces
        i1, i2, i3 = faces[fi]
        s1 = screen_verts[i1]
        s2 = screen_verts[i2]
        s3 = screen_verts[i3]
        area = edge_function(s1.x, s1.y, s2.x, s2.y, s3.x, s3.y)

        # Extended bounding box for soft edges
        margin = 3.0 / max(σ, eps)
        margin = min(margin, 50.0)  # cap to prevent huge bboxes
        bmin_x = max(floor(Int, min(s1.x, s2.x, s3.x) - margin), 1)
        bmax_x = min(ceil(Int, max(s1.x, s2.x, s3.x) + margin), W)
        bmin_y = max(floor(Int, min(s1.y, s2.y, s3.y) - margin), 1)
        bmax_y = min(ceil(Int, max(s1.y, s2.y, s3.y) + margin), H)

        valid = abs(area) > eps && bmax_x >= bmin_x && bmax_y >= bmin_y
        screen_tris[fi] = (s1=s1, s2=s2, s3=s3, color=face_colors[fi],
                           min_x=bmin_x, max_x=bmax_x, min_y=bmin_y, max_y=bmax_y,
                           area=area, valid=valid)
    end

    # Render each pixel
    image = Array{T}(undef, H, W, 3)
    for py in 1:H
        for px in 1:W
            cx = T(px) - T(0.5)
            cy = T(py) - T(0.5)

            # Accumulate soft contributions from all faces
            total_weight = zero(T)
            color_r = zero(T)
            color_g = zero(T)
            color_b = zero(T)

            for fi in 1:n_faces
                tri = screen_tris[fi]
                !tri.valid && continue
                !(tri.min_x <= px <= tri.max_x && tri.min_y <= py <= tri.max_y) && continue

                # Signed distance to triangle (minimum distance to any edge)
                d = signed_distance_to_triangle(cx, cy,
                    tri.s1.x, tri.s1.y, tri.s2.x, tri.s2.y, tri.s3.x, tri.s3.y)

                # Soft coverage via sigmoid
                coverage = sigmoid_approx(d / σ)

                # Depth-based weighting
                z_face = (tri.s1.z + tri.s2.z + tri.s3.z) / 3
                depth_weight = exp(-z_face / γ)

                w = coverage * depth_weight
                total_weight += w
                color_r += w * tri.color.r
                color_g += w * tri.color.g
                color_b += w * tri.color.b
            end

            if total_weight > eps
                inv_w = one(T) / total_weight
                # Blend with background
                alpha = one(T) - exp(-total_weight)
                image[py, px, 1] = alpha * color_r * inv_w + (one(T) - alpha) * bg.r
                image[py, px, 2] = alpha * color_g * inv_w + (one(T) - alpha) * bg.g
                image[py, px, 3] = alpha * color_b * inv_w + (one(T) - alpha) * bg.b
            else
                image[py, px, 1] = bg.r
                image[py, px, 2] = bg.g
                image[py, px, 3] = bg.b
            end
        end
    end

    return image
end

"""
Sigmoid approximation — smooth and AD-friendly.
"""
@inline function sigmoid_approx(x::T) where T
    one(T) / (one(T) + exp(-x))
end

"""
Signed distance from point (px,py) to triangle defined by (ax,ay), (bx,by), (cx,cy).
Positive inside, negative outside.
"""
function signed_distance_to_triangle(px, py, ax, ay, bx, by, cx, cy)
    # Barycentric test
    area = edge_function(ax, ay, bx, by, cx, cy)
    if abs(area) < 1e-20
        return typeof(px)(-1e10)
    end

    w0 = edge_function(bx, by, cx, cy, px, py) / area
    w1 = edge_function(cx, cy, ax, ay, px, py) / area
    w2 = edge_function(ax, ay, bx, by, px, py) / area

    if w0 >= 0 && w1 >= 0 && w2 >= 0
        # Inside — distance is min distance to any edge
        d1 = point_line_distance(px, py, ax, ay, bx, by)
        d2 = point_line_distance(px, py, bx, by, cx, cy)
        d3 = point_line_distance(px, py, cx, cy, ax, ay)
        return min(d1, d2, d3)
    else
        # Outside — negative of min distance to edges/vertices
        d1 = point_segment_distance(px, py, ax, ay, bx, by)
        d2 = point_segment_distance(px, py, bx, by, cx, cy)
        d3 = point_segment_distance(px, py, cx, cy, ax, ay)
        return -min(d1, d2, d3)
    end
end

"""
Distance from point to infinite line through (ax,ay)-(bx,by).
"""
@inline function point_line_distance(px, py, ax, ay, bx, by)
    dx = bx - ax
    dy = by - ay
    len = sqrt(dx^2 + dy^2)
    abs((px - ax) * dy - (py - ay) * dx) / max(len, 1e-20)
end

"""
Distance from point to line segment (ax,ay)-(bx,by).
AD-friendly: uses smooth min/max via clamping.
"""
function point_segment_distance(px, py, ax, ay, bx, by)
    dx = bx - ax
    dy = by - ay
    len_sq = dx^2 + dy^2
    if len_sq < 1e-20
        return sqrt((px - ax)^2 + (py - ay)^2)
    end
    t = clamp(((px - ax)*dx + (py - ay)*dy) / len_sq, zero(px), one(px))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    sqrt((px - proj_x)^2 + (py - proj_y)^2)
end

# ========================== High-level differentiable render ==========================

"""
Differentiable render of a scene — extracts geometry data and calls soft_render.
Suitable for wrapping in ForwardDiff.
"""
function soft_render_scene(scene::Scene, camera::AbstractCamera,
                           width::Int, height::Int;
                           sigma=1e-4, gamma=1e-4)
    config = SoftRasterizerConfig(sigma=sigma, gamma=gamma, bg_color=scene.background)

    proj = projection_matrix(camera)
    view = view_matrix(camera)
    vp = proj * view

    meshes = collect_meshes(scene)
    lights = collect_lights(scene)

    all_verts = Vec3{Float64}[]
    all_faces = NTuple{3,Int}[]
    all_colors = Color3{Float64}[]
    vert_offset = 0

    for mesh in meshes
        !is_visible(mesh) && continue
        world_mat = compute_world_matrix(mesh)
        geo = mesh.geometry

        # Transform vertices to world space
        for vi in 1:geo.n_vertices
            v = get_vertex(geo, vi)
            wv = mat4_transform_point(world_mat, v)
            push!(all_verts, wv)
        end

        # Compute face colors
        face_colors = shade_mesh_faces(geo, world_mat, mesh.material, lights, camera.position)

        for fi in 1:geo.n_faces
            i1, i2, i3 = get_face(geo, fi)
            push!(all_faces, (i1 + vert_offset, i2 + vert_offset, i3 + vert_offset))
            push!(all_colors, face_colors[fi])
        end
        vert_offset += geo.n_vertices
    end

    soft_render(all_verts, all_faces, all_colors, vp, width, height, config)
end

"""
Differentiable render with explicit parameters for AD.
`params` is a flat vector of parameters being optimized.
`param_injector!` is a function that injects params into the scene/camera before rendering.
"""
function differentiable_render(params::AbstractVector{T},
                               setup_fn::Function,
                               width::Int, height::Int;
                               sigma=1e-4, gamma=1e-4) where T
    # setup_fn returns (vertices, faces, face_colors, view_proj, bg_color)
    vertices, faces, face_colors, vp, bg = setup_fn(params)
    config = SoftRasterizerConfig(sigma=sigma, gamma=gamma, bg_color=bg)
    soft_render(vertices, faces, face_colors, vp, width, height, config)
end
