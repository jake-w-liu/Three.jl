# --------------------------------------------------------------------------
# Shadow mapping: render scene depth from a light's viewpoint, then test shaded
# points against that depth map. Directional lights use an orthographic light
# camera sized to the scene; spot/point lights use a perspective light camera.
# --------------------------------------------------------------------------

struct ShadowMap
    depth::Matrix{Float64}      # NDC depth from the light (smaller = nearer light); Inf = empty
    light_vp::Mat4{Float64}     # light view-projection
    bias::Float64
end

# World-space bounding sphere of all meshes under the scene.
function _scene_bounds(meshes)
    box = Box3()
    for mesh in meshes
        is_visible(mesh) || continue
        wm = compute_world_matrix(mesh); geo = mesh.geometry
        for vi in 1:geo.n_vertices
            box = box3_expand_by_point(box, mat4_transform_point(wm, get_vertex(geo, vi)))
        end
    end
    center = (box.min + box.max) * 0.5
    radius = norm(box.max - center)
    return (center, max(radius, 1e-3))
end

_safe_up(dir::Vec3) = abs(dir.y) > 0.99 ? Vec3(1.0, 0.0, 0.0) : Vec3(0.0, 1.0, 0.0)

function _light_view_proj(light::DirectionalLight, center::Vec3, radius)
    dir = normalize(light.position - light.target)        # toward the light
    eye = center + dir * (radius * 2.0)
    lv = mat4_look_at(eye, center, _safe_up(dir))
    s = radius * 1.1
    lp = mat4_orthographic(-s, s, -s, s, 0.01, radius * 4.0)
    return lp * lv
end

function _light_view_proj(light::SpotLight, center::Vec3, radius)
    lv = mat4_look_at(light.position, light.target, _safe_up(normalize(light.target - light.position)))
    lp = mat4_perspective(min(light.angle * 2.0, π * 0.9), 1.0, 0.05, radius * 6.0)
    return lp * lv
end

function _light_view_proj(light::PointLight, center::Vec3, radius)
    dir = normalize(center - light.position)
    lv = mat4_look_at(light.position, center, _safe_up(dir))
    lp = mat4_perspective(π/2, 1.0, 0.05, radius * 6.0)
    return lp * lv
end

# Depth-only triangle rasterization into the shadow buffer.
@inline function _raster_depth!(depth::Matrix{Float64}, W, H,
                                s1x, s1y, z1, s2x, s2y, z2, s3x, s3y, z3)
    area = edge_function(s1x, s1y, s2x, s2y, s3x, s3y)
    abs(area) < 1e-12 && return nothing
    inv_area = 1.0 / area
    min_x = max(floor(Int, min(s1x, s2x, s3x)), 1)
    max_x = min(ceil(Int, max(s1x, s2x, s3x)), W)
    min_y = max(floor(Int, min(s1y, s2y, s3y)), 1)
    max_y = min(ceil(Int, max(s1y, s2y, s3y)), H)
    @inbounds for py in min_y:max_y, px in min_x:max_x
        cx = px - 0.5; cy = py - 0.5
        w0 = edge_function(s2x, s2y, s3x, s3y, cx, cy) * inv_area
        w1 = edge_function(s3x, s3y, s1x, s1y, cx, cy) * inv_area
        w2 = edge_function(s1x, s1y, s2x, s2y, cx, cy) * inv_area
        if w0 >= 0 && w1 >= 0 && w2 >= 0
            z = w0 * z1 + w1 * z2 + w2 * z3
            z < depth[py, px] && (depth[py, px] = z)
        end
    end
    return nothing
end

"""
    compute_shadow_map(scene, light; resolution=512, bias=3e-3)

Render the scene's depth from `light`'s viewpoint into a [`ShadowMap`].
"""
function compute_shadow_map(scene, light; resolution::Int=512, bias=3e-3)
    meshes = collect_meshes(scene)
    center, radius = _scene_bounds(meshes)
    vp = _light_view_proj(light, center, radius)
    W = H = resolution
    depth = fill(Inf, H, W)
    for mesh in meshes
        is_visible(mesh) || continue
        wm = compute_world_matrix(mesh); geo = mesh.geometry
        mvp = vp * wm
        for fi in 1:geo.n_faces
            i1, i2, i3 = get_face(geo, fi)
            c1 = mat4_transform_vec4(mvp, _vh(get_vertex(geo, i1)))
            c2 = mat4_transform_vec4(mvp, _vh(get_vertex(geo, i2)))
            c3 = mat4_transform_vec4(mvp, _vh(get_vertex(geo, i3)))
            (c1.w <= 1e-6 || c2.w <= 1e-6 || c3.w <= 1e-6) && continue
            _raster_depth!(depth, W, H,
                (c1.x/c1.w+1)*0.5*W, (1-c1.y/c1.w)*0.5*H, c1.z/c1.w,
                (c2.x/c2.w+1)*0.5*W, (1-c2.y/c2.w)*0.5*H, c2.z/c2.w,
                (c3.x/c3.w+1)*0.5*W, (1-c3.y/c3.w)*0.5*H, c3.z/c3.w)
        end
    end
    return ShadowMap(depth, vp, bias)
end

@inline _vh(v::Vec3) = Vec4(v.x, v.y, v.z, 1.0)

"""Visibility of a world-space point from the shadow map's light: 1 lit, 0 occluded."""
function shadow_visibility(sm::ShadowMap, p::Vec3)
    c = mat4_transform_vec4(sm.light_vp, _vh(p))
    c.w <= 1e-6 && return 1.0
    ndcx = c.x / c.w; ndcy = c.y / c.w; ndcz = c.z / c.w
    (abs(ndcx) > 1 || abs(ndcy) > 1) && return 1.0          # outside the light frustum
    H, W = size(sm.depth)
    px = clamp(floor(Int, (ndcx + 1) * 0.5 * W) + 1, 1, W)
    py = clamp(floor(Int, (1 - ndcy) * 0.5 * H) + 1, 1, H)
    d = sm.depth[py, px]
    isinf(d) && return 1.0
    return ndcz - sm.bias > d ? 0.0 : 1.0                   # something nearer the light occludes
end

# Build shadow maps for all shadow-casting lights and a closure to query them.
function _build_shadow_query(scene, lights; resolution::Int=512)
    maps = IdDict{AbstractLight, ShadowMap}()
    for light in lights
        if !_is_fill_light(light) && hasfield(typeof(light), :cast_shadow) && getfield(light, :cast_shadow)
            maps[light] = compute_shadow_map(scene, light; resolution=resolution)
        end
    end
    isempty(maps) && return nothing
    return (light, p) -> haskey(maps, light) ? shadow_visibility(maps[light], p) : 1.0
end
