# --------------------------------------------------------------------------
# Raycaster: ray-mesh intersection for picking and queries (three.js Raycaster).
# Uses the Möller–Trumbore algorithm against world-space triangles.
# --------------------------------------------------------------------------

struct Intersection
    distance::Float64
    point::Vec3{Float64}
    object::AbstractObject3D
    face_index::Int
end

"""
Möller–Trumbore ray/triangle test. Returns the ray parameter `t > 0` at the
hit, or `nothing` if the ray misses. `dir` need not be normalised; `t` is then
in units of `dir`.
"""
function ray_triangle_intersect(origin::Vec3, dir::Vec3, a::Vec3, b::Vec3, c::Vec3; eps=1e-9)
    e1 = b - a; e2 = c - a
    p = cross(dir, e2)
    det = dot(e1, p)
    abs(det) < eps && return nothing             # ray parallel to triangle
    inv_det = 1 / det
    tvec = origin - a
    u = dot(tvec, p) * inv_det
    (u < 0 || u > 1) && return nothing
    q = cross(tvec, e1)
    v = dot(dir, q) * inv_det
    (v < 0 || u + v > 1) && return nothing
    t = dot(e2, q) * inv_det
    return t > eps ? t : nothing
end

mutable struct Raycaster
    ray::Ray{Float64}
    near::Float64
    far::Float64
end
Raycaster(origin::Vec3, dir::Vec3; near=0.0, far=Inf) =
    Raycaster(Ray(origin, normalize(dir)), near, far)

# Build the world-space ray through normalized device coords (x,y ∈ [-1,1]).
function _camera_ray(camera::AbstractCamera, ndc_x, ndc_y)
    inv_vp = mat4_inverse(projection_matrix(camera) * view_matrix(camera))
    p_near = mat4_transform_point(inv_vp, Vec3(ndc_x, ndc_y, -1.0))
    p_far  = mat4_transform_point(inv_vp, Vec3(ndc_x, ndc_y, 1.0))
    Ray(camera.position, normalize(p_far - p_near))
end

"""Aim the raycaster through screen NDC `(x,y)` from a camera (three.js `setFromCamera`)."""
function set_from_camera!(rc::Raycaster, camera::AbstractCamera, ndc_x, ndc_y)
    rc.ray = _camera_ray(camera, ndc_x, ndc_y)
    return rc
end

"""
Intersect the ray with every mesh under `root`, returning `Intersection`s
sorted by distance (nearest first), filtered to `[near, far]`.
"""
function raycast(rc::Raycaster, root::AbstractObject3D)
    hits = Intersection[]
    o = rc.ray.origin; d = rc.ray.direction
    for mesh in collect_meshes(root)
        is_visible(mesh) || continue
        wm = compute_world_matrix(mesh)
        geo = mesh.geometry
        @inbounds for fi in 1:geo.n_faces
            i1, i2, i3 = get_face(geo, fi)
            a = mat4_transform_point(wm, get_vertex(geo, i1))
            b = mat4_transform_point(wm, get_vertex(geo, i2))
            c = mat4_transform_point(wm, get_vertex(geo, i3))
            t = ray_triangle_intersect(o, d, a, b, c)
            if t !== nothing && rc.near <= t <= rc.far
                push!(hits, Intersection(t, o + d * t, mesh, fi))
            end
        end
    end
    sort!(hits, by = h -> h.distance)
    return hits
end
