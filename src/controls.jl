# --------------------------------------------------------------------------
# Controls, animation, and helpers (three.js examples/ counterparts), adapted
# for headless/programmatic use: camera manipulators, a Clock, keyframe
# animation, and line-based scene helpers.
# --------------------------------------------------------------------------

# ========================== Camera controls ==========================

mutable struct OrbitControls
    camera::PerspectiveCamera
    target::Vec3{Float64}
    # Damping/inertia (three.js OrbitControls.enableDamping). When enabled, each
    # interaction adds to a residual velocity that `orbit_update!` applies and
    # decays. Backward-compatible: defaults reproduce the original immediate
    # (non-damped) behaviour.
    enable_damping::Bool
    damping_factor::Float64
    # Internal residual velocity accumulated during the last interaction.
    v_azimuth::Float64       # pending azimuth delta (rad)
    v_polar::Float64         # pending polar delta (rad)
    v_zoom::Float64          # pending log-scale dolly (radius *= exp(v_zoom))
    v_pan::Vec3{Float64}     # pending world-space pan offset
end

# Positional/keyword constructors. The two original positional forms still work;
# damping fields are keyword-only with three.js-matching defaults.
function OrbitControls(cam::PerspectiveCamera, target::Vec3{Float64};
                       enable_damping::Bool=false, damping_factor::Real=0.05)
    OrbitControls(cam, target, enable_damping, Float64(damping_factor),
                  0.0, 0.0, 0.0, Vec3(0.0, 0.0, 0.0))
end
OrbitControls(cam::PerspectiveCamera; enable_damping::Bool=false, damping_factor::Real=0.05) =
    OrbitControls(cam, cam.target; enable_damping=enable_damping, damping_factor=damping_factor)

# Current spherical (radius, polar from +y, azimuth) of the camera about target.
_orbit_spherical(oc::OrbitControls) = cartesian_to_spherical(oc.camera.position - oc.target)

function _orbit_apply!(oc::OrbitControls, s::Spherical)
    oc.camera.position = oc.target + spherical_to_cartesian(s)
    oc.camera.target = oc.target
    return oc
end

"""Place the camera at an absolute orbit (azimuth, polar, radius) about the target."""
function orbit_set!(oc::OrbitControls; azimuth, polar, radius)
    _orbit_apply!(oc, Spherical(radius, clamp(polar, 1e-4, π - 1e-4), azimuth))
end

# Immediate (non-damped) primitives that always apply their deltas at once.
function _orbit_rotate_now!(oc::OrbitControls, d_azimuth, d_polar)
    s = _orbit_spherical(oc)
    _orbit_apply!(oc, Spherical(s.radius, clamp(s.phi + d_polar, 1e-4, π - 1e-4), s.theta + d_azimuth))
end
function _orbit_zoom_now!(oc::OrbitControls, factor)
    s = _orbit_spherical(oc)
    _orbit_apply!(oc, Spherical(s.radius * factor, s.phi, s.theta))
end
function _orbit_pan_now!(oc::OrbitControls, dx, dy)
    fwd = normalize(oc.target - oc.camera.position)
    right = normalize(cross(fwd, oc.camera.up))
    up = cross(right, fwd)
    shift = right * dx + up * dy
    oc.target = oc.target + shift
    oc.camera.position = oc.camera.position + shift
    oc.camera.target = oc.target
    return oc
end

"""
Rotate the camera around the target by angular deltas (radians).

With `enable_damping=false` (default) the rotation is applied immediately, exactly
as before. With damping enabled the deltas are added to the residual angular
velocity instead, to be consumed by `orbit_update!`.
"""
function orbit_rotate!(oc::OrbitControls, d_azimuth, d_polar)
    if oc.enable_damping
        oc.v_azimuth += d_azimuth
        oc.v_polar += d_polar
        return oc
    end
    return _orbit_rotate_now!(oc, d_azimuth, d_polar)
end

"""
Scale the orbit radius (dolly). `factor < 1` zooms in.

With damping enabled the dolly accumulates as a residual log-scale velocity
consumed by `orbit_update!`; non-damped behaviour is unchanged.
"""
function orbit_zoom!(oc::OrbitControls, factor)
    if oc.enable_damping
        oc.v_zoom += log(factor)
        return oc
    end
    return _orbit_zoom_now!(oc, factor)
end

"""
Pan the target (and camera) in the camera's right/up plane.

With damping enabled the pan offset accumulates as a residual world-space
velocity consumed by `orbit_update!`; non-damped behaviour is unchanged.
"""
function orbit_pan!(oc::OrbitControls, dx, dy)
    if oc.enable_damping
        fwd = normalize(oc.target - oc.camera.position)
        right = normalize(cross(fwd, oc.camera.up))
        up = cross(right, fwd)
        oc.v_pan = oc.v_pan + right * dx + up * dy
        return oc
    end
    return _orbit_pan_now!(oc, dx, dy)
end

"""
    orbit_update!(oc)

Advance the orbit by one frame. With damping disabled this is a no-op (state is
already current). With damping enabled it applies the residual rotation, dolly,
and pan velocities, then decays them by `(1 - damping_factor)` so motion eases
to a stop after the last interaction (three.js `OrbitControls.update`). Velocity
components below a small threshold are zeroed to avoid endless drift.
"""
function orbit_update!(oc::OrbitControls)
    oc.enable_damping || return oc
    decay = 1.0 - oc.damping_factor
    # Apply the accumulated residual velocities for this frame.
    if oc.v_azimuth != 0.0 || oc.v_polar != 0.0
        _orbit_rotate_now!(oc, oc.v_azimuth, oc.v_polar)
    end
    if oc.v_zoom != 0.0
        _orbit_zoom_now!(oc, exp(oc.v_zoom))
    end
    if oc.v_pan.x != 0.0 || oc.v_pan.y != 0.0 || oc.v_pan.z != 0.0
        # Pan velocity is already a world-space offset; apply it directly.
        oc.target = oc.target + oc.v_pan
        oc.camera.position = oc.camera.position + oc.v_pan
        oc.camera.target = oc.target
    end
    # Decay residual velocities; snap tiny remnants to zero.
    thresh = 1e-9
    oc.v_azimuth = abs(oc.v_azimuth * decay) < thresh ? 0.0 : oc.v_azimuth * decay
    oc.v_polar   = abs(oc.v_polar   * decay) < thresh ? 0.0 : oc.v_polar   * decay
    oc.v_zoom    = abs(oc.v_zoom    * decay) < thresh ? 0.0 : oc.v_zoom    * decay
    pan = oc.v_pan * decay
    oc.v_pan = (abs(pan.x) < thresh && abs(pan.y) < thresh && abs(pan.z) < thresh) ?
        Vec3(0.0, 0.0, 0.0) : pan
    return oc
end

# TrackballControls: free rotation about the target via screen-space deltas.
mutable struct TrackballControls
    camera::PerspectiveCamera
    target::Vec3{Float64}
end
TrackballControls(cam::PerspectiveCamera) = TrackballControls(cam, cam.target)

function trackball_rotate!(tc::TrackballControls, dx, dy)
    oc = OrbitControls(tc.camera, tc.target)
    orbit_rotate!(oc, dx, dy)
    tc.camera = oc.camera
    return tc
end

# FlyControls: first-person translation/rotation along the camera basis.
mutable struct FlyControls
    camera::PerspectiveCamera
end

"""Translate the camera (and its target) along forward/right/up axes."""
function fly_translate!(fc::FlyControls, forward, right, up)
    cam = fc.camera
    f = normalize(cam.target - cam.position)
    r = normalize(cross(f, cam.up))
    u = cross(r, f)
    shift = f * forward + r * right + u * up
    cam.position = cam.position + shift
    cam.target = cam.target + shift
    return fc
end

"""Yaw/pitch the camera in place (target orbits around the camera position)."""
function fly_rotate!(fc::FlyControls, yaw, pitch)
    cam = fc.camera
    dist = norm(cam.target - cam.position)
    dir = normalize(cam.target - cam.position)
    s = cartesian_to_spherical(dir)
    s2 = Spherical(1.0, clamp(s.phi - pitch, 1e-4, π - 1e-4), s.theta + yaw)
    cam.target = cam.position + spherical_to_cartesian(s2) * dist
    return fc
end

# ========================== Clock ==========================

mutable struct Clock
    start_time::Float64
    last_time::Float64
    running::Bool
end
Clock() = (t = time(); Clock(t, t, true))

clock_elapsed(c::Clock, now=time()) = now - c.start_time
function clock_delta!(c::Clock, now=time())
    d = now - c.last_time
    c.last_time = now
    return d
end

# ========================== Animation ==========================

abstract type AbstractKeyframeTrack end

# Scalar/Vec3 property track. `interpolation` selects the sampling mode:
#   :linear (default)  — piecewise linear, byte-identical to the original path.
#   :cubic             — Catmull-Rom spline through the keyframes.
struct KeyframeTrack <: AbstractKeyframeTrack
    target::AbstractObject3D
    property::Symbol                 # e.g. :position, :scale
    times::Vector{Float64}
    values::Vector{Vec3{Float64}}
    interpolation::Symbol            # :linear | :cubic
end

# Backward-compatible four-argument constructor: the original positional call
# `KeyframeTrack(target, property, times, values)` keeps working and defaults to
# linear; `interpolation=:cubic` selects the spline mode.
KeyframeTrack(target::AbstractObject3D, property::Symbol,
              times::Vector{Float64}, values::Vector{Vec3{Float64}};
              interpolation::Symbol=:linear) =
    KeyframeTrack(target, property, times, values, interpolation)

# Dedicated rotation track: quaternion keyframes interpolated with slerp between
# adjacent frames (three.js `QuaternionKeyframeTrack`), rather than componentwise
# linear blending which would not stay on the unit sphere.
struct QuaternionKeyframeTrack <: AbstractKeyframeTrack
    target::AbstractObject3D
    property::Symbol                          # e.g. :quaternion
    times::Vector{Float64}
    values::Vector{Quaternion{Float64}}
end

# Catmull-Rom spline tangent: derivative estimated from the neighbouring
# keyframes, clamped at the endpoints (matches three.js CubicInterpolant on a
# uniform-Catmull-Rom basis). Works for Real and Vec3 values.
@inline _cr_sub(a::Real, b::Real) = a - b
@inline _cr_sub(a::Vec3, b::Vec3) = a - b
@inline _cr_scale(a::Real, s) = a * s
@inline _cr_scale(a::Vec3, s) = a * s
@inline _cr_add(a::Real, b::Real) = a + b
@inline _cr_add(a::Vec3, b::Vec3) = a + b

# Hermite/Catmull-Rom evaluation between p1 (at α=0) and p2 (at α=1) with
# tangents m1, m2 (already scaled to the local segment), parameter α∈[0,1].
@inline function _hermite(p1, p2, m1, m2, α)
    α2 = α * α
    α3 = α2 * α
    h00 = 2α3 - 3α2 + 1
    h10 = α3 - 2α2 + α
    h01 = -2α3 + 3α2
    h11 = α3 - α2
    _cr_add(_cr_add(_cr_scale(p1, h00), _cr_scale(m1, h10)),
            _cr_add(_cr_scale(p2, h01), _cr_scale(m2, h11)))
end

"""
    interpolate_catmull_rom(times, values, t)

Catmull-Rom spline sample of `values` at sorted `times`, evaluated at `t`,
clamped to the endpoints. Tangents use the centripetal/uniform Catmull-Rom rule
on possibly non-uniform `times`; endpoint tangents are one-sided. `values` may be
reals or `Vec3`.
"""
function interpolate_catmull_rom(times::AbstractVector, values::AbstractVector, t)
    n = length(times)
    @assert n == length(values) && n >= 1 "times and values must align and be non-empty"
    t <= times[1] && return values[1]
    t >= times[n] && return values[n]
    n == 1 && return values[1]
    hi = searchsortedfirst(times, t)
    lo = hi - 1
    h = times[hi] - times[lo]
    α = (t - times[lo]) / h
    p1 = values[lo]; p2 = values[hi]
    # One-sided tangents at the ends, central tangents in the interior, scaled by
    # the local interval h so the Hermite basis matches the parameter spacing.
    if lo > 1
        m1 = _cr_scale(_cr_sub(values[hi], values[lo-1]), h / (times[hi] - times[lo-1]))
    else
        m1 = _cr_sub(values[hi], values[lo])           # forward difference
    end
    if hi < n
        m2 = _cr_scale(_cr_sub(values[hi+1], values[lo]), h / (times[hi+1] - times[lo]))
    else
        m2 = _cr_sub(values[hi], values[lo])           # backward difference
    end
    return _hermite(p1, p2, m1, m2, α)
end

# Sample a single track value at absolute time `t` according to its mode.
_track_value(tr::KeyframeTrack, t) =
    tr.interpolation === :cubic ? interpolate_catmull_rom(tr.times, tr.values, t) :
                                  interpolate_linear(tr.times, tr.values, t)

"""Slerp a quaternion track between adjacent keyframes (clamped at the ends)."""
function _track_value(tr::QuaternionKeyframeTrack, t)
    times = tr.times; values = tr.values
    n = length(times)
    t <= times[1] && return values[1]
    t >= times[n] && return values[n]
    hi = searchsortedfirst(times, t)
    lo = hi - 1
    α = (t - times[lo]) / (times[hi] - times[lo])
    return quat_slerp(values[lo], values[hi], α)
end

"""
    sample_track(track, t)

Interpolated value of a keyframe `track` at absolute time `t`, using the track's
own interpolation mode (linear, Catmull-Rom cubic, or quaternion slerp). Returns a
`Vec3` for property tracks and a `Quaternion` for a `QuaternionKeyframeTrack`.
"""
sample_track(track::AbstractKeyframeTrack, t) = _track_value(track, t)

# Quaternion → Euler (intrinsic XYZ order, matching the default `Euler`).
# Mirrors three.js `Euler.setFromQuaternion` for order XYZ.
function _quat_to_euler_xyz(q::Quaternion)
    qn = quat_normalize(q)
    x, y, z, w = qn.x, qn.y, qn.z, qn.w
    m13 = 2*(x*z + w*y)
    m13c = clamp(m13, -one(m13), one(m13))
    ey = asin(m13c)
    if abs(m13) < 0.9999999
        m23 = 2*(y*z - w*x)
        m33 = 1 - 2*(x*x + y*y)
        m12 = 2*(x*y - w*z)
        m11 = 1 - 2*(y*y + z*z)
        ex = atan(-m23, m33)
        ez = atan(-m12, m11)
    else                                   # gimbal lock
        m21 = 2*(x*y + w*z)
        m22 = 1 - 2*(x*x + z*z)
        ex = atan(m21, m22)
        ez = zero(ey)
    end
    return Euler(ex, ey, ez, :XYZ)
end

struct AnimationClip
    name::String
    duration::Float64
    tracks::Vector{AbstractKeyframeTrack}
end
function AnimationClip(name::String, tracks::AbstractVector{<:AbstractKeyframeTrack})
    duration = isempty(tracks) ? 0.0 :
        maximum(isempty(t.times) ? 0.0 : maximum(t.times) for t in tracks)
    AnimationClip(name, duration, collect(AbstractKeyframeTrack, tracks))
end

mutable struct AnimationMixer
    clip::AnimationClip
    time::Float64
end
AnimationMixer(clip::AnimationClip) = AnimationMixer(clip, 0.0)

# Write an interpolated value to a track target. Quaternion samples targeting the
# `:rotation` field are converted to an `Euler` so the typed field accepts them.
_write_track_value!(target, property::Symbol, v) = setproperty!(target, property, v)
function _write_track_value!(target, property::Symbol, v::Quaternion)
    if property === :rotation
        setproperty!(target, property, _quat_to_euler_xyz(v))
    else
        setproperty!(target, property, v)
    end
    return target
end

"""Sample the clip at absolute time `t` and write each track's value to its target."""
function mixer_set_time!(mixer::AnimationMixer, t)
    mixer.time = t
    for tr in mixer.clip.tracks
        v = _track_value(tr, t)
        _write_track_value!(tr.target, tr.property, v)
    end
    return mixer
end

mixer_update!(mixer::AnimationMixer, dt) = mixer_set_time!(mixer, mixer.time + dt)

# ========================== Helpers ==========================

_line_geo(positions::Vector{Float64}) =
    BufferGeometry(positions, Float64[], Float64[], collect(1:length(positions)÷3),
                   length(positions) ÷ 3, 0)

"""Three line segments along +x (red), +y (green), +z (blue)."""
function AxesHelper(size=1.0)
    pos = Float64[0,0,0, size,0,0,  0,0,0, 0,size,0,  0,0,0, 0,0,size]
    geo = _line_geo(pos)
    set_attribute!(geo, :color, Float64[1,0,0, 1,0,0,  0,1,0, 0,1,0,  0,0,1, 0,0,1], 3)
    LineSegments(geo, LineBasicMaterial(color=Color3(1.0,1.0,1.0)); name="AxesHelper")
end

"""A `divisions`×`divisions` grid of size `size` on the xz-plane."""
function GridHelper(size=10.0, divisions=10; color=Color3(0.5,0.5,0.5))
    pos = Float64[]
    step = size / divisions; half = size / 2
    for i in 0:divisions
        c = -half + i*step
        append!(pos, [c,0,-half, c,0,half])     # parallel to z
        append!(pos, [-half,0,c, half,0,c])     # parallel to x
    end
    LineSegments(_line_geo(pos), LineBasicMaterial(color=color); name="GridHelper")
end

# 12 edges of an axis-aligned box given as min/max corners.
function _box_edges(mn::Vec3, mx::Vec3)
    c = [Vec3(mn.x,mn.y,mn.z), Vec3(mx.x,mn.y,mn.z), Vec3(mx.x,mx.y,mn.z), Vec3(mn.x,mx.y,mn.z),
         Vec3(mn.x,mn.y,mx.z), Vec3(mx.x,mn.y,mx.z), Vec3(mx.x,mx.y,mx.z), Vec3(mn.x,mx.y,mx.z)]
    edges = [(1,2),(2,3),(3,4),(4,1), (5,6),(6,7),(7,8),(8,5), (1,5),(2,6),(3,7),(4,8)]
    pos = Float64[]
    for (a,b) in edges
        append!(pos, [c[a].x,c[a].y,c[a].z, c[b].x,c[b].y,c[b].z])
    end
    return pos
end

"""Wireframe of a mesh's (local) bounding box."""
function BoxHelper(obj; color=Color3(1.0,1.0,0.0))
    box = compute_bounding_box(obj.geometry)
    LineSegments(_line_geo(_box_edges(box.min, box.max)), LineBasicMaterial(color=color); name="BoxHelper")
end

"""Frustum wireframe (12 edges) of a camera from its inverse view-projection."""
function CameraHelper(camera::AbstractCamera; color=Color3(1.0,1.0,1.0))
    inv = mat4_inverse(projection_matrix(camera) * view_matrix(camera))
    corner(x,y,z) = mat4_transform_point(inv, Vec3(x,y,z))
    c = [corner(-1,-1,-1), corner(1,-1,-1), corner(1,1,-1), corner(-1,1,-1),
         corner(-1,-1, 1), corner(1,-1, 1), corner(1,1, 1), corner(-1,1, 1)]
    edges = [(1,2),(2,3),(3,4),(4,1), (5,6),(6,7),(7,8),(8,5), (1,5),(2,6),(3,7),(4,8)]
    pos = Float64[]
    for (a,b) in edges
        append!(pos, [c[a].x,c[a].y,c[a].z, c[b].x,c[b].y,c[b].z])
    end
    LineSegments(_line_geo(pos), LineBasicMaterial(color=color); name="CameraHelper")
end

"""A single segment from a directional light's position toward its target."""
function DirectionalLightHelper(light::DirectionalLight; color=Color3(1.0,1.0,0.0))
    p = light.position; t = light.target
    LineSegments(_line_geo(Float64[p.x,p.y,p.z, t.x,t.y,t.z]),
                 LineBasicMaterial(color=color); name="DirectionalLightHelper")
end

"""Cross-hair lines marking a point light's position."""
function PointLightHelper(light::PointLight, size=0.5; color=Color3(1.0,1.0,0.0))
    p = light.position
    pos = Float64[p.x-size,p.y,p.z, p.x+size,p.y,p.z,
                  p.x,p.y-size,p.z, p.x,p.y+size,p.z,
                  p.x,p.y,p.z-size, p.x,p.y,p.z+size]
    LineSegments(_line_geo(pos), LineBasicMaterial(color=color); name="PointLightHelper")
end

# Append the two endpoints of a segment a→b to a flat position buffer.
@inline function _push_seg!(pos::Vector{Float64}, a::Vec3, b::Vec3)
    push!(pos, a.x, a.y, a.z, b.x, b.y, b.z)
    return pos
end

# Orthonormal basis (u, v) spanning the plane perpendicular to unit vector `w`.
function _perp_basis(w::Vec3)
    ref = abs(w.y) < 0.99 ? Vec3(0.0, 1.0, 0.0) : Vec3(1.0, 0.0, 0.0)
    u = normalize(cross(ref, w))
    v = cross(w, u)
    return u, v
end

"""
Cone outline of a spot light: the apex sits at the light position and the base
circle marks the cone of half-angle `light.angle` at the distance to the target
(three.js `SpotLightHelper`). Four apex-to-rim spokes plus the base ring.
"""
function SpotLightHelper(light::SpotLight; color=light.color, segments::Int=16)
    apex = light.position
    axis = light.target - apex
    len = norm(axis)
    len = len > 0 ? len : 1.0
    dir = len > 0 ? normalize(axis) : Vec3(0.0, -1.0, 0.0)
    base = apex + dir * len
    r = len * tan(light.angle)              # cone base radius at the target plane
    u, v = _perp_basis(dir)
    pos = Float64[]
    # Base ring.
    prev = base + u * r
    for k in 1:segments
        φ = 2π * k / segments
        cur = base + (u * (r * cos(φ)) + v * (r * sin(φ)))
        _push_seg!(pos, prev, cur)
        prev = cur
    end
    # Four spokes from apex to the rim.
    for φ in (0.0, π/2, π, 3π/2)
        rim = base + (u * (r * cos(φ)) + v * (r * sin(φ)))
        _push_seg!(pos, apex, rim)
    end
    LineSegments(_line_geo(pos), LineBasicMaterial(color=color); name="SpotLightHelper")
end

"""
Small octahedron wireframe at a hemisphere light's position, sized by `size`
(three.js `HemisphereLightHelper` uses a sphere/octahedron icon). The upper apex
takes the sky colour, the lower apex the ground colour, stored as a per-vertex
`:color` attribute.
"""
function HemisphereLightHelper(light::HemisphereLight, size=1.0; color=light.color)
    p = light.position; s = size
    top = Vec3(p.x, p.y+s, p.z); bot = Vec3(p.x, p.y-s, p.z)
    px = Vec3(p.x+s, p.y, p.z); nx = Vec3(p.x-s, p.y, p.z)
    pz = Vec3(p.x, p.y, p.z+s); nz = Vec3(p.x, p.y, p.z-s)
    eq = (px, pz, nx, nz)
    pos = Float64[]
    # Equator ring.
    for k in 1:4
        _push_seg!(pos, eq[k], eq[mod1(k+1, 4)])
    end
    # Spokes to the two apices.
    for e in eq
        _push_seg!(pos, top, e)
        _push_seg!(pos, bot, e)
    end
    geo = _line_geo(pos)
    sky = light.color; grd = light.ground_color
    cols = Float64[]
    nseg = length(pos) ÷ 6
    # First 4 segments are the (mid-latitude) equator: blend; the remaining
    # spokes inherit the apex colour (sky for top spokes, ground for bottom).
    for seg in 1:nseg
        if seg <= 4
            for _ in 1:2; push!(cols, (sky.r+grd.r)/2, (sky.g+grd.g)/2, (sky.b+grd.b)/2); end
        elseif (seg - 5) % 2 == 0          # top-apex spoke
            push!(cols, sky.r, sky.g, sky.b, sky.r, sky.g, sky.b)
        else                                # bottom-apex spoke
            push!(cols, grd.r, grd.g, grd.b, grd.r, grd.g, grd.b)
        end
    end
    set_attribute!(geo, :color, cols, 3)
    LineSegments(geo, LineBasicMaterial(color=color); name="HemisphereLightHelper")
end

# World-space translation (column 4) of a 4×4 transform.
@inline _mat4_translation_vec(m::Mat4) =
    Vec3(mat4_get(m, 1, 4), mat4_get(m, 2, 4), mat4_get(m, 3, 4))

"""
Line segments connecting each bone to its parent bone in world space
(three.js `SkeletonHelper`). Bones whose parent is not itself a bone in the
skeleton are skipped, matching three.js which only links bone-to-bone.
"""
function SkeletonHelper(skeleton::Skeleton; color=Color3(0.0, 0.0, 1.0))
    boneset = Set(objectid(b) for b in skeleton.bones)
    pos = Float64[]
    for bone in skeleton.bones
        par = get_parent(bone)
        (par isa Bone && objectid(par) in boneset) || continue
        cw = _mat4_translation_vec(compute_world_matrix(bone))
        pw = _mat4_translation_vec(compute_world_matrix(par))
        _push_seg!(pos, pw, cw)
    end
    LineSegments(_line_geo(pos), LineBasicMaterial(color=color); name="SkeletonHelper")
end

"""
Square outline lying in `plane` with side length `size`, plus a segment along the
plane normal from the plane's nearest point to the origin (three.js
`PlaneHelper`). The plane is `n·x + d = 0`; its representative point is `-d·n`.
"""
function PlaneHelper(plane::Plane, size=1.0; color=Color3(1.0, 1.0, 0.0))
    n = normalize(plane.normal)
    center = n * (-plane.constant)          # closest point to the origin on the plane
    u, v = _perp_basis(n)
    h = size / 2
    c1 = center + (u *  h + v *  h)
    c2 = center + (u * -h + v *  h)
    c3 = center + (u * -h + v * -h)
    c4 = center + (u *  h + v * -h)
    pos = Float64[]
    _push_seg!(pos, c1, c2); _push_seg!(pos, c2, c3)
    _push_seg!(pos, c3, c4); _push_seg!(pos, c4, c1)
    _push_seg!(pos, center, center + n * h) # normal indicator
    LineSegments(_line_geo(pos), LineBasicMaterial(color=color); name="PlaneHelper")
end

"""
Polar grid on the xz-plane: `rings` concentric circles out to `radius` and
`sectors` evenly spaced radial spokes (three.js `PolarGridHelper`). Circles are
approximated by `circle_segments` chords each.
"""
function PolarGridHelper(radius=10.0, sectors::Int=16, rings::Int=8;
                         circle_segments::Int=64, color=Color3(0.5, 0.5, 0.5))
    pos = Float64[]
    # Radial spokes.
    for k in 0:sectors-1
        φ = 2π * k / sectors
        x = radius * cos(φ); z = radius * sin(φ)
        _push_seg!(pos, Vec3(0.0, 0.0, 0.0), Vec3(x, 0.0, z))
    end
    # Concentric rings.
    for ri in 1:rings
        r = radius * ri / rings
        prev = Vec3(r, 0.0, 0.0)
        for k in 1:circle_segments
            φ = 2π * k / circle_segments
            cur = Vec3(r * cos(φ), 0.0, r * sin(φ))
            _push_seg!(pos, prev, cur)
            prev = cur
        end
    end
    LineSegments(_line_geo(pos), LineBasicMaterial(color=color); name="PolarGridHelper")
end
