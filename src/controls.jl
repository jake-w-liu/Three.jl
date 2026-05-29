# --------------------------------------------------------------------------
# Controls, animation, and helpers (three.js examples/ counterparts), adapted
# for headless/programmatic use: camera manipulators, a Clock, keyframe
# animation, and line-based scene helpers.
# --------------------------------------------------------------------------

# ========================== Camera controls ==========================

mutable struct OrbitControls
    camera::PerspectiveCamera
    target::Vec3{Float64}
end
OrbitControls(cam::PerspectiveCamera) = OrbitControls(cam, cam.target)

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

"""Rotate the camera around the target by angular deltas (radians)."""
function orbit_rotate!(oc::OrbitControls, d_azimuth, d_polar)
    s = _orbit_spherical(oc)
    _orbit_apply!(oc, Spherical(s.radius, clamp(s.phi + d_polar, 1e-4, π - 1e-4), s.theta + d_azimuth))
end

"""Scale the orbit radius (dolly). `factor < 1` zooms in."""
function orbit_zoom!(oc::OrbitControls, factor)
    s = _orbit_spherical(oc)
    _orbit_apply!(oc, Spherical(s.radius * factor, s.phi, s.theta))
end

"""Pan the target (and camera) in the camera's right/up plane."""
function orbit_pan!(oc::OrbitControls, dx, dy)
    fwd = normalize(oc.target - oc.camera.position)
    right = normalize(cross(fwd, oc.camera.up))
    up = cross(right, fwd)
    shift = right * dx + up * dy
    oc.target = oc.target + shift
    oc.camera.position = oc.camera.position + shift
    oc.camera.target = oc.target
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

struct KeyframeTrack
    target::AbstractObject3D
    property::Symbol                 # e.g. :position, :scale
    times::Vector{Float64}
    values::Vector{Vec3{Float64}}
end

struct AnimationClip
    name::String
    duration::Float64
    tracks::Vector{KeyframeTrack}
end
AnimationClip(name::String, tracks::Vector{KeyframeTrack}) =
    AnimationClip(name, maximum(maximum(t.times) for t in tracks), tracks)

mutable struct AnimationMixer
    clip::AnimationClip
    time::Float64
end
AnimationMixer(clip::AnimationClip) = AnimationMixer(clip, 0.0)

"""Sample the clip at absolute time `t` and write each track's value to its target."""
function mixer_set_time!(mixer::AnimationMixer, t)
    mixer.time = t
    for tr in mixer.clip.tracks
        v = interpolate_linear(tr.times, tr.values, t)
        setproperty!(tr.target, tr.property, v)
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
