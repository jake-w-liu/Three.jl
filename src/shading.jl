# --------------------------------------------------------------------------
# Shading models: Phong, Lambert, PBR (simplified Cook-Torrance).
# Pure Julia, AD-compatible — no branching on types in hot path.
# --------------------------------------------------------------------------

"""
Compute Lambertian diffuse contribution for one light.
Returns Color3 — the diffuse color modulated by light.
"""
function shade_lambert(normal::Vec3, light_dir::Vec3, light_color::Color3,
                       light_intensity, surface_color::Color3)
    ndotl = max(dot(normal, light_dir), zero(light_intensity))
    surface_color * light_color * (ndotl * light_intensity)
end

"""
Blinn-Phong shading: diffuse + specular.
`view_dir` points from surface toward camera.
"""
function shade_phong(normal::Vec3, light_dir::Vec3, view_dir::Vec3,
                     light_color::Color3, light_intensity,
                     diffuse_color::Color3, specular_color::Color3,
                     shininess)
    # Diffuse
    ndotl = max(dot(normal, light_dir), zero(light_intensity))
    diffuse = diffuse_color * light_color * (ndotl * light_intensity)

    # Specular (Blinn-Phong half-vector)
    half_vec = normalize(light_dir + view_dir)
    ndoth = max(dot(normal, half_vec), zero(shininess))
    spec = specular_color * light_color * (ndoth^shininess * light_intensity)

    diffuse + spec
end

"""
Simplified PBR shading (Cook-Torrance with GGX distribution).
"""
function shade_pbr(normal::Vec3, light_dir::Vec3, view_dir::Vec3,
                   light_color::Color3, light_intensity,
                   albedo::Color3, metalness, roughness)
    α = roughness * roughness
    α2 = α * α

    ndotl = max(dot(normal, light_dir), zero(α))
    ndotv = max(dot(normal, view_dir), zero(α))

    h = normalize(light_dir + view_dir)
    ndoth = max(dot(normal, h), zero(α))
    vdoth = max(dot(view_dir, h), zero(α))

    # GGX/Trowbridge-Reitz normal distribution
    denom_d = ndoth * ndoth * (α2 - 1) + 1
    D = α2 / (π * denom_d * denom_d + 1e-7)

    # Schlick-GGX geometry (Smith model)
    k = (roughness + 1)^2 / 8
    G1_l = ndotl / (ndotl * (1 - k) + k + 1e-7)
    G1_v = ndotv / (ndotv * (1 - k) + k + 1e-7)
    G = G1_l * G1_v

    # Schlick Fresnel
    f0_val = lerp_scalar(0.04, 1.0, metalness)
    F0 = Color3(
        lerp_scalar(0.04, albedo.r, metalness),
        lerp_scalar(0.04, albedo.g, metalness),
        lerp_scalar(0.04, albedo.b, metalness)
    )
    F = Color3(
        F0.r + (1 - F0.r) * (1 - vdoth)^5,
        F0.g + (1 - F0.g) * (1 - vdoth)^5,
        F0.b + (1 - F0.b) * (1 - vdoth)^5
    )

    # Specular BRDF
    spec_num = D * G
    spec_den = 4 * ndotv * ndotl + 1e-7
    spec = Color3(
        F.r * spec_num / spec_den,
        F.g * spec_num / spec_den,
        F.b * spec_num / spec_den
    )

    # Diffuse BRDF (energy-conserving Lambert)
    kD = Color3(
        (1 - F.r) * (1 - metalness),
        (1 - F.g) * (1 - metalness),
        (1 - F.b) * (1 - metalness)
    )
    diffuse = Color3(
        kD.r * albedo.r / π,
        kD.g * albedo.g / π,
        kD.b * albedo.b / π
    )

    # Final
    result = Color3(
        (diffuse.r + spec.r) * light_color.r * light_intensity * ndotl,
        (diffuse.g + spec.g) * light_color.g * light_intensity * ndotl,
        (diffuse.b + spec.b) * light_color.b * light_intensity * ndotl
    )
    return result
end

lerp_scalar(a, b, t) = a + (b - a) * t

"""
Compute per-face shading for a mesh given scene lights and camera.
Returns Vector{Color3} with one color per face.
"""
# Flat-shading face normal: average of the authored per-vertex normals
# transformed to world space, so orientation follows the geometry's normals
# rather than its (possibly inconsistent) triangle winding. Falls back to the
# geometric winding normal when authored normals are absent or degenerate
# (e.g. a freshly loaded mesh before `compute_vertex_normals!`).
@inline function _flat_face_normal(geo::BufferGeometry, i1, i2, i3,
                                   v1::Vec3, v2::Vec3, v3::Vec3,
                                   normal_mat::Mat4, has_normals::Bool)
    geo_n = cross(v2 - v1, v3 - v1)
    if has_normals
        n = mat4_transform_direction(normal_mat, get_normal(geo, i1)) +
            mat4_transform_direction(normal_mat, get_normal(geo, i2)) +
            mat4_transform_direction(normal_mat, get_normal(geo, i3))
        nl = norm(n)
        nl > 1e-12 && return n / nl
    end
    gl = norm(geo_n)
    return gl > 1e-12 ? geo_n / gl : Vec3(0.0, 0.0, 1.0)
end

# Optional material texture field (nothing if the material has no such map).
@inline _material_field(m, f::Symbol) = hasfield(typeof(m), f) ? getfield(m, f) : nothing

# Centroid UV of a face (average of its three vertex UVs).
@inline function _face_centroid_uv(geo::BufferGeometry, i1, i2, i3)
    b1 = (i1-1)*2; b2 = (i2-1)*2; b3 = (i3-1)*2
    ((geo.uvs[b1+1] + geo.uvs[b2+1] + geo.uvs[b3+1]) / 3,
     (geo.uvs[b1+2] + geo.uvs[b2+2] + geo.uvs[b3+2]) / 3)
end

@inline _vertex_uv(geo::BufferGeometry, i) = (geo.uvs[(i-1)*2+1], geo.uvs[(i-1)*2+2])

# Perturb a face normal by a tangent-space normal map. The tangent frame is
# derived from the triangle's world positions and UV gradients (standard
# tangent-from-UV), then the sampled normal `2·texel-1` is rotated into world.
function _apply_normal_map(face_n::Vec3, nmap, u, v, p1::Vec3, p2::Vec3, p3::Vec3,
                           uv1, uv2, uv3)
    e1 = p2 - p1; e2 = p3 - p1
    du1 = uv2[1]-uv1[1]; dv1 = uv2[2]-uv1[2]
    du2 = uv3[1]-uv1[1]; dv2 = uv3[2]-uv1[2]
    det = du1*dv2 - du2*dv1
    abs(det) < 1e-12 && return face_n
    r = 1.0 / det
    T = (e1*dv2 - e2*dv1) * r
    tl = norm(T); tl < 1e-12 && return face_n
    T = T / tl
    T = T - face_n * dot(face_n, T)                  # Gram-Schmidt orthonormalize
    tl2 = norm(T); tl2 < 1e-12 && return face_n
    T = T / tl2
    B = cross(face_n, T)
    s = sample_texture(nmap, u, v)
    tn = Vec3(s.r*2 - 1, s.g*2 - 1, s.b*2 - 1)
    return normalize(T*tn.x + B*tn.y + face_n*tn.z)
end

# Per-face roughness override from a roughness map (glTF: G channel = roughness).
function _apply_roughness_map(m::MeshStandardMaterial, rmap, u, v)
    rg = sample_texture(rmap, u, v).g
    MeshStandardMaterial(color=m.color, emissive=m.emissive, metalness=m.metalness,
                         roughness=rg, opacity=m.opacity, transparent=m.transparent,
                         side=m.side, map=m.map, normal_map=m.normal_map,
                         roughness_map=m.roughness_map, ao_map=m.ao_map, emissive_map=m.emissive_map)
end
function _apply_roughness_map(m::MeshPhysicalMaterial, rmap, u, v)
    rg = sample_texture(rmap, u, v).g
    MeshPhysicalMaterial(color=m.color, emissive=m.emissive, metalness=m.metalness,
                         roughness=rg, clearcoat=m.clearcoat, clearcoat_roughness=m.clearcoat_roughness,
                         transmission=m.transmission, ior=m.ior, opacity=m.opacity,
                         transparent=m.transparent, side=m.side)
end
_apply_roughness_map(m::AbstractMaterial, rmap, u, v) = m   # other materials: ignore

shade_mesh_faces(geo::BufferGeometry, world_mat::Mat4, material::AbstractMaterial,
                 lights::Vector{<:AbstractLight}, cam_pos::Vec3; shadow_fn=nothing) =
    shade_mesh_faces!(Vector{Color3{Float64}}(undef, geo.n_faces),
                      geo, world_mat, material, lights, cam_pos; shadow_fn=shadow_fn)

# In-place variant: writes one colour per face into `colors` (resized to fit),
# so a caller can reuse the buffer across frames/meshes (bounded allocation).
function shade_mesh_faces!(colors::Vector{Color3{Float64}},
                           geo::BufferGeometry, world_mat::Mat4, material::AbstractMaterial,
                           lights::Vector{<:AbstractLight}, cam_pos::Vec3; shadow_fn=nothing)
    n_faces = geo.n_faces
    length(colors) == n_faces || resize!(colors, n_faces)

    # Normal matrix (transpose of inverse) for transforming normals to world space.
    normal_mat = mat4_transpose(mat4_inverse(world_mat))
    has_normals = length(geo.normals) >= geo.n_vertices * 3
    has_uvs = length(geo.uvs) >= geo.n_vertices * 2
    albedo_map = _material_field(material, :map)
    ao_map = _material_field(material, :ao_map)
    emissive_map = _material_field(material, :emissive_map)
    normal_map = _material_field(material, :normal_map)
    roughness_map = _material_field(material, :roughness_map)
    use_maps = has_uvs && (albedo_map !== nothing || ao_map !== nothing || emissive_map !== nothing ||
                           normal_map !== nothing || roughness_map !== nothing)

    for fi in 1:n_faces
        i1, i2, i3 = get_face(geo, fi)
        v1 = mat4_transform_point(world_mat, get_vertex(geo, i1))
        v2 = mat4_transform_point(world_mat, get_vertex(geo, i2))
        v3 = mat4_transform_point(world_mat, get_vertex(geo, i3))
        center = Vec3((v1.x+v2.x+v3.x)/3, (v1.y+v2.y+v3.y)/3, (v1.z+v2.z+v3.z)/3)

        if material isa MeshDepthMaterial
            d = norm(cam_pos - center)
            g = clamp((material.far - d) / (material.far - material.near), 0.0, 1.0)
            colors[fi] = Color3(g, g, g)                  # near → bright
            continue
        end

        face_n = _flat_face_normal(geo, i1, i2, i3, v1, v2, v3, normal_mat, has_normals)
        view_dir = normalize(cam_pos - center)
        eff_mat = material

        # normalMap perturbs the shading normal and roughnessMap overrides the
        # material roughness — both must apply BEFORE shading. Albedo/AO/emissive
        # modulate the result AFTER (they commute with diffuse lighting).
        if use_maps
            u, v = _face_centroid_uv(geo, i1, i2, i3)
            if normal_map !== nothing
                face_n = _apply_normal_map(face_n, normal_map, u, v, v1, v2, v3,
                                           _vertex_uv(geo, i1), _vertex_uv(geo, i2), _vertex_uv(geo, i3))
            end
            roughness_map !== nothing && (eff_mat = _apply_roughness_map(material, roughness_map, u, v))
        end

        color = shade_face(face_n, view_dir, center, eff_mat, lights; shadow_fn=shadow_fn)

        if use_maps
            u, v = _face_centroid_uv(geo, i1, i2, i3)
            albedo_map !== nothing && (color = color * sample_texture(albedo_map, u, v))
            if ao_map !== nothing
                ao = sample_texture(ao_map, u, v); color = Color3(color.r*ao.r, color.g*ao.g, color.b*ao.b)
            end
            emissive_map !== nothing && (color = color + sample_texture(emissive_map, u, v))
        end

        colors[fi] = clamp_color(color)
    end
    return colors
end

function shade_face(normal::Vec3, view_dir::Vec3, position::Vec3,
                    material::MeshBasicMaterial, lights; shadow_fn=nothing)
    material.color
end

# --------------------------------------------------------------------------
# Ambient / hemisphere lights are view-independent fill, not directional. They
# add uniform (hemisphere: normal-weighted) irradiance rather than an N·L term.
# `_is_fill_light` selects them; `_fill_color` returns the effective irradiance.
# --------------------------------------------------------------------------
@inline _is_fill_light(l::AbstractLight) =
    (l isa AmbientLight) || (l isa HemisphereLight) || (l isa LightProbe)

_fill_color(normal::Vec3, light::AmbientLight) = light.color * light.intensity
function _fill_color(normal::Vec3, light::HemisphereLight)
    w = clamp(normal.y * 0.5 + 0.5, zero(normal.y), one(normal.y))
    blended = Color3(light.color.r * w + light.ground_color.r * (1 - w),
                     light.color.g * w + light.ground_color.g * (1 - w),
                     light.color.b * w + light.ground_color.b * (1 - w))
    return blended * light.intensity
end
# Order-1 SH irradiance, clamped to non-negative per channel.
function _fill_color(normal::Vec3, light::LightProbe)
    c = light.coeffs
    r = c[1].r + c[2].r*normal.x + c[3].r*normal.y + c[4].r*normal.z
    g = c[1].g + c[2].g*normal.x + c[3].g*normal.y + c[4].g*normal.z
    b = c[1].b + c[2].b*normal.x + c[3].b*normal.y + c[4].b*normal.z
    Color3(max(r,0.0), max(g,0.0), max(b,0.0)) * light.intensity
end

# Environment (ambient/IBL) response of a metallic-roughness surface. Without
# this term metals (low diffuse, narrow specular) render black under directional
# light alone. Diffuse scales by (1-metalness); specular uses Schlick F0 (≈albedo
# for metals) attenuated by roughness.
function _pbr_ambient(normal::Vec3, albedo::Color3, metalness, roughness, fill::Color3)
    kd = one(metalness) - metalness
    F0r = lerp_scalar(0.04, albedo.r, metalness)
    F0g = lerp_scalar(0.04, albedo.g, metalness)
    F0b = lerp_scalar(0.04, albedo.b, metalness)
    spec_scale = one(roughness) - roughness * 0.7
    Color3((albedo.r * kd + F0r * spec_scale) * fill.r,
           (albedo.g * kd + F0g * spec_scale) * fill.g,
           (albedo.b * kd + F0b * spec_scale) * fill.b)
end

# Dielectric clearcoat highlight: Schlick-Fresnel-weighted Blinn-Phong lobe (F0 = 0.04).
@inline function _clearcoat_spec(normal::Vec3, light_dir::Vec3, view_dir::Vec3, cc_roughness)
    h = normalize(light_dir + view_dir)
    ndoth = max(dot(normal, h), zero(cc_roughness))
    vdoth = max(dot(view_dir, h), zero(cc_roughness))
    shininess = 2 / max((cc_roughness*cc_roughness)^2, 1e-4) + 2   # roughness → exponent
    fresnel = 0.04 + 0.96 * (1 - vdoth)^5
    return fresnel * ndoth^shininess
end

# Lit materials share one accumulation loop so direct lighting can be modulated
# by a per-light shadow visibility (`shadow_fn(light, position) ∈ [0,1]`). Each
# material supplies its fill response and its per-light direct response.
const LitMaterial = Union{MeshLambertMaterial, MeshPhongMaterial, MeshStandardMaterial,
                          MeshPhysicalMaterial, MeshToonMaterial}

_fill_response(m::MeshLambertMaterial,  n, fc) = m.color * fc
_fill_response(m::MeshPhongMaterial,    n, fc) = m.color * fc
_fill_response(m::MeshToonMaterial,     n, fc) = m.color * fc
_fill_response(m::MeshStandardMaterial, n, fc) = _pbr_ambient(n, m.color, m.metalness, m.roughness, fc)
_fill_response(m::MeshPhysicalMaterial, n, fc) = _pbr_ambient(n, m.color, m.metalness, m.roughness, fc)

_direct_response(m::MeshLambertMaterial, n, v, lc, li, ldir) =
    shade_lambert(n, ldir, lc, li, m.color)
_direct_response(m::MeshPhongMaterial, n, v, lc, li, ldir) =
    shade_phong(n, ldir, v, lc, li, m.color, m.specular, m.shininess)
_direct_response(m::MeshStandardMaterial, n, v, lc, li, ldir) =
    shade_pbr(n, ldir, v, lc, li, m.color, m.metalness, m.roughness)
function _direct_response(m::MeshPhysicalMaterial, n, v, lc, li, ldir)
    base = shade_pbr(n, ldir, v, lc, li, m.color, m.metalness, m.roughness)
    cc = m.clearcoat * _clearcoat_spec(n, ldir, v, m.clearcoat_roughness) * max(dot(n, ldir), 0.0)
    base + lc * (cc * li)
end
function _direct_response(m::MeshToonMaterial, n, v, lc, li, ldir)
    ndotl = max(dot(n, ldir), 0.0)
    banded = ceil(ndotl * m.gradient_steps) / m.gradient_steps
    m.color * lc * (banded * li)
end

function _shade_lit(m, normal::Vec3, view_dir::Vec3, position::Vec3, lights, shadow_fn)
    result = m.emissive
    for light in lights
        if _is_fill_light(light)
            result = result + _fill_response(m, normal, _fill_color(normal, light))
        else
            lc, li, ldir = light_contribution(light, position)
            vis = shadow_fn === nothing ? 1.0 : shadow_fn(light, position)
            vis <= 0.0 && continue
            result = result + _direct_response(m, normal, view_dir, lc, li, ldir) * vis
        end
    end
    return result
end

shade_face(normal::Vec3, view_dir::Vec3, position::Vec3, material::LitMaterial, lights;
           shadow_fn=nothing) = _shade_lit(material, normal, view_dir, position, lights, shadow_fn)

function shade_face(normal::Vec3, view_dir::Vec3, position::Vec3,
                    material::MeshNormalMaterial, lights; shadow_fn=nothing)
    Color3((normal.x + 1) / 2, (normal.y + 1) / 2, (normal.z + 1) / 2)
end

function shade_face(normal::Vec3, view_dir::Vec3, position::Vec3,
                    material::MeshMatcapMaterial, lights; shadow_fn=nothing)
    # Procedural matcap: brighter where the surface faces the viewer.
    f = max(dot(normal, view_dir), 0.0)
    material.color * (0.35 + 0.65 * f)
end

# MeshDepthMaterial is resolved in `shade_mesh_faces` (it needs the camera
# distance); this fallback keeps direct `shade_face` calls well-defined.
function shade_face(normal::Vec3, view_dir::Vec3, position::Vec3,
                    material::MeshDepthMaterial, lights; shadow_fn=nothing)
    Color3(0.5, 0.5, 0.5)
end

# ShaderMaterial: run the user-supplied CPU fragment program (the engine's
# analog of a GLSL fragment shader). Signature: (normal, view_dir, position,
# uniforms) -> Color3. Without a program, falls back to mid-gray.
function shade_face(normal::Vec3, view_dir::Vec3, position::Vec3,
                    material::ShaderMaterial, lights; shadow_fn=nothing)
    material.program === nothing && return Color3(0.5, 0.5, 0.5)
    return material.program(normal, view_dir, position, material.uniforms)
end

# Fallback (MeshBasicMaterial is defined earlier; this covers any other material).
function shade_face(normal::Vec3, view_dir::Vec3, position::Vec3,
                    material::AbstractMaterial, lights; shadow_fn=nothing)
    Color3(0.5, 0.5, 0.5)
end

# ========================== Light contribution ==========================

function light_contribution(light::AmbientLight, position::Vec3)
    (light.color, light.intensity, Vec3(0.0, 1.0, 0.0))
end

function light_contribution(light::DirectionalLight, position::Vec3)
    dir = normalize(light.position - light.target)
    (light.color, light.intensity, dir)
end

function light_contribution(light::PointLight, position::Vec3)
    diff = light.position - position
    dist = norm(diff)
    dir = diff / max(dist, 1e-10)
    attenuation = if light.distance > 0
        factor = max(1.0 - (dist / light.distance)^2, 0.0)
        factor / max(dist^light.decay, 1e-10)
    else
        1.0 / max(dist^light.decay, 1e-10)
    end
    (light.color, light.intensity * attenuation, dir)
end

function light_contribution(light::SpotLight, position::Vec3)
    diff = light.position - position
    dist = norm(diff)
    dir = diff / max(dist, 1e-10)

    # Spot cone
    target_dir = normalize(light.target - light.position)
    cos_angle = dot(-dir, target_dir)  # note: dir points toward light
    # Actually: dir points from surface to light, target_dir points from light to target
    cos_angle = dot(normalize(position - light.position), target_dir)
    cos_outer = cos(light.angle)
    cos_inner = cos(light.angle * (1 - light.penumbra))

    spot_effect = clamp((cos_angle - cos_outer) / max(cos_inner - cos_outer, 1e-10), 0.0, 1.0)

    attenuation = spot_effect / max(dist^light.decay, 1e-10)
    (light.color, light.intensity * attenuation, dir)
end

function light_contribution(light::HemisphereLight, position::Vec3)
    # Blend between sky and ground based on normal direction
    # We return the sky color; the shade function should handle hemisphere blending
    (light.color, light.intensity, Vec3(0.0, 1.0, 0.0))
end

function light_contribution(light::RectAreaLight, position::Vec3)
    diff = light.position - position
    dir = diff / max(norm(diff), 1e-10)
    (light.color, light.intensity, dir)
end

# Special handling for ambient in shade functions
function shade_face_with_ambient(normal, view_dir, position, material, lights)
    result = Color3(0.0, 0.0, 0.0)
    for light in lights
        if light isa AmbientLight
            mc = _material_color(material)
            result = result + mc * light.color * light.intensity
        elseif light isa HemisphereLight
            mc = _material_color(material)
            weight = dot(normal, Vec3(0.0, 1.0, 0.0)) * 0.5 + 0.5
            blended = Color3(
                light.color.r * weight + light.ground_color.r * (1 - weight),
                light.color.g * weight + light.ground_color.g * (1 - weight),
                light.color.b * weight + light.ground_color.b * (1 - weight)
            )
            result = result + mc * blended * light.intensity
        end
    end
    return result
end

_material_color(m::MeshBasicMaterial) = m.color
_material_color(m::MeshLambertMaterial) = m.color
_material_color(m::MeshPhongMaterial) = m.color
_material_color(m::MeshStandardMaterial) = m.color
_material_color(m::MeshNormalMaterial) = Color3(0.5, 0.5, 0.5)
_material_color(m::AbstractMaterial) = Color3(0.5, 0.5, 0.5)
