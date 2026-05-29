# --------------------------------------------------------------------------
# Material types mirroring three.js material hierarchy.
# --------------------------------------------------------------------------

abstract type AbstractMaterial end

# ========================== MeshBasicMaterial ==========================
# Unlit, flat color

struct MeshBasicMaterial <: AbstractMaterial
    color::Color3{Float64}
    opacity::Float64
    transparent::Bool
    wireframe::Bool
    side::Symbol  # :front, :back, :double
    map::Any      # optional albedo Texture
    vertex_colors::Bool   # modulate by geometry :color attribute when true
end

function MeshBasicMaterial(; color=Color3(1.0, 1.0, 1.0), opacity=1.0,
                            transparent=false, wireframe=false, side=:front, map=nothing,
                            vertex_colors=false)
    MeshBasicMaterial(color, opacity, transparent, wireframe, side, map, vertex_colors)
end

# ========================== MeshLambertMaterial ==========================
# Diffuse-only (Lambertian)

struct MeshLambertMaterial <: AbstractMaterial
    color::Color3{Float64}
    emissive::Color3{Float64}
    opacity::Float64
    transparent::Bool
    side::Symbol
    map::Any
    ao_map::Any
    emissive_map::Any
    vertex_colors::Bool   # modulate by geometry :color attribute when true
    light_map::Any        # baked indirect-lighting texture (multiplied in, like aoMap)
end

function MeshLambertMaterial(; color=Color3(1.0, 1.0, 1.0),
                              emissive=Color3(0.0, 0.0, 0.0),
                              opacity=1.0, transparent=false, side=:front,
                              map=nothing, ao_map=nothing, emissive_map=nothing,
                              vertex_colors=false, light_map=nothing)
    MeshLambertMaterial(color, emissive, opacity, transparent, side, map, ao_map, emissive_map,
                        vertex_colors, light_map)
end

# ========================== MeshPhongMaterial ==========================
# Blinn-Phong shading

struct MeshPhongMaterial <: AbstractMaterial
    color::Color3{Float64}
    specular::Color3{Float64}
    emissive::Color3{Float64}
    shininess::Float64
    opacity::Float64
    transparent::Bool
    side::Symbol
    map::Any
    light_map::Any        # baked indirect-lighting texture (multiplied in, like aoMap)
end

function MeshPhongMaterial(; color=Color3(1.0, 1.0, 1.0),
                            specular=Color3(0.066, 0.066, 0.066),
                            emissive=Color3(0.0, 0.0, 0.0),
                            shininess=30.0, opacity=1.0,
                            transparent=false, side=:front, map=nothing, light_map=nothing)
    MeshPhongMaterial(color, specular, emissive, shininess, opacity, transparent, side, map,
                      light_map)
end

# ========================== MeshStandardMaterial ==========================
# PBR (metallic-roughness workflow)

struct MeshStandardMaterial <: AbstractMaterial
    color::Color3{Float64}
    emissive::Color3{Float64}
    metalness::Float64
    roughness::Float64
    opacity::Float64
    transparent::Bool
    side::Symbol
    map::Any
    normal_map::Any
    roughness_map::Any
    ao_map::Any
    emissive_map::Any
    vertex_colors::Bool   # modulate by geometry :color attribute when true
    envmap::Any           # optional CubeTexture for reflection (IBL specular)
    light_map::Any        # baked indirect-lighting texture (multiplied in, like aoMap)
end

function MeshStandardMaterial(; color=Color3(1.0, 1.0, 1.0),
                               emissive=Color3(0.0, 0.0, 0.0),
                               metalness=0.0, roughness=1.0,
                               opacity=1.0, transparent=false, side=:front,
                               map=nothing, normal_map=nothing, roughness_map=nothing,
                               ao_map=nothing, emissive_map=nothing,
                               vertex_colors=false, envmap=nothing, light_map=nothing)
    MeshStandardMaterial(color, emissive, metalness, roughness, opacity, transparent, side,
                         map, normal_map, roughness_map, ao_map, emissive_map,
                         vertex_colors, envmap, light_map)
end

# ========================== MeshNormalMaterial ==========================
# Maps normals to RGB

struct MeshNormalMaterial <: AbstractMaterial
    opacity::Float64
    transparent::Bool
    side::Symbol
end

function MeshNormalMaterial(; opacity=1.0, transparent=false, side=:front)
    MeshNormalMaterial(opacity, transparent, side)
end

# ========================== LineBasicMaterial ==========================

struct LineBasicMaterial <: AbstractMaterial
    color::Color3{Float64}
    linewidth::Float64
    opacity::Float64
end

function LineBasicMaterial(; color=Color3(1.0, 1.0, 1.0), linewidth=1.0, opacity=1.0)
    LineBasicMaterial(color, linewidth, opacity)
end

# ========================== PointsMaterial ==========================

struct PointsMaterial <: AbstractMaterial
    color::Color3{Float64}
    size::Float64
    opacity::Float64
end

function PointsMaterial(; color=Color3(1.0, 1.0, 1.0), size=1.0, opacity=1.0)
    PointsMaterial(color, size, opacity)
end

# ========================== MeshPhysicalMaterial ==========================
# PBR extended with a dielectric clearcoat lobe, transmission and IOR.

struct MeshPhysicalMaterial <: AbstractMaterial
    color::Color3{Float64}
    emissive::Color3{Float64}
    metalness::Float64
    roughness::Float64
    clearcoat::Float64
    clearcoat_roughness::Float64
    transmission::Float64
    ior::Float64
    opacity::Float64
    transparent::Bool
    side::Symbol
    envmap::Any           # optional CubeTexture for reflection (IBL specular)
    # --- three.js MeshPhysicalMaterial extensions (added last, keyword defaults) ---
    sheen::Float64                 # retroreflective sheen strength (0 = off)
    sheen_color::Color3{Float64}   # tint of the sheen lobe
    sheen_roughness::Float64       # Charlie-distribution roughness (1 = broad)
    iridescence::Float64           # thin-film interference blend (0 = off)
    iridescence_ior::Float64       # refractive index of the thin film
    iridescence_thickness::Float64 # film thickness in nanometres
    light_map::Any                 # baked indirect-lighting texture (multiplied in)
end

function MeshPhysicalMaterial(; color=Color3(1.0,1.0,1.0), emissive=Color3(0.0,0.0,0.0),
                               metalness=0.0, roughness=1.0, clearcoat=0.0,
                               clearcoat_roughness=0.0, transmission=0.0, ior=1.5,
                               opacity=1.0, transparent=false, side=:front, envmap=nothing,
                               sheen=0.0, sheen_color=Color3(1.0,1.0,1.0), sheen_roughness=1.0,
                               iridescence=0.0, iridescence_ior=1.3, iridescence_thickness=400.0,
                               light_map=nothing)
    MeshPhysicalMaterial(color, emissive, metalness, roughness, clearcoat,
                         clearcoat_roughness, transmission, ior, opacity, transparent, side,
                         envmap, sheen, sheen_color, sheen_roughness,
                         iridescence, iridescence_ior, iridescence_thickness, light_map)
end

# ========================== MeshToonMaterial ==========================
# Quantized (cel-shaded) diffuse.

struct MeshToonMaterial <: AbstractMaterial
    color::Color3{Float64}
    emissive::Color3{Float64}
    gradient_steps::Int
    opacity::Float64
    transparent::Bool
    side::Symbol
end

function MeshToonMaterial(; color=Color3(1.0,1.0,1.0), emissive=Color3(0.0,0.0,0.0),
                           gradient_steps=3, opacity=1.0, transparent=false, side=:front)
    MeshToonMaterial(color, emissive, gradient_steps, opacity, transparent, side)
end

# ========================== MeshMatcapMaterial ==========================
# Material-capture shading: appearance baked into a sphere image indexed by the
# view-space normal. `matcap` is an optional texture; without one a procedural
# view-facing falloff is used.

struct MeshMatcapMaterial <: AbstractMaterial
    color::Color3{Float64}
    matcap::Any
    opacity::Float64
    transparent::Bool
    side::Symbol
end

function MeshMatcapMaterial(; color=Color3(1.0,1.0,1.0), matcap=nothing,
                             opacity=1.0, transparent=false, side=:front)
    MeshMatcapMaterial(color, matcap, opacity, transparent, side)
end

# ========================== MeshDepthMaterial ==========================
# Renders normalized camera-space depth as grayscale (near → bright).

struct MeshDepthMaterial <: AbstractMaterial
    near::Float64
    far::Float64
    opacity::Float64
    transparent::Bool
    side::Symbol
end

function MeshDepthMaterial(; near=0.1, far=100.0, opacity=1.0, transparent=false, side=:front)
    MeshDepthMaterial(near, far, opacity, transparent, side)
end

# ========================== ShaderMaterial ==========================
# Placeholder for custom GLSL

struct ShaderMaterial <: AbstractMaterial
    vertex_shader::String
    fragment_shader::String
    uniforms::Dict{String, Any}
    program::Any   # optional CPU fragment program: (normal, view_dir, position, uniforms) -> Color3
    side::Symbol
end

function ShaderMaterial(; vertex_shader="", fragment_shader="", uniforms=Dict{String,Any}(),
                         program=nothing, side=:front)
    ShaderMaterial(vertex_shader, fragment_shader, uniforms, program, side)
end
