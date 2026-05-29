# --------------------------------------------------------------------------
# Textures: 2D Texture/DataTexture/CanvasTexture, CubeTexture, DepthTexture,
# UV sampling with wrap modes (repeat/clamp/mirror) and filtering
# (nearest/bilinear), mipmaps, and procedural checker/grid generators.
# Image data is stored row-major H×W×C with row 1 = top; UV (0,0) = bottom-left.
# --------------------------------------------------------------------------

mutable struct Texture
    data::Array{Float64, 3}            # H × W × C
    wrap_s::Symbol                     # :repeat | :clamp | :mirror  (u)
    wrap_t::Symbol                     # :repeat | :clamp | :mirror  (v)
    filter::Symbol                     # :nearest | :bilinear
    mipmaps::Vector{Array{Float64, 3}} # optional pyramid (level 1 = base/2)
end

function Texture(data::Array{Float64,3}; wrap_s=:repeat, wrap_t=:repeat, filter=:bilinear)
    Texture(data, wrap_s, wrap_t, filter, Array{Float64,3}[])
end
DataTexture(data::Array{Float64,3}; kwargs...) = Texture(data; kwargs...)
CanvasTexture(data::Array{Float64,3}; kwargs...) = Texture(data; kwargs...)

# Single-channel depth texture.
DepthTexture(depth::Matrix{Float64}; kwargs...) =
    Texture(reshape(depth, size(depth,1), size(depth,2), 1); kwargs...)

# Wrap a 0-based integer pixel coordinate into [0, n-1] per the mode.
@inline function _wrap_coord(i::Int, n::Int, mode::Symbol)
    if mode === :repeat
        return mod(i, n)
    elseif mode === :clamp
        return clamp(i, 0, n - 1)
    else # :mirror — triangle reflection with period 2n
        p = mod(i, 2n)
        return p < n ? p : (2n - 1 - p)
    end
end

@inline function _texel(tex::Texture, ix::Int, iy::Int)
    H, W, C = size(tex.data)
    x = _wrap_coord(ix, W, tex.wrap_s) + 1
    y = _wrap_coord(iy, H, tex.wrap_t) + 1
    if C == 1
        g = tex.data[y, x, 1]; return Color3(g, g, g)
    end
    Color3(tex.data[y, x, 1], tex.data[y, x, 2], tex.data[y, x, 3])
end

"""
    sample_texture(tex, u, v) -> Color3

Sample the texture at UV `(u,v)` ∈ [0,1]² (outside handled by the wrap modes),
using nearest or bilinear filtering. `v=0` is the bottom row.
"""
function sample_texture(tex::Texture, u, v)
    H, W, _ = size(tex.data)
    fx = u * W - 0.5
    fy = (1 - v) * H - 0.5                       # flip v so v=0 maps to the bottom row
    if tex.filter === :nearest
        return _texel(tex, round(Int, fx), round(Int, fy))
    end
    x0 = floor(Int, fx); y0 = floor(Int, fy)
    tx = fx - x0; ty = fy - y0
    c00 = _texel(tex, x0,   y0);   c10 = _texel(tex, x0+1, y0)
    c01 = _texel(tex, x0,   y0+1); c11 = _texel(tex, x0+1, y0+1)
    top = c00 * (1 - tx) + c10 * tx
    bot = c01 * (1 - tx) + c11 * tx
    return top * (1 - ty) + bot * ty
end

# ========================== Mipmaps ==========================

"""Build a box-filtered mipmap pyramid down to 1×1 (three.js `generateMipmaps`)."""
function generate_mipmaps!(tex::Texture)
    empty!(tex.mipmaps)
    cur = tex.data
    while size(cur, 1) > 1 || size(cur, 2) > 1
        H, W, C = size(cur)
        nh = max(H ÷ 2, 1); nw = max(W ÷ 2, 1)
        nxt = zeros(Float64, nh, nw, C)
        @inbounds for c in 1:C, i in 1:nh, j in 1:nw
            i0 = min(2i-1, H); i1 = min(2i, H); j0 = min(2j-1, W); j1 = min(2j, W)
            nxt[i,j,c] = 0.25*(cur[i0,j0,c] + cur[i1,j0,c] + cur[i0,j1,c] + cur[i1,j1,c])
        end
        push!(tex.mipmaps, nxt)
        cur = nxt
    end
    return tex
end

"""Sample a discrete mip level (0 = base). Clamped to the available levels."""
function sample_texture_lod(tex::Texture, u, v, lod::Int)
    lod <= 0 && return sample_texture(tex, u, v)
    isempty(tex.mipmaps) && return sample_texture(tex, u, v)
    lvl = tex.mipmaps[min(lod, length(tex.mipmaps))]
    tmp = Texture(lvl; wrap_s=tex.wrap_s, wrap_t=tex.wrap_t, filter=tex.filter)
    return sample_texture(tmp, u, v)
end

# ========================== CubeTexture ==========================

struct CubeTexture
    faces::NTuple{6, Texture}   # +x, -x, +y, -y, +z, -z
end

"""Sample a cube map along direction `dir` (three.js cube-map convention)."""
function sample_cube(ct::CubeTexture, dir::Vec3)
    ax, ay, az = abs(dir.x), abs(dir.y), abs(dir.z)
    if ax >= ay && ax >= az
        if dir.x > 0; return sample_texture(ct.faces[1], 0.5 - dir.z/(2dir.x), 0.5 - dir.y/(2ax))
        else;         return sample_texture(ct.faces[2], 0.5 - dir.z/(2dir.x), 0.5 + dir.y/(2ax)); end
    elseif ay >= ax && ay >= az
        if dir.y > 0; return sample_texture(ct.faces[3], 0.5 + dir.x/(2ay), 0.5 + dir.z/(2dir.y))
        else;         return sample_texture(ct.faces[4], 0.5 + dir.x/(2ay), 0.5 - dir.z/(2dir.y)); end
    else
        if dir.z > 0; return sample_texture(ct.faces[5], 0.5 + dir.x/(2dir.z), 0.5 - dir.y/(2az))
        else;         return sample_texture(ct.faces[6], 0.5 + dir.x/(2dir.z), 0.5 + dir.y/(2az)); end
    end
end

# ========================== Procedural textures ==========================

"""`n`×`n`-cell checkerboard texture (each cell `cell` pixels) as an H×W×3 Texture."""
function checker_texture(; n::Int=8, cell::Int=8, a=Color3(1.0,1.0,1.0), b=Color3(0.0,0.0,0.0),
                          wrap_s=:repeat, wrap_t=:repeat, filter=:nearest)
    sz = n * cell
    data = Array{Float64}(undef, sz, sz, 3)
    @inbounds for i in 1:sz, j in 1:sz
        ci = (i - 1) ÷ cell; cj = (j - 1) ÷ cell
        col = iseven(ci + cj) ? a : b
        data[i,j,1] = col.r; data[i,j,2] = col.g; data[i,j,3] = col.b
    end
    Texture(data; wrap_s=wrap_s, wrap_t=wrap_t, filter=filter)
end

"""Grid texture: `line` color on grid lines every `cell` pixels, else `bg`."""
function grid_texture(; size_px::Int=64, cell::Int=16, line=Color3(0.0,0.0,0.0),
                       bg=Color3(1.0,1.0,1.0), thickness::Int=1, wrap_s=:repeat, wrap_t=:repeat)
    data = Array{Float64}(undef, size_px, size_px, 3)
    @inbounds for i in 1:size_px, j in 1:size_px
        on = ((i-1) % cell < thickness) || ((j-1) % cell < thickness)
        col = on ? line : bg
        data[i,j,1] = col.r; data[i,j,2] = col.g; data[i,j,3] = col.b
    end
    Texture(data; wrap_s=wrap_s, wrap_t=wrap_t, filter=:nearest)
end
