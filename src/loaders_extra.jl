# --------------------------------------------------------------------------
# Extended loaders: PNG decode (pure-Julia INFLATE), TextureLoader, OBJ .mtl
# materials, and a minimal glTF 2.0 loader (embedded base64 buffers). All pure
# Julia, no external dependencies.
# --------------------------------------------------------------------------

# ========================== DEFLATE / INFLATE ==========================

mutable struct _BitReader
    data::Vector{UInt8}
    pos::Int
    bitbuf::UInt32
    bitcnt::Int
end
_BitReader(d::Vector{UInt8}) = _BitReader(d, 1, UInt32(0), 0)

@inline function _getbit(br::_BitReader)
    if br.bitcnt == 0
        br.bitbuf = UInt32(br.data[br.pos]); br.pos += 1; br.bitcnt = 8
    end
    b = br.bitbuf & 0x1
    br.bitbuf >>= 1; br.bitcnt -= 1
    return Int(b)
end
@inline function _getbits(br::_BitReader, n::Int)
    v = 0
    for i in 0:n-1
        v |= _getbit(br) << i
    end
    return v
end

# Canonical Huffman decode table (RFC 1951). Decoding walks MSB-first one bit at
# a time, but each length check is O(1) instead of scanning all symbols.
#
# Fields, indexed by code length `len` (1..maxbits):
#   first_code[len]   smallest canonical code of that length
#   first_index[len]  offset into `symbols` where that length's symbols start
#   count[len]        number of symbols of that length
#   symbols           symbol values (0-based) sorted by (length, code), i.e. by
#                     ascending symbol index within each length — exactly the
#                     order `_build_huff` assigns consecutive codes, so
#                     symbol = symbols[first_index[len] + (code - first_code[len])].
struct _Huff
    maxbits::Int
    first_code::Vector{Int}
    first_index::Vector{Int}
    count::Vector{Int}
    symbols::Vector{Int}
end

# Build the canonical-Huffman fast-decode table from per-symbol code lengths.
# Produces the identical canonical code assignment as the previous (lengths,
# codes) form: within a length, codes increase in symbol-index order.
function _build_huff(lengths::Vector{Int})
    maxbits = isempty(lengths) ? 0 : maximum(lengths)
    if maxbits == 0
        return _Huff(0, Int[], Int[], Int[], Int[])
    end
    # Count codes per length and derive the first canonical code per length.
    blcount = zeros(Int, maxbits + 1)        # blcount[len+1] = #codes of length len
    @inbounds for l in lengths
        l > 0 && (blcount[l + 1] += 1)
    end
    first_code = zeros(Int, maxbits)         # first_code[len]
    count = zeros(Int, maxbits)              # count[len]
    code = 0
    @inbounds for len in 1:maxbits
        code = (code + blcount[len]) << 1    # matches _build_huff(prev) recurrence
        first_code[len] = code
        count[len] = blcount[len + 1]
    end
    # `symbols` holds symbol values grouped by length, in ascending symbol order.
    first_index = zeros(Int, maxbits)        # offset (0-based) into symbols per length
    acc = 0
    @inbounds for len in 1:maxbits
        first_index[len] = acc
        acc += count[len]
    end
    symbols = Vector{Int}(undef, acc)
    fill_pos = copy(first_index)             # running write cursor per length
    @inbounds for n in 1:length(lengths)
        l = lengths[n]
        if l > 0
            symbols[fill_pos[l] + 1] = n - 1 # 0-based symbol value
            fill_pos[l] += 1
        end
    end
    return _Huff(maxbits, first_code, first_index, count, symbols)
end

# Decode one symbol. O(code length) bit reads with O(1) work per bit.
@inline function _decode_sym(br::_BitReader, huff::_Huff)
    code = 0
    @inbounds for len in 1:huff.maxbits
        code = (code << 1) | _getbit(br)
        cnt = huff.count[len]
        if cnt > 0
            off = code - huff.first_code[len]
            if 0 <= off < cnt
                return huff.symbols[huff.first_index[len] + off + 1]
            end
        end
    end
    error("invalid Huffman code")
end

const _LEN_BASE   = [3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258]
const _LEN_EXTRA  = [0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0]
const _DIST_BASE  = [1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577]
const _DIST_EXTRA = [0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13]

function _fixed_huffs()
    litlen = Vector{Int}(undef, 288)
    for i in 1:288
        litlen[i] = i <= 144 ? 8 : i <= 256 ? 9 : i <= 280 ? 7 : 8
    end
    (_build_huff(litlen), _build_huff(fill(5, 30)))
end

function _read_dynamic(br::_BitReader)
    hlit = _getbits(br, 5) + 257
    hdist = _getbits(br, 5) + 1
    hclen = _getbits(br, 4) + 4
    order = [16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15]
    cl_lengths = zeros(Int, 19)
    for i in 1:hclen
        cl_lengths[order[i] + 1] = _getbits(br, 3)
    end
    cl_huff = _build_huff(cl_lengths)
    all_lengths = Int[]
    while length(all_lengths) < hlit + hdist
        sym = _decode_sym(br, cl_huff)
        if sym < 16
            push!(all_lengths, sym)
        elseif sym == 16
            rep = _getbits(br, 2) + 3; prev = all_lengths[end]
            append!(all_lengths, fill(prev, rep))
        elseif sym == 17
            append!(all_lengths, fill(0, _getbits(br, 3) + 3))
        else
            append!(all_lengths, fill(0, _getbits(br, 7) + 11))
        end
    end
    lit = _build_huff(all_lengths[1:hlit])
    dist = _build_huff(all_lengths[hlit+1:hlit+hdist])
    return (lit, dist)
end

function _inflate_block!(out::Vector{UInt8}, br::_BitReader, lit, dist)
    while true
        sym = _decode_sym(br, lit)
        if sym < 256
            push!(out, UInt8(sym))
        elseif sym == 256
            break
        else
            li = sym - 256
            len = _LEN_BASE[li] + _getbits(br, _LEN_EXTRA[li])
            dsym = _decode_sym(br, dist)
            d = _DIST_BASE[dsym + 1] + _getbits(br, _DIST_EXTRA[dsym + 1])
            start = length(out) - d
            for k in 1:len
                push!(out, out[start + k])
            end
        end
    end
end

"""Inflate a raw DEFLATE stream (no zlib header) to bytes."""
function inflate(data::Vector{UInt8})
    br = _BitReader(data); out = UInt8[]
    while true
        bfinal = _getbit(br); btype = _getbits(br, 2)
        if btype == 0
            br.bitcnt = 0                       # align to byte boundary
            len = Int(br.data[br.pos]) | (Int(br.data[br.pos + 1]) << 8)
            br.pos += 4                         # skip LEN(2) + NLEN(2)
            for _ in 1:len
                push!(out, br.data[br.pos]); br.pos += 1
            end
        elseif btype == 1
            lit, dist = _fixed_huffs(); _inflate_block!(out, br, lit, dist)
        elseif btype == 2
            lit, dist = _read_dynamic(br); _inflate_block!(out, br, lit, dist)
        else
            error("invalid DEFLATE block type 3")
        end
        bfinal == 1 && break
    end
    return out
end

zlib_inflate(data::Vector{UInt8}) = inflate(data[3:end])   # skip 2-byte zlib header; ignore Adler trailer

# ========================== PNG decode ==========================

@inline _rd_be32(b, i) = (Int(b[i]) << 24) | (Int(b[i+1]) << 16) | (Int(b[i+2]) << 8) | Int(b[i+3])

@inline function _paeth(a, b, c)
    p = a + b - c
    pa = abs(p - a); pb = abs(p - b); pc = abs(p - c)
    return pa <= pb && pa <= pc ? a : (pb <= pc ? b : c)
end

"""
    load_png(path) -> Array{Float64,3}

Decode an 8-bit PNG (grayscale, RGB, or RGBA) to an H×W×C array in [0,1].
Implements full INFLATE (stored/fixed/dynamic Huffman) and all five PNG filters.
"""
function load_png(path::String)
    bytes = read(path)
    bytes[1:8] == UInt8[137,80,78,71,13,10,26,10] || error("not a PNG file")
    pos = 9; W = 0; H = 0; bitdepth = 8; colortype = 2
    idat = UInt8[]
    while pos <= length(bytes)
        len = _rd_be32(bytes, pos); pos += 4
        ctype = String(bytes[pos:pos+3]); pos += 4
        if ctype == "IHDR"
            W = _rd_be32(bytes, pos); H = _rd_be32(bytes, pos+4)
            bitdepth = bytes[pos+8]; colortype = bytes[pos+9]
        elseif ctype == "IDAT"
            append!(idat, @view bytes[pos:pos+len-1])
        elseif ctype == "IEND"
            break
        end
        pos += len + 4                          # data + CRC
    end
    (bitdepth == 8 || bitdepth == 16) || error("only 8-bit and 16-bit PNG decode is supported")
    channels = colortype == 0 ? 1 : colortype == 2 ? 3 : colortype == 6 ? 4 :
               error("unsupported PNG color type $colortype")
    bps = bitdepth ÷ 8                          # bytes per sample
    bpp = channels * bps                        # bytes per pixel (filter window)
    raw = zlib_inflate(idat)
    stride = W * bpp
    img = Array{Float64}(undef, H, W, channels)
    norm = bitdepth == 16 ? 65535.0 : 255.0
    prev = zeros(UInt8, stride)
    p = 1
    for row in 1:H
        ftype = raw[p]; p += 1
        cur = Vector{UInt8}(raw[p:p+stride-1]); p += stride
        # Unfilter in place (filter window is one pixel = bpp bytes).
        for i in 1:stride
            a = i > bpp ? cur[i-bpp] : 0x00
            b = prev[i]
            c = i > bpp ? prev[i-bpp] : 0x00
            x = cur[i]
            cur[i] = if ftype == 0; x
                     elseif ftype == 1; x + a
                     elseif ftype == 2; x + b
                     elseif ftype == 3; x + UInt8((Int(a) + Int(b)) ÷ 2)
                     elseif ftype == 4; x + UInt8(_paeth(Int(a), Int(b), Int(c)))
                     else error("bad PNG filter $ftype") end
        end
        for j in 1:W, c in 1:channels
            base = (j-1)*bpp + (c-1)*bps + 1
            img[row, j, c] = bps == 2 ? ((Int(cur[base]) << 8) | Int(cur[base+1])) / norm :
                                        cur[base] / norm
        end
        prev = cur
    end
    return img
end

"""Load a PNG into a [`Texture`]."""
TextureLoader(path::String; kwargs...) = Texture(load_png(path); kwargs...)

# ========================== OBJ .mtl materials ==========================

"""
    load_mtl(path) -> Dict{String, MeshPhongMaterial}

Parse a Wavefront .mtl file: `newmtl`, `Kd` (diffuse), `Ks` (specular),
`Ns` (shininess), `Ke` (emissive), `d`/`Tr` (opacity).
"""
function load_mtl(path::String)
    mats = Dict{String, MeshPhongMaterial}()
    name = ""
    kd = Color3(1.0,1.0,1.0); ks = Color3(0.0,0.0,0.0); ke = Color3(0.0,0.0,0.0)
    ns = 30.0; d = 1.0
    function flush!()
        isempty(name) && return
        mats[name] = MeshPhongMaterial(color=kd, specular=ks, emissive=ke, shininess=ns,
                                       opacity=d, transparent=(d < 1.0))
    end
    for raw in eachline(path)
        t = split(strip(raw))
        isempty(t) && continue
        tag = t[1]
        if tag == "newmtl"
            flush!()
            name = t[2]; kd = Color3(1.0,1.0,1.0); ks = Color3(0.0,0.0,0.0)
            ke = Color3(0.0,0.0,0.0); ns = 30.0; d = 1.0
        elseif tag == "Kd"; kd = Color3(parse(Float64,t[2]), parse(Float64,t[3]), parse(Float64,t[4]))
        elseif tag == "Ks"; ks = Color3(parse(Float64,t[2]), parse(Float64,t[3]), parse(Float64,t[4]))
        elseif tag == "Ke"; ke = Color3(parse(Float64,t[2]), parse(Float64,t[3]), parse(Float64,t[4]))
        elseif tag == "Ns"; ns = parse(Float64, t[2])
        elseif tag == "d";  d = parse(Float64, t[2])
        elseif tag == "Tr"; d = 1.0 - parse(Float64, t[2])
        end
    end
    flush!()
    return mats
end

"""
    load_obj_groups(path) -> (geometry, face_material_names, materials)

Like [`load_obj`](@ref) but also returns, per triangle, the active `usemtl`
name and the material dictionary parsed from any referenced `mtllib`.
"""
function load_obj_groups(path::String)
    verts = Float64[]; file_normals = Float64[]
    out_pos = Float64[]; out_nrm = Float64[]; indices = Int[]
    face_mtl = String[]
    have_normals = false; out_vi = 0; cur_mtl = ""
    materials = Dict{String, MeshPhongMaterial}()
    parse_index(tok, n) = (i = parse(Int, tok); i < 0 ? n + i + 1 : i)
    dir = dirname(path)
    for raw in eachline(path)
        line = strip(raw)
        (isempty(line) || startswith(line, "#")) && continue
        t = split(line); tag = t[1]
        if tag == "v"
            push!(verts, parse(Float64,t[2]), parse(Float64,t[3]), parse(Float64,t[4]))
        elseif tag == "vn"
            push!(file_normals, parse(Float64,t[2]), parse(Float64,t[3]), parse(Float64,t[4])); have_normals = true
        elseif tag == "mtllib"
            mp = joinpath(dir, t[2]); isfile(mp) && merge!(materials, load_mtl(mp))
        elseif tag == "usemtl"
            cur_mtl = t[2]
        elseif tag == "f"
            nv = length(verts) ÷ 3; nn = length(file_normals) ÷ 3
            corners = t[2:end]
            for k in 2:(length(corners) - 1)
                for c in (corners[1], corners[k], corners[k+1])
                    sub = split(c, '/')
                    vidx = parse_index(sub[1], nv); base = (vidx-1)*3
                    push!(out_pos, verts[base+1], verts[base+2], verts[base+3])
                    if have_normals && length(sub) >= 3 && !isempty(sub[3])
                        nidx = parse_index(sub[3], nn); nb = (nidx-1)*3
                        push!(out_nrm, file_normals[nb+1], file_normals[nb+2], file_normals[nb+3])
                    else
                        push!(out_nrm, 0.0, 0.0, 0.0)
                    end
                    out_vi += 1; push!(indices, out_vi)
                end
                push!(face_mtl, cur_mtl)
            end
        end
    end
    nfaces = length(indices) ÷ 3
    geo = BufferGeometry(out_pos, out_nrm, Float64[], indices, out_vi, nfaces)
    # Recompute smooth normals when the file had none, or when ANY emitted vertex
    # normal is zero-length (some faces lacked vn) — otherwise those vertices
    # keep a degenerate (0,0,0) normal and shade black.
    needs_recompute = !have_normals
    if !needs_recompute
        @inbounds for b in 1:3:length(out_nrm)
            if out_nrm[b] == 0.0 && out_nrm[b+1] == 0.0 && out_nrm[b+2] == 0.0
                needs_recompute = true
                break
            end
        end
    end
    needs_recompute && compute_vertex_normals!(geo)
    return (geo, face_mtl, materials)
end

# ========================== Minimal JSON parser ==========================
# Supports objects, arrays, strings, numbers, true/false/null — enough for glTF.

mutable struct _JSONParser
    s::String
    i::Int
end

function _json_parse(s::String)
    p = _JSONParser(s, 1)
    _json_ws(p)
    v = _json_value(p)
    return v
end

@inline function _json_ws(p)
    n = ncodeunits(p.s)
    while p.i <= n && (p.s[p.i] in (' ', '\t', '\n', '\r'))
        p.i += 1
    end
end

function _json_value(p)
    c = p.s[p.i]
    if c == '{'; return _json_object(p)
    elseif c == '['; return _json_array(p)
    elseif c == '"'; return _json_string(p)
    elseif c == 't'; p.i += 4; return true
    elseif c == 'f'; p.i += 5; return false
    elseif c == 'n'; p.i += 4; return nothing
    else; return _json_number(p)
    end
end

function _json_object(p)
    d = Dict{String, Any}(); p.i += 1; _json_ws(p)
    p.s[p.i] == '}' && (p.i += 1; return d)
    while true
        _json_ws(p); key = _json_string(p); _json_ws(p)
        p.i += 1                                # ':'
        _json_ws(p); d[key] = _json_value(p); _json_ws(p)
        if p.s[p.i] == ','; p.i += 1
        else; p.i += 1; break; end              # '}'
    end
    return d
end

function _json_array(p)
    a = Any[]; p.i += 1; _json_ws(p)
    p.s[p.i] == ']' && (p.i += 1; return a)
    while true
        _json_ws(p); push!(a, _json_value(p)); _json_ws(p)
        if p.s[p.i] == ','; p.i += 1
        else; p.i += 1; break; end              # ']'
    end
    return a
end

function _json_string(p)
    p.i += 1; io = IOBuffer()
    while p.s[p.i] != '"'
        c = p.s[p.i]
        if c == '\\'
            p.i += 1; e = p.s[p.i]
            print(io, e == 'n' ? '\n' : e == 't' ? '\t' : e == 'r' ? '\r' : e)
        else
            print(io, c)
        end
        p.i += 1
    end
    p.i += 1
    return String(take!(io))
end

function _json_number(p)
    start = p.i; n = ncodeunits(p.s)
    while p.i <= n && (p.s[p.i] in ('-','+','.','e','E','0','1','2','3','4','5','6','7','8','9'))
        p.i += 1
    end
    return parse(Float64, p.s[start:p.i-1])
end

# ========================== base64 ==========================

const _B64_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
const _B64_LUT = let lut = fill(-1, 256); for (k, ch) in enumerate(_B64_CHARS); lut[Int(ch)+1] = k-1; end; lut end

function base64_decode(s::AbstractString)
    out = UInt8[]; acc = 0; nbits = 0
    for ch in s
        ch in ('=', '\n', '\r', ' ') && continue
        v = _B64_LUT[Int(ch) + 1]
        v < 0 && continue
        acc = (acc << 6) | v; nbits += 6
        if nbits >= 8
            nbits -= 8; push!(out, UInt8((acc >> nbits) & 0xff))
        end
    end
    return out
end

# ========================== glTF 2.0 ==========================

const _GLTF_COMP_SIZE = Dict("SCALAR"=>1, "VEC2"=>2, "VEC3"=>3, "VEC4"=>4, "MAT4"=>16)

function _gltf_read_buffer(buf::Dict, dir::String)
    uri = get(buf, "uri", nothing)
    uri === nothing && error("glTF buffer without uri (GLB not supported)")
    if startswith(uri, "data:")
        return base64_decode(split(uri, ",", limit=2)[2])
    else
        return read(joinpath(dir, uri))
    end
end

# Read accessor `ai` (0-based) as a vector of Float64 tuples / scalars.
function _gltf_accessor(gltf, buffers, ai::Int)
    acc = gltf["accessors"][ai + 1]
    bv = gltf["bufferViews"][Int(acc["bufferView"]) + 1]
    buf = buffers[Int(bv["buffer"]) + 1]
    offset = Int(get(bv, "byteOffset", 0.0)) + Int(get(acc, "byteOffset", 0.0))
    count = Int(acc["count"])
    ncomp = _GLTF_COMP_SIZE[acc["type"]]
    ctype = Int(acc["componentType"])     # 5126=float,5125=uint32,5123=ushort,5121=ubyte
    compbytes = (ctype == 5126 || ctype == 5125) ? 4 :
                ctype == 5123 ? 2 : ctype == 5121 ? 1 :
                error("glTF componentType $ctype")
    stride = Int(get(bv, "byteStride", 0.0))   # 0 (or absent) => tightly packed
    out = Vector{Float64}(undef, count * ncomp)
    io = IOBuffer(buf)
    read_comp() = ctype == 5126 ? Float64(read(io, Float32)) :
                  ctype == 5125 ? Float64(read(io, UInt32)) :
                  ctype == 5123 ? Float64(read(io, UInt16)) :
                  Float64(read(io, UInt8))
    if stride == 0 || stride == ncomp * compbytes
        seek(io, offset)
        for k in 1:count*ncomp
            out[k] = read_comp()
        end
    else
        # Interleaved buffer view: seek to each element start before reading.
        for e in 0:count-1
            seek(io, offset + e * stride)
            base = e * ncomp
            for c in 1:ncomp
                out[base + c] = read_comp()
            end
        end
    end
    return (out, ncomp, count)
end

function _gltf_material(gltf, mi)
    mi === nothing && return MeshStandardMaterial()
    m = gltf["materials"][Int(mi) + 1]
    pbr = get(m, "pbrMetallicRoughness", Dict{String,Any}())
    bc = get(pbr, "baseColorFactor", [1.0,1.0,1.0,1.0])
    MeshStandardMaterial(color=Color3(bc[1], bc[2], bc[3]),
                         metalness=Float64(get(pbr, "metallicFactor", 1.0)),
                         roughness=Float64(get(pbr, "roughnessFactor", 1.0)))
end

function _gltf_node_matrix(node)
    if haskey(node, "matrix")
        m = node["matrix"]
        return Mat4{Float64}(ntuple(k -> Float64(m[k]), 16))
    end
    t = get(node, "translation", [0.0,0.0,0.0])
    r = get(node, "rotation", [0.0,0.0,0.0,1.0])
    s = get(node, "scale", [1.0,1.0,1.0])
    T = mat4_translation(t[1], t[2], t[3])
    R = quat_to_mat4(Quaternion(r[1], r[2], r[3], r[4]))
    S = mat4_scaling(s[1], s[2], s[3])
    return T * R * S
end

# Decompose a column-major TRS matrix `M` into (position, rotation::Euler{:XYZ},
# scale), matching three.js `Matrix4.decompose` + `Euler.setFromRotationMatrix`
# (order XYZ). Returned components recompose as T*R*S exactly as
# `compute_local_matrix`.
function _gltf_decompose(M::Mat4)
    position = Vec3(mat4_get(M,1,4), mat4_get(M,2,4), mat4_get(M,3,4))
    # Column vectors of the upper-left 3x3.
    c1 = Vec3(mat4_get(M,1,1), mat4_get(M,2,1), mat4_get(M,3,1))
    c2 = Vec3(mat4_get(M,1,2), mat4_get(M,2,2), mat4_get(M,3,2))
    c3 = Vec3(mat4_get(M,1,3), mat4_get(M,2,3), mat4_get(M,3,3))
    sx = norm(c1); sy = norm(c2); sz = norm(c3)
    # Negative determinant means a reflected basis; three.js folds the sign into sx.
    det = dot(c1, cross(c2, c3))
    if det < 0
        sx = -sx
        c1 = -c1
    end
    # Pure-rotation columns (guard against zero scale).
    isx = sx == 0 ? zero(sx) : one(sx)/sx
    isy = sy == 0 ? zero(sy) : one(sy)/sy
    isz = sz == 0 ? zero(sz) : one(sz)/sz
    r1 = c1 * isx; r2 = c2 * isy; r3 = c3 * isz
    # R indexed [row, col]: column r1 -> (R11,R21,R31), r2 -> (R12,R22,R32),
    # r3 -> (R13,R23,R33).
    R11 = r1.x
    R12 = r2.x; R22 = r2.y; R32 = r2.z
    R13 = r3.x; R23 = r3.y; R33 = r3.z
    _y = asin(clamp(R13, -one(R13), one(R13)))
    if abs(R13) < 0.9999999
        _x = atan(-R23, R33)
        _z = atan(-R12, R11)
    else
        _x = atan(R32, R22)
        _z = zero(R13)
    end
    return (position, Euler(_x, _y, _z, :XYZ), Vec3(sx, sy, sz))
end

# Build a `Scene` from a parsed glTF document and its decoded buffers. Shared by
# `load_gltf` (text/embedded buffers) and `load_glb` (binary container, where
# the BIN chunk is supplied as buffer 0). `buffers` must already contain the raw
# bytes for every buffer referenced by the document.
function _gltf_build_scene(gltf, buffers)
    scene = Scene()

    function build_primitive(prim)
        attrs = prim["attributes"]
        pos, _, nverts = _gltf_accessor(gltf, buffers, Int(attrs["POSITION"]))
        normals = Float64[]
        if haskey(attrs, "NORMAL")
            normals, _, _ = _gltf_accessor(gltf, buffers, Int(attrs["NORMAL"]))
        end
        if haskey(prim, "indices")
            idxf, _, _ = _gltf_accessor(gltf, buffers, Int(prim["indices"]))
            indices = Int.(round.(idxf)) .+ 1
        else
            indices = collect(1:nverts)
        end
        geo = BufferGeometry(pos, normals, Float64[], indices, nverts, length(indices) ÷ 3)
        isempty(normals) && compute_vertex_normals!(geo)
        mat = _gltf_material(gltf, get(prim, "material", nothing))
        Mesh(geo, mat)
    end

    function add_node!(parent, node_idx)
        node = gltf["nodes"][node_idx + 1]
        grp = Group()
        M = _gltf_node_matrix(node)
        pos, rot, scl = _gltf_decompose(M)   # full TRS, not just translation
        grp.position = pos
        grp.rotation = rot
        grp.scale = scl
        add!(parent, grp)
        if haskey(node, "mesh")
            mesh_def = gltf["meshes"][Int(node["mesh"]) + 1]
            for prim in mesh_def["primitives"]
                add!(grp, build_primitive(prim))
            end
        end
        for child in get(node, "children", Any[])
            add_node!(grp, Int(child))
        end
    end

    scene_def = gltf["scenes"][Int(get(gltf, "scene", 0.0)) + 1]
    for n in scene_def["nodes"]
        add_node!(scene, Int(n))
    end
    return scene
end

"""
    load_gltf(path) -> Scene

Load a glTF 2.0 file (embedded base64 or external buffers) into a `Scene`.
Supports node transforms, mesh primitives (POSITION, NORMAL, indices) and
basic PBR metallic-roughness materials. Skinning/morph targets are ignored.
"""
function load_gltf(path::String)
    gltf = _json_parse(read(path, String))
    dir = dirname(path)
    buffers = [_gltf_read_buffer(b, dir) for b in gltf["buffers"]]
    return _gltf_build_scene(gltf, buffers)
end

# Resolve a glTF buffer that may reference the GLB binary chunk. A buffer with no
# `uri` is the embedded GLB binary buffer (buffer 0 by spec); otherwise behave
# exactly like `_gltf_read_buffer`.
function _glb_read_buffer(buf::Dict, dir::String, bin::Vector{UInt8})
    uri = get(buf, "uri", nothing)
    if uri === nothing
        return bin
    elseif startswith(uri, "data:")
        return base64_decode(split(uri, ",", limit=2)[2])
    else
        return read(joinpath(dir, uri))
    end
end

@inline _rd_le32(b, i) = Int(b[i]) | (Int(b[i+1]) << 8) | (Int(b[i+2]) << 16) | (Int(b[i+3]) << 24)

"""
    load_glb(path) -> Scene

Load a binary glTF (`.glb`) container into a `Scene`. Parses the 12-byte header
(magic `glTF`, version, total length) and the chunk list, extracts the JSON
chunk (type `0x4E4F534A`) and the optional binary chunk (type `0x004E4942`),
then reuses the glTF document logic. A buffer view without a `uri` reads from
the embedded binary chunk (buffer 0). Mirrors [`load_gltf`](@ref) output.
"""
function load_glb(path::String)
    bytes = read(path)
    length(bytes) >= 12 || error("GLB file too short")
    # 12-byte header: magic, version, total length (all little-endian uint32).
    magic = _rd_le32(bytes, 1)
    magic == 0x46546C67 || error("not a GLB file (bad magic)")   # 'glTF' little-endian
    version = _rd_le32(bytes, 5)
    version == 2 || error("unsupported GLB version $version")
    total = _rd_le32(bytes, 9)
    total = min(total, length(bytes))                            # tolerate over-long declared size

    json_bytes = UInt8[]
    bin_bytes = UInt8[]
    have_json = false
    pos = 13                                                     # first chunk header
    while pos + 8 <= total + 1
        clen = _rd_le32(bytes, pos)
        ctype = _rd_le32(bytes, pos + 4)
        dstart = pos + 8
        dend = dstart + clen - 1
        dend <= length(bytes) || error("GLB chunk exceeds file bounds")
        if ctype == 0x4E4F534A          # 'JSON'
            json_bytes = bytes[dstart:dend]
            have_json = true
        elseif ctype == 0x004E4942      # 'BIN\0'
            bin_bytes = bytes[dstart:dend]
        end
        # Chunks are 4-byte aligned; advance over any padding to the next header.
        pad = (4 - (clen % 4)) % 4
        pos = dend + 1 + pad
    end
    have_json || error("GLB has no JSON chunk")

    gltf = _json_parse(String(json_bytes))
    dir = dirname(path)
    buffers = [_glb_read_buffer(b, dir, bin_bytes) for b in gltf["buffers"]]
    return _gltf_build_scene(gltf, buffers)
end
