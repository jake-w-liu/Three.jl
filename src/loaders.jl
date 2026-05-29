# --------------------------------------------------------------------------
# Mesh loaders/writers: STL (binary + ASCII) and OBJ, plus smooth-normal
# computation. Loaders return a BufferGeometry usable by the rasterizer.
# Pure Julia, no external dependencies.
# --------------------------------------------------------------------------

"""
    compute_vertex_normals!(geo) -> geo

Recompute per-vertex normals as the area-weighted average of adjacent face
normals (smooth normals).  Overwrites `geo.normals`.
"""
function compute_vertex_normals!(geo::BufferGeometry)
    nv = geo.n_vertices
    acc = zeros(Float64, nv * 3)
    @inbounds for fi in 1:geo.n_faces
        i1, i2, i3 = get_face(geo, fi)
        v1 = get_vertex(geo, i1); v2 = get_vertex(geo, i2); v3 = get_vertex(geo, i3)
        # Cross product is proportional to face area, giving area weighting.
        fn = cross(v2 - v1, v3 - v1)
        for idx in (i1, i2, i3)
            base = (idx - 1) * 3
            acc[base+1] += fn.x; acc[base+2] += fn.y; acc[base+3] += fn.z
        end
    end
    @inbounds for vi in 1:nv
        base = (vi - 1) * 3
        nx, ny, nz = acc[base+1], acc[base+2], acc[base+3]
        len = sqrt(nx*nx + ny*ny + nz*nz)
        if len > 1e-20
            acc[base+1] = nx/len; acc[base+2] = ny/len; acc[base+3] = nz/len
        else
            acc[base+1] = 0.0; acc[base+2] = 0.0; acc[base+3] = 1.0
        end
    end
    geo.normals = acc
    return geo
end

# ========================== STL ==========================

"""
    save_stl_binary(path, geo) -> path

Write `geo` as a binary STL file (per-triangle facet normals computed from
geometry).  Round-trips with [`load_stl`](@ref).
"""
function save_stl_binary(path::String, geo::BufferGeometry)
    open(path, "w") do io
        write(io, zeros(UInt8, 80))                 # 80-byte header
        write(io, UInt32(geo.n_faces))
        for fi in 1:geo.n_faces
            i1, i2, i3 = get_face(geo, fi)
            v1 = get_vertex(geo, i1); v2 = get_vertex(geo, i2); v3 = get_vertex(geo, i3)
            n = compute_face_normal(geo, fi)
            for c in (Float32(n.x), Float32(n.y), Float32(n.z))
                write(io, c)
            end
            for v in (v1, v2, v3)
                write(io, Float32(v.x)); write(io, Float32(v.y)); write(io, Float32(v.z))
            end
            write(io, UInt16(0))                     # attribute byte count
        end
    end
    return path
end

function _looks_binary_stl(path::String)
    sz = filesize(path)
    sz < 84 && return false
    ntri = open(path, "r") do io
        seek(io, 80)
        read(io, UInt32)
    end
    return sz == 84 + 50 * Int(ntri)               # exact binary STL size
end

"""
    load_stl(path) -> BufferGeometry

Load an STL mesh, auto-detecting binary vs ASCII.  Each triangle contributes
three independent vertices; call [`compute_vertex_normals!`](@ref) afterward
for smooth shading.
"""
function load_stl(path::String)
    return _looks_binary_stl(path) ? _load_stl_binary(path) : _load_stl_ascii(path)
end

function _load_stl_binary(path::String)
    open(path, "r") do io
        seek(io, 80)
        ntri = Int(read(io, UInt32))
        positions = Vector{Float64}(undef, ntri * 9)
        normals = Vector{Float64}(undef, ntri * 9)
        indices = Vector{Int}(undef, ntri * 3)
        p = 1; vi = 0
        for _ in 1:ntri
            nx = Float64(read(io, Float32)); ny = Float64(read(io, Float32)); nz = Float64(read(io, Float32))
            for _v in 1:3
                x = Float64(read(io, Float32)); y = Float64(read(io, Float32)); z = Float64(read(io, Float32))
                positions[p] = x; positions[p+1] = y; positions[p+2] = z
                normals[p] = nx; normals[p+1] = ny; normals[p+2] = nz
                p += 3; vi += 1; indices[vi] = vi
            end
            read(io, UInt16)                         # attribute byte count
        end
        return BufferGeometry(positions, normals, Float64[], indices, ntri * 3, ntri)
    end
end

function _load_stl_ascii(path::String)
    positions = Float64[]; normals = Float64[]; indices = Int[]
    cur_n = (0.0, 0.0, 0.0); vi = 0
    for raw in eachline(path)
        line = strip(raw)
        if startswith(line, "facet normal")
            t = split(line)
            cur_n = (parse(Float64, t[3]), parse(Float64, t[4]), parse(Float64, t[5]))
        elseif startswith(line, "vertex")
            t = split(line)
            push!(positions, parse(Float64, t[2]), parse(Float64, t[3]), parse(Float64, t[4]))
            push!(normals, cur_n[1], cur_n[2], cur_n[3])
            vi += 1; push!(indices, vi)
        end
    end
    nfaces = length(indices) ÷ 3
    return BufferGeometry(positions, normals, Float64[], indices, vi, nfaces)
end

# ========================== OBJ ==========================

"""
    load_obj(path) -> BufferGeometry

Load a Wavefront OBJ mesh (vertices and faces; polygons fan-triangulated).
Normals are taken from the file when present, otherwise computed smoothly.
Texture coordinates and materials are ignored.
"""
function load_obj(path::String)
    verts = Float64[]            # v
    file_normals = Float64[]     # vn
    out_pos = Float64[]
    out_nrm = Float64[]
    indices = Int[]
    have_normals = false
    out_vi = 0

    parse_index(tok, n) = (i = parse(Int, tok); i < 0 ? n + i + 1 : i)

    for raw in eachline(path)
        line = strip(raw)
        (isempty(line) || startswith(line, "#")) && continue
        t = split(line)
        tag = t[1]
        if tag == "v"
            push!(verts, parse(Float64, t[2]), parse(Float64, t[3]), parse(Float64, t[4]))
        elseif tag == "vn"
            push!(file_normals, parse(Float64, t[2]), parse(Float64, t[3]), parse(Float64, t[4]))
            have_normals = true
        elseif tag == "f"
            nverts_v = length(verts) ÷ 3
            nverts_n = length(file_normals) ÷ 3
            corners = t[2:end]
            # Fan-triangulate polygon (corner 1, k, k+1).
            for k in 2:(length(corners) - 1)
                for c in (corners[1], corners[k], corners[k+1])
                    sub = split(c, '/')
                    vidx = parse_index(sub[1], nverts_v)
                    base = (vidx - 1) * 3
                    push!(out_pos, verts[base+1], verts[base+2], verts[base+3])
                    if have_normals && length(sub) >= 3 && !isempty(sub[3])
                        nidx = parse_index(sub[3], nverts_n)
                        nb = (nidx - 1) * 3
                        push!(out_nrm, file_normals[nb+1], file_normals[nb+2], file_normals[nb+3])
                    else
                        push!(out_nrm, 0.0, 0.0, 0.0)
                    end
                    out_vi += 1; push!(indices, out_vi)
                end
            end
        end
    end
    nfaces = length(indices) ÷ 3
    geo = BufferGeometry(out_pos, out_nrm, Float64[], indices, out_vi, nfaces)
    (!have_normals || all(==(0.0), out_nrm)) && compute_vertex_normals!(geo)
    return geo
end
