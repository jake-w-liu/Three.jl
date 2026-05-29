using Test
using Three
using ForwardDiff

@testset "Three.jl" begin

    @testset "Vec3 arithmetic" begin
        a = Vec3(1.0, 2.0, 3.0)
        b = Vec3(4.0, 5.0, 6.0)
        c = a + b
        @test c.x ≈ 5.0
        @test c.y ≈ 7.0
        @test c.z ≈ 9.0

        d = a - b
        @test d.x ≈ -3.0

        s = a * 2.0
        @test s.x ≈ 2.0 && s.y ≈ 4.0 && s.z ≈ 6.0

        @test dot(a, b) ≈ 32.0  # 1*4 + 2*5 + 3*6

        cr = cross(Vec3(1,0,0), Vec3(0,1,0))
        @test cr.x ≈ 0.0 && cr.y ≈ 0.0 && cr.z ≈ 1.0

        @test norm(Vec3(3.0, 4.0, 0.0)) ≈ 5.0

        n = normalize(Vec3(0.0, 0.0, 5.0))
        @test n.z ≈ 1.0
        @test norm(n) ≈ 1.0
    end

    @testset "Color3" begin
        c1 = Color3(0.5, 0.3, 0.1)
        c2 = Color3(0.2, 0.4, 0.6)
        c3 = c1 + c2
        @test c3.r ≈ 0.7
        @test c3.g ≈ 0.7
        @test c3.b ≈ 0.7

        c4 = c1 * 2.0
        @test c4.r ≈ 1.0

        c5 = c1 * c2  # component-wise
        @test c5.r ≈ 0.1

        c_hex = Color3(UInt32(0xFF8000))
        @test c_hex.r ≈ 1.0
        @test c_hex.g ≈ 128/255 atol=0.01
    end

    @testset "Mat4 identity and transform" begin
        I4 = Mat4()
        v = Vec4(1.0, 2.0, 3.0, 1.0)
        w = mat4_transform_vec4(I4, v)
        @test w.x ≈ 1.0 && w.y ≈ 2.0 && w.z ≈ 3.0 && w.w ≈ 1.0

        T = mat4_translation(10.0, 20.0, 30.0)
        p = mat4_transform_point(T, Vec3(1.0, 2.0, 3.0))
        @test p.x ≈ 11.0
        @test p.y ≈ 22.0
        @test p.z ≈ 33.0

        S = mat4_scaling(2.0, 3.0, 4.0)
        ps = mat4_transform_point(S, Vec3(1.0, 1.0, 1.0))
        @test ps.x ≈ 2.0 && ps.y ≈ 3.0 && ps.z ≈ 4.0
    end

    @testset "Mat4 rotation" begin
        # 90° rotation around Z: (1,0,0) → (0,1,0)
        Rz = mat4_rotation_z(π/2)
        p = mat4_transform_point(Rz, Vec3(1.0, 0.0, 0.0))
        @test p.x ≈ 0.0 atol=1e-12
        @test p.y ≈ 1.0 atol=1e-12
        @test p.z ≈ 0.0 atol=1e-12

        # 90° rotation around X: (0,1,0) → (0,0,1)
        Rx = mat4_rotation_x(π/2)
        p2 = mat4_transform_point(Rx, Vec3(0.0, 1.0, 0.0))
        @test p2.x ≈ 0.0 atol=1e-12
        @test p2.y ≈ 0.0 atol=1e-12
        @test p2.z ≈ 1.0 atol=1e-12
    end

    @testset "Mat4 inverse" begin
        T = mat4_translation(3.0, -7.0, 11.0)
        Tinv = mat4_inverse(T)
        prod = T * Tinv
        for i in 1:4
            for j in 1:4
                expected = i == j ? 1.0 : 0.0
                @test mat4_get(prod, i, j) ≈ expected atol=1e-10
            end
        end
    end

    @testset "Mat4 perspective projection" begin
        P = mat4_perspective(π/4, 1.0, 0.1, 100.0)
        # Point at origin along -Z should project to center
        p_clip = mat4_transform_vec4(P, Vec4(0.0, 0.0, -1.0, 1.0))
        ndc_x = p_clip.x / p_clip.w
        ndc_y = p_clip.y / p_clip.w
        @test ndc_x ≈ 0.0 atol=1e-10
        @test ndc_y ≈ 0.0 atol=1e-10
    end

    @testset "Quaternion" begin
        q = quat_from_euler(0.0, 0.0, π/2)
        m = quat_to_mat4(q)
        p = mat4_transform_point(m, Vec3(1.0, 0.0, 0.0))
        @test p.x ≈ 0.0 atol=1e-10
        @test p.y ≈ 1.0 atol=1e-10

        # Identity quaternion
        qi = Quaternion()
        mi = quat_to_mat4(qi)
        pi = mat4_transform_point(mi, Vec3(5.0, 3.0, 1.0))
        @test pi.x ≈ 5.0 && pi.y ≈ 3.0 && pi.z ≈ 1.0
    end

    @testset "BoxGeometry" begin
        geo = BoxGeometry(width=2.0, height=2.0, depth=2.0)
        @test geo.n_vertices == 24  # 4 per face × 6 faces
        @test geo.n_faces == 12     # 2 triangles per face × 6 faces
        @test length(geo.positions) == 24 * 3
        @test length(geo.normals) == 24 * 3
        @test length(geo.indices) == 12 * 3

        # Check vertex range
        for k in 1:3:length(geo.positions)
            @test -1.0 <= geo.positions[k] <= 1.0
            @test -1.0 <= geo.positions[k+1] <= 1.0
            @test -1.0 <= geo.positions[k+2] <= 1.0
        end
    end

    @testset "SphereGeometry" begin
        geo = SphereGeometry(radius=2.0, width_segments=16, height_segments=8)
        @test geo.n_vertices > 0
        @test geo.n_faces > 0

        # Check radius: all vertices should be at distance ~2 from origin
        for vi in 1:geo.n_vertices
            v = get_vertex(geo, vi)
            r = norm(v)
            @test r ≈ 2.0 atol=1e-10
        end
    end

    @testset "PlaneGeometry" begin
        geo = PlaneGeometry(width=4.0, height=4.0)
        @test geo.n_vertices == 4
        @test geo.n_faces == 2
    end

    @testset "CylinderGeometry" begin
        geo = CylinderGeometry(radius_top=1.0, radius_bottom=1.0, height=2.0)
        @test geo.n_faces > 0
        @test geo.n_vertices > 0
    end

    @testset "TorusGeometry" begin
        geo = TorusGeometry(radius=2.0, tube=0.5)
        @test geo.n_faces > 0
    end

    @testset "Scene graph" begin
        scene = Scene()
        geo = BoxGeometry()
        mat = MeshBasicMaterial(color=Color3(1.0, 0.0, 0.0))
        mesh = Mesh(geo, mat; name="RedCube")
        add!(scene, mesh)

        @test length(get_children(scene)) == 1
        @test get_parent(mesh) === scene
        @test mesh.name == "RedCube"

        meshes = collect_meshes(scene)
        @test length(meshes) == 1
        @test meshes[1] === mesh
    end

    @testset "Camera projection" begin
        cam = PerspectiveCamera(fov=π/4, aspect=1.0, near=0.1, far=100.0)
        P = projection_matrix(cam)
        V = view_matrix(cam)
        @test mat4_get(P, 1, 1) != 0.0
        @test mat4_get(V, 1, 1) != 0.0
    end

    @testset "Materials" begin
        m1 = MeshBasicMaterial(color=Color3(1.0, 0.0, 0.0))
        @test m1.color.r ≈ 1.0

        m2 = MeshPhongMaterial(shininess=64.0)
        @test m2.shininess ≈ 64.0

        m3 = MeshStandardMaterial(metalness=0.8, roughness=0.2)
        @test m3.metalness ≈ 0.8
        @test m3.roughness ≈ 0.2
    end

    @testset "Shading — Lambert" begin
        n = Vec3(0.0, 0.0, 1.0)
        l = Vec3(0.0, 0.0, 1.0)  # light from front
        lc = Color3(1.0, 1.0, 1.0)
        sc = Color3(0.5, 0.5, 0.5)
        c = shade_lambert(n, l, lc, 1.0, sc)
        @test c.r ≈ 0.5  # full illumination

        # Light from behind
        lb = Vec3(0.0, 0.0, -1.0)
        cb = shade_lambert(n, lb, lc, 1.0, sc)
        @test cb.r ≈ 0.0  # no illumination
    end

    @testset "Shading — Phong" begin
        n = Vec3(0.0, 0.0, 1.0)
        l = Vec3(0.0, 0.0, 1.0)
        v = Vec3(0.0, 0.0, 1.0)
        lc = Color3(1.0, 1.0, 1.0)
        dc = Color3(0.5, 0.5, 0.5)
        sc = Color3(1.0, 1.0, 1.0)
        c = shade_phong(n, l, v, lc, 1.0, dc, sc, 30.0)
        @test c.r > 0.5  # diffuse + specular > diffuse alone
    end

    @testset "RenderTarget" begin
        rt = RenderTarget(64, 48)
        @test rt.width == 64
        @test rt.height == 48
        @test size(rt.color) == (48, 64, 3)
        @test all(rt.color .== 0.0)

        clear!(rt, Color3(0.2, 0.3, 0.4))
        @test rt.color[1, 1, 1] ≈ 0.2
        @test rt.color[1, 1, 2] ≈ 0.3
        @test rt.color[1, 1, 3] ≈ 0.4
    end

    @testset "CPU rasterizer — basic render" begin
        scene = Scene(background=Color3(0.1, 0.1, 0.1))
        geo = BoxGeometry(width=1.0, height=1.0, depth=1.0)
        mat = MeshPhongMaterial(color=Color3(0.8, 0.2, 0.2))
        mesh = Mesh(geo, mat)
        add!(scene, mesh)

        light = DirectionalLight(color=Color3(1.0, 1.0, 1.0), intensity=1.0,
                                 position=Vec3(5.0, 5.0, 5.0))
        add!(scene, light)

        ambient = AmbientLight(intensity=0.3)
        add!(scene, ambient)

        cam = PerspectiveCamera(fov=π/4, aspect=1.0, near=0.1, far=100.0)
        cam.position = Vec3(0.0, 0.0, 3.0)

        rt = RenderTarget(32, 32)
        render!(rt, scene, cam)

        # Center pixel should not be background
        cy, cx = 16, 16
        pixel_r = rt.color[cy, cx, 1]
        @test pixel_r > 0.1  # something rendered, not just background

        # Corner pixel should be background (cube is small relative to viewport)
        corner_r = rt.color[1, 1, 1]
        @test corner_r ≈ 0.1 atol=0.05
    end

    @testset "Edge function" begin
        # Triangle (0,0), (10,0), (0,10) — CW winding gives negative area
        ef = edge_function(0.0, 0.0, 10.0, 0.0, 0.0, 10.0)
        @test abs(ef) ≈ 100.0  # |signed area * 2|

        # Barycentric coords for point (3,3) inside triangle
        w0 = edge_function(10.0, 0.0, 0.0, 10.0, 3.0, 3.0) / ef
        w1 = edge_function(0.0, 10.0, 0.0, 0.0, 3.0, 3.0) / ef
        w2 = edge_function(0.0, 0.0, 10.0, 0.0, 3.0, 3.0) / ef
        @test w0 + w1 + w2 ≈ 1.0 atol=1e-10
    end

    @testset "Soft rasterizer — signed distance" begin
        # Point at center of triangle (0,0)-(10,0)-(0,10) should be positive
        d = signed_distance_to_triangle(2.0, 2.0, 0.0, 0.0, 10.0, 0.0, 0.0, 10.0)
        @test d > 0.0

        # Point far outside should be negative
        d_out = signed_distance_to_triangle(-5.0, -5.0, 0.0, 0.0, 10.0, 0.0, 0.0, 10.0)
        @test d_out < 0.0
    end

    @testset "Soft rasterizer — basic render" begin
        verts = [Vec3(-0.5, -0.5, 0.0), Vec3(0.5, -0.5, 0.0),
                 Vec3(0.5, 0.5, 0.0), Vec3(-0.5, 0.5, 0.0)]
        faces = [(1, 2, 3), (1, 3, 4)]
        colors = [Color3(1.0, 0.0, 0.0), Color3(0.0, 1.0, 0.0)]

        vp = mat4_perspective(π/4, 1.0, 0.1, 100.0) *
             mat4_look_at(Vec3(0.0,0.0,3.0), Vec3(0.0,0.0,0.0), Vec3(0.0,1.0,0.0))
        config = SoftRasterizerConfig(sigma=1e-2, gamma=1.0)

        img = soft_render(verts, faces, colors, vp, 16, 16, config)
        @test size(img) == (16, 16, 3)
        # Center pixel should have some color (not just black background)
        center_brightness = img[8, 8, 1] + img[8, 8, 2] + img[8, 8, 3]
        @test center_brightness > 0.01
    end

    @testset "Loss functions" begin
        img1 = rand(8, 8, 3)
        img2 = copy(img1)

        @test loss_mse(img1, img2) ≈ 0.0 atol=1e-15
        @test loss_l1(img1, img2) ≈ 0.0 atol=1e-15

        # Different images should have positive loss
        img3 = rand(8, 8, 3)
        @test loss_mse(img1, img3) > 0.0
        @test loss_l1(img1, img3) > 0.0
    end

    @testset "ForwardDiff gradient through soft render" begin
        # Simple test: move a triangle along x, compute gradient of center pixel
        function _test_render_offset(x_offset)
            T = eltype(x_offset)
            verts = [Vec3(T(-0.5) + x_offset[1], T(-0.5), zero(T)),
                     Vec3(T(0.5) + x_offset[1], T(-0.5), zero(T)),
                     Vec3(zero(T) + x_offset[1], T(0.5), zero(T))]
            faces = [(1, 2, 3)]
            colors = [Color3(one(T), zero(T), zero(T))]
            vp = mat4_perspective(T(π/4), one(T), T(0.1), T(100)) *
                 mat4_look_at(Vec3(zero(T), zero(T), T(3)),
                              Vec3(zero(T), zero(T), zero(T)),
                              Vec3(zero(T), one(T), zero(T)))
            config = SoftRasterizerConfig(sigma=T(0.5), gamma=T(1.0),
                                          bg_color=Color3(zero(T), zero(T), zero(T)),
                                          eps=T(1e-8))
            img = soft_render(verts, faces, colors, vp, 8, 8, config)
            img[4, 4, 1]
        end

        grad = ForwardDiff.gradient(_test_render_offset, [0.0])
        @test length(grad) == 1
        @test isfinite(grad[1])

        # Numerical gradient for comparison
        f0 = _test_render_offset([0.0])
        f1 = _test_render_offset([0.001])
        fd_grad = (f1 - f0) / 0.001
        if abs(fd_grad) > 1e-10
            @test abs(grad[1] - fd_grad) / max(abs(fd_grad), 1e-8) < 1.0
        end
    end

    @testset "Numerical gradient utility" begin
        f(x) = sum(x .^ 2)
        g = numerical_gradient(f, [1.0, 2.0, 3.0])
        @test g[1] ≈ 2.0 atol=1e-4
        @test g[2] ≈ 4.0 atol=1e-4
        @test g[3] ≈ 6.0 atol=1e-4
    end

    @testset "I/O — PPM export" begin
        img = test_pattern(16, 16)
        @test size(img) == (16, 16, 3)
        # Just verify it doesn't error; we don't write to disk in tests
        tmpfile = tempname() * ".ppm"
        save_ppm(tmpfile, img)
        @test isfile(tmpfile)
        rm(tmpfile)
    end

    @testset "Merge geometries" begin
        g1 = BoxGeometry()
        g2 = SphereGeometry(width_segments=8, height_segments=4)
        merged = merge_geometries([g1, g2])
        @test merged.n_vertices == g1.n_vertices + g2.n_vertices
        @test merged.n_faces == g1.n_faces + g2.n_faces
    end

    @testset "PNG/PDF checksum oracles" begin
        # CRC-32/ISO-HDLC standard check value for "123456789".
        @test Three._crc32(Vector{UInt8}(codeunits("123456789"))) == 0xcbf43926
        # CRC of the IEND chunk type (no data) — fixed PNG constant.
        @test Three._crc32(Vector{UInt8}(codeunits("IEND"))) == 0xae426082
        # Adler-32 of "Wikipedia" — canonical reference value.
        @test Three._adler32(Vector{UInt8}(codeunits("Wikipedia"))) == 0x11e60398
    end

    @testset "I/O — PNG export structure" begin
        img = test_pattern(20, 12)   # width=20, height=12
        f = tempname() * ".png"
        save_png(f, img)
        @test isfile(f)
        bytes = read(f)
        @test bytes[1:8] == UInt8[137, 80, 78, 71, 13, 10, 26, 10]   # PNG signature
        # IHDR width/height (big-endian) at byte offset 16 (sig 8 + len 4 + type 4).
        w = (Int(bytes[17]) << 24) | (Int(bytes[18]) << 16) | (Int(bytes[19]) << 8) | Int(bytes[20])
        h = (Int(bytes[21]) << 24) | (Int(bytes[22]) << 16) | (Int(bytes[23]) << 8) | Int(bytes[24])
        @test w == 20 && h == 12
        @test bytes[end-7:end] == UInt8[0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82]  # IEND + CRC
        rm(f)
    end

    @testset "I/O — PDF export structure" begin
        img = test_pattern(16, 16)
        f = tempname() * ".pdf"
        save_pdf(f, img)
        @test isfile(f)
        bytes = read(f)
        @test String(bytes[1:8]) == "%PDF-1.4"
        tail = String(bytes[max(1, end-16):end])
        @test occursin("%%EOF", tail)
        rm(f)
    end

    @testset "Ambient is uniform fill, not directional (regression)" begin
        mat = MeshLambertMaterial(color=Color3(0.6, 0.6, 0.6))
        amb = AbstractLight[AmbientLight(intensity=0.5)]
        up   = shade_face(Vec3(0.0,  1.0, 0.0), Vec3(0.0, 0.0, 1.0), Vec3(), mat, amb)
        down = shade_face(Vec3(0.0, -1.0, 0.0), Vec3(0.0, 0.0, 1.0), Vec3(), mat, amb)
        @test up.r ≈ down.r              # ambient independent of surface orientation
        @test down.r > 0.0               # downward face not black under ambient
        @test up.r ≈ 0.6 * 0.5           # albedo × intensity
    end

    @testset "PBR metal not black under ambient (regression)" begin
        n = Vec3(0.0, 1.0, 0.0); v = Vec3(0.0, 0.0, 1.0)
        metal = MeshStandardMaterial(color=Color3(0.95, 0.78, 0.22),
                                     metalness=0.9, roughness=0.25)
        c = shade_face(n, v, Vec3(), metal, AbstractLight[AmbientLight(intensity=0.5)])
        @test (c.r + c.g + c.b) > 0.1    # metal reflects ambient, not black
        @test c.r > c.b                  # gold albedo preserved (red > blue)
    end

    @testset "Near-plane clipping — straddling geometry renders" begin
        # Large ground plane with corners behind the camera must render its
        # in-front portion (would vanish without near-plane clipping).
        scene = Scene(background=Color3(0.0, 0.0, 0.0))
        ground = Mesh(PlaneGeometry(width=50.0, height=50.0),
                      MeshLambertMaterial(color=Color3(0.85, 0.85, 0.85)))
        ground.rotation = Euler(-π/2, 0.0, 0.0)
        add!(scene, ground)
        add!(scene, AmbientLight(intensity=1.0))
        cam = PerspectiveCamera(fov=π/4, aspect=1.0, near=0.1, far=100.0)
        cam.position = Vec3(0.0, 2.0, 0.0)
        cam.target = Vec3(0.0, 0.0, -5.0)
        rt = RenderTarget(32, 32)
        render!(rt, scene, cam)
        lit = count(>(0.1), @view rt.color[:, :, 1])
        @test lit > 50                   # floor visible across many pixels
    end

    @testset "Smooth shading — continuity and correctness" begin
        scene = Scene(background=Color3(0.0, 0.0, 0.0))
        sph = Mesh(SphereGeometry(radius=1.0, width_segments=24, height_segments=12),
                   MeshLambertMaterial(color=Color3(0.8, 0.3, 0.3)))
        add!(scene, sph)
        add!(scene, AmbientLight(intensity=0.2))
        d = DirectionalLight(intensity=0.9, position=Vec3(3.0, 3.0, 4.0))
        d.target = Vec3(0.0, 0.0, 0.0)
        add!(scene, d)
        cam = PerspectiveCamera(fov=π/4, aspect=1.0, near=0.1, far=100.0)
        cam.position = Vec3(0.0, 0.0, 4.0)

        rtf = RenderTarget(100, 100); render!(rtf, scene, cam; shading=:flat)
        rts = RenderTarget(100, 100); render!(rts, scene, cam; shading=:smooth)

        # Same silhouette: both light a comparable number of pixels.
        litf = count(>(0.05), @view rtf.color[:, :, 1])
        lits = count(>(0.05), @view rts.color[:, :, 1])
        @test litf > 1000 && lits > 1000
        @test abs(lits - litf) < 0.1 * litf

        # Smooth shading yields a continuous gradient: many more distinct shades.
        uniq(rt) = length(unique(round.(vec(rt.color[:, :, 1]), digits=3)))
        @test uniq(rts) > 5 * uniq(rtf)

        # Invalid shading mode is rejected.
        @test_throws ArgumentError render!(RenderTarget(8, 8), scene, cam; shading=:bogus)
    end

    @testset "Smooth shading — constant normals match analytic Lambert" begin
        # A front-facing quad with all-front normals, lit head-on: smooth-shaded
        # pixels must equal the Lambert value albedo·intensity (N·L = 1).
        scene = Scene(background=Color3(0.0, 0.0, 0.0))
        quad = Mesh(PlaneGeometry(width=2.0, height=2.0),
                    MeshLambertMaterial(color=Color3(0.6, 0.4, 0.2)))
        add!(scene, quad)
        light = DirectionalLight(intensity=1.0, position=Vec3(0.0, 0.0, 5.0))
        light.target = Vec3(0.0, 0.0, 0.0)
        add!(scene, light)
        cam = PerspectiveCamera(fov=π/4, aspect=1.0, near=0.1, far=100.0)
        cam.position = Vec3(0.0, 0.0, 3.0)
        rt = RenderTarget(32, 32); render!(rt, scene, cam; shading=:smooth)
        # Center pixel: N=(0,0,1), L=(0,0,1) ⇒ N·L=1 ⇒ colour = albedo.
        @test rt.color[16, 16, 1] ≈ 0.6 atol=1e-6
        @test rt.color[16, 16, 2] ≈ 0.4 atol=1e-6
        @test rt.color[16, 16, 3] ≈ 0.2 atol=1e-6
    end

    @testset "Near-plane clipping — geometry fully behind camera is dropped" begin
        scene = Scene(background=Color3(0.0, 0.0, 0.0))
        box = Mesh(BoxGeometry(width=1.0, height=1.0, depth=1.0),
                   MeshBasicMaterial(color=Color3(1.0, 1.0, 1.0)))
        box.position = Vec3(0.0, 0.0, 10.0)   # behind a camera at z=3 looking -z
        add!(scene, box)
        cam = PerspectiveCamera(fov=π/4, aspect=1.0, near=0.1, far=100.0)
        cam.position = Vec3(0.0, 0.0, 3.0)
        rt = RenderTarget(32, 32)
        render!(rt, scene, cam)
        @test all(<(0.01), rt.color)     # nothing rendered, all background
    end

    @testset "No NaN/Inf in geometry data" begin
        for (name, geo) in [
            ("Box", BoxGeometry()),
            ("Sphere", SphereGeometry()),
            ("Plane", PlaneGeometry()),
            ("Cylinder", CylinderGeometry()),
            ("Cone", ConeGeometry()),
            ("Torus", TorusGeometry()),
            ("TorusKnot", TorusKnotGeometry()),
            ("Ring", RingGeometry()),
            ("Circle", CircleGeometry()),
            ("Icosahedron", IcosahedronGeometry()),
        ]
            @testset "$name" begin
                @test all(isfinite, geo.positions)
                @test all(isfinite, geo.normals)
                @test all(i -> 1 <= i <= geo.n_vertices, geo.indices)
            end
        end
    end

    @testset "compute_vertex_normals! — sphere" begin
        ws, hs = 16, 8
        geo = SphereGeometry(radius=1.0, width_segments=ws, height_segments=hs)
        compute_vertex_normals!(geo)
        # All smooth normals are unit length.
        for vi in 1:geo.n_vertices
            @test norm(get_normal(geo, vi)) ≈ 1.0 atol=1e-10
        end
        # Interior (non-pole, non-seam) vertices recover the analytic outward
        # normal (= vertex direction on a unit sphere). Pole and φ-seam vertices
        # are topologically degenerate (incomplete 1-ring) and excluded here.
        for (j, i) in ((4, 4), (4, 8), (3, 6), (5, 10))
            vi = j * (ws + 1) + i + 1
            v = get_vertex(geo, vi); n = get_normal(geo, vi)
            @test n.x ≈ v.x atol=0.02
            @test n.y ≈ v.y atol=0.02
            @test n.z ≈ v.z atol=0.02
        end
        # Mesh-wide outward orientation: a sign flip would drive this strongly
        # negative. Degenerate pole/seam vertices are tolerated via the mean.
        meandot = sum(dot(get_normal(geo, vi), normalize(get_vertex(geo, vi)))
                      for vi in 1:geo.n_vertices) / geo.n_vertices
        @test meandot > 0.95
    end

    @testset "STL binary round-trip" begin
        geo = BoxGeometry(width=1.0, height=2.0, depth=3.0)
        f = tempname() * ".stl"
        save_stl_binary(f, geo)
        @test isfile(f)
        @test filesize(f) == 84 + 50 * geo.n_faces   # exact binary STL size
        loaded = load_stl(f)
        @test loaded.n_faces == geo.n_faces
        @test loaded.n_vertices == geo.n_faces * 3    # STL: independent verts
        # Triangle 1 vertices match (within Float32 precision).
        a1, b1, c1 = get_face(geo, 1)
        la, lb, lc = get_face(loaded, 1)
        for (orig, ld) in ((a1, la), (b1, lb), (c1, lc))
            vo = get_vertex(geo, orig); vl = get_vertex(loaded, ld)
            @test vl.x ≈ vo.x atol=1e-6
            @test vl.y ≈ vo.y atol=1e-6
            @test vl.z ≈ vo.z atol=1e-6
        end
        rm(f)
    end

    @testset "STL ASCII parse" begin
        ascii = """
        solid tri
        facet normal 0 0 1
        outer loop
        vertex 0 0 0
        vertex 1 0 0
        vertex 0 1 0
        endloop
        endfacet
        endsolid tri
        """
        f = tempname() * ".stl"
        write(f, ascii)
        @test !Three._looks_binary_stl(f)            # detected as ASCII
        geo = load_stl(f)
        @test geo.n_faces == 1
        @test geo.n_vertices == 3
        @test get_vertex(geo, 2).x ≈ 1.0
        @test get_normal(geo, 1).z ≈ 1.0
        rm(f)
    end

    @testset "OBJ parse + fan triangulation" begin
        obj = """
        # quad
        v 0 0 0
        v 1 0 0
        v 1 1 0
        v 0 1 0
        f 1 2 3 4
        """
        f = tempname() * ".obj"
        write(f, obj)
        geo = load_obj(f)
        @test geo.n_faces == 2                        # quad fan → 2 triangles
        @test geo.n_vertices == 6
        @test get_vertex(geo, 1).x ≈ 0.0 && get_vertex(geo, 1).y ≈ 0.0
        @test get_vertex(geo, 2).x ≈ 1.0
        # Computed normals point +z for a planar quad in the z=0 plane.
        @test get_normal(geo, 1).z ≈ 1.0 atol=1e-10
        rm(f)
    end

    @testset "OBJ ↔ STL cross round-trip preserves triangle count" begin
        geo = IcosahedronGeometry(radius=1.0)
        f = tempname() * ".stl"
        save_stl_binary(f, geo)
        loaded = load_stl(f)
        @test loaded.n_faces == geo.n_faces == 20
        @test all(isfinite, loaded.positions)
        rm(f)
    end

    @testset "Euler — all six orders match axis-rotation products" begin
        x, y, z = 0.3, -0.5, 0.7
        Rx = mat4_rotation_x(x); Ry = mat4_rotation_y(y); Rz = mat4_rotation_z(z)
        # three.js intrinsic order L1L2L3 ⇒ rotation matrix R(L1)·R(L2)·R(L3).
        expected = Dict(:XYZ => Rx*Ry*Rz, :YXZ => Ry*Rx*Rz, :ZXY => Rz*Rx*Ry,
                        :ZYX => Rz*Ry*Rx, :YZX => Ry*Rz*Rx, :XZY => Rx*Rz*Ry)
        for (ord, M) in expected
            Q = quat_to_mat4(quat_from_euler(x, y, z; order=ord))
            for i in 1:4, j in 1:4
                @test mat4_get(Q, i, j) ≈ mat4_get(M, i, j) atol=1e-12
            end
        end
        # Single-axis rotation is order-independent.
        qa = quat_from_euler(0.4, 0.0, 0.0; order=:ZYX)
        qb = quat_from_euler(0.4, 0.0, 0.0; order=:XYZ)
        @test qa.x ≈ qb.x && qa.w ≈ qb.w
        @test_throws ArgumentError quat_from_euler(0.1, 0.2, 0.3; order=:ABC)
    end

    @testset "Quaternion slerp" begin
        qa = quat_from_euler(0.0, 0.0, 0.0)
        qb = quat_from_euler(0.0, 0.0, π/2)
        @test quat_slerp(qa, qb, 0.0).w ≈ qa.w
        @test quat_slerp(qa, qb, 1.0).z ≈ qb.z atol=1e-12
        # Halfway between identity and Rz(π/2) is Rz(π/4): z = sin(π/8).
        @test quat_slerp(qa, qb, 0.5).z ≈ sin(π/8) atol=1e-12
        # Result is a unit quaternion.
        q = quat_slerp(qa, qb, 0.37)
        @test q.x^2 + q.y^2 + q.z^2 + q.w^2 ≈ 1.0 atol=1e-12
        # Shorter-arc: slerp with the negated target gives the same rotation.
        qbn = Quaternion(-qb.x, -qb.y, -qb.z, -qb.w)
        qm1 = quat_slerp(qa, qb, 0.5); qm2 = quat_slerp(qa, qbn, 0.5)
        @test qm1.z ≈ qm2.z atol=1e-12
    end

    @testset "Quaternion setFromUnitVectors" begin
        q = quat_from_unit_vectors(Vec3(1.0,0,0), Vec3(0.0,1.0,0))
        r = mat4_transform_point(quat_to_mat4(q), Vec3(1.0,0,0))
        @test r.x ≈ 0.0 atol=1e-12
        @test r.y ≈ 1.0 atol=1e-12
        # Antiparallel case: x → -x.
        qo = quat_from_unit_vectors(Vec3(1.0,0,0), Vec3(-1.0,0,0))
        ro = mat4_transform_point(quat_to_mat4(qo), Vec3(1.0,0,0))
        @test ro.x ≈ -1.0 atol=1e-10
        # Arbitrary pair maps from → to.
        from = normalize(Vec3(0.2, 0.9, -0.3)); to = normalize(Vec3(-0.5, 0.1, 0.8))
        qa = quat_from_unit_vectors(from, to)
        ra = mat4_transform_point(quat_to_mat4(qa), from)
        @test ra.x ≈ to.x atol=1e-10
        @test ra.y ≈ to.y atol=1e-10
        @test ra.z ≈ to.z atol=1e-10
    end

    @testset "Triangle" begin
        tri = Triangle(Vec3(0.0,0,0), Vec3(1.0,0,0), Vec3(0.0,1.0,0))
        @test triangle_area(tri) ≈ 0.5
        @test triangle_normal(tri).z ≈ 1.0
        @test triangle_centroid(tri).x ≈ 1/3
        bc = triangle_barycentric(tri, Vec3(0.25, 0.25, 0.0))
        @test bc.x ≈ 0.5 && bc.y ≈ 0.25 && bc.z ≈ 0.25
        @test triangle_contains_point(tri, Vec3(0.25, 0.25, 0))
        @test !triangle_contains_point(tri, Vec3(2.0, 2.0, 0))
    end

    @testset "Line3" begin
        l = Line3(Vec3(0.0,0,0), Vec3(10.0,0,0))
        @test line3_length(l) ≈ 10.0
        @test line3_center(l).x ≈ 5.0
        @test line3_at(l, 0.3).x ≈ 3.0
        @test line3_closest_point(l, Vec3(3.0, 5.0, 0.0)).x ≈ 3.0
        @test line3_closest_point(l, Vec3(-4.0, 0, 0)).x ≈ 0.0          # clamped to start
        @test line3_closest_point(l, Vec3(-4.0, 0, 0); clamp_to_segment=false).x ≈ -4.0
    end

    @testset "Spherical / Cylindrical round-trip" begin
        for v in (Vec3(0.3,-0.5,0.7), Vec3(1.0,0,0), Vec3(0.0,2.0,0))
            s = cartesian_to_spherical(v); vs = spherical_to_cartesian(s)
            @test vs.x ≈ v.x atol=1e-12
            @test vs.y ≈ v.y atol=1e-12
            @test vs.z ≈ v.z atol=1e-12
            c = cartesian_to_cylindrical(v); vc = cylindrical_to_cartesian(c)
            @test vc.x ≈ v.x atol=1e-12
            @test vc.y ≈ v.y atol=1e-12
            @test vc.z ≈ v.z atol=1e-12
        end
        # Known value: +Y axis ⇒ phi = 0.
        @test cartesian_to_spherical(Vec3(0.0, 3.0, 0.0)).phi ≈ 0.0
    end

    @testset "Interpolant — linear keyframes" begin
        t = [0.0, 1.0, 3.0]; vals = [0.0, 10.0, 30.0]
        @test interpolate_linear(t, vals, -1.0) ≈ 0.0       # clamp low
        @test interpolate_linear(t, vals, 5.0) ≈ 30.0       # clamp high
        @test interpolate_linear(t, vals, 1.0) ≈ 10.0       # exact knot
        @test interpolate_linear(t, vals, 2.0) ≈ 20.0       # within [1,3]
        vv = interpolate_linear([0.0,1.0], [Vec3(0.,0,0), Vec3(2.,4,6)], 0.25)
        @test vv.x ≈ 0.5 && vv.y ≈ 1.0 && vv.z ≈ 1.5
    end

    @testset "Frustum culling" begin
        cam = PerspectiveCamera(fov=π/4, aspect=1.0, near=0.1, far=100.0)
        cam.position = Vec3(0.0, 0.0, 5.0); cam.target = Vec3(0.0, 0.0, 0.0)
        f = frustum_from_matrix(projection_matrix(cam) * view_matrix(cam))
        @test frustum_contains_point(f, Vec3(0.0, 0.0, 0.0))      # origin in view
        @test !frustum_contains_point(f, Vec3(0.0, 0.0, 10.0))    # behind camera (near)
        @test !frustum_contains_point(f, Vec3(0.0, 0.0, -200.0))  # beyond far
        @test !frustum_contains_point(f, Vec3(100.0, 0.0, 0.0))   # outside side plane
        @test frustum_intersects_sphere(f, BoundingSphere(Vec3(0.0,0,0), 0.5))
        @test !frustum_intersects_sphere(f, BoundingSphere(Vec3(0.0,0,1000.0), 0.5))
        @test frustum_intersects_box(f, Box3(Vec3(-0.5,-0.5,-0.5), Vec3(0.5,0.5,0.5)))
        @test !frustum_intersects_box(f, Box3(Vec3(50.0,50,50), Vec3(60.0,60,60)))
    end

    @testset "Layers bitmask" begin
        a = Layers(); b = Layers()
        @test layers_test(a, b)                       # both on channel 0
        layers_set!(b, 2)
        @test !layers_test(a, b)                      # disjoint channels
        layers_enable!(b, 0)
        @test layers_test(a, b)                       # channel 0 re-shared
        layers_disable!(b, 0)
        @test !layers_test(a, b)
        layers_enable_all!(b)
        @test layers_test(a, b)
        layers_disable_all!(b)
        @test !layers_test(a, b)
        layers_toggle!(b, 0)
        @test layers_test(a, b)
    end

    @testset "Named attributes + bounding volumes" begin
        g = BoxGeometry(width=2.0, height=4.0, depth=6.0)
        @test !has_attribute(g, :color)
        set_attribute!(g, :color, [1.0,0,0, 0,1.0,0], 3)
        @test has_attribute(g, :color)
        @test get_attribute(g, :color).item_size == 3
        bb = compute_bounding_box(g)
        @test bb.min.x ≈ -1.0 && bb.max.y ≈ 2.0 && bb.max.z ≈ 3.0
        bs = compute_bounding_sphere(g)
        @test bs.radius ≈ sqrt(1.0 + 4.0 + 9.0)       # half-diagonal of the box
        @test bs.center.x ≈ 0.0 && bs.center.y ≈ 0.0
    end

    @testset "World-matrix cache equals direct computation" begin
        scene = Scene()
        g = Group(); g.position = Vec3(1.0, 2.0, 3.0)
        m = Mesh(BoxGeometry(), MeshBasicMaterial()); m.position = Vec3(0.5, 0.0, -1.0)
        add!(scene, g); add!(g, m)
        cache = compute_world_matrices(scene)
        for obj in (scene, g, m)
            wc = cache[obj]; wd = compute_world_matrix(obj)
            for i in 1:4, j in 1:4
                @test mat4_get(wc, i, j) ≈ mat4_get(wd, i, j) atol=1e-12
            end
        end
    end

    @testset "LOD level selection" begin
        lod = LOD()
        near = Mesh(BoxGeometry(), MeshBasicMaterial())
        far  = Mesh(SphereGeometry(), MeshBasicMaterial())
        add_lod_level!(lod, 0.0, near)
        add_lod_level!(lod, 10.0, far)
        @test lod_select(lod, 5.0) === near
        @test lod_select(lod, 10.0) === far
        @test lod_select(lod, 100.0) === far
        @test lod_select(lod, 0.0) === near
    end

    @testset "Raycaster — ray/triangle and scene picking" begin
        # Single triangle hit / miss.
        @test ray_triangle_intersect(Vec3(0.25,0.25,1.0), Vec3(0.0,0,-1.0),
              Vec3(0.0,0,0), Vec3(1.0,0,0), Vec3(0.0,1.0,0)) ≈ 1.0
        @test ray_triangle_intersect(Vec3(2.0,2.0,1.0), Vec3(0.0,0,-1.0),
              Vec3(0.0,0,0), Vec3(1.0,0,0), Vec3(0.0,1.0,0)) === nothing
        # Scene picking: ray hits a unit box, front face at z=0.5 ⇒ distance 4.5.
        scene = Scene()
        box = Mesh(BoxGeometry(width=1.0, height=1.0, depth=1.0), MeshBasicMaterial())
        add!(scene, box)
        rc = Raycaster(Vec3(0.2, 0.2, 5.0), Vec3(0.0, 0.0, -1.0))
        hits = raycast(rc, scene)
        @test !isempty(hits)
        @test hits[1].distance ≈ 4.5 atol=1e-9
        @test hits[1].object === box
        @test issorted([h.distance for h in hits])     # sorted nearest-first
        # Off-axis ray misses entirely.
        @test isempty(raycast(Raycaster(Vec3(5.0,0,5), Vec3(0.0,0,-1.0)), scene))
        # set_from_camera through screen centre hits the box.
        cam = PerspectiveCamera(fov=π/4, aspect=1.0, near=0.1, far=100.0)
        cam.position = Vec3(0.0,0,5); cam.target = Vec3(0.0,0,0)
        set_from_camera!(rc, cam, 0.0, 0.0)
        @test !isempty(raycast(rc, scene))
    end

    @testset "Sprite billboard faces camera" begin
        cam = PerspectiveCamera(fov=π/4, aspect=1.0, near=0.1, far=100.0)
        cam.position = Vec3(0.0, 0.0, 5.0); cam.target = Vec3(0.0, 0.0, 0.0)
        sp = Sprite(MeshBasicMaterial()); sp.position = Vec3(0.0, 0.0, 0.0)
        M = sprite_world_matrix(sp, cam)
        # Sprite local +z maps to the camera forward axis (world +z here).
        f = mat4_transform_direction(M, Vec3(0.0,0,1.0))
        @test f.z ≈ 1.0 atol=1e-12
        @test f.x ≈ 0.0 atol=1e-12
        @test f.y ≈ 0.0 atol=1e-12
        # Translation column is the sprite world position.
        @test mat4_get(M, 1, 4) ≈ 0.0 && mat4_get(M, 3, 4) ≈ 0.0
    end

    @testset "SkinnedMesh — linear blend skinning" begin
        bone = Bone()                                   # bind pose: identity at origin
        skel = Skeleton([bone])
        geo = PlaneGeometry(width=2.0, height=2.0)      # 4 vertices
        idx = fill((1,1,1,1), geo.n_vertices)
        wts = fill((1.0,0.0,0.0,0.0), geo.n_vertices)
        sm = SkinnedMesh(geo, MeshBasicMaterial(), skel, idx, wts)
        # Bind pose ⇒ vertices unchanged.
        p0 = apply_skinning(sm)
        for vi in 1:geo.n_vertices
            v = get_vertex(geo, vi)
            @test p0[vi].x ≈ v.x && p0[vi].y ≈ v.y && p0[vi].z ≈ v.z
        end
        # Translate the bone ⇒ all (fully-weighted) vertices translate equally.
        bone.position = Vec3(3.0, -1.0, 2.0)
        p1 = apply_skinning(sm)
        for vi in 1:geo.n_vertices
            v = get_vertex(geo, vi)
            @test p1[vi].x ≈ v.x + 3.0 atol=1e-10
            @test p1[vi].y ≈ v.y - 1.0 atol=1e-10
            @test p1[vi].z ≈ v.z + 2.0 atol=1e-10
        end
    end

    @testset "InstancedMesh renders each instance" begin
        scene = Scene(background=Color3(0.0,0,0))
        im = InstancedMesh(BoxGeometry(width=1.0,height=1.0,depth=1.0),
                           MeshBasicMaterial(color=Color3(1.0,1.0,1.0)), 2)
        @test instanced_count(im) == 2
        set_instance_matrix!(im, 1, mat4_translation(-2.0, 0.0, 0.0))
        set_instance_matrix!(im, 2, mat4_translation( 2.0, 0.0, 0.0))
        add!(scene, im)
        cam = PerspectiveCamera(fov=π/4, aspect=1.0, near=0.1, far=100.0)
        cam.position = Vec3(0.0, 0.0, 8.0)
        rt = RenderTarget(64, 64)
        render!(rt, scene, cam)
        # Two separated boxes ⇒ lit pixels in both the left and right halves,
        # with a dark gap straddling the centre column.
        left  = count(>(0.5), @view rt.color[:, 1:24, 1])
        right = count(>(0.5), @view rt.color[:, 41:64, 1])
        center = count(>(0.5), @view rt.color[:, 30:35, 1])
        @test left > 20 && right > 20
        @test center == 0
    end

    @testset "StereoCamera eye offset" begin
        cam = PerspectiveCamera(fov=π/4, aspect=1.5, near=0.1, far=50.0)
        cam.position = Vec3(0.0, 0.0, 5.0); cam.target = Vec3(0.0, 0.0, 0.0)
        st = StereoCamera(eye_sep=0.2)
        stereo_update!(st, cam)
        # Camera looks down -z with up +y ⇒ world right is +x; eyes split on x.
        @test st.cameraL.position.x ≈ -0.1 atol=1e-12
        @test st.cameraR.position.x ≈  0.1 atol=1e-12
        @test st.cameraL.aspect ≈ 1.5 && st.cameraR.far ≈ 50.0
        @test (st.cameraR.position.x - st.cameraL.position.x) ≈ 0.2 atol=1e-12
    end

    @testset "CubeCamera six faces" begin
        cc = CubeCamera(near=0.5, far=200.0, position=Vec3(1.0, 2.0, 3.0))
        @test length(cc.cameras) == 6
        for c in cc.cameras
            @test c.fov ≈ π/2
            @test c.near ≈ 0.5 && c.far ≈ 200.0
            @test c.position.x ≈ 1.0 && c.position.y ≈ 2.0 && c.position.z ≈ 3.0
        end
        # Targets point along ±x, ±y, ±z relative to the camera position.
        dirs = [Vec3(c.target.x - 1.0, c.target.y - 2.0, c.target.z - 3.0) for c in cc.cameras]
        @test dirs[1].x ≈ 1.0 && dirs[2].x ≈ -1.0
        @test dirs[3].y ≈ 1.0 && dirs[4].y ≈ -1.0
        @test dirs[5].z ≈ 1.0 && dirs[6].z ≈ -1.0
    end

    @testset "ArrayCamera holds sub-cameras" begin
        cams = [PerspectiveCamera(), PerspectiveCamera()]
        ac = ArrayCamera(cams)
        @test length(ac.cameras) == 2
        @test length(ac.viewports) == 2
        ac2 = ArrayCamera(cams, [(0,0,400,300), (400,0,400,300)])
        @test ac2.viewports[2] == (400,0,400,300)
    end

    @testset "Polyhedra (+ subdivision)" begin
        for (geo, nf, r) in ((OctahedronGeometry(radius=2.0), 8, 2.0),
                             (TetrahedronGeometry(radius=1.5), 4, 1.5),
                             (DodecahedronGeometry(radius=1.0), 36, 1.0))
            @test geo.n_faces == nf
            @test all(isfinite, geo.positions)
            for vi in 1:geo.n_vertices
                @test norm(get_vertex(geo, vi)) ≈ r atol=1e-9   # projected to sphere
            end
            @test all(i -> 1 <= i <= geo.n_vertices, geo.indices)
        end
        # Subdivision: detail=d multiplies faces by (d+1)².
        @test OctahedronGeometry(detail=1).n_faces == 8 * 4
        @test TetrahedronGeometry(detail=2).n_faces == 4 * 9
        # Octahedron/Tetrahedron geometric winding agrees with stored outward normals.
        for geo in (OctahedronGeometry(), TetrahedronGeometry())
            for fi in 1:geo.n_faces
                i1,i2,i3 = get_face(geo, fi)
                v1=get_vertex(geo,i1); v2=get_vertex(geo,i2); v3=get_vertex(geo,i3)
                fn = cross(v2-v1, v3-v1)
                sn = get_normal(geo,i1)+get_normal(geo,i2)+get_normal(geo,i3)
                @test dot(fn, sn) > 0
            end
        end
    end

    @testset "LatheGeometry — vertical profile is a cylinder" begin
        lat = LatheGeometry([Vec2(1.0,0.0), Vec2(1.0,2.0)], segments=16)
        for vi in 1:lat.n_vertices
            v = get_vertex(lat, vi)
            @test sqrt(v.x^2 + v.z^2) ≈ 1.0 atol=1e-9
        end
        @test lat.n_faces == 16 * 1 * 2
    end

    @testset "TubeGeometry — straight path is a cylinder" begin
        tub = TubeGeometry([Vec3(0.0,0,0), Vec3(0.0,0,1.0), Vec3(0.0,0,2.0)];
                           radius=0.5, radial_segments=12)
        for vi in 1:tub.n_vertices
            v = get_vertex(tub, vi)
            @test sqrt(v.x^2 + v.y^2) ≈ 0.5 atol=1e-9     # distance from the z-axis
        end
    end

    @testset "Shape + Extrude" begin
        sq = [Vec2(0.0,0), Vec2(1.0,0), Vec2(1.0,1.0), Vec2(0.0,1.0)]
        sh = ShapeGeometry(sq)
        @test sh.n_faces == 2
        @test get_normal(sh, 1).z ≈ 1.0
        ex = ExtrudeGeometry(sq, depth=2.0)
        @test ex.n_faces == 12                            # 2 caps×2 + 4 walls×2
        bb = compute_bounding_box(ex)
        @test bb.min.z ≈ 0.0 && bb.max.z ≈ 2.0
        @test bb.min.x ≈ 0.0 && bb.max.x ≈ 1.0
    end

    @testset "CapsuleGeometry" begin
        radius, len = 0.5, 2.0; half = len/2
        cap = CapsuleGeometry(radius=radius, length=len, cap_segments=6, radial_segments=12)
        bb = compute_bounding_box(cap)
        @test (bb.max.y - bb.min.y) ≈ (len + 2*radius) atol=1e-9
        for vi in 1:cap.n_vertices
            v = get_vertex(cap, vi)
            cy = clamp(v.y, -half, half)                  # nearest spine point
            @test sqrt(v.x^2 + (v.y - cy)^2 + v.z^2) ≈ radius atol=1e-9
        end
    end

    @testset "Wireframe and Edges geometry" begin
        # EdgesGeometry of a cube: 12 feature edges (coplanar diagonals excluded).
        eg = edges_geometry(BoxGeometry())
        @test eg.n_faces == 0
        @test eg.n_vertices ÷ 2 == 12
        # WireframeGeometry shows all triangle edges including the quad diagonal.
        wf = wireframe_geometry(BoxGeometry())
        @test wf.n_vertices ÷ 2 == 30                     # 6 faces × (4 border + 1 diagonal)
        # Flat quad: 4 boundary edges (edges) vs 5 triangle edges (wireframe).
        @test edges_geometry(PlaneGeometry()).n_vertices ÷ 2 == 4
        @test wireframe_geometry(PlaneGeometry()).n_vertices ÷ 2 == 5
    end

    @testset "MeshToonMaterial — banded diffuse" begin
        m = MeshToonMaterial(color=Color3(1.0,1.0,1.0), gradient_steps=3)
        lights = AbstractLight[DirectionalLight(intensity=1.0, position=Vec3(0.0,0,1.0))]
        n = Vec3(0.0,0,1.0); v = Vec3(0.0,0,1.0)
        # Two normals in the same band quantize to the same colour.
        n1 = normalize(Vec3(0.05, 0.0, 1.0)); n2 = normalize(Vec3(0.10, 0.0, 1.0))
        c1 = shade_face(n1, v, Vec3(), m, lights)
        c2 = shade_face(n2, v, Vec3(), m, lights)
        @test c1.r ≈ c2.r                                  # banding ⇒ identical
        # Head-on lighting gives the top band (full intensity).
        @test shade_face(n, v, Vec3(), m, lights).r ≈ 1.0
    end

    @testset "MeshMatcapMaterial — view-facing falloff" begin
        m = MeshMatcapMaterial(color=Color3(1.0,1.0,1.0))
        v = Vec3(0.0,0,1.0)
        front = shade_face(Vec3(0.0,0,1.0), v, Vec3(), m, AbstractLight[])
        graze = shade_face(Vec3(1.0,0,0.0), v, Vec3(), m, AbstractLight[])
        @test front.r > graze.r                            # facing camera is brighter
        @test front.r ≈ 1.0 && graze.r ≈ 0.35
    end

    @testset "MeshPhysicalMaterial — clearcoat adds highlight" begin
        n = Vec3(0.0,0,1.0); v = Vec3(0.0,0,1.0)
        lights = AbstractLight[DirectionalLight(intensity=1.0, position=Vec3(0.0,0,1.0))]
        base = MeshPhysicalMaterial(color=Color3(0.5,0.0,0.0), metalness=0.0, roughness=0.5, clearcoat=0.0)
        coat = MeshPhysicalMaterial(color=Color3(0.5,0.0,0.0), metalness=0.0, roughness=0.5,
                                    clearcoat=1.0, clearcoat_roughness=0.1)
        cb = shade_face(n, v, Vec3(), base, lights)
        cc = shade_face(n, v, Vec3(), coat, lights)
        @test (cc.r + cc.g + cc.b) > (cb.r + cb.g + cb.b)   # coat brightens the highlight
    end

    @testset "MeshDepthMaterial — near brighter than far" begin
        scene = Scene(background=Color3(0.0,0,0))
        nearbox = Mesh(BoxGeometry(), MeshDepthMaterial(near=1.0, far=10.0)); nearbox.position = Vec3(-1.5,0,2.0)
        farbox  = Mesh(BoxGeometry(), MeshDepthMaterial(near=1.0, far=10.0)); farbox.position  = Vec3( 1.5,0,-3.0)
        add!(scene, nearbox); add!(scene, farbox)
        cam = PerspectiveCamera(fov=π/3, aspect=1.0, near=0.1, far=100.0); cam.position = Vec3(0.0,0,6.0)
        rt = RenderTarget(64, 64); render!(rt, scene, cam)
        nearbright = maximum(@view rt.color[:, 1:32, 1])
        farbright  = maximum(@view rt.color[:, 33:64, 1])
        @test nearbright > farbright                        # closer box renders brighter
    end

    @testset "Transparency — alpha compositing" begin
        scene = Scene(background=Color3(0.0, 0.0, 1.0))     # blue background
        quad = Mesh(PlaneGeometry(width=10.0, height=10.0),
                    MeshBasicMaterial(color=Color3(1.0,0,0), opacity=0.5, transparent=true))
        quad.position = Vec3(0.0, 0.0, 0.0)
        add!(scene, quad)
        cam = PerspectiveCamera(fov=π/4, aspect=1.0, near=0.1, far=100.0); cam.position = Vec3(0.0,0,3.0)
        rt = RenderTarget(32, 32); render!(rt, scene, cam)
        # Centre over the quad: 0.5·red + 0.5·blue ⇒ (0.5, 0, 0.5).
        @test rt.color[16,16,1] ≈ 0.5 atol=0.05
        @test rt.color[16,16,3] ≈ 0.5 atol=0.05
        @test is_transparent_material(quad.material)
        @test !is_transparent_material(MeshBasicMaterial(color=Color3(1.0,0,0)))
    end

    @testset "LightProbe — SH ambient fill" begin
        mat = MeshLambertMaterial(color=Color3(1.0,1.0,1.0))
        uni = AbstractLight[LightProbe(Color3(0.5,0.5,0.5))]
        up = shade_face(Vec3(0.0,1,0), Vec3(0.0,0,1), Vec3(), mat, uni)
        dn = shade_face(Vec3(0.0,-1,0), Vec3(0.0,0,1), Vec3(), mat, uni)
        @test up.r ≈ dn.r                                  # DC-only ⇒ orientation-independent
        @test up.r ≈ 0.5
        grad = AbstractLight[LightProbe(coeffs=(Color3(0.5,0.5,0.5), Color3(0.0,0,0),
                                                 Color3(0.5,0.5,0.5), Color3(0.0,0,0)))]
        gup = shade_face(Vec3(0.0,1,0), Vec3(0.0,0,1), Vec3(), mat, grad)
        gdn = shade_face(Vec3(0.0,-1,0), Vec3(0.0,0,1), Vec3(), mat, grad)
        @test gup.r > gdn.r                                # +y gradient ⇒ up brighter
    end

    @testset "RectAreaLight — facing vs behind" begin
        mat = MeshLambertMaterial(color=Color3(1.0,1.0,1.0))
        lights = AbstractLight[RectAreaLight(intensity=1.0, position=Vec3(0.0,0,5.0))]
        front = shade_face(Vec3(0.0,0,1.0), Vec3(0.0,0,1), Vec3(), mat, lights)
        back  = shade_face(Vec3(0.0,0,-1.0), Vec3(0.0,0,1), Vec3(), mat, lights)
        @test front.r > back.r
        @test back.r ≈ 0.0
    end

    @testset "Shadow mapping" begin
        scene = Scene()
        ground = Mesh(PlaneGeometry(width=40.0, height=40.0), MeshLambertMaterial())
        ground.rotation = Euler(-π/2, 0.0, 0.0); add!(scene, ground)
        box = Mesh(BoxGeometry(width=2.0,height=2.0,depth=2.0), MeshLambertMaterial())
        box.position = Vec3(0.0, 4.0, 0.0); add!(scene, box)
        key = DirectionalLight(position=Vec3(0.0,10.0,0.0)); key.target = Vec3(0.0,0,0)
        sm = compute_shadow_map(scene, key; resolution=512)
        @test shadow_visibility(sm, Vec3(0.0, 0.0, 0.0)) == 0.0      # under the box
        @test shadow_visibility(sm, Vec3(0.5, 0.0, 0.5)) == 0.0
        @test shadow_visibility(sm, Vec3(10.0, 0.0, 0.0)) == 1.0     # open ground
        @test shadow_visibility(sm, Vec3(0.0, 0.0, 10.0)) == 1.0

        # End-to-end: an angled key light casts the box's shadow onto visible
        # ground beside it. Enabling shadows can only darken, and darkens a region.
        scene2 = Scene()
        g2 = Mesh(PlaneGeometry(width=20.0,height=20.0), MeshLambertMaterial(color=Color3(0.9,0.9,0.9)))
        g2.rotation = Euler(-π/2,0.0,0.0); add!(scene2, g2)
        b2 = Mesh(BoxGeometry(width=2.0,height=2.0,depth=2.0), MeshLambertMaterial(color=Color3(0.8,0.2,0.2)))
        b2.position = Vec3(0.0, 1.0, 0.0); add!(scene2, b2)
        add!(scene2, AmbientLight(intensity=0.15))
        k2 = DirectionalLight(intensity=1.0, position=Vec3(6.0,8.0,2.0))
        k2.target = Vec3(0.0,0,0); k2.cast_shadow=true; add!(scene2, k2)
        cam = PerspectiveCamera(fov=π/4, aspect=1.0, near=0.1, far=100.0)
        cam.position = Vec3(0.0, 9.0, 12.0); cam.target = Vec3(0.0, 0.0, 0.0)
        rt_ns = RenderTarget(100,100); render!(rt_ns, scene2, cam; shadows=false)
        rt_s  = RenderTarget(100,100); render!(rt_s,  scene2, cam; shadows=true, shadow_resolution=512)
        diff = rt_ns.color[:,:,1] .- rt_s.color[:,:,1]
        @test all(>=(-1e-9), diff)                          # shadows never brighten
        @test count(>(0.1), diff) > 30                      # a visible shadow region exists
    end

    @testset "Tone mapping and sRGB" begin
        hdr = fill(4.0, 2, 2, 3)
        @test tone_map_reinhard(hdr)[1,1,1] ≈ 0.8           # 4/(1+4)
        @test 0.0 < tone_map_aces(hdr)[1,1,1] <= 1.0
        @test srgb_encode(fill(0.5, 1, 1, 3))[1,1,1] ≈ 0.7353569830524495 atol=1e-6
        @test srgb_to_linear(linear_to_srgb(0.3)) ≈ 0.3 atol=1e-10   # round-trip
        @test linear_to_srgb(0.0) ≈ 0.0
    end

    @testset "Supersample anti-aliasing" begin
        scene = Scene(background=Color3(0.0,0,0))
        add!(scene, Mesh(SphereGeometry(radius=1.0), MeshBasicMaterial(color=Color3(1.0,1,1))))
        cam = PerspectiveCamera(fov=π/4, aspect=1.0, near=0.1, far=100.0); cam.position = Vec3(0.0,0,4.0)
        aa = render_aa(scene, cam, 40, 40; ss=3)
        @test size(aa) == (40, 40, 3)
        # The sphere edge has partial-coverage (intermediate) values under AA.
        @test count(v -> 0.05 < v < 0.95, aa[:,:,1]) > 10
        @test maximum(aa) <= 1.0 + 1e-9
    end

    @testset "Tiled render equals render!" begin
        scene = Scene(background=Color3(0.05,0.05,0.1))
        add!(scene, Mesh(BoxGeometry(), MeshLambertMaterial(color=Color3(0.8,0.3,0.3))))
        add!(scene, AmbientLight(intensity=0.4))
        add!(scene, DirectionalLight(intensity=0.8, position=Vec3(3.0,3,4.0)))
        cam = PerspectiveCamera(fov=π/4, aspect=1.0, near=0.1, far=100.0); cam.position = Vec3(0.0,0,4.0)
        rt1 = RenderTarget(60,60); render!(rt1, scene, cam)
        rt2 = RenderTarget(60,60); render_tiled!(rt2, scene, cam; tiles=5)
        @test maximum(abs.(rt1.color .- rt2.color)) < 1e-12
    end

    @testset "Line and point rasterization" begin
        cam = PerspectiveCamera(fov=π/4, aspect=1.0, near=0.1, far=100.0); cam.position = Vec3(0.0,0,4.0)
        # Diagonal line.
        lg = BufferGeometry(); lg.positions = [-1.0,-1,0, 1.0,1.0,0]; lg.n_vertices = 2
        ls = Scene(background=Color3(0.0,0,0)); add!(ls, LineObject(lg, LineBasicMaterial(color=Color3(1.0,0,0))))
        rtl = RenderTarget(40,40); render!(rtl, ls, cam)
        @test count(>(0.5), rtl.color[:,:,1]) > 10          # line drew a streak of pixels
        # Single point at the origin.
        pg = BufferGeometry(); pg.positions = [0.0,0,0]; pg.n_vertices = 1
        ps = Scene(background=Color3(0.0,0,0)); add!(ps, PointsObject(pg, PointsMaterial(color=Color3(0.0,1,0), size=3.0)))
        rtp = RenderTarget(40,40); render!(rtp, ps, cam)
        @test count(>(0.5), rtp.color[:,:,2]) >= 1          # point lit
    end

    @testset "EffectComposer post-processing" begin
        img = rand(8, 8, 3)
        comp = EffectComposer(); add_pass!(comp, grayscale_pass)
        g = compose(comp, img)
        @test g[3,4,1] ≈ g[3,4,2] && g[3,4,2] ≈ g[3,4,3]   # grayscale ⇒ equal channels
        comp2 = EffectComposer(); add_pass!(comp2, reinhard_pass); add_pass!(comp2, srgb_pass)
        out = compose(comp2, fill(2.0, 4,4,3))
        @test all(0.0 .<= out .<= 1.0)
    end

    @testset "Backface culling and double-sided" begin
        cam = PerspectiveCamera(fov=π/4, aspect=1.0, near=0.1, far=100.0); cam.position = Vec3(0.0,0,4.0)
        # A plane flipped to face away from the camera.
        front = Scene(background=Color3(0.0,0,0))
        pf = Mesh(PlaneGeometry(width=4.0,height=4.0), MeshBasicMaterial(color=Color3(1.0,1,1), side=:front))
        pf.rotation = Euler(0.0, π, 0.0); add!(front, pf)
        rtf = RenderTarget(40,40); render!(rtf, front, cam)
        @test count(>(0.5), rtf.color[:,:,1]) == 0          # back face culled

        dbl = Scene(background=Color3(0.0,0,0))
        pd = Mesh(PlaneGeometry(width=4.0,height=4.0), MeshBasicMaterial(color=Color3(1.0,1,1), side=:double))
        pd.rotation = Euler(0.0, π, 0.0); add!(dbl, pd)
        rtd = RenderTarget(40,40); render!(rtd, dbl, cam)
        @test count(>(0.5), rtd.color[:,:,1]) > 100         # double-sided renders it
    end

    @testset "Per-mesh flatShading override" begin
        # Global smooth, but the mesh overrides to flat ⇒ far fewer distinct shades.
        function shades(flat_override)
            scene = Scene(background=Color3(0.0,0,0))
            sph = Mesh(SphereGeometry(radius=1.0, width_segments=24, height_segments=12),
                       MeshLambertMaterial(color=Color3(0.8,0.3,0.3)); flat_shading=flat_override)
            add!(scene, sph)
            add!(scene, AmbientLight(intensity=0.2))
            d = DirectionalLight(intensity=0.9, position=Vec3(3.0,3,4.0)); d.target = Vec3(0.0,0,0); add!(scene, d)
            cam = PerspectiveCamera(fov=π/4, aspect=1.0, near=0.1, far=100.0); cam.position = Vec3(0.0,0,4.0)
            rt = RenderTarget(80,80); render!(rt, scene, cam; shading=:smooth)
            length(unique(round.(vec(rt.color[:,:,1]), digits=3)))
        end
        @test shades(false) > 3 * shades(true)              # smooth has many more shades than flat
    end

    @testset "Texture sampling — checker, wrap, filter" begin
        tex = checker_texture(n=2, cell=1, a=Color3(1.0,1,1), b=Color3(0.0,0,0), filter=:nearest)
        # Quadrant centres alternate a/b.
        @test sample_texture(tex, 0.25, 0.75).r ≈ 1.0
        @test sample_texture(tex, 0.75, 0.75).r ≈ 0.0
        @test sample_texture(tex, 0.25, 0.25).r ≈ 0.0
        @test sample_texture(tex, 0.75, 0.25).r ≈ 1.0
        # Repeat wrap: u=1.25 samples the same texel as u=0.25.
        @test sample_texture(tex, 1.25, 0.75).r == sample_texture(tex, 0.25, 0.75).r
        # Clamp wrap holds the edge.
        texc = checker_texture(n=2, cell=1, filter=:nearest); texc.wrap_s = :clamp
        @test sample_texture(texc, 5.0, 0.5).r == sample_texture(texc, 0.999, 0.5).r
        # Mirror wrap reflects.
        texm = checker_texture(n=2, cell=1, filter=:nearest); texm.wrap_s = :mirror
        @test 0.0 <= sample_texture(texm, 1.25, 0.5).r <= 1.0
        # Bilinear gives an intermediate value at a texel boundary.
        texb = checker_texture(n=2, cell=1, filter=:bilinear)
        @test 0.0 < sample_texture(texb, 0.5, 0.5).r < 1.0
    end

    @testset "Mipmaps" begin
        t = DataTexture(rand(8,8,3)); generate_mipmaps!(t)
        @test length(t.mipmaps) == 3                       # 8 → 4 → 2 → 1
        @test size(t.mipmaps[1]) == (4,4,3)
        @test size(t.mipmaps[3]) == (1,1,3)
        # LOD 0 equals the base sample.
        @test sample_texture_lod(t, 0.5, 0.5, 0).r ≈ sample_texture(t, 0.5, 0.5).r
    end

    @testset "CubeTexture and DepthTexture" begin
        faces = ntuple(i -> DataTexture(fill(Float64(i)/6, 2,2,3)), 6)
        ct = CubeTexture(faces)
        @test sample_cube(ct, Vec3(1.0,0,0)).r ≈ 1/6        # +x face
        @test sample_cube(ct, Vec3(-1.0,0,0)).r ≈ 2/6       # -x face
        @test sample_cube(ct, Vec3(0.0,0,-1.0)).r ≈ 6/6     # -z face
        dt = DepthTexture(fill(0.7, 4, 4))
        @test sample_texture(dt, 0.5, 0.5).r ≈ 0.7          # single channel → gray
    end

    @testset "Material albedo map rendering" begin
        scene = Scene(background=Color3(0.0,0,0.5))
        pl = Mesh(PlaneGeometry(width=4.0,height=4.0, width_segments=16, height_segments=16),
                  MeshBasicMaterial(map=checker_texture(n=4, cell=4, filter=:nearest)))
        add!(scene, pl)
        cam = PerspectiveCamera(fov=π/4, aspect=1.0, near=0.1, far=100.0); cam.position = Vec3(0.0,0,5.0)
        rt = RenderTarget(96,96); render!(rt, scene, cam)
        @test count(>(0.9), rt.color[:,:,1]) > 100          # checker white cells
        @test count(<(0.1), @view rt.color[:,:,1]) > 100    # checker dark cells
    end

    @testset "DEFLATE inflate + base64" begin
        data = rand(UInt8, 1000)
        @test zlib_inflate(Three._zlib_store(data)) == data    # stored-block round-trip
        @test base64_decode("TWFu") == Vector{UInt8}(codeunits("Man"))
        @test base64_decode("TWE=") == Vector{UInt8}(codeunits("Ma"))
        @test base64_decode("TQ==") == Vector{UInt8}(codeunits("M"))
    end

    @testset "PNG decode round-trip" begin
        img = test_pattern(20, 12)
        f = tempname() * ".png"; save_png(f, img)
        dec = load_png(f)
        @test size(dec) == (12, 20, 3)
        @test maximum(abs.(dec .- img[:, :, 1:3])) <= 1/255 + 1e-9   # 8-bit quantization
        tex = TextureLoader(f); rm(f)
        @test tex isa Texture
        @test size(tex.data) == (12, 20, 3)
    end

    @testset "OBJ .mtl materials" begin
        dir = mktempdir()
        write(joinpath(dir, "m.mtl"), "newmtl red\nKd 0.8 0.2 0.1\nKs 0.5 0.5 0.5\nNs 64\n")
        mats = load_mtl(joinpath(dir, "m.mtl"))
        @test haskey(mats, "red")
        @test mats["red"].color.r ≈ 0.8 && mats["red"].color.g ≈ 0.2
        @test mats["red"].shininess ≈ 64.0
        # OBJ referencing the mtl with usemtl.
        write(joinpath(dir, "t.obj"),
              "mtllib m.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl red\nf 1 2 3\n")
        geo, face_mtl, m2 = load_obj_groups(joinpath(dir, "t.obj"))
        @test geo.n_faces == 1
        @test face_mtl == ["red"]
        @test haskey(m2, "red") && m2["red"].color.r ≈ 0.8
        rm(dir; recursive=true)
    end

    @testset "glTF 2.0 loader" begin
        dir = mktempdir()
        buf = vcat(reinterpret(UInt8, Float32[0,0,0, 1,0,0, 0,1,0]) |> collect,
                   reinterpret(UInt8, UInt16[0,1,2]) |> collect)
        write(joinpath(dir, "t.bin"), buf)
        gltf = """
        {"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],
        "meshes":[{"primitives":[{"attributes":{"POSITION":0},"indices":1}]}],
        "buffers":[{"byteLength":$(length(buf)),"uri":"t.bin"}],
        "bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":36},{"buffer":0,"byteOffset":36,"byteLength":6}],
        "accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"},
                     {"bufferView":1,"componentType":5123,"count":3,"type":"SCALAR"}]}
        """
        write(joinpath(dir, "t.gltf"), gltf)
        scene = load_gltf(joinpath(dir, "t.gltf"))
        meshes = collect_meshes(scene)
        @test length(meshes) == 1
        @test meshes[1].geometry.n_vertices == 3
        @test meshes[1].geometry.n_faces == 1
        v2 = get_vertex(meshes[1].geometry, 2)
        @test v2.x ≈ 1.0 && v2.y ≈ 0.0 && v2.z ≈ 0.0
        rm(dir; recursive=true)
    end

    @testset "OrbitControls" begin
        cam = PerspectiveCamera(); cam.position = Vec3(0.0,0,5.0); cam.target = Vec3(0.0,0,0)
        oc = OrbitControls(cam)
        r0 = norm(cam.position - oc.target)
        orbit_rotate!(oc, π/2, 0.0)
        @test norm(cam.position - oc.target) ≈ r0 atol=1e-9     # radius preserved on rotate
        @test abs(cam.position.x) > 1.0                          # camera actually moved
        orbit_zoom!(oc, 0.5)
        @test norm(cam.position - oc.target) ≈ r0 * 0.5 atol=1e-9
        orbit_set!(oc; azimuth=0.0, polar=π/2, radius=3.0)
        @test norm(cam.position - oc.target) ≈ 3.0 atol=1e-9
    end

    @testset "FlyControls" begin
        cam = PerspectiveCamera(); cam.position = Vec3(0.0,0,5.0); cam.target = Vec3(0.0,0,0)
        fc = FlyControls(cam)
        fly_translate!(fc, 1.0, 0.0, 0.0)                        # move forward (toward -z)
        @test cam.position.z ≈ 4.0 atol=1e-9
        @test cam.target.z ≈ -1.0 atol=1e-9                      # target shifts with camera
    end

    @testset "Clock" begin
        c = Clock()
        @test clock_elapsed(c, c.start_time + 2.0) ≈ 2.0
        @test clock_delta!(c, c.last_time + 0.5) ≈ 0.5
        @test clock_delta!(c, c.last_time + 0.25) ≈ 0.25         # last_time advanced
    end

    @testset "Animation — keyframe interpolation" begin
        mesh = Mesh(BoxGeometry(), MeshBasicMaterial())
        tr = KeyframeTrack(mesh, :position, [0.0, 1.0, 2.0],
                           [Vec3(0.0,0,0), Vec3(10.0,0,0), Vec3(10.0,5.0,0)])
        clip = AnimationClip("move", [tr])
        @test clip.duration ≈ 2.0
        mx = AnimationMixer(clip)
        mixer_set_time!(mx, 0.5)
        @test mesh.position.x ≈ 5.0
        mixer_update!(mx, 1.0)                                   # now t=1.5
        @test mesh.position.x ≈ 10.0
        @test mesh.position.y ≈ 2.5
    end

    @testset "Helpers" begin
        ax = AxesHelper(2.0)
        @test ax isa LineSegments
        @test ax.geometry.n_vertices == 6                        # 3 segments
        @test has_attribute(ax.geometry, :color)
        @test GridHelper(10.0, 5).geometry.n_vertices == 24      # 6 lines × 2 dirs × 2 verts
        bh = BoxHelper(Mesh(BoxGeometry(width=2.0,height=2.0,depth=2.0), MeshBasicMaterial()))
        @test bh.geometry.n_vertices == 24                       # 12 edges × 2 verts
        @test CameraHelper(PerspectiveCamera()).geometry.n_vertices == 24
        @test DirectionalLightHelper(DirectionalLight(position=Vec3(5.0,5,5))).geometry.n_vertices == 2
        @test PointLightHelper(PointLight(position=Vec3(1.0,1,1))).geometry.n_vertices == 6
    end

    @testset "Alpha PNG round-trip" begin
        rgba = rand(8, 10, 4)
        f = tempname() * ".png"; save_png_rgba(f, rgba)
        dec = load_png(f); rm(f)
        @test size(dec) == (8, 10, 4)
        @test maximum(abs.(dec .- rgba)) <= 1/255 + 1e-9
    end

    @testset "16-bit PNG round-trip" begin
        gray = [Float64(i*j)/(8*10) for i in 1:8, j in 1:10]
        f = tempname() * ".png"; save_png16(f, gray)
        dec = load_png(f); rm(f)
        @test size(dec) == (8, 10, 1)
        @test maximum(abs.(dec[:,:,1] .- gray)) <= 1/65535 + 1e-9   # finer than 8-bit's 1/255
    end

    @testset "Differentiable — high-dim vertex gradient" begin
        faces = [(1,2,3)]; fcols = [Color3(1.0,0,0)]
        vp = mat4_perspective(π/4,1.0,0.1,100.0) * mat4_look_at(Vec3(0.0,0,3.0), Vec3(0.0,0,0), Vec3(0.0,1,0))
        rf = vertex_render_fn(faces, fcols, vp, 16, 16; sigma=0.5, gamma=1.0)
        p0 = [-0.5,-0.5,0.0, 0.5,-0.5,0.0, 0.0,0.5,0.0]
        g = ForwardDiff.gradient(p -> sum(rf(p)), p0)
        @test length(g) == 9
        @test all(isfinite, g)
        @test count(!=(0.0), g) >= 6                    # most components respond
        # AD matches a central finite difference on the first x-coordinate.
        δ = 1e-4
        pp = copy(p0); pp[1] += δ; pm = copy(p0); pm[1] -= δ
        fd = (sum(rf(pp)) - sum(rf(pm))) / (2δ)
        @test abs(g[1] - fd) <= 1e-2 * (abs(fd) + 1)
    end

    @testset "Differentiable — vertex-position optimization" begin
        faces = [(1,2,3)]; fcols = [Color3(1.0,0,0)]
        vp = mat4_perspective(π/4,1.0,0.1,100.0) * mat4_look_at(Vec3(0.0,0,3.0), Vec3(0.0,0,0), Vec3(0.0,1,0))
        rf = vertex_render_fn(faces, fcols, vp, 16, 16; sigma=0.5, gamma=1.0)
        target = rf([-0.5,-0.5,0.0, 0.5,-0.5,0.0, 0.0,0.5,0.0])
        pinit = [-0.35,-0.6,0.0, 0.4,-0.38,0.0, 0.08,0.55,0.0]
        loss0 = loss_mse(rf(pinit), target)
        _, hist = optimize_vertices(pinit, faces, fcols, vp, target;
                                    W=16, H=16, sigma=0.5, gamma=1.0, lr=0.05, n_iters=40, verbose=false)
        @test hist[end] < 0.25 * loss0                  # optimization substantially reduces loss
    end

    @testset "Differentiable texture — face-color optimization" begin
        faces = [(1,2,3)]
        verts = [Vec3(-0.5,-0.5,0.0), Vec3(0.5,-0.5,0.0), Vec3(0.0,0.5,0.0)]
        vp = mat4_perspective(π/4,1.0,0.1,100.0) * mat4_look_at(Vec3(0.0,0,3.0), Vec3(0.0,0,0), Vec3(0.0,1,0))
        crf = color_render_fn(verts, faces, vp, 16, 16; sigma=0.5, gamma=1.0)
        target = crf([1.0, 0.0, 0.0])                   # red
        copt, chist = optimize_face_colors([0.0,0.0,1.0], verts, faces, vp, target;
                                           W=16, H=16, sigma=0.5, gamma=1.0, lr=0.1, n_iters=40, verbose=false)
        @test chist[end] < chist[1]
        @test copt[1] > 0.7 && copt[3] < 0.3            # converged toward red
    end

    @testset "Industrial scale — 100K+ triangles" begin
        sphere_faces = SphereGeometry(radius=0.7, width_segments=16, height_segments=8).n_faces
        n_inst = cld(100_000, sphere_faces)
        scene = build_instanced_scene(n_inst)
        tris = scene_triangle_count(scene)
        @test tris >= 100_000                            # venue-gate scale
        side = ceil(Int, cbrt(n_inst)); c = side * 1.0
        cam = PerspectiveCamera(fov=π/4, aspect=1.0, near=0.1, far=1000.0)
        cam.position = Vec3(c*2.5, c*2.5, c*4.0); cam.target = Vec3(c, c, c)
        rt = RenderTarget(128, 128); render!(rt, scene, cam)
        @test count(>(0.05), rt.color[:,:,1]) > 500      # the scene actually renders
        # RenderTarget buffers are reused across frames (identical re-render).
        s1 = copy(rt.color); render!(rt, scene, cam)
        @test rt.color == s1
    end

    @testset "Benchmark harness" begin
        scene = build_instanced_scene(80)                # ~18K triangles, fast
        cam = PerspectiveCamera(fov=π/4, aspect=1.0, near=0.1, far=500.0)
        cam.position = Vec3(10.0, 10.0, 16.0); cam.target = Vec3(4.0, 4.0, 4.0)
        br = benchmark_render(scene, cam, 96, 96; warmup=1, reps=5)
        @test br.reps == 5
        @test br.median_s > 0.0
        @test br.iqr_s >= 0.0
        @test br.min_s <= br.median_s                    # min ≤ median by construction
        @test br.triangles == scene_triangle_count(scene)
    end

    @testset "ShaderMaterial — executable fragment program" begin
        prog = (n, v, p, u) -> Color3(clamp(p.x, 0.0, 1.0), 0.0, get(u, "b", 0.0))
        sm = ShaderMaterial(program=prog, uniforms=Dict{String,Any}("b"=>0.25))
        c = shade_face(Vec3(0.0,0,1), Vec3(0.0,0,1), Vec3(0.7, 0.0, 0.0), sm, AbstractLight[])
        @test c.r ≈ 0.7
        @test c.b ≈ 0.25                                   # uniform read
        @test shade_face(Vec3(0.0,0,1), Vec3(0.0,0,1), Vec3(), ShaderMaterial(), AbstractLight[]).r ≈ 0.5
    end

    @testset "Normal map and roughness map" begin
        geo = PlaneGeometry(width=2.0, height=2.0)
        wm = Mat4{Float64}()
        light = AbstractLight[DirectionalLight(intensity=1.0, position=Vec3(1.0,1.0,1.0))]
        campos = Vec3(0.0, 0.0, 3.0)
        # A normal map that decodes to (0,0,1) leaves shading unchanged.
        flatmap = DataTexture(cat(fill(0.5,4,4), fill(0.5,4,4), fill(1.0,4,4); dims=3))
        mk(nm) = MeshStandardMaterial(color=Color3(0.8,0.8,0.8), metalness=0.0, roughness=1.0, normal_map=nm)
        base = shade_mesh_faces(geo, wm, mk(nothing), light, campos)
        same = shade_mesh_faces(geo, wm, mk(flatmap), light, campos)
        @test base[1].r ≈ same[1].r atol=1e-9
        # A tilted normal map changes the shaded result.
        tiltmap = DataTexture(cat(fill(0.95,4,4), fill(0.5,4,4), fill(0.7,4,4); dims=3))
        tilt = shade_mesh_faces(geo, wm, mk(tiltmap), light, campos)
        @test abs(tilt[1].r - base[1].r) > 1e-3
        # Roughness map overrides the material roughness per face (PBR result moves).
        rmap = DataTexture(fill(0.9, 4, 4, 3))
        sharp = shade_mesh_faces(geo, wm, MeshStandardMaterial(color=Color3(0.8,0.2,0.2), metalness=0.5, roughness=0.05), light, campos)
        mapped = shade_mesh_faces(geo, wm, MeshStandardMaterial(color=Color3(0.8,0.2,0.2), metalness=0.5, roughness=0.05, roughness_map=rmap), light, campos)
        @test abs(sharp[1].r - mapped[1].r) > 1e-4
    end

    @testset "In-renderer MSAA" begin
        scene = Scene(background=Color3(0.0,0,0))
        add!(scene, Mesh(SphereGeometry(radius=1.0), MeshBasicMaterial(color=Color3(1.0,1,1))))
        cam = PerspectiveCamera(fov=π/4, aspect=1.0, near=0.1, far=100.0); cam.position = Vec3(0.0,0,4.0)
        rt = RenderTarget(48, 48); render_msaa!(rt, scene, cam; samples=9)
        @test size(rt.color) == (48, 48, 3)
        @test count(v -> 0.05 < v < 0.95, rt.color[:,:,1]) > 10   # anti-aliased edge band
    end

    @testset "Pooled rendering — identical output, bounded allocation" begin
        scene = build_instanced_scene(40)
        cam = PerspectiveCamera(fov=π/4, aspect=1.0, near=0.1, far=500.0)
        cam.position = Vec3(8.0,8,14.0); cam.target = Vec3(3.0,3,3)
        r1 = RenderTarget(64,64); render!(r1, scene, cam)
        cache = RenderCache(); r2 = RenderTarget(64,64); render_pooled!(r2, scene, cam, cache)
        @test maximum(abs.(r1.color .- r2.color)) < 1e-12      # same image as render!
        a2 = @allocated render_pooled!(r2, scene, cam, cache)
        a3 = @allocated render_pooled!(r2, scene, cam, cache)
        @test a3 <= a2                                         # allocation does not grow per frame
    end

    @testset "Reverse-mode AD — matches ForwardDiff" begin
        # Analytic: ∇ Σxᵢ² = 2x.
        @test reverse_gradient(x -> sum(x.^2), [1.0,2.0,3.0,4.0]) == [2.0,4.0,6.0,8.0]
        # Non-trivial scalar function (exp/sqrt/max) vs ForwardDiff.
        hfun(x) = exp(x[1]) * sqrt(abs(x[2]) + 1) + max(x[1], x[3]) - x[2]/x[3]
        x0 = [0.4, -1.3, 2.1]
        @test maximum(abs.(reverse_gradient(hfun, x0) .- ForwardDiff.gradient(hfun, x0))) < 1e-10
        # Through the full soft rasterizer (high-dim vertex gradient).
        faces = [(1,2,3)]; fcols = [Color3(1.0,0,0)]
        vp = mat4_perspective(π/4,1.0,0.1,100.0) * mat4_look_at(Vec3(0.0,0,3.0), Vec3(0.0,0,0), Vec3(0.0,1,0))
        rf = vertex_render_fn(faces, fcols, vp, 12, 12; sigma=0.5, gamma=1.0)
        p0 = [-0.5,-0.5,0.0, 0.5,-0.5,0.0, 0.0,0.5,0.0]
        gr = reverse_gradient(p -> sum(rf(p)), p0)
        gf = ForwardDiff.gradient(p -> sum(rf(p)), p0)
        @test length(gr) == 9
        @test maximum(abs.(gr .- gf)) < 1e-8                   # reverse == forward
    end

end
