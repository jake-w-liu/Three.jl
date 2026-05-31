# --------------------------------------------------------------------------
# Browser/WebGL export for interactive examples.
#
# The exporter serializes Three.jl scene objects into a standalone HTML file
# with a small generic WebGL runtime. Scene construction, geometry generation,
# materials, transforms, instancing, and keyframe clips come from Three.jl.
# --------------------------------------------------------------------------

struct WebGLExportCase
    id::String
    title::String
    subtitle::String
    scene::Scene
    target::Vec3{Float64}
    radius::Float64
    height::Float64
    fov::Float64
    animations::Vector{AnimationClip}
end

function WebGLExportCase(id::String, title::String, subtitle::String, scene::Scene;
                         target=Vec3(0.0, 0.0, 0.0), radius::Real=8.0,
                         height::Real=3.0, fov::Real=pi/4,
                         animations::AbstractVector{AnimationClip}=AnimationClip[])
    WebGLExportCase(id, title, subtitle, scene, target, Float64(radius),
                    Float64(height), Float64(fov), collect(AnimationClip, animations))
end

_js_str(s::AbstractString) = "\"" * replace(s,
    "\\"=>"\\\\", "\""=>"\\\"", "\b"=>"\\b", "\f"=>"\\f", "\n"=>"\\n",
    "\r"=>"\\r", "\t"=>"\\t", "</"=>"<\\/") * "\""
_js_num(x::Real) = isfinite(Float64(x)) ? @sprintf("%.17g", Float64(x)) : "0"
_js_array(xs) = "[" * join((_js_num(x) for x in xs), ",") * "]"
_js_vec(v::Vec3) = "[" * _js_num(v.x) * "," * _js_num(v.y) * "," * _js_num(v.z) * "]"
_js_color(c::Color3) = "[" * _js_num(c.r) * "," * _js_num(c.g) * "," * _js_num(c.b) * "]"
_js_mat(m::Mat4) = _js_array(m.e)
_html_escape(s::AbstractString) = replace(s, "&"=>"&amp;", "<"=>"&lt;", ">"=>"&gt;", "\""=>"&quot;")

function _web_material_color(mat)
    if hasproperty(mat, :color)
        return getproperty(mat, :color)
    elseif mat isa MeshNormalMaterial
        return Color3(0.62, 0.86, 1.0)
    else
        return Color3(1.0, 1.0, 1.0)
    end
end

_web_material_size(mat) = hasproperty(mat, :size) ? Float64(getproperty(mat, :size)) : 4.0
_web_material_glow(mat) = mat isa MeshBasicMaterial ? 0.35 : mat isa PointsMaterial ? 1.0 : 0.08

function _web_geo_object(geo::BufferGeometry)
    normals = length(geo.normals) == length(geo.positions) ? geo.normals : zeros(Float64, length(geo.positions))
    indices = isempty(geo.indices) ? collect(1:geo.n_vertices) : geo.indices
    return "\"positions\":" * _js_array(geo.positions) *
           ",\"normals\":" * _js_array(normals) *
           ",\"indices\":" * _js_array(indices .- 1)
end

function _web_drawable_json(obj, world::Mat4; matrix=nothing, mode::String="triangles")
    geo = obj.geometry
    mat = obj.material
    m = matrix === nothing ? world : matrix
    floor = occursin("floor", lowercase(getproperty(obj, :name)))
    return "{" *
           "\"id\":" * string(obj.id) *
           ",\"name\":" * _js_str(getproperty(obj, :name)) *
           ",\"mode\":" * _js_str(mode) *
           ",\"matrix\":" * _js_mat(m) *
           ",\"color\":" * _js_color(_web_material_color(mat)) *
           ",\"pointSize\":" * _js_num(_web_material_size(mat)) *
           ",\"glow\":" * _js_num(_web_material_glow(mat)) *
           ",\"floor\":" * (floor ? "true" : "false") *
           "," * _web_geo_object(geo) *
           "}"
end

function _web_collect_drawables(root::AbstractObject3D)
    out = String[]
    function visit(obj::AbstractObject3D)
        is_visible(obj) || return
        world = compute_world_matrix(obj)
        if obj isa Mesh
            push!(out, _web_drawable_json(obj, world; mode="triangles"))
        elseif obj isa InstancedMesh
            parent = compute_world_matrix(obj)
            for im in obj.instance_matrices
                push!(out, _web_drawable_json(obj, parent * im; mode="triangles"))
            end
        elseif obj isa PointsObject
            push!(out, _web_drawable_json(obj, world; mode="points"))
        elseif obj isa LineObject
            push!(out, _web_drawable_json(obj, world; mode="line_strip"))
        elseif obj isa LineSegments
            push!(out, _web_drawable_json(obj, world; mode="lines"))
        end
        for child in get_children(obj)
            visit(child)
        end
    end
    visit(root)
    return out
end

function _web_track_json(tr::KeyframeTrack)
    values = Float64[]
    for v in tr.values
        append!(values, (v.x, v.y, v.z))
    end
    return "{" *
           "\"target\":" * string(tr.target.id) *
           ",\"property\":" * _js_str(String(tr.property)) *
           ",\"kind\":\"vec3\"" *
           ",\"times\":" * _js_array(tr.times) *
           ",\"values\":" * _js_array(values) *
           ",\"interpolation\":" * _js_str(String(tr.interpolation)) *
           "}"
end

function _web_track_json(tr::QuaternionKeyframeTrack)
    values = Float64[]
    for q in tr.values
        append!(values, (q.x, q.y, q.z, q.w))
    end
    return "{" *
           "\"target\":" * string(tr.target.id) *
           ",\"property\":" * _js_str(String(tr.property)) *
           ",\"kind\":\"quat\"" *
           ",\"times\":" * _js_array(tr.times) *
           ",\"values\":" * _js_array(values) *
           ",\"interpolation\":\"slerp\"" *
           "}"
end

function _web_clip_json(clip::AnimationClip)
    tracks = join((_web_track_json(t) for t in clip.tracks), ",")
    return "{" *
           "\"name\":" * _js_str(clip.name) *
           ",\"duration\":" * _js_num(clip.duration) *
           ",\"tracks\":[" * tracks * "]" *
           "}"
end

function _web_case_json(case::WebGLExportCase)
    return "{" *
           "\"id\":" * _js_str(case.id) *
           ",\"title\":" * _js_str(case.title) *
           ",\"subtitle\":" * _js_str(case.subtitle) *
           ",\"background\":" * _js_color(case.scene.background) *
           ",\"target\":" * _js_vec(case.target) *
           ",\"radius\":" * _js_num(case.radius) *
           ",\"height\":" * _js_num(case.height) *
           ",\"fov\":" * _js_num(case.fov) *
           ",\"objects\":[" * join(_web_collect_drawables(case.scene), ",") * "]" *
           ",\"animations\":[" * join((_web_clip_json(c) for c in case.animations), ",") * "]" *
           "}"
end

function _webgl_html(data_json::String, title::String)
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>$(_html_escape(title))</title>
  <style>
    :root { color-scheme: dark; --bg:#080b10; --panel:#111821; --text:#f2f7ff; --muted:#9eb0c4; --edge:#273443; --accent:#50c8ff; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    main { width:min(1220px, calc(100vw - 28px)); margin:0 auto; padding:24px 0 34px; }
    header { display:flex; justify-content:space-between; align-items:end; gap:16px; margin-bottom:16px; }
    h1 { margin:0 0 6px; font-size:26px; letter-spacing:0; }
    p { margin:0; color:var(--muted); line-height:1.45; }
    .layout { display:grid; grid-template-columns:minmax(0, 1fr) 300px; gap:16px; align-items:start; }
    .stage { border:1px solid var(--edge); border-radius:8px; overflow:hidden; background:#020305; }
    canvas { display:block; width:100%; aspect-ratio:16 / 10; touch-action:none; cursor:grab; }
    canvas:active { cursor:grabbing; }
    .bar { display:flex; justify-content:space-between; gap:12px; padding:12px 14px; background:var(--panel); border-top:1px solid var(--edge); }
    .bar strong { font-size:14px; }
    .bar span { color:var(--muted); font-size:13px; text-align:right; }
    .cases { display:grid; gap:10px; }
    button { text-align:left; border:1px solid var(--edge); border-radius:8px; background:var(--panel); color:var(--text); padding:12px; cursor:pointer; }
    button.active { border-color:var(--accent); background:#102233; box-shadow:0 0 0 1px rgba(80,200,255,.3) inset; }
    button strong { display:block; font-size:14px; margin-bottom:5px; }
    button span { display:block; color:var(--muted); font-size:12px; line-height:1.35; }
    @media (max-width: 840px) { header { display:block; } .layout { grid-template-columns:1fr; } .cases { grid-template-columns:repeat(auto-fit, minmax(220px,1fr)); } }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Three.jl Live WebGL Showcase</h1>
        <p>Scenes, geometry, materials, instancing, and keyframes are exported from Three.jl. Drag to orbit; wheel to zoom.</p>
      </div>
      <p id="stats">loading</p>
    </header>
    <section class="layout">
      <div class="stage">
        <canvas id="canvas"></canvas>
        <div class="bar"><strong id="title"></strong><span id="subtitle"></span></div>
      </div>
      <nav id="cases" class="cases"></nav>
    </section>
  </main>
  <script>
  const DATA = $data_json;
  const canvas = document.getElementById("canvas");
  const gl = canvas.getContext("webgl", {antialias:true});
  if (!gl) throw new Error("WebGL is not available");

  const M4 = {
    ident(){ return [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]; },
    mul(a,b){ const r=new Array(16); for(let c=0;c<4;c++) for(let row=0;row<4;row++) r[c*4+row]=a[row]*b[c*4]+a[4+row]*b[c*4+1]+a[8+row]*b[c*4+2]+a[12+row]*b[c*4+3]; return r; },
    translate(x,y,z){ return [1,0,0,0, 0,1,0,0, 0,0,1,0, x,y,z,1]; },
    scale(x,y,z){ return [x,0,0,0, 0,y,0,0, 0,0,z,0, 0,0,0,1]; },
    quat(q){ const x=q[0],y=q[1],z=q[2],w=q[3], x2=x+x,y2=y+y,z2=z+z, xx=x*x2,xy=x*y2,xz=x*z2, yy=y*y2,yz=y*z2,zz=z*z2, wx=w*x2,wy=w*y2,wz=w*z2; return [1-(yy+zz),xy+wz,xz-wy,0, xy-wz,1-(xx+zz),yz+wx,0, xz+wy,yz-wx,1-(xx+yy),0, 0,0,0,1]; },
    trs(p,q,s){ return M4.mul(M4.translate(p[0],p[1],p[2]), M4.mul(M4.quat(q), M4.scale(s[0],s[1],s[2]))); },
    perspective(fov, aspect, near, far){ const t=Math.tan(fov/2); return [1/(aspect*t),0,0,0, 0,1/t,0,0, 0,0,-(far+near)/(far-near),-1, 0,0,-2*far*near/(far-near),0]; },
    lookAt(eye,target,up){ const z=norm(sub(eye,target)), x=norm(cross(up,z)), y=cross(z,x); return [x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0, -dot(x,eye),-dot(y,eye),-dot(z,eye),1]; },
    normal3(m){ const a=m[0],b=m[1],c=m[2],d=m[4],e=m[5],f=m[6],g=m[8],h=m[9],i=m[10]; const A=e*i-f*h,B=-(d*i-f*g),C=d*h-e*g,D=-(b*i-c*h),E=a*i-c*g,F=-(a*h-b*g),G=b*f-c*e,H=-(a*f-c*d),I=a*e-b*d; const det=a*A+b*B+c*C; if(Math.abs(det)<1e-10) return [1,0,0,0,1,0,0,0,1]; const inv=1/det; return [A*inv,B*inv,C*inv,D*inv,E*inv,F*inv,G*inv,H*inv,I*inv]; }
  };
  const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];
  const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
  const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
  const norm=a=>{ const l=Math.hypot(a[0],a[1],a[2])||1; return [a[0]/l,a[1]/l,a[2]/l]; };
  const mix=(a,b,t)=>a+(b-a)*t;
  const mix3=(a,b,t)=>[mix(a[0],b[0],t),mix(a[1],b[1],t),mix(a[2],b[2],t)];
  function slerp(a,b,t){ let cos=a[0]*b[0]+a[1]*b[1]+a[2]*b[2]+a[3]*b[3]; if(cos<0){ b=[-b[0],-b[1],-b[2],-b[3]]; cos=-cos; } if(cos>.9995){ const q=[mix(a[0],b[0],t),mix(a[1],b[1],t),mix(a[2],b[2],t),mix(a[3],b[3],t)]; const l=Math.hypot(...q)||1; return q.map(v=>v/l); } const th=Math.acos(cos), s=Math.sin(th); return [Math.sin((1-t)*th)/s*a[0]+Math.sin(t*th)/s*b[0], Math.sin((1-t)*th)/s*a[1]+Math.sin(t*th)/s*b[1], Math.sin((1-t)*th)/s*a[2]+Math.sin(t*th)/s*b[2], Math.sin((1-t)*th)/s*a[3]+Math.sin(t*th)/s*b[3]]; }

  const VSH=`attribute vec3 aPosition; attribute vec3 aNormal; uniform mat4 uModel,uView,uProj; uniform mat3 uNormalMat; varying vec3 vNormal,vWorld; void main(){ vec4 w=uModel*vec4(aPosition,1.0); vWorld=w.xyz; vNormal=normalize(uNormalMat*aNormal); gl_Position=uProj*uView*w; }`;
  const FSH=`precision mediump float; varying vec3 vNormal,vWorld; uniform vec3 uColor,uCamera; uniform float uGlow; void main(){ vec3 n=normalize(vNormal); if(!gl_FrontFacing)n=-n; vec3 l=normalize(vec3(.55,.9,.42)); vec3 v=normalize(uCamera-vWorld); vec3 h=normalize(l+v); float d=max(dot(n,l),0.0); float s=pow(max(dot(n,h),0.0),32.0); vec3 c=uColor*(.18+.82*d)+s*vec3(.9,.95,1.0)+uGlow*uColor*.28; gl_FragColor=vec4(c,1.0); }`;
  const CVSH=`attribute vec3 aPosition; uniform mat4 uModel,uView,uProj; void main(){ gl_Position=uProj*uView*uModel*vec4(aPosition,1.0); }`;
  const CFSH=`precision mediump float; uniform vec3 uColor; uniform float uGlow; void main(){ gl_FragColor=vec4(uColor*(.75+.65*uGlow),1.0); }`;
  const PVSH=`attribute vec3 aPosition; uniform mat4 uModel,uView,uProj; uniform float uPointSize; void main(){ gl_Position=uProj*uView*uModel*vec4(aPosition,1.0); gl_PointSize=uPointSize; }`;
  const PFSH=`precision mediump float; uniform vec3 uColor; uniform float uGlow; void main(){ vec2 d=gl_PointCoord-vec2(.5); float r=dot(d,d); if(r>.25) discard; float a=smoothstep(.25,0.0,r); gl_FragColor=vec4(uColor*(.65+uGlow*.85)*a,1.0); }`;
  function shader(type,src){ const s=gl.createShader(type); gl.shaderSource(s,src); gl.compileShader(s); if(!gl.getShaderParameter(s,gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; }
  function program(vs,fs){ const p=gl.createProgram(); gl.attachShader(p,shader(gl.VERTEX_SHADER,vs)); gl.attachShader(p,shader(gl.FRAGMENT_SHADER,fs)); gl.linkProgram(p); if(!gl.getProgramParameter(p,gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p)); return p; }
  const meshProgram=program(VSH,FSH), colorProgram=program(CVSH,CFSH), pointProgram=program(PVSH,PFSH);
  function buf(data,target=gl.ARRAY_BUFFER,ctor=Float32Array){ const b=gl.createBuffer(); gl.bindBuffer(target,b); gl.bufferData(target,new ctor(data),gl.STATIC_DRAW); return b; }
  function buildObj(o){
    o.baseMatrix=o.matrix.slice(); o.animPos=[0,0,0]; o.animScale=[1,1,1]; o.animQuat=[0,0,0,1];
    o.posBuf=buf(o.positions); o.nrmBuf=buf(o.normals);
    const maxIndex=o.indices.reduce((m,v)=>Math.max(m,v),0);
    o.indexType=maxIndex>65535 ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT;
    if(o.indexType===gl.UNSIGNED_INT && !gl.getExtension("OES_element_index_uint")) throw new Error("OES_element_index_uint is required for this exported geometry");
    o.idxBuf=buf(o.indices,gl.ELEMENT_ARRAY_BUFFER,o.indexType===gl.UNSIGNED_INT ? Uint32Array : Uint16Array);
    o.count=o.indices.length; return o;
  }
  for(const c of DATA.cases) c.objects=c.objects.map(buildObj);
  const objectById = new Map(); for(const c of DATA.cases) for(const o of c.objects) if(!objectById.has(o.id)) objectById.set(o.id,[]); for(const c of DATA.cases) for(const o of c.objects) objectById.get(o.id).push(o);
  function attrib(p,name,b){ const loc=gl.getAttribLocation(p,name); if(loc<0)return; gl.bindBuffer(gl.ARRAY_BUFFER,b); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc,3,gl.FLOAT,false,0,0); }
  function sampleTrack(tr,t){ const times=tr.times; if(times.length===0) return null; const dur=times[times.length-1] || 1; t=((t%dur)+dur)%dur; let i=0; while(i+1<times.length && times[i+1]<t)i++; const stride=tr.kind==="quat"?4:3; if(tr.interpolation==="step") return tr.values.slice(i*stride,i*stride+stride); const a=times[i], b=times[Math.min(i+1,times.length-1)]; const u=b===a?0:(t-a)/(b-a); const p=tr.values.slice(i*stride,i*stride+stride), q=tr.values.slice(Math.min(i+1,times.length-1)*stride,Math.min(i+1,times.length-1)*stride+stride); return tr.kind==="quat"?slerp(p,q,u):mix3(p,q,u); }
  function applyAnimations(c,t){ for(const o of c.objects){ o.animPos=[0,0,0]; o.animScale=[1,1,1]; o.animQuat=[0,0,0,1]; o.matrix=o.baseMatrix; } for(const clip of c.animations) for(const tr of clip.tracks){ const v=sampleTrack(tr,t); const objs=objectById.get(tr.target)||[]; for(const o of objs){ if(tr.property==="position") o.animPos=v; else if(tr.property==="scale") o.animScale=v; else if(tr.property==="rotation") o.animQuat=v; o.matrix=M4.mul(o.baseMatrix,M4.trs(o.animPos,o.animQuat,o.animScale)); } } }
  function draw(o,view,proj,eye){ const p=o.mode==="points"?pointProgram:(o.mode==="triangles"?meshProgram:colorProgram); gl.useProgram(p); attrib(p,"aPosition",o.posBuf); if(o.mode==="triangles") attrib(p,"aNormal",o.nrmBuf); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,o.idxBuf); gl.uniformMatrix4fv(gl.getUniformLocation(p,"uModel"),false,new Float32Array(o.matrix)); gl.uniformMatrix4fv(gl.getUniformLocation(p,"uView"),false,new Float32Array(view)); gl.uniformMatrix4fv(gl.getUniformLocation(p,"uProj"),false,new Float32Array(proj)); gl.uniform3fv(gl.getUniformLocation(p,"uColor"),new Float32Array(o.color)); gl.uniform1f(gl.getUniformLocation(p,"uGlow"),o.glow||0); if(o.mode==="triangles"){ gl.uniformMatrix3fv(gl.getUniformLocation(p,"uNormalMat"),false,new Float32Array(M4.normal3(o.matrix))); gl.uniform3fv(gl.getUniformLocation(p,"uCamera"),new Float32Array(eye)); } if(o.mode==="points") gl.uniform1f(gl.getUniformLocation(p,"uPointSize"),Math.max(2,o.pointSize*1.5)); const mode=o.mode==="points"?gl.POINTS:(o.mode==="lines"?gl.LINES:(o.mode==="line_strip"?gl.LINE_STRIP:gl.TRIANGLES)); gl.drawElements(mode,o.count,o.indexType,0); }
  const nav=document.getElementById("cases"), titleEl=document.getElementById("title"), subEl=document.getElementById("subtitle"), stats=document.getElementById("stats");
  let active=DATA.cases[0], yaw=.65, pitch=.53, dist=active.radius, dragging=false, lx=0, ly=0;
  for(const c of DATA.cases){ const b=document.createElement("button"); b.dataset.case=c.id; const strong=document.createElement("strong"); strong.textContent=c.title; const span=document.createElement("span"); span.textContent=c.subtitle; b.append(strong,span); b.onclick=()=>setCase(c.id); nav.appendChild(b); }
  function setCase(id){ active=DATA.cases.find(c=>c.id===id); dist=active.radius; titleEl.textContent=active.title; subEl.textContent=active.subtitle; document.querySelectorAll("button[data-case]").forEach(b=>b.classList.toggle("active",b.dataset.case===id)); }
  function resize(){ const r=canvas.getBoundingClientRect(), dpr=Math.min(devicePixelRatio||1,2); const w=Math.max(1,Math.round(r.width*dpr)), h=Math.max(1,Math.round(r.height*dpr)); if(canvas.width!==w||canvas.height!==h){ canvas.width=w; canvas.height=h; } }
  function render(){ resize(); const t=performance.now()*.001; applyAnimations(active,t); const target=active.target; const eye=[target[0]+dist*Math.cos(pitch)*Math.cos(yaw), target[1]+dist*Math.sin(pitch), target[2]+dist*Math.cos(pitch)*Math.sin(yaw)]; const view=M4.lookAt(eye,target,[0,1,0]); const proj=M4.perspective(active.fov,canvas.width/canvas.height,.1,180); gl.viewport(0,0,canvas.width,canvas.height); gl.enable(gl.DEPTH_TEST); gl.disable(gl.CULL_FACE); gl.clearColor(active.background[0],active.background[1],active.background[2],1); gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT); let drawn=0; for(const o of active.objects){ if(o.floor && eye[1]<.02) continue; draw(o,view,proj,eye); drawn++; } stats.textContent=`\${drawn} draw items`; requestAnimationFrame(render); }
  canvas.addEventListener("pointerdown",e=>{ dragging=true; lx=e.clientX; ly=e.clientY; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener("pointermove",e=>{ if(!dragging)return; const dx=e.clientX-lx, dy=e.clientY-ly; lx=e.clientX; ly=e.clientY; yaw+=dx*.008; pitch=Math.max(-1.35,Math.min(1.35,pitch+dy*.006)); });
  canvas.addEventListener("pointerup",()=>dragging=false); canvas.addEventListener("pointercancel",()=>dragging=false);
  canvas.addEventListener("wheel",e=>{ e.preventDefault(); dist=Math.max(2.5,Math.min(24,dist*(1+Math.sign(e.deltaY)*.08))); },{passive:false});
  setCase(active.id); requestAnimationFrame(render);
  </script>
</body>
</html>
"""
end

"""
    save_webgl_html(path, cases; title="Three.jl Live WebGL Showcase")

Export one or more `WebGLExportCase`s to a standalone interactive HTML file.
The browser runtime is intentionally small; the scene data is produced from
Three.jl objects, materials, instancing, and optional `AnimationClip`s.
"""
function save_webgl_html(path::String, cases::AbstractVector{WebGLExportCase};
                         title::String="Three.jl Live WebGL Showcase")
    isempty(cases) && throw(ArgumentError("save_webgl_html requires at least one WebGLExportCase"))
    data = "{\"cases\":[" * join((_web_case_json(c) for c in cases), ",") * "]}"
    open(path, "w") do io
        write(io, _webgl_html(data, title))
    end
    return path
end
