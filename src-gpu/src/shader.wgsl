struct Camera {
    origin: vec4<f32>,
    lower_left_corner: vec4<f32>,
    horizontal: vec4<f32>,
    vertical: vec4<f32>,
    u: vec4<f32>,
    v: vec4<f32>,
    lens_radius: f32,
    _pc0: f32, _pc1: f32, _pc2: f32,
}

struct Sphere {
    center: vec4<f32>,
    radius: f32,
    material_id: u32,
    _ps0: u32, _ps1: u32,
}

struct Triangle {
    v0: vec4<f32>,
    v1: vec4<f32>,
    v2: vec4<f32>,
    n0: vec4<f32>,
    n1: vec4<f32>,
    n2: vec4<f32>,
    uv0: vec2<f32>,
    uv1: vec2<f32>,
    uv2: vec2<f32>,
    material_id: u32,
    _pad: u32,
}

struct BvhNode {
    bbox_min: vec3<f32>,
    left_or_first: u32,
    bbox_max: vec3<f32>,
    primitive_count: u32,
}

struct Material {
    albedo: vec4<f32>,
    fuzz: f32,
    ref_idx: f32,
    material_type: u32,
    tex_id: u32,
}

struct GpuTexture {
    data_offset: u32, width: u32, height: u32, _pt0: u32,
}

struct Uniforms {
    image_width: u32,
    image_height: u32,
    samples_per_pixel: u32,
    max_depth: u32,
    sphere_count: u32,
    triangle_count: u32,
    bvh_node_count: u32,
    light_count: u32,
    background: vec4<f32>,
    tex_count: u32,
    batch_offset: u32,
    batch_count: u32,
    tile_start_x: u32,
    tile_start_y: u32,
    tile_end_x: u32,
    tile_end_y: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> cam: Camera;
@group(0) @binding(2) var<storage, read> u: Uniforms;
@group(0) @binding(3) var<storage, read> spheres: array<Sphere>;
@group(0) @binding(4) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(5) var<storage, read> bvh_nodes: array<BvhNode>;
@group(0) @binding(6) var<storage, read> materials: array<Material>;
@group(0) @binding(7) var<storage, read> lights: array<u32>;
@group(0) @binding(8) var<storage, read> textures: array<GpuTexture>;
@group(0) @binding(9) var<storage, read> tex_data: array<f32>;

fn rand(seed: ptr<function, u32>) -> f32 {
    *seed = *seed * 1664525u + 1013904223u;
    return f32(*seed & 0x00FFFFFFu) / f32(0x01000000u);
}

fn rand_in_disk(seed: ptr<function, u32>) -> vec2<f32> {
    for (var i: u32 = 0u; i < 100u; i++) {
        let x = rand(seed) * 2.0 - 1.0;
        let y = rand(seed) * 2.0 - 1.0;
        if x * x + y * y < 1.0 { return vec2<f32>(x, y); }
    }
    return vec2<f32>(0.0);
}

fn rand_unit_vec(seed: ptr<function, u32>) -> vec3<f32> {
    for (var i: u32 = 0u; i < 100u; i++) {
        let x = rand(seed)*2.0-1.0; let y = rand(seed)*2.0-1.0; let z = rand(seed)*2.0-1.0;
        let l2 = x*x + y*y + z*z;
        if l2 > 0.0001 && l2 < 1.0 {
            let inv = 1.0 / sqrt(l2);
            return vec3<f32>(x*inv, y*inv, z*inv);
        }
    }
    // Fallback: use seed-driven direction instead of hardcoded (0,1,0)
    let a = rand(seed) * 6.28318530718;
    let b = rand(seed) * 2.0 - 1.0;
    let sb = sqrt(1.0 - b*b);
    return vec3<f32>(cos(a)*sb, sin(a)*sb, b);
}

fn rand_cosine_dir(seed: ptr<function, u32>) -> vec3<f32> {
    let r1 = rand(seed); let r2 = rand(seed);
    let z = sqrt(1.0 - r2);
    let phi = 6.28318530718 * r1;
    return vec3<f32>(cos(phi) * sqrt(r2), sin(phi) * sqrt(r2), z);
}

fn reflect(v: vec3<f32>, n: vec3<f32>) -> vec3<f32> { return v - 2.0 * dot(v, n) * n; }

fn refract2(uv: vec3<f32>, n: vec3<f32>, eta: f32) -> vec3<f32> {
    let ct = min(dot(-uv, n), 1.0);
    let rp = eta * (uv + ct * n);
    return rp - sqrt(abs(1.0 - dot(rp, rp))) * n;
}

fn onb_local(n: vec3<f32>, a: vec3<f32>) -> vec3<f32> {
    let w = normalize(n);
    // abs(w.x) > 0.9 → normal near X → use Y as auxiliary (NOT X!)
    let av = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(w.x) > 0.9);
    let v = normalize(cross(w, av));
    let u = cross(w, v);
    return a.x * u + a.y * v + a.z * w;
}

fn aabb_hit(ro: vec3<f32>, rd: vec3<f32>, tmin: f32, tmax: f32, bmin: vec3<f32>, bmax: vec3<f32>) -> bool {
    var tmn = tmin; var tmx = tmax;
    for (var a: u32 = 0u; a < 3u; a++) {
        let inv = 1.0 / rd[a];
        var t0 = (bmin[a] - ro[a]) * inv;
        var t1 = (bmax[a] - ro[a]) * inv;
        if inv < 0.0 { let t = t0; t0 = t1; t1 = t; }
        tmn = max(t0, tmn); tmx = min(t1, tmx);
        if tmx <= tmn { return false; }
    }
    return true;
}

fn hit_sphere(ro: vec3<f32>, rd: vec3<f32>, tmin: f32, tmax: f32, s: Sphere) -> f32 {
    let oc = ro - s.center.xyz;
    let a = dot(rd, rd);
    let hb = dot(oc, rd);
    let c = dot(oc, oc) - s.radius * s.radius;
    let d = hb * hb - a * c;
    if d < 0.0 { return tmax; }
    let sd = sqrt(d);
    var root = (-hb - sd) / a;
    if root < tmin || root > tmax { root = (-hb + sd) / a; if root < tmin || root > tmax { return tmax; } }
    return root;
}

fn hit_tri(ro: vec3<f32>, rd: vec3<f32>, tmin: f32, tmax: f32, tri: Triangle,
            uv_out: ptr<function, vec2<f32>>, norm_out: ptr<function, vec3<f32>>) -> f32 {
    let e1 = tri.v1.xyz - tri.v0.xyz;
    let e2 = tri.v2.xyz - tri.v0.xyz;
    let h = cross(rd, e2);
    let a = dot(e1, h);
    if abs(a) < 1e-12 { return tmax; }
    let f = 1.0 / a;
    let s = ro - tri.v0.xyz;
    let bu = f * dot(s, h);
    if bu < -1e-6 || bu > 1.0 + 1e-6 { return tmax; }
    let q = cross(s, e1);
    let bv = f * dot(rd, q);
    if bv < -1e-6 || bu + bv > 1.0 + 1e-6 { return tmax; }
    // Use raw barycentrics (already validated by edge checks above)
    let bw = 1.0 - bu - bv;
    *uv_out = tri.uv0 * bw + tri.uv1 * bu + tri.uv2 * bv;
    *norm_out = normalize(tri.n0.xyz * bw + tri.n1.xyz * bu + tri.n2.xyz * bv);
    let t = f * dot(e2, q);
    if t < tmin || t > tmax { return tmax; }
    return t;
}

struct Hit { t: f32, n: vec3<f32>, ff: bool, mid: u32, uv: vec2<f32>, }

fn hit_world(ro: vec3<f32>, rd: vec3<f32>, tmin: f32, tmax: f32) -> Hit {
    var result: Hit;
    result.t = tmax; result.mid = 0u; result.n = vec3<f32>(0.0); result.ff = false;

    if u.bvh_node_count > 0u {
        var stack: array<u32, 64>;
        var sp: u32 = 0u;
        stack[sp] = 0u; sp = 1u;
        for (var iter: u32 = 0u; iter < 50000u && sp > 0u; iter++) {
            sp = sp - 1u;
            let ni = stack[sp];
            let node = bvh_nodes[ni];
            if !aabb_hit(ro, rd, tmin, result.t, node.bbox_min, node.bbox_max) { continue; }
            if (node.primitive_count & 0x80000000u) != 0u {
                // Internal node: left_or_first = left child index, primitive_count low bits = right child index
                let left = node.left_or_first;
                let right = node.primitive_count & 0x7FFFFFFFu;
                stack[sp] = right; sp = sp + 1u;
                stack[sp] = left; sp = sp + 1u;
            } else {
                // Leaf node: left_or_first has primitive index, primitive_count = count
                let first = node.left_or_first;
                let count = node.primitive_count;
                let is_tri = (first >> 31u) != 0u;
                let idx = first & 0x7FFFFFFFu;
                if !is_tri {
                    for (var i: u32 = 0u; i < count; i++) {
                        let s = spheres[idx + i];
                        let t = hit_sphere(ro, rd, tmin, result.t, s);
                        if t < result.t {
                            result.t = t; result.mid = s.material_id;
                            let p = ro + t * rd;
                            let on = (p - s.center.xyz) / s.radius;
                            result.ff = dot(rd, on) < 0.0;
                            result.n = select(-on, on, result.ff);
                        }
                    }
                } else {
                    var uv: vec2<f32>;
                    var norm: vec3<f32>;
                    for (var i: u32 = 0u; i < count; i++) {
                        let tri = triangles[idx + i];
                        let t = hit_tri(ro, rd, tmin, result.t, tri, &uv, &norm);
                        if t < result.t {
                            result.t = t; result.mid = tri.material_id;
                            result.ff = dot(rd, norm) < 0.0;
                            result.n = select(-norm, norm, result.ff);
                            result.uv = uv;
                        }
                    }
                }
            }
        }
    } else {
        for (var i: u32 = 0u; i < u.sphere_count; i++) {
            let s = spheres[i];
            let t = hit_sphere(ro, rd, tmin, result.t, s);
            if t < result.t {
                result.t = t; result.mid = s.material_id;
                let p = ro + t * rd;
                let on = (p - s.center.xyz) / s.radius;
                result.ff = dot(rd, on) < 0.0;
                result.n = select(-on, on, result.ff);
                result.uv = vec2<f32>(0.0);
            }
        }
        var uv2: vec2<f32>;
        var norm2: vec3<f32>;
        for (var i: u32 = 0u; i < u.triangle_count; i++) {
            let tri = triangles[i];
            let t = hit_tri(ro, rd, tmin, result.t, tri, &uv2, &norm2);
            if t < result.t {
                result.t = t; result.mid = tri.material_id;
                result.ff = dot(rd, norm2) < 0.0;
                result.n = select(-norm2, norm2, result.ff);
                result.uv = uv2;
            }
        }
    }
    return result;
}

fn sample_texture(tex_id: u32, uv_input: vec2<f32>) -> vec3<f32> {
    if tex_id == 0u { return vec3<f32>(1.0); }
    let tex = textures[tex_id - 1u];
    let w = tex.width;
    let h = tex.height;
    var uu = uv_input.x - floor(uv_input.x);
    var vv = uv_input.y - floor(uv_input.y);
    var px = uu * f32(w) - 0.5;
    var py = vv * f32(h) - 0.5;
    // Wrap negative values to avoid large-u32 precision loss
    if px < 0.0 { px = px + f32(w); }
    if py < 0.0 { py = py + f32(h); }
    let xi = u32(px);
    let yi = u32(py);
    let fx = px - f32(xi);
    let fy = py - f32(yi);
    let x0 = xi % w;
    let x1 = (x0 + 1u) % w;
    let y0 = yi % h;
    let y1 = (y0 + 1u) % h;
    let offset = tex.data_offset;
    let row_stride = w * 3u;
    let i00 = offset + y0 * row_stride + x0 * 3u;
    let i10 = offset + y0 * row_stride + x1 * 3u;
    let i01 = offset + y1 * row_stride + x0 * 3u;
    let i11 = offset + y1 * row_stride + x1 * 3u;
    let c00 = vec3<f32>(tex_data[i00], tex_data[i00+1u], tex_data[i00+2u]);
    let c10 = vec3<f32>(tex_data[i10], tex_data[i10+1u], tex_data[i10+2u]);
    let c01 = vec3<f32>(tex_data[i01], tex_data[i01+1u], tex_data[i01+2u]);
    let c11 = vec3<f32>(tex_data[i11], tex_data[i11+1u], tex_data[i11+2u]);
    return mix(mix(c00, c10, fx), mix(c01, c11, fx), fy);
}

fn ray_color(ro: vec3<f32>, rd: vec3<f32>, seed: ptr<function, u32>) -> vec3<f32> {
    var col = vec3<f32>(0.0);
    var thr = vec3<f32>(1.0);
    var ro2 = ro; var rd2 = rd;

    for (var d: u32 = 0u; d < u.max_depth; d++) {
        let hit = hit_world(ro2, rd2, 0.001, 1e30);
        if hit.t >= 1e29 {
            let tb = 0.5 * (normalize(rd2).y + 1.0);
            col = col + thr * mix(vec3<f32>(0.5,0.7,0.9), vec3<f32>(0.95,0.95,1.0), tb);
            break;
        }
        let mat = materials[hit.mid];
        let p = ro2 + hit.t * rd2;
        let mt = mat.material_type;
        if mt == 3u { col = col + thr * mat.albedo.xyz; break; }
        if mt == 2u {
            let tex_color = sample_texture(mat.tex_id, hit.uv);
            let surface_color = mat.albedo.xyz * tex_color;
            var eta = 1.0 / mat.ref_idx; if !hit.ff { eta = mat.ref_idx; }
            let ud = normalize(rd2);
            let ct = min(dot(-ud, hit.n), 1.0);
            let st = sqrt(1.0 - ct * ct);
            var dir: vec3<f32>;
            if eta * st > 1.0 { dir = reflect(ud, hit.n); }
            else {
                let r0 = (1.0-eta)/(1.0+eta); let r0s = r0 * r0;
                if r0s + (1.0-r0s) * pow(1.0-ct, 5.0) > rand(seed) { dir = reflect(ud, hit.n); }
                else { dir = refract2(ud, hit.n, eta); }
            }
            rd2 = dir; ro2 = p + dir * 0.001;
            thr = thr * mix(vec3<f32>(1.0), surface_color, mat.fuzz);
            continue;
        }
        if mt == 1u {
            let tex_color = sample_texture(mat.tex_id, hit.uv);
            let surface_color = mat.albedo.xyz * tex_color;
            let refl = reflect(normalize(rd2), hit.n);
            let fuzz_val = max(mat.fuzz, 0.0001);
            let sdir = refl + fuzz_val * rand_unit_vec(seed);
            if dot(sdir, hit.n) <= 0.0 { break; }
            rd2 = sdir; thr = thr * surface_color; ro2 = p + hit.n * 0.002; continue;
        }
        // ── Diffuse (mt == 0u): standard path tracing loop ──
        let tex_color = sample_texture(mat.tex_id, hit.uv);
        let surface_color = mat.albedo.xyz * tex_color;

        // Step A: NEE — sample one random light per bounce
        if u.light_count > 0u {
            let li = u32(rand(seed) * f32(u.light_count)) % u.light_count;
            let ls = spheres[lights[li]];
            let lmat = materials[ls.material_id];
            let rnd_pt = rand_unit_vec(seed);
            let light_pt = ls.center.xyz + rnd_pt * ls.radius;
            let to_light = light_pt - p;
            let dist_sq = dot(to_light, to_light);
            let dist = sqrt(dist_sq);
            let wi = to_light / dist;
            let cos_emitter = max(abs(dot(rnd_pt, -wi)), 0.001);
            let dist_to_center = length(ls.center.xyz - p);
            let tmax_shadow = max(dist_to_center - ls.radius - 0.001, 0.001);
            let lhit = hit_world(p + hit.n * 0.01, wi, 0.001, tmax_shadow);
            if lhit.t >= tmax_shadow - 0.001 {
                let cos_recv = max(dot(hit.n, wi), 0.0);
                let area = 4.0 * 3.14159265359 * ls.radius * ls.radius;
                let light_pdf = (dist_sq / (cos_emitter * area)) / f32(u.light_count);
                let bsdf_pdf = cos_recv / 3.14159265359;
                let mis_w = light_pdf / (light_pdf + bsdf_pdf); // balance heuristic
                if light_pdf > 0.0 {
                    col = col + thr * surface_color * lmat.albedo.xyz * cos_recv
                        / (3.14159265359 * light_pdf) * mis_w;
                }
            }
        }

        // Step B: MixturePdf scatter — 50% light-guided, 50% cosine (matching CPU)
        var sdir: vec3<f32>;
        if u.light_count > 0u && rand(seed) < 0.5 {
            // HittablePdf: sample toward a random light
            let li2 = u32(rand(seed) * f32(u.light_count)) % u.light_count;
            let ls2 = spheres[lights[li2]];
            let rp = rand_unit_vec(seed);
            let lp2 = ls2.center.xyz + rp * ls2.radius;
            let tl = lp2 - p;
            let dsq = dot(tl, tl);
            if dsq < 0.0001 {
                sdir = onb_local(hit.n, rand_cosine_dir(seed));
                thr = thr * surface_color;
            } else {
                let d = sqrt(dsq);
                sdir = tl / d;
                let cr2 = max(dot(hit.n, sdir), 0.001);
                let ce2 = max(abs(dot(rp, -sdir)), 0.001);
                let area2 = 4.0 * 3.14159265359 * ls2.radius * ls2.radius;
                let light_pdf2 = (dsq / (ce2 * area2)) / f32(u.light_count);
                let cosine_pdf2 = cr2 / 3.14159265359;
                let mix_pdf2 = max(0.5 * light_pdf2 + 0.5 * cosine_pdf2, 0.0001);
                thr = thr * surface_color * cr2 / (3.14159265359 * mix_pdf2);
            }
        } else {
            sdir = onb_local(hit.n, rand_cosine_dir(seed));
            thr = thr * surface_color;
        }
        if dot(sdir, hit.n) < 0.001 { sdir = hit.n; }
        thr = max(thr, vec3<f32>(0.0));
        rd2 = sdir; ro2 = p + hit.n * 0.01;
    }
    return col;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x + u.tile_start_x;
    let y = gid.y + u.tile_start_y;
    if x >= u.tile_end_x || y >= u.tile_end_y { return; }

    // Stratified sampling: divide pixel into grid, one sample per cell
    let grid_dim = u32(floor(sqrt(f32(u.samples_per_pixel))));
    let grid_count = grid_dim * grid_dim;  // samples covered by stratification

    var seed: u32 = x * 1973u + y * 9277u + 2663u;
    // Advance seed to match absolute sample index (one sample = 4 LCG advances)
    for (var i: u32 = 0u; i < u.batch_offset; i++) {
        seed = seed * 1664525u + 1013904223u;
        seed = seed * 1664525u + 1013904223u;
        seed = seed * 1664525u + 1013904223u;
        seed = seed * 1664525u + 1013904223u;
    }

    var acc = vec3<f32>(0.0);
    for (var bi: u32 = 0u; bi < u.batch_count; bi++) {
        let abs_s = bi + u.batch_offset;  // absolute sample index (0..samples_per_pixel-1)

        // Advance seed deterministically for this absolute sample
        seed = seed * 1664525u + 1013904223u;
        let rx = rand(&seed);
        seed = seed * 1664525u + 1013904223u;
        let ry = rand(&seed);

        var du: f32;
        var dv: f32;
        if abs_s < grid_count {
            // Stratified: grid cell with random jitter inside
            let gx = abs_s % grid_dim;
            let gy = abs_s / grid_dim;
            du = (f32(x) + (f32(gx) + rx) / f32(grid_dim)) / f32(u.image_width - 1u);
            dv = (f32(u.image_height - 1u - y) + (f32(gy) + ry) / f32(grid_dim)) / f32(u.image_height - 1u);
        } else {
            // Extra samples beyond grid: full-pixel random jitter
            du = (f32(x) + rx) / f32(u.image_width - 1u);
            dv = (f32(u.image_height - 1u - y) + ry) / f32(u.image_height - 1u);
        }

        let orig = cam.origin.xyz;
        let dir = normalize(cam.lower_left_corner.xyz + du * cam.horizontal.xyz + dv * cam.vertical.xyz - cam.origin.xyz);
        acc = acc + ray_color(orig, dir, &seed);
    }
    let idx = y * u.image_width + x;
    output[idx] = output[idx] + vec4<f32>(acc.x, acc.y, acc.z, 0.0); // restore normal
}
