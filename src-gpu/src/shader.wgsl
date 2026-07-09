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

struct GpuParticle {
    pos_t0: vec4<f32>,
    pos_t1: vec4<f32>,
    radius: f32,
    material_id: u32,
    _pad: vec2<u32>,
}

struct GpuTexture {
    data_offset: u32, width: u32, height: u32, channels: u32,
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
    particle_count: u32,
    particle_offset: u32,
    background: vec4<f32>,
    tex_count: u32,
    batch_offset: u32,
    batch_count: u32,
    tile_start_x: u32,
    tile_start_y: u32,
    tile_end_x: u32,
    tile_end_y: u32,
    sun_dir: vec4<f32>,
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
@group(0) @binding(10) var<storage, read> particles: array<GpuParticle>;

fn hash_u32(x: u32) -> u32 {
    var a = x;
    a = (a ^ 61u) ^ (a >> 16u);
    a = a + (a << 3u);
    a = a ^ (a >> 4u);
    a = a * 0x27d4eb2du;
    a = a ^ (a >> 15u);
    return a;
}

fn rand(seed: ptr<function, u32>) -> f32 {
    *seed = *seed * 1664525u + 1013904223u;
    return f32(*seed >> 8u) / 16777216.0;
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

fn water_wave_normal(p: vec3<f32>, base_n: vec3<f32>) -> vec3<f32> {
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let av = select(up, vec3<f32>(1.0, 0.0, 0.0), abs(base_n.y) > 0.999);
    let t = normalize(cross(av, base_n));
    let b = cross(base_n, t);
    let coord = p * 0.4;
    let w1 = sin(coord.x*3.7 + coord.z*2.3) * cos(coord.z*4.1 - coord.x*1.7) * 0.022;
    let w2 = sin(coord.x*7.1 + coord.z*5.3 + 0.7) * cos(coord.z*8.2 - coord.x*3.9 + 1.1) * 0.009;
    let w3 = sin(coord.x*13.4 + coord.z*9.7 + 1.3) * cos(coord.z*11.8 - coord.x*7.2 + 0.4) * 0.003;
    let perturb = t * (w1 + w2 + w3) + b * (w1*0.7 + w2*0.8 + w3*0.9);
    return normalize(base_n + perturb);
}

fn onb_local(n: vec3<f32>, a: vec3<f32>) -> vec3<f32> {
    let w = normalize(n);
    let av = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(w.x) > 0.9);
    let v = normalize(cross(w, av));
    let u = cross(w, v);
    return a.x * u + a.y * v + a.z * w;
}

// Returns distance to AABB entry, or 1e30 if miss
fn aabb_hit_dist(ro: vec3<f32>, rd: vec3<f32>, tmin: f32, tmax: f32, bmin: vec3<f32>, bmax: vec3<f32>) -> f32 {
    var tmn = tmin; var tmx = tmax;
    for (var a: u32 = 0u; a < 3u; a++) {
        if abs(rd[a]) < 1e-12 {
            if ro[a] < bmin[a] || ro[a] > bmax[a] { return 1e30; }
            continue;
        }
        let inv = 1.0 / rd[a];
        var t0 = (bmin[a] - ro[a]) * inv;
        var t1 = (bmax[a] - ro[a]) * inv;
        if inv < 0.0 { let t = t0; t0 = t1; t1 = t; }
        tmn = max(t0, tmn); tmx = min(t1, tmx);
        if tmx <= tmn { return 1e30; }
    }
    return tmn;
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

fn hit_particle(ro: vec3<f32>, rd: vec3<f32>, tmin: f32, tmax: f32, p_idx: u32, r_time: f32) -> f32 {
    let p = particles[p_idx - u.particle_offset];
    let center = mix(p.pos_t0.xyz, p.pos_t1.xyz, r_time);
    let oc = ro - center;
    let a = dot(rd, rd);
    let hb = dot(oc, rd);
    let c = dot(oc, oc) - p.radius * p.radius;
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
    let bw = 1.0 - bu - bv;
    *uv_out = tri.uv0 * bw + tri.uv1 * bu + tri.uv2 * bv;
    *norm_out = normalize(tri.n0.xyz * bw + tri.n1.xyz * bu + tri.n2.xyz * bv);
    let t = f * dot(e2, q);
    if t < tmin || t > tmax { return tmax; }
    return t;
}

struct Hit { t: f32, n: vec3<f32>, ff: bool, mid: u32, uv: vec2<f32>, }

fn hit_world(ro: vec3<f32>, rd: vec3<f32>, tmin: f32, tmax: f32, r_time: f32) -> Hit {
    var result: Hit;
    result.t = tmax; result.mid = 0u; result.n = vec3<f32>(0.0); result.ff = false;

    if u.bvh_node_count > 0u {
        var stack: array<u32, 256>;
        var sp: u32 = 0u;
        let root_node = bvh_nodes[0u];
        if aabb_hit_dist(ro, rd, tmin, result.t, root_node.bbox_min, root_node.bbox_max) < 1e29 {
            stack[sp] = 0u; sp = 1u;
        }
        for (var iter: u32 = 0u; iter < 2000u && sp > 0u; iter++) {
            sp = sp - 1u;
            let ni = stack[sp];
            let node = bvh_nodes[ni];
            if (node.primitive_count & 0x80000000u) != 0u {
                let left = node.left_or_first;
                let right = node.primitive_count & 0x7FFFFFFFu;
                let left_node = bvh_nodes[left];
                let right_node = bvh_nodes[right];
                let d_left = aabb_hit_dist(ro, rd, tmin, result.t, left_node.bbox_min, left_node.bbox_max);
                let d_right = aabb_hit_dist(ro, rd, tmin, result.t, right_node.bbox_min, right_node.bbox_max);
                if d_left < 1e29 && d_right < 1e29 {
                    if d_left < d_right {
                        stack[sp] = right; sp = sp + 1u;
                        stack[sp] = left; sp = sp + 1u;
                    } else {
                        stack[sp] = left; sp = sp + 1u;
                        stack[sp] = right; sp = sp + 1u;
                    }
                } else if d_left < 1e29 {
                    stack[sp] = left; sp = sp + 1u;
                } else if d_right < 1e29 {
                    stack[sp] = right; sp = sp + 1u;
                }
            } else {
                let first = node.left_or_first;
                let count = node.primitive_count;
                let is_tri = (first >> 31u) != 0u;
                let idx = first & 0x7FFFFFFFu;
                if !is_tri {
                    for (var i: u32 = 0u; i < count; i++) {
                        let si = idx + i;
                        let s = spheres[si];
                        var t: f32;
                        if si >= u.particle_offset && si < u.particle_offset + u.particle_count {
                            t = hit_particle(ro, rd, tmin, result.t, si, r_time);
                        } else {
                            t = hit_sphere(ro, rd, tmin, result.t, s);
                        }
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
                            let mat = materials[tri.material_id];
                            if sample_texture_alpha(mat.tex_id, uv) < 0.5 { continue; }
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
            var t: f32;
            if i >= u.particle_offset && i < u.particle_offset + u.particle_count {
                t = hit_particle(ro, rd, tmin, result.t, i, r_time);
            } else {
                t = hit_sphere(ro, rd, tmin, result.t, s);
            }
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
                let mat = materials[tri.material_id];
                if sample_texture_alpha(mat.tex_id, uv2) < 0.5 { continue; }
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
    let ch = tex.channels;
    var uu = uv_input.x - floor(uv_input.x);
    var vv = uv_input.y - floor(uv_input.y);
    var px = uu * f32(w) - 0.5;
    var py = vv * f32(h) - 0.5;
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
    let row_stride = w * ch;
    let i00 = offset + y0 * row_stride + x0 * ch;
    let i10 = offset + y0 * row_stride + x1 * ch;
    let i01 = offset + y1 * row_stride + x0 * ch;
    let i11 = offset + y1 * row_stride + x1 * ch;
    let c00 = vec3<f32>(tex_data[i00], tex_data[i00+1u], tex_data[i00+2u]);
    let c10 = vec3<f32>(tex_data[i10], tex_data[i10+1u], tex_data[i10+2u]);
    let c01 = vec3<f32>(tex_data[i01], tex_data[i01+1u], tex_data[i01+2u]);
    let c11 = vec3<f32>(tex_data[i11], tex_data[i11+1u], tex_data[i11+2u]);
    return mix(mix(c00, c10, fx), mix(c01, c11, fx), fy);
}

fn sample_texture_alpha(tex_id: u32, uv_input: vec2<f32>) -> f32 {
    if tex_id == 0u { return 1.0; }
    let tex = textures[tex_id - 1u];
    if tex.channels < 4u { return 1.0; }
    let w = tex.width;
    let h = tex.height;
    var uu = uv_input.x - floor(uv_input.x);
    var vv = uv_input.y - floor(uv_input.y);
    if uu < 0.0 { uu = uu + 1.0; }
    if vv < 0.0 { vv = vv + 1.0; }
    let xi = u32(uu * f32(w)) % w;
    let yi = u32(vv * f32(h)) % h;
    let idx = tex.data_offset + yi * w * 4u + xi * 4u + 3u;
    return tex_data[idx];
}

fn hash3(p: vec3<f32>) -> f32 {
    let h = clamp(dot(p, vec3<f32>(127.1, 311.7, 74.7)), -1e6, 1e6);
    let s = sin(h) * 43758.5453;
    return select(0.5, fract(s), abs(s) < 1e10);
}

fn value_noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(hash3(i), hash3(i+vec3<f32>(1,0,0)), u.x),
            mix(hash3(i+vec3<f32>(0,1,0)), hash3(i+vec3<f32>(1,1,0)), u.x), u.y),
        mix(mix(hash3(i+vec3<f32>(0,0,1)), hash3(i+vec3<f32>(1,0,1)), u.x),
            mix(hash3(i+vec3<f32>(0,1,1)), hash3(i+vec3<f32>(1,1,1)), u.x), u.y),
        u.z);
}

fn fbm(p: vec3<f32>, octaves: u32) -> f32 {
    var val = 0.0; var amp = 0.5; var freq = 1.0; var acc = 0.0;
    for (var i = 0u; i < octaves; i++) {
        val += amp * value_noise(p * freq);
        acc += amp; freq *= 2.0; amp *= 0.5;
    }
    return val / acc;
}

fn bump_normal(p: vec3<f32>, n: vec3<f32>, field_val: f32, eps: f32, strength: f32) -> vec3<f32> {
    let dx = field_val - fbm(p - vec3<f32>(eps,0,0), 4u);
    let dy = field_val - fbm(p - vec3<f32>(0,eps,0), 4u);
    let dz = field_val - fbm(p - vec3<f32>(0,0,eps), 4u);
    let av = select(vec3<f32>(0,1,0), vec3<f32>(1,0,0), abs(n.y) > 0.999);
    let t = normalize(cross(av, n));
    let b = cross(n, t);
    let grad = vec3<f32>(dx, dy, dz);
    return normalize(n - (t*dot(grad,t) + b*dot(grad,b)) * (strength / eps));
}

fn stone_field(p: vec3<f32>) -> f32 {
    let coarse = fbm(p * 3.0, 3u);
    let fine   = fbm(p * 11.0 + vec3<f32>(1.7), 3u);
    let cracks = fbm(p * 25.0 + vec3<f32>(3.1), 2u);
    return coarse * 0.5 + fine * 0.3 + cracks * 0.2;
}

fn wood_field(p: vec3<f32>) -> f32 {
    // Isotropic wood grain — multi-scale FBM for rich surface detail
    let base  = fbm(p * 2.5, 3u);                          // coarse grain
    let grain = fbm(p * 8.0  + vec3<f32>(1.7, 2.3, 3.1), 3u); // medium detail
    let micro = fbm(p * 22.0 + vec3<f32>(5.1, 1.3, 4.7), 2u);  // fine pores
    return base * 0.35 + grain * 0.40 + micro * 0.25;
}

fn wood_roughness(p: vec3<f32>) -> f32 { return mix(0.45, 0.85, wood_field(p)); }

fn sample_hg(wo: vec3<f32>, g: f32, seed: ptr<function, u32>) -> vec3<f32> {
    let xi = rand(seed);
    var cos_theta: f32;
    if abs(g) < 0.001 {
        cos_theta = 1.0 - 2.0 * xi;
    } else {
        let g2 = g * g;
        let term = (1.0 - g2) / (1.0 - g + 2.0 * g * xi);
        cos_theta = (1.0 + g2 - term * term) / (2.0 * max(g, 0.001));
    }
    let sin_theta = sqrt(max(1.0 - cos_theta*cos_theta, 0.0));
    let phi = 6.28318530718 * rand(seed);
    let av = select(vec3<f32>(0,1,0), vec3<f32>(1,0,0), abs(wo.y) > 0.999);
    let t = normalize(cross(av, wo));
    let b = cross(wo, t);
    return wo * cos_theta + t * sin_theta * cos(phi) + b * sin_theta * sin(phi);
}

fn sky_color(rd: vec3<f32>) -> vec3<f32> {
    let sun_dir = normalize(u.sun_dir.xyz);
    let y = rd.y;

    let zenith   = vec3<f32>(0.015, 0.025, 0.12);
    let mid_sky  = vec3<f32>(0.55, 0.12, 0.28);
    let horizon  = vec3<f32>(0.98, 0.42, 0.12);
    let r_dark   = vec3<f32>(0.12, 0.04, 0.03);

    var sky_base: vec3<f32>;
    if y > -0.15 {
        let t = (y + 0.15) / 1.15;
        let mid_stop = 0.38;
        if t < mid_stop {
            let local_t = smoothstep(0.0, mid_stop, t);
            sky_base = mix(horizon, mid_sky, local_t);
        } else {
            let local_t = smoothstep(mid_stop, 1.0, t);
            sky_base = mix(mid_sky, zenith, local_t);
        }
    } else {
        let t_down = smoothstep(0.0, 1.0, clamp((-y - 0.15) * 2.5, 0.0, 1.0));
        sky_base = mix(horizon, r_dark, t_down);
    }

    let sun_cos = max(dot(rd, sun_dir), 0.0);
    let glow_wide = pow(sun_cos, 6.0) * vec3<f32>(1.0, 0.3, 0.05) * 2.0;
    let sun_disk  = pow(sun_cos, 400.0) * vec3<f32>(1.0, 0.85, 0.4) * 15.0;

    return sky_base * 0.25 + glow_wide * 0.35 + sun_disk;
}

fn ray_color(ro: vec3<f32>, rd: vec3<f32>, r_time: f32, seed: ptr<function, u32>) -> vec3<f32> {
    var col = vec3<f32>(0.0);
    var thr = vec3<f32>(1.0);
    var ro2 = ro; var rd2 = rd;
    var in_water = false;

    for (var d: u32 = 0u; d < u.max_depth; d++) {
        let hit = hit_world(ro2, rd2, 0.001, 1e30, r_time);

        if in_water {
            let sigma = vec3<f32>(0.35, 0.06, 0.025);
            if hit.t < 1e29 {
                thr = thr * exp(-sigma * min(hit.t, 20.0));
            } else {
                col = col + thr * sky_color(normalize(rd2)) * exp(-sigma * 5.0);
                break;
            }
        }
        in_water = false;

        if hit.t >= 1e29 {
            col = col + thr * sky_color(normalize(rd2));
            break;
        }
        let mat = materials[hit.mid];
        let p = ro2 + hit.t * rd2;
        let mt = mat.material_type;

        var use_n = hit.n;
        var use_fuzz = mat.fuzz;
        if mt == 5u {
            let fv = stone_field(p);
            use_n = bump_normal(p, hit.n, fv, 0.010, 0.04);
            use_fuzz = mix(0.4, 0.75, fv);
        }
        if mt == 6u {
            let wv = wood_field(p);
            use_n = bump_normal(p, hit.n, wv, 0.008, 0.03);
            use_fuzz = 0.70;
        }

        if mt == 3u {
            if d == 0u {
                var emit_color = mat.albedo.xyz;
                if dot(normalize(rd2), normalize(u.sun_dir.xyz)) > 0.95 {
                    emit_color = emit_color * vec3<f32>(1.0, 0.65, 0.22);
                }
                col = col + thr * emit_color;
            }
            break;
        }
        if mt == 2u {
            let is_water = mat.ref_idx > 1.3 && mat.ref_idx < 1.34;
            var use_n2 = hit.n;
            if is_water {
                use_n2 = water_wave_normal(p, hit.n);
            }

            let tex_color = sample_texture(mat.tex_id, hit.uv);
            let surface_color = mat.albedo.xyz * tex_color;
            var eta = 1.0 / mat.ref_idx; if !hit.ff { eta = mat.ref_idx; }
            let ud = normalize(rd2);
            let ct = min(dot(-ud, use_n2), 1.0);
            let st = sqrt(1.0 - ct * ct);
            var dir: vec3<f32>;
            var did_reflect = false;
            if eta * st > 1.0 { dir = reflect(ud, use_n2); did_reflect = true; }
            else {
                let r0 = (1.0-eta)/(1.0+eta); let r0s = r0 * r0;
                let fresnel = r0s + (1.0-r0s) * pow(1.0-ct, 5.0);
                if fresnel > rand(seed) { dir = reflect(ud, use_n2); did_reflect = true; }
                else { dir = refract2(ud, use_n2, eta); }
            }
            if is_water {
                if (!did_reflect && hit.ff) || (did_reflect && !hit.ff) {
                    in_water = true;
                }
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
        if mt == 4u {
            let tex_color = sample_texture(mat.tex_id, hit.uv);
            let base_color = mat.albedo.xyz * tex_color;
            let coat_rough = max(mat.fuzz, 0.005);
            let ud = normalize(rd2);
            let ct = abs(dot(ud, hit.n));
            let r0 = (mat.ref_idx - 1.0) / (mat.ref_idx + 1.0);
            let r02_phys = r0 * r0;
            let r02 = r02_phys;            // physical Fresnel only (no boost)
            let fresnel = r02 + (1.0 - r02) * pow(1.0 - ct, 5.0);
            if rand(seed) < fresnel {
                let refl = reflect(ud, hit.n);
                let sdir = refl + coat_rough * rand_unit_vec(seed);
                if dot(sdir, hit.n) <= 0.0 { break; }
                rd2 = sdir; thr = thr * mix(vec3<f32>(1.0), base_color, 0.88); ro2 = p + hit.n * 0.002; continue;
            }
            thr = thr * 0.92;  // minimal darkening — let the wood texture dominate
        }
        if mt == 7u {
            if rand(seed) < 0.15 {
                let ud2 = normalize(rd2);
                let refl = reflect(ud2, hit.n);
                let sheen_dir = normalize(refl + 0.65 * rand_unit_vec(seed));
                if dot(sheen_dir, hit.n) > 0.001 {
                    rd2 = sheen_dir; ro2 = p + hit.n * 0.002; continue;
                }
            }

            // ── Petal 3D: tangent-space pillow + per-petal random tilt ──
            let uv_center = vec2<f32>(0.5);
            let uv_dist = length(hit.uv - uv_center);
            let alpha_edge = 1.0 - smoothstep(0.0, 0.45, uv_dist);
            let thickness = mix(0.003, 0.035, alpha_edge);
            let pillow_offset = (hit.uv - uv_center) * 0.20;
            let hx = hash3(p * 50.0 + vec3<f32>(0.0, 1.7, 3.1));
            let hy = hash3(p * 50.0 + vec3<f32>(5.3, 0.0, 2.7));
            let local_perturb = vec3<f32>(pillow_offset.x + (hx-0.5)*0.25, pillow_offset.y + (hy-0.5)*0.25, 1.0);
            let petal_n = normalize(onb_local(hit.n, local_perturb));

            let tex_color = sample_texture(mat.tex_id, hit.uv);
            let base_color = mat.albedo.xyz * tex_color;
            let ior = 1.4;
            let ud = normalize(rd2);
            let ct_enter = abs(dot(ud, petal_n));
            let r0 = (ior-1.0)/(ior+1.0); let r02 = r0*r0;
            let fresnel = r02 + (1.0-r02)*pow(1.0-ct_enter, 5.0);

            if rand(seed) < fresnel {
                let refl = reflect(ud, petal_n);
                let rough_refl = refl + 0.2 * rand_unit_vec(seed);
                if dot(rough_refl, petal_n) <= 0.0 { break; }
                rd2 = normalize(rough_refl); thr = thr * base_color; ro2 = p + petal_n * 0.002; continue;
            }

            let sigma_a = vec3<f32>(0.001, 0.008, 0.003);  // minimal absorption
            let sigma_s = 6.0;   // long mean free path → 0-1 scatters → clear
            let sigma_t = sigma_s + sigma_a;
            let albedo_s = sigma_s / sigma_t;
            let sigma_t_avg = (sigma_t.x+sigma_t.y+sigma_t.z)/3.0;
            let inward_n = -petal_n;
            let g_hg = 0.7;

            var sss_thr = vec3<f32>(1.0);
            var sss_pos = p;
            var sss_dir = rd2;
            var escaped = false;
            var escaped_back = false;

            for (var sb: u32 = 0u; sb < 20u; sb++) {
                let d_free = -log(max(rand(seed), 0.0001)) / sigma_t_avg;
                sss_pos = sss_pos + d_free * sss_dir;
                let depth = dot(sss_pos - p, inward_n);

                if depth < 0.0 {
                    sss_pos = sss_pos - depth * inward_n;
                    escaped = true; escaped_back = false; break;
                }
                if depth > thickness {
                    sss_pos = sss_pos - (depth - thickness) * inward_n;
                    escaped = true; escaped_back = true; break;
                }

                sss_thr *= exp(-sigma_a * d_free) * albedo_s;

                let mp = max(sss_thr.x, max(sss_thr.y, sss_thr.z));
                if mp < 0.05 || rand(seed) > mp { break; }
                sss_thr /= mp;

                sss_dir = sample_hg(sss_dir, g_hg, seed);
            }

            if escaped {
                let exit_n = select(hit.n, -hit.n, escaped_back);
                ro2 = sss_pos + exit_n * 0.001;
                rd2 = normalize(sss_dir);
                thr = thr * sss_thr * base_color;
                if dot(rd2, exit_n) < 0.001 { rd2 = reflect(rd2, exit_n); }
            } else {
                col = col + thr * sss_thr * base_color * 0.1;
                break;
            }
            continue;
        }
        // ── Diffuse (mt == 0u or mt == 4u fallthrough) ──
        // Petal gaps: stochastic holes between clustered petals (diffuse only, not wood/stone)
        if mt == 0u && mat.fuzz > 0.0 && hash3(p * 30.0 + vec3<f32>(0.3, 0.7, 0.1)) > 0.55 {
            ro2 = p + rd2 * 0.01; continue;
        }
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
            let cos_raw = dot(use_n, wi);
            let petal_t = select(mat.fuzz, 0.0, mt == 5u || mt == 6u);
            var shadow_ro2 = p + use_n * 0.001;
            if cos_raw < 0.0 && petal_t > 0.0 {
                shadow_ro2 = p - use_n * 0.001;
            }
            let to_light_vec = light_pt - shadow_ro2;
            let dist_to_light = length(to_light_vec);
            let wi2 = to_light_vec / dist_to_light;
            let lhit = hit_world(shadow_ro2, wi2, 0.001, dist_to_light - 0.001, r_time);
            if lhit.t >= dist_to_light - 0.002 {
                var cos_recv = max(cos_raw, 0.0);
                if cos_raw < 0.0 && petal_t > 0.0 {
                    cos_recv = abs(cos_raw) * petal_t;
                }
                let area = 4.0 * 3.14159265359 * ls.radius * ls.radius;
                let light_pdf = (dist_to_light * dist_to_light / (cos_emitter * area)) / f32(u.light_count);
                let bsdf_pdf = max(cos_recv, 0.001) / 3.14159265359;
                let mis_w = light_pdf / (light_pdf + bsdf_pdf);
                if light_pdf > 0.0 {
                    var sun_light_color = lmat.albedo.xyz;
                    if dot(wi2, normalize(u.sun_dir.xyz)) > 0.95 {
                        sun_light_color = sun_light_color * vec3<f32>(1.0, 0.65, 0.22);
                    }
                    col = col + thr * surface_color * sun_light_color * cos_recv
                        / (3.14159265359 * light_pdf) * mis_w;
                }
            }
        }

        // Step B: MixturePdf scatter
        let petal_t2 = select(mat.fuzz, 0.0, mt == 5u || mt == 6u);
        var scatter_n = use_n;
        var transmission_color = vec3<f32>(1.0);
        if petal_t2 > 0.0 && rand(seed) < petal_t2 {
            scatter_n = -use_n;
            let sss_dist = abs(dot(rd2, use_n)) * 0.02;
            transmission_color = exp(-vec3<f32>(0.2, 2.5, 4.5) * sss_dist);
        }
        var sdir: vec3<f32>;
        if u.light_count > 0u && rand(seed) < 0.5 {
            let li2 = u32(rand(seed) * f32(u.light_count)) % u.light_count;
            let ls2 = spheres[lights[li2]];
            let rp = rand_unit_vec(seed);
            let lp2 = ls2.center.xyz + rp * ls2.radius;
            let tl = lp2 - p;
            let dsq = dot(tl, tl);
            if dsq < 0.0001 {
                sdir = onb_local(use_n, rand_cosine_dir(seed));
                thr = thr * surface_color;
            } else {
                let d = sqrt(dsq);
                sdir = tl / d;
                let cr2 = max(dot(use_n, sdir), 0.001);
                let ce2 = max(abs(dot(rp, -sdir)), 0.001);
                let area2 = 4.0 * 3.14159265359 * ls2.radius * ls2.radius;
                let light_pdf2 = (dsq / (ce2 * area2)) / f32(u.light_count);
                let cosine_pdf2 = cr2 / 3.14159265359;
                let mix_pdf2 = max(0.5 * light_pdf2 + 0.5 * cosine_pdf2, 0.0001);
                thr = thr * surface_color * cr2 / (3.14159265359 * mix_pdf2);
            }
        } else {
            sdir = onb_local(use_n, rand_cosine_dir(seed));
            thr = thr * surface_color;
        }
        if petal_t2 > 0.0 && rand(seed) < petal_t2 {
            sdir = -sdir;
            // 2-step mini random walk for subtle SSS
            var sss_thr = vec3<f32>(1.0);
            var sss_pos = p;
            var sss_dir2 = sdir;
            for (var ss: u32 = 0u; ss < 2u; ss++) {
                let d_free = -log(max(rand(seed), 0.001)) / 14.0;
                sss_pos = sss_pos + d_free * sss_dir2;
                sss_thr *= exp(-vec3<f32>(0.15, 2.0, 4.0) * d_free);
                sss_dir2 = sample_hg(sss_dir2, 0.5, seed);
            }
            sdir = normalize(sss_dir2);
            thr = thr * sss_thr * transmission_color;
        }
        if mt == 5u {
            let refl = reflect(normalize(rd2), use_n);
            let rough_refl = normalize(refl + use_fuzz * rand_unit_vec(seed));
            if dot(rough_refl, use_n) > 0.001 {
                sdir = normalize(mix(sdir, rough_refl, 0.06));
                thr = thr * mix(surface_color, vec3<f32>(1.0), 0.06);
            }
        }
        if mt == 6u {
            let refl = reflect(normalize(rd2), use_n);
            let rough_refl = normalize(refl + use_fuzz * rand_unit_vec(seed));
            if dot(rough_refl, use_n) > 0.001 {
                sdir = normalize(mix(sdir, rough_refl, 0.04));
                thr = thr * mix(surface_color, vec3<f32>(1.0), 0.04);
            }
        }
        if dot(sdir, use_n) < 0.001 { sdir = select(-use_n, use_n, dot(sdir, use_n) > 0.0); }
        thr = max(thr, vec3<f32>(0.0));
        let offset_sign = select(-1.0, 1.0, dot(sdir, use_n) > 0.0);
        rd2 = sdir; ro2 = p + use_n * 0.001 * offset_sign;
    }
    return col;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x + u.tile_start_x;
    let y = gid.y + u.tile_start_y;
    if x >= u.tile_end_x || y >= u.tile_end_y { return; }

    let grid_dim = u32(floor(sqrt(f32(u.samples_per_pixel))));
    let grid_count = grid_dim * grid_dim;

    var acc = vec3<f32>(0.0);
    for (var bi: u32 = 0u; bi < u.batch_count; bi++) {
        let abs_s = bi + u.batch_offset;
        var seed = hash_u32(x + y * u.image_width + abs_s * 1234567u);

        let rx = rand(&seed);
        let ry = rand(&seed);
        let r_time = rand(&seed);  // per-ray motion-blur timestamp

        var du: f32;
        var dv: f32;
        if abs_s < grid_count {
            let gx = abs_s % grid_dim;
            let gy = abs_s / grid_dim;
            du = (f32(x) + (f32(gx) + rx) / f32(grid_dim)) / f32(u.image_width - 1u);
            dv = (f32(u.image_height - 1u - y) + (f32(gy) + ry) / f32(grid_dim)) / f32(u.image_height - 1u);
        } else {
            du = (f32(x) + rx) / f32(u.image_width - 1u);
            dv = (f32(u.image_height - 1u - y) + ry) / f32(u.image_height - 1u);
        }

        let orig = cam.origin.xyz;
        let dir = normalize(cam.lower_left_corner.xyz + du * cam.horizontal.xyz + dv * cam.vertical.xyz - cam.origin.xyz);
        acc = acc + ray_color(orig, dir, r_time, &seed);
    }
    let idx = y * u.image_width + x;
    output[idx] = output[idx] + vec4<f32>(acc.x, acc.y, acc.z, 0.0);
}