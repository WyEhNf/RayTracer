mod camera;
mod vec3;

use bytemuck::{Pod, Zeroable};
use camera::GpuCamera;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::mpsc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuUniforms {
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
    _pad_particle: [u32; 2], // WGSL vec4 alignment
    background: [f32; 4],
    tex_count: u32,
    batch_offset: u32,
    batch_count: u32,
    tile_start_x: u32,
    tile_start_y: u32,
    tile_end_x: u32,
    tile_end_y: u32,
    _pad1: u32,          // WGSL vec4 16-byte alignment
    sun_dir: [f32; 4],   // xyz = sun direction (dusk), w = unused
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuTexture {
    data_offset: u32,
    width: u32,
    height: u32,
    channels: u32,  // 3=RGB, 4=RGBA
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuSphere {
    pub center: [f32; 4],
    pub radius: f32,
    pub material_id: u32,
    pub _pad: [u32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuTriangle {
    v0: [f32; 4],
    v1: [f32; 4],
    v2: [f32; 4],
    n0: [f32; 4],
    n1: [f32; 4],
    n2: [f32; 4],
    uv0: [f32; 2],
    uv1: [f32; 2],
    uv2: [f32; 2],
    material_id: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuMaterial {
    albedo: [f32; 4],
    fuzz: f32,
    ref_idx: f32,
    material_type: u32,
    tex_id: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuParticle {
    pos_t0: [f32; 4],  // position at shutter open
    pos_t1: [f32; 4],  // position at shutter close (t0 + fall + wind)
    radius: f32,
    material_id: u32,
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuBvhNode {
    bbox_min: [f32; 3],
    left_or_first: u32,
    bbox_max: [f32; 3],
    primitive_count: u32,
}

struct BvhPrim {
    bbox_min: [f32; 3],
    bbox_max: [f32; 3],
    center: [f32; 3],
    is_triangle: bool,
    index: u32,
}

fn bbox_sphere(s: &GpuSphere) -> ([f32; 3], [f32; 3]) {
    let mut min = [0.0f32; 3];
    let mut max = [0.0f32; 3];
    for a in 0..3 {
        min[a] = s.center[a] - s.radius;
        max[a] = s.center[a] + s.radius;
    }
    (min, max)
}

fn bbox_triangle(t: &GpuTriangle) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for a in 0..3 {
        min[a] = t.v0[a].min(t.v1[a]).min(t.v2[a]);
        max[a] = t.v0[a].max(t.v1[a]).max(t.v2[a]);
        min[a] -= 0.0001;
        max[a] += 0.0001;
    }
    (min, max)
}

fn build_all_bvh_reordered(
    spheres: &mut Vec<GpuSphere>,
    triangles: &mut Vec<GpuTriangle>,
) -> Vec<GpuBvhNode> {
    let mut prims: Vec<BvhPrim> = Vec::new();

    for (i, s) in spheres.iter().enumerate() {
        let (min, max) = bbox_sphere(s);
        prims.push(BvhPrim {
            bbox_min: min, bbox_max: max,
            center: [(min[0]+max[0])/2.0, (min[1]+max[1])/2.0, (min[2]+max[2])/2.0],
            is_triangle: false, index: i as u32,
        });
    }

    for (i, t) in triangles.iter().enumerate() {
        let (min, max) = bbox_triangle(t);
        prims.push(BvhPrim {
            bbox_min: min, bbox_max: max,
            center: [(min[0]+max[0])/2.0, (min[1]+max[1])/2.0, (min[2]+max[2])/2.0],
            is_triangle: true, index: i as u32,
        });
    }

    let sphere_base = 0u32;
    let tri_base = spheres.len() as u32;

    let nodes = build_unified_bvh_reorder(&mut prims, 8, spheres, triangles, sphere_base, tri_base);
    nodes
}

fn emit_leaf_nodes(prims: &[BvhPrim], bmin:[f32;3], bmax:[f32;3], spheres: &mut Vec<GpuSphere>, triangles: &mut Vec<GpuTriangle>) -> Vec<GpuBvhNode> {
    let mut out = Vec::new();
    let tri_count = prims.iter().filter(|p| p.is_triangle).count();
    if tri_count > 0 {
        let start = triangles.len() as u32;
        for p in prims.iter().filter(|p| p.is_triangle) { triangles.push(GpuTriangle{..triangles[p.index as usize]}); }
        out.push(GpuBvhNode{bbox_min:bmin, left_or_first:start|0x80000000u32, bbox_max:bmax, primitive_count:tri_count as u32});
    }
    let sph_count = prims.len()-tri_count;
    if sph_count > 0 {
        let start = spheres.len() as u32;
        for p in prims.iter().filter(|p|!p.is_triangle) { spheres.push(GpuSphere{..spheres[p.index as usize]}); }
        out.push(GpuBvhNode{bbox_min:bmin, left_or_first:start, bbox_max:bmax, primitive_count:sph_count as u32});
    }
    out
}

fn build_unified_bvh_reorder(
    prims: &mut [BvhPrim],
    leaf_max: usize,
    spheres: &mut Vec<GpuSphere>,
    triangles: &mut Vec<GpuTriangle>,
    _sphere_base: u32,
    _tri_base: u32,
) -> Vec<GpuBvhNode> {
    const INTERNAL_FLAG: u32 = 0x80000000;

    if prims.is_empty() {
        return vec![GpuBvhNode { bbox_min:[0.0;3], left_or_first:0, bbox_max:[0.0;3], primitive_count:0 }];
    }

    let mut bmin = prims[0].bbox_min;
    let mut bmax = prims[0].bbox_max;
    for p in &prims[1..] { for a in 0..3 {
        bmin[a] = bmin[a].min(p.bbox_min[a]); bmax[a] = bmax[a].max(p.bbox_max[a]);
    }}

    if prims.len() <= leaf_max {
        return emit_leaf_nodes(prims, bmin, bmax, spheres, triangles);
    }

    let ext = [bmax[0]-bmin[0], bmax[1]-bmin[1], bmax[2]-bmin[2]];
    let axis = if ext[0] >= ext[1] && ext[0] >= ext[2] { 0 }
        else if ext[1] >= ext[2] { 1 } else { 2 };

    // ── 16-bin SAH split ──
    const NB: usize = 16;
    let mut bin_cnt = [0u32; NB];
    let mut bin_min: [[f32; 3]; NB] = [[f32::MAX; 3]; NB];
    let mut bin_max: [[f32; 3]; NB] = [[f32::MIN; 3]; NB];

    let mut cmin = f32::MAX; let mut cmax = f32::MIN;
    for p in prims.iter() { cmin = cmin.min(p.center[axis]); cmax = cmax.max(p.center[axis]); }
    let cext = cmax - cmin;
    if cext < 1e-6 { return emit_leaf_nodes(prims, bmin, bmax, spheres, triangles); }

    let inv_bw = NB as f32 / cext;
    for (i, p) in prims.iter().enumerate() {
        let b = (((p.center[axis] - cmin) * inv_bw) as usize).min(NB - 1);
        bin_cnt[b] += 1;
        for a in 0..3 { bin_min[b][a] = bin_min[b][a].min(p.bbox_min[a]); bin_max[b][a] = bin_max[b][a].max(p.bbox_max[a]); }
        _ = i;
    }

    // Prefix sweep: left→right accumulated bounds + counts
    let mut lcnt = [0u32; NB-1]; let mut lmin = [f32::MAX;3]; let mut lmax = [f32::MIN;3];
    let mut lbnd: [([f32;3],[f32;3]); NB-1] = [([f32::MAX;3],[f32::MIN;3]); NB-1];
    let mut cl = 0u32;
    for i in 0..NB-1 { cl += bin_cnt[i]; lcnt[i] = cl;
        for a in 0..3 { lmin[a]=lmin[a].min(bin_min[i][a]); lmax[a]=lmax[a].max(bin_max[i][a]); }
        lbnd[i] = (lmin, lmax); }

    // Suffix sweep: find best split
    let node_sa = 2.0*((bmax[0]-bmin[0])*(bmax[1]-bmin[1])+(bmax[1]-bmin[1])*(bmax[2]-bmin[2])+(bmax[2]-bmin[2])*(bmax[0]-bmin[0]));
    let mut best_cost = f32::MAX; let mut best_bin = 0u32;
    let mut rmin = [f32::MAX;3]; let mut rmax = [f32::MIN;3]; let mut cr = 0u32;
    for i in (1..NB).rev() {
        cr += bin_cnt[i];
        for a in 0..3 { rmin[a]=rmin[a].min(bin_min[i][a]); rmax[a]=rmax[a].max(bin_max[i][a]); }
        let nl = lcnt[i-1] as f32; let nr = cr as f32;
        if nl < 1.0 || nr < 1.0 { continue; }
        let (lm,lM) = lbnd[i-1]; let le = [lM[0]-lm[0],lM[1]-lm[1],lM[2]-lm[2]];
        let ls = 2.0*(le[0]*le[1]+le[1]*le[2]+le[2]*le[0]);
        let re = [rmax[0]-rmin[0],rmax[1]-rmin[1],rmax[2]-rmin[2]];
        let rs = 2.0*(re[0]*re[1]+re[1]*re[2]+re[2]*re[0]);
        let cost = 0.5 + (ls*nl + rs*nr)/node_sa;
        if cost < best_cost { best_cost = cost; best_bin = i as u32; }
    }

    // SAH termination: stop if leaf is cheaper than split
    if best_cost >= prims.len() as f32 { return emit_leaf_nodes(prims, bmin, bmax, spheres, triangles); }

    // Partition at best_bin boundary
    let split_v = cmin + (best_bin as f32 + 0.5) * cext / NB as f32;
    let (mut i, mut j) = (0usize, prims.len()-1);
    loop {
        while i < prims.len() && prims[i].center[axis] < split_v { i += 1; }
        while j > 0 && prims[j].center[axis] >= split_v { j = if j > 0 { j-1 } else { 0 }; if j == 0 { break; } }
        if i >= j { break; }
        prims.swap(i, j); i += 1; if j > 0 { j -= 1; }
    }
    let mid = if i == 0 { 1.min(prims.len()-1) } else if i >= prims.len() { prims.len()-1 } else { i };
    let (left_slice, right_slice) = prims.split_at_mut(mid);
    let mut left = build_unified_bvh_reorder(left_slice, leaf_max, spheres, triangles, _sphere_base, _tri_base);
    let mut right = build_unified_bvh_reorder(right_slice, leaf_max, spheres, triangles, _sphere_base, _tri_base);
    let left_size = left.len() as u32;

    // Adjust left subtree internal node indices: offset +1 (root is at index 0)
    for node in left.iter_mut() {
        if (node.primitive_count & INTERNAL_FLAG) != 0 {
            node.left_or_first += 1;
            let rc = node.primitive_count & !INTERNAL_FLAG;
            node.primitive_count = (rc + 1) | INTERNAL_FLAG;
        }
    }
    // Adjust right subtree internal node indices: offset +1 + left_size
    for node in right.iter_mut() {
        if (node.primitive_count & INTERNAL_FLAG) != 0 {
            node.left_or_first += 1 + left_size;
            let rc = node.primitive_count & !INTERNAL_FLAG;
            node.primitive_count = (rc + 1 + left_size) | INTERNAL_FLAG;
        }
    }

    // Root: left child at index 1, right child at index 1 + left_size
    let root = GpuBvhNode {
        bbox_min: bmin,
        left_or_first: 1,
        bbox_max: bmax,
        primitive_count: (1 + left_size) | INTERNAL_FLAG,
    };
    let mut nodes = vec![root];
    nodes.append(&mut left);
    nodes.append(&mut right);
    nodes
}

fn parse_mtl(path: &str) -> Vec<(String, GpuMaterial)> {
    let mut materials: Vec<(String, GpuMaterial)> = Vec::new();
    let file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return materials,
    };
    let reader = BufReader::new(file);
    let mut current_name = String::new();
    let mut current_kd = [0.8, 0.8, 0.8];

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };
        let line = line.trim();
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }
        match parts[0] {
            "newmtl" if parts.len() >= 2 => {
                if !current_name.is_empty() {
                    materials.push((
                        current_name.clone(),
                        GpuMaterial {
                            albedo: [current_kd[0], current_kd[1], current_kd[2], 0.0],
                            fuzz: 0.0,
                            ref_idx: 1.0,
                            material_type: 0,
                            tex_id: 0,
                        },
                    ));
                }
                current_name = parts[1].to_string();
                current_kd = [0.8, 0.8, 0.8];
            }
            "Kd" if parts.len() >= 4 => {
                current_kd = [
                    parts[1].parse().unwrap_or(0.8),
                    parts[2].parse().unwrap_or(0.8),
                    parts[3].parse().unwrap_or(0.8),
                ];
            }
            _ => {}
        }
    }
    if !current_name.is_empty() {
        materials.push((
            current_name,
            GpuMaterial {
                albedo: [current_kd[0], current_kd[1], current_kd[2], 0.0],
                fuzz: 0.0,
                ref_idx: 1.0,
                material_type: 0,
                tex_id: 0,
            },
        ));
    }
    materials
}

fn load_obj(
    path: &str,
    default_mat_id: u32,
) -> (Vec<GpuTriangle>, Vec<GpuMaterial>, Vec<String>) {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Cannot open OBJ {}: {}", path, e);
            return (Vec::new(), Vec::new(), Vec::new());
        }
    };

    let obj_dir = Path::new(path).parent().map(|p| p.to_path_buf());
    let reader = BufReader::new(file);
    let mut vertices: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut texcoords: Vec<[f32; 2]> = Vec::new();
    let mut triangles: Vec<GpuTriangle> = Vec::new();
    let mut materials: Vec<GpuMaterial> = Vec::new();
    let mut material_names: Vec<String> = Vec::new();
    let mut mtl_map: HashMap<String, u32> = HashMap::new();
    let mut current_mat_id: u32 = default_mat_id;

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };
        let line = line.trim();
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "mtllib" if parts.len() >= 2 => {
                let mtl_name = parts[1..].join(" ");
                let mtl_path = if let Some(ref dir) = obj_dir {
                    dir.join(&mtl_name).to_string_lossy().to_string()
                } else {
                    mtl_name
                };
                let parsed = parse_mtl(&mtl_path);
                println!("MTL {} -> {} materials", mtl_path, parsed.len());
                for (name, mat) in &parsed {
                    let id = materials.len() as u32;
                    materials.push(*mat);
                    material_names.push(name.clone());
                    mtl_map.insert(name.clone(), id);
                }
            }
            "usemtl" if parts.len() >= 2 => {
                let name = parts[1];
                if let Some(&id) = mtl_map.get(name) {
                    current_mat_id = id;
                }
            }
            "v" if parts.len() >= 4 => {
                vertices.push([
                    parts[1].parse().unwrap_or(0.0),
                    parts[2].parse().unwrap_or(0.0),
                    parts[3].parse().unwrap_or(0.0),
                ]);
            }
            "vt" if parts.len() >= 3 => {
                texcoords.push([
                    parts[1].parse().unwrap_or(0.0),
                    1.0 - parts[2].parse::<f32>().unwrap_or(0.0),  // flip V for image coords
                ]);
            }
            "vn" if parts.len() >= 4 => {
                normals.push([
                    parts[1].parse().unwrap_or(0.0),
                    parts[2].parse().unwrap_or(0.0),
                    parts[3].parse().unwrap_or(0.0),
                ]);
            }
            "f" if parts.len() >= 4 => {
                let mut fv: Vec<(usize, Option<usize>, Option<usize>)> = Vec::new();
                for f in &parts[1..] {
                    let comps: Vec<&str> = f.split('/').collect();
                    let vi = comps[0]
                        .parse::<i64>()
                        .map(|n| {
                            if n < 0 { (vertices.len() as i64 + n) as usize }
                            else { (n - 1) as usize }
                        })
                        .unwrap_or(0);
                    let ti = comps.get(1).and_then(|s| {
                        if s.is_empty() { None }
                        else {
                            s.parse::<i64>().ok().map(|n| {
                                if n < 0 { (texcoords.len() as i64 + n) as usize }
                                else { (n - 1) as usize }
                            })
                        }
                    });
                    let ni = comps.get(2).and_then(|s| {
                        if s.is_empty() { None }
                        else {
                            s.parse::<i64>().ok().map(|n| {
                                if n < 0 { (normals.len() as i64 + n) as usize }
                                else { (n - 1) as usize }
                            })
                        }
                    });
                    fv.push((vi, ti, ni));
                }
                for k in 1..fv.len() - 1 {
                    let (i0, t0, n0) = fv[0];
                    let (i1, t1, n1) = fv[k];
                    let (i2, t2, n2) = fv[k + 1];
                    if i0 >= vertices.len() || i1 >= vertices.len() || i2 >= vertices.len() {
                        continue;
                    }
                    let def_n = [0.0, 1.0, 0.0];
                    let def_uv = [0.0, 0.0];
                    let nn0 = n0.and_then(|i| normals.get(i).copied()).unwrap_or(def_n);
                    let nn1 = n1.and_then(|i| normals.get(i).copied()).unwrap_or(def_n);
                    let nn2 = n2.and_then(|i| normals.get(i).copied()).unwrap_or(def_n);
                    let uv0_val = t0.and_then(|i| texcoords.get(i).copied()).unwrap_or(def_uv);
                    let uv1_val = t1.and_then(|i| texcoords.get(i).copied()).unwrap_or(def_uv);
                    let uv2_val = t2.and_then(|i| texcoords.get(i).copied()).unwrap_or(def_uv);
                    triangles.push(GpuTriangle {
                        v0: [vertices[i0][0], vertices[i0][1], vertices[i0][2], 0.0],
                        v1: [vertices[i1][0], vertices[i1][1], vertices[i1][2], 0.0],
                        v2: [vertices[i2][0], vertices[i2][1], vertices[i2][2], 0.0],
                        n0: [nn0[0], nn0[1], nn0[2], 0.0],
                        n1: [nn1[0], nn1[1], nn1[2], 0.0],
                        n2: [nn2[0], nn2[1], nn2[2], 0.0],
                        uv0: uv0_val,
                        uv1: uv1_val,
                        uv2: uv2_val,
                        material_id: current_mat_id,
                        _pad: 0,
                    });
                }
            }
            _ => {}
        }
    }

    println!(
        "OBJ: {} verts, {} texcoords, {} normals, {} triangles",
        vertices.len(),
        texcoords.len(),
        normals.len(),
        triangles.len()
    );
    (triangles, materials, material_names)
}

fn load_textures(
    material_names: &[String],
    obj_dir: &Path,
) -> (Vec<GpuTexture>, Vec<f32>) {
    let mut tex_data: Vec<f32> = Vec::new();
    let mut textures: Vec<GpuTexture> = Vec::new();

    // Map: texture filename (without extension) → list of material name substrings to match
    let tex_files: &[(&str, &[&str])] = &[
        ("肌.png", &["肌"]),
        ("面.png", &["面"]),
        ("发.png", &["髮", "髮spa+", "前髮", "前髮spa+"]),
        ("服.png", &["服", "裙", "新規", "蝴蝶結", "結1", "帶", "帶內", "配飾", "褲"]),
    ];

    let max_tex_size: u32 = 2048;

    for (filename, _matches) in tex_files {
        let tex_path = obj_dir.join(filename);
        let final_path = if tex_path.exists() {
            tex_path
        } else {
            let alt = Path::new("assets").join(filename);
            if alt.exists() { alt } else { continue; }
        };
        let mut img = match image::open(&final_path) {
            Ok(i) => i.to_rgba8(),
            Err(_) => continue,
        };
        // Downscale if too large
        let (mut w, mut h) = (img.width(), img.height());
        if w > max_tex_size || h > max_tex_size {
            let scale = max_tex_size as f64 / w.max(h) as f64;
            let new_w = (w as f64 * scale) as u32;
            let new_h = (h as f64 * scale) as u32;
            img = image::imageops::resize(&img, new_w, new_h, image::imageops::FilterType::Lanczos3);
            w = new_w;
            h = new_h;
            println!("  Resized {} ({}x{})", filename, w, h);
        }
        let offset = tex_data.len() as u32;
        for y in 0..h {
            for x in 0..w {
                let p = img.get_pixel(x, y);
                tex_data.push(p[0] as f32 / 255.0);
                tex_data.push(p[1] as f32 / 255.0);
                tex_data.push(p[2] as f32 / 255.0);
                tex_data.push(p[3] as f32 / 255.0);
            }
        }
        textures.push(GpuTexture { data_offset: offset, width: w, height: h, channels: 4 });
    }

    println!("Loaded {} textures ({} floats)", textures.len(), tex_data.len());
    (textures, tex_data)
}

fn assign_tex_id(name: &str, _textures: &[GpuTexture]) -> u32 {
    // Textures loaded in order: 1=肌, 2=面, 3=发, 4=服
    if name.contains("肌") || name.contains("二重") { return 1; }
    if name.contains("面") { return 2; }
    if name.contains("髮") || name.contains("前髮") { return 3; }
    if name.contains("服") || name.contains("裙") || name.contains("新規")
        || name.contains("蝴蝶結") || name.contains("結1") || name.contains("帶")
        || name.contains("配飾") || name.contains("褲") { return 4; }
    0
}

// ─── Scene (鳴神大社) texture & material helpers ───

fn load_scene_textures(obj_dir: &Path) -> (Vec<GpuTexture>, Vec<f32>) {
    let mut tex_data: Vec<f32> = Vec::new();
    let mut textures: Vec<GpuTexture> = Vec::new();

    let tex_dir = if obj_dir.join("Tex").exists() {
        obj_dir.join("Tex")
    } else if Path::new("../assets/Tex").exists() {
        Path::new("../assets/Tex").to_path_buf()
    } else if Path::new("assets/Tex").exists() {
        Path::new("assets/Tex").to_path_buf()
    } else {
        eprintln!("Scene Tex/ directory not found");
        return (textures, tex_data);
    };

    let max_tex_size: u32 = 1024;

    // Material NAME number → texture file mapping
    // Verified against Ray渲染材质包/*.fx NORMAL_MAP_FILE references
    // tex_map[N] = base color texture for 材質N
    let tex_map: [Option<&str>; 41] = [
        None,                          // 0 unused
        Some("Tex_0002.png"),          // 材質1  — fx:N=Tex_0002_N
        Some("Tex_0003.png"),          // 材質2  — fx:N=Tex_0003_N
        Some("Tex_0004.png"),          // 材質3  — fx:N=Tex_0004_N
        Some("Tex_0005.png"),          // 材質4  — fx:N=Tex_0005_N
        Some("Tex_0007.png"),          // 材質5  — fx:N=Tex_0007_N
        Some("Tex_0008.png"),          // 材質6  — fx:N=Tex_0008_N
        Some("Tex_0009.png"),          // 材質7  — fx:N=Tex_0009_N
        Some("Tex_0010.png"),          // 材質8  — fx:N=Tex_0010_N
        Some("Tex_0011.png"),          // 材質9  — fx:N=Tex_0011_N
        Some("Tex_0012.png"),          // 材質10 — fx:N=Tex_0012_N
        Some("Tex_0013.png"),          // 材質11 — fx:N=Tex_0013_N
        Some("Tex_0014.png"),          // 材質12 — fx:N=Tex_0014_N
        Some("Tex_0015.png"),          // 材質13 — fx:N=Tex_0015_N
        Some("Tex_0006_3.tga"),        // 材質14 — petal (α=Tex_0006_A)
        Some("Tex_0018.png"),          // 材質15 — fx:N=Tex_0018_N
        Some("Tex_0019.png"),          // 材質16 — fx:N=Tex_0019_N  (was incorrectly skipped as normal map)
        Some("Tex_0020.png"),          // 材質17 — smooth=1.0 specular material
        Some("Tex_0021.png"),          // 材質18 — fx:N=Tex_0021_N
        Some("Tex_0025.png"),          // 材質19 — fx:N=Tex_0025_N
        Some("Tex_0024.png"),          // 材質20 → 0024
        Some("Tex_0026.png"),          // 材質21 — fx:N=Tex_0026_N
        Some("Tex_0027.png"),          // 材質22 — fx:N=Tex_0027_N
        Some("Tex_0028.png"),          // 材質23 — fx:N=Tex_0028_N
        Some("Tex_0029.png"),          // 材質24 — fx:N=Tex_0029_N
        Some("Tex_0030.png"),          // 材質25 — fx:N=Tex_0030_N
        Some("Tex_0031.png"),          // 材質26 — fx:N=Tex_0031_N
        Some("Tex_0032.png"),          // 材質27 — fx:N=Tex_0032_N
        Some("Tex_0033.png"),          // 材質28 — fx:N=Tex_0033_N
        Some("Tex_0034.png"),          // 材質29 — fx:N=Tex_0034_N
        Some("Tex_0035.png"),          // 材質30 — fx:N=Tex_0035_N
        Some("Tex_0038_3.tga"),        // 材質31 — petal (α=Tex_0006_A)
        Some("Tex_0040.png"),          // 材質32 — fx:N=Tex_0040_N
        Some("Tex_0015.png"),          // 材質33 — fx: smooth=1.0, FIXED: use 0015
        Some("Tex_0017.png"),          // 材質34 — fx:N=Tex_0017_N
        Some("Tex_0015.png"),          // 材質35 — fx:N=Tex_0015_N  (FIXED: was Tex_0001)
        Some("Tex_0016.png"),          // 材質36 — fx:N=Tex_0016_N
        Some("Tex_0023.png"),          // 材質37 — fx:N=Tex_0023_N
        Some("Tex_0022.png"),          // 材質38 — fx:N=Tex_0022_N
        Some("Tex_0015.png"),          // 材質39 — fx:N=Tex_0015_N  (FIXED: was Tex_0037_3.tga)
        Some("Tex_0015.png"),          // 材質40 — fx:N=Tex_0015_N  (FIXED: was Tex_0039)
    ];

    for mtl_idx in 1u32..=40 {
        let filename = match tex_map[mtl_idx as usize] {
            Some(f) => f,
            None => {
                eprintln!("  MTL idx {} — missing, using white", mtl_idx);
                let offset = tex_data.len() as u32;
                tex_data.push(1.0); tex_data.push(1.0); tex_data.push(1.0); tex_data.push(1.0);
                textures.push(GpuTexture { data_offset: offset, width: 1, height: 1, channels: 4 });
                continue;
            }
        };
        let tex_path = tex_dir.join(filename);
        if !tex_path.exists() {
            eprintln!("  {} missing, using white", filename);
            let offset = tex_data.len() as u32;
            tex_data.push(1.0); tex_data.push(1.0); tex_data.push(1.0); tex_data.push(1.0);
            textures.push(GpuTexture { data_offset: offset, width: 1, height: 1, channels: 4 });
            continue;
        }
        let mut img = match image::open(&tex_path) {
            Ok(i) => i.to_rgba8(),
            Err(e) => {
                eprintln!("  Cannot open {}: {}", filename, e);
                let offset = tex_data.len() as u32;
                tex_data.push(1.0); tex_data.push(1.0); tex_data.push(1.0); tex_data.push(1.0);
                textures.push(GpuTexture { data_offset: offset, width: 1, height: 1, channels: 4 });
                continue;
            }
        };
        let (mut w, mut h) = (img.width(), img.height());
        if w > max_tex_size || h > max_tex_size {
            let scale = max_tex_size as f64 / w.max(h) as f64;
            let new_w = (w as f64 * scale) as u32;
            let new_h = (h as f64 * scale) as u32;
            img = image::imageops::resize(&img, new_w, new_h, image::imageops::FilterType::Lanczos3);
            w = new_w;
            h = new_h;
            println!("  Resized {} ({}x{})", filename, w, h);
        }
        let offset = tex_data.len() as u32;
        for y in 0..h {
            for x in 0..w {
                let p = img.get_pixel(x, y);
                tex_data.push(p[0] as f32 / 255.0);
                tex_data.push(p[1] as f32 / 255.0);
                tex_data.push(p[2] as f32 / 255.0);
                tex_data.push(p[3] as f32 / 255.0);
            }
        }
        // Debug: sample average pixel colour
        let sample_count = (w * h).min(1000);
        let step = (w * h / sample_count).max(1);
        let (mut sr, mut sg, mut sb) = (0u64, 0u64, 0u64);
        for i in (0..w*h).step_by(step as usize) {
            let p = img.get_pixel(i % w, i / w);
            sr += p[0] as u64; sg += p[1] as u64; sb += p[2] as u64;
        }
        let n = (sample_count as u64).max(1);
        let avg_r = sr as f64 / n as f64;
        let avg_g = sg as f64 / n as f64;
        let avg_b = sb as f64 / n as f64;
        // Detect non-base-colour textures (conservative, only extreme cases)
        let avg_lum = (avg_r + avg_g + avg_b) / 3.0;
        let is_blank = avg_lum > 252.0;       // pure-white blank/mask
        let is_ao = avg_lum < 12.0;            // near-black AO/utility
        if is_blank || is_ao {
            println!("  {} avg RGB=({:.0},{:.0},{:.0}) — non-colour, using mid-grey", filename, avg_r, avg_g, avg_b);
            tex_data.truncate(offset as usize);
            tex_data.push(0.5); tex_data.push(0.5); tex_data.push(0.5); tex_data.push(1.0);
            textures.push(GpuTexture { data_offset: offset, width: 1, height: 1, channels: 4 });
        } else {
            println!("  {} avg RGB=({:.0},{:.0},{:.0})", filename, avg_r, avg_g, avg_b);
            textures.push(GpuTexture { data_offset: offset, width: w, height: h, channels: 4 });
        }
    }

    println!(
        "Loaded {} scene textures ({} floats, {:.1} MB)",
        textures.len(),
        tex_data.len(),
        tex_data.len() as f64 * 4.0 / (1024.0 * 1024.0)
    );
    (textures, tex_data)
}

fn create_scene(obj_path: Option<&str>) -> (Vec<GpuSphere>, Vec<GpuTriangle>, Vec<GpuMaterial>, Vec<u32>, Vec<GpuTexture>, Vec<f32>) {
    let mut materials = vec![
        GpuMaterial { albedo: [0.5,0.5,0.5,0.0], fuzz:0.0, ref_idx:1.0, material_type:0, tex_id:0 },
        GpuMaterial { albedo: [0.4,0.2,0.1,0.0], fuzz:0.0, ref_idx:1.0, material_type:0, tex_id:0 },
        GpuMaterial { albedo: [0.7,0.6,0.5,0.0], fuzz:0.0, ref_idx:1.0, material_type:1, tex_id:0 },
        GpuMaterial { albedo: [1.0,1.0,1.0,0.0], fuzz:0.0, ref_idx:1.5, material_type:2, tex_id:0 },
        GpuMaterial { albedo: [15.0,15.0,15.0,0.0], fuzz:0.0, ref_idx:1.0, material_type:3, tex_id:0 },
    ];

    let default_mat_count = materials.len() as u32;

    let mut spheres = vec![
        GpuSphere { center: [0.0, -1000.0, 0.0, 0.0], radius: 1000.0, material_id: 0, _pad: [0; 2] },
        GpuSphere { center: [0.0, 1.0, 0.0, 0.0], radius: 1.0, material_id: 3, _pad: [0; 2] },
        GpuSphere { center: [-4.0, 1.0, 0.0, 0.0], radius: 1.0, material_id: 1, _pad: [0; 2] },
        GpuSphere { center: [4.0, 1.0, 0.0, 0.0], radius: 1.0, material_id: 2, _pad: [0; 2] },
    ];

    for a in -11..11 {
        for b in -11..11 {
            let choose_mat = vec3::random_double();
            let center = [
                a as f32 + 0.9 * vec3::random_double(),
                0.2,
                b as f32 + 0.9 * vec3::random_double(),
                0.0,
            ];
            let dx = center[0] - 4.0;
            let dz = center[2] - 0.0;
            if (dx * dx + dz * dz).sqrt() > 0.9 {
                let mat_id = if choose_mat < 0.8 {
                    0
                } else if choose_mat < 0.95 {
                    2
                } else {
                    3
                };
                spheres.push(GpuSphere {
                    center,
                    radius: 0.2,
                    material_id: mat_id,
                    _pad: [0; 2],
                });
            }
        }
    }

    let mut triangles: Vec<GpuTriangle> = Vec::new();
    let mut tex_data_vec: Vec<f32> = Vec::new();
    let mut tex_info: Vec<GpuTexture> = Vec::new();
    let mut lights: Vec<u32> = Vec::new();

    if let Some(path) = obj_path {
        let (obj_tris, obj_mats, obj_mat_names) = load_obj(path, 0);
        triangles = obj_tris;
        for m in obj_mats {
            materials.push(m);
        }

        // Load textures and map to materials
        let obj_dir = Path::new(path).parent().unwrap_or(Path::new("."));
        let (texs, tdata) = load_textures(&obj_mat_names, obj_dir);
        tex_info = texs;
        tex_data_vec = tdata;

        // Assign tex_id + material type to each OBJ material
        for (i, name) in obj_mat_names.iter().enumerate() {
            let mat_idx = default_mat_count as usize + i;
            let m = &mut materials[mat_idx];
            m.tex_id = assign_tex_id(name, &tex_info);

            // --- Material type, fuzz, and color based on body part ---
            match name.as_str() {
                // === Metal (type 1) — only for clearly metallic items, very low fuzz ===
                "神之眼" => { m.material_type = 1; m.fuzz = 0.03; m.albedo = [0.3, 0.65, 0.85, 0.0]; }
                "配飾" => { m.material_type = 1; m.fuzz = 0.08; }
                "髮饰" => { m.material_type = 1; m.fuzz = 0.10; }
                "帶" | "帶內" => { m.material_type = 1; m.fuzz = 0.12; }
                "服" | "褲" | "新規" => { m.material_type = 1; m.fuzz = 0.08; }
                "裙" => { m.material_type = 0; }

                // === Skin: restore texture with proper albedo tint ===
                "肌" | "面1" | "面2" | "二重" => { m.material_type = 0; m.albedo = [0.95, 0.75, 0.65, 0.0]; }
                // === Hair: light blue tint ===
                "髮" | "前髮" | "髮spa+" | "前髮spa+" => { m.material_type = 0; m.albedo = [0.80, 0.85, 0.95, 0.0]; }

                // === Dielectric (type 2) ===
                // === Eye layers: 白目(sclera) diffuse | 目(iris) glass | 星目(pupil) dark glass ===
                "白目" => { m.material_type = 0; m.albedo = [1.0, 1.0, 1.0, 0.0]; }
                "目" => { m.material_type = 2; m.ref_idx = 1.45; m.fuzz = 0.35; m.albedo = [0.12, 0.15, 0.42, 0.0]; }
                "星目" => { m.material_type = 2; m.ref_idx = 1.45; m.fuzz = 0.80; m.albedo = [0.03, 0.03, 0.06, 0.0]; }

                // === Diffuse (type 0) — everything else ===
                _ => {
                    m.material_type = 0;
                    if m.tex_id == 0 {
                        let color: [f32; 4] = match name.as_str() {
                            "口舌" => [0.9, 0.35, 0.35, 0.0],
                            "白目" => [1.0, 1.0, 1.0, 0.0],
                            "眉" | "睫" => [0.1, 0.08, 0.12, 0.0],
                            "齒" => [1.0, 1.0, 1.0, 0.0],
                            _ => [0.8, 0.8, 0.8, 0.0],
                        };
                        m.albedo = color;
                    }
                }
            }
        }

        if !triangles.is_empty() {
            for t in triangles.iter_mut() {
                t.material_id += default_mat_count;
            }

            let mut bmin = [f32::MAX; 3];
            let mut bmax = [f32::MIN; 3];
            for t in &triangles {
                for a in 0..3 {
                    bmin[a] = bmin[a].min(t.v0[a]).min(t.v1[a]).min(t.v2[a]);
                    bmax[a] = bmax[a].max(t.v0[a]).max(t.v1[a]).max(t.v2[a]);
                }
            }
            println!(
                "Model bbox: ({:.2},{:.2},{:.2}) - ({:.2},{:.2},{:.2})",
                bmin[0], bmin[1], bmin[2], bmax[0], bmax[1], bmax[2]
            );
            let cy = (bmin[1] + bmax[1]) / 2.0;
            let sx = bmax[0] - bmin[0];
            let sy = bmax[1] - bmin[1];
            let sz = bmax[2] - bmin[2];
            println!("Model center Y: {:.2}, size: {:.2} x {:.2} x {:.2}", cy, sx, sy, sz);

            spheres.clear();

            let model_size = sx.max(sy).max(sz);
            let cx = (bmin[0] + bmax[0]) / 2.0;
            let cz = (bmin[2] + bmax[2]) / 2.0;

            // Ground sphere (not a light)
            spheres.push(GpuSphere {
                center: [cx, bmin[1] - 2.0, cz, 0.0],
                radius: 2.0, material_id: 0, _pad: [0; 2],
            });

            // --- Create lights: track sphere indices, not material IDs ---
            let key_light_idx = spheres.len() as u32;
            materials[4] = GpuMaterial {
                albedo: [30.0, 28.0, 24.0, 0.0], fuzz: 0.0, ref_idx: 1.0,
                material_type: 3, tex_id: 0,
            };
            spheres.push(GpuSphere {
                center: [cx + model_size * 1.5, cy + model_size * 0.5, cz - model_size * 1.5, 0.0],
                radius: model_size * 0.3, material_id: 4, _pad: [0; 2],
            });

            let fill_mat_id = materials.len() as u32;
            materials.push(GpuMaterial {
                albedo: [12.0, 14.0, 20.0, 0.0], fuzz: 0.0, ref_idx: 1.0,
                material_type: 3, tex_id: 0,
            });
            let fill_light_idx = spheres.len() as u32;
            spheres.push(GpuSphere {
                center: [cx - model_size * 1.2, cy + model_size * 0.3, cz + model_size * 1.0, 0.0],
                radius: model_size * 0.25, material_id: fill_mat_id, _pad: [0; 2],
            });

            let rim_mat_id = materials.len() as u32;
            materials.push(GpuMaterial {
                albedo: [20.0, 20.0, 25.0, 0.0], fuzz: 0.0, ref_idx: 1.0,
                material_type: 3, tex_id: 0,
            });
            let rim_light_idx = spheres.len() as u32;
            spheres.push(GpuSphere {
                center: [cx + model_size * 0.5, cy + model_size * 1.0, cz + model_size * 1.5, 0.0],
                radius: model_size * 0.2, material_id: rim_mat_id, _pad: [0; 2],
            });

            lights = vec![key_light_idx, fill_light_idx, rim_light_idx];
        }
    }

    println!(
        "Scene: {} spheres, {} triangles, {} materials",
        spheres.len(),
        triangles.len(),
        materials.len()
    );

    (spheres, triangles, materials, lights, tex_info, tex_data_vec)
}

// ─── Scene (鳴神大社) creation ────────────────────────────────────────────

fn generate_petal_particles(
    _tris: &[GpuTriangle],
    material_id: u32,
) -> (Vec<GpuParticle>, Vec<GpuSphere>) {
    use vec3::random_double;
    let mut particles: Vec<GpuParticle> = Vec::new();
    let mut spheres: Vec<GpuSphere> = Vec::new();

    let count = 6000u32;
    let shutter = 0.08f32; // seconds of motion blur

    // Spawn volumes: (cx, cy, cz, rx, rz)
    let zones: [(f32, f32, f32, f32, f32); 4] = [
        (8.0, 18.0, 12.0, 14.0, 8.0),    // right tree cluster
        (-12.0, 16.0, 5.0, 16.0, 10.0),  // left tree cluster
        (0.0, 22.0, -8.0, 20.0, 12.0),   // rear high trees
        (5.0, 8.0, 25.0, 12.0, 7.0),     // front approach
    ];

    for _i in 0..count {
        // Pick a random zone
        let zi = (random_double() * zones.len() as f32) as usize % zones.len();
        let (cx, cy, cz, rx, rz) = zones[zi];

        // Random position in ellipsoid
        let angle = random_double() * 6.28318530718;
        let r = random_double().sqrt();
        let h = (random_double() * 2.0 - 1.0) * 0.7;
        let px = cx + r * angle.cos() * rx;
        let py = cy + h * 15.0;
        let pz = cz + r * angle.sin() * rz;

        // Sway + wind drift
        let wind_x = (random_double() - 0.5) * 3.0;
        let wind_z = (random_double() - 0.5) * 3.0;
        let fall_speed = 18.0 + random_double() * 22.0;
        let fall_dist = fall_speed * shutter;

        let p0x = px;
        let p0y = py + random_double() * 5.0;
        let p0z = pz;
        let p1x = px + wind_x;
        let p1y = py - fall_dist;
        let p1z = pz + wind_z;

        let radius = 0.25 + random_double() * 0.45;

        // BVH sphere: center = midpoint, radius covers full sweep
        let mid_x = (p0x + p1x) * 0.5;
        let mid_y = (p0y + p1y) * 0.5;
        let mid_z = (p0z + p1z) * 0.5;
        let sweep = ((p1x-p0x).powi(2) + (p1y-p0y).powi(2) + (p1z-p0z).powi(2)).sqrt();
        let bvh_radius = radius + sweep * 0.5;

        particles.push(GpuParticle {
            pos_t0: [p0x, p0y, p0z, 0.0],
            pos_t1: [p1x, p1y, p1z, 0.0],
            radius,
            material_id,
            _pad: [0; 2],
        });
        spheres.push(GpuSphere {
            center: [mid_x, mid_y, mid_z, 0.0],
            radius: bvh_radius,
            material_id,
            _pad: [0; 2],
        });
    }

    // Near-camera bokeh particles (large, close)
    let bokeh_count = 80u32;
    for _i in 0..bokeh_count {
        let px = (random_double() - 0.5) * 50.0;
        let py = 5.0 + random_double() * 20.0;
        let pz = 40.0 + random_double() * 25.0;

        let fall_dist = (12.0 + random_double() * 15.0) * shutter;
        let wind_x = (random_double() - 0.5) * 2.0;
        let wind_z = (random_double() - 0.5) * 2.0;

        let radius = 1.2 + random_double() * 2.5;
        let p0x = px; let p0y = py; let p0z = pz;
        let p1x = px + wind_x; let p1y = py - fall_dist; let p1z = pz + wind_z;

        let mid = [(p0x+p1x)*0.5, (p0y+p1y)*0.5, (p0z+p1z)*0.5, 0.0];
        let sweep = ((p1x-p0x).powi(2) + (p1y-p0y).powi(2) + (p1z-p0z).powi(2)).sqrt();

        particles.push(GpuParticle {
            pos_t0: [p0x, p0y, p0z, 0.0], pos_t1: [p1x, p1y, p1z, 0.0],
            radius, material_id, _pad: [0; 2],
        });
        spheres.push(GpuSphere {
            center: mid, radius: radius + sweep * 0.5, material_id, _pad: [0; 2],
        });
    }

    println!("Generated {} petal particles ({} bokeh)", count + bokeh_count, bokeh_count);
    (particles, spheres)
}

fn create_scene_environment(
    obj_path: &str,
) -> (
    Vec<GpuSphere>,
    Vec<GpuTriangle>,
    Vec<GpuMaterial>,
    Vec<u32>,
    Vec<GpuTexture>,
    Vec<f32>,
) {
    // Base materials — same defaults as the character scene
    let mut materials = vec![
        GpuMaterial { albedo: [0.5, 0.5, 0.5, 0.0], fuzz: 0.0, ref_idx: 1.0, material_type: 0, tex_id: 0 }, // 0: ground grey
        GpuMaterial { albedo: [0.4, 0.2, 0.1, 0.0], fuzz: 0.0, ref_idx: 1.0, material_type: 0, tex_id: 0 }, // 1: brown diffuse
        GpuMaterial { albedo: [0.7, 0.6, 0.5, 0.0], fuzz: 0.0, ref_idx: 1.0, material_type: 1, tex_id: 0 }, // 2: metal
        GpuMaterial { albedo: [1.0, 1.0, 1.0, 0.0], fuzz: 0.0, ref_idx: 1.5, material_type: 2, tex_id: 0 }, // 3: dielectric
        GpuMaterial { albedo: [15.0, 15.0, 15.0, 0.0], fuzz: 0.0, ref_idx: 1.0, material_type: 3, tex_id: 0 }, // 4: light (placeholder, will be overridden)
    ];
    let default_mat_count = materials.len() as u32;

    // Load OBJ
    let (mut triangles, obj_mats, obj_mat_names) = load_obj(obj_path, 0);

    // Append OBJ materials — all start as diffuse, white albedo (texture provides colour)
    for m in obj_mats {
        materials.push(m);
    }

    // Load scene textures from Tex/ directory
    let obj_dir = Path::new(obj_path).parent().unwrap_or(Path::new("."));
    let (tex_info, tex_data_vec) = load_scene_textures(obj_dir);

    // Assign tex_id by material NAME number (材質N → N, uses tex_map)
    // Material properties verified against Ray渲染材质包/*.fx
    for (i, name) in obj_mat_names.iter().enumerate() {
        let mat_idx = default_mat_count as usize + i;
        let m = &mut materials[mat_idx];
        // Parse material name: 材質N → N
        let tid = if name.starts_with("材質") {
            name[6..].parse::<u32>().unwrap_or(0)
        } else {
            0
        };
        m.tex_id = tid;
        m.material_type = 0; // default: diffuse
        m.fuzz = 0.0;
        m.ref_idx = 1.0;

        match tid {
            // ── Clear Coat (type 4): lacquered wood — sharp Fresnel coat + diffuse base ──
            1 | 3 | 7 | 8 | 11 | 34 => { m.material_type = 4; m.ref_idx = 1.5; m.fuzz = 0.70; m.albedo = [0.55, 0.55, 0.55, 0.0]; } // matte lacquer

            // ── Metal/Glossy (type 1) ──
            16 => { m.material_type = 1; m.fuzz = 0.2; }   // fx: smoothness=0.8
            35 => { m.material_type = 1; m.fuzz = 0.3; }   // fx: smoothness=0.7 + α=Tex_0006_A

            // ── Dielectric/Glass (type 2) ──
            17 => { m.material_type = 2; m.ref_idx = 1.45; m.fuzz = 0.0; }  // fx: smooth=1.0, baseSpec=0.02
            33 => { m.material_type = 2; m.ref_idx = 1.333; m.fuzz = 0.15; m.albedo = [0.1, 0.25, 0.4, 0.0]; }  // Water: IOR=1.333 + wave + Beer

            // ── SSS Petal (type 7): Random Walk subsurface scattering ──
            14 | 31 | 39 | 40 => { m.material_type = 7; m.fuzz = 0.5; m.albedo = [1.0, 0.75, 0.7, 0.0]; }

            // ── Procedural Stone (type 5): bump-mapped rock surface ──
            5 | 10 | 21..=30 | 38 => { m.material_type = 5; m.fuzz = 0.6; }

            // ── Procedural Wood (type 6): anisotropic grain + bump ──
            19 | 20 | 32 => { m.material_type = 6; m.fuzz = 0.5; }

            _ => {} // all others: diffuse (type 0), smoothness=0.0
        }

            let type_name = match m.material_type {
                1 => "METAL",
                2 => "GLASS",
                4 => "CLEARCOAT",
                5 => "STONE",
                6 => "WOOD",
                7 => "SSS",
                _ => if m.fuzz > 0.0 { "TRANSLUCENT" } else { "DIFFUSE" },
            };
        println!("  {} → tex_id {} ({})", name, tid, type_name);
    }

    // Offset triangle material IDs past the default materials
    if !triangles.is_empty() {
        for t in triangles.iter_mut() {
            t.material_id += default_mat_count;
        }
    }

    // Bounding box of the scene triangles
    let mut bmin = [f32::MAX; 3];
    let mut bmax = [f32::MIN; 3];
    for t in &triangles {
        for a in 0..3 {
            bmin[a] = bmin[a].min(t.v0[a]).min(t.v1[a]).min(t.v2[a]);
            bmax[a] = bmax[a].max(t.v0[a]).max(t.v1[a]).max(t.v2[a]);
        }
    }
    let cx = (bmin[0] + bmax[0]) / 2.0;
    // let cy = (bmin[1] + bmax[1]) / 2.0;
    let cz = (bmin[2] + bmax[2]) / 2.0;
    let model_size = (bmax[0] - bmin[0])
        .max(bmax[1] - bmin[1])
        .max(bmax[2] - bmin[2]);
    println!(
        "Scene bbox: ({:.1},{:.1},{:.1}) - ({:.1},{:.1},{:.1})  size={:.1}",
        bmin[0], bmin[1], bmin[2], bmax[0], bmax[1], bmax[2], model_size
    );

    // ── Scene lights: placed outside the camera frustum ──
    // Camera is at ~(cx+0.42*s, bmin+0.9*s, cz+1.46*s) looking at ~(cx, bmin+0.4*s, cz)
    // FOV 40°.  Lights go above, behind (+z), and to the far sides.
    let mut spheres: Vec<GpuSphere> = Vec::new();
    let mut lights: Vec<u32> = Vec::new();

    let mut add_light = |center: [f32; 3], radius: f32, albedo: [f32; 3]| {
        let mat_id = materials.len() as u32;
        materials.push(GpuMaterial {
            albedo: [albedo[0], albedo[1], albedo[2], 0.0],
            fuzz: 0.0, ref_idx: 1.0, material_type: 3, tex_id: 0,
        });
        let idx = spheres.len() as u32;
        spheres.push(GpuSphere {
            center: [center[0], center[1], center[2], 0.0],
            radius, material_id: mat_id, _pad: [0; 2],
        });
        lights.push(idx);
    };

    let s = model_size;

    // Single sun light — placed far in the dusk sun direction
    // Sun dir: ~(0.5, 0.15, -0.7) — low western sky at dusk
    let sun_dir = [0.5_f32, 0.15, -0.7];
    let sun_len = (sun_dir[0]*sun_dir[0] + sun_dir[1]*sun_dir[1] + sun_dir[2]*sun_dir[2]).sqrt();
    let sun_dx = sun_dir[0] / sun_len;
    let sun_dy = sun_dir[1] / sun_len;
    let sun_dz = sun_dir[2] / sun_len;
    let sun_dist = s * 10.0;
    add_light([cx + sun_dx * sun_dist, bmax[1] + sun_dy * sun_dist + s * 0.5, cz + sun_dz * sun_dist],
              s * 3.0, [40.0, 30.0, 18.0]);  // warm dusk sun

    println!(
        "Scene: {} spheres, {} triangles, {} materials, {} lights",
        spheres.len(),
        triangles.len(),
        materials.len(),
        lights.len()
    );

    (spheres, triangles, materials, lights, tex_info, tex_data_vec)
}

fn validate_bvh(nodes: &[GpuBvhNode], sphere_total: u32, tri_total: u32) {
    const INTERNAL_FLAG: u32 = 0x80000000;
    for (i, node) in nodes.iter().enumerate() {
        if (node.primitive_count & INTERNAL_FLAG) != 0 {
            // Internal node: has left and right child pointers
            let left = node.left_or_first;
            let right = node.primitive_count & !INTERNAL_FLAG;
            if left as usize >= nodes.len() || right as usize >= nodes.len() {
                panic!(
                    "BVH node {} has invalid children: left={}, right={}, total={}",
                    i, left, right, nodes.len()
                );
            }
        } else {
            // Leaf node: references primitives
            let first = node.left_or_first & 0x7FFFFFFFu32;
            let is_tri = (node.left_or_first >> 31) != 0;
            let max = if is_tri { tri_total } else { sphere_total };
            if first + node.primitive_count > max {
                panic!(
                    "BVH node {} leaf has out-of-range {} indices: first={}, count={}, total={}",
                    i, if is_tri { "tri" } else { "sphere" }, first, node.primitive_count, max
                );
            }
        }
    }
}

async fn run() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    // Explicitly pick the NVIDIA discrete GPU (not integrated)
    let adapter = instance
        .enumerate_adapters(wgpu::Backends::PRIMARY)
        .into_iter()
        .find(|a| {
            let name = a.get_info().name.to_lowercase();
            name.contains("nvidia") || name.contains("rtx")
        })
        .or_else(|| {
            instance
                .enumerate_adapters(wgpu::Backends::PRIMARY)
                .into_iter()
                .max_by_key(|a| match a.get_info().device_type {
                    wgpu::DeviceType::DiscreteGpu => 2,
                    wgpu::DeviceType::IntegratedGpu => 1,
                    _ => 0,
                })
        })
        .expect("No GPU adapter found");
    println!(
        "GPU: {} ({:?})",
        adapter.get_info().name,
        adapter.get_info().device_type
    );
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: 12,
                    max_buffer_size: 1073741824,
                    max_storage_buffer_binding_size: 1073741824,
                    ..Default::default()
                },
                ..Default::default()
            },
            None,
        )
        .await
        .expect("Failed to create GPU device");

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Raytracer Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
    });

    // ── CLI: "cargo run -- scene" renders the shrine, default = character ──
    let args: Vec<String> = std::env::args().collect();
    let render_scene = args.len() > 1 && args[1] == "scene";

    let mut image_width: u32 = 6400;
    let mut image_height: u32 = 6400;
    let mut samples_per_pixel: u32 = 200;
    let mut max_depth: u32 = 20;

    // Scene-mode: smaller output for faster iteration
    if render_scene {
        image_width = 3200;
        image_height = 3200;
        samples_per_pixel = 128;
        max_depth = 50;
    }

    let (mut spheres, triangles, mut materials, lights, tex_info, tex_data_vec) = if render_scene {
        let scene_path = if Path::new("../assets/Scene.obj").exists() {
            "../assets/Scene.obj"
        } else if Path::new("assets/Scene.obj").exists() {
            "assets/Scene.obj"
        } else {
            eprintln!("Scene.obj not found!");
            return;
        };
        println!("━━━ Rendering SCENE: {} ━━━", scene_path);
        create_scene_environment(scene_path)
    } else {
        let obj_path = if Path::new("../assets/ayaka.obj").exists() {
            Some("../assets/ayaka.obj")
        } else if Path::new("assets/ayaka.obj").exists() {
            Some("assets/ayaka.obj")
        } else {
            None
        };
        println!("━━━ Rendering CHARACTER ━━━");
        create_scene(obj_path)
    };

    let has_model = !triangles.is_empty();

    let bg: [f32; 4] = if render_scene {
        [0.45, 0.55, 0.75, 0.0] // slightly deeper sky blue for outdoor scene
    } else {
        [0.5, 0.7, 0.9, 0.0]
    };

    let cam_origin: [f32; 3];
    let cam_lookat: [f32; 3];
    let cam_vfov: f32;

    if has_model {
        let mut bmin = [f32::MAX; 3];
        let mut bmax = [f32::MIN; 3];
        for t in &triangles {
            for a in 0..3 {
                bmin[a] = bmin[a].min(t.v0[a]).min(t.v1[a]).min(t.v2[a]);
                bmax[a] = bmax[a].max(t.v0[a]).max(t.v1[a]).max(t.v2[a]);
            }
        }
        let cx = (bmin[0] + bmax[0]) / 2.0;
        let cy = (bmin[1] + bmax[1]) / 2.0;
        let cz = (bmin[2] + bmax[2]) / 2.0;

        if render_scene {
            // ── Scene camera: fill 90% of frame with the model ──
            let model_size = (bmax[0] - bmin[0])
                .max(bmax[1] - bmin[1])
                .max(bmax[2] - bmin[2]);
            // FOV 40° — natural architectural perspective
            // dist so that model_size occupies 90% of the vertical frame
            cam_vfov = 40.0;
            let half_h = (cam_vfov / 2.0).to_radians().tan();
            let desired_visible = model_size / 1.08; // model overfills the frame
            let dist = (desired_visible / 2.0) / half_h;
            // Elevated 25° front view
            let angle = 25.0_f32.to_radians();
            let elev = model_size * 0.4;
            cam_origin = [
                cx + dist * angle.sin(),
                bmin[1] + model_size * 0.5 + elev,
                cz + dist * angle.cos(),
            ];
            cam_lookat = [cx, bmin[1] + model_size * 0.4, cz];
        } else {
            // ── Character camera: portrait 45° front-right ──
            let dist = (bmax[2] - bmin[2]).max(bmax[1] - bmin[1]) * 2.5 + 2.0;
            let angle = 45.0_f32.to_radians();
            cam_origin = [
                cx + dist * angle.sin(),
                cy + (bmax[1] - bmin[1]) * 0.15,
                cz + dist * angle.cos(),
            ];
            cam_lookat = [cx, cy, cz];
            cam_vfov = 28.0;

            spheres.push(GpuSphere {
                center: [cx, bmin[1] - 2.0, cz, 0.0],
                radius: 2.0,
                material_id: 0,
                _pad: [0; 2],
            });
        }
    } else {
        cam_origin = [13.0, 2.0, 3.0];
        cam_lookat = [0.0, 0.0, 0.0];
        cam_vfov = 20.0;
    }

    let debug_spheres_only = false;
    let use_triangles: Vec<GpuTriangle> = if debug_spheres_only { Vec::new() } else { triangles.clone() };

    if spheres.is_empty() && use_triangles.is_empty() {
        eprintln!("Nothing to render");
        return;
    }

    // ── Generate petal particles (DISABLED) ──
    let particle_count: u32 = 0;
    let particle_offset: u32 = spheres.len() as u32;
    let particle_vec: Vec<GpuParticle> = Vec::new();
    // let petal_mat_id = materials.len() as u32;
    // materials.push(GpuMaterial { ... });
    // let (particle_vec, particle_spheres) = generate_petal_particles(&use_triangles, petal_mat_id);
    // spheres.extend(particle_spheres);

    let mut final_spheres = spheres.clone();
    let mut final_tris = use_triangles.clone();
    let final_bvh = build_all_bvh_reordered(&mut final_spheres, &mut final_tris);
    validate_bvh(&final_bvh, final_spheres.len() as u32, final_tris.len() as u32);

    let uniforms = GpuUniforms {
        image_width,
        image_height,
        samples_per_pixel,
        max_depth,
        sphere_count: final_spheres.len() as u32,
        triangle_count: final_tris.len() as u32,
        bvh_node_count: final_bvh.len() as u32,
        light_count: lights.len() as u32,
        particle_count,
        particle_offset,
        _pad_particle: [0; 2],
        background: bg,
        tex_count: tex_info.len() as u32,
        batch_offset: 0,
        batch_count: 0,
        tile_start_x: 0,
        tile_start_y: 0,
        tile_end_x: 0,
        tile_end_y: 0,
        _pad1: 0,
        sun_dir: [0.580, 0.174, -0.812, 0.0], // dusk sun: low western sky
    };

    println!(
        "GPU Raytracer: {}x{} {}spp, {} spheres, {} triangles, {} BVH nodes",
        image_width, image_height, samples_per_pixel, spheres.len(), triangles.len(), final_bvh.len()
    );

    let camera = GpuCamera::new(
        cam_origin, cam_lookat, [0.0, 1.0, 0.0],
        cam_vfov, 1.0, 0.0, 10.0,
    );

    let output_size = (image_width * image_height) as u64;
    // Create and zero-initialize the output buffer (shader accumulates into it)
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: output_size * 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });
    {
        let mut mapping = output_buffer.slice(..).get_mapped_range_mut();
        mapping.fill(0);
    }
    output_buffer.unmap();
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: output_size * 16,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create uniform buffer (will be updated each batch via queue.write_buffer)
    let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Uniform Buffer"),
        size: std::mem::size_of::<GpuUniforms>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Camera Buffer"),
        contents: bytemuck::cast_slice(&[camera]),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let dummy_sphere = GpuSphere { center: [0.0;4], radius:0.0, material_id:0, _pad:[0;2] };
    let sphere_slice: &[GpuSphere] = if final_spheres.is_empty() { std::slice::from_ref(&dummy_sphere) } else { &final_spheres };
    let sphere_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Sphere Buffer"),
        contents: bytemuck::cast_slice(sphere_slice),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let dummy_tri = GpuTriangle {
        v0:[0.0;4], v1:[0.0;4], v2:[0.0;4],
        n0:[0.0;4], n1:[0.0;4], n2:[0.0,1.0,0.0,0.0],
        uv0:[0.0;2], uv1:[0.0;2], uv2:[0.0;2],
        material_id:0, _pad:0,
    };
    let tri_slice: &[GpuTriangle] = if final_tris.is_empty() { std::slice::from_ref(&dummy_tri) } else { &final_tris };
    let tri_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Triangle Buffer"),
        contents: bytemuck::cast_slice(tri_slice),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let material_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Material Buffer"),
        contents: bytemuck::cast_slice(&materials),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let bvh_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("BVH Buffer"),
        contents: bytemuck::cast_slice(&final_bvh),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let light_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Light Buffer"),
        contents: bytemuck::cast_slice(&lights),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Texture metadata buffer
    let tex_slice: &[GpuTexture] = if tex_info.is_empty() {
        &[GpuTexture { data_offset: 0, width: 0, height: 0, channels: 3 }]
    } else {
        &tex_info
    };
    let tex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Texture Buffer"),
        contents: bytemuck::cast_slice(tex_slice),
        usage: wgpu::BufferUsages::STORAGE,
    });
    // Texture pixel data buffer
    let tex_data_slice: &[f32] = if tex_data_vec.is_empty() { &[0.0, 0.0, 0.0] } else { &tex_data_vec };
    let tex_data_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Texture Data Buffer"),
        contents: bytemuck::cast_slice(tex_data_slice),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Particle buffer (dummy if disabled)
    let dummy_particle = GpuParticle {
        pos_t0: [0.0; 4], pos_t1: [0.0; 4], radius: 0.0, material_id: 0, _pad: [0; 2],
    };
    let particle_slice: &[GpuParticle] = if particle_vec.is_empty() { std::slice::from_ref(&dummy_particle) } else { &particle_vec };
    let particle_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Particle Buffer"),
        contents: bytemuck::cast_slice(particle_slice),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: camera_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: sphere_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: tri_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: bvh_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: material_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: light_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: tex_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: tex_data_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: particle_buf.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });

    let start = std::time::Instant::now();

    // Tile-based batch rendering: split image into tiles, split samples into batches.
    // Each dispatch = one tile × one sample batch, keeping per-dispatch work low.
    let tile_size: u32 = 512;
    let samples_per_batch: u32 = 8;
    let num_batches = (samples_per_pixel + samples_per_batch - 1) / samples_per_batch;

    let mut total_dispatches: u32 = 0;
    for ty in (0..image_height).step_by(tile_size as usize) {
        let tile_h = (ty + tile_size).min(image_height) - ty;
        for tx in (0..image_width).step_by(tile_size as usize) {
            let tile_w = (tx + tile_size).min(image_width) - tx;
            let wg_x = (tile_w + 7) / 8;
            let wg_y = (tile_h + 7) / 8;

            for batch in 0..num_batches {
                let batch_offset = batch * samples_per_batch;
                let batch_count = (samples_per_pixel - batch_offset).min(samples_per_batch);

                let batch_uniforms = GpuUniforms {
                    batch_offset,
                    batch_count,
                    tile_start_x: tx,
                    tile_start_y: ty,
                    tile_end_x: tx + tile_w,
                    tile_end_y: ty + tile_h,
                    ..uniforms
                };
                queue.write_buffer(&uniform_buffer, 0, bytemuck::cast_slice(&[batch_uniforms]));

                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Raytracing Pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&compute_pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.dispatch_workgroups(wg_x, wg_y, 1);
                }
                queue.submit(Some(encoder.finish()));
                device.poll(wgpu::Maintain::Wait);
                total_dispatches += 1;
            }
        }
    }

    println!("Total dispatches: {}", total_dispatches);

    // Copy accumulated output to staging buffer and read back
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size * 16);
    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (tx, rx) = mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).ok();
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let data = buffer_slice.get_mapped_range();
    let pixels: &[f32] = bytemuck::cast_slice(&data);

    let elapsed = start.elapsed();
    println!("Render time: {:.1}s", elapsed.as_secs_f64());

    // Apply gamma correction on CPU (shader outputs raw accumulated color)
    let scale = 1.0 / samples_per_pixel as f32;
    let mut img: image::RgbImage = image::ImageBuffer::new(image_width, image_height);
    for y in 0..image_height {
        for x in 0..image_width {
            let idx = (y * image_width + x) as usize;
            // Linear HDR colour
            let lr = pixels[idx * 4] * scale;
            let lg = pixels[idx * 4 + 1] * scale;
            let lb = pixels[idx * 4 + 2] * scale;
            // Luminance-preserving Reinhard tone mapping (preserves colour ratios)
            let lum = 0.2126 * lr + 0.7152 * lg + 0.0722 * lb;
            let mapped_lum = lum / (1.0 + lum);
            let scale_tm = if lum > 0.0001 { mapped_lum / lum } else { 1.0 };
            let tr = lr * scale_tm;
            let tg = lg * scale_tm;
            let tb = lb * scale_tm;
            // sRGB gamma (≈2.2)
            let r = (256.0 * tr.powf(1.0/2.2).clamp(0.0, 0.999)) as u8;
            let g = (256.0 * tg.powf(1.0/2.2).clamp(0.0, 0.999)) as u8;
            let b = (256.0 * tb.powf(1.0/2.2).clamp(0.0, 0.999)) as u8;
            img.put_pixel(x, y, image::Rgb([r, g, b]));
        }
    }
    drop(data);
    staging_buffer.unmap();

    let path = if render_scene { "output/gpu/image.png" } else { "output/gpu/image2.png" };
    std::fs::create_dir_all("output/gpu").ok();
    img.save(path).expect("Cannot save image");
    println!("Output: {}", path);
}

fn main() {
    pollster::block_on(run());
}
