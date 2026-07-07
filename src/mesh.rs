use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;

use crate::hittable::Hittable;
use crate::material::Material;
use crate::triangle::Triangle;
use crate::vec3::{Point3, Vec3};

use crate::image_texture::ImageTexture;
use crate::lambertian::Lambertian;
use crate::solid_color::SolidColor;
use crate::texture::Texture;
use crate::vec3::Color;

pub struct Mesh {
    pub triangles: Vec<Arc<dyn Hittable>>,
}

fn parse_mtl(mtl_path: &str) -> HashMap<String, Arc<dyn Material>> {
    let mut materials: HashMap<String, Arc<dyn Material>> = HashMap::new();
    let mtl_dir = Path::new(mtl_path).parent().map(|p| p.to_path_buf());

    let file = match File::open(mtl_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("  Cannot open MTL file {}: {}", mtl_path, e);
            return materials;
        }
    };

    let reader = BufReader::new(file);
    let mut current_name = String::new();
    let mut current_kd = Color::new(0.8, 0.8, 0.8);
    let mut current_map_kd = String::new();

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "newmtl" if parts.len() >= 2 => {
                if !current_name.is_empty() {
                    let mat = build_material(
                        &current_kd,
                        &current_map_kd,
                        mtl_dir.as_ref(),
                    );
                    materials.insert(current_name.clone(), mat);
                }
                current_name = parts[1].to_string();
                current_kd = Color::new(0.8, 0.8, 0.8);
                current_map_kd.clear();
            }
            "Kd" if parts.len() >= 4 => {
                current_kd = Color::new(
                    parts[1].parse().unwrap_or(0.8),
                    parts[2].parse().unwrap_or(0.8),
                    parts[3].parse().unwrap_or(0.8),
                );
            }
            "map_Kd" if parts.len() >= 2 => {
                current_map_kd = parts[1..].join(" ");
            }
            _ => {}
        }
    }

    if !current_name.is_empty() {
        let mat = build_material(&current_kd, &current_map_kd, mtl_dir.as_ref());
        println!(
            "  MTL material '{}': Kd=({:.2},{:.2},{:.2}) map='{}'",
            current_name, current_kd.x, current_kd.y, current_kd.z, current_map_kd
        );
        materials.insert(current_name, mat);
    }

    materials
}

fn build_material(
    kd: &Color,
    map_kd: &str,
    mtl_dir: Option<&std::path::PathBuf>,
) -> Arc<dyn Material> {
    if !map_kd.is_empty() {
        let tex_path = if let Some(dir) = mtl_dir {
            dir.join(map_kd).to_string_lossy().to_string()
        } else {
            map_kd.to_string()
        };
        let tex: Arc<dyn Texture> = Arc::new(ImageTexture::new(&tex_path));
        Arc::new(Lambertian::new(tex))
    } else {
        Arc::new(Lambertian::new(Arc::new(SolidColor::new(*kd))))
    }
}

impl Mesh {
    pub fn from_obj(path: &str, default_material: Arc<dyn Material>) -> Self {
        let file = match File::open(path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Cannot open OBJ file {}: {}", path, e);
                return Self {
                    triangles: Vec::new(),
                };
            }
        };

        let obj_dir = Path::new(path)
            .parent()
            .map(|p| p.to_path_buf());

        let reader = BufReader::new(file);
        let mut vertices: Vec<Point3> = Vec::new();
        let mut normals: Vec<Vec3> = Vec::new();
        let mut triangles: Vec<Arc<dyn Hittable>> = Vec::new();

        let mut materials: HashMap<String, Arc<dyn Material>> = HashMap::new();
        let mut current_material: Arc<dyn Material> = Arc::clone(&default_material);

        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => continue,
            };
            let line = line.trim();

            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();

            match parts[0] {
                "mtllib" if parts.len() >= 2 => {
                    let mtl_name = parts[1..].join(" ");
                    let mtl_path = if let Some(ref dir) = obj_dir {
                        dir.join(&mtl_name).to_string_lossy().to_string()
                    } else {
                        mtl_name
                    };
                    println!("Loading MTL: {}", mtl_path);
                    materials = parse_mtl(&mtl_path);
                    println!("  -> {} materials loaded", materials.len());
                }
                "usemtl" if parts.len() >= 2 => {
                    let name = parts[1];
                    if let Some(mat) = materials.get(name) {
                        current_material = Arc::clone(mat);
                    } else {
                        println!("  usemtl '{}' not found in MTL", name);
                    }
                }
                "v" if parts.len() >= 4 => {
                    let x: f64 = parts[1].parse().unwrap_or(0.0);
                    let y: f64 = parts[2].parse().unwrap_or(0.0);
                    let z: f64 = parts[3].parse().unwrap_or(0.0);
                    vertices.push(Point3::new(x, y, z));
                }
                "vn" if parts.len() >= 4 => {
                    let x: f64 = parts[1].parse().unwrap_or(0.0);
                    let y: f64 = parts[2].parse().unwrap_or(0.0);
                    let z: f64 = parts[3].parse().unwrap_or(0.0);
                    normals.push(Vec3::new(x, y, z));
                }
                "f" if parts.len() >= 4 => {
                    let face_vertices: Vec<&str> = parts[1..].to_vec();
                    let mut tri_verts: Vec<(usize, Option<usize>)> = Vec::new();

                    for fv in &face_vertices {
                        let comps: Vec<&str> = fv.split('/').collect();
                        let vi: usize = comps[0]
                            .parse::<i64>()
                            .map(|n| {
                                if n < 0 {
                                    (vertices.len() as i64 + n) as usize
                                } else {
                                    (n - 1) as usize
                                }
                            })
                            .unwrap_or(0);
                        let ni = if comps.len() >= 3 && !comps[2].is_empty() {
                            comps[2]
                                .parse::<i64>()
                                .map(|n| {
                                    if n < 0 {
                                        (normals.len() as i64 + n) as usize
                                    } else {
                                        (n - 1) as usize
                                    }
                                })
                                .ok()
                        } else {
                            None
                        };
                        tri_verts.push((vi, ni));
                    }

                    for k in 1..tri_verts.len() - 1 {
                        let (i0, n0_idx) = tri_verts[0];
                        let (i1, n1_idx) = tri_verts[k];
                        let (i2, n2_idx) = tri_verts[k + 1];

                        if i0 >= vertices.len()
                            || i1 >= vertices.len()
                            || i2 >= vertices.len()
                        {
                            continue;
                        }

                        let n0 = n0_idx
                            .and_then(|idx| normals.get(idx).copied())
                            .unwrap_or(Vec3::new(0.0, 1.0, 0.0));
                        let n1 = n1_idx
                            .and_then(|idx| normals.get(idx).copied())
                            .unwrap_or(Vec3::new(0.0, 1.0, 0.0));
                        let n2 = n2_idx
                            .and_then(|idx| normals.get(idx).copied())
                            .unwrap_or(Vec3::new(0.0, 1.0, 0.0));

                        let tri = Arc::new(Triangle::new(
                            vertices[i0],
                            vertices[i1],
                            vertices[i2],
                            n0,
                            n1,
                            n2,
                            Arc::clone(&current_material),
                        ));
                        triangles.push(tri);
                    }
                }
                _ => {}
            }
        }

        Self { triangles }
    }

    pub fn count(&self) -> usize {
        self.triangles.len()
    }
}
