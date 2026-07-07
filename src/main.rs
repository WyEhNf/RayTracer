mod aabb;
mod box_object;
mod bvh;
mod camera;
mod checker_texture;
mod constant_medium;
mod dielectric;
mod diffuse_light;
mod hittable;
mod hittable_list;
mod image_texture;
mod instance;
mod isotropic;
mod lambertian;
mod material;
mod mesh;
mod metal;
mod onb;
mod pdf;
mod perlin;
mod ray;
mod rect;
mod solid_color;
mod sphere;
mod texture;
mod triangle;
mod utils;
mod vec3;

use std::sync::Arc;

use console::style;
use image::{ImageBuffer, RgbImage};
use indicatif::ProgressBar;

use bvh::BvhNode;
use camera::Camera;
use diffuse_light::DiffuseLight;
use hittable::Hittable;
use hittable_list::HittableList;
use instance::{FlipFace, RotateY};
use lambertian::Lambertian;
use material::{Material, ScatterType};
use mesh::Mesh;
use pdf::{HittablePdf, MixturePdf, Pdf};
use ray::Ray;
use solid_color::SolidColor;
use vec3::{Color, Point3, Vec3};

use crate::rect::XzRect;
use crate::utils::random_double;

fn ray_color(
    ray: &Ray,
    background: &Color,
    world: &dyn Hittable,
    lights: Arc<dyn Hittable>,
    depth: u32,
) -> Color {
    if depth == 0 {
        return Color::new(0.0, 0.0, 0.0);
    }

    if let Some(rec) = world.hit(ray, 0.001, f64::INFINITY) {
        let emitted = rec.material.emitted(rec.u, rec.v, &rec.p);

        match rec.material.scatter(ray, &rec) {
            Some(ScatterType::Specular {
                attenuation,
                scattered_ray,
            }) => {
                return emitted
                    + attenuation
                        * ray_color(
                            &scattered_ray,
                            background,
                            world,
                            Arc::clone(&lights),
                            depth - 1,
                        );
            }
            Some(ScatterType::Diffuse { attenuation, pdf }) => {
                let light_pdf: Arc<dyn Pdf> =
                    Arc::new(HittablePdf::new(rec.p, Arc::clone(&lights)));
                let mixture: Arc<dyn Pdf> =
                    Arc::new(MixturePdf::new(Arc::clone(&pdf) as Arc<dyn Pdf>, light_pdf));

                let scattered_dir = mixture.generate();
                let pdf_val = mixture.value(&scattered_dir);

                if pdf_val < 1e-12 {
                    return emitted;
                }

                let scattered_ray = Ray::new_at_time(rec.p, scattered_dir, ray.time);
                let scattering_pdf =
                    rec.material.scattering_pdf(ray, &rec, &scattered_dir);

                let brdf_color = attenuation
                    * scattering_pdf
                    * ray_color(
                        &scattered_ray,
                        background,
                        world,
                        Arc::clone(&lights),
                        depth - 1,
                    )
                    / pdf_val;

                return emitted + brdf_color;
            }
            None => return emitted,
        }
    }

    let t = 0.5 * (ray.direction.y / ray.direction.length() + 1.0);
    (1.0 - t) * Color::new(0.02, 0.02, 0.05) + t * Color::new(0.05, 0.05, 0.15)
}

fn write_color(pixel: &mut image::Rgb<u8>, color: &Color, samples_per_pixel: u32) {
    let scale = 1.0 / samples_per_pixel as f64;
    let r = (color.x * scale).sqrt();
    let g = (color.y * scale).sqrt();
    let b = (color.z * scale).sqrt();

    let r = (256.0 * r.clamp(0.0, 0.999)) as u8;
    let g = (256.0 * g.clamp(0.0, 0.999)) as u8;
    let b = (256.0 * b.clamp(0.0, 0.999)) as u8;

    *pixel = image::Rgb([r, g, b]);
}

fn compute_bbox(triangles: &[Arc<dyn Hittable>]) -> (Point3, Point3) {
    let mut bmin = Point3::new(f64::MAX, f64::MAX, f64::MAX);
    let mut bmax = Point3::new(f64::MIN, f64::MIN, f64::MIN);
    for t in triangles.iter().take(200) {
        if let Some(bb) = t.bounding_box(0.0, 1.0) {
            bmin.x = bmin.x.min(bb.min.x);
            bmin.y = bmin.y.min(bb.min.y);
            bmin.z = bmin.z.min(bb.min.z);
            bmax.x = bmax.x.max(bb.max.x);
            bmax.y = bmax.y.max(bb.max.y);
            bmax.z = bmax.z.max(bb.max.z);
        }
    }
    (bmin, bmax)
}

fn main() {
    let obj_path = "assets/ayaka.obj";

    let model_mat: Arc<dyn Material> =
        Arc::new(Lambertian::new(Arc::new(SolidColor::new(Color::new(
            0.85, 0.82, 0.78,
        )))));

    let mesh = Mesh::from_obj(obj_path, Arc::clone(&model_mat));
    let tris = mesh.triangles;
    println!("Loaded {} triangles", tris.len());

    let (sample_bb_min, sample_bb_max) = compute_bbox(&tris);
    println!(
        "Model bbox: ({:.2},{:.2},{:.2}) - ({:.2},{:.2},{:.2})",
        sample_bb_min.x, sample_bb_min.y, sample_bb_min.z,
        sample_bb_max.x, sample_bb_max.y, sample_bb_max.z,
    );

    let cx = (sample_bb_min.x + sample_bb_max.x) / 2.0;
    let cy = (sample_bb_min.y + sample_bb_max.y) / 2.0;
    let cz = (sample_bb_min.z + sample_bb_max.z) / 2.0;
    let sy = sample_bb_max.y - sample_bb_min.y;
    let sz = sample_bb_max.z - sample_bb_min.z;
    let model_size = sy.max(sz);

    let mut world = HittableList::new();
    let mut lights = HittableList::new();

    let mut tri_objects = tris;
    println!("Building BVH for {} triangles...", tri_objects.len());
    let mesh_bvh = BvhNode::build(&mut tri_objects, 0.0, 1.0);
    let rotated = RotateY::new(mesh_bvh, 180.0);
    world.add(Arc::new(rotated));

    let ground_y = sample_bb_min.y - model_size * 0.02;
    let ground_mat: Arc<dyn Material> =
        Arc::new(Lambertian::new(Arc::new(SolidColor::new(Color::new(
            0.18, 0.18, 0.22,
        )))));
    world.add(Arc::new(XzRect::new(
        cx - model_size * 3.0, cx + model_size * 3.0,
        cz - model_size * 3.0, cz + model_size * 3.0,
        ground_y, ground_mat,
    )));

    let key_light: Arc<dyn Material> =
        Arc::new(DiffuseLight::new(Arc::new(SolidColor::new(Color::new(
            40.0, 37.0, 30.0,
        )))));
    let key_rect: Arc<dyn Hittable> = Arc::new(FlipFace::new(Arc::new(XzRect::new(
        cx - model_size * 0.6, cx + model_size * 0.6,
        cz - model_size * 2.0, cz - model_size * 1.2,
        sample_bb_max.y + model_size * 0.6, Arc::clone(&key_light),
    ))));
    world.add(Arc::clone(&key_rect));
    lights.add(key_rect);

    let fill_light: Arc<dyn Material> =
        Arc::new(DiffuseLight::new(Arc::new(SolidColor::new(Color::new(
            12.0, 14.0, 20.0,
        )))));
    let fill_rect: Arc<dyn Hittable> = Arc::new(FlipFace::new(Arc::new(XzRect::new(
        cx + model_size * 0.8, cx + model_size * 1.5,
        cz - model_size * 1.5, cz + model_size * 0.5,
        sample_bb_max.y + model_size * 0.3, Arc::clone(&fill_light),
    ))));
    world.add(Arc::clone(&fill_rect));
    lights.add(fill_rect);

    let rim_light: Arc<dyn Material> =
        Arc::new(DiffuseLight::new(Arc::new(SolidColor::new(Color::new(
            30.0, 30.0, 35.0,
        )))));
    let rim_rect: Arc<dyn Hittable> = Arc::new(FlipFace::new(Arc::new(XzRect::new(
        cx - model_size * 0.5, cx + model_size * 0.5,
        cz + model_size * 1.2, cz + model_size * 2.0,
        sample_bb_max.y + model_size * 0.5, Arc::clone(&rim_light),
    ))));
    world.add(Arc::clone(&rim_rect));
    lights.add(rim_rect);

    let aspect_ratio = 1.0;
    let image_width: u32 = 1200;
    let image_height: u32 = (image_width as f64 / aspect_ratio) as u32;
    let image_height = if image_height < 1 { 1 } else { image_height };

    let samples_per_pixel: u32 = 2000;
    let max_depth: u32 = 40;

    let path = std::path::Path::new("output/portrait/ayaka.png");
    let prefix = path.parent().unwrap();
    std::fs::create_dir_all(prefix).expect("Cannot create all the parents");

    let mut img: RgbImage = ImageBuffer::new(image_width, image_height);

    let cam_dist = model_size * 3.5;
    let lookfrom = Point3::new(cx, cy + model_size * 0.15, cz - cam_dist);
    let lookat = Point3::new(cx, cy + model_size * 0.1, cz);
    let vup = Vec3::new(0.0, 1.0, 0.0);
    let focus_dist = cam_dist;
    let defocus_angle = 0.0;
    let vfov = 28.0;
    let time0 = 0.0;
    let time1 = 1.0;

    let camera = Camera::new(
        lookfrom, lookat, vup, vfov, aspect_ratio, defocus_angle, focus_dist, time0,
        time1,
    );

    println!(
        "Camera: lookfrom=({:.1},{:.1},{:.1}) lookat=({:.1},{:.1},{:.1}) vfov={:.0}",
        lookfrom.x, lookfrom.y, lookfrom.z, lookat.x, lookat.y, lookat.z, vfov
    );

    let mut objects: Vec<Arc<dyn Hittable>> = Vec::new();
    std::mem::swap(&mut world.objects, &mut objects);
    let world: Arc<dyn Hittable> = BvhNode::build(&mut objects, time0, time1);
    let lights: Arc<dyn Hittable> = Arc::new(lights);

    let progress = if option_env!("CI").unwrap_or_default() == "true" {
        ProgressBar::hidden()
    } else {
        ProgressBar::new(image_height as u64)
    };

    println!(
        "Rendering {}x{} {}spp depth={}...",
        image_width, image_height, samples_per_pixel, max_depth
    );

    for j in (0..image_height).rev() {
        for i in 0..image_width {
            let mut pixel_color = Color::new(0.0, 0.0, 0.0);
            for _ in 0..samples_per_pixel {
                let u = (i as f64 + random_double()) / (image_width - 1) as f64;
                let v = ((image_height - 1 - j) as f64 + random_double())
                    / (image_height - 1) as f64;
                let ray = camera.get_ray(u, v);
                pixel_color += ray_color(&ray, &Color::new(0.0, 0.0, 0.0), &*world, Arc::clone(&lights), max_depth);
            }
            let pixel = img.get_pixel_mut(i, j);
            write_color(pixel, &pixel_color, samples_per_pixel);
        }
        progress.inc(1);
    }
    progress.finish();

    println!(
        "Output image as \"{}\"",
        style(path.to_str().unwrap()).yellow()
    );
    img.save(path).expect("Cannot save the image to the file");
}
