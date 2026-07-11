
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
use checker_texture::CheckerTexture;
use constant_medium::ConstantMedium;
use dielectric::Dielectric;
use diffuse_light::DiffuseLight;
use hittable::Hittable;
use hittable_list::HittableList;
use instance::{FlipFace, RotateY, Translate};
use isotropic::Isotropic;
use lambertian::Lambertian;
use material::{Material, ScatterType};
use metal::Metal;
use pdf::{HittablePdf, MixturePdf, Pdf};
use ray::Ray;
use solid_color::SolidColor;
use sphere::Sphere;
use vec3::{Color, Point3, Vec3};
use crate::utils::{random_double, random_range};

fn ray_color(
    ray: &Ray,
    world: &dyn Hittable,
    lights: Arc<dyn Hittable>,
    depth: u32,
) -> Color {
    if depth == 0 { return Color::new(0.0, 0.0, 0.0); }

    if let Some(rec) = world.hit(ray, 0.001, f64::INFINITY) {
        let emitted = rec.material.emitted(rec.u, rec.v, &rec.p);
        match rec.material.scatter(ray, &rec) {
            Some(ScatterType::Specular { attenuation, scattered_ray }) => {
                return emitted + attenuation * ray_color(&scattered_ray, world, Arc::clone(&lights), depth - 1);
            }
            Some(ScatterType::Diffuse { attenuation, pdf }) => {
                let light_pdf = Arc::new(HittablePdf::new(rec.p, Arc::clone(&lights)));
                let mixture = Arc::new(MixturePdf::new(Arc::clone(&pdf) as Arc<dyn Pdf>, light_pdf));
                let scattered_dir = mixture.generate();
                let pdf_val = mixture.value(&scattered_dir);
                if pdf_val < 1e-12 { return emitted; }
                let scattered_ray = Ray::new_at_time(rec.p, scattered_dir, ray.time);
                let scattering_pdf = rec.material.scattering_pdf(ray, &rec, &scattered_dir);
                let brdf_color = attenuation * scattering_pdf
                    * ray_color(&scattered_ray, world, Arc::clone(&lights), depth - 1) / pdf_val;
                return emitted + brdf_color;
            }
            None => return emitted,
        }
    }
    let t = 0.5 * (ray.direction.y / ray.direction.length() + 1.0);
    (1.0 - t) * Color::new(0.35, 0.35, 0.5) + t * Color::new(0.7, 0.7, 0.9)
}

fn write_color(pixel: &mut image::Rgb<u8>, color: &Color, samples_per_pixel: u32) {
    let scale = 1.0 / samples_per_pixel as f64;
    let r = (256.0 * (color.x * scale).sqrt().clamp(0.0, 0.999)) as u8;
    let g = (256.0 * (color.y * scale).sqrt().clamp(0.0, 0.999)) as u8;
    let b = (256.0 * (color.z * scale).sqrt().clamp(0.0, 0.999)) as u8;
    *pixel = image::Rgb([r, g, b]);
}

fn render(world: &dyn Hittable, lights: Arc<dyn Hittable>, cam: &Camera,
          image_width: u32, aspect_ratio: f64, samples_per_pixel: u32, max_depth: u32,
          output_path: &str) {
    let image_height = ((image_width as f64 / aspect_ratio) as u32).max(1);
    let prefix = std::path::Path::new(output_path).parent().unwrap();
    std::fs::create_dir_all(prefix).expect("Cannot create dirs");
    let mut img: RgbImage = ImageBuffer::new(image_width, image_height);
    let progress = ProgressBar::new(image_height as u64);
    println!("  {}x{} {}spp depth={}", image_width, image_height, samples_per_pixel, max_depth);
    for j in (0..image_height).rev() {
        for i in 0..image_width {
            let mut pixel_color = Color::new(0.0, 0.0, 0.0);
            for _ in 0..samples_per_pixel {
                let u = (i as f64 + random_double()) / (image_width - 1) as f64;
                let v = ((image_height - 1 - j) as f64 + random_double()) / (image_height - 1) as f64;
                let ray = cam.get_ray(u, v);
                pixel_color += ray_color(&ray, world, Arc::clone(&lights), max_depth);
            }
            write_color(img.get_pixel_mut(i, j), &pixel_color, samples_per_pixel);
        }
        progress.inc(1);
    }
    progress.finish();
    img.save(output_path).expect("Cannot save");
    println!("  -> {}", style(output_path).yellow());
}

// ===== BOOK 1 CHAPTER SCENES =====

fn b1_ch5_surface_normals() -> (HittableList, HittableList, Camera) {
    let mut world = HittableList::new();
    world.add(Arc::new(Sphere::new_static(Point3::new(0.0, 0.0, -1.0), 0.5, Arc::new(Lambertian::new(Arc::new(SolidColor::new(Color::new(0.7, 0.3, 0.3))))))));
    world.add(Arc::new(Sphere::new_static(Point3::new(0.0, -100.5, -1.0), 100.0, Arc::new(Lambertian::new(Arc::new(SolidColor::new(Color::new(0.8, 0.8, 0.0))))))));
    let lights = HittableList::new();
    let cam = Camera::new(Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 0.0, -1.0), Vec3::new(0.0, 1.0, 0.0), 90.0, 16.0/9.0, 0.0, 1.0, 0.0, 1.0);
    (world, lights, cam)
}

fn b1_ch6_7_diffuse() -> (HittableList, HittableList, Camera) {
    let mut world = HittableList::new();
    world.add(Arc::new(Sphere::new_static(Point3::new(0.0, 0.0, -1.0), 0.5, Arc::new(Lambertian::new(Arc::new(SolidColor::new(Color::new(0.7, 0.3, 0.3))))))));
    world.add(Arc::new(Sphere::new_static(Point3::new(0.0, -100.5, -1.0), 100.0, Arc::new(Lambertian::new(Arc::new(SolidColor::new(Color::new(0.8, 0.8, 0.0))))))));
    let lights = HittableList::new();
    let cam = Camera::new(Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 0.0, -1.0), Vec3::new(0.0, 1.0, 0.0), 90.0, 16.0/9.0, 0.0, 1.0, 0.0, 1.0);
    (world, lights, cam)
}

fn b1_ch9_metal() -> (HittableList, HittableList, Camera) {
    let mut world = HittableList::new();
    let mat_ground = Arc::new(Lambertian::new(Arc::new(SolidColor::new(Color::new(0.8, 0.8, 0.0)))));
    let mat_center = Arc::new(Lambertian::new(Arc::new(SolidColor::new(Color::new(0.7, 0.3, 0.3)))));
    let mat_left = Arc::new(Metal::new(Color::new(0.8, 0.8, 0.8), 0.3));
    let mat_right = Arc::new(Metal::new(Color::new(0.8, 0.6, 0.2), 1.0));
    world.add(Arc::new(Sphere::new_static(Point3::new(0.0, -100.5, -1.0), 100.0, mat_ground)));
    world.add(Arc::new(Sphere::new_static(Point3::new(0.0, 0.0, -1.0), 0.5, mat_center)));
    world.add(Arc::new(Sphere::new_static(Point3::new(-1.0, 0.0, -1.0), 0.5, mat_left)));
    world.add(Arc::new(Sphere::new_static(Point3::new(1.0, 0.0, -1.0), 0.5, mat_right)));
    let lights = HittableList::new();
    let cam = Camera::new(Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 0.0, -1.0), Vec3::new(0.0, 1.0, 0.0), 90.0, 16.0/9.0, 0.0, 1.0, 0.0, 1.0);
    (world, lights, cam)
}

fn b1_ch10_dielectric() -> (HittableList, HittableList, Camera) {
    let mut world = HittableList::new();
    world.add(Arc::new(Sphere::new_static(Point3::new(0.0, -100.5, -1.0), 100.0, Arc::new(Lambertian::new(Arc::new(SolidColor::new(Color::new(0.8, 0.8, 0.0))))))));
    world.add(Arc::new(Sphere::new_static(Point3::new(0.0, 0.0, -1.0), 0.5, Arc::new(Lambertian::new(Arc::new(SolidColor::new(Color::new(0.1, 0.2, 0.5))))))));
    world.add(Arc::new(Sphere::new_static(Point3::new(-1.0, 0.0, -1.0), 0.5, Arc::new(Dielectric::new(1.5)))));
    world.add(Arc::new(Sphere::new_static(Point3::new(-1.0, 0.0, -1.0), -0.4, Arc::new(Dielectric::new(1.5)))));
    world.add(Arc::new(Sphere::new_static(Point3::new(1.0, 0.0, -1.0), 0.5, Arc::new(Metal::new(Color::new(0.8, 0.6, 0.2), 0.0)))));
    let lights = HittableList::new();
    let cam = Camera::new(Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 0.0, -1.0), Vec3::new(0.0, 1.0, 0.0), 90.0, 16.0/9.0, 0.0, 1.0, 0.0, 1.0);
    (world, lights, cam)
}

fn b1_ch11_camera() -> (HittableList, HittableList, Camera) {
    let mut world = HittableList::new();
    world.add(Arc::new(Sphere::new_static(Point3::new(0.0, -100.5, -1.0), 100.0, Arc::new(Lambertian::new(Arc::new(SolidColor::new(Color::new(0.8, 0.8, 0.0))))))));
    world.add(Arc::new(Sphere::new_static(Point3::new(0.0, 0.0, -1.0), 0.5, Arc::new(Lambertian::new(Arc::new(SolidColor::new(Color::new(0.1, 0.2, 0.5))))))));
    world.add(Arc::new(Sphere::new_static(Point3::new(-1.0, 0.0, -1.0), 0.5, Arc::new(Dielectric::new(1.5)))));
    world.add(Arc::new(Sphere::new_static(Point3::new(1.0, 0.0, -1.0), 0.5, Arc::new(Metal::new(Color::new(0.8, 0.6, 0.2), 0.0)))));
    let lights = HittableList::new();
    let cam = Camera::new(Point3::new(-2.0, 2.0, 1.0), Point3::new(0.0, 0.0, -1.0), Vec3::new(0.0, 1.0, 0.0), 20.0, 16.0/9.0, 0.0, 1.0, 0.0, 1.0);
    (world, lights, cam)
}

fn b1_ch12_defocus() -> (HittableList, HittableList, Camera) {
    let mut world = HittableList::new();
    world.add(Arc::new(Sphere::new_static(Point3::new(0.0, -100.5, -1.0), 100.0, Arc::new(Lambertian::new(Arc::new(SolidColor::new(Color::new(0.8, 0.8, 0.0))))))));
    world.add(Arc::new(Sphere::new_static(Point3::new(0.0, 0.0, -1.0), 0.5, Arc::new(Lambertian::new(Arc::new(SolidColor::new(Color::new(0.1, 0.2, 0.5))))))));
    world.add(Arc::new(Sphere::new_static(Point3::new(-1.0, 0.0, -1.0), 0.5, Arc::new(Dielectric::new(1.5)))));
    world.add(Arc::new(Sphere::new_static(Point3::new(1.0, 0.0, -1.0), 0.5, Arc::new(Metal::new(Color::new(0.8, 0.6, 0.2), 0.0)))));
    let lights = HittableList::new();
    let cam = Camera::new(Point3::new(3.0, 3.0, 2.0), Point3::new(0.0, 0.0, -1.0), Vec3::new(0.0, 1.0, 0.0), 20.0, 16.0/9.0, 0.6, 10.0, 0.0, 1.0);
    (world, lights, cam)
}

fn b1_ch13_final() -> (HittableList, HittableList, Camera) {
    let mut world = HittableList::new();
    let ground = Arc::new(Lambertian::new(Arc::new(CheckerTexture::new(0.32, Arc::new(SolidColor::new(Color::new(0.2, 0.3, 0.1))), Arc::new(SolidColor::new(Color::new(0.9, 0.9, 0.9)))))));
    world.add(Arc::new(Sphere::new_static(Point3::new(0.0, -1000.0, 0.0), 1000.0, ground)));
    for a in -11..11 {
        for b in -11..11 {
            let choose_mat = random_double();
            let center = Point3::new(a as f64 + 0.9*random_double(), 0.2, b as f64 + 0.9*random_double());
            if (center - Point3::new(4.0, 0.2, 0.0)).length() > 0.9 {
                if choose_mat < 0.8 {
                    let albedo = Color::new(random_double()*random_double(), random_double()*random_double(), random_double()*random_double());
                    world.add(Arc::new(Sphere::new_static(center, 0.2, Arc::new(Lambertian::new(Arc::new(SolidColor::new(albedo)))))));
                } else if choose_mat < 0.95 {
                    let albedo = Color::new(random_range(0.5,1.0), random_range(0.5,1.0), random_range(0.5,1.0));
                    let fuzz = crate::utils::random_double() * 0.5;
                    world.add(Arc::new(Sphere::new_static(center, 0.2, Arc::new(Metal::new(albedo, fuzz)))));
                } else {
                    world.add(Arc::new(Sphere::new_static(center, 0.2, Arc::new(Dielectric::new(1.5)))));
                }
            }
        }
    }
    world.add(Arc::new(Sphere::new_static(Point3::new(0.0, 1.0, 0.0), 1.0, Arc::new(Dielectric::new(1.5)))));
    world.add(Arc::new(Sphere::new_static(Point3::new(-4.0, 1.0, 0.0), 1.0, Arc::new(Lambertian::new(Arc::new(SolidColor::new(Color::new(0.4, 0.2, 0.1))))))));
    world.add(Arc::new(Sphere::new_static(Point3::new(4.0, 1.0, 0.0), 1.0, Arc::new(Metal::new(Color::new(0.7, 0.6, 0.5), 0.0)))));
    let lights = HittableList::new();
    let cam = Camera::new(Point3::new(13.0, 2.0, 3.0), Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 20.0, 16.0/9.0, 0.6, 10.0, 0.0, 1.0);
    (world, lights, cam)
}

// ===== BOOK 2 CHAPTER SCENES =====

fn b2_ch1_motion_blur() -> (HittableList, HittableList, Camera) {
    let mut world = HittableList::new();
    world.add(Arc::new(Sphere::new_static(Point3::new(0.0, -100.5, -1.0), 100.0, Arc::new(Lambertian::new(Arc::new(SolidColor::new(Color::new(0.8, 0.8, 0.0))))))));
    world.add(Arc::new(Sphere::new_static(Point3::new(-1.0, 0.0, -1.0), 0.5, Arc::new(Dielectric::new(1.5)))));
    world.add(Arc::new(Sphere::new_static(Point3::new(1.0, 0.0, -1.0), 0.5, Arc::new(Metal::new(Color::new(0.8, 0.6, 0.2), 0.0)))));
    // Moving sphere: center → center+right, motion blurred
    world.add(Arc::new(Sphere::new_moving(
        Point3::new(0.0, 0.0, -1.0),
        Point3::new(0.5, 0.0, -1.0),
        0.5,
        Arc::new(Lambertian::new(Arc::new(SolidColor::new(Color::new(0.1, 0.2, 0.5))))),
    )));
    let lights = HittableList::new();
    let cam = Camera::new(Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 0.0, -1.0), Vec3::new(0.0, 1.0, 0.0), 90.0, 16.0/9.0, 0.0, 1.0, 0.0, 1.0);
    (world, lights, cam)
}

fn b2_ch2_bvh() -> (HittableList, HittableList, Camera) {
    // Same as b1_ch13 but wrapped in BVH via the render function
    b1_ch13_final()
}

fn b2_ch3_textures() -> (HittableList, HittableList, Camera) {
    let mut world = HittableList::new();
    let checker = Arc::new(Lambertian::new(Arc::new(CheckerTexture::new(0.2, Arc::new(SolidColor::new(Color::new(0.2, 0.3, 0.1))), Arc::new(SolidColor::new(Color::new(0.9, 0.9, 0.9)))))));
    world.add(Arc::new(Sphere::new_static(Point3::new(0.0, -10.0, 0.0), 10.0, Arc::clone(&checker) as Arc<dyn Material>)));
    world.add(Arc::new(Sphere::new_static(Point3::new(0.0, 10.0, 0.0), 10.0, checker)));
    let lights = HittableList::new();
    let cam = Camera::new(Point3::new(13.0, 2.0, 3.0), Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 20.0, 16.0/9.0, 0.0, 10.0, 0.0, 1.0);
    (world, lights, cam)
}

fn b1_ch13_final_motion() -> (HittableList, HittableList, Camera) {
    let mut world = HittableList::new();
    let ground = Arc::new(Lambertian::new(Arc::new(CheckerTexture::new(0.32, Arc::new(SolidColor::new(Color::new(0.2, 0.3, 0.1))), Arc::new(SolidColor::new(Color::new(0.9, 0.9, 0.9)))))));
    world.add(Arc::new(Sphere::new_static(Point3::new(0.0, -1000.0, 0.0), 1000.0, ground)));
    for a in -11..11 {
        for b in -11..11 {
            let choose_mat = random_double();
            let center = Point3::new(a as f64 + 0.9*random_double(), 0.2, b as f64 + 0.9*random_double());
            if (center - Point3::new(4.0, 0.2, 0.0)).length() > 0.9 {
                if choose_mat < 0.8 {
                    let albedo = Color::new(random_double()*random_double(), random_double()*random_double(), random_double()*random_double());
                    let center2 = center + Vec3::new(0.0, random_range(0.0, 0.5), 0.0);
                    if random_double() < 0.5 {
                        world.add(Arc::new(Sphere::new_moving(center, center2, 0.2, Arc::new(Lambertian::new(Arc::new(SolidColor::new(albedo)))))));
                    } else {
                        world.add(Arc::new(Sphere::new_static(center, 0.2, Arc::new(Lambertian::new(Arc::new(SolidColor::new(albedo)))))));
                    }
                } else if choose_mat < 0.95 {
                    let albedo = Color::new(random_range(0.5,1.0), random_range(0.5,1.0), random_range(0.5,1.0));
                    let fuzz = random_double() * 0.5;
                    world.add(Arc::new(Sphere::new_static(center, 0.2, Arc::new(Metal::new(albedo, fuzz)))));
                } else {
                    world.add(Arc::new(Sphere::new_static(center, 0.2, Arc::new(Dielectric::new(1.5)))));
                }
            }
        }
    }
    world.add(Arc::new(Sphere::new_static(Point3::new(0.0, 1.0, 0.0), 1.0, Arc::new(Dielectric::new(1.5)))));
    world.add(Arc::new(Sphere::new_static(Point3::new(-4.0, 1.0, 0.0), 1.0, Arc::new(Lambertian::new(Arc::new(SolidColor::new(Color::new(0.4, 0.2, 0.1))))))));
    world.add(Arc::new(Sphere::new_static(Point3::new(4.0, 1.0, 0.0), 1.0, Arc::new(Metal::new(Color::new(0.7, 0.6, 0.5), 0.0)))));
    let lights = HittableList::new();
    let cam = Camera::new(Point3::new(13.0, 2.0, 3.0), Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 20.0, 16.0/9.0, 0.6, 10.0, 0.0, 1.0);
    (world, lights, cam)
}

fn b2_ch4_perlin() -> (HittableList, HittableList, Camera) {
    use crate::perlin::Perlin;
    use crate::texture::Texture;
    struct NoiseTexture { scale: f64, noise: Perlin }
    impl NoiseTexture { fn new(s: f64) -> Self { Self { scale: s, noise: Perlin::new() } } }
    impl Texture for NoiseTexture {
        fn value(&self, _u: f64, _v: f64, p: &Point3) -> Color {
            Color::new(1.0,1.0,1.0) * 0.5 * (1.0 + (self.scale * p.z + 10.0 * self.noise.turb(p, 7)).sin())
        }
    }
    let mut world = HittableList::new();
    let mut lights = HittableList::new();
    let noise_tex = Arc::new(NoiseTexture::new(4.0));
    world.add(Arc::new(Sphere::new_static(Point3::new(0.0, -1000.0, 0.0), 1000.0, Arc::new(Lambertian::new(Arc::clone(&noise_tex) as Arc<dyn Texture>)))));
    world.add(Arc::new(Sphere::new_static(Point3::new(0.0, 2.0, 0.0), 2.0, Arc::new(Lambertian::new(Arc::clone(&noise_tex) as Arc<dyn Texture>)))));
    let light_mat: Arc<dyn Material> = Arc::new(DiffuseLight::new(Arc::new(SolidColor::new(Color::new(4.0,4.0,4.0)))));
    let light_rect = Arc::new(crate::rect::XzRect::new(-3.0, 3.0, -3.0, 3.0, 5.0, Arc::clone(&light_mat)));
    let light_rect = Arc::new(FlipFace::new(light_rect));
    world.add(Arc::clone(&light_rect) as Arc<dyn Hittable>);
    lights.add(light_rect);
    let cam = Camera::new(Point3::new(13.0, 2.0, 3.0), Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 20.0, 16.0/9.0, 0.0, 10.0, 0.0, 1.0);
    (world, lights, cam)
}

fn b2_ch8_volumes() -> (HittableList, HittableList, Camera) {
    // Cornell box with blocks of smoke
    let mut world = HittableList::new();
    let mut lights = HittableList::new();
    let red = Arc::new(Lambertian::new(Arc::new(SolidColor::new(Color::new(0.65, 0.05, 0.05)))));
    let white = Arc::new(Lambertian::new(Arc::new(SolidColor::new(Color::new(0.73, 0.73, 0.73)))));
    let green = Arc::new(Lambertian::new(Arc::new(SolidColor::new(Color::new(0.12, 0.45, 0.15)))));
    let light_mat: Arc<dyn Material> = Arc::new(DiffuseLight::new(Arc::new(SolidColor::new(Color::new(15.0,15.0,15.0)))));
    // Cornell box walls
    world.add(Arc::new(crate::rect::YzRect::new(0.0, 555.0, 0.0, 555.0, 555.0, green)));
    world.add(Arc::new(crate::rect::YzRect::new(0.0, 555.0, 0.0, 555.0, 0.0, red)));
    let light_rect = Arc::new(crate::rect::XzRect::new(213.0, 343.0, 227.0, 332.0, 554.0, Arc::clone(&light_mat)));
    let light_rect = Arc::new(FlipFace::new(light_rect));
    world.add(Arc::clone(&light_rect) as Arc<dyn Hittable>);
    lights.add(light_rect);
    world.add(Arc::new(crate::rect::XzRect::new(0.0, 555.0, 0.0, 555.0, 0.0, Arc::clone(&white) as Arc<dyn Material>)));
    world.add(Arc::new(crate::rect::XzRect::new(0.0, 555.0, 0.0, 555.0, 555.0, Arc::clone(&white) as Arc<dyn Material>)));
    world.add(Arc::new(crate::rect::XyRect::new(0.0, 555.0, 0.0, 555.0, 555.0, Arc::clone(&white) as Arc<dyn Material>)));
    // Two smoke blocks
    let box1 = Arc::new(crate::box_object::BoxObject::new(Point3::new(0.0,0.0,0.0), Point3::new(165.0,330.0,165.0), Arc::clone(&white) as Arc<dyn Material>));
    let box1 = Arc::new(RotateY::new(box1, 15.0));
    let box1 = Arc::new(Translate::new(box1, Vec3::new(265.0,0.0,295.0)));
    world.add(Arc::new(ConstantMedium::new(box1, 0.01, Arc::new(Isotropic::new(Arc::new(SolidColor::new(Color::new(0.95,0.95,0.95))))))));
    let box2 = Arc::new(crate::box_object::BoxObject::new(Point3::new(0.0,0.0,0.0), Point3::new(165.0,165.0,165.0), Arc::clone(&white) as Arc<dyn Material>));
    let box2 = Arc::new(RotateY::new(box2, -18.0));
    let box2 = Arc::new(Translate::new(box2, Vec3::new(130.0,0.0,65.0)));
    world.add(Arc::new(ConstantMedium::new(box2, 0.01, Arc::new(Isotropic::new(Arc::new(SolidColor::new(Color::new(0.95,0.95,0.95))))))));
    let cam = Camera::new(Point3::new(278.0, 278.0, -800.0), Point3::new(278.0, 278.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 40.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    (world, lights, cam)
}

fn b2_ch5_image() -> (HittableList, HittableList, Camera) {
    use crate::image_texture::ImageTexture;
    let earth_tex = Arc::new(ImageTexture::new("assets/earthmap.jpg"));
    let earth = Arc::new(Lambertian::new(earth_tex));
    let mut world = HittableList::new();
    let lookfrom = Point3::new(13.0, 2.0, 3.0);
    let lookat = Point3::new(0.0, 0.0, 0.0);
    world.add(Arc::new(Sphere::new_static(Point3::new(0.0, 0.0, 0.0), 2.0, earth)));
    let lights = HittableList::new();
    let cam = Camera::new(lookfrom, lookat, Vec3::new(0.0, 1.0, 0.0), 20.0, 16.0/9.0, 0.0, 10.0, 0.0, 1.0);
    (world, lights, cam)
}

fn b2_ch6_lights() -> (HittableList, HittableList, Camera) {
    use crate::perlin::Perlin;
    use crate::rect::XyRect;
    use crate::texture::Texture;
    struct NoiseTexture { scale: f64, noise: Perlin }
    impl NoiseTexture { fn new(s: f64) -> Self { Self { scale: s, noise: Perlin::new() } } }
    impl Texture for NoiseTexture {
        fn value(&self, _u: f64, _v: f64, p: &Point3) -> Color {
            Color::new(1.0,1.0,1.0) * 0.5 * (1.0 + (self.scale * p.z + 10.0 * self.noise.turb(p, 7)).sin())
        }
    }
    let mut world = HittableList::new();
    let mut lights = HittableList::new();
    let noise_tex = Arc::new(NoiseTexture::new(4.0));
    let noise_mat: Arc<dyn Material> = Arc::new(Lambertian::new(Arc::clone(&noise_tex) as Arc<dyn Texture>));
    world.add(Arc::new(Sphere::new_static(Point3::new(0.0, -1000.0, 0.0), 1000.0, Arc::clone(&noise_mat))));
    world.add(Arc::new(Sphere::new_static(Point3::new(0.0, 2.0, 0.0), 2.0, Arc::clone(&noise_mat))));
    let light_mat: Arc<dyn Material> = Arc::new(DiffuseLight::new(Arc::new(SolidColor::new(Color::new(4.0,4.0,4.0)))));
    let light_rect = Arc::new(XyRect::new(3.0, 5.0, 1.0, 3.0, -2.0, Arc::clone(&light_mat)));
    world.add(Arc::clone(&light_rect) as Arc<dyn Hittable>);
    lights.add(light_rect);
    let cam = Camera::new(Point3::new(26.0, 3.0, 6.0), Point3::new(0.0, 2.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 20.0, 16.0/9.0, 0.0, 1.0, 0.0, 1.0);
    (world, lights, cam)
}

fn b2_ch7_instances() -> (HittableList, HittableList, Camera) {
    let mut world = HittableList::new();
    let mut lights = HittableList::new();
    let red = Arc::new(Lambertian::new(Arc::new(SolidColor::new(Color::new(0.65, 0.05, 0.05)))));
    let white = Arc::new(Lambertian::new(Arc::new(SolidColor::new(Color::new(0.73, 0.73, 0.73)))));
    let green = Arc::new(Lambertian::new(Arc::new(SolidColor::new(Color::new(0.12, 0.45, 0.15)))));
    let light_mat: Arc<dyn Material> = Arc::new(DiffuseLight::new(Arc::new(SolidColor::new(Color::new(7.0,7.0,7.0)))));
    // Cornell box walls
    world.add(Arc::new(crate::rect::YzRect::new(0.0, 555.0, 0.0, 555.0, 555.0, Arc::clone(&green) as Arc<dyn Material>)));
    world.add(Arc::new(crate::rect::YzRect::new(0.0, 555.0, 0.0, 555.0, 0.0, Arc::clone(&red) as Arc<dyn Material>)));
    let light_rect = Arc::new(crate::rect::XzRect::new(213.0, 343.0, 227.0, 332.0, 554.0, Arc::clone(&light_mat)));
    let light_rect = Arc::new(FlipFace::new(light_rect));
    world.add(Arc::clone(&light_rect) as Arc<dyn Hittable>);
    lights.add(light_rect);
    world.add(Arc::new(crate::rect::XzRect::new(0.0, 555.0, 0.0, 555.0, 0.0, Arc::clone(&white) as Arc<dyn Material>)));
    world.add(Arc::new(crate::rect::XzRect::new(0.0, 555.0, 0.0, 555.0, 555.0, Arc::clone(&white) as Arc<dyn Material>)));
    world.add(Arc::new(crate::rect::XyRect::new(0.0, 555.0, 0.0, 555.0, 555.0, Arc::clone(&white) as Arc<dyn Material>)));
    // Rotated boxes
    let box1 = Arc::new(crate::box_object::BoxObject::new(Point3::new(0.0,0.0,0.0), Point3::new(165.0,330.0,165.0), Arc::clone(&white) as Arc<dyn Material>));
    let box1 = Arc::new(RotateY::new(box1, 15.0));
    let box1 = Arc::new(Translate::new(box1, Vec3::new(265.0,0.0,295.0)));
    world.add(box1);
    let box2 = Arc::new(crate::box_object::BoxObject::new(Point3::new(0.0,0.0,0.0), Point3::new(165.0,165.0,165.0), white));
    let box2 = Arc::new(RotateY::new(box2, -18.0));
    let box2 = Arc::new(Translate::new(box2, Vec3::new(130.0,0.0,65.0)));
    world.add(box2);
    world.add(Arc::new(Sphere::new_static(Point3::new(400.0, 200.0, 400.0), 50.0, Arc::clone(&red) as Arc<dyn Material>)));
    world.add(Arc::new(Sphere::new_static(Point3::new(190.0, 90.0, 190.0), 90.0, Arc::clone(&green) as Arc<dyn Material>)));
    let cam = Camera::new(Point3::new(278.0, 278.0, -800.0), Point3::new(278.0, 278.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 40.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    (world, lights, cam)
}

fn b2_ch9_final() -> (HittableList, HittableList, Camera) {
    let mut world = HittableList::new();
    let mut lights = HittableList::new();
    let mut boxes1 = HittableList::new();
    let ground = Arc::new(Lambertian::new(Arc::new(SolidColor::new(Color::new(0.48, 0.83, 0.53)))));
    let white = Arc::new(Lambertian::new(Arc::new(SolidColor::new(Color::new(0.73, 0.73, 0.73)))));
    let light_mat: Arc<dyn Material> = Arc::new(DiffuseLight::new(Arc::new(SolidColor::new(Color::new(7.0,7.0,7.0)))));
    for i in 0..20 {
        for j in 0..20 {
            let w = 100.0;
            let x0 = -1000.0 + i as f64 * w;
            let z0 = -1000.0 + j as f64 * w;
            let y0 = 0.0;
            let x1 = x0 + w;
            let y1 = random_double() * 100.0 + 1.0;
            let z1 = z0 + w;
            if ((x0+x1)/2.0 - 400.0).hypot((z0+z1)/2.0 - 400.0) > 50.0 {
                boxes1.add(Arc::new(crate::box_object::BoxObject::new(Point3::new(x0,y0,z0), Point3::new(x1,y1,z1), Arc::clone(&ground) as Arc<dyn Material>)));
            }
        }
    }
    world.add(Arc::new(boxes1));
    let light_rect = Arc::new(crate::rect::XzRect::new(123.0, 423.0, 147.0, 412.0, 550.0, Arc::clone(&light_mat)));
    let light_rect = Arc::new(FlipFace::new(light_rect));
    world.add(Arc::clone(&light_rect) as Arc<dyn Hittable>);
    lights.add(light_rect);
    world.add(Arc::new(Sphere::new_static(Point3::new(260.0, 150.0, 45.0), 50.0, Arc::new(Dielectric::new(1.5)))));
    world.add(Arc::new(Sphere::new_static(Point3::new(0.0, 150.0, 145.0), 50.0, Arc::new(Metal::new(Color::new(0.8, 0.8, 0.9), 1.0)))));
    world.add(Arc::new(ConstantMedium::new(Arc::new(Sphere::new_static(Point3::new(360.0,150.0,145.0), 70.0, Arc::clone(&white) as Arc<dyn Material>)), 0.01, Arc::new(Isotropic::new(Arc::new(SolidColor::new(Color::new(0.2,0.4,0.9))))))));
    world.add(Arc::new(ConstantMedium::new(Arc::new(Sphere::new_static(Point3::new(0.0,0.0,0.0), 5000.0, Arc::clone(&white) as Arc<dyn Material>)), 0.0001, Arc::new(Isotropic::new(Arc::new(SolidColor::new(Color::new(1.0,1.0,1.0))))))));
    world.add(Arc::new(Sphere::new_static(Point3::new(400.0, 200.0, 400.0), 100.0, Arc::new(Lambertian::new(Arc::new(SolidColor::new(Color::new(0.7,0.3,0.3))))))));
    let cam = Camera::new(Point3::new(478.0, 278.0, -600.0), Point3::new(278.0, 278.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 40.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    (world, lights, cam)
}

// ===== MAIN =====

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let chapter = if args.len() > 1 { args[1].as_str() } else { "all" };

    let all_chapters: Vec<(&str, fn() -> (HittableList, HittableList, Camera), u32, u32, u32, &str)> = vec![
        ("book1/ch05_normals",      b1_ch5_surface_normals as fn() -> _,  1024, 100, 50, "output/book1/ch05_normals.png"),
        ("book1/ch06_diffuse",      b1_ch6_7_diffuse as fn() -> _,      1024, 100, 50, "output/book1/ch06_diffuse.png"),
        ("book1/ch09_metal",        b1_ch9_metal as fn() -> _,          1024, 100, 50, "output/book1/ch09_metal.png"),
        ("book1/ch10_dielectric",   b1_ch10_dielectric as fn() -> _,    1024, 100, 50, "output/book1/ch10_dielectric.png"),
        ("book1/ch11_camera",       b1_ch11_camera as fn() -> _,        1024, 100, 50, "output/book1/ch11_camera.png"),
        ("book1/ch12_defocus",      b1_ch12_defocus as fn() -> _,       1024, 100, 50, "output/book1/ch12_defocus.png"),
        ("book1/ch13_motion",       b1_ch13_final_motion as fn() -> _, 1024, 100, 50, "output/book1/ch13_motion.png"),
        ("book2/ch1_motion",        b2_ch1_motion_blur as fn() -> _,    1024, 100, 50, "output/book2/ch01_motion.png"),
        ("book2/ch3_textures",      b2_ch3_textures as fn() -> _,       1024, 100, 50, "output/book2/ch03_textures.png"),
        ("book2/ch4_perlin",        b2_ch4_perlin as fn() -> _,         1024, 100, 50, "output/book2/ch04_perlin.png"),
        ("book2/ch5_image",         b2_ch5_image as fn() -> _,          1024, 100, 50, "output/book2/ch05_image.png"),
        ("book2/ch6_lights",        b2_ch6_lights as fn() -> _,         1024, 100, 50, "output/book2/ch06_lights.png"),
        ("book2/ch7_instances",     b2_ch7_instances as fn() -> _,      1024, 100, 40, "output/book2/ch07_instances.png"),
        ("book2/ch8_volumes",       b2_ch8_volumes as fn() -> _,        1024, 100, 40, "output/book2/ch08_volumes.png"),
        ("book2/ch1_motion",        b2_ch1_motion_blur as fn() -> _,     400, 100, 50, "output/book2/ch01_motion.png"),
        ("book2/ch3_textures",      b2_ch3_textures as fn() -> _,        400, 100, 50, "output/book2/ch03_textures.png"),
        ("book2/ch4_perlin",        b2_ch4_perlin as fn() -> _,          400, 100, 50, "output/book2/ch04_perlin.png"),
        ("book2/ch5_image",         b2_ch5_image as fn() -> _,           400, 100, 50, "output/book2/ch05_image.png"),
        ("book2/ch6_lights",        b2_ch6_lights as fn() -> _,          400, 100, 50, "output/book2/ch06_lights.png"),
        ("book2/ch7_instances",     b2_ch7_instances as fn() -> _,       400, 100, 40, "output/book2/ch07_instances.png"),
        ("book2/ch8_volumes",       b2_ch8_volumes as fn() -> _,         400, 100, 40, "output/book2/ch08_volumes.png"),
    ];

    for (name, scene_fn, width, spp, depth, path) in &all_chapters {
        if chapter != "all" && *name != chapter { continue; }
        println!("\n=== {} ===", style(name).cyan());
        let (world, lights, cam) = scene_fn();
        let mut objects: Vec<Arc<dyn Hittable>> = Vec::new();
        let mut world_mut = world;
        std::mem::swap(&mut world_mut.objects, &mut objects);
        let bvh_world: Arc<dyn Hittable> = BvhNode::build(&mut objects, 0.0, 1.0);
        let lights_arc: Arc<dyn Hittable> = Arc::new(lights);
        let aspect = if name.contains("instances") || name.contains("volumes") || name.contains("final") { 1.0 } else { 16.0/9.0 };
        render(&*bvh_world, lights_arc, &cam, *width, aspect, *spp, *depth, path);
    }
}