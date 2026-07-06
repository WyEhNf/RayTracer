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
mod metal;
mod perlin;
mod ray;
mod rect;
mod solid_color;
mod sphere;
mod texture;
mod utils;
mod vec3;

use std::sync::Arc;

use console::style;
use image::{ImageBuffer, RgbImage};
use indicatif::ProgressBar;

use bvh::BvhNode;
use camera::Camera;
use constant_medium::ConstantMedium;
use dielectric::Dielectric;
use diffuse_light::DiffuseLight;
use hittable::Hittable;
use hittable_list::HittableList;
use image_texture::ImageTexture;
use instance::{FlipFace, RotateY, Translate};
use isotropic::Isotropic;
use lambertian::Lambertian;
use material::Material;
use metal::Metal;
use perlin::NoiseTexture;
use ray::Ray;
use solid_color::SolidColor;
use sphere::Sphere;
use vec3::{Color, Point3, Vec3};

use crate::box_object::BoxObject;
use crate::rect::XzRect;
use crate::utils::random_double;

fn ray_color(
    ray: &Ray,
    background: &Color,
    world: &dyn Hittable,
    depth: u32,
) -> Color {
    if depth == 0 {
        return Color::new(0.0, 0.0, 0.0);
    }

    if let Some(rec) = world.hit(ray, 0.001, f64::INFINITY) {
        let emitted = rec.material.emitted(rec.u, rec.v, &rec.p);
        if let Some(scatter) = rec.material.scatter(ray, &rec) {
            return emitted
                + scatter.attenuation
                    * ray_color(&scatter.scattered_ray, background, world, depth - 1);
        }
        return emitted;
    }

    *background
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

fn final_scene() -> HittableList {
    let ground: Arc<dyn Material> =
        Arc::new(Lambertian::new(Arc::new(SolidColor::new(Color::new(
            0.48, 0.83, 0.53,
        )))));

    let mut boxes1 = HittableList::new();
    let boxes_per_side = 20;
    for i in 0..boxes_per_side {
        for j in 0..boxes_per_side {
            let w = 100.0;
            let x0 = -1000.0 + i as f64 * w;
            let z0 = -1000.0 + j as f64 * w;
            let y0 = 0.0;
            let x1 = x0 + w;
            let y1 = crate::utils::random_range(1.0, 101.0);
            let z1 = z0 + w;

            boxes1.add(Arc::new(BoxObject::new(
                Point3::new(x0, y0, z0),
                Point3::new(x1, y1, z1),
                Arc::clone(&ground),
            )));
        }
    }

    let mut world = HittableList::new();

    let mut objects: Vec<Arc<dyn Hittable>> = Vec::new();
    std::mem::swap(&mut boxes1.objects, &mut objects);
    world.add(BvhNode::build(&mut objects, 0.0, 1.0));

    let light: Arc<dyn Material> =
        Arc::new(DiffuseLight::new(Arc::new(SolidColor::new(Color::new(
            7.0, 7.0, 7.0,
        )))));
    world.add(Arc::new(FlipFace::new(Arc::new(XzRect::new(
        123.0, 423.0, 147.0, 412.0, 554.0, light,
    )))));

    let center1 = Point3::new(400.0, 400.0, 200.0);
    let center2 = center1 + Vec3::new(30.0, 0.0, 0.0);
    let moving_sphere_material: Arc<dyn Material> =
        Arc::new(Lambertian::new(Arc::new(SolidColor::new(Color::new(
            0.7, 0.3, 0.1,
        )))));
    world.add(Arc::new(Sphere::new_moving(
        center1,
        center2,
        50.0,
        moving_sphere_material,
    )));

    world.add(Arc::new(Sphere::new_static(
        Point3::new(260.0, 150.0, 45.0),
        50.0,
        Arc::new(Dielectric::new(1.5)),
    )));

    world.add(Arc::new(Sphere::new_static(
        Point3::new(0.0, 150.0, 145.0),
        50.0,
        Arc::new(Metal::new(Color::new(0.8, 0.8, 0.9), 1.0)),
    )));

    let boundary: Arc<dyn Hittable> = Arc::new(Sphere::new_static(
        Point3::new(360.0, 150.0, 145.0),
        70.0,
        Arc::new(Dielectric::new(1.5)),
    ));
    world.add(Arc::clone(&boundary));
    world.add(Arc::new(ConstantMedium::new(
        Arc::clone(&boundary),
        0.2,
        Arc::new(Isotropic::new(Arc::new(SolidColor::new(Color::new(
            0.2, 0.4, 0.9,
        ))))),
    )));

    let mist_boundary: Arc<dyn Hittable> = Arc::new(Sphere::new_static(
        Point3::new(0.0, 0.0, 0.0),
        5000.0,
        Arc::new(Dielectric::new(1.5)),
    ));
    world.add(Arc::new(ConstantMedium::new(
        mist_boundary,
        0.0001,
        Arc::new(Isotropic::new(Arc::new(SolidColor::new(Color::new(
            1.0, 1.0, 1.0,
        ))))),
    )));

    let earth_texture = Arc::new(ImageTexture::new("assets/earthmap.jpg"));
    let earth_surface: Arc<dyn Material> = Arc::new(Lambertian::new(earth_texture));
    world.add(Arc::new(Sphere::new_static(
        Point3::new(400.0, 200.0, 400.0),
        100.0,
        earth_surface,
    )));

    let pertext = Arc::new(NoiseTexture::new(0.2));
    let noise_surface: Arc<dyn Material> = Arc::new(Lambertian::new(pertext));
    world.add(Arc::new(Sphere::new_static(
        Point3::new(220.0, 280.0, 300.0),
        80.0,
        noise_surface,
    )));

    let white: Arc<dyn Material> =
        Arc::new(Lambertian::new(Arc::new(SolidColor::new(Color::new(
            0.73, 0.73, 0.73,
        )))));
    let mut boxes2 = HittableList::new();
    let ns = 1000;
    for _ in 0..ns {
        let p = Point3::new(
            crate::utils::random_range(0.0, 165.0),
            crate::utils::random_range(0.0, 165.0),
            crate::utils::random_range(0.0, 165.0),
        );
        boxes2.add(Arc::new(Sphere::new_static(p, 10.0, Arc::clone(&white))));
    }

    let mut objects2: Vec<Arc<dyn Hittable>> = Vec::new();
    std::mem::swap(&mut boxes2.objects, &mut objects2);
    let bvh2 = BvhNode::build(&mut objects2, 0.0, 1.0);
    let rotated = Arc::new(RotateY::new(bvh2, 15.0));
    let translated = Arc::new(Translate::new(rotated, Vec3::new(-100.0, 270.0, 395.0)));
    world.add(translated);

    world
}

fn main() {
    let aspect_ratio = 1.0;
    let image_width: u32 = 800;
    let image_height: u32 = (image_width as f64 / aspect_ratio) as u32;
    let image_height = if image_height < 1 { 1 } else { image_height };

    let samples_per_pixel: u32 = 1000;
    let max_depth: u32 = 40;

    let background = Color::new(0.0, 0.0, 0.0);

    let path = std::path::Path::new("output/book2/image3.png");
    let prefix = path.parent().unwrap();
    std::fs::create_dir_all(prefix).expect("Cannot create all the parents");

    let mut img: RgbImage = ImageBuffer::new(image_width, image_height);

    let lookfrom = Point3::new(478.0, 278.0, -600.0);
    let lookat = Point3::new(278.0, 278.0, 0.0);
    let vup = Vec3::new(0.0, 1.0, 0.0);
    let focus_dist = 10.0;
    let defocus_angle = 0.0;
    let vfov = 40.0;
    let time0 = 0.0;
    let time1 = 1.0;

    let camera = Camera::new(
        lookfrom, lookat, vup, vfov, aspect_ratio, defocus_angle, focus_dist, time0,
        time1,
    );

    let mut world_scene = final_scene();
    let mut objects: Vec<Arc<dyn Hittable>> = Vec::new();
    std::mem::swap(&mut world_scene.objects, &mut objects);
    let world: Arc<dyn Hittable> = BvhNode::build(&mut objects, time0, time1);

    let progress = if option_env!("CI").unwrap_or_default() == "true" {
        ProgressBar::hidden()
    } else {
        ProgressBar::new(image_height as u64)
    };

    for j in (0..image_height).rev() {
        for i in 0..image_width {
            let mut pixel_color = Color::new(0.0, 0.0, 0.0);
            for _ in 0..samples_per_pixel {
                let u = (i as f64 + random_double()) / (image_width - 1) as f64;
                let v = ((image_height - 1 - j) as f64 + random_double())
                    / (image_height - 1) as f64;
                let ray = camera.get_ray(u, v);
                pixel_color += ray_color(&ray, &background, &*world, max_depth);
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
