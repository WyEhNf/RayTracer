use std::sync::Arc;

use crate::aabb::Aabb;
use crate::hittable::{HitRecord, Hittable};
use crate::material::Material;
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3, dot};

fn get_sphere_uv(p: &Point3) -> (f64, f64) {
    let theta = (-p.y).acos();
    let phi = (-p.z).atan2(p.x) + std::f64::consts::PI;
    let u = phi / (2.0 * std::f64::consts::PI);
    let v = theta / std::f64::consts::PI;
    (u, v)
}

pub struct Sphere {
    center: Ray,
    radius: f64,
    material: Arc<dyn Material>,
}

impl Sphere {
    pub fn new_static(center: Point3, radius: f64, material: Arc<dyn Material>) -> Self {
        Self {
            center: Ray::new(center, Vec3::new(0.0, 0.0, 0.0)),
            radius,
            material,
        }
    }

    pub fn new_moving(
        center1: Point3,
        center2: Point3,
        radius: f64,
        material: Arc<dyn Material>,
    ) -> Self {
        Self {
            center: Ray::new(center1, center2 - center1),
            radius,
            material,
        }
    }
}

impl Hittable for Sphere {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let current_center = self.center.at(ray.time);
        let oc = ray.origin - current_center;
        let a = ray.direction.length_squared();
        let half_b = dot(&oc, &ray.direction);
        let c = oc.length_squared() - self.radius * self.radius;

        let discriminant = half_b * half_b - a * c;
        if discriminant < 0.0 {
            return None;
        }

        let sqrtd = discriminant.sqrt();

        let mut root = (-half_b - sqrtd) / a;
        if root < t_min || root > t_max {
            root = (-half_b + sqrtd) / a;
            if root < t_min || root > t_max {
                return None;
            }
        }

        let p = ray.at(root);
        let outward_normal = (p - current_center) / self.radius;
        let (u, v) = get_sphere_uv(&outward_normal);

        let mut rec = HitRecord {
            p,
            normal: outward_normal,
            t: root,
            u,
            v,
            front_face: false,
            material: Arc::clone(&self.material),
        };
        rec.set_face_normal(ray, &outward_normal);

        Some(rec)
    }

    fn bounding_box(&self, time0: f64, time1: f64) -> Option<Aabb> {
        let center0 = self.center.at(time0);
        let center1 = self.center.at(time1);
        let r = self.radius.abs();
        let rvec = Vec3::new(r, r, r);
        let box0 = Aabb::new(center0 - rvec, center0 + rvec);
        let box1 = Aabb::new(center1 - rvec, center1 + rvec);
        Some(Aabb::surrounding(&box0, &box1))
    }
}
