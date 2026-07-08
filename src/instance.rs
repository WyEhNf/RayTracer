use std::sync::Arc;

use crate::aabb::Aabb;
use crate::hittable::{HitRecord, Hittable};
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3};

pub struct Translate {
    ptr: Arc<dyn Hittable>,
    offset: Vec3,
}

impl Translate {
    pub fn new(ptr: Arc<dyn Hittable>, offset: Vec3) -> Self {
        Self { ptr, offset }
    }
}

impl Hittable for Translate {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let moved_ray = Ray::new_at_time(ray.origin - self.offset, ray.direction, ray.time);
        self.ptr.hit(&moved_ray, t_min, t_max).map(|mut rec| {
            rec.p += self.offset;
            rec.set_face_normal(&moved_ray, &rec.normal);
            rec
        })
    }

    fn bounding_box(&self, time0: f64, time1: f64) -> Option<Aabb> {
        self.ptr.bounding_box(time0, time1).map(|b| {
            Aabb::new(b.min + self.offset, b.max + self.offset)
        })
    }
}

pub struct RotateY {
    ptr: Arc<dyn Hittable>,
    sin_theta: f64,
    cos_theta: f64,
    bbox: Option<Aabb>,
}

impl RotateY {
    pub fn new(ptr: Arc<dyn Hittable>, angle: f64) -> Self {
        let radians = angle.to_radians();
        let sin_theta = radians.sin();
        let cos_theta = radians.cos();
        let bbox = ptr.bounding_box(0.0, 1.0).map(|b| {
            let mut min = Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
            let mut max = Point3::new(
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
            );
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        let x = i as f64 * b.max.x + (1 - i) as f64 * b.min.x;
                        let y = j as f64 * b.max.y + (1 - j) as f64 * b.min.y;
                        let z = k as f64 * b.max.z + (1 - k) as f64 * b.min.z;
                        let new_x = cos_theta * x + sin_theta * z;
                        let new_z = -sin_theta * x + cos_theta * z;
                        min.x = min.x.min(new_x);
                        max.x = max.x.max(new_x);
                        min.y = min.y.min(y);
                        max.y = max.y.max(y);
                        min.z = min.z.min(new_z);
                        max.z = max.z.max(new_z);
                    }
                }
            }
            Aabb::new(min, max)
        });
        Self {
            ptr,
            sin_theta,
            cos_theta,
            bbox,
        }
    }
}

impl Hittable for RotateY {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let mut origin = ray.origin;
        let mut direction = ray.direction;
        origin.x = self.cos_theta * ray.origin.x - self.sin_theta * ray.origin.z;
        origin.z = self.sin_theta * ray.origin.x + self.cos_theta * ray.origin.z;
        direction.x = self.cos_theta * ray.direction.x - self.sin_theta * ray.direction.z;
        direction.z = self.sin_theta * ray.direction.x + self.cos_theta * ray.direction.z;
        let rotated_ray = Ray::new_at_time(origin, direction, ray.time);

        self.ptr.hit(&rotated_ray, t_min, t_max).map(|mut rec| {
            let mut p = rec.p;
            let mut normal = rec.normal;
            p.x = self.cos_theta * rec.p.x + self.sin_theta * rec.p.z;
            p.z = -self.sin_theta * rec.p.x + self.cos_theta * rec.p.z;
            normal.x = self.cos_theta * rec.normal.x + self.sin_theta * rec.normal.z;
            normal.z = -self.sin_theta * rec.normal.x + self.cos_theta * rec.normal.z;
            rec.p = p;
            rec.set_face_normal(&rotated_ray, &normal);
            rec
        })
    }

    fn bounding_box(&self, _time0: f64, _time1: f64) -> Option<Aabb> {
        self.bbox.clone()
    }
}

pub struct FlipFace {
    ptr: Arc<dyn Hittable>,
}

impl FlipFace {
    pub fn new(ptr: Arc<dyn Hittable>) -> Self {
        Self { ptr }
    }
}

impl Hittable for FlipFace {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        self.ptr.hit(ray, t_min, t_max).map(|mut rec| {
            rec.front_face = !rec.front_face;
            rec
        })
    }

    fn bounding_box(&self, time0: f64, time1: f64) -> Option<Aabb> {
        self.ptr.bounding_box(time0, time1)
    }

    fn pdf_value(&self, origin: &Point3, direction: &Vec3) -> f64 {
        self.ptr.pdf_value(origin, direction)
    }

    fn random(&self, origin: &Point3) -> Vec3 {
        self.ptr.random(origin)
    }
}
