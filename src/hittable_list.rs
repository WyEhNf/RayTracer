use std::sync::Arc;

use crate::aabb::Aabb;
use crate::hittable::{HitRecord, Hittable};
use crate::ray::Ray;
use crate::utils::random_double;
use crate::vec3::{Point3, Vec3};

pub struct HittableList {
    pub objects: Vec<Arc<dyn Hittable>>,
}

impl HittableList {
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
        }
    }

    pub fn add(&mut self, object: Arc<dyn Hittable>) {
        self.objects.push(object);
    }
}

impl Hittable for HittableList {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let mut closest = t_max;
        let mut result = None;

        for obj in &self.objects {
            if let Some(rec) = obj.hit(ray, t_min, closest) {
                closest = rec.t;
                result = Some(rec);
            }
        }

        result
    }

    fn bounding_box(&self, time0: f64, time1: f64) -> Option<Aabb> {
        if self.objects.is_empty() {
            return None;
        }
        let mut result: Option<Aabb> = None;
        for obj in &self.objects {
            if let Some(bbox) = obj.bounding_box(time0, time1) {
                result = Some(match result {
                    Some(r) => Aabb::surrounding(&r, &bbox),
                    None => bbox,
                });
            } else {
                return None;
            }
        }
        result
    }

    fn pdf_value(&self, origin: &Point3, direction: &Vec3) -> f64 {
        if self.objects.is_empty() {
            return 0.0;
        }
        let weight = 1.0 / self.objects.len() as f64;
        self.objects
            .iter()
            .map(|obj| weight * obj.pdf_value(origin, direction))
            .sum()
    }

    fn random(&self, origin: &Point3) -> Vec3 {
        if self.objects.is_empty() {
            return Vec3::new(1.0, 0.0, 0.0);
        }
        let idx = (random_double() * self.objects.len() as f64) as usize;
        let idx = idx.min(self.objects.len() - 1);
        self.objects[idx].random(origin)
    }
}
