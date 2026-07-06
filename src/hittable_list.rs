use std::sync::Arc;

use crate::aabb::Aabb;
use crate::hittable::{HitRecord, Hittable};
use crate::ray::Ray;

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

    pub fn clear(&mut self) {
        self.objects.clear();
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
}
