use std::sync::Arc;

use crate::aabb::Aabb;
use crate::hittable::{HitRecord, Hittable};
use crate::material::Material;
use crate::ray::Ray;
use crate::utils::random_double;
use crate::vec3::Vec3;

pub struct ConstantMedium {
    boundary: Arc<dyn Hittable>,
    neg_inv_density: f64,
    phase_function: Arc<dyn Material>,
}

impl ConstantMedium {
    pub fn new(
        boundary: Arc<dyn Hittable>,
        density: f64,
        phase_function: Arc<dyn Material>,
    ) -> Self {
        Self {
            boundary,
            neg_inv_density: -1.0 / density,
            phase_function,
        }
    }
}

impl Hittable for ConstantMedium {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let mut rec1 = self
            .boundary
            .hit(ray, f64::NEG_INFINITY, f64::INFINITY)?;
        let mut rec2 = self
            .boundary
            .hit(ray, rec1.t + 0.0001, f64::INFINITY)?;

        rec1.t = rec1.t.max(t_min);
        rec2.t = rec2.t.min(t_max);

        if rec1.t >= rec2.t {
            return None;
        }

        let ray_length = ray.direction.length();
        let distance_inside_boundary = (rec2.t - rec1.t) * ray_length;
        let hit_distance = self.neg_inv_density * random_double().ln();

        if hit_distance > distance_inside_boundary {
            return None;
        }

        let t = rec1.t + hit_distance / ray_length;
        let p = ray.at(t);

        Some(HitRecord {
            p,
            t,
            normal: Vec3::new(1.0, 0.0, 0.0),
            front_face: true,
            u: 0.0,
            v: 0.0,
            material: Arc::clone(&self.phase_function),
        })
    }

    fn bounding_box(&self, time0: f64, time1: f64) -> Option<Aabb> {
        self.boundary.bounding_box(time0, time1)
    }
}
