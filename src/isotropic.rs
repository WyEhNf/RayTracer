use std::sync::Arc;

use crate::hittable::HitRecord;
use crate::material::{Material, ScatterRecord};
use crate::ray::Ray;
use crate::texture::Texture;
use crate::vec3::random_unit_vector;

pub struct Isotropic {
    albedo: Arc<dyn Texture>,
}

impl Isotropic {
    pub fn new(albedo: Arc<dyn Texture>) -> Self {
        Self { albedo }
    }
}

impl Material for Isotropic {
    fn scatter(&self, ray: &Ray, rec: &HitRecord) -> Option<ScatterRecord> {
        let scattered = Ray::new_at_time(rec.p, random_unit_vector(), ray.time);
        Some(ScatterRecord {
            attenuation: self.albedo.value(rec.u, rec.v, &rec.p),
            scattered_ray: scattered,
        })
    }
}
