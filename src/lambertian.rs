use std::sync::Arc;

use crate::hittable::HitRecord;
use crate::material::Material;
use crate::ray::Ray;
use crate::texture::Texture;
use crate::vec3::random_in_hemisphere;

pub struct Lambertian {
    albedo: Arc<dyn Texture>,
}

impl Lambertian {
    pub fn new(albedo: Arc<dyn Texture>) -> Self {
        Self { albedo }
    }
}

impl Material for Lambertian {
    fn scatter(&self, ray: &Ray, rec: &HitRecord) -> Option<super::material::ScatterRecord> {
        let mut scatter_direction = rec.normal + random_in_hemisphere(&rec.normal);

        if scatter_direction.near_zero() {
            scatter_direction = rec.normal;
        }

        Some(super::material::ScatterRecord {
            attenuation: self.albedo.value(rec.u, rec.v, &rec.p),
            scattered_ray: Ray::new_at_time(rec.p, scatter_direction, ray.time),
        })
    }
}
