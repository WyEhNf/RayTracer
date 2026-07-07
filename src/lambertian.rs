use std::sync::Arc;

use crate::hittable::HitRecord;
use crate::material::{Material, ScatterType};
use crate::pdf::{CosinePdf, Pdf};
use crate::ray::Ray;
use crate::texture::Texture;
use crate::vec3::{Vec3, dot, unit_vector};

pub struct Lambertian {
    albedo: Arc<dyn Texture>,
}

impl Lambertian {
    pub fn new(albedo: Arc<dyn Texture>) -> Self {
        Self { albedo }
    }
}

impl Material for Lambertian {
    fn scatter(&self, _ray: &Ray, rec: &HitRecord) -> Option<ScatterType> {
        let pdf: Arc<dyn Pdf> = Arc::new(CosinePdf::new(&rec.normal));
        Some(ScatterType::Diffuse {
            attenuation: self.albedo.value(rec.u, rec.v, &rec.p),
            pdf,
        })
    }

    fn scattering_pdf(&self, _ray: &Ray, rec: &HitRecord, scattered: &Vec3) -> f64 {
        let cosine = dot(&rec.normal, &unit_vector(scattered));
        if cosine <= 0.0 {
            0.0
        } else {
            cosine / crate::utils::PI
        }
    }
}
