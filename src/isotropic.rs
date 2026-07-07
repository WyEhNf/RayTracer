use std::sync::Arc;

use crate::hittable::HitRecord;
use crate::material::{Material, ScatterType};
use crate::pdf::Pdf;
use crate::ray::Ray;
use crate::texture::Texture;
use crate::utils::PI;
use crate::vec3::{Vec3, unit_vector};

struct UniformSpherePdf;

impl Pdf for UniformSpherePdf {
    fn value(&self, _direction: &Vec3) -> f64 {
        1.0 / (4.0 * PI)
    }

    fn generate(&self) -> Vec3 {
        unit_vector(&crate::vec3::random_in_unit_sphere())
    }
}

pub struct Isotropic {
    albedo: Arc<dyn Texture>,
}

impl Isotropic {
    pub fn new(albedo: Arc<dyn Texture>) -> Self {
        Self { albedo }
    }
}

impl Material for Isotropic {
    fn scatter(&self, _ray: &Ray, rec: &HitRecord) -> Option<ScatterType> {
        let pdf: Arc<dyn Pdf> = Arc::new(UniformSpherePdf);
        Some(ScatterType::Diffuse {
            attenuation: self.albedo.value(rec.u, rec.v, &rec.p),
            pdf,
        })
    }

    fn scattering_pdf(&self, _ray: &Ray, _rec: &HitRecord, _scattered: &Vec3) -> f64 {
        1.0 / (4.0 * PI)
    }
}
