use std::sync::Arc;

use crate::hittable::HitRecord;
use crate::material::{Material, ScatterType};
use crate::pdf::CosinePdf;
use crate::ray::Ray;
use crate::utils::PI;
use crate::vec3::{Color, Vec3, dot, random_in_unit_sphere, reflect, unit_vector};

pub struct Metal {
    pub albedo: Color,
    pub fuzz: f64,
}

impl Metal {
    pub fn new(albedo: Color, fuzz: f64) -> Self {
        Self {
            albedo,
            fuzz: if fuzz < 1.0 { fuzz } else { 1.0 },
        }
    }
}

impl Material for Metal {
    fn scatter(&self, ray: &Ray, rec: &HitRecord) -> Option<ScatterType> {
        let reflected = reflect(&unit_vector(&ray.direction), &rec.normal);
        let scattered_dir = reflected + self.fuzz * random_in_unit_sphere();

        if dot(&scattered_dir, &rec.normal) <= 0.0 {
            return None;
        }

        if self.fuzz < 0.001 {
            Some(ScatterType::Specular {
                attenuation: self.albedo,
                scattered_ray: Ray::new_at_time(rec.p, reflected, ray.time),
            })
        } else {
            let pdf = Arc::new(CosinePdf::new(&rec.normal));
            Some(ScatterType::Diffuse {
                attenuation: self.albedo,
                pdf,
            })
        }
    }

    fn scattering_pdf(&self, _ray: &Ray, rec: &HitRecord, scattered: &Vec3) -> f64 {
        let cosine = dot(&rec.normal, &unit_vector(scattered));
        if cosine <= 0.0 {
            0.0
        } else {
            cosine / PI
        }
    }
}
