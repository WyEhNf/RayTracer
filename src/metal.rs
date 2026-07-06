use crate::hittable::HitRecord;
use crate::material::{Material, ScatterRecord};
use crate::ray::Ray;
use crate::vec3::{Color, dot, random_in_unit_sphere, reflect, unit_vector};

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
    fn scatter(&self, ray: &Ray, rec: &HitRecord) -> Option<ScatterRecord> {
        let reflected = reflect(&unit_vector(&ray.direction), &rec.normal);
        let scattered_dir = reflected + self.fuzz * random_in_unit_sphere();

        if dot(&scattered_dir, &rec.normal) > 0.0 {
            Some(ScatterRecord {
                attenuation: self.albedo,
                scattered_ray: Ray::new(rec.p, scattered_dir),
            })
        } else {
            None
        }
    }
}
