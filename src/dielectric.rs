use crate::hittable::HitRecord;
use crate::material::{Material, ScatterRecord};
use crate::ray::Ray;
use crate::utils::random_double;
use crate::vec3::{Color, dot, reflect, refract, unit_vector};

pub struct Dielectric {
    pub ref_idx: f64,
}

impl Dielectric {
    pub fn new(ref_idx: f64) -> Self {
        Self { ref_idx }
    }
}

fn reflectance(cosine: f64, ref_idx: f64) -> f64 {
    let r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    let r0 = r0 * r0;
    r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
}

impl Material for Dielectric {
    fn scatter(&self, ray: &Ray, rec: &HitRecord) -> Option<ScatterRecord> {
        let refraction_ratio = if rec.front_face {
            1.0 / self.ref_idx
        } else {
            self.ref_idx
        };

        let unit_direction = unit_vector(&ray.direction);
        let cos_theta = f64::min(dot(&-unit_direction, &rec.normal), 1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

        let cannot_refract = refraction_ratio * sin_theta > 1.0;
        let direction =
            if cannot_refract || reflectance(cos_theta, refraction_ratio) > random_double() {
                reflect(&unit_direction, &rec.normal)
            } else {
                refract(&unit_direction, &rec.normal, refraction_ratio)
            };

        Some(ScatterRecord {
            attenuation: Color::new(1.0, 1.0, 1.0),
            scattered_ray: Ray::new_at_time(rec.p, direction, ray.time),
        })
    }
}
