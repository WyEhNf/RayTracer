use crate::hittable::HitRecord;
use crate::ray::Ray;
use crate::vec3::{Color, Point3};

pub struct ScatterRecord {
    pub attenuation: Color,
    pub scattered_ray: Ray,
}

pub trait Material: Send + Sync {
    fn scatter(&self, ray: &Ray, rec: &HitRecord) -> Option<ScatterRecord>;
    fn emitted(&self, _u: f64, _v: f64, _p: &Point3) -> Color {
        Color::new(0.0, 0.0, 0.0)
    }
}
