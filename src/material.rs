use std::sync::Arc;

use crate::hittable::HitRecord;
use crate::pdf::Pdf;
use crate::ray::Ray;
use crate::vec3::{Color, Point3, Vec3};

pub enum ScatterType {
    Diffuse {
        attenuation: Color,
        pdf: Arc<dyn Pdf>,
    },
    Specular {
        attenuation: Color,
        scattered_ray: Ray,
    },
    NormalVis {
        normal: Vec3,
    },
}

pub trait Material: Send + Sync {
    fn scatter(&self, ray: &Ray, rec: &HitRecord) -> Option<ScatterType>;
    fn scattering_pdf(&self, _ray: &Ray, _rec: &HitRecord, _scattered: &Vec3) -> f64 {
        0.0
    }
    fn emitted(&self, _u: f64, _v: f64, _p: &Point3) -> Color {
        Color::new(0.0, 0.0, 0.0)
    }
}
