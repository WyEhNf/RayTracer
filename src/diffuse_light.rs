use std::sync::Arc;

use crate::hittable::HitRecord;
use crate::material::{Material, ScatterType};
use crate::ray::Ray;
use crate::texture::Texture;
use crate::vec3::Color;

pub struct DiffuseLight {
    emit: Arc<dyn Texture>,
}

impl DiffuseLight {
    pub fn new(emit: Arc<dyn Texture>) -> Self {
        Self { emit }
    }
}

impl Material for DiffuseLight {
    fn scatter(&self, _ray: &Ray, _rec: &HitRecord) -> Option<ScatterType> {
        None
    }

    fn emitted(&self, u: f64, v: f64, p: &crate::vec3::Point3) -> Color {
        self.emit.value(u, v, p)
    }
}
