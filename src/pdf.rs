use std::sync::Arc;

use crate::hittable::Hittable;
use crate::onb::Onb;
use crate::utils::{random_double, PI};
use crate::vec3::{Point3, Vec3, unit_vector};

pub trait Pdf: Send + Sync {
    fn value(&self, direction: &Vec3) -> f64;
    fn generate(&self) -> Vec3;
}

pub struct CosinePdf {
    uvw: Onb,
}

impl CosinePdf {
    pub fn new(w: &Vec3) -> Self {
        Self {
            uvw: Onb::build_from_w(w),
        }
    }
}

impl Pdf for CosinePdf {
    fn value(&self, direction: &Vec3) -> f64 {
        let cosine = crate::vec3::dot(&unit_vector(direction), &self.uvw.w());
        if cosine <= 0.0 {
            0.0
        } else {
            cosine / PI
        }
    }

    fn generate(&self) -> Vec3 {
        let r1 = random_double();
        let r2 = random_double();
        let z = (1.0 - r2).sqrt();
        let phi = 2.0 * PI * r1;
        let x = phi.cos() * r2.sqrt();
        let y = phi.sin() * r2.sqrt();
        self.uvw.local(&Vec3::new(x, y, z))
    }
}

pub struct HittablePdf {
    origin: Point3,
    objects: Arc<dyn Hittable>,
}

impl HittablePdf {
    pub fn new(origin: Point3, objects: Arc<dyn Hittable>) -> Self {
        Self { origin, objects }
    }
}

impl Pdf for HittablePdf {
    fn value(&self, direction: &Vec3) -> f64 {
        self.objects.pdf_value(&self.origin, direction)
    }

    fn generate(&self) -> Vec3 {
        self.objects.random(&self.origin)
    }
}

pub struct MixturePdf {
    p0: Arc<dyn Pdf>,
    p1: Arc<dyn Pdf>,
}

impl MixturePdf {
    pub fn new(p0: Arc<dyn Pdf>, p1: Arc<dyn Pdf>) -> Self {
        Self { p0, p1 }
    }
}

impl Pdf for MixturePdf {
    fn value(&self, direction: &Vec3) -> f64 {
        0.5 * self.p0.value(direction) + 0.5 * self.p1.value(direction)
    }

    fn generate(&self) -> Vec3 {
        if random_double() < 0.5 {
            self.p0.generate()
        } else {
            self.p1.generate()
        }
    }
}

pub struct SpherePdf {
    origin: Point3,
    center: Point3,
    radius: f64,
}

impl SpherePdf {
    pub fn new(origin: Point3, center: Point3, radius: f64) -> Self {
        Self {
            origin,
            center,
            radius,
        }
    }
}

impl Pdf for SpherePdf {
    fn value(&self, direction: &Vec3) -> f64 {
        let oc = self.origin - self.center;
        let a = direction.length_squared();
        let half_b = crate::vec3::dot(&oc, direction);
        let c = oc.length_squared() - self.radius * self.radius;
        let discriminant = half_b * half_b - a * c;
        if discriminant < 0.0 {
            return 0.0;
        }
        let dist_to_center = (self.center - self.origin).length();
        let sin_theta_max = (self.radius / dist_to_center).min(1.0);
        let cos_theta_max = (1.0 - sin_theta_max * sin_theta_max).abs().sqrt();
        let solid_angle = 2.0 * PI * (1.0 - cos_theta_max);
        if solid_angle < 1e-12 {
            return 1.0;
        }
        1.0 / solid_angle
    }

    fn generate(&self) -> Vec3 {
        let dir_to_center = unit_vector(&(self.center - self.origin));
        let onb = Onb::build_from_w(&dir_to_center);
        let dist_to_center = (self.center - self.origin).length();
        let sin_theta_max = (self.radius / dist_to_center).min(1.0);
        let cos_theta_max = (1.0 - sin_theta_max * sin_theta_max).abs().sqrt();
        let r1 = random_double();
        let cos_theta = 1.0 + r1 * (cos_theta_max - 1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).abs().sqrt();
        let phi = 2.0 * PI * random_double();
        onb.local(&Vec3::new(
            sin_theta * phi.cos(),
            sin_theta * phi.sin(),
            cos_theta,
        ))
    }
}
