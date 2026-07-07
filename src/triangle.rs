use std::sync::Arc;

use crate::aabb::Aabb;
use crate::hittable::{HitRecord, Hittable};
use crate::material::Material;
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3, cross, dot};

pub struct Triangle {
    v0: Point3,
    v1: Point3,
    v2: Point3,
    n0: Vec3,
    n1: Vec3,
    n2: Vec3,
    material: Arc<dyn Material>,
}

impl Triangle {
    pub fn new(
        v0: Point3,
        v1: Point3,
        v2: Point3,
        n0: Vec3,
        n1: Vec3,
        n2: Vec3,
        material: Arc<dyn Material>,
    ) -> Self {
        Self {
            v0,
            v1,
            v2,
            n0,
            n1,
            n2,
            material,
        }
    }

    pub fn with_flat_normals(
        v0: Point3,
        v1: Point3,
        v2: Point3,
        material: Arc<dyn Material>,
    ) -> Self {
        let n = unit_normal(&v0, &v1, &v2);
        Self {
            v0,
            v1,
            v2,
            n0: n,
            n1: n,
            n2: n,
            material,
        }
    }
}

pub fn unit_normal(v0: &Point3, v1: &Point3, v2: &Point3) -> Vec3 {
    let e1 = *v1 - *v0;
    let e2 = *v2 - *v0;
    let n = cross(&e1, &e2);
    let len = n.length();
    if len > 0.0 {
        n / len
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    }
}

impl Hittable for Triangle {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let e1 = self.v1 - self.v0;
        let e2 = self.v2 - self.v0;

        let h = cross(&ray.direction, &e2);
        let a = dot(&e1, &h);

        if a.abs() < 1e-12 {
            return None;
        }

        let f = 1.0 / a;
        let s = ray.origin - self.v0;
        let u = f * dot(&s, &h);

        if u < 0.0 || u > 1.0 {
            return None;
        }

        let q = cross(&s, &e1);
        let v = f * dot(&ray.direction, &q);

        if v < 0.0 || u + v > 1.0 {
            return None;
        }

        let t = f * dot(&e2, &q);

        if t < t_min || t > t_max {
            return None;
        }

        let p = ray.at(t);

        let w = 1.0 - u - v;
        let normal = self.n0 * w + self.n1 * u + self.n2 * v;
        let normal = if normal.length_squared() > 0.0 {
            normal
        } else {
            unit_normal(&self.v0, &self.v1, &self.v2)
        };

        let mut rec = HitRecord {
            p,
            normal,
            t,
            u,
            v,
            front_face: false,
            material: Arc::clone(&self.material),
        };
        rec.set_face_normal(ray, &normal);

        Some(rec)
    }

    fn bounding_box(&self, _time0: f64, _time1: f64) -> Option<Aabb> {
        let min = Point3::new(
            self.v0.x.min(self.v1.x).min(self.v2.x),
            self.v0.y.min(self.v1.y).min(self.v2.y),
            self.v0.z.min(self.v1.z).min(self.v2.z),
        );
        let max = Point3::new(
            self.v0.x.max(self.v1.x).max(self.v2.x),
            self.v0.y.max(self.v1.y).max(self.v2.y),
            self.v0.z.max(self.v1.z).max(self.v2.z),
        );
        let eps = 0.0001;
        let pad = Vec3::new(eps, eps, eps);
        Some(Aabb::new(min - pad, max + pad))
    }
}
