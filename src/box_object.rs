use std::sync::Arc;

use crate::aabb::Aabb;
use crate::hittable::{HitRecord, Hittable};
use crate::hittable_list::HittableList;
use crate::material::Material;
use crate::ray::Ray;
use crate::rect::{XyRect, XzRect, YzRect};
use crate::vec3::Point3;

pub struct BoxObject {
    box_min: Point3,
    box_max: Point3,
    sides: HittableList,
}

impl BoxObject {
    pub fn new(p0: Point3, p1: Point3, material: Arc<dyn Material>) -> Self {
        let mut sides = HittableList::new();
        sides.add(Arc::new(XyRect::new(
            p0.x, p1.x, p0.y, p1.y, p1.z, Arc::clone(&material),
        )));
        sides.add(Arc::new(XyRect::new(
            p0.x, p1.x, p0.y, p1.y, p0.z, Arc::clone(&material),
        )));
        sides.add(Arc::new(XzRect::new(
            p0.x, p1.x, p0.z, p1.z, p1.y, Arc::clone(&material),
        )));
        sides.add(Arc::new(XzRect::new(
            p0.x, p1.x, p0.z, p1.z, p0.y, Arc::clone(&material),
        )));
        sides.add(Arc::new(YzRect::new(
            p0.y, p1.y, p0.z, p1.z, p1.x, Arc::clone(&material),
        )));
        sides.add(Arc::new(YzRect::new(
            p0.y, p1.y, p0.z, p1.z, p0.x, Arc::clone(&material),
        )));
        Self {
            box_min: p0,
            box_max: p1,
            sides,
        }
    }
}

impl Hittable for BoxObject {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        self.sides.hit(ray, t_min, t_max)
    }

    fn bounding_box(&self, _time0: f64, _time1: f64) -> Option<Aabb> {
        Some(Aabb::new(self.box_min, self.box_max))
    }
}
