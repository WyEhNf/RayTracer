use std::sync::Arc;

use crate::aabb::Aabb;
use crate::hittable::{HitRecord, Hittable};
use crate::ray::Ray;
use crate::utils::random_int;

pub struct BvhNode {
    left: Arc<dyn Hittable>,
    right: Arc<dyn Hittable>,
    bbox: Aabb,
}

impl BvhNode {
    pub fn build(
        objects: &mut [Arc<dyn Hittable>],
        time0: f64,
        time1: f64,
    ) -> Arc<dyn Hittable> {
        let axis = random_int(0, 2) as usize;
        objects.sort_by(|a, b| {
            let box_a = a.bounding_box(time0, time1).unwrap();
            let box_b = b.bounding_box(time0, time1).unwrap();
            box_a.min[axis]
                .partial_cmp(&box_b.min[axis])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        match objects.len() {
            1 => objects[0].clone(),
            2 => {
                let left = objects[0].clone();
                let right = objects[1].clone();
                let bbox = Aabb::surrounding(
                    &left.bounding_box(time0, time1).unwrap(),
                    &right.bounding_box(time0, time1).unwrap(),
                );
                Arc::new(BvhNode { left, right, bbox })
            }
            n => {
                let mid = n / 2;
                let left = BvhNode::build(&mut objects[..mid], time0, time1);
                let right = BvhNode::build(&mut objects[mid..], time0, time1);
                let bbox = Aabb::surrounding(
                    &left.bounding_box(time0, time1).unwrap(),
                    &right.bounding_box(time0, time1).unwrap(),
                );
                Arc::new(BvhNode { left, right, bbox })
            }
        }
    }
}

impl Hittable for BvhNode {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        if !self.bbox.hit(ray, t_min, t_max) {
            return None;
        }
        let hit_left = self.left.hit(ray, t_min, t_max);
        let t_max_right = hit_left.as_ref().map_or(t_max, |r| r.t);
        let hit_right = self.right.hit(ray, t_min, t_max_right);
        hit_right.or(hit_left)
    }

    fn bounding_box(&self, _time0: f64, _time1: f64) -> Option<Aabb> {
        Some(self.bbox.clone())
    }
}
