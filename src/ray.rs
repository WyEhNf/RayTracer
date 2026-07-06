use crate::vec3::{Point3, Vec3};

#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: Point3,
    pub direction: Vec3,
    pub time: f64,
}

impl Ray {
    pub fn new(origin: Point3, direction: Vec3) -> Self {
        Self {
            origin,
            direction,
            time: 0.0,
        }
    }

    pub fn new_at_time(origin: Point3, direction: Vec3, time: f64) -> Self {
        Self {
            origin,
            direction,
            time,
        }
    }

    pub fn at(&self, t: f64) -> Point3 {
        self.origin + t * self.direction
    }
}
