use crate::ray::Ray;
use crate::utils::random_range;
use crate::vec3::{Point3, Vec3, cross, random_in_unit_disk, unit_vector};

pub struct Camera {
    origin: Point3,
    lower_left_corner: Point3,
    horizontal: Vec3,
    vertical: Vec3,
    u: Vec3,
    v: Vec3,
    lens_radius: f64,
    time0: f64,
    time1: f64,
}

impl Camera {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        lookfrom: Point3,
        lookat: Point3,
        vup: Vec3,
        vfov: f64,
        aspect_ratio: f64,
        defocus_angle: f64,
        focus_dist: f64,
        time0: f64,
        time1: f64,
    ) -> Self {
        let theta = vfov.to_radians();
        let h = (theta / 2.0).tan();
        let viewport_height = 2.0 * h * focus_dist;
        let viewport_width = aspect_ratio * viewport_height;

        let w = unit_vector(&(lookfrom - lookat));
        let u = unit_vector(&cross(&vup, &w));
        let v = cross(&w, &u);

        let origin = lookfrom;
        let horizontal = viewport_width * u;
        let vertical = viewport_height * v;
        let lower_left_corner =
            origin - horizontal / 2.0 - vertical / 2.0 - w * focus_dist;

        let lens_radius = (defocus_angle / 2.0).to_radians().tan() * focus_dist;

        Self {
            origin,
            lower_left_corner,
            horizontal,
            vertical,
            u,
            v,
            lens_radius,
            time0,
            time1,
        }
    }

    pub fn get_ray(&self, s: f64, t: f64) -> Ray {
        let rd = self.lens_radius * random_in_unit_disk();
        let offset = self.u * rd.x + self.v * rd.y;
        let ray_time = random_range(self.time0, self.time1);

        Ray::new_at_time(
            self.origin + offset,
            self.lower_left_corner + s * self.horizontal + t * self.vertical
                - self.origin
                - offset,
            ray_time,
        )
    }
}
