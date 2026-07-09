use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuCamera {
    pub origin: [f32; 4],
    pub lower_left_corner: [f32; 4],
    pub horizontal: [f32; 4],
    pub vertical: [f32; 4],
    pub u: [f32; 4],
    pub v: [f32; 4],
    pub lens_radius: f32,
    pub _pad: [f32; 3],
}

impl GpuCamera {
    pub fn new(
        lookfrom: [f32; 3],
        lookat: [f32; 3],
        vup: [f32; 3],
        vfov: f32,
        aspect_ratio: f32,
        defocus_angle: f32,
        focus_dist: f32,
    ) -> Self {
        let theta = vfov.to_radians();
        let h = (theta / 2.0).tan();
        let viewport_height = 2.0 * h * focus_dist;
        let viewport_width = aspect_ratio * viewport_height;

        let w = unit_vector(&sub(&lookfrom, &lookat));
        let u = unit_vector(&cross(&vup, &w));
        let v = cross(&w, &u);

        let origin = lookfrom;
        let horizontal = scale(&u, viewport_width);
        let vertical = scale(&v, viewport_height);
        let llc = sub(
            &sub(&sub(&origin, &scale(&horizontal, 0.5)), &scale(&vertical, 0.5)),
            &scale(&w, focus_dist),
        );

        let lens_radius = (defocus_angle / 2.0).to_radians().tan() * focus_dist;

        Self {
            origin: [origin[0], origin[1], origin[2], 0.0],
            lower_left_corner: [llc[0], llc[1], llc[2], 0.0],
            horizontal: [horizontal[0], horizontal[1], horizontal[2], 0.0],
            vertical: [vertical[0], vertical[1], vertical[2], 0.0],
            u: [u[0], u[1], u[2], 0.0],
            v: [v[0], v[1], v[2], 0.0],
            lens_radius,
            _pad: [0.0; 3],
        }
    }
}

fn sub(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(v: &[f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn dot(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn unit_vector(v: &[f32; 3]) -> [f32; 3] {
    let len = (dot(v, v)).sqrt();
    [v[0] / len, v[1] / len, v[2] / len]
}
