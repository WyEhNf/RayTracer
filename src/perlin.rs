use crate::vec3::{Vec3, dot, unit_vector};

pub struct Perlin {
    ranvec: [Vec3; 256],
    perm_x: [usize; 256],
    perm_y: [usize; 256],
    perm_z: [usize; 256],
}

fn perlin_generate_perm() -> [usize; 256] {
    let mut p = [0usize; 256];
    for i in 0..256 {
        p[i] = i;
    }
    permute(&mut p);
    p
}

fn permute(p: &mut [usize; 256]) {
    for i in (1..256).rev() {
        let target = (crate::utils::random_double() * (i + 1) as f64) as usize;
        p.swap(i, target);
    }
}

impl Perlin {
    pub fn new() -> Self {
        let ranvec = std::array::from_fn(|_| {
            unit_vector(&Vec3::new(
                crate::utils::random_range(-1.0, 1.0),
                crate::utils::random_range(-1.0, 1.0),
                crate::utils::random_range(-1.0, 1.0),
            ))
        });
        let perm_x = perlin_generate_perm();
        let perm_y = perlin_generate_perm();
        let perm_z = perlin_generate_perm();
        Self {
            ranvec,
            perm_x,
            perm_y,
            perm_z,
        }
    }

    pub fn noise(&self, p: &Point3) -> f64 {
        let i = p.x.floor() as i32;
        let j = p.y.floor() as i32;
        let k = p.z.floor() as i32;
        let u = p.x - p.x.floor();
        let v = p.y - p.y.floor();
        let w = p.z - p.z.floor();

        let uu = u * u * (3.0 - 2.0 * u);
        let vv = v * v * (3.0 - 2.0 * v);
        let ww = w * w * (3.0 - 2.0 * w);

        let mut accum = 0.0;
        for di in 0..2 {
            for dj in 0..2 {
                for dk in 0..2 {
                    let idx = self.perm_x[((i + di as i32) & 255) as usize]
                        ^ self.perm_y[((j + dj as i32) & 255) as usize]
                        ^ self.perm_z[((k + dk as i32) & 255) as usize];
                    let c = self.ranvec[idx];
                    let weight = Vec3::new(u - di as f64, v - dj as f64, w - dk as f64);
                    accum += (di as f64 * uu + (1 - di) as f64 * (1.0 - uu))
                        * (dj as f64 * vv + (1 - dj) as f64 * (1.0 - vv))
                        * (dk as f64 * ww + (1 - dk) as f64 * (1.0 - ww))
                        * dot(&c, &weight);
                }
            }
        }
        accum
    }

    pub fn turb(&self, p: &Point3, depth: u32) -> f64 {
        let mut accum = 0.0;
        let mut temp_p = *p;
        let mut weight = 1.0;
        for _ in 0..depth {
            accum += weight * self.noise(&temp_p);
            weight *= 0.5;
            temp_p = temp_p * 2.0;
        }
        accum.abs()
    }
}
