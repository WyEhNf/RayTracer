use crate::texture::Texture;
use crate::vec3::{Color, Point3};

pub struct ImageTexture {
    data: Vec<u8>,
    width: u32,
    height: u32,
}

impl ImageTexture {
    pub fn new(path: &str) -> Self {
        let img = image::ImageReader::open(path)
            .ok()
            .and_then(|r| r.decode().ok())
            .map(|i| i.to_rgb8());
        match img {
            Some(buf) => {
                let (width, height) = buf.dimensions();
                Self {
                    data: buf.into_raw(),
                    width,
                    height,
                }
            }
            None => Self {
                data: Vec::new(),
                width: 0,
                height: 0,
            },
        }
    }
}

impl Texture for ImageTexture {
    fn value(&self, u: f64, v: f64, _p: &Point3) -> Color {
        if self.data.is_empty() {
            return Color::new(1.0, 0.0, 1.0);
        }
        let i = ((u.clamp(0.0, 1.0) * self.width as f64) as u32).min(self.width - 1);
        let j = (((1.0 - v).clamp(0.0, 1.0) * self.height as f64) as u32).min(self.height - 1);
        let idx = (3 * i + 3 * self.width * j) as usize;
        let r = self.data[idx] as f64 / 255.0;
        let g = self.data[idx + 1] as f64 / 255.0;
        let b = self.data[idx + 2] as f64 / 255.0;
        Color::new(r * r, g * g, b * b)
    }
}
