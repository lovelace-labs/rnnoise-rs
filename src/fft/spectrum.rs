//! Complex value, matching C `kiss_fft_cpx` (`{ float r; float i; }`).

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Cpx {
    pub r: f32,
    pub i: f32,
}

impl Cpx {
    #[inline]
    pub const fn new(r: f32, i: f32) -> Self {
        Cpx { r, i }
    }
}
