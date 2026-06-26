/// Interpolation constructors for polynomial series.
///
/// Unlike least-squares fitting, interpolation usually constructs a polynomial
/// passing exactly through the supplied points, subject to numerical precision.
pub trait InterpolatePolynomialSeries<E>: Sized {
    /// Error returned when interpolation fails.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Construct an interpolating polynomial through `(x, y)` data.
    fn interpolate(xs: &[E], ys: &[E]) -> Result<Self, Self::Error>;
}
