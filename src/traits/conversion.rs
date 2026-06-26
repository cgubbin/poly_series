/// Conversion between polynomial bases.
///
/// This is useful when, for example, a Chebyshev representation is used for
/// stable fitting/evaluation but a monomial representation is needed for
/// interoperability.
pub trait ConvertPolynomialBasis<Target> {
    /// Error returned when basis conversion fails.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Convert this polynomial series into another basis.
    fn convert(&self) -> Result<Target, Self::Error>;
}
