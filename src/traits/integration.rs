use crate::PolynomialSeries;

use std::ops::Range;

/// Integration behaviour for polynomial series.
pub trait PolynomialIntegrals<E>: PolynomialSeries<E> {
    /// Return an antiderivative of the polynomial.
    ///
    /// The integration constant should normally be zero unless otherwise
    /// documented by the implementation.
    fn antiderivative(&self) -> Self;

    /// Return an antiderivative with the supplied constant term.
    fn antiderivative_with_constant(&self, constant: E) -> Self;

    /// Compute the definite integral over a physical interval.
    fn definite_integral(&self, interval: Range<E>) -> E;

    /// Compute the definite integral over the full physical domain.
    fn integral_over_domain(&self) -> E {
        self.definite_integral(self.domain())
    }
}
