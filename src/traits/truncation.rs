use crate::PolynomialSeries;

/// Approximation and simplification operations for polynomial series.
pub trait PolynomialApproximation<E>: PolynomialSeries<E> {
    /// Return a polynomial truncated to at most `degree`.
    ///
    /// Coefficients above `degree` should be discarded.
    fn truncate(&self, degree: usize) -> Self;

    /// Remove insignificant high-order coefficients.
    ///
    /// Implementations should remove trailing coefficients whose absolute value
    /// is less than or equal to `tolerance`.
    fn trim(&self, tolerance: E) -> Self;

    /// Return a lower-degree approximation if possible.
    ///
    /// This is usually equivalent to `trim`, but allows implementations to use
    /// basis-specific simplification rules.
    fn simplify(&self, tolerance: E) -> Self {
        self.trim(tolerance)
    }
}
