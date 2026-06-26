use std::ops::Range;

use crate::PolynomialSeries;

/// Access to the coefficients of a polynomial series.
///
/// Coefficients are expressed in the native basis of the implementing type.
/// For example, a `ChebyshevSeries` should return Chebyshev coefficients, not
/// monomial coefficients.
pub trait PolynomialCoefficients<E>: PolynomialSeries<E> {
    type Error: std::error::Error + Send + Sync + 'static;
    /// Return the coefficients in basis order.
    ///
    /// For most bases this means:
    ///
    /// - `coefficients()[0]` is the constant coefficient
    /// - `coefficients()[1]` multiplies the first basis polynomial
    /// - `coefficients()[n]` multiplies the nth basis polynomial
    fn coefficients(&self) -> &[E];

    /// Construct a polynomial series from native-basis coefficients.
    ///
    /// Implementations should validate that the coefficients are finite and
    /// that the domain is valid.
    fn from_coefficients(domain: Range<E>, coefficients: Vec<E>) -> Result<Self, Self::Error>;

    /// Change the polynomial coefficients in constant basis series
    ///
    /// Implementations should validate that the coefficients are finite and
    /// that the domain is valid.
    fn with_coefficients(&self, coefficients: Vec<E>) -> Result<Self, Self::Error>;

    /// Return the number of stored coefficients.
    fn len(&self) -> usize {
        self.coefficients().len()
    }

    /// Return `true` if no coefficients are stored.
    fn is_empty(&self) -> bool {
        self.coefficients().is_empty()
    }

    /// Return the coefficient at `index`, if present.
    fn coefficient(&self, index: usize) -> Option<&E> {
        self.coefficients().get(index)
    }

    /// Return the constant coefficient, if present.
    fn constant_coefficient(&self) -> Option<&E> {
        self.coefficient(0)
    }

    /// Return the highest-order stored coefficient, if present.
    fn leading_coefficient(&self) -> Option<&E> {
        self.coefficients().last()
    }
}
