use std::ops::Range;

mod coefficients;
mod conversion;
mod domain;
mod evaluate;
#[cfg(feature = "linalg")]
mod fitting;
mod integration;
#[cfg(feature = "linalg")]
mod interpolation;
#[cfg(feature = "linalg")]
mod root;
mod truncation;

pub use coefficients::PolynomialCoefficients;
pub use conversion::ConvertPolynomialBasis;
pub use domain::PolynomialDomain;
pub use evaluate::PolynomialEvaluateMany;
#[cfg(feature = "linalg")]
pub use fitting::{FitPolynomialSeries, FitReport};
pub use integration::PolynomialIntegrals;
#[cfg(feature = "linalg")]
pub use interpolation::InterpolatePolynomialSeries;
#[cfg(feature = "linalg")]
pub use root::PolynomialRoots;
pub use truncation::PolynomialApproximation;

/// Core behaviour shared by all polynomial series.
///
/// This trait represents a polynomial expressed in some basis, for example
/// monomial, Chebyshev, Legendre, or another orthogonal basis.
///
/// The polynomial is defined on a physical domain `domain = a..b`, but many
/// polynomial bases are most naturally evaluated on a scaled coordinate,
/// commonly `t ∈ [-1, 1]`.
pub trait PolynomialSeries<E>: Clone + Sized {
    /// Evaluate the polynomial at a scaled coordinate.
    fn evaluate_scaled(&self, t: E) -> E;

    /// Evaluate the polynomial at a physical coordinate.
    ///
    /// Implementations should map `x` from the physical domain into the
    /// internal scaled coordinate before evaluation.
    fn evaluate(&self, x: E) -> E;

    /// Return the first derivative of the polynomial.
    ///
    /// The derivative should be with respect to the physical coordinate `x`,
    /// not merely the scaled coordinate, unless the type explicitly documents
    /// otherwise.
    fn first_derivative(&self) -> Self;

    /// Return the derivative of the requested order.
    ///
    /// `order = 0` returns `self`.
    fn derivative(&self, order: usize) -> Self {
        let mut current = self.clone();

        for _ in 0..order {
            current = current.first_derivative();
        }

        current
    }

    /// Return the degree of the polynomial.
    ///
    /// The zero polynomial may be represented as degree `0`, or by a special
    /// empty coefficient representation. Implementations should document which
    /// convention they use.
    fn degree(&self) -> usize;

    /// Return the physical domain on which the polynomial is defined.
    fn domain(&self) -> Range<E>;

    /// Return a zero polynomial on the given physical domain.
    fn zero(domain: Range<E>) -> Self;

    /// Return `true` if the series represents the zero polynomial.
    fn is_zero(&self) -> bool;
}
