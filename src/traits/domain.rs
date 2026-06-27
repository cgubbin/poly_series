use crate::{
    PolynomialSeries,
    scaling::{clamp_to_domain, from_scaled, to_scaled},
};

use num_traits::{Float, FromPrimitive};
use std::ops::Range;

/// Utilities for mapping between physical and scaled coordinates.
pub trait PolynomialDomain<E: Float + FromPrimitive>: PolynomialSeries<E> {
    type Error: std::error::Error + Send + Sync + 'static;
    /// Return the lower endpoint of the physical domain.
    fn lower_bound(&self) -> E;

    /// Return the upper endpoint of the physical domain.
    fn upper_bound(&self) -> E;

    /// Return `true` if `x` lies inside the physical domain.
    fn contains(&self, x: E) -> bool {
        self.lower_bound() <= x && x <= self.upper_bound()
    }

    /// Clamp `x` to the physical domain.
    fn clamp(&self, x: E) -> E {
        clamp_to_domain(x, &(self.lower_bound()..self.upper_bound()))
    }

    /// Map a physical coordinate `x ∈ [a, b]` to the scaled coordinate.
    ///
    /// For Chebyshev-like bases this usually maps `[a, b]` to `[-1, 1]`.
    fn to_scaled(&self, x: E) -> E {
        to_scaled(x, &(self.lower_bound()..self.upper_bound()))
    }

    /// Map a scaled coordinate back to the physical coordinate.
    ///
    /// For Chebyshev-like bases this usually maps `[-1, 1]` to `[a, b]`.
    #[allow(clippy::wrong_self_convention)]
    fn from_scaled(&self, t: E) -> E {
        from_scaled(t, &(self.lower_bound()..self.upper_bound()))
    }

    /// Return a copy of the polynomial with a different physical domain.
    ///
    /// This should preserve the stored coefficients and only change the domain
    /// metadata. It does not generally represent the same mathematical function
    /// of `x`; it represents the same function of the scaled coordinate.
    fn with_domain(&self, domain: Range<E>) -> Result<Self, Self::Error>;
}
