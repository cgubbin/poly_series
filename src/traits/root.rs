use crate::PolynomialSeries;

use std::ops::Range;

/// Root-finding behaviour for polynomial series.
///
/// Root finding is intentionally separate from [`PolynomialSeries`], because it
/// may require additional numerical algorithms, matrix decompositions, complex
/// arithmetic, or external linear algebra backends.
pub trait PolynomialRoots<E>: PolynomialSeries<E>
where
    E: PartialOrd + Copy,
{
    /// Error returned by the root-finding implementation.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Return all real roots representable by this implementation.
    fn roots(&self) -> Result<Vec<E>, Self::Error>;

    /// Return all roots inside the physical domain of the polynomial.
    ///
    /// Domain membership uses closed interval semantics:
    ///
    /// ```text
    /// domain.start <= root <= domain.end
    /// ```
    fn roots_in_domain(&self) -> Result<Vec<E>, Self::Error> {
        let domain = self.domain();
        self.roots_in_window(domain)
    }

    /// Return all roots inside `window`.
    ///
    /// Window membership uses closed interval semantics:
    ///
    /// ```text
    /// window.start <= root <= window.end
    /// ```
    fn roots_in_window(&self, window: Range<E>) -> Result<Vec<E>, Self::Error> {
        Ok(self
            .roots()?
            .into_iter()
            .filter(|root| window.start <= *root && *root <= window.end)
            .collect())
    }

    /// Return `true` if there is at least one root inside `window`.
    fn has_root_in_window(&self, window: Range<E>) -> Result<bool, Self::Error> {
        Ok(!self.roots_in_window(window)?.is_empty())
    }

    /// Return `true` if there is at least one root inside the physical domain.
    fn has_root_in_domain(&self) -> Result<bool, Self::Error> {
        Ok(!self.roots_in_domain()?.is_empty())
    }

    /// Return `true` if no stationary point is detected inside the domain.
    ///
    /// This checks whether the first derivative has roots in the physical
    /// domain. If the derivative has no roots inside the domain, the polynomial
    /// is monotonic on that domain.
    ///
    /// The converse is not always true: a monotonic polynomial may have a
    /// derivative root where the derivative touches zero without changing sign,
    /// for example `x^3`.
    fn is_monotonic(&self) -> Result<bool, Self::Error> {
        Ok(!self.first_derivative().has_root_in_domain()?)
    }
}
