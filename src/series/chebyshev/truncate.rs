use crate::PolynomialApproximation;

use super::ChebyshevSeries;

use num_traits::{Float, FromPrimitive};

impl<E: Float + FromPrimitive> PolynomialApproximation<E> for ChebyshevSeries<E> {
    fn truncate(&self, degree: usize) -> Self {
        Self {
            coefficients: self.coefficients.truncate(degree),
            domain: self.domain.clone(),
        }
    }

    fn trim(&self, tolerance: E) -> Self {
        Self {
            coefficients: self.coefficients.trim_by_absolute_tolerance(tolerance),
            domain: self.domain.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PolynomialApproximation, PolynomialSeries};

    fn series(coefficients: Vec<f64>) -> ChebyshevSeries<f64> {
        ChebyshevSeries::new(coefficients, 10.0..20.0).unwrap()
    }

    #[test]
    fn truncate_to_lower_degree_discards_high_order_coefficients() {
        let p = series(vec![1.0, 2.0, 3.0, 4.0]);

        let q = p.truncate(1);

        assert_eq!(q.coefficients(), &[1.0, 2.0]);
        assert_eq!(q.degree(), 1);
    }

    #[test]
    fn truncate_to_zero_keeps_constant_coefficient() {
        let p = series(vec![1.0, 2.0, 3.0]);

        let q = p.truncate(0);

        assert_eq!(q.coefficients(), &[1.0]);
        assert_eq!(q.degree(), 0);
    }

    #[test]
    fn truncate_above_degree_returns_same_coefficients() {
        let p = series(vec![1.0, 2.0, 3.0]);

        let q = p.truncate(10);

        assert_eq!(q.coefficients(), p.coefficients());
        assert_eq!(q.degree(), p.degree());
    }

    #[test]
    fn truncate_preserves_domain() {
        let p = series(vec![1.0, 2.0, 3.0]);

        let q = p.truncate(1);

        assert_eq!(q.domain(), 10.0..20.0);
    }

    #[test]
    fn trim_removes_trailing_coefficients_below_tolerance() {
        let p = series(vec![1.0, 2.0, 1.0e-14, -1.0e-15]);

        let q = p.trim(1.0e-12);

        assert_eq!(q.coefficients(), &[1.0, 2.0]);
        assert_eq!(q.degree(), 1);
    }

    #[test]
    fn trim_keeps_non_trailing_small_coefficients() {
        let p = series(vec![1.0, 1.0e-14, 2.0]);

        let q = p.trim(1.0e-12);

        assert_eq!(q.coefficients(), &[1.0, 1.0e-14, 2.0]);
        assert_eq!(q.degree(), 2);
    }

    #[test]
    fn trim_uses_absolute_value_of_tolerance() {
        let p = series(vec![1.0, 2.0, 1.0e-14]);

        let q = p.trim(-1.0e-12);

        assert_eq!(q.coefficients(), &[1.0, 2.0]);
    }

    #[test]
    fn trim_all_coefficients_returns_canonical_zero() {
        let p = series(vec![1.0e-14, -1.0e-15]);

        let q = p.trim(1.0e-12);

        assert_eq!(q.coefficients(), &[0.0]);
        assert!(q.is_zero());
    }

    #[test]
    fn trim_preserves_domain() {
        let p = series(vec![1.0, 2.0, 1.0e-14]);

        let q = p.trim(1.0e-12);

        assert_eq!(q.domain(), 10.0..20.0);
    }
}
