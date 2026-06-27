use crate::{PolynomialCoefficients, PolynomialSeries};

use super::{ChebyshevError, ChebyshevSeries};

use num_traits::{Float, FromPrimitive};
use std::ops::Range;

impl<E> PolynomialCoefficients<E> for ChebyshevSeries<E>
where
    E: Float + FromPrimitive,
{
    type Error = ChebyshevError;
    fn coefficients(&self) -> &[E] {
        self.coefficients.as_slice()
    }

    fn from_coefficients(domain: Range<E>, coefficients: Vec<E>) -> Result<Self, ChebyshevError> {
        Self::new(coefficients, domain)
    }

    fn with_coefficients(&self, coefficients: Vec<E>) -> Result<Self, ChebyshevError> {
        Self::new(coefficients, self.domain())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PolynomialCoefficients, PolynomialSeries};

    fn series(coefficients: Vec<f64>) -> ChebyshevSeries<f64> {
        ChebyshevSeries::new(coefficients, -1.0..1.0).unwrap()
    }

    #[test]
    fn coefficients_returns_native_basis_coefficients() {
        let p = series(vec![1.0, 2.0, 3.0]);

        assert_eq!(p.coefficients(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn from_coefficients_constructs_valid_series() {
        let p = ChebyshevSeries::from_coefficients(10.0..20.0, vec![1.0, 2.0, 3.0]).unwrap();

        assert_eq!(p.coefficients(), &[1.0, 2.0, 3.0]);
        assert_eq!(p.domain(), 10.0..20.0);
        assert_eq!(p.degree(), 2);
    }

    #[test]
    fn from_coefficients_trims_trailing_zeroes() {
        let p = ChebyshevSeries::from_coefficients(-1.0..1.0, vec![1.0, 2.0, 0.0, 0.0]).unwrap();

        assert_eq!(p.coefficients(), &[1.0, 2.0]);
        assert_eq!(p.degree(), 1);
    }

    #[test]
    fn from_coefficients_canonicalises_empty_coefficients_to_zero() {
        let p = ChebyshevSeries::from_coefficients(-1.0..1.0, vec![]).unwrap();

        assert_eq!(p.coefficients(), &[0.0]);
        assert!(p.is_zero());
        assert_eq!(p.degree(), 0);
    }

    #[test]
    fn from_coefficients_rejects_nan_coefficients() {
        let err = ChebyshevSeries::from_coefficients(-1.0..1.0, vec![1.0, f64::NAN]).unwrap_err();

        assert!(matches!(err, ChebyshevError::InvalidCoefficients));
    }

    #[test]
    fn from_coefficients_rejects_infinite_coefficients() {
        let err =
            ChebyshevSeries::from_coefficients(-1.0..1.0, vec![1.0, f64::INFINITY]).unwrap_err();

        assert!(matches!(err, ChebyshevError::InvalidCoefficients));
    }

    #[test]
    fn from_coefficients_rejects_invalid_domain() {
        let err = ChebyshevSeries::from_coefficients(1.0..-1.0, vec![1.0, 2.0]).unwrap_err();

        assert!(matches!(err, ChebyshevError::InvalidDomain));
    }

    #[test]
    fn with_coefficients_preserves_domain() {
        let p = series(vec![1.0, 2.0]);
        let q = p.with_coefficients(vec![3.0, 4.0, 5.0]).unwrap();

        assert_eq!(q.coefficients(), &[3.0, 4.0, 5.0]);
        assert_eq!(q.domain(), p.domain());
    }

    #[test]
    fn with_coefficients_trims_trailing_zeroes() {
        let p = series(vec![1.0, 2.0]);
        let q = p.with_coefficients(vec![3.0, 4.0, 0.0, 0.0]).unwrap();

        assert_eq!(q.coefficients(), &[3.0, 4.0]);
        assert_eq!(q.degree(), 1);
    }

    #[test]
    fn with_coefficients_rejects_invalid_coefficients() {
        let p = series(vec![1.0, 2.0]);

        let err = p.with_coefficients(vec![1.0, f64::NAN]).unwrap_err();

        assert!(matches!(err, ChebyshevError::InvalidCoefficients));
    }

    #[test]
    fn len_returns_number_of_stored_coefficients() {
        let p = series(vec![1.0, 2.0, 3.0]);

        assert_eq!(p.len(), 3);
    }

    #[test]
    fn is_empty_is_false_for_canonical_zero_series() {
        let p = ChebyshevSeries::<f64>::zero(-1.0..1.0);

        assert!(!p.is_empty());
    }

    #[test]
    fn coefficient_returns_requested_coefficient() {
        let p = series(vec![1.0, 2.0, 3.0]);

        assert_eq!(p.coefficient(0), Some(&1.0));
        assert_eq!(p.coefficient(1), Some(&2.0));
        assert_eq!(p.coefficient(2), Some(&3.0));
        assert_eq!(p.coefficient(3), None);
    }

    #[test]
    fn constant_coefficient_returns_first_coefficient() {
        let p = series(vec![1.0, 2.0, 3.0]);

        assert_eq!(p.constant_coefficient(), Some(&1.0));
    }

    #[test]
    fn leading_coefficient_returns_last_stored_coefficient() {
        let p = series(vec![1.0, 2.0, 3.0]);

        assert_eq!(p.leading_coefficient(), Some(&3.0));
    }
}
