use crate::{
    PolynomialDomain, PolynomialSeries,
    scaling::{from_scaled, to_scaled},
};

use super::{ChebyshevError, ChebyshevSeries};

use num_traits::{Float, FromPrimitive};
use std::ops::Range;

impl<E> PolynomialDomain<E> for ChebyshevSeries<E>
where
    E: Float + FromPrimitive,
{
    type Error = ChebyshevError;

    fn lower_bound(&self) -> E {
        self.domain.start
    }

    fn upper_bound(&self) -> E {
        self.domain.end
    }

    fn with_domain(&self, domain: Range<E>) -> Result<Self, Self::Error> {
        Self::new(self.coefficients.clone(), domain)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PolynomialDomain, PolynomialSeries};

    const EPS: f64 = 1.0e-12;

    fn assert_close(lhs: f64, rhs: f64) {
        assert!(
            (lhs - rhs).abs() <= EPS,
            "expected {lhs} ≈ {rhs}, difference = {}",
            (lhs - rhs).abs()
        );
    }

    fn series() -> ChebyshevSeries<f64> {
        ChebyshevSeries::new(vec![1.0, 2.0, 3.0], 10.0..20.0).unwrap()
    }

    #[test]
    fn lower_bound_returns_domain_start() {
        let p = series();

        assert_close(p.lower_bound(), 10.0);
    }

    #[test]
    fn upper_bound_returns_domain_end() {
        let p = series();

        assert_close(p.upper_bound(), 20.0);
    }

    #[test]
    fn contains_uses_closed_interval_semantics() {
        let p = series();

        assert!(p.contains(10.0));
        assert!(p.contains(15.0));
        assert!(p.contains(20.0));

        assert!(!p.contains(9.999));
        assert!(!p.contains(20.001));
    }

    #[test]
    fn clamp_returns_value_inside_domain_unchanged() {
        let p = series();

        assert_close(p.clamp(15.0), 15.0);
    }

    #[test]
    fn clamp_maps_values_below_domain_to_lower_bound() {
        let p = series();

        assert_close(p.clamp(5.0), 10.0);
    }

    #[test]
    fn clamp_maps_values_above_domain_to_upper_bound() {
        let p = series();

        assert_close(p.clamp(25.0), 20.0);
    }

    #[test]
    fn to_scaled_maps_domain_start_to_minus_one() {
        let p = series();

        assert_close(p.to_scaled(10.0), -1.0);
    }

    #[test]
    fn to_scaled_maps_domain_midpoint_to_zero() {
        let p = series();

        assert_close(p.to_scaled(15.0), 0.0);
    }

    #[test]
    fn to_scaled_maps_domain_end_to_one() {
        let p = series();

        assert_close(p.to_scaled(20.0), 1.0);
    }

    #[test]
    fn from_scaled_maps_minus_one_to_domain_start() {
        let p = series();

        assert_close(p.from_scaled(-1.0), 10.0);
    }

    #[test]
    fn from_scaled_maps_zero_to_domain_midpoint() {
        let p = series();

        assert_close(p.from_scaled(0.0), 15.0);
    }

    #[test]
    fn from_scaled_maps_one_to_domain_end() {
        let p = series();

        assert_close(p.from_scaled(1.0), 20.0);
    }

    #[test]
    fn scaling_round_trip_physical_to_scaled_to_physical() {
        let p = series();

        for x in [10.0, 12.5, 15.0, 17.5, 20.0] {
            assert_close(p.from_scaled(p.to_scaled(x)), x);
        }
    }

    #[test]
    fn scaling_round_trip_scaled_to_physical_to_scaled() {
        let p = series();

        for t in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            assert_close(p.to_scaled(p.from_scaled(t)), t);
        }
    }

    #[test]
    fn with_domain_changes_domain() {
        let p = series();
        let q = p.with_domain(-1.0..1.0).unwrap();

        assert_eq!(q.domain(), -1.0..1.0);
        assert_close(q.lower_bound(), -1.0);
        assert_close(q.upper_bound(), 1.0);
    }

    #[test]
    fn with_domain_preserves_coefficients_as_scaled_function() {
        let p = series();
        let q = p.with_domain(-1.0..1.0).unwrap();

        assert_eq!(q.coefficients(), p.coefficients());

        for t in [-1.0, -0.25, 0.0, 0.5, 1.0] {
            assert_close(q.evaluate_scaled(t), p.evaluate_scaled(t));
        }
    }

    #[test]
    fn with_domain_does_not_preserve_physical_function_in_general() {
        let p = series();
        let q = p.with_domain(-1.0..1.0).unwrap();

        assert_ne!(q.evaluate(0.0), p.evaluate(0.0));
    }

    #[test]
    fn with_domain_rejects_reversed_domain() {
        let p = series();

        let err = p.with_domain(1.0..-1.0).unwrap_err();

        assert!(matches!(err, ChebyshevError::InvalidDomain));
    }

    #[test]
    fn with_domain_rejects_zero_width_domain() {
        let p = series();

        let err = p.with_domain(1.0..1.0).unwrap_err();

        assert!(matches!(err, ChebyshevError::InvalidDomain));
    }

    #[test]
    fn with_domain_rejects_non_finite_domain() {
        let p = series();

        let err = p.with_domain(f64::NEG_INFINITY..1.0).unwrap_err();

        assert!(matches!(err, ChebyshevError::InvalidDomain));
    }
}
