use crate::{InterpolatePolynomialSeries, scaling::to_scaled};

use super::{ChebyshevError, ChebyshevSeries};

use ndarray::{Array1, Array2};
use ndarray_linalg::{Lapack, Scalar, Solve};
use num_traits::{Float, FromPrimitive};
use std::ops::Range;

impl<E> InterpolatePolynomialSeries<E> for ChebyshevSeries<E>
where
    E: Float + FromPrimitive + Scalar<Real = E> + Lapack,
{
    type Error = ChebyshevError;

    fn interpolate(xs: &[E], ys: &[E]) -> Result<Self, Self::Error> {
        validate_interpolation_data(xs, ys)?;

        let domain = interpolation_domain(xs)?;

        Self::interpolate_on_domain(xs, ys, domain)
    }
}

impl<E> ChebyshevSeries<E>
where
    E: Float + FromPrimitive + Scalar<Real = E> + Lapack,
{
    pub fn interpolate_on_domain(
        xs: &[E],
        ys: &[E],
        domain: Range<E>,
    ) -> Result<Self, ChebyshevError> {
        validate_interpolation_data(xs, ys)?;

        let degree = xs.len() - 1;
        let mut matrix = Array2::<E>::zeros((xs.len(), xs.len()));

        for (row, &x) in xs.iter().enumerate() {
            let t = to_scaled(x, &domain);
            let basis = chebyshev_basis_values(t, degree);

            for (col, value) in basis.into_iter().enumerate() {
                matrix[[row, col]] = value;
            }
        }

        let rhs = Array1::from_vec(ys.to_vec());
        let coefficients = matrix.solve_into(rhs)?.to_vec();

        Self::new(coefficients, domain)
    }
}

fn validate_interpolation_data<E>(xs: &[E], ys: &[E]) -> Result<(), ChebyshevError>
where
    E: Float,
{
    if xs.is_empty() {
        return Err(ChebyshevError::EmptyInput);
    }

    if xs.len() != ys.len() {
        return Err(ChebyshevError::LengthMismatch);
    }

    if xs.iter().any(|x| !x.is_finite()) || ys.iter().any(|y| !y.is_finite()) {
        return Err(ChebyshevError::InvalidData);
    }

    for i in 0..xs.len() {
        for j in i + 1..xs.len() {
            if xs[i] == xs[j] {
                return Err(ChebyshevError::DuplicateAbscissa);
            }
        }
    }

    Ok(())
}

fn interpolation_domain<E>(xs: &[E]) -> Result<std::ops::Range<E>, ChebyshevError>
where
    E: Float,
{
    let mut lower = xs[0];
    let mut upper = xs[0];

    for &x in &xs[1..] {
        lower = lower.min(x);
        upper = upper.max(x);
    }

    if lower == upper {
        return Err(ChebyshevError::InvalidDomain);
    }

    Ok(lower..upper)
}

fn chebyshev_basis_values<E>(t: E, degree: usize) -> Vec<E>
where
    E: Float,
{
    let mut values = vec![E::zero(); degree + 1];

    values[0] = E::one();

    if degree == 0 {
        return values;
    }

    values[1] = t;

    let two = E::one() + E::one();

    for n in 2..=degree {
        values[n] = two * t * values[n - 1] - values[n - 2];
    }

    values
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{InterpolatePolynomialSeries, PolynomialCoefficients, PolynomialSeries};

    const EPS: f64 = 1.0e-10;

    fn assert_close(lhs: f64, rhs: f64) {
        assert!(
            (lhs - rhs).abs() <= EPS,
            "expected {lhs} ≈ {rhs}, difference = {}",
            (lhs - rhs).abs()
        );
    }

    #[test]
    fn interpolate_rejects_empty_input() {
        let err = ChebyshevSeries::<f64>::interpolate(&[], &[]).unwrap_err();

        assert!(matches!(err, ChebyshevError::EmptyInput));
    }

    #[test]
    fn interpolate_rejects_length_mismatch() {
        let err = ChebyshevSeries::<f64>::interpolate(&[0.0, 1.0], &[1.0]).unwrap_err();

        assert!(matches!(err, ChebyshevError::LengthMismatch));
    }

    #[test]
    fn interpolate_rejects_nan_x() {
        let err = ChebyshevSeries::<f64>::interpolate(&[0.0, f64::NAN], &[1.0, 2.0]).unwrap_err();

        assert!(matches!(err, ChebyshevError::InvalidData));
    }

    #[test]
    fn interpolate_rejects_nan_y() {
        let err = ChebyshevSeries::<f64>::interpolate(&[0.0, 1.0], &[1.0, f64::NAN]).unwrap_err();

        assert!(matches!(err, ChebyshevError::InvalidData));
    }

    #[test]
    fn interpolate_rejects_infinite_x() {
        let err =
            ChebyshevSeries::<f64>::interpolate(&[0.0, f64::INFINITY], &[1.0, 2.0]).unwrap_err();

        assert!(matches!(err, ChebyshevError::InvalidData));
    }

    #[test]
    fn interpolate_rejects_duplicate_x_values() {
        let err = ChebyshevSeries::<f64>::interpolate(&[0.0, 0.0], &[1.0, 2.0]).unwrap_err();

        assert!(matches!(err, ChebyshevError::DuplicateAbscissa));
    }

    #[test]
    fn interpolate_single_point_needs_explicit_domain() {
        let err = ChebyshevSeries::<f64>::interpolate(&[2.0], &[5.0]).unwrap_err();

        assert!(matches!(err, ChebyshevError::InvalidDomain));
    }

    #[test]
    fn interpolate_on_domain_single_point_constructs_constant_series() {
        let p = ChebyshevSeries::<f64>::interpolate_on_domain(&[2.0], &[5.0], 0.0..10.0).unwrap();

        assert_eq!(p.degree(), 0);
        assert_close(p.evaluate(0.0), 5.0);
        assert_close(p.evaluate(2.0), 5.0);
        assert_close(p.evaluate(10.0), 5.0);
    }

    #[test]
    fn interpolate_linear_function_on_inferred_domain() {
        let xs = [0.0, 10.0];
        let ys = [1.0, 21.0];

        let p = ChebyshevSeries::<f64>::interpolate(&xs, &ys).unwrap();

        assert_eq!(p.domain(), 0.0..10.0);
        assert_eq!(p.degree(), 1);

        for (&x, &y) in xs.iter().zip(ys.iter()) {
            assert_close(p.evaluate(x), y);
        }

        assert_close(p.evaluate(5.0), 11.0);
    }

    #[test]
    fn interpolate_quadratic_function_on_inferred_domain() {
        let xs = [-1.0, 0.0, 1.0];
        let ys = [1.0, 0.0, 1.0];

        let p = ChebyshevSeries::<f64>::interpolate(&xs, &ys).unwrap();

        assert_eq!(p.domain(), -1.0..1.0);
        assert_eq!(p.degree(), 2);

        for (&x, &y) in xs.iter().zip(ys.iter()) {
            assert_close(p.evaluate(x), y);
        }

        assert_close(p.evaluate(0.5), 0.25);
    }

    #[test]
    fn interpolate_cubic_function_on_inferred_domain() {
        let xs = [-1.0, 0.0, 1.0, 2.0];
        let ys = [-1.0, 0.0, 1.0, 8.0];

        let p = ChebyshevSeries::<f64>::interpolate(&xs, &ys).unwrap();

        for (&x, &y) in xs.iter().zip(ys.iter()) {
            assert_close(p.evaluate(x), y);
        }

        assert_close(p.evaluate(1.5), 3.375);
    }

    #[test]
    fn interpolate_on_domain_preserves_requested_domain() {
        let xs = [0.0, 10.0];
        let ys = [1.0, 21.0];

        let p = ChebyshevSeries::<f64>::interpolate_on_domain(&xs, &ys, -10.0..20.0).unwrap();

        assert_eq!(p.domain(), -10.0..20.0);

        for (&x, &y) in xs.iter().zip(ys.iter()) {
            assert_close(p.evaluate(x), y);
        }
    }

    #[test]
    fn interpolate_on_domain_rejects_invalid_domain() {
        let err = ChebyshevSeries::<f64>::interpolate_on_domain(&[0.0, 1.0], &[0.0, 1.0], 1.0..1.0)
            .unwrap_err();

        assert!(matches!(err, ChebyshevError::InvalidDomain));
    }

    #[test]
    fn interpolate_on_domain_points_need_not_span_domain() {
        let xs = [2.0, 4.0, 6.0];
        let ys = [4.0, 16.0, 36.0];

        let p = ChebyshevSeries::<f64>::interpolate_on_domain(&xs, &ys, 0.0..10.0).unwrap();

        assert_eq!(p.domain(), 0.0..10.0);

        for (&x, &y) in xs.iter().zip(ys.iter()) {
            assert_close(p.evaluate(x), y);
        }

        assert_close(p.evaluate(5.0), 25.0);
    }

    #[test]
    fn interpolation_coefficients_are_finite() {
        let xs = [-1.0, 0.0, 1.0];
        let ys = [1.0, 0.0, 1.0];

        let p = ChebyshevSeries::<f64>::interpolate(&xs, &ys).unwrap();

        assert!(p.coefficients().iter().all(|c| c.is_finite()));
    }
}
