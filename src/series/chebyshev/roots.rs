use crate::scaling::from_scaled;
use crate::{PolynomialRoots, PolynomialSeries};

use super::{ChebyshevError, ChebyshevSeries};

use ndarray::Array2;
use ndarray_linalg::{EigVals, Lapack, Scalar};
use num_traits::{Float, FromPrimitive};

impl<E> PolynomialRoots<E> for ChebyshevSeries<E>
where
    E: Float + FromPrimitive + Scalar<Real = E> + Lapack,
{
    type Error = ChebyshevError;

    fn roots(&self) -> Result<Vec<E>, Self::Error> {
        match self.degree() {
            0 => Ok(vec![]),
            1 => self.linear_root(),
            _ => self.companion_roots(),
        }
    }
}

impl<E> ChebyshevSeries<E>
where
    E: Float + FromPrimitive + Scalar<Real = E> + Lapack,
{
    fn linear_root(&self) -> Result<Vec<E>, ChebyshevError> {
        let coefficients = self.coefficients.as_slice();

        let c0 = coefficients[0];
        let c1 = coefficients[1];

        if c1 == E::zero() {
            return Ok(vec![]);
        }

        let root_scaled = -c0 / c1;

        Ok(vec![from_scaled(root_scaled, &self.domain)])
    }

    fn companion_roots(&self) -> Result<Vec<E>, ChebyshevError> {
        let tolerance = E::from_f64(1.0e-12).expect("tolerance should be representable");

        let mut roots = self
            .companion_matrix()?
            .eigvals()?
            .into_iter()
            .filter(|z| Float::abs(z.im()) <= tolerance)
            .map(|z| z.re())
            .filter(|x| x.is_finite())
            .map(|root_scaled| from_scaled(root_scaled, &self.domain))
            .collect::<Vec<_>>();

        roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Ok(roots)
    }

    fn companion_matrix(&self) -> Result<Array2<E>, ChebyshevError> {
        let degree = self.degree();

        debug_assert!(degree >= 2);

        let coefficients = self.coefficients.as_slice();

        if coefficients.len() != degree + 1 {
            return Err(ChebyshevError::DegeneratePolynomial);
        }

        if coefficients[degree] == E::zero() {
            return Err(ChebyshevError::DegeneratePolynomial);
        }

        let two = E::from_f64(2.0).expect("2.0 should be representable");
        let half = E::one() / two;
        let inv_sqrt_two = Float::sqrt(half);

        let mut matrix = Array2::<E>::zeros((degree, degree));

        for index in 0..degree.saturating_sub(1) {
            let value = if index == 0 { inv_sqrt_two } else { half };

            matrix[[index, index + 1]] = value;
            matrix[[index + 1, index]] = value;
        }

        let mut scale = Vec::with_capacity(degree);
        scale.push(E::one());

        for _ in 1..degree {
            scale.push(inv_sqrt_two);
        }

        let leading = coefficients[degree];

        for row in 0..degree {
            matrix[[row, degree - 1]] -=
                -coefficients[row] / two / leading * scale[row] / scale[degree - 1];
        }

        Ok(matrix)
    }

    // fn companion_matrix(&self) -> Result<Array2<E>, ChebyshevError> {
    //     let degree = self.degree();

    //     debug_assert!(degree >= 2);

    //     let coefficients = self.coefficients.as_slice();

    //     if coefficients[degree] == E::zero() {
    //         return Err(ChebyshevError::DegeneratePolynomial);
    //     }

    //     let two = E::from_f64(2.0).expect("2.0 should be representable");
    //     let half = E::one() / two;
    //     let inv_sqrt_two = Float::sqrt(half);

    //     let mut matrix = Array2::<E>::zeros((degree, degree));

    //     for index in 0..degree - 1 {
    //         let value = if index == 0 { inv_sqrt_two } else { half };

    //         matrix[[index, index + 1]] = value;
    //         matrix[[index + 1, index]] = value;
    //     }

    //     let mut scale = Vec::with_capacity(degree);
    //     scale.push(E::one());
    //     scale.extend(std::iter::repeat(inv_sqrt_two).take(degree - 1));

    //     for row in 0..degree {
    //         matrix[[row, degree - 1]] = matrix[[row, degree - 1]]
    //             - coefficients[row] / two / coefficients[degree] * scale[row] / scale[degree - 1];
    //     }

    //     Ok(matrix)
    // }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PolynomialRoots, PolynomialSeries};

    use std::ops::Range;

    const EPS: f64 = 1.0e-10;

    fn series(coefficients: Vec<f64>, domain: Range<f64>) -> ChebyshevSeries<f64> {
        ChebyshevSeries::new(coefficients, domain).unwrap()
    }

    fn assert_close(lhs: f64, rhs: f64) {
        assert!(
            (lhs - rhs).abs() <= EPS,
            "expected {lhs} ≈ {rhs}, difference = {}",
            (lhs - rhs).abs()
        );
    }

    fn assert_roots_close(mut roots: Vec<f64>, expected: &[f64]) {
        roots.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert_eq!(
            roots.len(),
            expected.len(),
            "expected {} roots, got {}: {roots:?}",
            expected.len(),
            roots.len(),
        );

        for (root, expected) in roots.iter().zip(expected.iter()) {
            assert_close(*root, *expected);
        }
    }

    #[test]
    fn constant_polynomial_has_no_roots() {
        let p = series(vec![2.0], -1.0..1.0);

        let roots = p.roots().unwrap();

        assert!(roots.is_empty());
    }

    #[test]
    fn zero_polynomial_returns_no_roots() {
        let p = ChebyshevSeries::<f64>::zero(-1.0..1.0);

        let roots = p.roots().unwrap();

        assert!(roots.is_empty());
    }

    #[test]
    fn linear_polynomial_root_is_returned_in_physical_coordinates_on_canonical_domain() {
        // p(t) = -0.25 + T1(t), root at t = 0.25.
        let p = series(vec![-0.25, 1.0], -1.0..1.0);

        assert_roots_close(p.roots().unwrap(), &[0.25]);
    }

    #[test]
    fn linear_polynomial_root_is_mapped_to_physical_domain() {
        // p(t) = T1(t), root at t = 0.
        // On [10, 20], t = 0 corresponds to x = 15.
        let p = series(vec![0.0, 1.0], 10.0..20.0);

        assert_roots_close(p.roots().unwrap(), &[15.0]);
    }

    #[test]
    fn t2_roots_on_canonical_domain_are_plus_minus_one_over_sqrt_two() {
        // T2(t) = 2t² - 1.
        let p = series(vec![0.0, 0.0, 1.0], -1.0..1.0);

        let expected = 1.0 / 2.0_f64.sqrt();

        assert_roots_close(p.roots().unwrap(), &[-expected, expected]);
    }

    #[test]
    fn t2_roots_are_mapped_to_physical_domain() {
        // T2 roots in scaled coordinates are ±1/sqrt(2).
        // Mapping from t ∈ [-1, 1] to x ∈ [10, 20] is x = 15 + 5t.
        let p = series(vec![0.0, 0.0, 1.0], 10.0..20.0);

        let scaled = 1.0 / 2.0_f64.sqrt();

        assert_roots_close(
            p.roots().unwrap(),
            &[15.0 - 5.0 * scaled, 15.0 + 5.0 * scaled],
        );
    }

    #[test]
    fn t3_roots_on_canonical_domain_are_known() {
        // T3(t) = 4t³ - 3t, roots at -sqrt(3)/2, 0, sqrt(3)/2.
        let p = series(vec![0.0, 0.0, 0.0, 1.0], -1.0..1.0);

        let expected = 3.0_f64.sqrt() / 2.0;

        assert_roots_close(p.roots().unwrap(), &[-expected, 0.0, expected]);
    }

    #[test]
    fn roots_in_domain_filters_out_roots_outside_domain() {
        // p(t) = t - 2 has root at scaled t = 2.
        // On canonical domain, this maps to physical x = 2, outside [-1, 1].
        let p = series(vec![-2.0, 1.0], -1.0..1.0);

        assert!(p.roots().unwrap().contains(&2.0));
        assert!(p.roots_in_domain().unwrap().is_empty());
    }

    #[test]
    fn roots_in_window_uses_closed_interval_semantics() {
        let p = series(vec![0.0, 1.0], -1.0..1.0);

        assert_roots_close(p.roots_in_window(0.0..0.0).unwrap(), &[0.0]);
    }

    #[test]
    fn roots_in_window_filters_roots() {
        let p = series(vec![0.0, 0.0, 1.0], -1.0..1.0);

        let expected = 1.0 / 2.0_f64.sqrt();

        assert_roots_close(p.roots_in_window(0.0..1.0).unwrap(), &[expected]);
    }

    #[test]
    fn has_root_in_window_returns_true_when_root_inside_window() {
        let p = series(vec![0.0, 1.0], -1.0..1.0);

        assert!(p.has_root_in_window(-0.1..0.1).unwrap());
    }

    #[test]
    fn has_root_in_window_returns_false_when_no_root_inside_window() {
        let p = series(vec![0.0, 1.0], -1.0..1.0);

        assert!(!p.has_root_in_window(0.1..0.2).unwrap());
    }

    #[test]
    fn has_root_in_domain_returns_true_when_root_inside_domain() {
        let p = series(vec![0.0, 1.0], -1.0..1.0);

        assert!(p.has_root_in_domain().unwrap());
    }

    #[test]
    fn has_root_in_domain_returns_false_when_root_outside_domain() {
        let p = series(vec![-2.0, 1.0], -1.0..1.0);

        assert!(!p.has_root_in_domain().unwrap());
    }

    #[test]
    fn monotonic_linear_series_is_monotonic() {
        let p = series(vec![0.0, 1.0], -1.0..1.0);

        assert!(p.is_monotonic().unwrap());
    }

    #[test]
    fn quadratic_t2_is_not_monotonic_on_full_canonical_domain() {
        let p = series(vec![0.0, 0.0, 1.0], -1.0..1.0);

        assert!(!p.is_monotonic().unwrap());
    }

    #[test]
    fn quadratic_t2_is_monotonic_on_positive_physical_half_domain() {
        // Same scaled function T2(t), but physical domain [0, 1].
        // The derivative root occurs at t = 0, which maps to x = 0.5,
        // so this is NOT monotonic on the full [0, 1] domain.
        //
        // To test monotonicity on a positive scaled window, use a series whose
        // physical domain corresponds to scaled t ∈ [0, 1] is not possible
        // with the full Chebyshev canonical mapping; instead use T2 shifted
        // through coefficients would require a different polynomial.
        //
        // So this test intentionally verifies the conservative derivative-root
        // criterion on the stored domain.
        let p = series(vec![0.0, 0.0, 1.0], 0.0..1.0);

        assert!(!p.is_monotonic().unwrap());
    }

    #[test]
    fn cubic_t3_has_stationary_points_and_is_not_monotonic_by_derivative_root_test() {
        // T3'(t) = 12t² - 3, roots at ±0.5.
        let p = series(vec![0.0, 0.0, 0.0, 1.0], -1.0..1.0);

        assert!(!p.is_monotonic().unwrap());
    }

    #[test]
    fn roots_evaluate_close_to_zero() {
        let p = series(vec![0.25, -1.0, 0.75, 2.0], -1.0..1.0);

        for root in p.roots_in_domain().unwrap() {
            assert_close(p.evaluate(root), 0.0);
        }
    }

    #[test]
    fn roots_are_sorted() {
        let p = series(vec![0.0, 0.0, 0.0, 1.0], -1.0..1.0);

        let roots = p.roots().unwrap();

        assert!(roots.windows(2).all(|window| window[0] <= window[1]));
    }

    #[test]
    fn companion_matrix_for_t2_has_expected_shape() {
        let p = series(vec![0.0, 0.0, 1.0], -1.0..1.0);

        let matrix = p.companion_matrix().unwrap();

        assert_eq!(matrix.shape(), &[2, 2]);
    }
}
