use crate::{FitPolynomialSeries, FitReport, scaling::to_scaled};

use super::{ChebyshevError, ChebyshevSeries};

use ndarray::{Array1, Array2};
use ndarray_linalg::{Inverse, Lapack, LeastSquaresSvdInto, Scalar};
use num_traits::{Float, FromPrimitive};
use std::ops::Range;

impl<E> FitPolynomialSeries<E> for ChebyshevSeries<E>
where
    E: Float + FromPrimitive + Scalar<Real = E> + Lapack,
{
    type Error = ChebyshevError;

    fn fit_report(xs: &[E], ys: &[E], degree: usize) -> Result<FitReport<E, Self>, Self::Error> {
        let domain = fitting_domain(xs)?;
        Self::fit_report_on_domain(xs, ys, degree, domain)
    }

    fn fit_weighted_report(
        xs: &[E],
        ys: &[E],
        weights: &[E],
        degree: usize,
    ) -> Result<FitReport<E, Self>, Self::Error> {
        let domain = fitting_domain(xs)?;
        Self::fit_weighted_report_on_domain(xs, ys, weights, degree, domain)
    }
}

impl<E> ChebyshevSeries<E>
where
    E: Float + FromPrimitive + Scalar<Real = E> + Lapack,
{
    pub fn fit_on_domain(
        xs: &[E],
        ys: &[E],
        degree: usize,
        domain: Range<E>,
    ) -> Result<Self, ChebyshevError> {
        Ok(Self::fit_report_on_domain(xs, ys, degree, domain)?.series)
    }

    pub fn fit_weighted_on_domain(
        xs: &[E],
        ys: &[E],
        weights: &[E],
        degree: usize,
        domain: Range<E>,
    ) -> Result<Self, ChebyshevError> {
        Ok(Self::fit_weighted_report_on_domain(xs, ys, weights, degree, domain)?.series)
    }

    pub fn fit_report(
        xs: &[E],
        ys: &[E],
        degree: usize,
    ) -> Result<FitReport<E, Self>, ChebyshevError> {
        let domain = fitting_domain(xs)?;
        Self::fit_report_on_domain(xs, ys, degree, domain)
    }

    pub fn fit_report_on_domain(
        xs: &[E],
        ys: &[E],
        degree: usize,
        domain: Range<E>,
    ) -> Result<FitReport<E, Self>, ChebyshevError> {
        validate_fit_data(xs, ys, degree)?;
        validate_domain(&domain)?;

        let design = design_matrix(xs, degree, &domain);
        let rhs = Array1::from_vec(ys.to_vec());

        fit_report_from_design(xs, ys, degree, domain, design.clone(), design, rhs)
    }

    pub fn fit_weighted_report(
        xs: &[E],
        ys: &[E],
        weights: &[E],
        degree: usize,
    ) -> Result<FitReport<E, Self>, ChebyshevError> {
        let domain = fitting_domain(xs)?;
        Self::fit_weighted_report_on_domain(xs, ys, weights, degree, domain)
    }

    pub fn fit_weighted_report_on_domain(
        xs: &[E],
        ys: &[E],
        weights: &[E],
        degree: usize,
        domain: Range<E>,
    ) -> Result<FitReport<E, Self>, ChebyshevError> {
        validate_fit_data(xs, ys, degree)?;
        validate_weights(xs, weights)?;
        validate_domain(&domain)?;

        let unweighted_design = design_matrix(xs, degree, &domain);
        let mut weighted_design = unweighted_design.clone();
        let mut weighted_rhs = Array1::from_vec(ys.to_vec());

        for row in 0..xs.len() {
            let scale = Scalar::sqrt(weights[row]);

            for col in 0..=degree {
                weighted_design[[row, col]] *= scale;
            }

            weighted_rhs[row] *= scale;
        }

        fit_report_from_design(
            xs,
            ys,
            degree,
            domain,
            unweighted_design,
            weighted_design,
            weighted_rhs,
        )
    }
}

fn fit_report_from_design<E>(
    xs: &[E],
    ys: &[E],
    degree: usize,
    domain: Range<E>,
    unweighted_design: Array2<E>,
    fitted_design: Array2<E>,
    fitted_rhs: Array1<E>,
) -> Result<FitReport<E, ChebyshevSeries<E>>, ChebyshevError>
where
    E: Float + FromPrimitive + Scalar<Real = E> + Lapack,
{
    let coefficients_array = fitted_design
        .clone()
        .least_squares_into(fitted_rhs)?
        .solution;
    let coefficients = coefficients_array.to_vec();

    let series = ChebyshevSeries::new(coefficients.clone(), domain)?;

    let fitted_values_array = unweighted_design.dot(&coefficients_array);
    let fitted_values = fitted_values_array.to_vec();

    let residuals = ys
        .iter()
        .zip(fitted_values.iter())
        .map(|(&y, &fit)| y - fit)
        .collect::<Vec<_>>();

    let residual_sum_of_squares = residuals
        .iter()
        .fold(E::zero(), |sum, &residual| sum + residual * residual);

    let parameters = degree + 1;
    let degrees_of_freedom = xs.len().saturating_sub(parameters);

    let residual_variance = if degrees_of_freedom > 0 {
        Some(
            residual_sum_of_squares
                / E::from_usize(degrees_of_freedom).expect("usize should be representable"),
        )
    } else {
        None
    };

    let covariance = if let Some(residual_variance) = residual_variance {
        let normal = fitted_design.t().dot(&fitted_design);
        let inverse = normal.inv()?;
        let covariance = inverse.mapv(|value| value * residual_variance);

        Some(array2_to_vecs(covariance))
    } else {
        None
    };

    Ok(FitReport {
        series,
        coefficients,
        covariance,
        fitted_values,
        residuals,
        degrees_of_freedom,
        residual_sum_of_squares,
        residual_variance,
    })
}

fn array2_to_vecs<E>(array: Array2<E>) -> Vec<Vec<E>>
where
    E: Copy,
{
    array.rows().into_iter().map(|row| row.to_vec()).collect()
}

fn validate_fit_data<E>(xs: &[E], ys: &[E], degree: usize) -> Result<(), ChebyshevError>
where
    E: Float,
{
    if xs.is_empty() {
        return Err(ChebyshevError::EmptyInput);
    }

    if xs.len() != ys.len() {
        return Err(ChebyshevError::LengthMismatch);
    }

    if xs.len() < degree + 1 {
        return Err(ChebyshevError::UnderdeterminedFit);
    }

    if xs.iter().any(|x| !x.is_finite()) || ys.iter().any(|y| !y.is_finite()) {
        return Err(ChebyshevError::InvalidData);
    }

    Ok(())
}

fn validate_weights<E>(xs: &[E], weights: &[E]) -> Result<(), ChebyshevError>
where
    E: Float,
{
    if xs.len() != weights.len() {
        return Err(ChebyshevError::LengthMismatch);
    }

    if weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight < E::zero())
    {
        return Err(ChebyshevError::InvalidWeights);
    }

    Ok(())
}

fn validate_domain<E>(domain: &Range<E>) -> Result<(), ChebyshevError>
where
    E: Float,
{
    if domain.start.is_finite() && domain.end.is_finite() && domain.start < domain.end {
        Ok(())
    } else {
        Err(ChebyshevError::InvalidDomain)
    }
}

fn fitting_domain<E>(xs: &[E]) -> Result<Range<E>, ChebyshevError>
where
    E: Float,
{
    if xs.is_empty() {
        return Err(ChebyshevError::EmptyInput);
    }

    if xs.iter().any(|x| !x.is_finite()) {
        return Err(ChebyshevError::InvalidData);
    }

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

fn design_matrix<E>(xs: &[E], degree: usize, domain: &Range<E>) -> Array2<E>
where
    E: Float + FromPrimitive,
{
    let mut matrix = Array2::<E>::zeros((xs.len(), degree + 1));

    for (row, &x) in xs.iter().enumerate() {
        let t = to_scaled(x, domain);
        let basis = chebyshev_basis_values(t, degree);

        for (col, value) in basis.into_iter().enumerate() {
            matrix[[row, col]] = value;
        }
    }

    matrix
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
    use crate::{FitPolynomialSeries, PolynomialSeries};

    const EPS: f64 = 1.0e-9;

    fn assert_close(lhs: f64, rhs: f64) {
        assert!(
            (lhs - rhs).abs() <= EPS,
            "expected {lhs} ≈ {rhs}, difference = {}",
            (lhs - rhs).abs()
        );
    }

    #[test]
    fn fit_rejects_empty_input() {
        let err = ChebyshevSeries::<f64>::fit(&[], &[], 0).unwrap_err();

        assert!(matches!(err, ChebyshevError::EmptyInput));
    }

    #[test]
    fn fit_rejects_length_mismatch() {
        let err = ChebyshevSeries::<f64>::fit(&[0.0, 1.0], &[1.0], 1).unwrap_err();

        assert!(matches!(err, ChebyshevError::LengthMismatch));
    }

    #[test]
    fn fit_rejects_underdetermined_problem() {
        let err = ChebyshevSeries::<f64>::fit(&[0.0, 1.0], &[0.0, 1.0], 2).unwrap_err();

        assert!(matches!(err, ChebyshevError::UnderdeterminedFit));
    }

    #[test]
    fn fit_rejects_non_finite_x() {
        let err = ChebyshevSeries::<f64>::fit(&[0.0, f64::NAN], &[1.0, 2.0], 1).unwrap_err();

        assert!(matches!(err, ChebyshevError::InvalidData));
    }

    #[test]
    fn fit_rejects_non_finite_y() {
        let err = ChebyshevSeries::<f64>::fit(&[0.0, 1.0], &[1.0, f64::INFINITY], 1).unwrap_err();

        assert!(matches!(err, ChebyshevError::InvalidData));
    }

    #[test]
    fn fit_on_domain_rejects_invalid_domain() {
        let err = ChebyshevSeries::<f64>::fit_on_domain(&[0.0, 1.0], &[0.0, 1.0], 1, 1.0..1.0)
            .unwrap_err();

        assert!(matches!(err, ChebyshevError::InvalidDomain));
    }

    #[test]
    fn fit_weighted_rejects_length_mismatch() {
        let err =
            ChebyshevSeries::<f64>::fit_weighted(&[0.0, 1.0], &[0.0, 1.0], &[1.0], 1).unwrap_err();

        assert!(matches!(err, ChebyshevError::LengthMismatch));
    }

    #[test]
    fn fit_weighted_rejects_negative_weight() {
        let err = ChebyshevSeries::<f64>::fit_weighted(&[0.0, 1.0], &[0.0, 1.0], &[1.0, -1.0], 1)
            .unwrap_err();

        assert!(matches!(err, ChebyshevError::InvalidWeights));
    }

    #[test]
    fn fit_weighted_rejects_nan_weight() {
        let err =
            ChebyshevSeries::<f64>::fit_weighted(&[0.0, 1.0], &[0.0, 1.0], &[1.0, f64::NAN], 1)
                .unwrap_err();

        assert!(matches!(err, ChebyshevError::InvalidWeights));
    }

    #[test]
    fn fit_linear_exact_data() {
        let xs = [0.0, 5.0, 10.0];
        let ys = [1.0, 11.0, 21.0];

        let p = ChebyshevSeries::<f64>::fit(&xs, &ys, 1).unwrap();

        assert_eq!(p.domain(), 0.0..10.0);
        assert_eq!(p.degree(), 1);

        for (&x, &y) in xs.iter().zip(ys.iter()) {
            assert_close(p.evaluate(x), y);
        }

        assert_close(p.evaluate(2.5), 6.0);
        assert_close(p.evaluate(7.5), 16.0);
    }

    #[test]
    fn fit_quadratic_exact_data() {
        let xs = [-1.0, -0.5, 0.0, 0.5, 1.0];
        let ys = xs.map(|x| x * x);

        let p = ChebyshevSeries::<f64>::fit(&xs, &ys, 2).unwrap();

        for (&x, &y) in xs.iter().zip(ys.iter()) {
            assert_close(p.evaluate(x), y);
        }

        assert_close(p.evaluate(0.25), 0.0625);
    }

    #[test]
    fn fit_on_domain_preserves_requested_domain() {
        let xs = [2.0, 4.0, 6.0];
        let ys = [4.0, 8.0, 12.0];

        let p = ChebyshevSeries::<f64>::fit_on_domain(&xs, &ys, 1, 0.0..10.0).unwrap();

        assert_eq!(p.domain(), 0.0..10.0);

        for (&x, &y) in xs.iter().zip(ys.iter()) {
            assert_close(p.evaluate(x), y);
        }
    }

    #[test]
    fn fit_report_contains_series_and_coefficients() {
        let xs = [0.0, 5.0, 10.0];
        let ys = [1.0, 11.0, 21.0];

        let report = ChebyshevSeries::<f64>::fit_report(&xs, &ys, 1).unwrap();

        assert_eq!(report.series.domain(), 0.0..10.0);
        assert_eq!(report.series.degree(), 1);
        assert_eq!(report.coefficients, report.series.coefficients());

        for (&x, &y) in xs.iter().zip(ys.iter()) {
            assert_close(report.series.evaluate(x), y);
        }
    }

    #[test]
    fn fit_report_contains_fitted_values() {
        let xs = [0.0, 5.0, 10.0];
        let ys = [1.0, 11.0, 21.0];

        let report = ChebyshevSeries::<f64>::fit_report(&xs, &ys, 1).unwrap();

        assert_eq!(report.fitted_values.len(), xs.len());

        for (&fit, &y) in report.fitted_values.iter().zip(ys.iter()) {
            assert_close(fit, y);
        }
    }

    #[test]
    fn fit_report_contains_residuals() {
        let xs = [0.0, 5.0, 10.0];
        let ys = [1.0, 11.0, 21.0];

        let report = ChebyshevSeries::<f64>::fit_report(&xs, &ys, 1).unwrap();

        assert_eq!(report.residuals.len(), xs.len());

        for &residual in &report.residuals {
            assert_close(residual, 0.0);
        }

        assert_close(report.residual_sum_of_squares, 0.0);
    }

    #[test]
    fn fit_report_has_correct_degrees_of_freedom() {
        let xs = [0.0, 2.5, 5.0, 7.5, 10.0];
        let ys = [1.0, 6.0, 11.0, 16.0, 21.0];

        let report = ChebyshevSeries::<f64>::fit_report(&xs, &ys, 1).unwrap();

        assert_eq!(report.degrees_of_freedom, 3);
        assert!(report.residual_variance.is_some());
    }

    #[test]
    fn fit_report_has_no_residual_variance_for_exactly_determined_fit() {
        let xs = [0.0, 10.0];
        let ys = [1.0, 21.0];

        let report = ChebyshevSeries::<f64>::fit_report(&xs, &ys, 1).unwrap();

        assert_eq!(report.degrees_of_freedom, 0);
        assert!(report.residual_variance.is_none());
        assert!(report.covariance.is_none());
    }

    #[test]
    fn fit_report_covariance_has_expected_shape_when_overdetermined() {
        let xs = [0.0, 2.5, 5.0, 7.5, 10.0];
        let ys = [1.0, 6.0, 11.0, 16.0, 21.0];

        let report = ChebyshevSeries::<f64>::fit_report(&xs, &ys, 1).unwrap();

        let covariance = report.covariance.unwrap();

        assert_eq!(covariance.len(), 2);
        assert_eq!(covariance[0].len(), 2);
        assert_eq!(covariance[1].len(), 2);
    }

    #[test]
    fn weighted_fit_matches_unweighted_fit_for_uniform_weights() {
        let xs = [0.0, 2.5, 5.0, 7.5, 10.0];
        let ys = [1.0, 6.1, 10.9, 16.2, 20.8];
        let weights = [1.0; 5];

        let unweighted = ChebyshevSeries::<f64>::fit(&xs, &ys, 1).unwrap();
        let weighted = ChebyshevSeries::<f64>::fit_weighted(&xs, &ys, &weights, 1).unwrap();

        assert_eq!(weighted.domain(), unweighted.domain());

        for x in xs {
            assert_close(weighted.evaluate(x), unweighted.evaluate(x));
        }
    }

    #[test]
    fn weighted_fit_report_contains_requested_series() {
        let xs = [0.0, 5.0, 10.0];
        let ys = [1.0, 11.0, 21.0];
        let weights = [1.0, 2.0, 3.0];

        let report = ChebyshevSeries::<f64>::fit_weighted_report(&xs, &ys, &weights, 1).unwrap();

        assert_eq!(report.series.domain(), 0.0..10.0);
        assert_eq!(report.series.degree(), 1);

        for (&x, &y) in xs.iter().zip(ys.iter()) {
            assert_close(report.series.evaluate(x), y);
        }
    }

    #[test]
    fn weighted_fit_changes_solution_when_outlier_has_low_weight() {
        let xs = [0.0, 1.0, 2.0];
        let ys = [0.0, 1.0, 100.0];

        let unweighted = ChebyshevSeries::<f64>::fit(&xs, &ys, 1).unwrap();

        let weights = [1.0, 1.0, 1.0e-6];
        let weighted = ChebyshevSeries::<f64>::fit_weighted(&xs, &ys, &weights, 1).unwrap();

        let unweighted_error_at_one = (unweighted.evaluate(1.0) - 1.0).abs();
        let weighted_error_at_one = (weighted.evaluate(1.0) - 1.0).abs();

        assert!(weighted_error_at_one < unweighted_error_at_one);
    }

    #[test]
    fn fit_report_on_domain_works_with_points_inside_larger_domain() {
        let xs = [2.0, 4.0, 6.0, 8.0];
        let ys = [4.0, 8.0, 12.0, 16.0];

        let report = ChebyshevSeries::<f64>::fit_report_on_domain(&xs, &ys, 1, 0.0..10.0).unwrap();

        assert_eq!(report.series.domain(), 0.0..10.0);

        for (&x, &y) in xs.iter().zip(ys.iter()) {
            assert_close(report.series.evaluate(x), y);
        }
    }
}
