/// Result of fitting a polynomial series to data.
#[derive(Clone, Debug)]
pub struct FitReport<E, S> {
    /// The fitted polynomial series.
    pub series: S,

    /// Native-basis fitted coefficients.
    pub coefficients: Vec<E>,

    /// Estimated coefficient covariance matrix, if available.
    pub covariance: Option<Vec<Vec<E>>>,

    /// Fitted values at the input abscissae.
    pub fitted_values: Vec<E>,

    /// Residuals `y_i - fitted_i`.
    pub residuals: Vec<E>,

    /// Number of residual degrees of freedom.
    pub degrees_of_freedom: usize,

    /// Sum of squared residuals.
    pub residual_sum_of_squares: E,

    /// Residual variance estimate, if degrees of freedom are positive.
    pub residual_variance: Option<E>,
}

/// Construction by least-squares fitting.
pub trait FitPolynomialSeries<E>: Sized {
    /// Error returned when fitting fails.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Fit a polynomial and return diagnostics.
    fn fit_report(xs: &[E], ys: &[E], degree: usize) -> Result<FitReport<E, Self>, Self::Error>;

    /// Fit a weighted polynomial and return diagnostics.
    fn fit_weighted_report(
        xs: &[E],
        ys: &[E],
        weights: &[E],
        degree: usize,
    ) -> Result<FitReport<E, Self>, Self::Error>;

    /// Fit a polynomial and return only the fitted series.
    fn fit(xs: &[E], ys: &[E], degree: usize) -> Result<Self, Self::Error> {
        Ok(Self::fit_report(xs, ys, degree)?.series)
    }

    /// Fit a weighted polynomial and return only the fitted series.
    fn fit_weighted(xs: &[E], ys: &[E], weights: &[E], degree: usize) -> Result<Self, Self::Error> {
        Ok(Self::fit_weighted_report(xs, ys, weights, degree)?.series)
    }
}
