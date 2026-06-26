use crate::PolynomialSeries;

/// Batch evaluation of a polynomial series.
pub trait PolynomialEvaluateMany<E>: PolynomialSeries<E> {
    /// Evaluate the polynomial at many physical coordinates.
    fn evaluate_many<I>(&self, xs: I) -> Vec<E>
    where
        I: IntoIterator<Item = E>,
    {
        xs.into_iter().map(|x| self.evaluate(x)).collect()
    }

    /// Evaluate the polynomial at many scaled coordinates.
    fn evaluate_scaled_many<I>(&self, ts: I) -> Vec<E>
    where
        I: IntoIterator<Item = E>,
    {
        ts.into_iter().map(|t| self.evaluate_scaled(t)).collect()
    }
}

impl<T, E> PolynomialEvaluateMany<E> for T where T: PolynomialSeries<E> {}
