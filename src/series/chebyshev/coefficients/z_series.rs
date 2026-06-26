use num_traits::Float;

use super::CSeries;

/// Laurent-like z-series representation of a Chebyshev series.
///
/// This is an internal representation used to multiply Chebyshev series by
/// convolution.
///
/// The identity is:
///
/// ```text
/// T_n(t) = 1/2 (z^n + z^-n)
/// ```
///
/// so a Chebyshev series can be mapped to a symmetric z-series, multiplied by
/// convolution, then mapped back to Chebyshev coefficients.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ZSeries<E>(Vec<E>);

impl<E> From<CSeries<E>> for ZSeries<E>
where
    E: Copy
        + num_traits::Zero
        + num_traits::One
        + std::ops::Add<Output = E>
        + std::ops::Div<Output = E>,
{
    fn from(value: CSeries<E>) -> Self {
        let n = value.len();

        if n == 0 {
            return Self(vec![E::zero()]);
        }

        let two = E::one() + E::one();
        let mut z = vec![E::zero(); 2 * n - 1];

        for (index, coefficient) in value.into_iter().enumerate() {
            let scaled = coefficient / two;
            z[n - 1 + index] = z[n - 1 + index] + scaled;
            z[n - 1 - index] = z[n - 1 - index] + scaled;
        }

        Self(z)
    }
}

impl<E> ZSeries<E> {
    /// Create a new coefficient series.
    pub(crate) fn new(coefficients: Vec<E>) -> Self {
        Self(coefficients)
    }

    pub(crate) fn as_slice(&self) -> &[E] {
        &self.0
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [E] {
        &mut self.0
    }

    pub(crate) fn into_vec(self) -> Vec<E> {
        self.0
    }

    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub(crate) fn iter(&self) -> std::slice::Iter<'_, E> {
        self.0.iter()
    }

    pub(crate) fn into_iter(self) -> std::vec::IntoIter<E> {
        self.0.into_iter()
    }

    pub(crate) fn coefficient(&self, idx: usize) -> Option<&E> {
        self.0.get(idx)
    }
}

fn convolve<E>(lhs: &[E], rhs: &[E]) -> Vec<E>
where
    E: Copy + num_traits::Zero + std::ops::Add<Output = E> + std::ops::Mul<Output = E>,
{
    if lhs.is_empty() || rhs.is_empty() {
        return vec![];
    }

    let mut output = vec![E::zero(); lhs.len() + rhs.len() - 1];

    for (i, &a) in lhs.iter().enumerate() {
        for (j, &b) in rhs.iter().enumerate() {
            output[i + j] = output[i + j] + a * b;
        }
    }

    output
}

impl<E> std::ops::Mul for ZSeries<E>
where
    E: Float,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(convolve(&self.0, &rhs.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convolve_empty_input_returns_empty_output() {
        let output = convolve::<f64>(&[], &[1.0, 2.0]);

        assert!(output.is_empty());
    }

    #[test]
    fn convolve_singletons_multiplies_values() {
        let output = convolve(&[2.0], &[3.0]);

        assert_eq!(output, vec![6.0]);
    }

    #[test]
    fn convolve_general_vectors() {
        let output = convolve(&[1.0, 2.0, 3.0], &[4.0, 5.0]);

        assert_eq!(output, vec![4.0, 13.0, 22.0, 15.0]);
    }
}
