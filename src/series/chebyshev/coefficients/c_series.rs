use num_traits::Float;

use super::ZSeries;

/// Coefficients of a Chebyshev series.
///
/// A `CSeries` stores the coefficients `c_n` in
///
/// ```text
/// p(t) = Σ c_n T_n(t)
/// ```
///
/// Coefficients are stored in ascending order of Chebyshev degree:
///
/// ```text
/// [c_0, c_1, c_2, ...]
/// ```
///
/// The zero polynomial is represented as `[0]`, not as an empty array.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct CSeries<E>(Vec<E>);

impl<E> From<ZSeries<E>> for CSeries<E>
where
    E: Float,
{
    fn from(value: ZSeries<E>) -> Self {
        let n = (value.len() + 1) / 2;
        let two = E::one() + E::one();

        let mut coefficients = Vec::with_capacity(n);

        for index in 0..n {
            let mut coefficient = *value.coefficient(n - 1 + index).unwrap() * two;

            if index == 0 {
                coefficient = coefficient / two;
            }

            coefficients.push(coefficient);
        }

        Self::new(coefficients).trimmed()
    }
}

impl<E> CSeries<E> {
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
}

impl<E> From<Vec<E>> for CSeries<E> {
    fn from(value: Vec<E>) -> Self {
        Self::new(value)
    }
}

impl<E> CSeries<E>
where
    E: Float,
{
    /// Return the zero Chebyshev series.
    pub(crate) fn zero() -> Self {
        Self(vec![E::zero()])
    }

    /// Return the polynomial degree.
    ///
    /// The zero polynomial has degree zero by convention.
    pub(crate) fn degree(&self) -> usize {
        self.len().saturating_sub(1)
    }

    /// Return `true` if every stored coefficient is zero.
    pub(crate) fn is_zero(&self) -> bool {
        self.0.iter().all(|coefficient| *coefficient == E::zero())
    }

    /// Return a coefficient series with trailing zero coefficients removed.
    ///
    /// The result is never empty. If all coefficients are zero, this returns
    /// the canonical zero representation `[0]`.
    pub(crate) fn trimmed(mut self) -> Self {
        while self.0.len() > 1 && self.0.last().copied() == Some(E::zero()) {
            self.0.pop();
        }

        if self.0.is_empty() {
            Self::zero()
        } else {
            self
        }
    }

    pub(crate) fn truncate(&self, degree: usize) -> Self {
        let len = degree.saturating_add(1).min(self.len());

        Self::from(self.as_slice()[..len].to_vec()).trimmed()
    }

    pub(crate) fn trim_by_absolute_tolerance(&self, tolerance: E) -> Self {
        let tolerance = tolerance.abs();

        let Some(last_significant_index) = self
            .as_slice()
            .iter()
            .rposition(|coefficient| coefficient.abs() > tolerance)
        else {
            return Self::zero();
        };

        Self::from(self.as_slice()[..=last_significant_index].to_vec()).trimmed()
    }
}

impl<E> CSeries<E>
where
    E: Copy + std::ops::Mul<Output = E>,
{
    /// Multiply all coefficients by `factor`.
    pub(crate) fn scale_mut(&mut self, factor: E) {
        for coefficient in &mut self.0 {
            *coefficient = *coefficient * factor;
        }
    }
}

impl<E> std::ops::Add for CSeries<E>
where
    E: Float,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let len = self.len().max(rhs.len());
        let mut coefficients = Vec::with_capacity(len);

        for index in 0..len {
            let lhs = self.0.get(index).copied().unwrap_or_else(E::zero);
            let rhs = rhs.0.get(index).copied().unwrap_or_else(E::zero);
            coefficients.push(lhs + rhs);
        }

        Self::new(coefficients).trimmed()
    }
}

impl<E> std::ops::AddAssign for CSeries<E>
where
    E: Float,
{
    fn add_assign(&mut self, other: Self) {
        *self = self.clone() + other;
    }
}

impl<E> std::ops::Sub for CSeries<E>
where
    E: Float,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let len = self.len().max(rhs.len());
        let mut coefficients = Vec::with_capacity(len);

        for index in 0..len {
            let lhs = self.0.get(index).copied().unwrap_or_else(E::zero);
            let rhs = rhs.0.get(index).copied().unwrap_or_else(E::zero);
            coefficients.push(lhs - rhs);
        }

        Self::new(coefficients).trimmed()
    }
}

impl<E> std::ops::SubAssign for CSeries<E>
where
    E: Float,
{
    fn sub_assign(&mut self, other: Self) {
        *self = self.clone() - other;
    }
}

impl<E> std::ops::Mul for CSeries<E>
where
    E: Float,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let lhs = ZSeries::from(self);
        let rhs = ZSeries::from(rhs);

        Self::from(lhs * rhs).trimmed()
    }
}

impl<E> std::ops::MulAssign for CSeries<E>
where
    E: Float,
{
    fn mul_assign(&mut self, other: Self) {
        *self = self.clone() * other;
    }
}
