use std::ops::{Add, Div, Mul, Neg, Sub};

use num_traits::{Float, FromPrimitive};

use super::ChebyshevSeries;

fn assert_same_domain<E>(lhs: &ChebyshevSeries<E>, rhs: &ChebyshevSeries<E>)
where
    E: Float,
{
    assert!(
        lhs.domain.start == rhs.domain.start && lhs.domain.end == rhs.domain.end,
        "cannot combine Chebyshev series defined on different domains"
    );
}

impl<E> Add for ChebyshevSeries<E>
where
    E: Float + FromPrimitive,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        assert_same_domain(&self, &rhs);

        Self {
            coefficients: self.coefficients + rhs.coefficients,
            domain: self.domain,
        }
    }
}

impl<E> Sub for ChebyshevSeries<E>
where
    E: Float + FromPrimitive,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        assert_same_domain(&self, &rhs);

        Self {
            coefficients: self.coefficients - rhs.coefficients,
            domain: self.domain,
        }
    }
}

impl<E> Mul for ChebyshevSeries<E>
where
    E: Float + FromPrimitive,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        assert_same_domain(&self, &rhs);

        Self {
            coefficients: self.coefficients * rhs.coefficients,
            domain: self.domain,
        }
    }
}

impl<E> Neg for ChebyshevSeries<E>
where
    E: Float + FromPrimitive,
{
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.coefficients.scale_mut(-E::one());
        self
    }
}

impl<E> Mul<E> for ChebyshevSeries<E>
where
    E: Float + FromPrimitive,
{
    type Output = Self;

    fn mul(mut self, rhs: E) -> Self::Output {
        self.coefficients.scale_mut(rhs);
        self
    }
}

impl<E> Div<E> for ChebyshevSeries<E>
where
    E: Float + FromPrimitive,
{
    type Output = Self;

    fn div(mut self, rhs: E) -> Self::Output {
        self.coefficients.scale_mut(E::one() / rhs);
        self
    }
}
