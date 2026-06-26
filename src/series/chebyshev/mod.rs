//! Chebyshev polynomial series.
//!
//! This module implements polynomials represented in the first-kind Chebyshev
//! basis,
//!
//! ```text
//! p(t) = Σ cₙ Tₙ(t),
//! ```
//!
//! where `Tₙ(t)` is the Chebyshev polynomial of the first kind and the
//! coefficients `cₙ` are stored in ascending order of polynomial degree.
//!
//! ## Why Chebyshev polynomials?
//!
//! Chebyshev series are often preferred over the monomial basis for numerical
//! computation because they provide:
//!
//! - excellent numerical stability for high-order polynomials,
//! - near-minimax polynomial approximations,
//! - efficient evaluation using Clenshaw's algorithm,
//! - stable differentiation and integration,
//! - efficient multiplication through z-series convolution.
//!
//! Consequently, Chebyshev representations are widely used in approximation
//! theory, numerical analysis, calibration, and scientific computing.
//!
//! ## Domains
//!
//! Chebyshev polynomials are naturally defined on the canonical interval
//! `[-1, 1]`.
//!
//! Each [`ChebyshevSeries`] stores the physical domain over which it is defined.
//! Evaluation on physical coordinates automatically rescales inputs between the
//! physical domain and the canonical interval.
//!
//! ## Internal representation
//!
//! Two coefficient representations are used internally:
//!
//! - **C-series** store the ordinary Chebyshev coefficients.
//! - **Z-series** provide a symmetric Laurent-series representation that allows
//!   multiplication to be performed efficiently as a convolution.
//!
//! Users normally interact only with [`ChebyshevSeries`]. The coefficient types
//! are implementation details.
//!
//! ## Features
//!
//! Depending on the enabled crate features, Chebyshev series support:
//!
//! - evaluation,
//! - differentiation,
//! - integration,
//! - arithmetic operations,
//! - root finding,
//! - polynomial fitting,
//! - interpolation.
//!
//! Root finding and fitting require the optional `linalg` feature.
//!
//! # Example
//!
//! ```
//! use polynomial_series::{ChebyshevSeries, PolynomialSeries};
//!
//! let p = ChebyshevSeries::new(
//!     vec![1.0, 2.0, 3.0],
//!     -1.0..1.0,
//! ).unwrap();
//!
//! let y = p.evaluate(0.25);
//!
//! let dp = p.first_derivative();
//!
//! # let _ = (y, dp);
//! ```

mod coeff;
mod coefficients;
mod domain;
mod error;
mod integrate;
mod ops;
mod truncate;

#[cfg(feature = "linalg")]
mod fitting;

#[cfg(feature = "linalg")]
mod interpolate;

#[cfg(feature = "linalg")]
mod roots;

use crate::{
    PolynomialSeries,
    scaling::{is_valid_domain, to_scaled},
};
use coefficients::CSeries;
use error::ChebyshevError;

use num_traits::{Float, FromPrimitive};

use std::ops::Range;

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ChebyshevSeries<E> {
    coefficients: CSeries<E>,
    domain: Range<E>,
}

impl<E> ChebyshevSeries<E> {
    pub fn new<C>(coefficients: C, domain: Range<E>) -> Result<Self, ChebyshevError>
    where
        C: Into<CSeries<E>>,
        E: Float,
    {
        if !is_valid_domain(&domain) {
            return Err(ChebyshevError::InvalidDomain);
        }

        let coefficients = coefficients.into().trimmed();

        if coefficients.iter().any(|c| !c.is_finite()) {
            return Err(ChebyshevError::InvalidCoefficients);
        }

        Ok(Self {
            coefficients,
            domain,
        })
    }

    pub fn coefficients(&self) -> &[E] {
        self.coefficients.as_slice()
    }

    pub fn into_coefficients(self) -> Vec<E> {
        self.coefficients.into_vec()
    }
    fn first_derivative_scaled(&self) -> Self
    where
        E: Float + FromPrimitive,
    {
        let degree = self.degree();

        if degree == 0 {
            return Self::zero(self.domain());
        }

        let c = self.coefficients.as_slice();
        let mut d = vec![E::zero(); degree];

        let two = E::from_f64(2.0).expect("2.0 should be representable");

        // Highest derivative coefficient.
        d[degree - 1] =
            two * E::from_usize(degree).expect("usize should be representable") * c[degree];

        // Second-highest derivative coefficient.
        if degree >= 2 {
            d[degree - 2] = two
                * E::from_usize(degree - 1).expect("usize should be representable")
                * c[degree - 1];
        }

        // Remaining interior coefficients.
        if degree >= 3 {
            for k in (1..=degree - 3).rev() {
                d[k] = d[k + 2]
                    + two * E::from_usize(k + 1).expect("usize should be representable") * c[k + 1];
            }
        }

        // Constant coefficient.
        d[0] = if degree == 1 {
            c[1]
        } else if degree == 2 {
            c[1]
        } else {
            c[1] + d[2] / two
        };

        Self {
            coefficients: CSeries::from(d).trimmed(),
            domain: self.domain(),
        }
    }

    fn first_derivative_physical(&self) -> Self
    where
        E: Float + FromPrimitive,
    {
        let mut derivative = self.first_derivative_scaled();

        let scale = E::from_f64(2.0).expect("2.0 should convert to scalar")
            / (self.domain.end - self.domain.start);

        derivative.coefficients.scale_mut(scale);

        derivative
    }
}

impl<E> PolynomialSeries<E> for ChebyshevSeries<E>
where
    E: Float + FromPrimitive,
{
    fn evaluate_scaled(&self, t: E) -> E {
        let mut coefficients = self.coefficients.iter();
        let (c0, c1) = if self.degree() < 2 {
            (
                coefficients.next().copied().unwrap_or_else(E::zero),
                coefficients.next().copied().unwrap_or_else(E::zero),
            )
        } else {
            let t2 = t + t;
            let mut coefficients = coefficients.rev();
            let mut c1 = coefficients.next().copied().unwrap();
            let mut c0 = coefficients.next().copied().unwrap();
            for cnext in coefficients {
                let tmp = c0;
                c0 = *cnext - c1;
                c1 = tmp + c1 * t2;
            }
            (c0, c1)
        };
        c0 + t * c1
    }

    fn evaluate(&self, x: E) -> E {
        self.evaluate_scaled(to_scaled(x, &self.domain))
    }

    fn first_derivative(&self) -> Self {
        self.first_derivative_physical()
    }

    fn degree(&self) -> usize {
        self.coefficients.degree()
    }

    fn domain(&self) -> Range<E> {
        self.domain.clone()
    }

    fn zero(domain: Range<E>) -> Self {
        Self {
            coefficients: CSeries::zero(),
            domain,
        }
    }

    fn is_zero(&self) -> bool {
        self.coefficients.is_zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::PolynomialSeries;

    const EPS: f64 = 1.0e-12;

    fn assert_close(lhs: f64, rhs: f64) {
        assert!(
            (lhs - rhs).abs() <= EPS,
            "expected {lhs} ≈ {rhs}, difference = {}",
            (lhs - rhs).abs()
        );
    }

    fn series(coefficients: Vec<f64>, domain: Range<f64>) -> ChebyshevSeries<f64> {
        ChebyshevSeries::new(coefficients, domain).unwrap()
    }

    #[test]
    fn zero_series_has_degree_zero_and_evaluates_to_zero() {
        let p = ChebyshevSeries::<f64>::zero(-1.0..1.0);

        assert_eq!(p.degree(), 0);
        assert!(p.is_zero());
        assert_close(p.evaluate_scaled(-1.0), 0.0);
        assert_close(p.evaluate_scaled(0.0), 0.0);
        assert_close(p.evaluate_scaled(1.0), 0.0);
    }

    #[test]
    fn constant_series_evaluates_to_constant() {
        let p = series(vec![3.5], -1.0..1.0);

        assert_eq!(p.degree(), 0);
        assert_close(p.evaluate_scaled(-1.0), 3.5);
        assert_close(p.evaluate_scaled(0.0), 3.5);
        assert_close(p.evaluate_scaled(1.0), 3.5);
    }

    #[test]
    fn t1_series_evaluates_to_t() {
        let p = series(vec![0.0, 1.0], -1.0..1.0);

        assert_close(p.evaluate_scaled(-1.0), -1.0);
        assert_close(p.evaluate_scaled(-0.25), -0.25);
        assert_close(p.evaluate_scaled(0.0), 0.0);
        assert_close(p.evaluate_scaled(0.75), 0.75);
        assert_close(p.evaluate_scaled(1.0), 1.0);
    }

    #[test]
    fn t2_series_evaluates_to_two_t_squared_minus_one() {
        let p = series(vec![0.0, 0.0, 1.0], -1.0..1.0);

        for t in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            assert_close(p.evaluate_scaled(t), 2.0 * t * t - 1.0);
        }
    }

    #[test]
    fn general_series_evaluates_known_chebyshev_expansion() {
        // p(t) = 1 + 2T1(t) + 3T2(t)
        //      = 1 + 2t + 3(2t^2 - 1)
        //      = 6t^2 + 2t - 2
        let p = series(vec![1.0, 2.0, 3.0], -1.0..1.0);

        for t in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            let expected = 6.0 * t * t + 2.0 * t - 2.0;
            assert_close(p.evaluate_scaled(t), expected);
        }
    }

    #[test]
    fn evaluate_maps_physical_domain_to_scaled_domain() {
        let p = series(vec![0.0, 1.0], 10.0..20.0);

        assert_close(p.evaluate(10.0), -1.0);
        assert_close(p.evaluate(15.0), 0.0);
        assert_close(p.evaluate(20.0), 1.0);
    }

    #[test]
    fn derivative_of_constant_is_zero() {
        let p = series(vec![3.0], -1.0..1.0);
        let dp = p.first_derivative();

        assert_eq!(dp.degree(), 0);
        assert!(dp.is_zero());
        assert_close(dp.evaluate(0.0), 0.0);
    }

    #[test]
    fn derivative_of_t1_on_canonical_domain_is_one() {
        let p = series(vec![0.0, 1.0], -1.0..1.0);
        let dp = p.first_derivative();

        assert_eq!(dp.degree(), 0);
        assert_close(dp.evaluate(-1.0), 1.0);
        assert_close(dp.evaluate(0.0), 1.0);
        assert_close(dp.evaluate(1.0), 1.0);
    }

    #[test]
    fn derivative_of_t2_on_canonical_domain_is_four_t() {
        let p = series(vec![0.0, 0.0, 1.0], -1.0..1.0);
        let dp = p.first_derivative();

        // d/dt T2(t) = d/dt (2t^2 - 1) = 4t
        assert_eq!(dp.degree(), 1);

        for x in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            assert_close(dp.evaluate(x), 4.0 * x);
        }
    }

    #[test]
    fn derivative_is_with_respect_to_physical_coordinate() {
        // p(t) = T1(t) = t
        //
        // domain x ∈ [10, 20]
        // t = 2(x - 10)/10 - 1
        // t = x/5 - 3
        //
        // dp/dx = 1/5 = 0.2
        let p = series(vec![0.0, 1.0], 10.0..20.0);
        let dp = p.first_derivative();

        assert_eq!(dp.degree(), 0);
        assert_close(dp.evaluate(10.0), 0.2);
        assert_close(dp.evaluate(15.0), 0.2);
        assert_close(dp.evaluate(20.0), 0.2);
    }

    #[test]
    fn derivative_of_t2_respects_physical_domain_scaling() {
        // p(t) = T2(t) = 2t^2 - 1
        //
        // x ∈ [10, 20]
        // t = x/5 - 3
        // dp/dx = (dp/dt)(dt/dx) = 4t * 0.2 = 0.8t
        let p = series(vec![0.0, 0.0, 1.0], 10.0..20.0);
        let dp = p.first_derivative();

        assert_eq!(dp.degree(), 1);

        for x in [10.0, 12.5, 15.0, 17.5, 20.0] {
            let t = (x / 5.0) - 3.0;
            assert_close(dp.evaluate(x), 0.8 * t);
        }
    }

    #[test]
    fn second_derivative_uses_repeated_physical_derivatives() {
        // p(t) = T2(t) = 2t^2 - 1
        //
        // On x ∈ [10, 20], t = x/5 - 3.
        //
        // dp/dx = 0.8t
        // d²p/dx² = 0.8 dt/dx = 0.16
        let p = series(vec![0.0, 0.0, 1.0], 10.0..20.0);
        let d2p = p.derivative(2);

        assert_eq!(d2p.degree(), 0);

        for x in [10.0, 12.5, 15.0, 17.5, 20.0] {
            assert_close(d2p.evaluate(x), 0.16);
        }
    }

    #[test]
    fn derivative_order_zero_returns_same_series() {
        let p = series(vec![1.0, 2.0, 3.0], -1.0..1.0);
        let dp = p.derivative(0);

        assert_eq!(dp.coefficients(), p.coefficients());
        assert_eq!(dp.domain(), p.domain());
    }

    #[test]
    fn derivative_order_greater_than_degree_returns_zero() {
        let p = series(vec![1.0, 2.0], -1.0..1.0);
        let dp = p.derivative(2);

        assert!(dp.is_zero());
        assert_eq!(dp.degree(), 0);
    }

    #[test]
    fn domain_is_preserved_by_derivative() {
        let p = series(vec![1.0, 2.0, 3.0], 10.0..20.0);
        let dp = p.first_derivative();

        assert_eq!(dp.domain(), 10.0..20.0);
    }
}
