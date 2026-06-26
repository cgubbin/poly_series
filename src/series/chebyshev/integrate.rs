use crate::{PolynomialIntegrals, PolynomialSeries};

use super::ChebyshevSeries;

use num_traits::{Float, FromPrimitive};
use std::ops::Range;

impl<E> PolynomialIntegrals<E> for ChebyshevSeries<E>
where
    E: Float + FromPrimitive,
{
    fn antiderivative(&self) -> Self {
        self.antiderivative_with_constant(E::zero())
    }

    fn antiderivative_with_constant(&self, constant: E) -> Self {
        let mut coefficients = integrate_chebyshev_coefficients(self.coefficients.as_slice());

        // Convert from integral with respect to scaled coordinate `t` to
        // integral with respect to physical coordinate `x`.
        //
        // t = 2(x - a)/(b - a) - 1
        // dx/dt = (b - a)/2
        let scale = (self.domain.end - self.domain.start)
            / E::from_f64(2.0).expect("2.0 should be representable");

        for coefficient in &mut coefficients {
            *coefficient = *coefficient * scale;
        }

        coefficients[0] = coefficients[0] + constant;

        Self::new(coefficients, self.domain.clone()).unwrap()
    }

    fn definite_integral(&self, interval: Range<E>) -> E {
        let antiderivative = self.antiderivative();

        antiderivative.evaluate(interval.end) - antiderivative.evaluate(interval.start)
    }
}

// fn integrate_chebyshev_coefficients<E>(coefficients: &[E]) -> Vec<E>
// where
//     E: Float + FromPrimitive,
// {
//     if coefficients.is_empty() {
//         return vec![E::zero()];
//     }

//     let mut integral = vec![E::zero(); coefficients.len() + 1];

//     for (n, &coefficient) in coefficients.iter().enumerate() {
//         match n {
//             0 => {
//                 // ∫ T0(t) dt = T1(t)
//                 integral[1] = integral[1] + coefficient;
//             }
//             1 => {
//                 // ∫ T1(t) dt = T2(t) / 4
//                 integral[2] = integral[2]
//                     + coefficient / E::from_f64(4.0).expect("4.0 should be representable");
//             }
//             _ => {
//                 // ∫ Tn(t) dt =
//                 //     T_{n+1}(t) / (2(n + 1))
//                 //   - T_{n-1}(t) / (2(n - 1))
//                 let two = E::from_f64(2.0).expect("2.0 should be representable");

//                 let lower = two * E::from_usize(n - 1).expect("usize should be representable");

//                 let upper = two * E::from_usize(n + 1).expect("usize should be representable");

//                 integral[n - 1] = integral[n - 1] - coefficient / lower;
//                 integral[n + 1] = integral[n + 1] + coefficient / upper;
//             }
//         }
//     }

//     integral
// }
fn integrate_chebyshev_coefficients<E>(coefficients: &[E]) -> Vec<E>
where
    E: Float + FromPrimitive,
{
    if coefficients.is_empty() {
        return vec![E::zero()];
    }

    let degree = coefficients.len() - 1;
    let mut integral = vec![E::zero(); coefficients.len() + 1];

    if degree == 0 {
        integral[1] = coefficients[0];
        return integral;
    }

    let two = E::from_f64(2.0).expect("2.0 should be representable");
    let four = E::from_f64(4.0).expect("4.0 should be representable");

    // c0 T0 integrates to c0 T1.
    integral[1] = integral[1] + coefficients[0];

    // c1 T1 integrates to c1 T2 / 4.
    integral[2] = integral[2] + coefficients[1] / four;

    for n in 2..=degree {
        let n_minus_one = E::from_usize(n - 1).expect("usize should be representable");
        let n_plus_one = E::from_usize(n + 1).expect("usize should be representable");

        integral[n - 1] = integral[n - 1] - coefficients[n] / (two * n_minus_one);
        integral[n + 1] = integral[n + 1] + coefficients[n] / (two * n_plus_one);
    }

    integral
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PolynomialIntegrals, PolynomialSeries};

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
    fn antiderivative_of_zero_is_zero() {
        let p = ChebyshevSeries::<f64>::zero(-1.0..1.0);
        let integral = p.antiderivative();

        assert!(integral.is_zero());
        assert_eq!(integral.degree(), 0);
    }

    #[test]
    fn antiderivative_of_constant_on_canonical_domain_is_linear() {
        let p = series(vec![2.0], -1.0..1.0);
        let integral = p.antiderivative();

        assert_eq!(integral.degree(), 1);

        for x in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            assert_close(integral.evaluate(x), 2.0 * x);
        }
    }

    #[test]
    fn antiderivative_with_constant_adds_constant_term() {
        let p = series(vec![2.0], -1.0..1.0);
        let integral = p.antiderivative_with_constant(5.0);

        for x in [-1.0, 0.0, 1.0] {
            assert_close(integral.evaluate(x), 2.0 * x + 5.0);
        }
    }

    #[test]
    fn antiderivative_of_t1_round_trips_through_derivative() {
        let p = series(vec![0.0, 1.0], -1.0..1.0);

        let recovered = p.antiderivative().first_derivative();

        for x in [-1.0, -0.75, -0.25, 0.0, 0.25, 0.75, 1.0] {
            assert_close(recovered.evaluate(x), p.evaluate(x));
        }
    }

    #[test]
    fn antiderivative_of_t2_round_trips_through_derivative() {
        let p = series(vec![0.0, 0.0, 1.0], -1.0..1.0);

        let recovered = p.antiderivative().first_derivative();

        for x in [-1.0, -0.75, -0.25, 0.0, 0.25, 0.75, 1.0] {
            assert_close(recovered.evaluate(x), p.evaluate(x));
        }
    }

    #[test]
    fn antiderivative_of_general_series_round_trips_on_canonical_domain() {
        let p = series(vec![1.0, 2.0, 3.0, 4.0], -1.0..1.0);

        let recovered = p.antiderivative().first_derivative();

        for x in [-1.0, -0.75, -0.25, 0.0, 0.25, 0.75, 1.0] {
            assert_close(recovered.evaluate(x), p.evaluate(x));
        }
    }

    #[test]
    fn antiderivative_of_general_series_round_trips_on_physical_domain() {
        let p = series(vec![1.0, 2.0, 3.0, 4.0], 10.0..20.0);

        let recovered = p.antiderivative().first_derivative();

        for x in [10.0, 11.25, 13.75, 15.0, 17.5, 20.0] {
            assert_close(recovered.evaluate(x), p.evaluate(x));
        }
    }

    #[test]
    fn antiderivative_derivative_round_trip_on_canonical_domain() {
        let p = series(vec![1.0, 2.0, 3.0, 4.0], -1.0..1.0);
        let integral = p.antiderivative();
        let recovered = integral.first_derivative();

        for x in [-1.0, -0.75, -0.25, 0.0, 0.5, 1.0] {
            assert_close(recovered.evaluate(x), p.evaluate(x));
        }
    }

    #[test]
    fn antiderivative_derivative_round_trip_on_physical_domain() {
        let p = series(vec![1.0, 2.0, 3.0, 4.0], 10.0..20.0);
        let integral = p.antiderivative();
        let recovered = integral.first_derivative();

        for x in [10.0, 12.5, 15.0, 17.5, 20.0] {
            assert_close(recovered.evaluate(x), p.evaluate(x));
        }
    }

    #[test]
    fn antiderivative_respects_physical_domain_scaling_for_constant() {
        let p = series(vec![2.0], 10.0..20.0);
        let integral = p.antiderivative();

        // Integral with respect to x of constant 2 is 2x + C.
        // With zero Chebyshev integration constant this antiderivative is
        // represented as 10*T1(t), equivalent to 2x - 30 on [10, 20].
        assert_close(integral.evaluate(10.0), -10.0);
        assert_close(integral.evaluate(15.0), 0.0);
        assert_close(integral.evaluate(20.0), 10.0);

        let derivative = integral.first_derivative();

        for x in [10.0, 15.0, 20.0] {
            assert_close(derivative.evaluate(x), 2.0);
        }
    }

    #[test]
    fn definite_integral_of_constant_over_canonical_domain() {
        let p = series(vec![2.0], -1.0..1.0);

        assert_close(p.definite_integral(-1.0..1.0), 4.0);
    }

    #[test]
    fn definite_integral_of_constant_over_physical_domain() {
        let p = series(vec![2.0], 10.0..20.0);

        assert_close(p.definite_integral(10.0..20.0), 20.0);
    }

    #[test]
    fn definite_integral_of_t1_over_symmetric_domain_is_zero() {
        let p = series(vec![0.0, 1.0], -1.0..1.0);

        assert_close(p.definite_integral(-1.0..1.0), 0.0);
    }

    #[test]
    fn definite_integral_of_t2_over_canonical_domain() {
        let p = series(vec![0.0, 0.0, 1.0], -1.0..1.0);

        // ∫_{-1}^{1} (2x² - 1) dx = -2/3
        assert_close(p.definite_integral(-1.0..1.0), -2.0 / 3.0);
    }

    #[test]
    fn definite_integral_over_subinterval() {
        let p = series(vec![2.0], 10.0..20.0);

        assert_close(p.definite_integral(12.0..18.0), 12.0);
    }

    #[test]
    fn integral_over_domain_uses_full_domain() {
        let p = series(vec![2.0], 10.0..20.0);

        assert_close(p.integral_over_domain(), 20.0);
    }

    #[test]
    fn definite_integral_is_independent_of_antiderivative_constant() {
        let p = series(vec![1.0, 2.0, 3.0], -1.0..1.0);

        let f0 = p.antiderivative();
        let f1 = p.antiderivative_with_constant(100.0);

        let direct = p.definite_integral(-0.5..0.75);
        let via_f0 = f0.evaluate(0.75) - f0.evaluate(-0.5);
        let via_f1 = f1.evaluate(0.75) - f1.evaluate(-0.5);

        assert_close(direct, via_f0);
        assert_close(direct, via_f1);
    }

    #[test]
    fn antiderivative_preserves_domain() {
        let p = series(vec![1.0, 2.0, 3.0], 10.0..20.0);
        let integral = p.antiderivative();

        assert_eq!(integral.domain(), 10.0..20.0);
    }
}
