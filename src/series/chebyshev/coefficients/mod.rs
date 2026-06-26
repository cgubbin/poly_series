mod c_series;
mod z_series;

pub(crate) use c_series::CSeries;
pub(crate) use z_series::ZSeries;

#[cfg(test)]
mod tests {
    use super::*;

    fn c(values: &[f64]) -> CSeries<f64> {
        CSeries::from(values.to_vec())
    }

    fn assert_coefficients_eq(series: &CSeries<f64>, expected: &[f64]) {
        assert_eq!(series.as_slice(), expected);
    }

    #[test]
    fn zero_is_represented_by_single_zero_coefficient() {
        let series = CSeries::<f64>::zero();

        assert_coefficients_eq(&series, &[0.0]);
        assert_eq!(series.len(), 1);
        assert_eq!(series.degree(), 0);
        assert!(series.is_zero());
    }

    #[test]
    fn degree_is_len_minus_one_for_non_empty_series() {
        assert_eq!(c(&[1.0]).degree(), 0);
        assert_eq!(c(&[1.0, 2.0]).degree(), 1);
        assert_eq!(c(&[1.0, 2.0, 3.0]).degree(), 2);
    }

    #[test]
    fn empty_series_has_degree_zero_by_convention() {
        let series = CSeries::<f64>::from(vec![]);

        assert_eq!(series.degree(), 0);
    }

    #[test]
    fn trimmed_removes_trailing_zeroes() {
        let series = c(&[1.0, 2.0, 0.0, 0.0]).trimmed();

        assert_coefficients_eq(&series, &[1.0, 2.0]);
    }

    #[test]
    fn trimmed_keeps_non_trailing_zeroes() {
        let series = c(&[1.0, 0.0, 3.0, 0.0]).trimmed();

        assert_coefficients_eq(&series, &[1.0, 0.0, 3.0]);
    }

    #[test]
    fn trimmed_all_zero_series_returns_canonical_zero() {
        let series = c(&[0.0, 0.0, 0.0]).trimmed();

        assert_coefficients_eq(&series, &[0.0]);
        assert!(series.is_zero());
    }

    #[test]
    fn trimmed_empty_series_returns_canonical_zero() {
        let series = CSeries::<f64>::from(vec![]).trimmed();

        assert_coefficients_eq(&series, &[0.0]);
    }

    #[test]
    fn scale_mut_scales_all_coefficients() {
        let mut series = c(&[1.0, -2.0, 3.0]);

        series.scale_mut(2.0);

        assert_coefficients_eq(&series, &[2.0, -4.0, 6.0]);
    }

    #[test]
    fn addition_with_equal_lengths_adds_coefficients() {
        let result = c(&[1.0, 2.0, 3.0]) + c(&[4.0, 5.0, 6.0]);

        assert_coefficients_eq(&result, &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn addition_pads_shorter_rhs() {
        let result = c(&[1.0, 2.0, 3.0]) + c(&[4.0]);

        assert_coefficients_eq(&result, &[5.0, 2.0, 3.0]);
    }

    #[test]
    fn addition_pads_shorter_lhs() {
        let result = c(&[4.0]) + c(&[1.0, 2.0, 3.0]);

        assert_coefficients_eq(&result, &[5.0, 2.0, 3.0]);
    }

    #[test]
    fn addition_trims_trailing_zeroes() {
        let result = c(&[1.0, 2.0, 3.0]) + c(&[0.0, 0.0, -3.0]);

        assert_coefficients_eq(&result, &[1.0, 2.0]);
    }

    #[test]
    fn addition_of_opposites_returns_canonical_zero() {
        let result = c(&[1.0, 2.0, 3.0]) + c(&[-1.0, -2.0, -3.0]);

        assert_coefficients_eq(&result, &[0.0]);
        assert!(result.is_zero());
    }

    #[test]
    fn subtraction_with_equal_lengths_subtracts_coefficients() {
        let result = c(&[5.0, 7.0, 9.0]) - c(&[1.0, 2.0, 3.0]);

        assert_coefficients_eq(&result, &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn subtraction_pads_shorter_rhs() {
        let result = c(&[1.0, 2.0, 3.0]) - c(&[4.0]);

        assert_coefficients_eq(&result, &[-3.0, 2.0, 3.0]);
    }

    #[test]
    fn subtraction_pads_shorter_lhs() {
        let result = c(&[4.0]) - c(&[1.0, 2.0, 3.0]);

        assert_coefficients_eq(&result, &[3.0, -2.0, -3.0]);
    }

    #[test]
    fn subtraction_trims_trailing_zeroes() {
        let result = c(&[1.0, 2.0, 3.0]) - c(&[0.0, 0.0, 3.0]);

        assert_coefficients_eq(&result, &[1.0, 2.0]);
    }

    #[test]
    fn subtraction_from_self_returns_canonical_zero() {
        let result = c(&[1.0, 2.0, 3.0]) - c(&[1.0, 2.0, 3.0]);

        assert_coefficients_eq(&result, &[0.0]);
        assert!(result.is_zero());
    }

    #[test]
    fn cseries_to_zseries_constant() {
        let z = ZSeries::from(c(&[4.0]));

        assert_eq!(z.into_vec(), vec![4.0]);
    }

    #[test]
    fn cseries_to_zseries_linear() {
        // c0 + c1 T1(t)
        //
        // T1 = (z + z^-1) / 2
        //
        // [2, 4] maps to [2, 2, 2] in powers z^-1, z^0, z^1.
        let z = ZSeries::from(c(&[2.0, 4.0]));

        assert_eq!(z.into_vec(), vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn zseries_round_trip_constant() {
        let original = c(&[3.0]);
        let round_trip = CSeries::from(ZSeries::from(original.clone()));

        assert_eq!(round_trip, original);
    }

    #[test]
    fn zseries_round_trip_general_series() {
        let original = c(&[1.0, -2.0, 3.5, 0.25]);
        let round_trip = CSeries::from(ZSeries::from(original.clone()));

        assert_eq!(round_trip, original);
    }

    #[test]
    fn multiplication_by_zero_returns_zero() {
        let result = c(&[1.0, 2.0, 3.0]) * CSeries::zero();

        assert_coefficients_eq(&result, &[0.0]);
    }

    #[test]
    fn multiplication_by_one_returns_original() {
        let result = c(&[1.0, -2.0, 3.0]) * c(&[1.0]);

        assert_coefficients_eq(&result, &[1.0, -2.0, 3.0]);
    }

    #[test]
    fn multiplication_of_constants_multiplies_constants() {
        let result = c(&[2.0]) * c(&[3.0]);

        assert_coefficients_eq(&result, &[6.0]);
    }

    #[test]
    fn multiplication_t1_by_t1_uses_chebyshev_identity() {
        // T1 * T1 = x^2 = (T2 + T0) / 2
        let result = c(&[0.0, 1.0]) * c(&[0.0, 1.0]);

        assert_coefficients_eq(&result, &[0.5, 0.0, 0.5]);
    }

    #[test]
    fn multiplication_t1_by_t2_uses_chebyshev_product_identity() {
        // T1 * T2 = (T3 + T1) / 2
        let result = c(&[0.0, 1.0]) * c(&[0.0, 0.0, 1.0]);

        assert_coefficients_eq(&result, &[0.0, 0.5, 0.0, 0.5]);
    }

    #[test]
    fn multiplication_t2_by_t2_uses_chebyshev_product_identity() {
        // T2 * T2 = (T4 + T0) / 2
        let result = c(&[0.0, 0.0, 1.0]) * c(&[0.0, 0.0, 1.0]);

        assert_coefficients_eq(&result, &[0.5, 0.0, 0.0, 0.0, 0.5]);
    }

    #[test]
    fn multiplication_general_case_matches_known_expansion() {
        // (1 + 2T1 + 3T2) * (4 + 5T1)
        //
        // Product identities:
        // T1*T1 = 0.5T0 + 0.5T2
        // T2*T1 = 0.5T1 + 0.5T3
        //
        // = 4
        // + 5T1
        // + 8T1
        // + 10T1*T1
        // + 12T2
        // + 15T2*T1
        //
        // = 9T0 + 20.5T1 + 17T2 + 7.5T3
        let result = c(&[1.0, 2.0, 3.0]) * c(&[4.0, 5.0]);

        assert_coefficients_eq(&result, &[9.0, 20.5, 17.0, 7.5]);
    }

    #[test]
    fn multiplication_trims_trailing_zeroes() {
        let result = c(&[0.0, 1.0]) * c(&[0.0]);

        assert_coefficients_eq(&result, &[0.0]);
    }

    #[test]
    fn multiplication_is_commutative_for_test_case() {
        let lhs = c(&[1.0, 2.0, 3.0]);
        let rhs = c(&[4.0, 5.0]);

        assert_eq!(lhs.clone() * rhs.clone(), rhs * lhs);
    }

    #[test]
    fn multiplication_degree_is_sum_of_degrees_when_leading_terms_do_not_cancel() {
        let lhs = c(&[1.0, 2.0, 3.0]);
        let rhs = c(&[4.0, 5.0]);

        let result = lhs * rhs;

        assert_eq!(result.degree(), 3);
    }
}
