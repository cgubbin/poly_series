#[derive(Debug, thiserror::Error)]
pub enum ChebyshevError {
    #[error("domain must be finite and have positive width")]
    InvalidDomain,

    #[error("coefficients must be finite")]
    InvalidCoefficients,

    #[error("interpolation requires at least one point")]
    EmptyInput,

    #[error("x and y data must have equal length")]
    LengthMismatch,

    #[error("weights must have the same length as x and y data")]
    WeightLengthMismatch,

    #[error("interpolation points must be finite")]
    InvalidData,

    #[error("interpolation points must have distinct x values")]
    DuplicateAbscissa,

    #[error("root finding requires a non-zero leading coefficient")]
    DegeneratePolynomial,

    #[cfg(feature = "linalg")]
    #[error("failure in linear algebra operation")]
    Linalg(#[from] ndarray_linalg::error::LinalgError),

    #[cfg(feature = "linalg")]
    #[error("shape error")]
    Shape(#[from] ndarray::ShapeError),

    #[error("weights must be finite and non-negative")]
    InvalidWeights,

    #[error("polynomial degree is too high for the number of observations")]
    UnderdeterminedFit,
}
