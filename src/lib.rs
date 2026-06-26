//! # Polynomial Series
//!
//! A library for constructing and manipulating polynomial series in arbitrary
//! polynomial bases.
//!
//! The crate provides common operations such as
//!
//! - evaluation,
//! - differentiation,
//! - integration,
//! - interpolation,
//! - least-squares fitting,
//! - root finding,
//! - coefficient manipulation,
//! - domain scaling, and
//! - polynomial approximation.
//!
//! The core APIs are expressed as traits, allowing different polynomial bases
//! (Chebyshev, Legendre, Bernstein, power series, and others) to expose a
//! common interface while retaining basis-specific implementations.
//!
//! ## Philosophy
//!
//! This crate separates **polynomial algebra** from **application-specific
//! fitting**.
//!
//! It provides the mathematical operations required to construct and manipulate
//! polynomial series, while higher-level crates can build calibration,
//! uncertainty analysis, optimisation, or statistical workflows on top.
//!
//! ## Implemented Bases
//!
//! Currently implemented:
//!
//! - Chebyshev series
//!
//! Future implementations may include
//!
//! - Legendre series
//! - Power series
//! - Bernstein polynomials
//! - Hermite polynomials
//! - Laguerre polynomials
//!
//! ## Features
//!
//! Basic polynomial operations require no external numerical libraries.
//!
//! Some functionality requires solving linear systems or eigenvalue problems.
//! These are enabled through the optional `linalg` feature together with a
//! selected BLAS/LAPACK backend.
//!
//! ```toml
//! [dependencies]
//! polynomial-series = { version = "...", features = ["linalg-openblas-system"] }
//! ```
//!
//! Available backends include
//!
//! - `linalg-openblas-system`
//! - `linalg-openblas-static`
//! - `linalg-netlib-system`
//! - `linalg-netlib-static`
//! - `linalg-intel-mkl-system`
//! - `linalg-intel-mkl-static`
//!
//! ## Examples
//!
//! ### Constructing a polynomial
//!
//! ```rust
//! use polynomial_series::{
//!     PolynomialSeries,
//!     PolynomialCoefficients,
//!     ChebyshevSeries,
//! };
//!
//! let p = ChebyshevSeries::from_coefficients(
//!     -1.0..1.0,
//!     vec![1.0, 2.0, 3.0],
//! ).unwrap();
//!
//! let y = p.evaluate(0.25);
//! ```
//!
//! ### Differentiation
//!
//! ```rust
//! # use polynomial_series::{
//! #     PolynomialSeries,
//! #     PolynomialCoefficients,
//! #     ChebyshevSeries,
//! # };
//! # let p = ChebyshevSeries::from_coefficients(
//! #     -1.0..1.0,
//! #     vec![1.0,2.0,3.0],
//! # ).unwrap();
//! let dp = p.first_derivative();
//! let d2p = p.derivative(2);
//! ```
//!
//! ### Integration
//!
//! ```rust
//! # use polynomial_series::{
//! #     PolynomialSeries,
//! #     PolynomialIntegrals,
//! #     PolynomialCoefficients,
//! #     ChebyshevSeries,
//! # };
//! # let p = ChebyshevSeries::from_coefficients(
//! #     -1.0..1.0,
//! #     vec![1.0,2.0],
//! # ).unwrap();
//! let area = p.integral_over_domain();
//! ```
//!
//! ### Least-squares fitting
//!
//! ```rust,no_run
//! use polynomial_series::{
//!     FitPolynomialSeries,
//!     ChebyshevSeries,
//! };
//!
//! let xs = [0.0, 1.0, 2.0, 3.0];
//! let ys = [1.0, 2.0, 5.0, 10.0];
//!
//! let report = ChebyshevSeries::fit_report(&xs, &ys, 2).unwrap();
//!
//! println!("RSS = {}", report.residual_sum_of_squares);
//! ```
//!
//! ### Root finding
//!
//! ```rust,no_run
//! use polynomial_series::{
//!     PolynomialRoots,
//!     PolynomialCoefficients,
//!     ChebyshevSeries,
//! };
//!
//! let p = ChebyshevSeries::from_coefficients(
//!     -1.0..1.0,
//!     vec![0.0, 0.0, 1.0],
//! ).unwrap();
//!
//! let roots = p.roots().unwrap();
//! ```
//!
//! ## Design
//!
//! The crate is organised around small orthogonal traits:
//!
//! | Trait | Purpose |
//! |-------|---------|
//! | `PolynomialSeries` | Core polynomial operations |
//! | `PolynomialCoefficients` | Access to basis coefficients |
//! | `PolynomialDomain` | Physical and scaled domains |
//! | `PolynomialApproximation` | Truncation and coefficient pruning |
//! | `PolynomialIntegrals` | Antiderivatives and definite integrals |
//! | `PolynomialRoots` | Root finding |
//! | `InterpolatePolynomialSeries` | Exact interpolation |
//! | `FitPolynomialSeries` | Least-squares fitting |
//!
//! Individual polynomial bases implement whichever traits are mathematically
//! meaningful and computationally supported.

#![allow(dead_code)]

mod error;
mod scaling;
mod series;
mod traits;

pub use series::ChebyshevSeries;
pub use traits::{
    ConvertPolynomialBasis, PolynomialApproximation, PolynomialCoefficients, PolynomialDomain,
    PolynomialEvaluateMany, PolynomialIntegrals, PolynomialSeries,
};

#[cfg(feature = "linalg")]
pub use traits::{FitPolynomialSeries, FitReport, InterpolatePolynomialSeries, PolynomialRoots};
