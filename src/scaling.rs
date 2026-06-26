use num_traits::{Float, FromPrimitive};
use std::ops::Range;

/// Map a physical coordinate `x ∈ [a, b]` to `t ∈ [-1, 1]`.
pub fn to_scaled<E>(x: E, domain: &Range<E>) -> E
where
    E: Float + FromPrimitive,
{
    let two = E::from_f64(2.0).expect("2.0 should be representable");
    let one = E::one();

    two * (x - domain.start) / (domain.end - domain.start) - one
}

/// Map a scaled coordinate `t ∈ [-1, 1]` to `x ∈ [a, b]`.
pub fn from_scaled<E>(t: E, domain: &Range<E>) -> E
where
    E: Float + FromPrimitive,
{
    let two = E::from_f64(2.0).expect("2.0 should be representable");

    domain.start + (t + E::one()) * (domain.end - domain.start) / two
}

/// Return the physical width of a domain.
pub fn domain_width<E>(domain: &Range<E>) -> E
where
    E: Float,
{
    domain.end - domain.start
}

/// Return the midpoint of a domain.
pub fn domain_midpoint<E>(domain: &Range<E>) -> E
where
    E: Float,
{
    (domain.start + domain.end) / (E::one() + E::one())
}

/// Return `true` if the domain is finite and has positive width.
pub fn is_valid_domain<E>(domain: &Range<E>) -> bool
where
    E: Float,
{
    domain.start.is_finite() && domain.end.is_finite() && domain.start < domain.end
}

/// Clamp a value to the physical domain.
pub fn clamp_to_domain<E>(x: E, domain: &Range<E>) -> E
where
    E: Float,
{
    if x < domain.start {
        domain.start
    } else if x > domain.end {
        domain.end
    } else {
        x
    }
}
