use std::ops::Mul;

use nalgebra::{Rotation3, Unit, Vector3};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    X,
    Y,
    Z,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sign {
    Positive,
    Negative,
}

impl Mul for Sign {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Sign::Positive, Sign::Positive) => Sign::Positive,
            (Sign::Positive, Sign::Negative) => Sign::Negative,
            (Sign::Negative, Sign::Positive) => Sign::Negative,
            (Sign::Negative, Sign::Negative) => Sign::Positive,
        }
    }
}

impl Axis {
    pub fn to_unit(self) -> Unit<Vector3<f32>> {
        match self {
            Axis::X => Vector3::x_axis(),
            Axis::Y => Vector3::y_axis(),
            Axis::Z => Vector3::z_axis(),
        }
    }
}

// these fields can be public since they have no invariants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BasisVector {
    pub axis: Axis,
    pub sign: Sign,
}

impl BasisVector {
    pub fn to_vector3(self) -> Vector3<f32> {
        let v = self.axis.to_unit().into_inner();

        match self.sign {
            Sign::Positive => v,
            Sign::Negative => -v,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuarterTurn {
    Zero,
    Quarter,
    Half,
    ThreeQuarters,
}

const PX: BasisVector = BasisVector {
    axis: Axis::X,
    sign: Sign::Positive,
};
const PY: BasisVector = BasisVector {
    axis: Axis::Y,
    sign: Sign::Positive,
};
const PZ: BasisVector = BasisVector {
    axis: Axis::Z,
    sign: Sign::Positive,
};
const MX: BasisVector = BasisVector {
    axis: Axis::X,
    sign: Sign::Negative,
};
const MY: BasisVector = BasisVector {
    axis: Axis::Y,
    sign: Sign::Negative,
};
const MZ: BasisVector = BasisVector {
    axis: Axis::Z,
    sign: Sign::Negative,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CubeRotation {
    basis: [BasisVector; 3],
}

impl Default for CubeRotation {
    fn default() -> Self {
        Self {
            basis: [PX, PY, PZ],
        }
    }
}

impl CubeRotation {
    pub fn from_axis_turn(axis: Axis, turn: QuarterTurn) -> Self {
        match (axis, turn) {
            (_, QuarterTurn::Zero) => Self {
                basis: [PX, PY, PZ],
            },
            (Axis::X, QuarterTurn::Quarter) => Self {
                basis: [PX, PZ, MY],
            },
            (Axis::X, QuarterTurn::Half) => Self {
                basis: [PX, MY, MZ],
            },
            (Axis::X, QuarterTurn::ThreeQuarters) => Self {
                basis: [PX, MZ, PY],
            },
            (Axis::Y, QuarterTurn::Quarter) => Self {
                basis: [MZ, PY, PX],
            },
            (Axis::Y, QuarterTurn::Half) => Self {
                basis: [MX, PY, MZ],
            },
            (Axis::Y, QuarterTurn::ThreeQuarters) => Self {
                basis: [PZ, PY, MX],
            },
            (Axis::Z, QuarterTurn::Quarter) => Self {
                basis: [PY, MX, PZ],
            },
            (Axis::Z, QuarterTurn::Half) => Self {
                basis: [MX, MY, PZ],
            },
            (Axis::Z, QuarterTurn::ThreeQuarters) => Self {
                basis: [MY, PX, PZ],
            },
        }
    }

    pub fn as_basis(&self) -> &[BasisVector; 3] {
        &self.basis
    }

    pub fn to_rotation3(self) -> Rotation3<f32> {
        Rotation3::from_basis_unchecked(&self.basis.map(|b| b.to_vector3()))
    }
}

impl Mul for CubeRotation {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let basis = rhs.basis.map(|right| {
            let left = self.basis[right.axis as usize];

            BasisVector {
                axis: left.axis,
                sign: left.sign * right.sign,
            }
        });

        Self { basis }
    }
}
