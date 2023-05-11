use std::ops::Mul;

use nalgebra::{Rotation3, Vector3};

const S4_LUT: [[usize; 24]; 24] = [
    [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    ],
    [
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 23, 22,
    ],
    [
        2, 4, 0, 5, 1, 3, 8, 10, 6, 11, 7, 9, 14, 16, 12, 17, 13, 15, 20, 22, 18, 23, 19, 21,
    ],
    [
        3, 5, 1, 4, 0, 2, 9, 11, 7, 10, 6, 8, 15, 17, 13, 16, 12, 14, 21, 23, 19, 22, 18, 20,
    ],
    [
        4, 2, 5, 0, 3, 1, 10, 8, 11, 6, 9, 7, 16, 14, 17, 12, 15, 13, 22, 20, 23, 18, 21, 19,
    ],
    [
        5, 3, 4, 1, 2, 0, 11, 9, 10, 7, 8, 6, 17, 15, 16, 13, 14, 12, 23, 21, 22, 19, 20, 18,
    ],
    [
        6, 7, 12, 13, 18, 19, 0, 1, 14, 15, 20, 21, 2, 3, 8, 9, 22, 23, 4, 5, 10, 11, 16, 17,
    ],
    [
        7, 6, 13, 12, 19, 18, 1, 0, 15, 14, 21, 20, 3, 2, 9, 8, 23, 22, 5, 4, 11, 10, 17, 16,
    ],
    [
        8, 10, 14, 16, 20, 22, 2, 4, 12, 17, 18, 23, 0, 5, 6, 11, 19, 21, 1, 3, 7, 9, 13, 15,
    ],
    [
        9, 11, 15, 17, 21, 23, 3, 5, 13, 16, 19, 22, 1, 4, 7, 10, 18, 20, 0, 2, 6, 8, 12, 14,
    ],
    [
        10, 8, 16, 14, 22, 20, 4, 2, 17, 12, 23, 18, 5, 0, 11, 6, 21, 19, 3, 1, 9, 7, 15, 13,
    ],
    [
        11, 9, 17, 15, 23, 21, 5, 3, 16, 13, 22, 19, 4, 1, 10, 7, 20, 18, 2, 0, 8, 6, 14, 12,
    ],
    [
        12, 18, 6, 19, 7, 13, 14, 20, 0, 21, 1, 15, 8, 22, 2, 23, 3, 9, 10, 16, 4, 17, 5, 11,
    ],
    [
        13, 19, 7, 18, 6, 12, 15, 21, 1, 20, 0, 14, 9, 23, 3, 22, 2, 8, 11, 17, 5, 16, 4, 10,
    ],
    [
        14, 20, 8, 22, 10, 16, 12, 18, 2, 23, 4, 17, 6, 19, 0, 21, 5, 11, 7, 13, 1, 15, 3, 9,
    ],
    [
        15, 21, 9, 23, 11, 17, 13, 19, 3, 22, 5, 16, 7, 18, 1, 20, 4, 10, 6, 12, 0, 14, 2, 8,
    ],
    [
        16, 22, 10, 20, 8, 14, 17, 23, 4, 18, 2, 12, 11, 21, 5, 19, 0, 6, 9, 15, 3, 13, 1, 7,
    ],
    [
        17, 23, 11, 21, 9, 15, 16, 22, 5, 19, 3, 13, 10, 20, 4, 18, 1, 7, 8, 14, 2, 12, 0, 6,
    ],
    [
        18, 12, 19, 6, 13, 7, 20, 14, 21, 0, 15, 1, 22, 8, 23, 2, 9, 3, 16, 10, 17, 4, 11, 5,
    ],
    [
        19, 13, 18, 7, 12, 6, 21, 15, 20, 1, 14, 0, 23, 9, 22, 3, 8, 2, 17, 11, 16, 5, 10, 4,
    ],
    [
        20, 14, 22, 8, 16, 10, 18, 12, 23, 2, 17, 4, 19, 6, 21, 0, 11, 5, 13, 7, 15, 1, 9, 3,
    ],
    [
        21, 15, 23, 9, 17, 11, 19, 13, 22, 3, 16, 5, 18, 7, 20, 1, 10, 4, 12, 6, 14, 0, 8, 2,
    ],
    [
        22, 16, 20, 10, 14, 8, 23, 17, 18, 4, 12, 2, 21, 11, 19, 5, 6, 0, 15, 9, 13, 3, 7, 1,
    ],
    [
        23, 17, 21, 11, 15, 9, 22, 16, 19, 5, 13, 3, 20, 10, 18, 4, 7, 1, 14, 8, 12, 2, 6, 0,
    ],
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CubeRotation(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    X,
    Y,
    Z,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Turn {
    None,
    Ccw,
    Half,
    Cw,
}

impl Mul for CubeRotation {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(S4_LUT[self.0][rhs.0])
    }
}

impl CubeRotation {
    pub fn to_rotation3(self) -> Rotation3<f32> {
        fn rotation(a: Vector3<f32>, b: Vector3<f32>, c: Vector3<f32>) -> Rotation3<f32> {
            Rotation3::from_basis_unchecked(&[a, b, c])
        }
        fn x_axis() -> Vector3<f32> {
            Vector3::x_axis().into_inner()
        }
        fn y_axis() -> Vector3<f32> {
            Vector3::y_axis().into_inner()
        }
        fn z_axis() -> Vector3<f32> {
            Vector3::z_axis().into_inner()
        }

        // 0 -> [0, 1, 2, 3]
        // 1 -> [0, 1, 3, 2]
        // 2 -> [0, 2, 1, 3]
        // 3 -> [0, 2, 3, 1]
        // 4 -> [0, 3, 1, 2]
        // 5 -> [0, 3, 2, 1]
        // 6 -> [1, 0, 2, 3]
        // 7 -> [1, 0, 3, 2]
        // 8 -> [1, 2, 0, 3]
        // 9 -> [1, 2, 3, 0]
        // 10 -> [1, 3, 0, 2]
        // 11 -> [1, 3, 2, 0]
        // 12 -> [2, 0, 1, 3]
        // 13 -> [2, 0, 3, 1]
        // 14 -> [2, 1, 0, 3]
        // 15 -> [2, 1, 3, 0]
        // 16 -> [2, 3, 0, 1]
        // 17 -> [2, 3, 1, 0]
        // 18 -> [3, 0, 1, 2]
        // 19 -> [3, 0, 2, 1]
        // 20 -> [3, 1, 0, 2]
        // 21 -> [3, 1, 2, 0]
        // 22 -> [3, 2, 0, 1]
        // 23 -> [3, 2, 1, 0]
        match self.0 {
            0 => rotation(x_axis(), y_axis(), z_axis()),
            1 => rotation(-x_axis(), -z_axis(), -y_axis()),
            2 => rotation(-z_axis(), -y_axis(), -x_axis()),
            3 => rotation(y_axis(), z_axis(), x_axis()),
            4 => rotation(z_axis(), x_axis(), y_axis()),
            5 => rotation(-y_axis(), -x_axis(), -z_axis()),
            6 => rotation(-x_axis(), z_axis(), y_axis()),
            7 => rotation(x_axis(), -y_axis(), -z_axis()),
            8 => rotation(z_axis(), -x_axis(), -y_axis()),
            9 => rotation(-y_axis(), x_axis(), z_axis()),
            10 => rotation(-z_axis(), y_axis(), x_axis()),
            11 => rotation(y_axis(), -z_axis(), -x_axis()),
            12 => rotation(-y_axis(), -z_axis(), x_axis()),
            13 => rotation(z_axis(), y_axis(), -x_axis()),
            14 => rotation(y_axis(), x_axis(), -z_axis()),
            15 => rotation(-z_axis(), -x_axis(), y_axis()),
            16 => rotation(-x_axis(), -y_axis(), z_axis()),
            17 => rotation(x_axis(), z_axis(), -y_axis()),
            18 => rotation(y_axis(), -x_axis(), z_axis()),
            19 => rotation(-z_axis(), x_axis(), -y_axis()),
            20 => rotation(-y_axis(), z_axis(), -x_axis()),
            21 => rotation(z_axis(), -y_axis(), x_axis()),
            22 => rotation(x_axis(), -z_axis(), y_axis()),
            23 => rotation(-x_axis(), y_axis(), -z_axis()),
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    type S4 = [usize; 4];

    use itertools::Itertools;

    fn to_s4(mut k: usize) -> S4 {
        let mut nums = (0..4).collect_vec();
        let order = (1..=4)
            .map(|d| {
                let r = k % d;
                k /= d;
                r
            })
            .collect_vec();

        let mut res = Vec::with_capacity(4);
        for i in order.into_iter().rev() {
            res.push(nums.remove(i));
        }

        res.try_into().unwrap()
    }

    fn from_s4(s: S4) -> usize {
        let mut nums = (0..4).collect_vec();
        let mut k = 0;

        for (d, x) in (1..=4).rev().zip(&s) {
            let i = nums.binary_search(x).unwrap();
            nums.remove(i);
            k = k * d + i;
        }

        k
    }

    fn multiply(a: S4, b: S4) -> S4 {
        let mut res = [0; 4];

        for (x, i) in res.iter_mut().zip(a) {
            *x = b[i];
        }

        res
    }

    #[test]
    fn test() {
        // z-rotation
        let rot = CubeRotation(from_s4([3, 0, 1, 2]));
        let mut x = CubeRotation(from_s4([3, 2, 1, 0]));

        for _ in 0..4 {
            println!("{}", x.0);
            x = rot * x;
        }
    }

    #[test]
    fn check_multiplication() {
        for i in 0..24 {
            for j in 0..24 {
                let a = CubeRotation(i);
                let b = CubeRotation(j);
                let c = a * b;
                println!("{} * {} = {}", i, j, c.0);
                assert_eq!(c.to_rotation3(), a.to_rotation3() * b.to_rotation3());
            }
        }
    }
}