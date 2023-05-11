use itertools::{iproduct, Either, Itertools};

use crate::{
    rotation::{Axis, BasisVector, CubeRotation, QuarterTurn, Sign},
    CUBE_SIZE, N_CUBIES,
};

const N_FACE_CUBIES: usize = CUBE_SIZE.pow(2);

type Layer = [usize; N_FACE_CUBIES];

pub const LAYERS: [Layer; 9] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8],
    [9, 10, 11, 12, 13, 14, 15, 16, 17],
    [18, 19, 20, 21, 22, 23, 24, 25, 26],
    [2, 1, 0, 11, 10, 9, 20, 19, 18],
    [5, 4, 3, 14, 13, 12, 23, 22, 21],
    [8, 7, 6, 17, 16, 15, 26, 25, 24],
    [0, 3, 6, 9, 12, 15, 18, 21, 24],
    [1, 4, 7, 10, 13, 16, 19, 22, 25],
    [2, 5, 8, 11, 14, 17, 20, 23, 26],
];

#[derive(Debug)]
pub struct Cube {
    rotations: [CubeRotation; N_CUBIES],
    idxs: [usize; N_CUBIES],
}

impl Default for Cube {
    fn default() -> Self {
        Self {
            rotations: [Default::default(); N_CUBIES],
            idxs: (0..N_CUBIES).collect_vec().try_into().unwrap(),
        }
    }
}

impl Cube {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn rotate_layer(&mut self, layer_idx: usize, turn: QuarterTurn) {
        assert!(
            layer_idx < LAYERS.len(),
            "layer index {} is out of bounds",
            layer_idx
        );

        let axis = match layer_idx / CUBE_SIZE {
            0 => Axis::X,
            1 => Axis::Y,
            2 => Axis::Z,
            _ => unreachable!(),
        };
        let rotation = CubeRotation::from_axis_turn(axis, turn);
        let layer = LAYERS[layer_idx];

        for i in layer.iter().map(|&i| self.idxs[i]) {
            self.rotations[i] = rotation * self.rotations[i];
        }

        match turn {
            QuarterTurn::Quarter => {
                for orbit in [[0, 2, 8, 6], [1, 5, 7, 3]] {
                    let tmp = self.idxs[layer[orbit[0]]];
                    for i in 0..3 {
                        self.idxs[layer[orbit[i]]] = self.idxs[layer[orbit[i + 1]]];
                    }
                    self.idxs[layer[orbit[3]]] = tmp;
                }
            }
            QuarterTurn::Half => {
                for [i, j] in [[0, 8], [2, 6], [1, 7], [3, 5]] {
                    self.idxs.swap(layer[i], layer[j]);
                }
            }
            QuarterTurn::ThreeQuarters => {
                for orbit in [[0, 6, 8, 2], [1, 3, 7, 5]] {
                    let tmp = self.idxs[layer[orbit[0]]];
                    for i in 0..3 {
                        self.idxs[layer[orbit[i]]] = self.idxs[layer[orbit[i + 1]]];
                    }
                    self.idxs[layer[orbit[3]]] = tmp;
                }
            }
            _ => {}
        };
    }

    pub fn rotations(&self) -> &[CubeRotation; N_CUBIES] {
        &self.rotations
    }

    pub fn cubie_indices(&self) -> &[usize; N_CUBIES] {
        &self.idxs
    }

    fn orientation(&self) -> CubeRotation {
        const CENTER_CUBIE: usize = 13;
        self.rotations[CENTER_CUBIE]
    }

    pub fn is_solved(&self) -> bool {
        const FRINGE_CUBIES: [usize; 20] = [
            0, 1, 2, 3, 5, 6, 7, 8, 9, 11, 15, 17, 18, 19, 20, 21, 23, 24, 25, 26,
        ];

        let orientation = self.orientation();

        // this works because a cubie can only be in the correct position if it is rotated
        // correctly
        FRINGE_CUBIES
            .into_iter()
            .map(|i| self.rotations[i])
            .all(|r| r == orientation)
    }

    pub fn reset(&mut self) {
        // this is a bit involved if we want to maintain the cube orientation
        // using the orientation basis, we can generate the iteration order
        let orientation = self.orientation();
        let basis = orientation.basis();

        // fill order for indices
        let iter = iproduct!(
            index_iter(basis[0]),
            index_iter(basis[1]),
            index_iter(basis[2])
        )
        .map(|(a, b, c)| a + b + c);

        for (i, idx) in iter.enumerate() {
            self.idxs[idx] = i;
        }

        // reset orientations
        self.rotations.fill(orientation);
    }
}

fn index_iter(v: BasisVector) -> impl Iterator<Item = usize> + Clone {
    let f = match v.axis {
        Axis::X => |i| 9 * i,
        Axis::Y => |i| 3 * i,
        Axis::Z => |i| i,
    };
    let iter = (0..CUBE_SIZE).map(f);
    match v.sign {
        Sign::Positive => Either::Right(iter),
        Sign::Negative => Either::Left(iter.rev()),
    }
}
