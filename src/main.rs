use std::{f32::consts::PI, time::Instant};

use glium::{
    glutin::{
        dpi::PhysicalPosition,
        event::{ElementState, Event, MouseButton, WindowEvent},
        event_loop::{ControlFlow, EventLoop},
        window::WindowBuilder,
        ContextBuilder,
    },
    implement_vertex, uniform, Display, IndexBuffer, Program, Surface, VertexBuffer,
};
use itertools::{iproduct, Either, Itertools};
use nalgebra::{Matrix4, UnitQuaternion, Vector3};

type Color = [f32; 3];

// each vertex should have a color associated with it
#[derive(Clone, Copy)]
struct Vertex {
    position: [f32; 3],
    color: Color,
}

implement_vertex!(Vertex, position, color);

const RED: Color = [1.0, 0.0, 0.0];
const ORANGE: Color = [1.0, 0.5, 0.0];
const BLUE: Color = [0.0, 0.0, 1.0];
const GREEN: Color = [0.0, 1.0, 0.0];
const YELLOW: Color = [1.0, 1.0, 0.0];
const WHITE: Color = [1.0, 1.0, 1.0];
const GREY: Color = [0.5, 0.5, 0.5];

// for tracking the current click state
// TODO: add enum fields
#[derive(Default)]
enum State {
    #[default]
    Released,
    CubeRotation,
    ClickedFace,
    FaceTurn {
        face: Vec<usize>,
        rotation: UnitQuaternion<f32>,
    },
}

const N_CUBIES: usize = 27;
const N_FACE_CUBIES: usize = 9;
const N_CUBE_FACES: usize = 6;
const VERTS_PER_CUBIE: usize = 12;

const CUBE_INDICES: [[usize; 4]; N_CUBE_FACES] = [
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [0, 1, 4, 5],
    [2, 3, 6, 7],
    [0, 2, 4, 6],
    [1, 3, 5, 7],
];

fn rect_indices(n: usize) -> Vec<u16> {
    let n_rects = n * N_CUBE_FACES;
    let mut indices = Vec::with_capacity(n_rects);

    for i in 0..n_rects as u16 {
        indices.push(4 * i);
        indices.push(4 * i + 1);
        indices.push(4 * i + 2);
        indices.push(4 * i + 1);
        indices.push(4 * i + 2);
        indices.push(4 * i + 3);
    }

    indices
}

fn main() {
    let event_loop = EventLoop::new();
    let wb = WindowBuilder::new();
    let cb = ContextBuilder::new().with_depth_buffer(24);
    let display = Display::new(wb, cb, &event_loop).unwrap();

    // let vertices = VertexBuffer::new(&display, &CUBE).unwrap();

    // these indices stay static
    let indices = IndexBuffer::new(
        &display,
        glium::index::PrimitiveType::TrianglesList,
        &rect_indices(N_CUBIES),
    )
    .unwrap();

    let corners = {
        let shifts = [-1.0, 1.0f32].into_iter();
        iproduct!(shifts.clone(), shifts.clone(), shifts)
            .map(|(x, y, z)| Vector3::new(x, y, z))
            .collect_vec()
    };
    let translations = {
        let shifts = [-2.0, 0.0, 2.0f32].into_iter();
        iproduct!(shifts.clone(), shifts.clone(), shifts)
            .map(|(x, y, z)| Vector3::new(x, y, z))
            .collect_vec()
    };

    let vertex_shader_src = include_str!("vertex_shader.vert");
    let fragment_shader_src = include_str!("fragment_shader.frag");

    let program =
        Program::from_source(&display, vertex_shader_src, fragment_shader_src, None).unwrap();

    // mutable state of the cubie rotations
    let mut rotations: Vec<UnitQuaternion<f32>> = vec![Default::default(); N_CUBIES];

    let mut mouse_pos = (0.0, 0.0);
    let mut pressed = false;
    let mut state = State::default();
    let mut cube_rotation: UnitQuaternion<f32> = UnitQuaternion::default();
    event_loop.run(move |event, _, control_flow| {
        let next_frame_time = Instant::now() + std::time::Duration::from_nanos(16_666_667);
        *control_flow = ControlFlow::WaitUntil(next_frame_time);

        let mut target = display.draw();
        target.clear_color_and_depth((0.8, 0.8, 0.8, 1.0), 1.0);

        let mut vertices: Vec<Vertex> = Vec::with_capacity(VERTS_PER_CUBIE * N_CUBIES);

        // an attempt at deduplication
        let (mut face, face_rtn) = if let State::FaceTurn { face, rotation, .. } = &state {
            (Either::Right(face.iter().cloned()).peekable(), *rotation)
        } else {
            (
                Either::Left(std::iter::empty()).peekable(),
                Default::default(),
            )
        };

        for (i, (&tln, &rtn)) in translations.iter().zip(rotations.iter()).enumerate() {
            // very ugly way of calculating colors lmao
            let colors = [
                if tln.x == -2.0 { RED } else { GREY },
                if tln.x == 2.0 { ORANGE } else { GREY },
                if tln.y == -2.0 { BLUE } else { GREY },
                if tln.y == 2.0 { GREEN } else { GREY },
                if tln.z == -2.0 { YELLOW } else { GREY },
                if tln.z == 2.0 { WHITE } else { GREY },
            ];

            // might need to calculate a new rotation, depending on whether the face is
            // rotated and whether this cubie matches
            let rotation = if face.next_if(|&j| i == j).is_some() {
                face_rtn * rtn
            } else {
                rtn
            };

            // now we can iterate over vertices for each cubie face
            for (cubie_face, color) in CUBE_INDICES.into_iter().zip(colors) {
                for pos in cubie_face.map(|i| corners[i]) {
                    vertices.push(Vertex {
                        position: *rotation.transform_vector(&(pos + tln)).as_ref(),
                        color,
                    })
                }
            }
        }

        let vertices = VertexBuffer::new(&display, &vertices).unwrap();

        let model = cube_rotation
            .to_homogeneous()
            .append_scaling(0.05)
            .append_translation(&Vector3::new(0.0, 0.0, -1.0));

        let perspective = {
            let (width, height) = target.get_dimensions();
            Matrix4::new_perspective(width as f32 / height as f32, PI / 3.0, 0.1, 1024.0)
        };

        let uniforms = uniform! {
            model: model.as_ref().to_owned(),
            perspective: perspective.as_ref().to_owned(),
        };

        let params = glium::DrawParameters {
            depth: glium::Depth {
                test: glium::DepthTest::IfLess,
                write: true,
                ..Default::default()
            },
            ..Default::default()
        };

        target
            .draw(&vertices, &indices, &program, &uniforms, &params)
            .unwrap();
        target.finish().unwrap();

        if let Event::WindowEvent { event, .. } = event {
            match event {
                WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }
                WindowEvent::MouseInput {
                    state,
                    button: MouseButton::Left,
                    ..
                } => {
                    pressed = state == ElementState::Pressed;
                }
                WindowEvent::CursorMoved {
                    position: PhysicalPosition { x, y },
                    ..
                } => {
                    // update position
                    let old_pos = mouse_pos;

                    mouse_pos = (x, y);

                    // if not pressed, just update location
                    if !pressed {
                        return;
                    }

                    // otherwise, calculate a delta
                    let (dx, dy) = ((x - old_pos.0) as f32, (y - old_pos.1) as f32);

                    // convert delta to a rotation
                    let d_rotation =
                        UnitQuaternion::from_scaled_axis(Vector3::new(dy, dx, 0.0).scale(0.01));

                    // and apply it
                    cube_rotation = d_rotation * cube_rotation;
                }
                _ => {}
            }
        }
    });
}

// planning:
// the changing data structure is just the one that stores a rotation for each cubie
// apart from that:
//  - translation of each cubie
//  - colors
// ^ these can all be static
//
// for collision detection, cast a ray in the z-direction and check the closest cube face that
// intersects
// action based on the first click:
//  - if outside, treat this as cube rotation
//  - if inside:
//      - the first click determines the face
//      - the movement determines which way the face should turn (now store a line together with
//      the face)
