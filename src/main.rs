mod cube;
mod rotation;

use std::{
    collections::HashSet,
    f32::consts::{FRAC_PI_2, PI},
};

use cube::Cube;
use glium::{
    draw_parameters::PolygonOffset,
    glutin::{
        dpi::PhysicalPosition,
        event::{
            ElementState, Event, KeyboardInput, MouseButton, MouseScrollDelta, VirtualKeyCode,
            WindowEvent,
        },
        event_loop::{ControlFlow, EventLoop},
        window::WindowBuilder,
        ContextBuilder,
    },
    implement_vertex,
    index::PrimitiveType,
    uniform, Display, IndexBuffer, Program, Surface, VertexBuffer,
};
use itertools::{iproduct, Itertools};
use nalgebra::{
    Perspective3, Point2, Point3, Rotation3, Similarity3, Translation3, Unit, Vector2, Vector3,
};
use rand::{seq::SliceRandom, Rng};
use rotation::{Axis, QuarterTurn};
use strum::{EnumIter, IntoEnumIterator};

use crate::cube::LAYERS;

type Color = [f32; 3];
type LayerIdx = usize;

// each vertex should have a color associated with it
#[derive(Clone, Copy)]
struct Vertex {
    position: [f32; 3],
    color: Color,
}

implement_vertex!(Vertex, position, color);

const CUBE_SIZE: usize = 3;
const N_CUBIES: usize = CUBE_SIZE.pow(3);
const N_CUBE_FACES: usize = 6;

const CUBIE_HALF_WIDTH: f32 = 1.0;
const CUBIE_WIDTH: f32 = CUBIE_HALF_WIDTH * 2.0;
const MIN_SHIFT: f32 = -CUBIE_WIDTH;
const MAX_SHIFT: f32 = CUBIE_WIDTH;

const FACE_DIST: f32 = CUBIE_HALF_WIDTH * (CUBE_SIZE as f32);
const FACE_TURN_RATE: f32 = 4.0;

// for tracking the current click state
#[derive(Default, Debug)]
enum State {
    #[default]
    Released,
    CubeRotation {
        button: MouseButton,
    },
    ClickedFace(ClickedFace),
    LayerTurn(LayerTurn),
}

#[derive(Debug)]
struct ClickedFace {
    pos: Point2<f32>,
    screen_axes: [Unit<Vector2<f32>>; 2],
    layers: [LayerIdx; 2],
}

#[derive(Debug)]
struct LayerTurn {
    pos: Point2<f32>,
    screen_axis: Unit<Vector2<f32>>,
    layer_idx: LayerIdx,
    angle: f32,
}

#[derive(Debug, Clone, Copy, EnumIter)]
enum Face {
    Left,
    Right,
    Front,
    Back,
    Down,
    Up,
}

fn face_indices(n: usize) -> Vec<u16> {
    const FACE_INDICES: [u16; 6] = [0, 1, 2, 1, 2, 3];

    let n_rects = n * N_CUBE_FACES;

    (0..n_rects as u16)
        .cartesian_product(FACE_INDICES)
        .map(|(i, j)| 4 * i + j)
        .collect_vec()
}

fn edge_indices(n: usize) -> Vec<u16> {
    const EDGE_INDICES: [u16; 24] = [
        0, 1, 2, 3, 4, 5, 6, 7, 0, 2, 1, 3, 4, 6, 5, 7, 0, 4, 1, 5, 2, 6, 3, 7,
    ];

    (0..n as u16)
        .cartesian_product(EDGE_INDICES)
        .map(|(i, j)| 8 * i + j)
        .collect_vec()
}

fn intersecting_face(start: Point3<f32>, dir: Vector3<f32>) -> Option<(Face, Point3<f32>)> {
    let mut best_face: Option<Face> = None;
    let mut best_dist = f32::INFINITY;
    let mut best_intersection: Point3<f32> = Default::default();

    const FACE_DIRS: [Vector3<f32>; N_CUBE_FACES] = [
        Vector3::new(-1.0, 0.0, 0.0),
        Vector3::new(1.0, 0.0, 0.0),
        Vector3::new(0.0, -1.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
        Vector3::new(0.0, 0.0, -1.0),
        Vector3::new(0.0, 0.0, 1.0),
    ];

    for (face, normal) in Face::iter().zip(FACE_DIRS) {
        // skip if line and plane are parallel
        let dot = normal.dot(&dir);
        if dot == 0.0 {
            continue;
        }

        let plane_point: Point3<f32> = (normal * FACE_DIST).into();
        let dist = (plane_point - start).dot(&normal) / dot;
        let intersection = start + dist * dir;

        let range = -FACE_DIST..=FACE_DIST;

        let in_face = match face {
            Face::Left | Face::Right => {
                range.contains(&intersection.y) && range.contains(&intersection.z)
            }
            Face::Front | Face::Back => {
                range.contains(&intersection.x) && range.contains(&intersection.z)
            }
            Face::Up | Face::Down => {
                range.contains(&intersection.x) && range.contains(&intersection.y)
            }
        };
        if !in_face {
            continue;
        }

        // then distance check
        // must be non-negative and smaller than best
        if dist < 0.0 || dist >= best_dist {
            continue;
        }

        // otherwise, everything is good
        best_face = Some(face);
        best_dist = dist;
        best_intersection = intersection;
    }

    best_face.map(|face| (face, best_intersection))
}

fn mouse_to_screen_coords(mouse_pos: (f64, f64), dimensions: (u32, u32)) -> Point2<f32> {
    let (click_x, click_y) = mouse_pos;
    let (width, height) = dimensions;
    // need to invert y because of how mouse coordinates work
    Point2::new(
        (2.0 * click_x / width as f64 - 1.0) as _,
        -(2.0 * click_y / height as f64 - 1.0) as _,
    )
}

fn clicked_face(
    pos: Point2<f32>,
    model: &Similarity3<f32>,
    perspective: &Perspective3<f32>,
) -> Option<ClickedFace> {
    // we want to find an intersection with the ray parallel to the z-direction, going through the
    // point clicked on the screen

    // NOTE: depth values range from -1 to 1
    // near plane is at z = -1
    let start = Point3::new(pos.x, pos.y, -1.0);
    let end = Point3::new(pos.x, pos.y, 1.0);

    let inverse = |p| model.inverse_transform_point(&perspective.unproject_point(p));

    // we need to apply the inverse of the perspective and model transformations to put them in the
    // same coordinates as the original cube
    let start = inverse(&start);
    let end = inverse(&end);

    let dir = end - start;

    // find the intersection or break early
    let (face, cube_pos) = intersecting_face(start, dir)?;

    // NOTE: for consistent face rotations later, we make sure that the cube_axes are only ever the
    // positive x, y, or z axes.

    let screen_axes = {
        let raw_axes = match face {
            Face::Left => [Vector3::z_axis(), -Vector3::y_axis()],
            Face::Right => [-Vector3::z_axis(), Vector3::y_axis()],
            Face::Front => [Vector3::x_axis(), -Vector3::z_axis()],
            Face::Back => [-Vector3::x_axis(), Vector3::z_axis()],
            Face::Down => [Vector3::y_axis(), -Vector3::x_axis()],
            Face::Up => [-Vector3::y_axis(), Vector3::x_axis()],
        };
        let axes = raw_axes.map(|v| {
            let w = perspective.project_point(&model.transform_point(&v.into_inner().into()));
            Vector2::new(w.x, w.y)
        });
        if axes.iter().any(|v| v.norm() == 0.0) {
            return None;
        }
        axes.map(Unit::new_normalize)
    };

    fn layer_index(x: f32) -> usize {
        ((x + FACE_DIST) / CUBIE_WIDTH).floor().clamp(0.0, 2.0) as usize
    }
    // too lazy to deal with the magic numbers here
    let layers = {
        match face {
            Face::Left | Face::Right => [3 + layer_index(cube_pos.y), 6 + layer_index(cube_pos.z)],
            Face::Front | Face::Back => [6 + layer_index(cube_pos.z), layer_index(cube_pos.x)],
            Face::Down | Face::Up => [layer_index(cube_pos.x), 3 + layer_index(cube_pos.y)],
        }
    };

    Some(ClickedFace {
        pos,
        screen_axes,
        layers,
    })
}

fn layer_to_axis(layer_idx: usize) -> Axis {
    match layer_idx / CUBE_SIZE {
        0 => Axis::X,
        1 => Axis::Y,
        2 => Axis::Z,
        _ => panic!("layer index {} is out of bounds", layer_idx),
    }
}

fn layer_turn(
    ClickedFace {
        pos,
        screen_axes,
        layers,
    }: &ClickedFace,
    new_pos: Point2<f32>,
) -> Option<LayerTurn> {
    const TURN_THRESHOLD: f32 = 0.05;

    let pos = *pos;
    let delta = new_pos - pos;

    if delta.norm() < TURN_THRESHOLD {
        return None;
    }

    // check which axis delta is more aligned with
    let dots = screen_axes.map(|a| a.dot(&delta));

    let i = if dots[0].abs() > dots[1].abs() { 0 } else { 1 };

    Some(LayerTurn {
        pos,
        screen_axis: screen_axes[i],
        layer_idx: layers[i],
        angle: (dots[i] * FACE_TURN_RATE).rem_euclid(2.0 * PI),
    })
}

fn update_layer_turn(
    LayerTurn {
        pos,
        screen_axis,
        angle,
        ..
    }: &mut LayerTurn,
    new_pos: Point2<f32>,
) {
    let delta = new_pos - *pos;
    let dot = screen_axis.dot(&delta);
    *angle = (dot * FACE_TURN_RATE).rem_euclid(2.0 * PI);
}

fn main() {
    let event_loop = EventLoop::new();
    let wb = WindowBuilder::new().with_title("Rubik's Cube");
    // .with_resizable(false)
    // .with_inner_size(PhysicalSize::new(600, 600));
    let cb = ContextBuilder::new()
        .with_depth_buffer(24)
        .with_vsync(false);

    let display = Display::new(wb, cb, &event_loop).unwrap();

    // these indices stay static
    let face_indices = IndexBuffer::new(
        &display,
        PrimitiveType::TrianglesList,
        &face_indices(N_CUBIES),
    )
    .unwrap();
    let edge_indices =
        IndexBuffer::new(&display, PrimitiveType::LinesList, &edge_indices(N_CUBIES)).unwrap();

    let corners = {
        let shifts = [-CUBIE_HALF_WIDTH, CUBIE_HALF_WIDTH].into_iter();
        iproduct!(shifts.clone(), shifts.clone(), shifts)
            .map(|(x, y, z)| Vector3::new(x, y, z))
            .collect_vec()
    };
    let translations = {
        let shifts = [-CUBIE_WIDTH, 0.0, CUBIE_WIDTH].into_iter();
        iproduct!(shifts.clone(), shifts.clone(), shifts)
            .map(|(x, y, z)| Vector3::new(x, y, z))
            .collect_vec()
    };

    let vertex_shader_src = include_str!("vertex_shader.vert");
    let fragment_shader_src = include_str!("fragment_shader.frag");

    let program =
        Program::from_source(&display, vertex_shader_src, fragment_shader_src, None).unwrap();

    // mutable state of the cubie rotations
    let mut cube = Cube::new();

    let mut mouse_pos = (0.0, 0.0);
    let mut state = State::default();
    let mut cube_rotation: Rotation3<f32> = Rotation3::default();

    let mut zoom = 1.0;

    event_loop.run(move |event, _, control_flow| {
        if !matches!(
            event,
            Event::RedrawRequested(..) | Event::WindowEvent { .. }
        ) {
            return;
        }

        let dimensions = display.get_framebuffer_dimensions();

        const CUBE_SCALE: f32 = 0.03;
        let model = Similarity3::from_parts(
            Translation3::new(0.0, 0.0, -1.0 / zoom),
            cube_rotation.into(),
            CUBE_SCALE,
        );

        let perspective = {
            let (width, height) = dimensions;
            Perspective3::new(width as f32 / height as f32, PI / 6.0, 0.1, 1024.0)
        };

        // NOTE: manually requesting redraws makes everything much faster!
        if let Event::RedrawRequested(..) = event {
            let mut target = display.draw();

            const LIGHT_GREEN: (f32, f32, f32, f32) = (0.45, 0.91, 0.48, 1.0);
            const LIGHT_GREY: (f32, f32, f32, f32) = (0.9, 0.9, 0.9, 1.0);

            let background_color = if cube.is_solved() {
                LIGHT_GREEN
            } else {
                LIGHT_GREY
            };

            target.clear_color_and_depth(background_color, 1.0);

            const VERTS_PER_CUBIE: usize = 12;
            const EDGES_PER_CUBIE: usize = 12;

            let mut face_vertices: Vec<Vertex> = Vec::with_capacity(VERTS_PER_CUBIE * N_CUBIES);
            let mut edge_vertices: Vec<Vertex> = Vec::with_capacity(EDGES_PER_CUBIE * N_CUBIES);

            // an attempt at deduplication
            let (face, face_rtn) = if let State::LayerTurn(LayerTurn {
                layer_idx, angle, ..
            }) = &state
            {
                (
                    LAYERS[*layer_idx]
                        .into_iter()
                        .map(|i| cube.cubie_indices()[i])
                        .collect(),
                    Rotation3::from_axis_angle(&layer_to_axis(*layer_idx).to_unit(), *angle),
                )
            } else {
                (HashSet::new(), Default::default())
            };

            for (i, (&tln, &rtn)) in translations.iter().zip(cube.rotations()).enumerate() {
                const RED: Color = [1.0, 0.0, 0.0];
                const ORANGE: Color = [1.0, 0.5, 0.0];
                const BLUE: Color = [0.0, 0.0, 1.0];
                const GREEN: Color = [0.0, 1.0, 0.0];
                const YELLOW: Color = [1.0, 1.0, 0.0];
                const WHITE: Color = [1.0, 1.0, 1.0];
                const GREY: Color = [0.3, 0.3, 0.3];
                const BLACK: Color = [0.0, 0.0, 0.0];

                const CUBE_INDICES: [[usize; 4]; N_CUBE_FACES] = [
                    [0, 1, 2, 3],
                    [4, 5, 6, 7],
                    [0, 1, 4, 5],
                    [2, 3, 6, 7],
                    [0, 2, 4, 6],
                    [1, 3, 5, 7],
                ];

                // very ugly way of calculating colors lmao
                let colors = [
                    if tln.x == MIN_SHIFT { RED } else { GREY },
                    if tln.x == MAX_SHIFT { ORANGE } else { GREY },
                    if tln.y == MIN_SHIFT { BLUE } else { GREY },
                    if tln.y == MAX_SHIFT { GREEN } else { GREY },
                    if tln.z == MIN_SHIFT { YELLOW } else { GREY },
                    if tln.z == MAX_SHIFT { WHITE } else { GREY },
                ];

                // might need to calculate a new rotation, depending on whether the face is
                // rotated and whether this cubie matches
                let rotation = if face.contains(&i) {
                    face_rtn * rtn.to_rotation3()
                } else {
                    rtn.to_rotation3()
                };

                // now we can iterate over vertices for each cubie face
                for (cubie_face, color) in CUBE_INDICES.into_iter().zip(colors) {
                    for pos in cubie_face.map(|i| corners[i]) {
                        face_vertices.push(Vertex {
                            position: *rotation.transform_vector(&(pos + tln)).as_ref(),
                            color,
                        })
                    }
                }

                for &pos in &corners {
                    edge_vertices.push(Vertex {
                        position: *rotation.transform_vector(&(pos + tln)).as_ref(),
                        color: BLACK,
                    })
                }
            }

            let face_vertices = VertexBuffer::new(&display, &face_vertices).unwrap();

            let uniforms = uniform! {
                model: model.to_homogeneous().as_ref().to_owned(),
                perspective: perspective.as_matrix().as_ref().to_owned(),
            };

            let params = glium::DrawParameters {
                depth: glium::Depth {
                    test: glium::DepthTest::IfLess,
                    write: true,
                    ..Default::default()
                },
                line_width: Some(4.0),
                // NOTE: positive polygon offset = farther back
                polygon_offset: PolygonOffset {
                    fill: true,
                    units: 1.0,
                    ..Default::default()
                },
                ..Default::default()
            };

            target
                .draw(&face_vertices, &face_indices, &program, &uniforms, &params)
                .unwrap();

            let edge_vertices = VertexBuffer::new(&display, &edge_vertices).unwrap();
            target
                .draw(&edge_vertices, &edge_indices, &program, &uniforms, &params)
                .unwrap();

            target.finish().unwrap();
        } else if let Event::WindowEvent { event, .. } = event {
            match event {
                WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }
                WindowEvent::MouseInput {
                    state: ElementState::Pressed,
                    button: button @ (MouseButton::Left | MouseButton::Right),
                    ..
                } => {
                    // in the case that button is already pressed, we don't want to do anything
                    if !matches!(state, State::Released) {
                        return;
                    }

                    let pos = mouse_to_screen_coords(mouse_pos, dimensions);
                    state = if let (MouseButton::Left, Some(clicked)) =
                        (button, clicked_face(pos, &model, &perspective))
                    {
                        State::ClickedFace(clicked)
                    } else {
                        State::CubeRotation { button }
                    };
                    display.gl_window().window().request_redraw();
                }
                WindowEvent::MouseInput {
                    state: ElementState::Released,
                    button: button @ (MouseButton::Left | MouseButton::Right),
                    ..
                } => {
                    // the released button should match the one that was pressed, otherwise don't
                    // do anything
                    let button_match = match state {
                        State::Released => false,
                        State::CubeRotation { button: expected } => button == expected,
                        State::ClickedFace(_) => button == MouseButton::Left,
                        State::LayerTurn(_) => button == MouseButton::Left,
                    };

                    if !button_match {
                        return;
                    }

                    if let State::LayerTurn(LayerTurn {
                        layer_idx, angle, ..
                    }) = state
                    {
                        let multiple = (angle / FRAC_PI_2).round();
                        let turn = match multiple as i32 {
                            0 | 4 => QuarterTurn::Zero,
                            1 => QuarterTurn::Quarter,
                            2 => QuarterTurn::Half,
                            3 => QuarterTurn::ThreeQuarters,
                            _ => unreachable!("multiple should always be between 0 and 4"),
                        };
                        cube.rotate_layer(layer_idx, turn);
                        display.gl_window().window().request_redraw();
                    }

                    state = State::Released;
                }
                WindowEvent::CursorMoved {
                    position: PhysicalPosition { x, y },
                    ..
                } => {
                    match &mut state {
                        State::CubeRotation { .. } => {
                            const CUBE_ROTATION_RATE: f32 = 0.007;

                            // calculate a delta
                            let (x0, y0) = mouse_pos;
                            let (dx, dy) = ((x - x0) as f32, (y - y0) as f32);

                            // convert delta to a rotation
                            let d_rotation = Rotation3::from_scaled_axis(
                                Vector3::new(dy, dx, 0.0).scale(CUBE_ROTATION_RATE),
                            );

                            // and apply it
                            cube_rotation = d_rotation * cube_rotation;
                            display.gl_window().window().request_redraw();
                        }
                        State::ClickedFace(clicked) => {
                            let new_pos = mouse_to_screen_coords((x, y), dimensions);

                            if let Some(turn) = layer_turn(clicked, new_pos) {
                                state = State::LayerTurn(turn);
                                display.gl_window().window().request_redraw();
                            }
                        }
                        State::LayerTurn(turn) => {
                            // this should just modify the layer turn
                            let new_pos = mouse_to_screen_coords((x, y), dimensions);
                            update_layer_turn(turn, new_pos);
                            display.gl_window().window().request_redraw();
                        }
                        _ => {}
                    };

                    mouse_pos = (x, y);
                }
                WindowEvent::MouseWheel {
                    delta: MouseScrollDelta::LineDelta(_, cols),
                    ..
                } => {
                    const ZOOM_STEP: f32 = 0.05;
                    const MIN_ZOOM: f32 = 0.5;
                    const MAX_ZOOM: f32 = 1.5;

                    zoom = (zoom + cols * ZOOM_STEP).clamp(MIN_ZOOM, MAX_ZOOM);
                    display.gl_window().window().request_redraw();
                }
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(keycode),
                            ..
                        },
                    ..
                } => match keycode {
                    VirtualKeyCode::Q => {
                        *control_flow = ControlFlow::Exit;
                    }
                    VirtualKeyCode::R => {
                        // reset everything
                        cube.reset();
                        display.gl_window().window().request_redraw();
                    }
                    VirtualKeyCode::S => {
                        // random face turns
                        for _ in 0..100 {
                            const FRINGE_LAYERS: [usize; 6] = [0, 2, 3, 5, 6, 8];

                            let mut rng = rand::thread_rng();

                            let layer_idx = *FRINGE_LAYERS.choose(&mut rng).unwrap();
                            let turn = match rng.gen_range(1..=3) {
                                1 => QuarterTurn::Quarter,
                                2 => QuarterTurn::Half,
                                3 => QuarterTurn::ThreeQuarters,
                                _ => unreachable!(),
                            };
                            cube.rotate_layer(layer_idx, turn);
                        }
                        display.gl_window().window().request_redraw();
                    }
                    _ => {}
                },
                _ => {}
            }
        }
    });
}
