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
use nalgebra::{Matrix4, UnitQuaternion, Vector3};

// each vertex should have a color associated with it
#[derive(Clone, Copy)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

implement_vertex!(Vertex, position, color);

const WHITE: [f32; 3] = [1.0, 1.0, 1.0];
const RED: [f32; 3] = [1.0, 0.0, 0.0];
const GREEN: [f32; 3] = [0.0, 1.0, 0.0];
const BLUE: [f32; 3] = [0.0, 0.0, 1.0];
const YELLOW: [f32; 3] = [1.0, 1.0, 0.0];
const ORANGE: [f32; 3] = [1.0, 0.5, 0.0];

#[rustfmt::skip]
const CORNERS: [[f32; 3]; 8] = [
    [-1.0, -1.0, -1.0],
    [-1.0, -1.0,  1.0],
    [-1.0,  1.0, -1.0],
    [-1.0,  1.0,  1.0],
    [ 1.0, -1.0, -1.0],
    [ 1.0, -1.0,  1.0],
    [ 1.0,  1.0, -1.0],
    [ 1.0,  1.0,  1.0f32],
];

#[rustfmt::skip]
const CUBE: [Vertex; 24] = [
    Vertex { position: CORNERS[4], color: WHITE },
    Vertex { position: CORNERS[5], color: WHITE },
    Vertex { position: CORNERS[6], color: WHITE },
    Vertex { position: CORNERS[7], color: WHITE },

    Vertex { position: CORNERS[0], color: BLUE },
    Vertex { position: CORNERS[1], color: BLUE },
    Vertex { position: CORNERS[4], color: BLUE },
    Vertex { position: CORNERS[5], color: BLUE },

    Vertex { position: CORNERS[0], color: ORANGE },
    Vertex { position: CORNERS[2], color: ORANGE },
    Vertex { position: CORNERS[4], color: ORANGE },
    Vertex { position: CORNERS[6], color: ORANGE },

    Vertex { position: CORNERS[2], color: GREEN },
    Vertex { position: CORNERS[3], color: GREEN },
    Vertex { position: CORNERS[6], color: GREEN },
    Vertex { position: CORNERS[7], color: GREEN },

    Vertex { position: CORNERS[1], color: RED },
    Vertex { position: CORNERS[3], color: RED },
    Vertex { position: CORNERS[5], color: RED },
    Vertex { position: CORNERS[7], color: RED },

    Vertex { position: CORNERS[0], color: YELLOW },
    Vertex { position: CORNERS[1], color: YELLOW },
    Vertex { position: CORNERS[2], color: YELLOW },
    Vertex { position: CORNERS[3], color: YELLOW },
];

#[rustfmt::skip]
const INDICES: [u16; 36] = [
    0, 1, 2, 1, 2, 3,
    4, 5, 6, 5, 6, 7,
    8, 9, 10, 9, 10, 11,
    12, 13, 14, 13, 14, 15,
    16, 17, 18, 17, 18, 19,
    20, 21, 22, 21, 22, 23,
];

fn main() {
    let event_loop = EventLoop::new();
    let wb = WindowBuilder::new();
    let cb = ContextBuilder::new().with_depth_buffer(24);
    let display = Display::new(wb, cb, &event_loop).unwrap();

    let vertices = VertexBuffer::new(&display, &CUBE).unwrap();
    let indices = IndexBuffer::new(
        &display,
        glium::index::PrimitiveType::TrianglesList,
        &INDICES,
    )
    .unwrap();

    let vertex_shader_src = include_str!("vertex_shader.vert");
    let fragment_shader_src = include_str!("fragment_shader.frag");

    let program =
        Program::from_source(&display, vertex_shader_src, fragment_shader_src, None).unwrap();

    let mut rotation: UnitQuaternion<f32> = UnitQuaternion::default();
    let mut mouse_pos = (0.0, 0.0);
    let mut pressed = false;
    event_loop.run(move |event, _, control_flow| {
        let next_frame_time = Instant::now() + std::time::Duration::from_nanos(16_666_667);
        *control_flow = ControlFlow::WaitUntil(next_frame_time);

        let mut target = display.draw();
        target.clear_color_and_depth((0.8, 0.8, 0.8, 1.0), 1.0);

        let model = rotation
            .to_homogeneous()
            .append_scaling(0.2)
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
                    rotation = d_rotation * rotation;
                }
                _ => {}
            }
        }
    });
}
