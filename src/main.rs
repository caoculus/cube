use std::{f32::consts::PI, time::Instant};

use glium::{
    glutin::{
        event::{Event, StartCause, WindowEvent},
        event_loop::{ControlFlow, EventLoop},
        window::WindowBuilder,
        ContextBuilder,
    },
    implement_vertex, uniform, Display, IndexBuffer, Program, Surface, VertexBuffer,
};

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

    Vertex { position: CORNERS[0], color: RED },
    Vertex { position: CORNERS[2], color: RED },
    Vertex { position: CORNERS[4], color: RED },
    Vertex { position: CORNERS[6], color: RED },

    Vertex { position: CORNERS[2], color: GREEN },
    Vertex { position: CORNERS[3], color: GREEN },
    Vertex { position: CORNERS[6], color: GREEN },
    Vertex { position: CORNERS[7], color: GREEN },

    Vertex { position: CORNERS[1], color: ORANGE },
    Vertex { position: CORNERS[3], color: ORANGE },
    Vertex { position: CORNERS[5], color: ORANGE },
    Vertex { position: CORNERS[7], color: ORANGE },

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

    let mut t: f32 = 0.0;
    event_loop.run(move |event, _, control_flow| {
        let next_frame_time = Instant::now() + std::time::Duration::from_nanos(16_666_667);
        *control_flow = ControlFlow::WaitUntil(next_frame_time);

        let mut target = display.draw();
        target.clear_color(0.8, 0.8, 0.8, 1.0); // light grey

        let uniforms = uniform! {
            matrix: [
                [0.1 * t.cos(), 0.0, 0.1 * t.sin(), 0.0],
                [0.0, 0.1, 0.0, 0.0],
                [0.1 * -t.sin(), 0.0, 0.1 * t.cos(), 0.0],
                [0.0, 0.0, 1.0, 1.0f32],
            ]
        };

        t += 0.01;
        if t > 2.0 * PI {
            t -= 2.0 * PI;
        }

        target
            .draw(
                &vertices,
                &indices,
                &program,
                &uniforms,
                &Default::default(),
            )
            .unwrap();
        target.finish().unwrap();

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                    return;
                }
                // WindowEvent::MouseInput { device_id, state, button, modifiers }
                _ => return,
            },
            Event::NewEvents(cause) => match cause {
                StartCause::ResumeTimeReached { .. } => (),
                StartCause::Init => (),
                _ => return,
            },
            _ => return,
        }
    });
}
