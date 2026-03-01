#![cfg(not(target_family = "wasm"))]

use gpui::{DevicePixels, Scene, Size};
use gpui_wgpu::{CustomShaderDescriptor, WgpuCustomShader, WgpuRenderer, WgpuSurfaceConfig};
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowAttributes, WindowId},
};

const CUBE_SHADER: &str = r#"
struct Uniforms {
    time: f32,
    aspect_ratio: f32,
    _padding0: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
};

const CUBE_VERTICES: array<vec3<f32>, 36> = array<vec3<f32>, 36>(
    vec3<f32>(-1.0, -1.0,  1.0), vec3<f32>( 1.0, -1.0,  1.0), vec3<f32>( 1.0,  1.0,  1.0),
    vec3<f32>(-1.0, -1.0,  1.0), vec3<f32>( 1.0,  1.0,  1.0), vec3<f32>(-1.0,  1.0,  1.0),

    vec3<f32>( 1.0, -1.0, -1.0), vec3<f32>(-1.0, -1.0, -1.0), vec3<f32>(-1.0,  1.0, -1.0),
    vec3<f32>( 1.0, -1.0, -1.0), vec3<f32>(-1.0,  1.0, -1.0), vec3<f32>( 1.0,  1.0, -1.0),

    vec3<f32>(-1.0, -1.0, -1.0), vec3<f32>(-1.0, -1.0,  1.0), vec3<f32>(-1.0,  1.0,  1.0),
    vec3<f32>(-1.0, -1.0, -1.0), vec3<f32>(-1.0,  1.0,  1.0), vec3<f32>(-1.0,  1.0, -1.0),

    vec3<f32>( 1.0, -1.0,  1.0), vec3<f32>( 1.0, -1.0, -1.0), vec3<f32>( 1.0,  1.0, -1.0),
    vec3<f32>( 1.0, -1.0,  1.0), vec3<f32>( 1.0,  1.0, -1.0), vec3<f32>( 1.0,  1.0,  1.0),

    vec3<f32>(-1.0,  1.0,  1.0), vec3<f32>( 1.0,  1.0,  1.0), vec3<f32>( 1.0,  1.0, -1.0),
    vec3<f32>(-1.0,  1.0,  1.0), vec3<f32>( 1.0,  1.0, -1.0), vec3<f32>(-1.0,  1.0, -1.0),

    vec3<f32>(-1.0, -1.0, -1.0), vec3<f32>( 1.0, -1.0, -1.0), vec3<f32>( 1.0, -1.0,  1.0),
    vec3<f32>(-1.0, -1.0, -1.0), vec3<f32>( 1.0, -1.0,  1.0), vec3<f32>(-1.0, -1.0,  1.0)
);

fn rotate_y(position: vec3<f32>, angle: f32) -> vec3<f32> {
    let s = sin(angle);
    let c = cos(angle);
    return vec3<f32>(
        c * position.x + s * position.z,
        position.y,
        -s * position.x + c * position.z
    );
}

fn rotate_x(position: vec3<f32>, angle: f32) -> vec3<f32> {
    let s = sin(angle);
    let c = cos(angle);
    return vec3<f32>(
        position.x,
        c * position.y - s * position.z,
        s * position.y + c * position.z
    );
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let base_position = CUBE_VERTICES[vertex_index];
    let rotated = rotate_x(rotate_y(base_position, uniforms.time * 1.1), uniforms.time * 0.7);
    let translated = rotated + vec3<f32>(0.0, 0.0, 4.0);

    let fov = 1.2;
    let clip_x = (translated.x / translated.z) * fov / uniforms.aspect_ratio;
    let clip_y = (translated.y / translated.z) * fov;
    let clip_z = translated.z / 10.0;

    var output: VertexOutput;
    output.clip_position = vec4<f32>(clip_x, clip_y, clip_z, 1.0);
    output.normal = normalize(rotated);
    return output;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let light_direction = normalize(vec3<f32>(0.4, 0.7, -0.6));
    let lambert = max(dot(in.normal, light_direction), 0.0);
    let base = vec3<f32>(0.15, 0.45, 0.85);
    let color = base * (0.2 + 0.8 * lambert);
    return vec4<f32>(color, 1.0);
}
"#;

#[repr(C)]
#[derive(Copy, Clone)]
struct ShaderUniforms {
    time: f32,
    aspect_ratio: f32,
    padding0: [f32; 2],
}

impl ShaderUniforms {
    fn as_bytes(&self) -> &[u8] {
        let pointer = self as *const ShaderUniforms as *const u8;
        let length = std::mem::size_of::<ShaderUniforms>();
        unsafe { std::slice::from_raw_parts(pointer, length) }
    }
}

struct AppState {
    window: Option<Window>,
    window_id: Option<WindowId>,
    renderer: Option<WgpuRenderer>,
    shader: Option<WgpuCustomShader>,
    scene: Scene,
    start: Instant,
}

impl AppState {
    fn new() -> Self {
        Self {
            window: None,
            window_id: None,
            renderer: None,
            shader: None,
            scene: Scene::default(),
            start: Instant::now(),
        }
    }

    fn create_renderer(window: &Window) -> anyhow::Result<(WgpuRenderer, WgpuCustomShader)> {
        let mut context = None;
        let size = window.inner_size();
        let renderer = WgpuRenderer::new(
            &mut context,
            window,
            WgpuSurfaceConfig {
                size: Size {
                    width: DevicePixels(size.width.max(1) as i32),
                    height: DevicePixels(size.height.max(1) as i32),
                },
                transparent: false,
            },
        )?;

        let aspect_ratio = if size.height == 0 {
            1.0
        } else {
            size.width as f32 / size.height as f32
        };
        let uniforms = ShaderUniforms {
            time: 0.0,
            aspect_ratio,
            padding0: [0.0, 0.0],
        };

        let shader = renderer.create_custom_shader(CustomShaderDescriptor {
            label: Some("spinning_cube_shader"),
            shader_source: CUBE_SHADER,
            vertex_entry: "vs_main",
            fragment_entry: "fs_main",
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            vertex_count: 36,
            instance_count: 1,
            uniform_bytes: Some(uniforms.as_bytes()),
            blend_state: Some(wgpu::BlendState::REPLACE),
        })?;

        Ok((renderer, shader))
    }

    fn draw_frame(&mut self) {
        let Some(window) = self.window.as_ref() else {
            return;
        };
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };
        let Some(shader) = self.shader.as_ref() else {
            return;
        };

        let window_size = window.inner_size();
        let aspect_ratio = if window_size.height == 0 {
            1.0
        } else {
            window_size.width as f32 / window_size.height as f32
        };
        let uniforms = ShaderUniforms {
            time: self.start.elapsed().as_secs_f32(),
            aspect_ratio,
            padding0: [0.0, 0.0],
        };

        if let Err(error) = renderer.update_custom_shader_uniforms(shader, uniforms.as_bytes()) {
            log::error!("Failed to update cube uniforms: {error:#}");
            return;
        }

        renderer.draw_with_custom_shader(&self.scene, shader);
    }
}

impl ApplicationHandler for AppState {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attributes = WindowAttributes::default()
            .with_title("GPUI WGPU Spinning Cube")
            .with_inner_size(winit::dpi::LogicalSize::new(960.0, 640.0));
        let window = match event_loop.create_window(attributes) {
            Ok(window) => window,
            Err(error) => {
                log::error!("Failed to create window: {error:#}");
                event_loop.exit();
                return;
            }
        };

        let window_id = window.id();
        let (renderer, shader) = match Self::create_renderer(&window) {
            Ok(result) => result,
            Err(error) => {
                log::error!("Failed to create renderer: {error:#}");
                event_loop.exit();
                return;
            }
        };

        self.window = Some(window);
        self.window_id = Some(window_id);
        self.renderer = Some(renderer);
        self.shader = Some(shader);
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if Some(window_id) != self.window_id {
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.update_drawable_size(Size {
                        width: DevicePixels(size.width.max(1) as i32),
                        height: DevicePixels(size.height.max(1) as i32),
                    });
                }
            }
            WindowEvent::RedrawRequested => {
                self.draw_frame();
                if let Some(window) = self.window.as_ref() {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
}

fn main() -> anyhow::Result<()> {
    let event_loop = EventLoop::new()?;
    let mut app = AppState::new();
    event_loop.run_app(&mut app)?;
    Ok(())
}
