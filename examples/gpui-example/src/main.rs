#![cfg_attr(target_family = "wasm", no_main)]

use gpui::{
    App, Bounds, Context, Menu, MenuItem, MouseButton, SystemMenuType, Window, WindowBounds,
    WindowOptions, actions, div, prelude::*, px, rgb, shader_surface, size,
};
use gpui_platform::application;
use gpui_wgpu::{
    GlobalCustomShaderConfig, RenderPrimitive, ShaderSurfaceDraw, register_named_custom_shader,
    submit_render_primitive,
};

const SPINNING_CUBE_SHADER_WGSL: &str = r#"
struct Params {
    time: f32,
    width: f32,
    height: f32,
    _pad: f32,
    viewport_rect: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> params: Params;

struct VsOut {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VsOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );

    var out: VsOut;
    out.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    return out;
}

fn rot_x(a: f32) -> mat3x3<f32> {
    let c = cos(a);
    let s = sin(a);
    return mat3x3<f32>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, c, s),
        vec3<f32>(0.0, -s, c)
    );
}

fn rot_y(a: f32) -> mat3x3<f32> {
    let c = cos(a);
    let s = sin(a);
    return mat3x3<f32>(
        vec3<f32>(c, 0.0, -s),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(s, 0.0, c)
    );
}

fn sdf_box(p: vec3<f32>, b: vec3<f32>) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn map_scene(p: vec3<f32>, t: f32) -> f32 {
    let r = rot_y(t * 0.9) * rot_x(t * 0.6);
    let q = r * p;
    return sdf_box(q, vec3<f32>(0.55));
}

fn march(ro: vec3<f32>, rd: vec3<f32>, t: f32) -> f32 {
    var dist = 0.0;
    for (var i: i32 = 0; i < 96; i = i + 1) {
        let p = ro + rd * dist;
        let d = map_scene(p, t);
        if (d < 0.001) {
            return dist;
        }
        dist = dist + d;
        if (dist > 20.0) {
            break;
        }
    }
    return -1.0;
}

fn estimate_normal(p: vec3<f32>, t: f32) -> vec3<f32> {
    let e = 0.0015;
    let x = vec3<f32>(e, 0.0, 0.0);
    let y = vec3<f32>(0.0, e, 0.0);
    let z = vec3<f32>(0.0, 0.0, e);

    return normalize(vec3<f32>(
        map_scene(p + x, t) - map_scene(p - x, t),
        map_scene(p + y, t) - map_scene(p - y, t),
        map_scene(p + z, t) - map_scene(p - z, t)
    ));
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let resolution = vec2<f32>(max(params.width, 1.0), max(params.height, 1.0));
    let screen_uv = frag_coord.xy / resolution;

    let viewport_min = params.viewport_rect.xy;
    let viewport_max = params.viewport_rect.zw;

    if (
        screen_uv.x < viewport_min.x ||
        screen_uv.x > viewport_max.x ||
        screen_uv.y < viewport_min.y ||
        screen_uv.y > viewport_max.y
    ) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let viewport_uv = (screen_uv - viewport_min) / max(viewport_max - viewport_min, vec2<f32>(0.0001, 0.0001));
    var uv = viewport_uv * 2.0 - vec2<f32>(1.0, 1.0);
    let viewport_size = max((viewport_max - viewport_min) * resolution, vec2<f32>(1.0, 1.0));
    uv.x = uv.x * (viewport_size.x / viewport_size.y);

    let time = params.time;
    let ro = vec3<f32>(0.0, 0.0, -3.0);
    let rd = normalize(vec3<f32>(uv, 1.8));

    let hit = march(ro, rd, time);
    if (hit < 0.0) {
        let bg = vec3<f32>(0.05, 0.08, 0.13) + 0.09 * vec3<f32>(uv.y + 0.5);
        return vec4<f32>(bg, 1.0);
    }

    let p = ro + rd * hit;
    let n = estimate_normal(p, time);
    let light_dir = normalize(vec3<f32>(-0.4, 0.7, -0.5));
    let diffuse = max(dot(n, light_dir), 0.0);
    let rim = pow(1.0 - max(dot(n, -rd), 0.0), 2.5);

    let base = vec3<f32>(0.20, 0.75, 1.0);
    let color = base * (0.2 + 0.8 * diffuse) + rim * vec3<f32>(0.9, 0.82, 1.0);
    return vec4<f32>(color, 1.0);
}
"#;
const SPINNING_CUBE_SHADER_KEY: &str = "spinning_cube_custom_shader";

struct UiWithShaderApp {
    shader_enabled: bool,
}

impl Render for UiWithShaderApp {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        window.request_animation_frame();

        div()
            .size_full()
            .bg(rgb(0x070D14))
            .text_color(rgb(0xFFFFFF))
            .p_4()
            .flex()
            .gap_4()
            .child(
                div()
                    .w(px(380.0))
                    .h_full()
                    .rounded_lg()
                    .bg(rgb(0x0E1722))
                    .border_1()
                    .border_color(rgb(0x223347))
                    .p_4()
                    .flex()
                    .flex_col()
                    .gap_3()
                    .child(div().text_xl().child("Control Panel"))
                    .child(div().text_base().text_color(rgb(0xE7F2FF)).child(
                        "Text + UI should remain visible while shader animates on the right.",
                    ))
                    .child(div().mt_2().child(control_button(
                        if self.shader_enabled {
                            "Disable Shader"
                        } else {
                            "Enable Shader"
                        },
                        cx,
                        |this, _cx| {
                            this.shader_enabled = !this.shader_enabled;
                        },
                    )))
                    .child(
                        div()
                            .mt_2()
                            .rounded_md()
                            .bg(rgb(0x162232))
                            .border_1()
                            .border_color(rgb(0x2A3F57))
                            .px_3()
                            .py_2()
                            .child("Primary Button"),
                    )
                    .child(
                        div()
                            .rounded_md()
                            .bg(rgb(0x162232))
                            .border_1()
                            .border_color(rgb(0x2A3F57))
                            .px_3()
                            .py_2()
                            .child("Secondary Button"),
                    )
                    .child(
                        div()
                            .rounded_md()
                            .bg(rgb(0x111E2D))
                            .border_1()
                            .border_color(rgb(0x2A3F57))
                            .p_3()
                            .text_sm()
                            .text_color(rgb(0xFFFFFF))
                            .child(if self.shader_enabled {
                                "Status: shader enabled"
                            } else {
                                "Status: shader disabled"
                            }),
                    ),
            )
            .child(
                div()
                    .flex_1()
                    .h_full()
                    .rounded_lg()
                    .bg(rgb(0x0A121D))
                    .border_1()
                    .border_color(rgb(0x21354B))
                    .p_3()
                    .flex()
                    .flex_col()
                    .gap_2()
                    .child(
                        div()
                            .text_base()
                            .text_color(rgb(0xFFFFFF))
                            .child("Shader Viewport"),
                    )
                    .child(
                        div()
                            .text_sm()
                            .text_color(rgb(0xCFE2F5))
                            .child("Spinning cube is clipped to the shader surface bounds."),
                    )
                    .child(
                        div()
                            .flex_1()
                            .rounded_md()
                            .bg(rgb(0x09121C))
                            .border_1()
                            .border_color(rgb(0x2A3F57))
                            .child(cube_shader_surface(self.shader_enabled)),
                    )
            )
    }
}

fn configure_shader() {
    let mut config = GlobalCustomShaderConfig::default();
    config.label = Some(SPINNING_CUBE_SHADER_KEY.to_string());
    config.shader_source = SPINNING_CUBE_SHADER_WGSL.to_string();
    config.uniform_bytes = Some(vec![0; 32]);
    config.animate_uniforms_with_time = true;

    register_named_custom_shader(SPINNING_CUBE_SHADER_KEY, config);
}

fn cube_shader_surface(enabled: bool) -> impl IntoElement {
    shader_surface(move |bounds, window, _cx| {
        if !enabled {
            return;
        }

        let viewport = window.viewport_size();
        let viewport_width = f32::from(viewport.width).max(1.0);
        let viewport_height = f32::from(viewport.height).max(1.0);

        let origin_x = f32::from(bounds.origin.x);
        let origin_y = f32::from(bounds.origin.y);
        let width = f32::from(bounds.size.width);
        let height = f32::from(bounds.size.height);

        let min_x = (origin_x / viewport_width).clamp(0.0, 1.0);
        let min_y = (origin_y / viewport_height).clamp(0.0, 1.0);
        let max_x = ((origin_x + width) / viewport_width).clamp(0.0, 1.0);
        let max_y = ((origin_y + height) / viewport_height).clamp(0.0, 1.0);

        let uniform_values = [
            0.0f32,
            viewport_width,
            viewport_height,
            0.0,
            min_x,
            min_y,
            max_x,
            max_y,
        ];
        let uniform_bytes: Vec<u8> = uniform_values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect();

        submit_render_primitive(RenderPrimitive {
            draw: Some(ShaderSurfaceDraw {
                shader_key: SPINNING_CUBE_SHADER_KEY.to_string(),
                normalized_bounds: [min_x, min_y, max_x, max_y],
                uniform_bytes: Some(uniform_bytes),
                texture_key: None,
            }),
            texture_key: None,
            textures_2d: Vec::new(),
            textures_nv12: Vec::new(),
            textures_3d: Vec::new(),
        });
    })
    .size_full()
}

fn control_button(
    label: &str,
    cx: &mut Context<UiWithShaderApp>,
    on_click: impl Fn(&mut UiWithShaderApp, &mut Context<UiWithShaderApp>) + 'static,
) -> impl IntoElement {
    let label: gpui::SharedString = label.to_string().into();

    div()
        .rounded_md()
        .bg(rgb(0x162232))
        .border_1()
        .border_color(rgb(0x2A3F57))
        .px_3()
        .py_2()
        .cursor_pointer()
        .on_mouse_down(
            MouseButton::Left,
            cx.listener(move |this, _event, _window, cx| {
                on_click(this, cx);
            }),
        )
        .child(label)
}

fn run_app() {
    configure_shader();

    application().run(|cx: &mut App| {
        cx.on_action(quit);
        set_app_menus(cx);

        let bounds = Bounds::centered(None, size(px(1100.0), px(700.0)), cx);

        cx.open_window(
            WindowOptions {
                titlebar: Some(gpui::TitlebarOptions {
                    title: Some("wGPUI Text + Shader Check".into()),
                    appears_transparent: false,
                    ..Default::default()
                }),
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_, cx| {
                cx.new(|_| UiWithShaderApp {
                    shader_enabled: true,
                })
            },
        )
        .expect("failed to open GPUI window");

        cx.activate(true);
    });
}

fn set_app_menus(cx: &mut App) {
    cx.set_menus(vec![Menu {
        name: "gpui_example".into(),
        items: vec![
            MenuItem::os_submenu("Services", SystemMenuType::Services),
            MenuItem::separator(),
            MenuItem::action("Quit", Quit),
        ],
    }]);
}

actions!(gpui_example, [Quit]);

fn quit(_: &Quit, cx: &mut App) {
    cx.quit();
}

#[cfg(not(target_family = "wasm"))]
fn main() {
    run_app();
}

#[cfg(target_family = "wasm")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() {
    gpui_platform::web_init();
    run_app();
}
