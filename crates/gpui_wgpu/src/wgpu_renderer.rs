use crate::{WgpuAtlas, WgpuContext};
use bytemuck::{Pod, Zeroable};
use gpui::{
    AtlasTextureId, Background, Bounds, DevicePixels, GpuSpecs, MonochromeSprite,
    PaintShaderSurface, Path, Point, PolychromeSprite, PrimitiveBatch, Quad, ScaledPixels, Scene,
    ShaderSurfaceDraw, Shadow, Size, SubpixelSprite, Underline, get_gamma_correction_ratios,
};
use log::warn;
#[cfg(not(target_family = "wasm"))]
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::any::{Any, TypeId};
use std::borrow::Cow;
use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GlobalParams {
    viewport_size: [f32; 2],
    premultiplied_alpha: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PodBounds {
    origin: [f32; 2],
    size: [f32; 2],
}

impl From<Bounds<ScaledPixels>> for PodBounds {
    fn from(bounds: Bounds<ScaledPixels>) -> Self {
        Self {
            origin: [bounds.origin.x.0, bounds.origin.y.0],
            size: [bounds.size.width.0, bounds.size.height.0],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SurfaceParams {
    bounds: PodBounds,
    content_mask: PodBounds,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GammaParams {
    gamma_ratios: [f32; 4],
    grayscale_enhanced_contrast: f32,
    subpixel_enhanced_contrast: f32,
    _pad: [f32; 2],
}

#[derive(Clone, Debug)]
#[repr(C)]
struct PathSprite {
    bounds: Bounds<ScaledPixels>,
}

#[derive(Clone, Debug)]
#[repr(C)]
struct PathRasterizationVertex {
    xy_position: Point<ScaledPixels>,
    st_position: Point<f32>,
    color: Background,
    bounds: Bounds<ScaledPixels>,
}

pub struct WgpuSurfaceConfig {
    pub size: Size<DevicePixels>,
    pub transparent: bool,
}

const DEFAULT_CUSTOM_SHADER_WGSL: &str = r#"
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );
    let xy = positions[vertex_index];
    return vec4<f32>(xy, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
"#;

pub struct CustomShaderDescriptor<'a> {
    pub label: Option<&'a str>,
    pub shader_source: &'a str,
    pub vertex_entry: &'a str,
    pub fragment_entry: &'a str,
    pub primitive_topology: wgpu::PrimitiveTopology,
    pub vertex_count: u32,
    pub instance_count: u32,
    pub uniform_bytes: Option<&'a [u8]>,
    pub blend_state: Option<wgpu::BlendState>,
    pub texture_mode: CustomShaderTextureMode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CustomShaderTextureMode {
    #[default]
    None,
    VideoPlayerFull,
}

#[derive(Clone)]
pub struct GlobalCustomShaderConfig {
    pub label: Option<String>,
    pub shader_source: String,
    pub vertex_entry: String,
    pub fragment_entry: String,
    pub primitive_topology: wgpu::PrimitiveTopology,
    pub vertex_count: u32,
    pub instance_count: u32,
    pub uniform_bytes: Option<Vec<u8>>,
    pub blend_state: Option<wgpu::BlendState>,
    pub animate_uniforms_with_time: bool,
    pub texture_mode: CustomShaderTextureMode,
}

impl Default for GlobalCustomShaderConfig {
    fn default() -> Self {
        Self {
            label: Some("gpui_custom_shader".to_string()),
            shader_source: DEFAULT_CUSTOM_SHADER_WGSL.to_string(),
            vertex_entry: "vs_main".to_string(),
            fragment_entry: "fs_main".to_string(),
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            vertex_count: 3,
            instance_count: 1,
            uniform_bytes: None,
            blend_state: None,
            animate_uniforms_with_time: false,
            texture_mode: CustomShaderTextureMode::None,
        }
    }
}

static GLOBAL_CUSTOM_SHADER_CONFIG: OnceLock<Mutex<Option<GlobalCustomShaderConfig>>> =
    OnceLock::new();
static GLOBAL_CUSTOM_SHADER_RUNTIME: OnceLock<Mutex<GlobalCustomShaderRuntime>> = OnceLock::new();
static NAMED_CUSTOM_SHADER_CONFIGS: OnceLock<Mutex<HashMap<String, GlobalCustomShaderConfig>>> =
    OnceLock::new();
static SHADER_SURFACE_DRAWS: OnceLock<Mutex<Vec<ShaderSurfaceDraw>>> = OnceLock::new();
static LAST_SHADER_SURFACE_DRAWS: OnceLock<Mutex<Vec<ShaderSurfaceDraw>>> = OnceLock::new();
static SHADER_SURFACE_TEXTURE_UPDATES: OnceLock<Mutex<HashMap<String, ShaderSurfaceTextureData>>> =
    OnceLock::new();
static SHADER_SURFACE_TEXTURE_UPDATES_NV12: OnceLock<
    Mutex<HashMap<String, ShaderSurfaceTexturePairData>>,
> = OnceLock::new();
static SHADER_SURFACE_TEXTURE3D_UPDATES: OnceLock<Mutex<HashMap<String, ShaderSurfaceTexture3dData>>> =
    OnceLock::new();
static SHADER_SURFACE_TEXTURE_UPDATES_CAPTION: OnceLock<
    Mutex<HashMap<String, ShaderSurfaceTextureData>>,
> = OnceLock::new();
static SHADER_SURFACE_TEXTURE_UPDATES_RGBA_VIDEO: OnceLock<
    Mutex<HashMap<String, ShaderSurfaceTextureData>>,
> = OnceLock::new();
static SHADER_SURFACE_TEXTURE_UPDATES_BROLL: OnceLock<
    Mutex<HashMap<String, ShaderSurfaceTextureData>>,
> = OnceLock::new();
static SHADER_SURFACE_TEXTURE_UPDATES_BROLL_NV12: OnceLock<
    Mutex<HashMap<String, ShaderSurfaceTexturePairData>>,
> = OnceLock::new();
static SHADER_SURFACE_TEXTURE_UPDATES_BROLL_LUT3D: OnceLock<
    Mutex<HashMap<String, ShaderSurfaceTexture3dData>>,
> = OnceLock::new();
static PRIMITIVE_PIPELINES: OnceLock<Mutex<HashMap<TypeId, Box<dyn Any + Send + Sync>>>> =
    OnceLock::new();
static PRIMITIVE_RENDER_CONTEXT: OnceLock<Mutex<Option<PrimitiveRenderContext>>> = OnceLock::new();

fn global_custom_shader_config() -> &'static Mutex<Option<GlobalCustomShaderConfig>> {
    GLOBAL_CUSTOM_SHADER_CONFIG.get_or_init(|| Mutex::new(None))
}

#[derive(Clone)]
struct GlobalCustomShaderRuntime {
    enabled: bool,
    paused: bool,
    time_scale: f32,
    uniform_bytes: Option<Vec<u8>>,
}

impl Default for GlobalCustomShaderRuntime {
    fn default() -> Self {
        Self {
            enabled: true,
            paused: false,
            time_scale: 1.0,
            uniform_bytes: None,
        }
    }
}

fn global_custom_shader_runtime() -> &'static Mutex<GlobalCustomShaderRuntime> {
    GLOBAL_CUSTOM_SHADER_RUNTIME.get_or_init(|| Mutex::new(GlobalCustomShaderRuntime::default()))
}

fn named_custom_shader_configs() -> &'static Mutex<HashMap<String, GlobalCustomShaderConfig>> {
    NAMED_CUSTOM_SHADER_CONFIGS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn render_primitive_draws() -> &'static Mutex<Vec<ShaderSurfaceDraw>> {
    SHADER_SURFACE_DRAWS.get_or_init(|| Mutex::new(Vec::new()))
}

fn render_primitive_texture_updates() -> &'static Mutex<HashMap<String, ShaderSurfaceTextureData>> {
    SHADER_SURFACE_TEXTURE_UPDATES.get_or_init(|| Mutex::new(HashMap::new()))
}

fn render_primitive_texture_updates_pair()
-> &'static Mutex<HashMap<String, ShaderSurfaceTexturePairData>> {
    SHADER_SURFACE_TEXTURE_UPDATES_NV12.get_or_init(|| Mutex::new(HashMap::new()))
}

fn render_primitive_texture_3d_updates() -> &'static Mutex<HashMap<String, ShaderSurfaceTexture3dData>> {
    SHADER_SURFACE_TEXTURE3D_UPDATES.get_or_init(|| Mutex::new(HashMap::new()))
}

fn render_primitive_texture_caption_updates() -> &'static Mutex<HashMap<String, ShaderSurfaceTextureData>> {
    SHADER_SURFACE_TEXTURE_UPDATES_CAPTION.get_or_init(|| Mutex::new(HashMap::new()))
}

fn render_primitive_texture_rgba_video_updates()
-> &'static Mutex<HashMap<String, ShaderSurfaceTextureData>> {
    SHADER_SURFACE_TEXTURE_UPDATES_RGBA_VIDEO.get_or_init(|| Mutex::new(HashMap::new()))
}

fn render_primitive_texture_broll_updates() -> &'static Mutex<HashMap<String, ShaderSurfaceTextureData>> {
    SHADER_SURFACE_TEXTURE_UPDATES_BROLL.get_or_init(|| Mutex::new(HashMap::new()))
}

fn render_primitive_texture_broll_nv12_updates()
-> &'static Mutex<HashMap<String, ShaderSurfaceTexturePairData>> {
    SHADER_SURFACE_TEXTURE_UPDATES_BROLL_NV12.get_or_init(|| Mutex::new(HashMap::new()))
}

fn render_primitive_texture_broll_lut3d_updates()
-> &'static Mutex<HashMap<String, ShaderSurfaceTexture3dData>> {
    SHADER_SURFACE_TEXTURE_UPDATES_BROLL_LUT3D.get_or_init(|| Mutex::new(HashMap::new()))
}

fn primitive_pipelines() -> &'static Mutex<HashMap<TypeId, Box<dyn Any + Send + Sync>>> {
    PRIMITIVE_PIPELINES.get_or_init(|| Mutex::new(HashMap::new()))
}

#[derive(Clone)]
struct PrimitiveRenderContext {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    format: wgpu::TextureFormat,
}

fn primitive_render_context() -> &'static Mutex<Option<PrimitiveRenderContext>> {
    PRIMITIVE_RENDER_CONTEXT.get_or_init(|| Mutex::new(None))
}


fn set_primitive_render_context(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    format: wgpu::TextureFormat,
) {
    if let Ok(mut slot) = primitive_render_context().lock() {
        *slot = Some(PrimitiveRenderContext {
            device,
            queue,
            format,
        });
    }
}

fn last_render_primitive_draws() -> &'static Mutex<Vec<ShaderSurfaceDraw>> {
    LAST_SHADER_SURFACE_DRAWS.get_or_init(|| Mutex::new(Vec::new()))
}

fn take_render_primitive_draws() -> Vec<ShaderSurfaceDraw> {
    let draws = render_primitive_draws()
        .lock()
        .map(|mut draws| std::mem::take(&mut *draws))
        .unwrap_or_default();

    if draws.is_empty() {
        // No new draws were queued (present-only frame without a draw pass).
        // Reuse the previous frame's draws to avoid a black flash.
        return last_render_primitive_draws()
            .lock()
            .map(|last| last.clone())
            .unwrap_or_default();
    }

    // Stash a copy for potential reuse on the next present-only frame.
    if let Ok(mut last) = last_render_primitive_draws().lock() {
        *last = draws.clone();
    }

    draws
}

fn take_render_primitive_texture_updates() -> HashMap<String, ShaderSurfaceTextureData> {
    render_primitive_texture_updates()
        .lock()
        .map(|mut updates| std::mem::take(&mut *updates))
        .unwrap_or_default()
}

fn take_render_primitive_texture_updates_pair() -> HashMap<String, ShaderSurfaceTexturePairData> {
    render_primitive_texture_updates_pair()
        .lock()
        .map(|mut updates| std::mem::take(&mut *updates))
        .unwrap_or_default()
}

fn take_render_primitive_texture_3d_updates() -> HashMap<String, ShaderSurfaceTexture3dData> {
    render_primitive_texture_3d_updates()
        .lock()
        .map(|mut updates| std::mem::take(&mut *updates))
        .unwrap_or_default()
}

fn take_render_primitive_texture_caption_updates() -> HashMap<String, ShaderSurfaceTextureData> {
    render_primitive_texture_caption_updates()
        .lock()
        .map(|mut updates| std::mem::take(&mut *updates))
        .unwrap_or_default()
}

fn take_render_primitive_texture_rgba_video_updates() -> HashMap<String, ShaderSurfaceTextureData> {
    render_primitive_texture_rgba_video_updates()
        .lock()
        .map(|mut updates| std::mem::take(&mut *updates))
        .unwrap_or_default()
}

fn take_render_primitive_texture_broll_updates() -> HashMap<String, ShaderSurfaceTextureData> {
    render_primitive_texture_broll_updates()
        .lock()
        .map(|mut updates| std::mem::take(&mut *updates))
        .unwrap_or_default()
}

fn take_render_primitive_texture_broll_nv12_updates() -> HashMap<String, ShaderSurfaceTexturePairData> {
    render_primitive_texture_broll_nv12_updates()
        .lock()
        .map(|mut updates| std::mem::take(&mut *updates))
        .unwrap_or_default()
}

fn take_render_primitive_texture_broll_lut3d_updates() -> HashMap<String, ShaderSurfaceTexture3dData> {
    render_primitive_texture_broll_lut3d_updates()
        .lock()
        .map(|mut updates| std::mem::take(&mut *updates))
        .unwrap_or_default()
}

pub fn set_global_custom_shader(config: Option<GlobalCustomShaderConfig>) {
    if let Ok(mut slot) = global_custom_shader_config().lock() {
        *slot = config;
    }
}

pub fn set_global_custom_shader_enabled(enabled: bool) {
    if let Ok(mut runtime) = global_custom_shader_runtime().lock() {
        runtime.enabled = enabled;
    }
}

pub fn set_global_custom_shader_paused(paused: bool) {
    if let Ok(mut runtime) = global_custom_shader_runtime().lock() {
        runtime.paused = paused;
    }
}

pub fn set_global_custom_shader_time_scale(time_scale: f32) {
    if let Ok(mut runtime) = global_custom_shader_runtime().lock() {
        runtime.time_scale = time_scale.max(0.0);
    }
}

pub fn set_global_custom_shader_uniform_bytes(uniform_bytes: Option<Vec<u8>>) {
    if let Ok(mut runtime) = global_custom_shader_runtime().lock() {
        runtime.uniform_bytes = uniform_bytes;
    }
}

pub fn register_named_custom_shader(
    shader_key: impl Into<String>,
    config: GlobalCustomShaderConfig,
) {
    if let Ok(mut configs) = named_custom_shader_configs().lock() {
        configs.insert(shader_key.into(), config);
    }
}

pub fn unregister_named_custom_shader(shader_key: &str) {
    if let Ok(mut configs) = named_custom_shader_configs().lock() {
        configs.remove(shader_key);
    }
}

fn queue_render_primitive_draw(draw: ShaderSurfaceDraw) {
    if let Ok(mut draws) = render_primitive_draws().lock() {
        draws.push(draw);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShaderSurfaceTextureFormat {
    Bgra8Unorm,
    Rgba8Unorm,
}

#[derive(Clone, Debug)]
struct ShaderSurfaceTextureData {
    pub key: String,
    pub width: u32,
    pub height: u32,
    pub bytes: Vec<u8>,
    pub format: ShaderSurfaceTextureFormat,
}

#[derive(Clone, Debug)]
struct ShaderSurfaceTexturePairData {
    pub key: String,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub data: Vec<u8>,
}

#[derive(Clone, Debug)]
struct ShaderSurfaceTexture3dData {
    pub key: String,
    pub size: u32,
    pub data: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct RenderPrimitiveTexture2d {
    pub binding: u32,
    pub width: u32,
    pub height: u32,
    pub bytes: Vec<u8>,
    pub format: ShaderSurfaceTextureFormat,
}

impl RenderPrimitiveTexture2d {
    pub fn new(
        binding: u32,
        width: u32,
        height: u32,
        bytes: Vec<u8>,
        format: ShaderSurfaceTextureFormat,
    ) -> Self {
        Self {
            binding,
            width,
            height,
            bytes,
            format,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RenderPrimitiveTexturePair {
    pub y_binding: u32,
    pub uv_binding: u32,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub data: Vec<u8>,
}

impl RenderPrimitiveTexturePair {
    pub fn new(
        y_binding: u32,
        uv_binding: u32,
        width: u32,
        height: u32,
        stride: u32,
        data: Vec<u8>,
    ) -> Self {
        Self {
            y_binding,
            uv_binding,
            width,
            height,
            stride,
            data,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RenderPrimitiveTexture3d {
    pub binding: u32,
    pub size: u32,
    pub data: Vec<u8>,
}

impl RenderPrimitiveTexture3d {
    pub fn new(binding: u32, size: u32, data: Vec<u8>) -> Self {
        Self {
            binding,
            size,
            data,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct RenderPrimitive {
    pub draw: Option<ShaderSurfaceDraw>,
    pub texture_key: Option<String>,
    pub textures_2d: Vec<RenderPrimitiveTexture2d>,
    pub textures_nv12: Vec<RenderPrimitiveTexturePair>,
    pub textures_3d: Vec<RenderPrimitiveTexture3d>,
}

impl RenderPrimitive {
    pub fn new(texture_key: impl Into<String>) -> Self {
        Self {
            draw: None,
            texture_key: Some(texture_key.into()),
            textures_2d: Vec::new(),
            textures_nv12: Vec::new(),
            textures_3d: Vec::new(),
        }
    }

    pub fn draw(mut self, draw: ShaderSurfaceDraw) -> Self {
        self.draw = Some(draw);
        self
    }

    pub fn with_texture_2d(mut self, texture: RenderPrimitiveTexture2d) -> Self {
        self.textures_2d.push(texture);
        self
    }

    pub fn with_texture_nv12(mut self, texture: RenderPrimitiveTexturePair) -> Self {
        self.textures_nv12.push(texture);
        self
    }

    pub fn with_texture_3d(mut self, texture: RenderPrimitiveTexture3d) -> Self {
        self.textures_3d.push(texture);
        self
    }
}

pub trait PrimitivePipeline: Send + Sync + 'static {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self
    where
        Self: Sized;
}

#[derive(Clone, Copy, Debug)]
pub struct PrimitiveBounds {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct PrimitiveViewport {
    pub width: f32,
    pub height: f32,
}

pub trait Primitive: std::fmt::Debug + Send + Sync + 'static {
    type Pipeline: PrimitivePipeline;

    fn prepare(
        &self,
        pipeline: &mut Self::Pipeline,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bounds: &PrimitiveBounds,
        viewport: &PrimitiveViewport,
    );
    fn render(
        &self,
        pipeline: &Self::Pipeline,
        bounds: &PrimitiveBounds,
        viewport: &PrimitiveViewport,
    ) -> RenderPrimitive;
}

pub fn draw_primitive<P: Primitive>(
    bounds: PrimitiveBounds,
    viewport: PrimitiveViewport,
    primitive: P,
) {
    let context = match primitive_render_context().lock() {
        Ok(guard) => guard.clone(),
        Err(_) => None,
    };
    let Some(context) = context else {
        return;
    };

    let type_id = TypeId::of::<P::Pipeline>();
    let mut pipelines = match primitive_pipelines().lock() {
        Ok(guard) => guard,
        Err(_) => return,
    };

    pipelines
        .entry(type_id)
        .or_insert_with(|| {
            Box::new(<P::Pipeline as PrimitivePipeline>::new(
                &context.device,
                &context.queue,
                context.format,
            ))
        });

    {
        let Some(pipeline_any) = pipelines.get_mut(&type_id) else {
            return;
        };
        let Some(pipeline) = pipeline_any.downcast_mut::<P::Pipeline>() else {
            return;
        };
        primitive.prepare(
            pipeline,
            &context.device,
            &context.queue,
            &bounds,
            &viewport,
        );
    }

    let render_primitive = {
        let Some(pipeline_any) = pipelines.get(&type_id) else {
            return;
        };
        let Some(pipeline) = pipeline_any.downcast_ref::<P::Pipeline>() else {
            return;
        };
        primitive.render(pipeline, &bounds, &viewport)
    };

    drop(pipelines);
    submit_render_primitive(render_primitive);
}

fn upsert_shader_surface_texture(texture: ShaderSurfaceTextureData) {
    if let Ok(mut updates) = render_primitive_texture_updates().lock() {
        updates.insert(texture.key.clone(), texture);
    }
}

fn upsert_shader_surface_texture_pair(texture: ShaderSurfaceTexturePairData) {
    if let Ok(mut updates) = render_primitive_texture_updates_pair().lock() {
        updates.insert(texture.key.clone(), texture);
    }
}

fn upsert_shader_surface_texture_3d(lut: ShaderSurfaceTexture3dData) {
    if let Ok(mut updates) = render_primitive_texture_3d_updates().lock() {
        updates.insert(lut.key.clone(), lut);
    }
}

fn upsert_shader_surface_caption_texture(texture: ShaderSurfaceTextureData) {
    if let Ok(mut updates) = render_primitive_texture_caption_updates().lock() {
        updates.insert(texture.key.clone(), texture);
    }
}

fn upsert_shader_surface_rgba_video_texture(texture: ShaderSurfaceTextureData) {
    if let Ok(mut updates) = render_primitive_texture_rgba_video_updates().lock() {
        updates.insert(texture.key.clone(), texture);
    }
}

fn upsert_shader_surface_broll_texture(texture: ShaderSurfaceTextureData) {
    if let Ok(mut updates) = render_primitive_texture_broll_updates().lock() {
        updates.insert(texture.key.clone(), texture);
    }
}

fn upsert_shader_surface_broll_texture_pair(texture: ShaderSurfaceTexturePairData) {
    if let Ok(mut updates) = render_primitive_texture_broll_nv12_updates().lock() {
        updates.insert(texture.key.clone(), texture);
    }
}

fn upsert_shader_surface_broll_texture_3d(lut: ShaderSurfaceTexture3dData) {
    if let Ok(mut updates) = render_primitive_texture_broll_lut3d_updates().lock() {
        updates.insert(lut.key.clone(), lut);
    }
}

pub fn submit_render_primitive(primitive: RenderPrimitive) {
    if let Some(draw) = primitive.draw {
        queue_render_primitive_draw(draw);
    }

    let Some(texture_key) = primitive.texture_key else {
        return;
    };

    for tex in primitive.textures_2d {
        let update = ShaderSurfaceTextureData {
            key: texture_key.clone(),
            width: tex.width,
            height: tex.height,
            bytes: tex.bytes,
            format: tex.format,
        };
        match tex.binding {
            6 => upsert_shader_surface_caption_texture(update),
            8 => upsert_shader_surface_rgba_video_texture(update),
            9 => upsert_shader_surface_broll_texture(update),
            _ => upsert_shader_surface_texture(update),
        }
    }

    for tex in primitive.textures_nv12 {
        let update = ShaderSurfaceTexturePairData {
            key: texture_key.clone(),
            width: tex.width,
            height: tex.height,
            stride: tex.stride,
            data: tex.data,
        };
        match (tex.y_binding, tex.uv_binding) {
            (13, 14) => upsert_shader_surface_broll_texture_pair(update),
            _ => upsert_shader_surface_texture_pair(update),
        }
    }

    for tex in primitive.textures_3d {
        let update = ShaderSurfaceTexture3dData {
            key: texture_key.clone(),
            size: tex.size,
            data: tex.data,
        };
        match tex.binding {
            11 => upsert_shader_surface_broll_texture_3d(update),
            _ => upsert_shader_surface_texture_3d(update),
        }
    }
}

/// Clear the cached last-frame draws so that no shader surface is rendered
/// on the next frame. Without this, the renderer reuses the previous frame's
/// draws on present-only frames, which causes stale video surfaces to persist
/// after navigating away from a page that uses shader surfaces.
pub fn clear_last_render_primitive_draws() {
    if let Ok(mut last) = last_render_primitive_draws().lock() {
        last.clear();
    }
}

pub fn remove_render_primitive_texture(key: &str) {
    if let Ok(mut updates) = render_primitive_texture_updates().lock() {
        updates.remove(key);
    }
    if let Ok(mut updates) = render_primitive_texture_updates_pair().lock() {
        updates.remove(key);
    }
    if let Ok(mut updates) = render_primitive_texture_3d_updates().lock() {
        updates.remove(key);
    }
    if let Ok(mut updates) = render_primitive_texture_caption_updates().lock() {
        updates.remove(key);
    }
    if let Ok(mut updates) = render_primitive_texture_rgba_video_updates().lock() {
        updates.remove(key);
    }
    if let Ok(mut updates) = render_primitive_texture_broll_updates().lock() {
        updates.remove(key);
    }
    if let Ok(mut updates) = render_primitive_texture_broll_nv12_updates().lock() {
        updates.remove(key);
    }
    if let Ok(mut updates) = render_primitive_texture_broll_lut3d_updates().lock() {
        updates.remove(key);
    }
}

impl<'a> Default for CustomShaderDescriptor<'a> {
    fn default() -> Self {
        Self {
            label: Some("gpui_custom_shader"),
            shader_source: DEFAULT_CUSTOM_SHADER_WGSL,
            vertex_entry: "vs_main",
            fragment_entry: "fs_main",
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            vertex_count: 3,
            instance_count: 1,
            uniform_bytes: None,
            blend_state: None,
            texture_mode: CustomShaderTextureMode::None,
        }
    }
}

pub struct WgpuCustomShader {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: Option<wgpu::Buffer>,
    bind_group_layout: Option<wgpu::BindGroupLayout>,
    uniform_size: Option<NonZeroU64>,
    texture_mode: CustomShaderTextureMode,
    vertex_count: u32,
    instance_count: u32,
}

struct NamedCustomShader {
    shader: WgpuCustomShader,
    uniform_template: Option<Vec<u8>>,
    animate_uniforms_with_time: bool,
}

struct ShaderSurfaceTextureGpu {
    texture_primary: wgpu::Texture,
    view_primary: wgpu::TextureView,
    texture_secondary: Option<wgpu::Texture>,
    view_secondary: Option<wgpu::TextureView>,
    mode: CustomShaderTextureMode,
    width: u32,
    height: u32,
    texture_3d: Option<wgpu::Texture>,
    view_3d: Option<wgpu::TextureView>,
    texture_caption: Option<wgpu::Texture>,
    view_caption: Option<wgpu::TextureView>,
    texture_rgba: Option<wgpu::Texture>,
    view_rgba: Option<wgpu::TextureView>,
    texture_broll: Option<wgpu::Texture>,
    view_broll: Option<wgpu::TextureView>,
    texture_broll_y: Option<wgpu::Texture>,
    view_broll_y: Option<wgpu::TextureView>,
    texture_broll_uv: Option<wgpu::Texture>,
    view_broll_uv: Option<wgpu::TextureView>,
    texture_broll_3d: Option<wgpu::Texture>,
    view_broll_3d: Option<wgpu::TextureView>,
}

struct WgpuPipelines {
    quads: wgpu::RenderPipeline,
    shadows: wgpu::RenderPipeline,
    path_rasterization: wgpu::RenderPipeline,
    paths: wgpu::RenderPipeline,
    underlines: wgpu::RenderPipeline,
    mono_sprites: wgpu::RenderPipeline,
    subpixel_sprites: Option<wgpu::RenderPipeline>,
    poly_sprites: wgpu::RenderPipeline,
    #[allow(dead_code)]
    surfaces: wgpu::RenderPipeline,
}

struct WgpuBindGroupLayouts {
    globals: wgpu::BindGroupLayout,
    instances: wgpu::BindGroupLayout,
    instances_with_texture: wgpu::BindGroupLayout,
    surfaces: wgpu::BindGroupLayout,
}

pub struct WgpuRenderer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    pipelines: WgpuPipelines,
    bind_group_layouts: WgpuBindGroupLayouts,
    atlas: Arc<WgpuAtlas>,
    atlas_sampler: wgpu::Sampler,
    globals_buffer: wgpu::Buffer,
    path_globals_offset: u64,
    gamma_offset: u64,
    globals_bind_group: wgpu::BindGroup,
    path_globals_bind_group: wgpu::BindGroup,
    instance_buffer: wgpu::Buffer,
    instance_buffer_capacity: u64,
    max_buffer_size: u64,
    storage_buffer_alignment: u64,
    path_intermediate_texture: Option<wgpu::Texture>,
    path_intermediate_view: Option<wgpu::TextureView>,
    path_msaa_texture: Option<wgpu::Texture>,
    path_msaa_view: Option<wgpu::TextureView>,
    rendering_params: RenderingParameters,
    dual_source_blending: bool,
    adapter_info: wgpu::AdapterInfo,
    transparent_alpha_mode: wgpu::CompositeAlphaMode,
    opaque_alpha_mode: wgpu::CompositeAlphaMode,
    max_texture_size: u32,
    global_custom_shader: Option<WgpuCustomShader>,
    global_custom_shader_init_failed: bool,
    global_custom_shader_uniform_template: Option<Vec<u8>>,
    global_custom_shader_animate_uniforms: bool,
    global_custom_shader_last_tick: Instant,
    global_custom_shader_time_seconds: f32,
    named_custom_shaders: HashMap<String, NamedCustomShader>,
    shader_surface_textures: HashMap<String, ShaderSurfaceTextureGpu>,
    custom_shader_fallback_texture_view: wgpu::TextureView,
    custom_shader_fallback_view_3d: wgpu::TextureView,
}

impl WgpuRenderer {
    /// Creates a new WgpuRenderer from raw window handles.
    ///
    /// # Safety
    /// The caller must ensure that the window handle remains valid for the lifetime
    /// of the returned renderer.
    #[cfg(not(target_family = "wasm"))]
    pub fn new<W: HasWindowHandle + HasDisplayHandle>(
        gpu_context: &mut Option<WgpuContext>,
        window: &W,
        config: WgpuSurfaceConfig,
    ) -> anyhow::Result<Self> {
        let window_handle = window
            .window_handle()
            .map_err(|e| anyhow::anyhow!("Failed to get window handle: {e}"))?;
        let display_handle = window
            .display_handle()
            .map_err(|e| anyhow::anyhow!("Failed to get display handle: {e}"))?;

        let target = wgpu::SurfaceTargetUnsafe::RawHandle {
            raw_display_handle: display_handle.as_raw(),
            raw_window_handle: window_handle.as_raw(),
        };

        // Use the existing context's instance if available, otherwise create a new one.
        // The surface must be created with the same instance that will be used for
        // adapter selection, otherwise wgpu will panic.
        let instance = gpu_context
            .as_ref()
            .map(|ctx| ctx.instance.clone())
            .unwrap_or_else(WgpuContext::instance);

        // Safety: The caller guarantees that the window handle is valid for the
        // lifetime of this renderer. In practice, the RawWindow struct is created
        // from the native window handles and the surface is dropped before the window.
        let surface = unsafe {
            instance
                .create_surface_unsafe(target)
                .map_err(|e| anyhow::anyhow!("Failed to create surface: {e}"))?
        };

        let context = match gpu_context {
            Some(context) => {
                context.check_compatible_with_surface(&surface)?;
                context
            }
            None => gpu_context.insert(WgpuContext::new(instance, &surface)?),
        };

        Self::new_with_surface(context, surface, config)
    }

    #[cfg(target_family = "wasm")]
    pub fn new_from_canvas(
        context: &WgpuContext,
        canvas: &web_sys::HtmlCanvasElement,
        config: WgpuSurfaceConfig,
    ) -> anyhow::Result<Self> {
        let surface = context
            .instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .map_err(|e| anyhow::anyhow!("Failed to create surface: {e}"))?;
        Self::new_with_surface(context, surface, config)
    }

    pub fn new_with_surface(
        context: &WgpuContext,
        surface: wgpu::Surface<'static>,
        config: WgpuSurfaceConfig,
    ) -> anyhow::Result<Self> {
        let surface_caps = surface.get_capabilities(&context.adapter);
        let preferred_formats = [
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Rgba8Unorm,
        ];
        let surface_format = preferred_formats
            .iter()
            .find(|f| surface_caps.formats.contains(f))
            .copied()
            .or_else(|| surface_caps.formats.iter().find(|f| !f.is_srgb()).copied())
            .or_else(|| surface_caps.formats.first().copied())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Surface reports no supported texture formats for adapter {:?}",
                    context.adapter.get_info().name
                )
            })?;

        let pick_alpha_mode =
            |preferences: &[wgpu::CompositeAlphaMode]| -> anyhow::Result<wgpu::CompositeAlphaMode> {
                preferences
                    .iter()
                    .find(|p| surface_caps.alpha_modes.contains(p))
                    .copied()
                    .or_else(|| surface_caps.alpha_modes.first().copied())
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "Surface reports no supported alpha modes for adapter {:?}",
                            context.adapter.get_info().name
                        )
                    })
            };

        let transparent_alpha_mode = pick_alpha_mode(&[
            wgpu::CompositeAlphaMode::PreMultiplied,
            wgpu::CompositeAlphaMode::Inherit,
        ])?;

        let opaque_alpha_mode = pick_alpha_mode(&[
            wgpu::CompositeAlphaMode::Opaque,
            wgpu::CompositeAlphaMode::Inherit,
        ])?;

        let alpha_mode = if config.transparent {
            transparent_alpha_mode
        } else {
            opaque_alpha_mode
        };

        let device = Arc::clone(&context.device);
        let max_texture_size = device.limits().max_texture_dimension_2d;

        let requested_width = config.size.width.0 as u32;
        let requested_height = config.size.height.0 as u32;
        let clamped_width = requested_width.min(max_texture_size);
        let clamped_height = requested_height.min(max_texture_size);

        if clamped_width != requested_width || clamped_height != requested_height {
            warn!(
                "Requested surface size ({}, {}) exceeds maximum texture dimension {}. \
                 Clamping to ({}, {}). Window content may not fill the entire window.",
                requested_width, requested_height, max_texture_size, clamped_width, clamped_height
            );
        }

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: clamped_width.max(1),
            height: clamped_height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode,
            view_formats: vec![],
        };
        surface.configure(&context.device, &surface_config);

        let queue = Arc::clone(&context.queue);
        let dual_source_blending = context.supports_dual_source_blending();

        let rendering_params = RenderingParameters::new(&context.adapter, surface_format);
        let bind_group_layouts = Self::create_bind_group_layouts(&device);
        let pipelines = Self::create_pipelines(
            &device,
            &bind_group_layouts,
            surface_format,
            alpha_mode,
            rendering_params.path_sample_count,
            dual_source_blending,
        );

        let atlas = Arc::new(WgpuAtlas::new(Arc::clone(&device), Arc::clone(&queue)));
        let atlas_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("atlas_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let custom_shader_fallback_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("custom_shader_fallback_texture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &custom_shader_fallback_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[0, 0, 0, 0],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let custom_shader_fallback_texture_view =
            custom_shader_fallback_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let custom_shader_fallback_3d = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("custom_shader_fallback_3d"),
            size: wgpu::Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 2,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let identity_lut_data: [u8; 32] = [
            0, 0, 0, 255, 255, 0, 0, 255, 0, 255, 0, 255, 255, 255, 0, 255, 0, 0, 255, 255, 255,
            0, 255, 255, 0, 255, 255, 255, 255, 255, 255, 255,
        ];
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &custom_shader_fallback_3d,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &identity_lut_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(8),
                rows_per_image: Some(2),
            },
            wgpu::Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 2,
            },
        );
        let custom_shader_fallback_view_3d =
            custom_shader_fallback_3d.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D3),
                ..Default::default()
            });

        let uniform_alignment = device.limits().min_uniform_buffer_offset_alignment as u64;
        let globals_size = std::mem::size_of::<GlobalParams>() as u64;
        let gamma_size = std::mem::size_of::<GammaParams>() as u64;
        let path_globals_offset = globals_size.next_multiple_of(uniform_alignment);
        let gamma_offset = (path_globals_offset + globals_size).next_multiple_of(uniform_alignment);

        let globals_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals_buffer"),
            size: gamma_offset + gamma_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let max_buffer_size = device.limits().max_buffer_size;
        let storage_buffer_alignment = device.limits().min_storage_buffer_offset_alignment as u64;
        let initial_instance_buffer_capacity = 2 * 1024 * 1024;
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instance_buffer"),
            size: initial_instance_buffer_capacity,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let globals_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals_bind_group"),
            layout: &bind_group_layouts.globals,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &globals_buffer,
                        offset: 0,
                        size: Some(NonZeroU64::new(globals_size).unwrap()),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &globals_buffer,
                        offset: gamma_offset,
                        size: Some(NonZeroU64::new(gamma_size).unwrap()),
                    }),
                },
            ],
        });

        let path_globals_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("path_globals_bind_group"),
            layout: &bind_group_layouts.globals,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &globals_buffer,
                        offset: path_globals_offset,
                        size: Some(NonZeroU64::new(globals_size).unwrap()),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &globals_buffer,
                        offset: gamma_offset,
                        size: Some(NonZeroU64::new(gamma_size).unwrap()),
                    }),
                },
            ],
        });

        let adapter_info = context.adapter.get_info();
        set_primitive_render_context(Arc::clone(&device), Arc::clone(&queue), surface_format);

        Ok(Self {
            device,
            queue,
            surface,
            surface_config,
            pipelines,
            bind_group_layouts,
            atlas,
            atlas_sampler,
            globals_buffer,
            path_globals_offset,
            gamma_offset,
            globals_bind_group,
            path_globals_bind_group,
            instance_buffer,
            instance_buffer_capacity: initial_instance_buffer_capacity,
            max_buffer_size,
            storage_buffer_alignment,
            // Defer intermediate texture creation to first draw call via ensure_intermediate_textures().
            // This avoids panics when the device/surface is in an invalid state during initialization.
            path_intermediate_texture: None,
            path_intermediate_view: None,
            path_msaa_texture: None,
            path_msaa_view: None,
            rendering_params,
            dual_source_blending,
            adapter_info,
            transparent_alpha_mode,
            opaque_alpha_mode,
            max_texture_size,
            global_custom_shader: None,
            global_custom_shader_init_failed: false,
            global_custom_shader_uniform_template: None,
            global_custom_shader_animate_uniforms: false,
            global_custom_shader_last_tick: Instant::now(),
            global_custom_shader_time_seconds: 0.0,
            named_custom_shaders: HashMap::new(),
            shader_surface_textures: HashMap::new(),
            custom_shader_fallback_texture_view,
            custom_shader_fallback_view_3d,
        })
    }

    fn create_bind_group_layouts(device: &wgpu::Device) -> WgpuBindGroupLayouts {
        let globals =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("globals_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(
                                std::mem::size_of::<GlobalParams>() as u64
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(
                                std::mem::size_of::<GammaParams>() as u64
                            ),
                        },
                        count: None,
                    },
                ],
            });

        let storage_buffer_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let instances = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("instances_layout"),
            entries: &[storage_buffer_entry(0)],
        });

        let instances_with_texture =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("instances_with_texture_layout"),
                entries: &[
                    storage_buffer_entry(0),
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let surfaces = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("surfaces_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<SurfaceParams>() as u64
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        WgpuBindGroupLayouts {
            globals,
            instances,
            instances_with_texture,
            surfaces,
        }
    }

    fn create_pipelines(
        device: &wgpu::Device,
        layouts: &WgpuBindGroupLayouts,
        surface_format: wgpu::TextureFormat,
        alpha_mode: wgpu::CompositeAlphaMode,
        path_sample_count: u32,
        dual_source_blending: bool,
    ) -> WgpuPipelines {
        let base_shader_source = include_str!("shaders.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpui_shaders"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(base_shader_source)),
        });

        let subpixel_shader_source = include_str!("shaders_subpixel.wgsl");
        let subpixel_shader_module = if dual_source_blending {
            let combined = format!(
                "enable dual_source_blending;\n{base_shader_source}\n{subpixel_shader_source}"
            );
            Some(device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gpui_subpixel_shaders"),
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(combined)),
            }))
        } else {
            None
        };

        let blend_mode = match alpha_mode {
            wgpu::CompositeAlphaMode::PreMultiplied => {
                wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING
            }
            _ => wgpu::BlendState::ALPHA_BLENDING,
        };

        let color_target = wgpu::ColorTargetState {
            format: surface_format,
            blend: Some(blend_mode),
            write_mask: wgpu::ColorWrites::ALL,
        };

        let create_pipeline = |name: &str,
                               vs_entry: &str,
                               fs_entry: &str,
                               globals_layout: &wgpu::BindGroupLayout,
                               data_layout: &wgpu::BindGroupLayout,
                               topology: wgpu::PrimitiveTopology,
                               color_targets: &[Option<wgpu::ColorTargetState>],
                               sample_count: u32,
                               module: &wgpu::ShaderModule| {
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{name}_layout")),
                bind_group_layouts: &[globals_layout, data_layout],
                immediate_size: 0,
            });

            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(name),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some(vs_entry),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some(fs_entry),
                    targets: color_targets,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: None,
            })
        };

        let quads = create_pipeline(
            "quads",
            "vs_quad",
            "fs_quad",
            &layouts.globals,
            &layouts.instances,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target.clone())],
            1,
            &shader_module,
        );

        let shadows = create_pipeline(
            "shadows",
            "vs_shadow",
            "fs_shadow",
            &layouts.globals,
            &layouts.instances,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target.clone())],
            1,
            &shader_module,
        );

        let path_rasterization = create_pipeline(
            "path_rasterization",
            "vs_path_rasterization",
            "fs_path_rasterization",
            &layouts.globals,
            &layouts.instances,
            wgpu::PrimitiveTopology::TriangleList,
            &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            path_sample_count,
            &shader_module,
        );

        let paths_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };

        let paths = create_pipeline(
            "paths",
            "vs_path",
            "fs_path",
            &layouts.globals,
            &layouts.instances_with_texture,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(paths_blend),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            1,
            &shader_module,
        );

        let underlines = create_pipeline(
            "underlines",
            "vs_underline",
            "fs_underline",
            &layouts.globals,
            &layouts.instances,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target.clone())],
            1,
            &shader_module,
        );

        let mono_sprites = create_pipeline(
            "mono_sprites",
            "vs_mono_sprite",
            "fs_mono_sprite",
            &layouts.globals,
            &layouts.instances_with_texture,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target.clone())],
            1,
            &shader_module,
        );

        let subpixel_sprites = if let Some(subpixel_module) = &subpixel_shader_module {
            let subpixel_blend = wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Src1,
                    dst_factor: wgpu::BlendFactor::OneMinusSrc1,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
            };

            Some(create_pipeline(
                "subpixel_sprites",
                "vs_subpixel_sprite",
                "fs_subpixel_sprite",
                &layouts.globals,
                &layouts.instances_with_texture,
                wgpu::PrimitiveTopology::TriangleStrip,
                &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(subpixel_blend),
                    write_mask: wgpu::ColorWrites::COLOR,
                })],
                1,
                subpixel_module,
            ))
        } else {
            None
        };

        let poly_sprites = create_pipeline(
            "poly_sprites",
            "vs_poly_sprite",
            "fs_poly_sprite",
            &layouts.globals,
            &layouts.instances_with_texture,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target.clone())],
            1,
            &shader_module,
        );

        let surfaces = create_pipeline(
            "surfaces",
            "vs_surface",
            "fs_surface",
            &layouts.globals,
            &layouts.surfaces,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target)],
            1,
            &shader_module,
        );

        WgpuPipelines {
            quads,
            shadows,
            path_rasterization,
            paths,
            underlines,
            mono_sprites,
            subpixel_sprites,
            poly_sprites,
            surfaces,
        }
    }

    fn create_path_intermediate(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("path_intermediate"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn create_msaa_if_needed(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        sample_count: u32,
    ) -> Option<(wgpu::Texture, wgpu::TextureView)> {
        if sample_count <= 1 {
            return None;
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("path_msaa"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Some((texture, view))
    }

    pub fn update_drawable_size(&mut self, size: Size<DevicePixels>) {
        let width = size.width.0 as u32;
        let height = size.height.0 as u32;

        if width != self.surface_config.width || height != self.surface_config.height {
            let clamped_width = width.min(self.max_texture_size);
            let clamped_height = height.min(self.max_texture_size);

            if clamped_width != width || clamped_height != height {
                warn!(
                    "Requested surface size ({}, {}) exceeds maximum texture dimension {}. \
                     Clamping to ({}, {}). Window content may not fill the entire window.",
                    width, height, self.max_texture_size, clamped_width, clamped_height
                );
            }

            // Wait for any in-flight GPU work to complete before destroying textures
            if let Err(e) = self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            }) {
                warn!("Failed to poll device during resize: {e:?}");
            }

            // Destroy old textures before allocating new ones to avoid GPU memory spikes
            if let Some(ref texture) = self.path_intermediate_texture {
                texture.destroy();
            }
            if let Some(ref texture) = self.path_msaa_texture {
                texture.destroy();
            }

            self.surface_config.width = clamped_width.max(1);
            self.surface_config.height = clamped_height.max(1);
            self.surface.configure(&self.device, &self.surface_config);

            // Invalidate intermediate textures - they will be lazily recreated
            // in draw() after we confirm the surface is healthy. This avoids
            // panics when the device/surface is in an invalid state during resize.
            self.path_intermediate_texture = None;
            self.path_intermediate_view = None;
            self.path_msaa_texture = None;
            self.path_msaa_view = None;
        }
    }

    fn ensure_intermediate_textures(&mut self) {
        if self.path_intermediate_texture.is_some() {
            return;
        }

        let (path_intermediate_texture, path_intermediate_view) = {
            let (t, v) = Self::create_path_intermediate(
                &self.device,
                self.surface_config.format,
                self.surface_config.width,
                self.surface_config.height,
            );
            (Some(t), Some(v))
        };
        self.path_intermediate_texture = path_intermediate_texture;
        self.path_intermediate_view = path_intermediate_view;

        let (path_msaa_texture, path_msaa_view) = Self::create_msaa_if_needed(
            &self.device,
            self.surface_config.format,
            self.surface_config.width,
            self.surface_config.height,
            self.rendering_params.path_sample_count,
        )
        .map(|(t, v)| (Some(t), Some(v)))
        .unwrap_or((None, None));
        self.path_msaa_texture = path_msaa_texture;
        self.path_msaa_view = path_msaa_view;
    }

    pub fn update_transparency(&mut self, transparent: bool) {
        let new_alpha_mode = if transparent {
            self.transparent_alpha_mode
        } else {
            self.opaque_alpha_mode
        };

        if new_alpha_mode != self.surface_config.alpha_mode {
            self.surface_config.alpha_mode = new_alpha_mode;
            self.surface.configure(&self.device, &self.surface_config);
            self.pipelines = Self::create_pipelines(
                &self.device,
                &self.bind_group_layouts,
                self.surface_config.format,
                self.surface_config.alpha_mode,
                self.rendering_params.path_sample_count,
                self.dual_source_blending,
            );
        }
    }

    #[allow(dead_code)]
    pub fn viewport_size(&self) -> Size<DevicePixels> {
        Size {
            width: DevicePixels(self.surface_config.width as i32),
            height: DevicePixels(self.surface_config.height as i32),
        }
    }

    pub fn sprite_atlas(&self) -> &Arc<WgpuAtlas> {
        &self.atlas
    }

    pub fn supports_dual_source_blending(&self) -> bool {
        self.dual_source_blending
    }

    pub fn gpu_specs(&self) -> GpuSpecs {
        GpuSpecs {
            is_software_emulated: self.adapter_info.device_type == wgpu::DeviceType::Cpu,
            device_name: self.adapter_info.name.clone(),
            driver_name: self.adapter_info.driver.clone(),
            driver_info: self.adapter_info.driver_info.clone(),
        }
    }

    pub fn max_texture_size(&self) -> u32 {
        self.max_texture_size
    }

    pub fn draw(&mut self, scene: &Scene) {
        self.ensure_global_custom_shader();
        let shader_surface_draws = take_render_primitive_draws();
        self.ensure_named_custom_shaders(&shader_surface_draws);
        let scene_shader_surface_draws: Vec<_> = scene
            .shader_surfaces
            .iter()
            .map(|surface| surface.draw.clone())
            .collect();
        self.ensure_named_custom_shaders(&scene_shader_surface_draws);
        let runtime = global_custom_shader_runtime()
            .lock()
            .map(|runtime| runtime.clone())
            .unwrap_or_default();
        let custom_shader_time_seconds = self.advance_custom_shader_time(&runtime);

        if let Some(shader) = self.global_custom_shader.take() {
            if runtime.enabled {
                if let Err(error) = self.update_global_custom_shader_uniforms(
                    &shader,
                    runtime.clone(),
                    custom_shader_time_seconds,
                ) {
                    log::error!("Failed to update global custom shader uniforms: {error:#}");
                }
                self.draw_internal(
                    scene,
                    Some(&shader),
                    &shader_surface_draws,
                    custom_shader_time_seconds,
                );
            } else {
                self.draw_internal(scene, None, &shader_surface_draws, custom_shader_time_seconds);
            }
            self.global_custom_shader = Some(shader);
        } else {
            self.draw_internal(scene, None, &shader_surface_draws, custom_shader_time_seconds);
        }
    }

    pub fn draw_with_custom_shader(&mut self, scene: &Scene, shader: &WgpuCustomShader) {
        let runtime = global_custom_shader_runtime()
            .lock()
            .map(|runtime| runtime.clone())
            .unwrap_or_default();
        let custom_shader_time_seconds = self.advance_custom_shader_time(&runtime);
        self.draw_internal(scene, Some(shader), &[], custom_shader_time_seconds);
    }

    pub fn create_custom_shader(
        &self,
        descriptor: CustomShaderDescriptor<'_>,
    ) -> anyhow::Result<WgpuCustomShader> {
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: descriptor.label,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(descriptor.shader_source)),
            });

        let uniform_size = descriptor
            .uniform_bytes
            .map(|bytes| {
                NonZeroU64::new(bytes.len() as u64)
                    .ok_or_else(|| anyhow::anyhow!("uniform_bytes cannot be empty"))
            })
            .transpose()?;

        let uniform_buffer = if let Some(bytes) = descriptor.uniform_bytes {
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: descriptor.label,
                size: bytes.len() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(&buffer, 0, bytes);
            Some(buffer)
        } else {
            None
        };

        let mut bind_group_layout_entries = Vec::new();
        if uniform_buffer.is_some() {
            bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: uniform_size,
                },
                count: None,
            });
        }
        match descriptor.texture_mode {
            CustomShaderTextureMode::VideoPlayerFull => {
                let tex2d_entry = |binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                };
                let tex3d_entry = |binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                };
                let sampler_entry = |binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                };

                bind_group_layout_entries.push(tex2d_entry(1));
                bind_group_layout_entries.push(tex2d_entry(2));
                bind_group_layout_entries.push(sampler_entry(3));
                bind_group_layout_entries.push(tex3d_entry(4));
                bind_group_layout_entries.push(sampler_entry(5));
                bind_group_layout_entries.push(tex2d_entry(6));
                bind_group_layout_entries.push(sampler_entry(7));
                bind_group_layout_entries.push(tex2d_entry(8));
                bind_group_layout_entries.push(tex2d_entry(9));
                bind_group_layout_entries.push(sampler_entry(10));
                bind_group_layout_entries.push(tex3d_entry(11));
                bind_group_layout_entries.push(sampler_entry(12));
                bind_group_layout_entries.push(tex2d_entry(13));
                bind_group_layout_entries.push(tex2d_entry(14));
            }
            CustomShaderTextureMode::None => {}
        }

        let bind_group_layout = if bind_group_layout_entries.is_empty() {
            None
        } else {
            Some(self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: descriptor.label,
                entries: &bind_group_layout_entries,
            }))
        };

        let default_blend = match self.surface_config.alpha_mode {
            wgpu::CompositeAlphaMode::PreMultiplied => {
                wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING
            }
            _ => wgpu::BlendState::ALPHA_BLENDING,
        };

        let target = wgpu::ColorTargetState {
            format: self.surface_config.format,
            blend: Some(descriptor.blend_state.unwrap_or(default_blend)),
            write_mask: wgpu::ColorWrites::ALL,
        };

        let layout_refs: [&wgpu::BindGroupLayout; 1];
        let bind_group_layouts = if let Some(layout) = bind_group_layout.as_ref() {
            layout_refs = [layout];
            &layout_refs[..]
        } else {
            &[]
        };

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: descriptor.label,
                bind_group_layouts,
                immediate_size: 0,
            });

        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: descriptor.label,
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some(descriptor.vertex_entry),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(descriptor.fragment_entry),
                    targets: &[Some(target)],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: descriptor.primitive_topology,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: None,
            });

        Ok(WgpuCustomShader {
            pipeline,
            uniform_buffer,
            bind_group_layout,
            uniform_size,
            texture_mode: descriptor.texture_mode,
            vertex_count: descriptor.vertex_count,
            instance_count: descriptor.instance_count,
        })
    }

    pub fn update_custom_shader_uniforms(
        &self,
        shader: &WgpuCustomShader,
        bytes: &[u8],
    ) -> anyhow::Result<()> {
        let Some(buffer) = shader.uniform_buffer.as_ref() else {
            anyhow::bail!("shader does not define uniforms");
        };
        let Some(size) = shader.uniform_size else {
            anyhow::bail!("shader does not define uniform size");
        };
        anyhow::ensure!(
            bytes.len() == size.get() as usize,
            "uniform size mismatch: expected {}, got {}",
            size.get(),
            bytes.len()
        );
        self.queue.write_buffer(buffer, 0, bytes);
        Ok(())
    }

    fn ensure_global_custom_shader(&mut self) {
        if self.global_custom_shader.is_some() || self.global_custom_shader_init_failed {
            return;
        }

        let config = global_custom_shader_config()
            .lock()
            .ok()
            .and_then(|slot| slot.clone());
        let Some(config) = config else {
            return;
        };

        let descriptor = CustomShaderDescriptor {
            label: config.label.as_deref(),
            shader_source: &config.shader_source,
            vertex_entry: &config.vertex_entry,
            fragment_entry: &config.fragment_entry,
            primitive_topology: config.primitive_topology,
            vertex_count: config.vertex_count,
            instance_count: config.instance_count,
            uniform_bytes: config.uniform_bytes.as_deref(),
            blend_state: config.blend_state,
            texture_mode: config.texture_mode,
        };

        match self.create_custom_shader(descriptor) {
            Ok(shader) => {
                self.global_custom_shader = Some(shader);
                self.global_custom_shader_uniform_template = config.uniform_bytes.clone();
                self.global_custom_shader_animate_uniforms = config.animate_uniforms_with_time;
                self.global_custom_shader_last_tick = Instant::now();
                self.global_custom_shader_time_seconds = 0.0;
            }
            Err(error) => {
                self.global_custom_shader_init_failed = true;
                log::error!("Failed to initialize global custom shader: {error:#}");
            }
        }
    }

    fn update_global_custom_shader_uniforms(
        &mut self,
        shader: &WgpuCustomShader,
        runtime: GlobalCustomShaderRuntime,
        custom_shader_time_seconds: f32,
    ) -> anyhow::Result<()> {
        if !self.global_custom_shader_animate_uniforms {
            return Ok(());
        }

        let Some(uniform_size) = shader.uniform_size else {
            return Ok(());
        };

        let mut bytes = self
            .global_custom_shader_uniform_template
            .clone()
            .unwrap_or_default();
        bytes.resize(uniform_size.get() as usize, 0);

        self.apply_standard_time_uniforms(&mut bytes, custom_shader_time_seconds);

        if let Some(runtime_bytes) = runtime.uniform_bytes.as_ref() {
            let len = bytes.len().min(runtime_bytes.len());
            bytes[..len].copy_from_slice(&runtime_bytes[..len]);
            self.apply_standard_time_uniforms(&mut bytes, custom_shader_time_seconds);
        }

        self.update_custom_shader_uniforms(shader, &bytes)
    }

    fn draw_internal(
        &mut self,
        scene: &Scene,
        custom_shader: Option<&WgpuCustomShader>,
        shader_surface_draws: &[ShaderSurfaceDraw],
        custom_shader_time_seconds: f32,
    ) {
        self.atlas.before_frame();
        self.sync_shader_surface_textures();

        let frame = match self.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.surface.configure(&self.device, &self.surface_config);
                return;
            }
            Err(e) => {
                log::error!("Failed to acquire surface texture: {e}");
                return;
            }
        };

        // Now that we know the surface is healthy, ensure intermediate textures exist
        self.ensure_intermediate_textures();

        let frame_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let gamma_params = GammaParams {
            gamma_ratios: self.rendering_params.gamma_ratios,
            grayscale_enhanced_contrast: self.rendering_params.grayscale_enhanced_contrast,
            subpixel_enhanced_contrast: self.rendering_params.subpixel_enhanced_contrast,
            _pad: [0.0; 2],
        };

        let globals = GlobalParams {
            viewport_size: [
                self.surface_config.width as f32,
                self.surface_config.height as f32,
            ],
            premultiplied_alpha: if self.surface_config.alpha_mode
                == wgpu::CompositeAlphaMode::PreMultiplied
            {
                1
            } else {
                0
            },
            pad: 0,
        };

        let path_globals = GlobalParams {
            premultiplied_alpha: 0,
            ..globals
        };

        self.queue
            .write_buffer(&self.globals_buffer, 0, bytemuck::bytes_of(&globals));
        self.queue.write_buffer(
            &self.globals_buffer,
            self.path_globals_offset,
            bytemuck::bytes_of(&path_globals),
        );
        self.queue.write_buffer(
            &self.globals_buffer,
            self.gamma_offset,
            bytemuck::bytes_of(&gamma_params),
        );

        loop {
            let mut instance_offset: u64 = 0;
            let mut overflow = false;
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("main_encoder"),
                });

            {
                let _clear_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("clear_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &frame_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
            }

            for draw in shader_surface_draws {
                let Some(named_shader) = self.named_custom_shaders.get(&draw.shader_key) else {
                    continue;
                };
                if let Err(error) = self.update_named_custom_shader_uniforms(
                    named_shader,
                    draw.uniform_bytes.as_deref(),
                    custom_shader_time_seconds,
                ) {
                    log::error!(
                        "Failed to update shader surface uniforms for '{}': {error:#}",
                        draw.shader_key
                    );
                    continue;
                }

                let Some((x, y, width, height)) = self.normalized_bounds_to_scissor(draw) else {
                    continue;
                };

                if named_shader.shader.vertex_count == 0 || named_shader.shader.instance_count == 0
                {
                    continue;
                }

                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("shader_surface_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &frame_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
                pass.set_scissor_rect(x, y, width, height);
                let texture_entry = draw
                    .texture_key
                    .as_ref()
                    .and_then(|key| self.shader_surface_textures.get(key));
                self.draw_custom_shader_pass(&named_shader.shader, &mut pass, texture_entry);
            }

            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("main_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &frame_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });

                for batch in scene.batches() {
                    let ok = match batch {
                        PrimitiveBatch::Quads(range) => {
                            self.draw_quads(&scene.quads[range], &mut instance_offset, &mut pass)
                        }
                        PrimitiveBatch::Shadows(range) => self.draw_shadows(
                            &scene.shadows[range],
                            &mut instance_offset,
                            &mut pass,
                        ),
                        PrimitiveBatch::Paths(range) => {
                            let paths = &scene.paths[range];
                            if paths.is_empty() {
                                continue;
                            }

                            drop(pass);

                            let did_draw = self.draw_paths_to_intermediate(
                                &mut encoder,
                                paths,
                                &mut instance_offset,
                            );

                            pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("main_pass_continued"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &frame_view,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Load,
                                        store: wgpu::StoreOp::Store,
                                    },
                                    depth_slice: None,
                                })],
                                depth_stencil_attachment: None,
                                ..Default::default()
                            });

                            if did_draw {
                                self.draw_paths_from_intermediate(
                                    paths,
                                    &mut instance_offset,
                                    &mut pass,
                                )
                            } else {
                                false
                            }
                        }
                        PrimitiveBatch::Underlines(range) => self.draw_underlines(
                            &scene.underlines[range],
                            &mut instance_offset,
                            &mut pass,
                        ),
                        PrimitiveBatch::MonochromeSprites { texture_id, range } => self
                            .draw_monochrome_sprites(
                                &scene.monochrome_sprites[range],
                                texture_id,
                                &mut instance_offset,
                                &mut pass,
                            ),
                        PrimitiveBatch::SubpixelSprites { texture_id, range } => self
                            .draw_subpixel_sprites(
                                &scene.subpixel_sprites[range],
                                texture_id,
                                &mut instance_offset,
                                &mut pass,
                            ),
                        PrimitiveBatch::PolychromeSprites { texture_id, range } => self
                            .draw_polychrome_sprites(
                                &scene.polychrome_sprites[range],
                                texture_id,
                                &mut instance_offset,
                                &mut pass,
                            ),
                        PrimitiveBatch::ShaderSurfaces(range) => {
                            let shader_surfaces = &scene.shader_surfaces[range];
                            if shader_surfaces.is_empty() {
                                continue;
                            }

                            drop(pass);

                            self.draw_shader_surfaces(
                                &mut encoder,
                                &frame_view,
                                shader_surfaces,
                                custom_shader_time_seconds,
                            );

                            pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("main_pass_continued"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &frame_view,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Load,
                                        store: wgpu::StoreOp::Store,
                                    },
                                    depth_slice: None,
                                })],
                                depth_stencil_attachment: None,
                                ..Default::default()
                            });

                            true
                        }
                        PrimitiveBatch::Surfaces(_surfaces) => {
                            // Surfaces are macOS-only for video playback
                            // Not implemented for Linux/wgpu
                            true
                        }
                    };
                    if !ok {
                        overflow = true;
                        break;
                    }
                }
            }

            if let Some(shader) = custom_shader {
                if shader.vertex_count > 0 && shader.instance_count > 0 {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("custom_shader_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &frame_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: None,
                        ..Default::default()
                    });
                    self.draw_custom_shader_pass(shader, &mut pass, None);
                }
            }

            if overflow {
                drop(encoder);
                if self.instance_buffer_capacity >= self.max_buffer_size {
                    log::error!(
                        "instance buffer size grew too large: {}",
                        self.instance_buffer_capacity
                    );
                    frame.present();
                    return;
                }
                self.grow_instance_buffer();
                continue;
            }

            self.queue.submit(std::iter::once(encoder.finish()));
            frame.present();
            return;
        }
    }

    fn draw_custom_shader_pass(
        &self,
        shader: &WgpuCustomShader,
        pass: &mut wgpu::RenderPass<'_>,
        texture_entry: Option<&ShaderSurfaceTextureGpu>,
    ) {
        pass.set_pipeline(&shader.pipeline);
        if let Some(bind_group) = self.create_custom_shader_bind_group(shader, texture_entry) {
            pass.set_bind_group(0, &bind_group, &[]);
        }
        pass.draw(0..shader.vertex_count, 0..shader.instance_count);
    }

    fn create_custom_shader_bind_group(
        &self,
        shader: &WgpuCustomShader,
        texture_entry: Option<&ShaderSurfaceTextureGpu>,
    ) -> Option<wgpu::BindGroup> {
        let layout = shader.bind_group_layout.as_ref()?;
        let mut entries = Vec::new();

        if let (Some(buffer), Some(size)) = (shader.uniform_buffer.as_ref(), shader.uniform_size) {
            entries.push(wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer,
                    offset: 0,
                    size: Some(size),
                }),
            });
        }

        match shader.texture_mode {
            CustomShaderTextureMode::VideoPlayerFull => {
                let view_y = texture_entry
                    .map(|entry| &entry.view_primary)
                    .unwrap_or(&self.custom_shader_fallback_texture_view);
                let view_uv = texture_entry
                    .and_then(|entry| entry.view_secondary.as_ref())
                    .unwrap_or(&self.custom_shader_fallback_texture_view);
                let lut_view = texture_entry
                    .and_then(|entry| entry.view_3d.as_ref())
                    .unwrap_or(&self.custom_shader_fallback_view_3d);
                let caption_view = texture_entry
                    .and_then(|entry| entry.view_caption.as_ref())
                    .unwrap_or(&self.custom_shader_fallback_texture_view);
                let rgba_view = texture_entry
                    .and_then(|entry| entry.view_rgba.as_ref())
                    .unwrap_or(&self.custom_shader_fallback_texture_view);
                let broll_view = texture_entry
                    .and_then(|entry| entry.view_broll.as_ref())
                    .unwrap_or(&self.custom_shader_fallback_texture_view);
                let broll_lut_view = texture_entry
                    .and_then(|entry| entry.view_broll_3d.as_ref())
                    .unwrap_or(&self.custom_shader_fallback_view_3d);
                let broll_y_view = texture_entry
                    .and_then(|entry| entry.view_broll_y.as_ref())
                    .unwrap_or(&self.custom_shader_fallback_texture_view);
                let broll_uv_view = texture_entry
                    .and_then(|entry| entry.view_broll_uv.as_ref())
                    .unwrap_or(&self.custom_shader_fallback_texture_view);

                entries.push(wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(view_y),
                });
                entries.push(wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(view_uv),
                });
                entries.push(wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.atlas_sampler),
                });
                entries.push(wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(lut_view),
                });
                entries.push(wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&self.atlas_sampler),
                });
                entries.push(wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(caption_view),
                });
                entries.push(wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Sampler(&self.atlas_sampler),
                });
                entries.push(wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(rgba_view),
                });
                entries.push(wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(broll_view),
                });
                entries.push(wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::Sampler(&self.atlas_sampler),
                });
                entries.push(wgpu::BindGroupEntry {
                    binding: 11,
                    resource: wgpu::BindingResource::TextureView(broll_lut_view),
                });
                entries.push(wgpu::BindGroupEntry {
                    binding: 12,
                    resource: wgpu::BindingResource::Sampler(&self.atlas_sampler),
                });
                entries.push(wgpu::BindGroupEntry {
                    binding: 13,
                    resource: wgpu::BindingResource::TextureView(broll_y_view),
                });
                entries.push(wgpu::BindGroupEntry {
                    binding: 14,
                    resource: wgpu::BindingResource::TextureView(broll_uv_view),
                });
            }
            CustomShaderTextureMode::None => {}
        }

        Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("custom_shader_bind_group"),
            layout,
            entries: &entries,
        }))
    }

    fn ensure_named_custom_shaders(&mut self, draws: &[ShaderSurfaceDraw]) {
        let configured_shaders = named_custom_shader_configs()
            .lock()
            .map(|configs| configs.clone())
            .unwrap_or_default();

        for draw in draws {
            if self.named_custom_shaders.contains_key(&draw.shader_key) {
                continue;
            }
            let Some(config) = configured_shaders.get(&draw.shader_key).cloned() else {
                log::warn!(
                    "Shader surface referenced unknown shader key '{}'",
                    draw.shader_key
                );
                continue;
            };

            let descriptor = CustomShaderDescriptor {
                label: config.label.as_deref(),
                shader_source: &config.shader_source,
                vertex_entry: &config.vertex_entry,
                fragment_entry: &config.fragment_entry,
                primitive_topology: config.primitive_topology,
                vertex_count: config.vertex_count,
                instance_count: config.instance_count,
                uniform_bytes: config.uniform_bytes.as_deref(),
                blend_state: config.blend_state,
                texture_mode: config.texture_mode,
            };

            match self.create_custom_shader(descriptor) {
                Ok(shader) => {
                    self.named_custom_shaders.insert(
                        draw.shader_key.clone(),
                        NamedCustomShader {
                            shader,
                            uniform_template: config.uniform_bytes,
                            animate_uniforms_with_time: config.animate_uniforms_with_time,
                        },
                    );
                }
                Err(error) => {
                    log::error!(
                        "Failed to initialize named custom shader '{}': {error:#}",
                        draw.shader_key
                    );
                }
            }
        }
    }

    fn advance_custom_shader_time(&mut self, runtime: &GlobalCustomShaderRuntime) -> f32 {
        let now = Instant::now();
        if !runtime.paused {
            let dt = now
                .saturating_duration_since(self.global_custom_shader_last_tick)
                .as_secs_f32();
            self.global_custom_shader_time_seconds += dt * runtime.time_scale;
        }
        self.global_custom_shader_last_tick = now;
        self.global_custom_shader_time_seconds
    }

    fn apply_standard_time_uniforms(&self, bytes: &mut [u8], custom_shader_time_seconds: f32) {
        if bytes.len() < 16 {
            return;
        }
        let params = [
            custom_shader_time_seconds,
            self.surface_config.width as f32,
            self.surface_config.height as f32,
            0.0f32,
        ];
        bytes[..16].copy_from_slice(bytemuck::bytes_of(&params));
    }

    fn update_named_custom_shader_uniforms(
        &self,
        named_shader: &NamedCustomShader,
        draw_uniform_bytes: Option<&[u8]>,
        custom_shader_time_seconds: f32,
    ) -> anyhow::Result<()> {
        let Some(uniform_size) = named_shader.shader.uniform_size else {
            return Ok(());
        };

        let mut bytes = if let Some(draw_uniform_bytes) = draw_uniform_bytes {
            let mut bytes = draw_uniform_bytes.to_vec();
            bytes.resize(uniform_size.get() as usize, 0);
            bytes
        } else {
            let mut bytes = named_shader.uniform_template.clone().unwrap_or_default();
            bytes.resize(uniform_size.get() as usize, 0);
            bytes
        };

        if named_shader.animate_uniforms_with_time {
            self.apply_standard_time_uniforms(&mut bytes, custom_shader_time_seconds);
        }

        self.update_custom_shader_uniforms(&named_shader.shader, &bytes)
    }

    fn sync_shader_surface_textures(&mut self) {
        let updates_rgba = take_render_primitive_texture_updates();
        let updates_pair = take_render_primitive_texture_updates_pair();
        let updates_3d = take_render_primitive_texture_3d_updates();
        let updates_caption = take_render_primitive_texture_caption_updates();
        let updates_rgba_video = take_render_primitive_texture_rgba_video_updates();
        let updates_broll = take_render_primitive_texture_broll_updates();
        let updates_broll_nv12 = take_render_primitive_texture_broll_nv12_updates();
        let updates_broll_lut3d = take_render_primitive_texture_broll_lut3d_updates();

        for (key, update) in updates_rgba {
            if update.width == 0 || update.height == 0 {
                continue;
            }

            let expected_len = update.width as usize * update.height as usize * 4;
            if update.bytes.len() < expected_len {
                continue;
            }

            let needs_recreate = self
                .shader_surface_textures
                .get(&key)
                .map(|existing| {
                    existing.width != update.width
                        || existing.height != update.height
                        || existing.mode != CustomShaderTextureMode::VideoPlayerFull
                })
                .unwrap_or(true);

            if needs_recreate {
                let texture_primary = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("shader_surface_texture"),
                    size: wgpu::Extent3d {
                        width: update.width,
                        height: update.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: Self::shader_surface_texture_format(update.format),
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                let view_primary =
                    texture_primary.create_view(&wgpu::TextureViewDescriptor::default());
                self.shader_surface_textures.insert(
                    key.clone(),
                    ShaderSurfaceTextureGpu {
                        texture_primary,
                        view_primary,
                        texture_secondary: None,
                        view_secondary: None,
                        mode: CustomShaderTextureMode::VideoPlayerFull,
                        width: update.width,
                        height: update.height,
                        texture_3d: None,
                        view_3d: None,
                        texture_caption: None,
                        view_caption: None,
                        texture_rgba: None,
                        view_rgba: None,
                        texture_broll: None,
                        view_broll: None,
                        texture_broll_y: None,
                        view_broll_y: None,
                        texture_broll_uv: None,
                        view_broll_uv: None,
                        texture_broll_3d: None,
                        view_broll_3d: None,
                    },
                );
            }

            let Some(entry) = self.shader_surface_textures.get(&key) else {
                continue;
            };

            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &entry.texture_primary,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &update.bytes[..expected_len],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(update.width * 4),
                    rows_per_image: Some(update.height),
                },
                wgpu::Extent3d {
                    width: update.width,
                    height: update.height,
                    depth_or_array_layers: 1,
                },
            );
        }

        for (key, update) in updates_pair {
            if update.width == 0 || update.height == 0 || update.stride == 0 {
                continue;
            }

            let y_plane_size = (update.stride * update.height) as usize;
            let uv_plane_size = (update.stride * (update.height / 2)) as usize;
            let total_size = y_plane_size + uv_plane_size;
            if update.data.len() < total_size {
                continue;
            }

            let needs_recreate = self
                .shader_surface_textures
                .get(&key)
                .map(|existing| {
                    existing.width != update.width
                        || existing.height != update.height
                        || existing.mode != CustomShaderTextureMode::VideoPlayerFull
                })
                .unwrap_or(true);

            if needs_recreate {
                let texture_primary = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("shader_surface_texture_nv12_y"),
                    size: wgpu::Extent3d {
                        width: update.width,
                        height: update.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::R8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                let view_primary =
                    texture_primary.create_view(&wgpu::TextureViewDescriptor::default());

                let texture_secondary = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("shader_surface_texture_nv12_uv"),
                    size: wgpu::Extent3d {
                        width: update.width / 2,
                        height: update.height / 2,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rg8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                let view_secondary =
                    texture_secondary.create_view(&wgpu::TextureViewDescriptor::default());

                self.shader_surface_textures.insert(
                    key.clone(),
                    ShaderSurfaceTextureGpu {
                        texture_primary,
                        view_primary,
                        texture_secondary: Some(texture_secondary),
                        view_secondary: Some(view_secondary),
                        mode: CustomShaderTextureMode::VideoPlayerFull,
                        width: update.width,
                        height: update.height,
                        texture_3d: None,
                        view_3d: None,
                        texture_caption: None,
                        view_caption: None,
                        texture_rgba: None,
                        view_rgba: None,
                        texture_broll: None,
                        view_broll: None,
                        texture_broll_y: None,
                        view_broll_y: None,
                        texture_broll_uv: None,
                        view_broll_uv: None,
                        texture_broll_3d: None,
                        view_broll_3d: None,
                    },
                );
            }

            let Some(entry) = self.shader_surface_textures.get(&key) else {
                continue;
            };
            let Some(texture_uv) = entry.texture_secondary.as_ref() else {
                continue;
            };

            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &entry.texture_primary,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &update.data[..y_plane_size],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(update.stride),
                    rows_per_image: Some(update.height),
                },
                wgpu::Extent3d {
                    width: update.width,
                    height: update.height,
                    depth_or_array_layers: 1,
                },
            );

            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: texture_uv,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &update.data[y_plane_size..total_size],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(update.stride),
                    rows_per_image: Some(update.height / 2),
                },
                wgpu::Extent3d {
                    width: update.width / 2,
                    height: update.height / 2,
                    depth_or_array_layers: 1,
                },
            );
        }

        for (key, lut) in updates_3d {
            if lut.size == 0 {
                continue;
            }
            let Some(entry) = self.shader_surface_textures.get_mut(&key) else {
                continue;
            };

            let expected_len = (lut.size as usize)
                .saturating_mul(lut.size as usize)
                .saturating_mul(lut.size as usize)
                .saturating_mul(4);
            if lut.data.len() < expected_len {
                continue;
            }

            let lut_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("shader_surface_texture3d"),
                size: wgpu::Extent3d {
                    width: lut.size,
                    height: lut.size,
                    depth_or_array_layers: lut.size,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let lut_view = lut_texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D3),
                ..Default::default()
            });

            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &lut_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &lut.data[..expected_len],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(lut.size * 4),
                    rows_per_image: Some(lut.size),
                },
                wgpu::Extent3d {
                    width: lut.size,
                    height: lut.size,
                    depth_or_array_layers: lut.size,
                },
            );

            entry.texture_3d = Some(lut_texture);
            entry.view_3d = Some(lut_view);
            entry.mode = CustomShaderTextureMode::VideoPlayerFull;
        }

        for (key, update) in updates_caption {
            self.apply_aux_2d_texture_update(&key, update, |entry, texture, view| {
                entry.texture_caption = Some(texture);
                entry.view_caption = Some(view);
            });
        }

        for (key, update) in updates_rgba_video {
            self.apply_aux_2d_texture_update(&key, update, |entry, texture, view| {
                entry.texture_rgba = Some(texture);
                entry.view_rgba = Some(view);
            });
        }

        for (key, update) in updates_broll {
            self.apply_aux_2d_texture_update(&key, update, |entry, texture, view| {
                entry.texture_broll = Some(texture);
                entry.view_broll = Some(view);
            });
        }

        for (key, update) in updates_broll_nv12 {
            self.apply_aux_nv12_texture_update(&key, update, |entry, y_tex, y_view, uv_tex, uv_view| {
                entry.texture_broll_y = Some(y_tex);
                entry.view_broll_y = Some(y_view);
                entry.texture_broll_uv = Some(uv_tex);
                entry.view_broll_uv = Some(uv_view);
            });
        }

        for (key, lut) in updates_broll_lut3d {
            if lut.size == 0 {
                continue;
            }
            let Some(entry) = self.shader_surface_textures.get_mut(&key) else {
                continue;
            };

            let expected_len = (lut.size as usize)
                .saturating_mul(lut.size as usize)
                .saturating_mul(lut.size as usize)
                .saturating_mul(4);
            if lut.data.len() < expected_len {
                continue;
            }

            let lut_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("shader_surface_texture3d_broll"),
                size: wgpu::Extent3d {
                    width: lut.size,
                    height: lut.size,
                    depth_or_array_layers: lut.size,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let lut_view = lut_texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D3),
                ..Default::default()
            });

            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &lut_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &lut.data[..expected_len],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(lut.size * 4),
                    rows_per_image: Some(lut.size),
                },
                wgpu::Extent3d {
                    width: lut.size,
                    height: lut.size,
                    depth_or_array_layers: lut.size,
                },
            );

            entry.texture_broll_3d = Some(lut_texture);
            entry.view_broll_3d = Some(lut_view);
        }
    }

    fn apply_aux_2d_texture_update<F>(
        &mut self,
        key: &str,
        update: ShaderSurfaceTextureData,
        mut assign: F,
    ) where
        F: FnMut(&mut ShaderSurfaceTextureGpu, wgpu::Texture, wgpu::TextureView),
    {
        if update.width == 0 || update.height == 0 {
            return;
        }
        let expected_len = update.width as usize * update.height as usize * 4;
        if update.bytes.len() < expected_len {
            return;
        }
        let Some(entry) = self.shader_surface_textures.get_mut(key) else {
            return;
        };
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shader_surface_texture_aux2d"),
            size: wgpu::Extent3d {
                width: update.width,
                height: update.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::shader_surface_texture_format(update.format),
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &update.bytes[..expected_len],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(update.width * 4),
                rows_per_image: Some(update.height),
            },
            wgpu::Extent3d {
                width: update.width,
                height: update.height,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        assign(entry, texture, view);
    }

    fn apply_aux_nv12_texture_update<F>(
        &mut self,
        key: &str,
        update: ShaderSurfaceTexturePairData,
        mut assign: F,
    ) where
        F: FnMut(
            &mut ShaderSurfaceTextureGpu,
            wgpu::Texture,
            wgpu::TextureView,
            wgpu::Texture,
            wgpu::TextureView,
        ),
    {
        if update.width == 0 || update.height == 0 || update.stride == 0 {
            return;
        }
        let y_plane_size = (update.stride * update.height) as usize;
        let uv_plane_size = (update.stride * (update.height / 2)) as usize;
        let total_size = y_plane_size + uv_plane_size;
        if update.data.len() < total_size {
            return;
        }
        let Some(entry) = self.shader_surface_textures.get_mut(key) else {
            return;
        };

        let texture_y = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shader_surface_texture_aux_nv12_y"),
            size: wgpu::Extent3d {
                width: update.width,
                height: update.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let texture_uv = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shader_surface_texture_aux_nv12_uv"),
            size: wgpu::Extent3d {
                width: update.width / 2,
                height: update.height / 2,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture_y,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &update.data[..y_plane_size],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(update.stride),
                rows_per_image: Some(update.height),
            },
            wgpu::Extent3d {
                width: update.width,
                height: update.height,
                depth_or_array_layers: 1,
            },
        );

        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture_uv,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &update.data[y_plane_size..total_size],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(update.stride),
                rows_per_image: Some(update.height / 2),
            },
            wgpu::Extent3d {
                width: update.width / 2,
                height: update.height / 2,
                depth_or_array_layers: 1,
            },
        );

        let view_y = texture_y.create_view(&wgpu::TextureViewDescriptor::default());
        let view_uv = texture_uv.create_view(&wgpu::TextureViewDescriptor::default());
        assign(entry, texture_y, view_y, texture_uv, view_uv);
    }

    fn shader_surface_texture_format(format: ShaderSurfaceTextureFormat) -> wgpu::TextureFormat {
        match format {
            ShaderSurfaceTextureFormat::Bgra8Unorm => wgpu::TextureFormat::Bgra8Unorm,
            ShaderSurfaceTextureFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
        }
    }

    fn normalized_bounds_to_scissor(
        &self,
        draw: &ShaderSurfaceDraw,
    ) -> Option<(u32, u32, u32, u32)> {
        let [min_x, min_y, max_x, max_y] = draw.normalized_bounds;
        let min_x = min_x.clamp(0.0, 1.0);
        let min_y = min_y.clamp(0.0, 1.0);
        let max_x = max_x.clamp(0.0, 1.0);
        let max_y = max_y.clamp(0.0, 1.0);

        if max_x <= min_x || max_y <= min_y {
            return None;
        }

        let surface_width = self.surface_config.width as f32;
        let surface_height = self.surface_config.height as f32;

        let x = (min_x * surface_width).floor().max(0.0) as u32;
        let y = (min_y * surface_height).floor().max(0.0) as u32;
        let max_x_px = (max_x * surface_width).ceil().max(0.0) as u32;
        let max_y_px = (max_y * surface_height).ceil().max(0.0) as u32;

        let width = max_x_px.saturating_sub(x);
        let height = max_y_px.saturating_sub(y);
        if width == 0 || height == 0 {
            return None;
        }

        Some((x, y, width, height))
    }

    fn shader_surface_bounds_to_scissor(
        &self,
        shader_surface: &PaintShaderSurface,
    ) -> Option<(u32, u32, u32, u32)> {
        let clipped_bounds = shader_surface
            .bounds
            .intersect(&shader_surface.content_mask.bounds);
        if clipped_bounds.is_empty() {
            return None;
        }

        let surface_width = self.surface_config.width as f32;
        let surface_height = self.surface_config.height as f32;

        let min_x = clipped_bounds.origin.x.0.clamp(0.0, surface_width);
        let min_y = clipped_bounds.origin.y.0.clamp(0.0, surface_height);
        let max_x =
            (clipped_bounds.origin.x.0 + clipped_bounds.size.width.0).clamp(0.0, surface_width);
        let max_y =
            (clipped_bounds.origin.y.0 + clipped_bounds.size.height.0).clamp(0.0, surface_height);

        if max_x <= min_x || max_y <= min_y {
            return None;
        }

        let x = min_x.floor() as u32;
        let y = min_y.floor() as u32;
        let max_x_px = max_x.ceil() as u32;
        let max_y_px = max_y.ceil() as u32;
        let width = max_x_px.saturating_sub(x);
        let height = max_y_px.saturating_sub(y);
        if width == 0 || height == 0 {
            return None;
        }

        Some((x, y, width, height))
    }

    fn draw_shader_surfaces(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        frame_view: &wgpu::TextureView,
        shader_surfaces: &[PaintShaderSurface],
        custom_shader_time_seconds: f32,
    ) {
        for shader_surface in shader_surfaces {
            let draw = &shader_surface.draw;
            let Some(named_shader) = self.named_custom_shaders.get(&draw.shader_key) else {
                continue;
            };
            if let Err(error) = self.update_named_custom_shader_uniforms(
                named_shader,
                draw.uniform_bytes.as_deref(),
                custom_shader_time_seconds,
            ) {
                log::error!(
                    "Failed to update shader surface uniforms for '{}': {error:#}",
                    draw.shader_key
                );
                continue;
            }

            let Some((x, y, width, height)) = self.shader_surface_bounds_to_scissor(shader_surface)
            else {
                continue;
            };

            if named_shader.shader.vertex_count == 0 || named_shader.shader.instance_count == 0 {
                continue;
            }

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene_shader_surface_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: frame_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_scissor_rect(x, y, width, height);
            let texture_entry = draw
                .texture_key
                .as_ref()
                .and_then(|key| self.shader_surface_textures.get(key));
            self.draw_custom_shader_pass(&named_shader.shader, &mut pass, texture_entry);
        }
    }

    fn draw_quads(
        &self,
        quads: &[Quad],
        instance_offset: &mut u64,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> bool {
        let data = unsafe { Self::instance_bytes(quads) };
        self.draw_instances(
            data,
            quads.len() as u32,
            &self.pipelines.quads,
            instance_offset,
            pass,
        )
    }

    fn draw_shadows(
        &self,
        shadows: &[Shadow],
        instance_offset: &mut u64,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> bool {
        let data = unsafe { Self::instance_bytes(shadows) };
        self.draw_instances(
            data,
            shadows.len() as u32,
            &self.pipelines.shadows,
            instance_offset,
            pass,
        )
    }

    fn draw_underlines(
        &self,
        underlines: &[Underline],
        instance_offset: &mut u64,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> bool {
        let data = unsafe { Self::instance_bytes(underlines) };
        self.draw_instances(
            data,
            underlines.len() as u32,
            &self.pipelines.underlines,
            instance_offset,
            pass,
        )
    }

    fn draw_monochrome_sprites(
        &self,
        sprites: &[MonochromeSprite],
        texture_id: AtlasTextureId,
        instance_offset: &mut u64,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> bool {
        let tex_info = self.atlas.get_texture_info(texture_id);
        let data = unsafe { Self::instance_bytes(sprites) };
        self.draw_instances_with_texture(
            data,
            sprites.len() as u32,
            &tex_info.view,
            &self.pipelines.mono_sprites,
            instance_offset,
            pass,
        )
    }

    fn draw_subpixel_sprites(
        &self,
        sprites: &[SubpixelSprite],
        texture_id: AtlasTextureId,
        instance_offset: &mut u64,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> bool {
        let tex_info = self.atlas.get_texture_info(texture_id);
        let data = unsafe { Self::instance_bytes(sprites) };
        let pipeline = self
            .pipelines
            .subpixel_sprites
            .as_ref()
            .unwrap_or(&self.pipelines.mono_sprites);
        self.draw_instances_with_texture(
            data,
            sprites.len() as u32,
            &tex_info.view,
            pipeline,
            instance_offset,
            pass,
        )
    }

    fn draw_polychrome_sprites(
        &self,
        sprites: &[PolychromeSprite],
        texture_id: AtlasTextureId,
        instance_offset: &mut u64,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> bool {
        let tex_info = self.atlas.get_texture_info(texture_id);
        let data = unsafe { Self::instance_bytes(sprites) };
        self.draw_instances_with_texture(
            data,
            sprites.len() as u32,
            &tex_info.view,
            &self.pipelines.poly_sprites,
            instance_offset,
            pass,
        )
    }

    fn draw_instances(
        &self,
        data: &[u8],
        instance_count: u32,
        pipeline: &wgpu::RenderPipeline,
        instance_offset: &mut u64,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> bool {
        if instance_count == 0 {
            return true;
        }
        let Some((offset, size)) = self.write_to_instance_buffer(instance_offset, data) else {
            return false;
        };
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bind_group_layouts.instances,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.instance_binding(offset, size),
            }],
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.globals_bind_group, &[]);
        pass.set_bind_group(1, &bind_group, &[]);
        pass.draw(0..4, 0..instance_count);
        true
    }

    fn draw_instances_with_texture(
        &self,
        data: &[u8],
        instance_count: u32,
        texture_view: &wgpu::TextureView,
        pipeline: &wgpu::RenderPipeline,
        instance_offset: &mut u64,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> bool {
        if instance_count == 0 {
            return true;
        }
        let Some((offset, size)) = self.write_to_instance_buffer(instance_offset, data) else {
            return false;
        };
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bind_group_layouts.instances_with_texture,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.instance_binding(offset, size),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.atlas_sampler),
                },
            ],
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.globals_bind_group, &[]);
        pass.set_bind_group(1, &bind_group, &[]);
        pass.draw(0..4, 0..instance_count);
        true
    }

    unsafe fn instance_bytes<T>(instances: &[T]) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                instances.as_ptr() as *const u8,
                std::mem::size_of_val(instances),
            )
        }
    }

    fn draw_paths_from_intermediate(
        &self,
        paths: &[Path<ScaledPixels>],
        instance_offset: &mut u64,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> bool {
        let first_path = &paths[0];
        let sprites: Vec<PathSprite> = if paths.last().map(|p| &p.order) == Some(&first_path.order)
        {
            paths
                .iter()
                .map(|p| PathSprite {
                    bounds: p.clipped_bounds(),
                })
                .collect()
        } else {
            let mut bounds = first_path.clipped_bounds();
            for path in paths.iter().skip(1) {
                bounds = bounds.union(&path.clipped_bounds());
            }
            vec![PathSprite { bounds }]
        };

        let Some(path_intermediate_view) = self.path_intermediate_view.as_ref() else {
            return true;
        };

        let sprite_data = unsafe { Self::instance_bytes(&sprites) };
        self.draw_instances_with_texture(
            sprite_data,
            sprites.len() as u32,
            path_intermediate_view,
            &self.pipelines.paths,
            instance_offset,
            pass,
        )
    }

    fn draw_paths_to_intermediate(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        paths: &[Path<ScaledPixels>],
        instance_offset: &mut u64,
    ) -> bool {
        let mut vertices = Vec::new();
        for path in paths {
            let bounds = path.clipped_bounds();
            vertices.extend(path.vertices.iter().map(|v| PathRasterizationVertex {
                xy_position: v.xy_position,
                st_position: v.st_position,
                color: path.color,
                bounds,
            }));
        }

        if vertices.is_empty() {
            return true;
        }

        let vertex_data = unsafe { Self::instance_bytes(&vertices) };
        let Some((vertex_offset, vertex_size)) =
            self.write_to_instance_buffer(instance_offset, vertex_data)
        else {
            return false;
        };

        let data_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("path_rasterization_bind_group"),
            layout: &self.bind_group_layouts.instances,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.instance_binding(vertex_offset, vertex_size),
            }],
        });

        let Some(path_intermediate_view) = self.path_intermediate_view.as_ref() else {
            return true;
        };

        let (target_view, resolve_target) = if let Some(ref msaa_view) = self.path_msaa_view {
            (msaa_view, Some(path_intermediate_view))
        } else {
            (path_intermediate_view, None)
        };

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("path_rasterization_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            pass.set_pipeline(&self.pipelines.path_rasterization);
            pass.set_bind_group(0, &self.path_globals_bind_group, &[]);
            pass.set_bind_group(1, &data_bind_group, &[]);
            pass.draw(0..vertices.len() as u32, 0..1);
        }

        true
    }

    fn grow_instance_buffer(&mut self) {
        let new_capacity = (self.instance_buffer_capacity * 2).min(self.max_buffer_size);
        log::info!("increased instance buffer size to {}", new_capacity);
        self.instance_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instance_buffer"),
            size: new_capacity,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.instance_buffer_capacity = new_capacity;
    }

    fn write_to_instance_buffer(
        &self,
        instance_offset: &mut u64,
        data: &[u8],
    ) -> Option<(u64, NonZeroU64)> {
        let offset = (*instance_offset).next_multiple_of(self.storage_buffer_alignment);
        let size = (data.len() as u64).max(16);
        if offset + size > self.instance_buffer_capacity {
            return None;
        }
        self.queue.write_buffer(&self.instance_buffer, offset, data);
        *instance_offset = offset + size;
        Some((offset, NonZeroU64::new(size).expect("size is at least 16")))
    }

    fn instance_binding(&self, offset: u64, size: NonZeroU64) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.instance_buffer,
            offset,
            size: Some(size),
        })
    }

    pub fn destroy(&mut self) {
        // wgpu resources are automatically cleaned up when dropped
    }
}

struct RenderingParameters {
    path_sample_count: u32,
    gamma_ratios: [f32; 4],
    grayscale_enhanced_contrast: f32,
    subpixel_enhanced_contrast: f32,
}

impl RenderingParameters {
    fn new(adapter: &wgpu::Adapter, surface_format: wgpu::TextureFormat) -> Self {
        use std::env;

        let format_features = adapter.get_texture_format_features(surface_format);
        let path_sample_count = [4, 2, 1]
            .into_iter()
            .find(|&n| format_features.flags.sample_count_supported(n))
            .unwrap_or(1);

        let gamma = env::var("ZED_FONTS_GAMMA")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.8_f32)
            .clamp(1.0, 2.2);
        let gamma_ratios = get_gamma_correction_ratios(gamma);

        let grayscale_enhanced_contrast = env::var("ZED_FONTS_GRAYSCALE_ENHANCED_CONTRAST")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.0_f32)
            .max(0.0);

        let subpixel_enhanced_contrast = env::var("ZED_FONTS_SUBPIXEL_ENHANCED_CONTRAST")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.5_f32)
            .max(0.0);

        Self {
            path_sample_count,
            gamma_ratios,
            grayscale_enhanced_contrast,
            subpixel_enhanced_contrast,
        }
    }
}
