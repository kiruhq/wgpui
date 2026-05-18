use anyhow::Result;
use gpui::{DevicePixels, Scene, Size};
use gpui_wgpu::{WgpuContext, WgpuRenderer, WgpuSurfaceConfig};
#[cfg(any(test, feature = "test-support"))]
use image::RgbaImage;
use parking_lot::Mutex;
use raw_window_handle as rwh;
use std::{ffi::c_void, ptr::NonNull, sync::Arc};

pub(crate) type Context = Arc<Mutex<Option<WgpuContext>>>;
pub(crate) type Renderer = WgpuMacRenderer;

pub(crate) unsafe fn new_renderer(
    context: Context,
    _native_window: *mut c_void,
    native_view: *mut c_void,
    bounds: Size<f32>,
    transparent: bool,
) -> Renderer {
    let native_view = match NonNull::new(native_view) {
        Some(native_view) => native_view,
        None => {
            log::error!("Failed to initialize wgpu renderer: native view pointer is null");
            std::process::exit(1);
        }
    };

    let raw_window = RawWindow { native_view };
    let config = WgpuSurfaceConfig {
        size: Size {
            width: DevicePixels(bounds.width.max(1.0).round() as i32),
            height: DevicePixels(bounds.height.max(1.0).round() as i32),
        },
        transparent,
    };

    let renderer = {
        let mut context = context.lock();
        match WgpuRenderer::new(&mut context, &raw_window, config) {
            Ok(renderer) => renderer,
            Err(error) => {
                log::error!("Failed to initialize wgpu renderer: {error:#}");
                std::process::exit(1);
            }
        }
    };

    WgpuMacRenderer { renderer }
}

pub(crate) struct WgpuMacRenderer {
    renderer: WgpuRenderer,
}

impl WgpuMacRenderer {
    pub fn sprite_atlas(&self) -> &Arc<gpui_wgpu::WgpuAtlas> {
        self.renderer.sprite_atlas()
    }

    pub fn set_presents_with_transaction(&mut self, _presents_with_transaction: bool) {}

    pub fn update_drawable_size(&mut self, size: Size<DevicePixels>) {
        self.renderer.update_drawable_size(size);
    }

    pub fn update_transparency(&mut self, transparent: bool) {
        self.renderer.update_transparency(transparent);
    }

    pub fn destroy(&mut self) {
        self.renderer.destroy();
    }

    pub fn draw(&mut self, scene: &Scene) {
        self.renderer.draw(scene);
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn render_to_image(&mut self, _scene: &Scene) -> Result<RgbaImage> {
        anyhow::bail!("render_to_image is not supported for the macOS wgpu renderer");
    }
}

#[derive(Clone, Debug)]
struct RawWindow {
    native_view: NonNull<c_void>,
}

// SAFETY: `native_view` is an opaque NSView pointer borrowed from the platform window.
// We only pass it through raw-window-handle to create the WGPU surface and do not mutate it.
unsafe impl Send for RawWindow {}
unsafe impl Sync for RawWindow {}

impl rwh::HasWindowHandle for RawWindow {
    fn window_handle(&self) -> Result<rwh::WindowHandle<'_>, rwh::HandleError> {
        // SAFETY: `native_view` points to an NSView that outlives this raw handle.
        unsafe {
            Ok(rwh::WindowHandle::borrow_raw(rwh::RawWindowHandle::AppKit(
                rwh::AppKitWindowHandle::new(self.native_view),
            )))
        }
    }
}

impl rwh::HasDisplayHandle for RawWindow {
    fn display_handle(&self) -> Result<rwh::DisplayHandle<'_>, rwh::HandleError> {
        // SAFETY: AppKit display handles do not borrow data.
        unsafe {
            Ok(rwh::DisplayHandle::borrow_raw(
                rwh::RawDisplayHandle::AppKit(rwh::AppKitDisplayHandle::new()),
            ))
        }
    }
}
