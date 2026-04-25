use anyhow::Result;
use gpui::{DevicePixels, GpuSpecs, PlatformAtlas, Scene, Size};
use gpui_wgpu::{GpuContext, WgpuRenderer, WgpuSurfaceConfig};
use raw_window_handle as rwh;
use std::{num::NonZeroIsize, sync::Arc};
use windows::Win32::Foundation::HWND;

pub(crate) type Context = GpuContext;
pub(crate) type Renderer = WgpuWinRenderer;

pub(crate) fn new_renderer(
    context: Context,
    hwnd: HWND,
    bounds: Size<f32>,
    transparent: bool,
) -> Renderer {
    let raw_window = RawWindow { hwnd };
    let config = WgpuSurfaceConfig {
        size: Size {
            width: DevicePixels(bounds.width.max(1.0).round() as i32),
            height: DevicePixels(bounds.height.max(1.0).round() as i32),
        },
        transparent,
        preferred_present_mode: None,
    };

    let renderer = match WgpuRenderer::new(context, &raw_window, config, None) {
        Ok(renderer) => renderer,
        Err(error) => {
            log::error!("Failed to initialize wgpu renderer: {error:#}");
            std::process::exit(1);
        }
    };

    WgpuWinRenderer { renderer }
}

pub(crate) struct WgpuWinRenderer {
    renderer: WgpuRenderer,
}

impl WgpuWinRenderer {
    pub fn sprite_atlas(&self) -> Arc<dyn PlatformAtlas> {
        self.renderer.sprite_atlas().clone()
    }

    pub fn resize(&mut self, size: Size<DevicePixels>) -> Result<()> {
        self.renderer.update_drawable_size(size);
        Ok(())
    }

    pub fn draw(
        &mut self,
        scene: &Scene,
        _background: gpui::WindowBackgroundAppearance,
    ) -> Result<()> {
        self.renderer.draw(scene);
        Ok(())
    }

    pub fn gpu_specs(&self) -> Result<GpuSpecs> {
        Ok(self.renderer.gpu_specs())
    }

    pub fn mark_drawable(&mut self) {
        // No-op for wgpu renderer — it's always drawable.
    }

    pub fn handle_device_lost(&mut self, _devices: &super::DirectXDevices) -> Result<()> {
        // wgpu handles device lost internally
        Ok(())
    }

    pub fn destroy(&mut self) {
        self.renderer.destroy();
    }
}

#[derive(Clone, Debug)]
struct RawWindow {
    hwnd: HWND,
}

// SAFETY: HWND is an opaque handle borrowed from the platform window.
// We only pass it through raw-window-handle to create the wgpu surface.
unsafe impl Send for RawWindow {}
unsafe impl Sync for RawWindow {}

impl rwh::HasWindowHandle for RawWindow {
    fn window_handle(&self) -> Result<rwh::WindowHandle<'_>, rwh::HandleError> {
        let raw = rwh::Win32WindowHandle::new(unsafe {
            NonZeroIsize::new_unchecked(self.hwnd.0 as isize)
        })
        .into();
        Ok(unsafe { rwh::WindowHandle::borrow_raw(raw) })
    }
}

impl rwh::HasDisplayHandle for RawWindow {
    fn display_handle(&self) -> Result<rwh::DisplayHandle<'_>, rwh::HandleError> {
        Ok(unsafe {
            rwh::DisplayHandle::borrow_raw(rwh::RawDisplayHandle::Windows(
                rwh::WindowsDisplayHandle::new(),
            ))
        })
    }
}
