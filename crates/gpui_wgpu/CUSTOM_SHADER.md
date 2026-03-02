# GPUI WGPU Custom Shader

This documents the custom shader support in `gpui_wgpu`.

## Rendering Model

Custom shaders run as an extra render pass after normal GPUI scene rendering.

Order:
1. GPUI scene draws (`quads`, `text`, `sprites`, etc.).
2. Custom shader pass draws into the same surface with `LoadOp::Load`.

This makes custom shaders good for overlays, post effects, and region-based effects.

## Rust API

Core types and functions:
- `CustomShaderDescriptor`
- `WgpuCustomShader`
- `WgpuRenderer::create_custom_shader(...)`
- `WgpuRenderer::update_custom_shader_uniforms(...)`
- `WgpuRenderer::draw_with_custom_shader(...)`

There is also a global mode:
- `set_global_custom_shader(...)`
- `set_global_custom_shader_enabled(...)`
- `set_global_custom_shader_paused(...)`
- `set_global_custom_shader_time_scale(...)`
- `set_global_custom_shader_uniform_bytes(...)`

When global mode is configured, `WgpuRenderer::draw(...)` automatically applies the custom pass.

Named shader-surface mode:
- `register_named_custom_shader(...)`
- `unregister_named_custom_shader(...)`
- `submit_render_primitive(...)`
- `ShaderSurfaceDraw`

With this mode, shaders are assigned by key per surface draw, so different UI regions can use different shaders.

## Shader Contract

Current contract for custom passes:
- No vertex buffers are bound.
- If uniforms are provided, they are bound at `@group(0) @binding(0)`.
- Draw call is `draw(0..vertex_count, 0..instance_count)`.
- You should typically generate geometry procedurally with `@builtin(vertex_index)`.

Recommended minimum WGSL pattern:

```wgsl
@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    return vec4<f32>(p[i], 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}
```

## Uniforms

`update_custom_shader_uniforms` requires exact byte size match.

- If shader expects a uniform struct of `N` bytes, your Rust byte slice must be exactly `N`.
- Empty uniform buffers are rejected.

Global mode convenience:
- If `animate_uniforms_with_time = true`, the first 16 bytes are updated each frame as:
  - `time_seconds: f32`
  - `surface_width: f32`
  - `surface_height: f32`
  - `reserved: f32`
- Extra bytes after the first 16 are preserved, so you can pass custom fields (for example a shader viewport rectangle).

## Embedding In A UI Element

You can use GPUI's `shader_surface(...)` element:
1. In `shader_surface` prepaint, read the element `bounds`.
2. Convert bounds to normalized window coordinates.
3. Submit a `RenderPrimitive` containing a `ShaderSurfaceDraw` with `shader_key`, `normalized_bounds`, and optional `uniform_bytes`.
4. In WGSL, discard or output transparent outside that rectangle (or rely on renderer scissor).

This lets shader output track a real layout region rather than hardcoded screen UV values.

## Blend and Alpha Behavior

You can override blend state in `CustomShaderDescriptor.blend_state`.

Important for preserving UI/text visibility:
- Outside your effect region, return transparent output (`alpha = 0.0`), not opaque color.
- If you draw opaque full-screen color, you will hide underlying text/UI.
- For most overlays, default alpha blending is the correct choice.

## Video Surface Notes

For a future video surface style effect:
- Use screen-space clipping in fragment shader (for example with `@builtin(position)` and a viewport rect uniform).
- Output transparent pixels outside the video rectangle.
- Keep compositing in shader for YUV->RGB or color grading if needed.
- If you need texture sampling from a video frame, the current custom pass API does not yet expose custom texture bindings; that requires API extension.

## Example

See:
- `examples/gpui-example/src/main.rs`

That example registers a named shader and assigns it to a `shader_surface` element via:
- `register_named_custom_shader(...)`
- `submit_render_primitive(...)`
