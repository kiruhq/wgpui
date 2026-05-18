use crate::{
    App, Bounds, Element, ElementId, GlobalElementId, InspectorElementId, IntoElement, LayoutId,
    Pixels, Style, StyleRefinement, Styled, Window,
};
use refineable::Refineable as _;
use std::{any::Any, sync::Arc};

/// A layout element intended for renderer-owned surfaces.
pub struct ExternalSurface {
    handle: Option<Arc<dyn Any>>,
    style: StyleRefinement,
}

/// Construct an external renderer surface element.
pub fn external_surface(handle: Arc<dyn Any>) -> ExternalSurface {
    ExternalSurface {
        handle: Some(handle),
        style: StyleRefinement::default(),
    }
}

impl IntoElement for ExternalSurface {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Element for ExternalSurface {
    type RequestLayoutState = Style;
    type PrepaintState = ();

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        let mut style = Style::default();
        style.refine(&self.style);
        let layout_id = window.request_layout(style.clone(), [], cx);
        (layout_id, style)
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        _window: &mut Window,
        _cx: &mut App,
    ) -> Self::PrepaintState {
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _style: &mut Style,
        _prepaint: &mut Self::PrepaintState,
        window: &mut Window,
        _cx: &mut App,
    ) {
        if let Some(handle) = self.handle.take() {
            window.paint_layer(bounds, |window| {
                window.paint_external_surface(bounds, handle);
            });
        }
    }
}

impl Styled for ExternalSurface {
    fn style(&mut self) -> &mut StyleRefinement {
        &mut self.style
    }
}
