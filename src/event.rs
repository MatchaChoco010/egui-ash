#[cfg(feature = "accesskit")]
use egui_winit::accesskit_winit;
pub use egui_winit::winit;

pub enum AppEvent {
    NewEvents(winit::event::StartCause),
    Suspended,
    Resumed,
    MainEventsCleared,
    RedrawEventsCleared,
    LoopDestroyed,
}

pub enum Event<'a> {
    DeferredViewportCreated {
        viewport_id: egui::ViewportId,
        window: &'a winit::window::Window,
    },
    ViewportEvent {
        viewport_id: egui::ViewportId,
        event: winit::event::WindowEvent<'a>,
    },
    AppEvent {
        event: AppEvent,
    },
    DeviceEvent {
        device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    },
    #[cfg(feature = "accesskit")]
    AccessKitActionRequest(accesskit_winit::ActionRequestEvent),
}
