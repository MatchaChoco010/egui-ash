use ash::extensions::khr::Swapchain;
use egui_winit::winit::{self, event_loop::EventLoopBuilder};
use raw_window_handle::HasRawDisplayHandle;
use std::{ffi::CStr, mem::ManuallyDrop};

use crate::{
    app::{App, AppCreator, CreationContext},
    event,
    integration::{Integration, IntegrationEvent},
    renderer::ImageRegistry,
    Allocator, Theme,
};
#[cfg(feature = "persistence")]
use crate::{storage, utils};

/// egui-ash run option.
pub struct RunOption {
    /// window clear color.
    pub clear_color: [f32; 4],
    /// viewport builder for root window.
    pub viewport_builder: Option<egui::ViewportBuilder>,
    /// follow system theme.
    pub follow_system_theme: bool,
    /// default theme.
    pub default_theme: Theme,
    #[cfg(feature = "persistence")]
    pub persistent_windows: bool,
    #[cfg(feature = "persistence")]
    pub persistent_egui_memory: bool,
}
impl Default for RunOption {
    fn default() -> Self {
        Self {
            clear_color: [0.0, 0.0, 0.0, 1.0],
            viewport_builder: None,
            follow_system_theme: true,
            default_theme: Theme::Light,
            #[cfg(feature = "persistence")]
            persistent_windows: true,
            #[cfg(feature = "persistence")]
            persistent_egui_memory: true,
        }
    }
}

///egui-ash run function.
///
/// ```
/// fn main() {
///     egui_winit_ash::run("my_app", MyAppCreator, RunOption::default());
/// }
/// ```
pub fn run<C: AppCreator<A> + 'static, A: Allocator + 'static>(
    app_id: impl Into<String>,
    creator: C,
    run_option: RunOption,
) {
    let app_id = app_id.into();

    let device_extensions = [Swapchain::name().to_owned()];

    let event_loop = EventLoopBuilder::<IntegrationEvent>::with_user_event().build();

    #[cfg(feature = "persistence")]
    let storage = storage::Storage::from_app_id(&app_id).expect("Failed to create storage");

    let context = egui::Context::default();
    #[cfg(feature = "persistence")]
    if run_option.persistent_egui_memory {
        if let Some(memory) = storage.get_egui_memory() {
            context.memory_mut(|m| *m = memory);
        }
    }

    context.set_embed_viewports(false);
    match run_option.default_theme {
        Theme::Light => {
            context.set_visuals(egui::Visuals::light());
        }
        Theme::Dark => {
            context.set_visuals(egui::Visuals::dark());
        }
    }

    #[allow(unused_mut)] // only mutable when persistence feature is enabled
    let main_window = if let Some(mut viewport_builder) = run_option.viewport_builder {
        #[cfg(feature = "persistence")]
        if run_option.persistent_windows {
            let window_settings = storage
                .get_windows()
                .and_then(|windows| windows.get(&egui::ViewportId::ROOT).map(|s| s.to_owned()))
                .map(|mut settings| {
                    let egui_zoom_factor = context.zoom_factor();
                    settings.clamp_size_to_sane_values(utils::largest_monitor_point_size(
                        egui_zoom_factor,
                        &event_loop,
                    ));
                    settings.clamp_position_to_monitors(egui_zoom_factor, &event_loop);
                    settings.to_owned()
                });

            if let Some(window_settings) = window_settings {
                viewport_builder = window_settings.initialize_viewport_builder(viewport_builder);
            }
        }

        egui_winit::create_winit_window_builder(
            &context,
            &event_loop,
            viewport_builder.with_visible(false),
        )
        .with_visible(false)
        .build(&event_loop)
        .unwrap()
    } else {
        winit::window::WindowBuilder::new()
            .with_title("egui-ash")
            .with_visible(false)
            .build(&event_loop)
            .unwrap()
    };

    let instance_extensions =
        ash_window::enumerate_required_extensions(event_loop.raw_display_handle()).unwrap();
    let instance_extensions = instance_extensions
        .into_iter()
        .map(|&ext| unsafe { CStr::from_ptr(ext).to_owned() })
        .collect::<Vec<_>>();

    let (image_registry, image_registry_receiver) = ImageRegistry::new();

    let cc = CreationContext {
        main_window: &main_window,
        #[cfg(feature = "persistence")]
        storage: &storage,
        context: context.clone(),
        required_instance_extensions: instance_extensions,
        required_device_extensions: device_extensions.into_iter().collect(),
        image_registry,
    };
    let (mut app, render_state) = creator.create(cc);

    // ManuallyDrop is required because the integration object needs to be dropped before
    // the app drops for gpu_allocator drop order reasons.
    let mut integration = ManuallyDrop::new(Integration::new(
        &app_id,
        &event_loop,
        context,
        main_window,
        render_state,
        run_option.clear_color,
        image_registry_receiver,
        #[cfg(feature = "persistence")]
        storage,
        #[cfg(feature = "persistence")]
        run_option.persistent_windows,
        #[cfg(feature = "persistence")]
        run_option.persistent_egui_memory,
    ));

    event_loop.run(move |event, event_loop, control_flow| match event {
        winit::event::Event::NewEvents(start_cause) => {
            match start_cause {
                winit::event::StartCause::ResumeTimeReached { .. } => {
                    integration.paint_all(event_loop, control_flow, &mut app);
                }
                _ => (),
            }
            let app_event = event::Event::AppEvent {
                event: event::AppEvent::NewEvents(start_cause),
            };
            app.handle_event(app_event);
        }
        winit::event::Event::WindowEvent {
            event, window_id, ..
        } => {
            let consumed = integration.handle_window_event(
                window_id,
                &event,
                control_flow,
                run_option.follow_system_theme,
            );
            if consumed {
                return;
            }

            let Some(viewport_id) = integration.viewport_id_from_window_id(window_id) else {
                return;
            };
            let viewport_event = event::Event::ViewportEvent { viewport_id, event };
            app.handle_event(viewport_event);
        }
        winit::event::Event::DeviceEvent { device_id, event } => {
            let device_event = event::Event::DeviceEvent { device_id, event };
            app.handle_event(device_event);
        }
        #[allow(unused_variables)] // only used when accesskit feature is enabled
        winit::event::Event::UserEvent(integration_event) => {
            #[cfg(feature = "accesskit")]
            {
                integration.handle_accesskit_event(
                    &integration_event.accesskit,
                    event_loop,
                    control_flow,
                    &mut app,
                );
                let user_event = event::Event::AccessKitActionRequest(integration_event.accesskit);
                app.handle_event(user_event);
            }
        }
        winit::event::Event::Suspended => {
            let app_event = event::Event::AppEvent {
                event: event::AppEvent::Suspended,
            };
            app.handle_event(app_event);
        }
        winit::event::Event::Resumed => {
            let app_event = event::Event::AppEvent {
                event: event::AppEvent::Resumed,
            };
            app.handle_event(app_event);
            integration.paint_all(event_loop, control_flow, &mut app);
        }
        winit::event::Event::MainEventsCleared => {
            let app_event = event::Event::AppEvent {
                event: event::AppEvent::MainEventsCleared,
            };
            app.handle_event(app_event);
        }
        winit::event::Event::RedrawRequested(window_id) => {
            integration.paint(event_loop, control_flow, window_id, &mut app);
        }
        winit::event::Event::RedrawEventsCleared => {
            let app_event = event::Event::AppEvent {
                event: event::AppEvent::RedrawEventsCleared,
            };
            app.handle_event(app_event);
        }
        winit::event::Event::LoopDestroyed => {
            let app_event = event::Event::AppEvent {
                event: event::AppEvent::LoopDestroyed,
            };
            app.handle_event(app_event);
            #[cfg(feature = "persistence")]
            integration.save(&mut app);
            integration.destroy();
            unsafe {
                ManuallyDrop::drop(&mut integration);
            }
        }
    });
}
