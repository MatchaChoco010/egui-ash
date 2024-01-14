use ash::{
    extensions::{
        ext::DebugUtils,
        khr::{Surface, Swapchain},
    },
    vk, Device, Entry, Instance,
};
use egui_ash::{
    raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle},
    winit, App, AppCreator, AshRenderState, CreationContext, HandleRedraw, RunOption, Theme,
};
use gpu_allocator::vulkan::*;
use std::{
    collections::HashSet,
    ffi::CString,
    mem::ManuallyDrop,
    sync::{Arc, Mutex},
};

mod renderer;
use renderer::Renderer;

const EGUI_ASH_EXAMPLE_ROTATION: &str = "ROTATION_Y";

struct MyApp {
    entry: Arc<Entry>,
    instance: Arc<Instance>,
    device: Arc<Device>,
    debug_utils_loader: DebugUtils,
    debug_messenger: vk::DebugUtilsMessengerEXT,
    physical_device: vk::PhysicalDevice,
    surface_loader: Arc<Surface>,
    swapchain_loader: Arc<Swapchain>,
    surface: vk::SurfaceKHR,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    allocator: ManuallyDrop<Arc<Mutex<Allocator>>>,

    renderer: Renderer,

    theme: Theme,
    text: String,
    rotate_y: f32,
}
impl App for MyApp {
    fn save(&mut self, _storage: &mut egui_ash::storage::Storage) {
        _storage.set_value(EGUI_ASH_EXAMPLE_ROTATION, &self.rotate_y);
    }
    fn ui(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("my_side_panel").show(&ctx, |ui| {
            ui.heading("Hello");
            ui.label("Hello egui!");
            ui.separator();
            ui.horizontal(|ui| {
                ui.label("Theme");
                let id = ui.make_persistent_id("theme_combo_box_side");
                egui::ComboBox::from_id_source(id)
                    .selected_text(format!("{:?}", self.theme))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.theme, Theme::Dark, "Dark");
                        ui.selectable_value(&mut self.theme, Theme::Light, "Light");
                    });
            });
            ui.separator();
            ui.hyperlink("https://github.com/emilk/egui");
            ui.separator();
            ui.text_edit_singleline(&mut self.text);
            ui.separator();
            ui.label("Rotate");
            ui.add(egui::widgets::Slider::new(
                &mut self.rotate_y,
                -180.0..=180.0,
            ));
        });
        egui::Window::new("My Window")
            .id(egui::Id::new("my_window"))
            .resizable(true)
            .scroll2([true, true])
            .show(&ctx, |ui| {
                ui.heading("Hello");
                ui.label("Hello egui!");
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("Theme");
                    let id = ui.make_persistent_id("theme_combo_box_window");
                    egui::ComboBox::from_id_source(id)
                        .selected_text(format!("{:?}", self.theme))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.theme, Theme::Dark, "Dark");
                            ui.selectable_value(&mut self.theme, Theme::Light, "Light");
                        });
                });
                ui.separator();
                ui.hyperlink("https://github.com/emilk/egui");
                ui.separator();
                ui.text_edit_singleline(&mut self.text);
                ui.separator();
                ui.label("Rotate");
                ui.add(egui::widgets::Slider::new(
                    &mut self.rotate_y,
                    -180.0..=180.0,
                ));
            });

        match self.theme {
            Theme::Dark => ctx.set_visuals(egui::style::Visuals::dark()),
            Theme::Light => ctx.set_visuals(egui::style::Visuals::light()),
        }
    }

    fn request_redraw(&mut self, _viewport_id: egui::ViewportId) -> HandleRedraw {
        HandleRedraw::Handle(Box::new({
            let renderer = self.renderer.clone();
            let rotate_y = self.rotate_y;
            move |size, egui_cmd| renderer.render(size.width, size.height, egui_cmd, rotate_y)
        }))
    }
}
impl Drop for MyApp {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();

            self.renderer.destroy();

            self.device.destroy_command_pool(self.command_pool, None);
            self.surface_loader.destroy_surface(self.surface, None);
            ManuallyDrop::drop(&mut self.allocator);
            self.device.destroy_device(None);
            if self.debug_messenger != vk::DebugUtilsMessengerEXT::null() {
                self.debug_utils_loader
                    .destroy_debug_utils_messenger(self.debug_messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

struct MyAppCreator;
impl MyAppCreator {
    const ENABLE_VALIDATION_LAYERS: bool = true;
    const VALIDATION: [&'static str; 1] = ["VK_LAYER_KHRONOS_validation"];

    unsafe extern "system" fn vulkan_debug_utils_callback(
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        message_types: vk::DebugUtilsMessageTypeFlagsEXT,
        p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
        _p_user_data: *mut std::ffi::c_void,
    ) -> vk::Bool32 {
        let severity = match message_severity {
            vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => "[VERBOSE]",
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => "[WARNING]",
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => "[ERROR]",
            vk::DebugUtilsMessageSeverityFlagsEXT::INFO => "[INFO]",
            _ => panic!("[UNKNOWN]"),
        };
        let types = match message_types {
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "[GENERAL]",
            vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "[PERFORMANCE]",
            vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "[VALIDATION]",
            _ => panic!("[UNKNOWN]"),
        };
        let message = std::ffi::CStr::from_ptr((*p_callback_data).p_message);
        println!("[DEBUG]{}{}{:?}", severity, types, message);

        vk::FALSE
    }

    fn create_entry() -> Entry {
        Entry::linked()
    }

    fn create_instance(
        required_instance_extensions: &[CString],
        entry: &Entry,
    ) -> (Instance, DebugUtils, vk::DebugUtilsMessengerEXT) {
        let mut debug_utils_messenger_create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .flags(vk::DebugUtilsMessengerCreateFlagsEXT::empty())
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                // | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                // | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
            )
            .pfn_user_callback(Some(Self::vulkan_debug_utils_callback))
            .build();

        let app_name = std::ffi::CString::new("egui-winit-ash example simple").unwrap();
        let app_info = vk::ApplicationInfo::builder()
            .application_name(&app_name)
            .application_version(vk::make_api_version(1, 0, 0, 0))
            .api_version(vk::API_VERSION_1_0);
        let mut extension_names = vec![DebugUtils::name().as_ptr()];
        for ext in required_instance_extensions {
            let name = ext.as_ptr();
            extension_names.push(name);
        }
        let raw_layer_names = Self::VALIDATION
            .iter()
            .map(|l| std::ffi::CString::new(*l).unwrap())
            .collect::<Vec<_>>();
        let layer_names = raw_layer_names
            .iter()
            .map(|l| l.as_ptr())
            .collect::<Vec<*const i8>>();
        let instance_create_info = vk::InstanceCreateInfo::builder()
            .push_next(&mut debug_utils_messenger_create_info)
            .application_info(&app_info)
            .enabled_extension_names(&extension_names);
        let instance_create_info = if Self::ENABLE_VALIDATION_LAYERS {
            instance_create_info.enabled_layer_names(&layer_names)
        } else {
            instance_create_info
        };
        let instance = unsafe {
            entry
                .create_instance(&instance_create_info, None)
                .expect("Failed to create instance")
        };

        // setup debug utils
        let debug_utils_loader = DebugUtils::new(&entry, &instance);
        let debug_messenger = if Self::ENABLE_VALIDATION_LAYERS {
            unsafe {
                debug_utils_loader
                    .create_debug_utils_messenger(&debug_utils_messenger_create_info, None)
                    .expect("Failed to create debug utils messenger")
            }
        } else {
            vk::DebugUtilsMessengerEXT::null()
        };

        (instance, debug_utils_loader, debug_messenger)
    }

    fn create_surface_loader(entry: &Entry, instance: &Instance) -> Surface {
        Surface::new(&entry, &instance)
    }

    fn create_swapchain_loader(instance: &Instance, device: &Device) -> Swapchain {
        Swapchain::new(&instance, &device)
    }

    fn create_surface(
        entry: &Entry,
        instance: &Instance,
        window: &winit::window::Window,
    ) -> vk::SurfaceKHR {
        unsafe {
            ash_window::create_surface(
                entry,
                instance,
                window.raw_display_handle(),
                window.raw_window_handle(),
                None,
            )
            .expect("Failed to create surface")
        }
    }

    fn create_physical_device(
        instance: &Instance,
        surface_loader: &Surface,
        surface: vk::SurfaceKHR,
        required_device_extensions: &[CString],
    ) -> (vk::PhysicalDevice, vk::PhysicalDeviceMemoryProperties, u32) {
        let mut queue_family_index: Option<usize> = None;

        let (physical_device, physical_device_memory_properties) = {
            let physical_devices = unsafe {
                instance
                    .enumerate_physical_devices()
                    .expect("Failed to enumerate physical devices")
            };
            let physical_device = physical_devices.into_iter().find(|physical_device| {
                let queue_families = unsafe {
                    instance.get_physical_device_queue_family_properties(*physical_device)
                };
                for (i, queue_family) in queue_families.iter().enumerate() {
                    let mut graphics_queue = false;
                    let mut present_queue = false;
                    if queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                        graphics_queue = true;
                    }
                    let present_support = unsafe {
                        surface_loader
                            .get_physical_device_surface_support(
                                *physical_device,
                                i as u32,
                                surface,
                            )
                            .unwrap()
                    };
                    if present_support {
                        present_queue = true;
                    }
                    if graphics_queue && present_queue {
                        queue_family_index = Some(i);
                        break;
                    }
                }
                let is_queue_family_supported = queue_family_index.is_some();

                // check device extensions
                let device_extensions = unsafe {
                    instance
                        .enumerate_device_extension_properties(*physical_device)
                        .unwrap()
                };
                let mut device_extensions_name = vec![];
                for device_extension in device_extensions {
                    let name = unsafe {
                        std::ffi::CStr::from_ptr(device_extension.extension_name.as_ptr())
                            .to_owned()
                    };
                    device_extensions_name.push(name);
                }
                let mut required_extensions = HashSet::new();
                for extension in required_device_extensions.iter() {
                    required_extensions.insert(extension.to_owned());
                }
                for extension_name in device_extensions_name {
                    required_extensions.remove(&extension_name);
                }
                let is_device_extension_supported = required_extensions.is_empty();

                // check swapchain support
                let surface_formats = unsafe {
                    surface_loader
                        .get_physical_device_surface_formats(*physical_device, surface)
                        .unwrap()
                };
                let surface_present_modes = unsafe {
                    surface_loader
                        .get_physical_device_surface_present_modes(*physical_device, surface)
                        .unwrap()
                };
                let is_swapchain_supported =
                    !surface_formats.is_empty() && !surface_present_modes.is_empty();

                is_queue_family_supported && is_swapchain_supported && is_device_extension_supported
            });
            let physical_device = physical_device.expect("Failed to get physical device");
            let physical_device_memory_properties =
                unsafe { instance.get_physical_device_memory_properties(physical_device) };

            (physical_device, physical_device_memory_properties)
        };

        (
            physical_device,
            physical_device_memory_properties,
            queue_family_index.unwrap() as u32,
        )
    }

    fn create_device(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        queue_family_index: u32,
        required_device_extensions: &[CString],
    ) -> (Device, vk::Queue) {
        let queue_priorities = [1.0_f32];
        let mut queue_create_infos = vec![];
        let queue_create_info = vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities)
            .build();
        queue_create_infos.push(queue_create_info);

        let physical_device_features = vk::PhysicalDeviceFeatures::builder().build();

        let enable_extension_names = required_device_extensions
            .iter()
            .map(|s| s.as_ptr())
            .collect::<Vec<_>>();

        // device create info
        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_create_infos)
            .enabled_features(&physical_device_features)
            .enabled_extension_names(&enable_extension_names);

        // create device
        let device = unsafe {
            instance
                .create_device(physical_device, &device_create_info, None)
                .expect("Failed to create device")
        };

        // get device queue
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        (device, queue)
    }

    fn create_command_pool(device: &Device, queue_family_index: u32) -> vk::CommandPool {
        let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family_index);
        unsafe {
            device
                .create_command_pool(&command_pool_create_info, None)
                .expect("Failed to create command pool")
        }
    }
}
impl AppCreator<Arc<Mutex<Allocator>>> for MyAppCreator {
    type App = MyApp;

    fn create(&self, cc: CreationContext) -> (Self::App, AshRenderState<Arc<Mutex<Allocator>>>) {
        // create vk objects
        let entry = Self::create_entry();
        let (instance, debug_utils_loader, debug_messenger) =
            Self::create_instance(&cc.required_instance_extensions, &entry);
        let surface_loader = Self::create_surface_loader(&entry, &instance);
        let surface = Self::create_surface(&entry, &instance, cc.main_window);
        let (physical_device, _physical_device_memory_properties, queue_family_index) =
            Self::create_physical_device(
                &instance,
                &surface_loader,
                surface,
                &cc.required_device_extensions,
            );
        let (device, queue) = Self::create_device(
            &instance,
            physical_device,
            queue_family_index,
            &cc.required_device_extensions,
        );
        let swapchain_loader = Self::create_swapchain_loader(&instance, &device);
        let command_pool = Self::create_command_pool(&device, queue_family_index);

        // create allocator
        let allocator = {
            Allocator::new(&AllocatorCreateDesc {
                instance: instance.clone(),
                device: device.clone(),
                physical_device,
                debug_settings: Default::default(),
                buffer_device_address: false,
                allocation_sizes: Default::default(),
            })
            .expect("Failed to create allocator")
        };
        let allocator = Arc::new(Mutex::new(allocator));

        // setup context
        cc.context.set_visuals(egui::style::Visuals::dark());

        let device = Arc::new(device);
        let surface_loader = Arc::new(surface_loader);
        let swapchain_loader = Arc::new(swapchain_loader);
        let app = MyApp {
            entry: Arc::new(entry),
            instance: Arc::new(instance),
            device: device.clone(),
            debug_utils_loader,
            debug_messenger,
            physical_device,
            surface_loader: surface_loader.clone(),
            swapchain_loader: swapchain_loader.clone(),
            surface,
            queue,
            command_pool,
            allocator: ManuallyDrop::new(allocator.clone()),

            renderer: Renderer::new(
                physical_device,
                device.clone(),
                surface_loader.clone(),
                swapchain_loader.clone(),
                allocator.clone(),
                surface,
                queue_family_index,
                queue,
                command_pool,
                1000,
                800,
            ),

            theme: if cc.context.style().visuals.dark_mode {
                Theme::Dark
            } else {
                Theme::Light
            },
            text: String::from("Hello text!"),
            rotate_y: cc.storage.get_value(EGUI_ASH_EXAMPLE_ROTATION).unwrap_or_default(),
        };
        let ash_render_state = AshRenderState {
            entry: app.entry.clone(),
            instance: app.instance.clone(),
            physical_device: app.physical_device,
            device,
            surface_loader: app.surface_loader.clone(),
            swapchain_loader: app.swapchain_loader.clone(),
            queue: app.queue,
            queue_family_index,
            command_pool: app.command_pool,
            allocator: allocator.clone(),
        };

        (app, ash_render_state)
    }
}

fn main() {
    egui_ash::run(
        "egui-ash-vulkan",
        MyAppCreator,
        RunOption {
            viewport_builder: Some(
                egui::ViewportBuilder::default()
                    .with_title("egui-winit-ash")
                    .with_inner_size(egui::vec2(1000.0, 800.0)),
            ),
            follow_system_theme: false,
            default_theme: Theme::Dark,
            ..Default::default()
        },
    )
}
