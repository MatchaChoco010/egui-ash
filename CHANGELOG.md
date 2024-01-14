# Change Log

## [0.4.0] - 2024-01-14
### Added
- `egui_cmd.swapchain_recreate_required()` for change scale factor etc.

### Changed
- exit signal now receive `std::process::ExitCode`.

### Fixed
- fix error for ui zoom.
- fix error by forgetting to destroy image and image view when zoom factor changes.
- fix now restore main window position and size when `persistent_windows` is `true`.
  - Note: there is currently a bug in egui itself in saving the scale factor and window position. https://github.com/emilk/egui/issues/3797

### Update
- update egui from 0.24.2 to 0.25.0.
- update egui-winit from 0.24.1 to 0.25.0.

## [0.3.0] - 2024-01-05
### Added
- add exit_signal API to close exit app in code.
- add ability to change `vk::PresentModeKHR`.

### Fixed
- fix error in unregister_user_texture.

### Update
- update gpu-allocator from 0.24.0 to 0.25.0.

## [0.2.0] - 2024-01-01
### Changed
- remove Arc from AshRenderState.
- control flow to poll and present mode to FIFO.
- update example code.

### Fixed
- fix unused import.
- remove unnecessary build() of ash method.
- fix error when import spirv binary in examples.

## [0.1.1] - 2023-12-14
### Fixed
- fix README.md documentation.

## [0.1.0] - 2023-12-14
### Added
- initial release
