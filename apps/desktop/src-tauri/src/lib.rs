use std::{
    env,
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    thread,
    time::Duration,
};

use serde::{Deserialize, Serialize};
use tauri::{
    AppHandle, Emitter, Manager, PhysicalPosition, PhysicalSize, Position, Size, State, WebviewUrl,
    WebviewWindow, WebviewWindowBuilder,
};

const WALLPAPER_LABEL: &str = "wallpaper";
const GENERATOR_BASE_URL: &str = "http://127.0.0.1:4173/api/jobs/";

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct WallpaperConfig {
    scene_path: String,
    background_path: Option<String>,
    intensity: f64,
    depth_gain: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
struct CursorPayload {
    x: f64,
    y: f64,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct SettingsPayload {
    intensity: f64,
    depth_gain: f64,
}

#[derive(Clone, Serialize)]
struct StatusPayload {
    state: String,
    message: Option<String>,
}

#[derive(Default)]
struct WallpaperState {
    config: Mutex<Option<WallpaperConfig>>,
    cursor_generation: Arc<AtomicU64>,
}

#[derive(Default)]
struct GeneratorState {
    child: Mutex<Option<Child>>,
}

impl Drop for GeneratorState {
    fn drop(&mut self) {
        if let Ok(child) = self.child.get_mut()
            && let Some(child) = child.as_mut()
        {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

#[tauri::command]
fn start_scene_generator(state: State<'_, GeneratorState>) -> Result<(), String> {
    let mut child = state.child.lock().map_err(|_| "模型服务状态锁异常。")?;
    if let Some(process) = child.as_mut() {
        if process
            .try_wait()
            .map_err(|error| error.to_string())?
            .is_none()
        {
            return Ok(());
        }
        *child = None;
    }

    let root = find_generator_root()
        .ok_or("没有找到本机 SHARP 运行环境。请在项目目录运行 npm run setup:sharp。")?;
    let python = root.join(".venv/bin/python");
    let server = root.join("sharp_server.py");
    let process = Command::new(&python)
        .arg(&server)
        .current_dir(&root)
        .env("PORT", "4173")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| format!("无法启动本机 SHARP 服务：{error}"))?;
    *child = Some(process);
    Ok(())
}

#[tauri::command]
fn apply_wallpaper(
    app: AppHandle,
    state: State<'_, WallpaperState>,
    mut config: WallpaperConfig,
) -> Result<(), String> {
    config.scene_path = validate_scene_source(&config.scene_path)?;
    config.background_path = config
        .background_path
        .as_deref()
        .map(|path| validate_file(path, &["jpg", "jpeg", "png", "webp"]))
        .transpose()?;
    config.intensity = config.intensity.clamp(0.0, 1.5);
    config.depth_gain = config.depth_gain.clamp(0.4, 2.2);

    state.cursor_generation.fetch_add(1, Ordering::SeqCst);
    if let Some(existing) = app.get_webview_window(WALLPAPER_LABEL) {
        existing.close().map_err(|error| error.to_string())?;
    }
    *state.config.lock().map_err(|_| "壁纸状态锁异常。")? = Some(config);

    let monitor = app
        .primary_monitor()
        .map_err(|error| error.to_string())?
        .ok_or("没有检测到主显示器。")?;
    let scale_factor = monitor.scale_factor();
    let logical_position = monitor.position().to_logical::<f64>(scale_factor);
    let logical_size = monitor.size().to_logical::<f64>(scale_factor);
    let window = WebviewWindowBuilder::new(
        &app,
        WALLPAPER_LABEL,
        WebviewUrl::App("wallpaper.html".into()),
    )
    .title("Spatial Wallpaper Renderer")
    .decorations(false)
    .resizable(false)
    .focusable(false)
    .skip_taskbar(true)
    .shadow(false)
    .always_on_bottom(true)
    .visible(false)
    .position(logical_position.x, logical_position.y)
    .inner_size(logical_size.width, logical_size.height)
    .build()
    .map_err(|error| error.to_string())?;

    fit_wallpaper_to_primary_monitor(&app, &window)?;
    window
        .set_ignore_cursor_events(true)
        .map_err(|error| error.to_string())?;
    window
        .set_visible_on_all_workspaces(true)
        .map_err(|error| error.to_string())?;
    configure_desktop_window(&window)?;
    Ok(())
}

#[tauri::command]
fn get_wallpaper_config(state: State<'_, WallpaperState>) -> Result<WallpaperConfig, String> {
    state
        .config
        .lock()
        .map_err(|_| "壁纸状态锁异常。".to_string())?
        .clone()
        .ok_or_else(|| "没有待加载的壁纸配置。".to_string())
}

#[tauri::command]
fn update_wallpaper_settings(
    app: AppHandle,
    state: State<'_, WallpaperState>,
    intensity: f64,
    depth_gain: f64,
) -> Result<(), String> {
    let payload = SettingsPayload {
        intensity: intensity.clamp(0.0, 1.5),
        depth_gain: depth_gain.clamp(0.4, 2.2),
    };
    if let Some(config) = state
        .config
        .lock()
        .map_err(|_| "壁纸状态锁异常。")?
        .as_mut()
    {
        config.intensity = payload.intensity;
        config.depth_gain = payload.depth_gain;
    }
    app.emit_to(WALLPAPER_LABEL, "wallpaper-settings", payload)
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn wallpaper_ready(app: AppHandle, state: State<'_, WallpaperState>) -> Result<(), String> {
    let window = app
        .get_webview_window(WALLPAPER_LABEL)
        .ok_or("壁纸窗口不存在。")?;
    window.show().map_err(|error| error.to_string())?;
    fit_wallpaper_to_primary_monitor(&app, &window)?;
    order_desktop_window_back(&window)?;

    let surface_message = wallpaper_surface_description(&app, &window).ok();

    let generation = state.cursor_generation.fetch_add(1, Ordering::SeqCst) + 1;
    start_cursor_loop(
        app.clone(),
        generation,
        Arc::clone(&state.cursor_generation),
    );
    app.emit_to(
        "main",
        "wallpaper-status",
        StatusPayload {
            state: "running".into(),
            message: surface_message,
        },
    )
    .map_err(|error| error.to_string())
}

#[tauri::command]
fn wallpaper_failed(
    app: AppHandle,
    state: State<'_, WallpaperState>,
    message: String,
) -> Result<(), String> {
    state.cursor_generation.fetch_add(1, Ordering::SeqCst);
    if let Some(window) = app.get_webview_window(WALLPAPER_LABEL) {
        let _ = window.close();
    }
    app.emit_to(
        "main",
        "wallpaper-status",
        StatusPayload {
            state: "error".into(),
            message: Some(message),
        },
    )
    .map_err(|error| error.to_string())
}

#[tauri::command]
fn stop_wallpaper(app: AppHandle, state: State<'_, WallpaperState>) -> Result<(), String> {
    state.cursor_generation.fetch_add(1, Ordering::SeqCst);
    if let Some(window) = app.get_webview_window(WALLPAPER_LABEL) {
        window.close().map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn validate_file(path: &str, allowed_extensions: &[&str]) -> Result<String, String> {
    let file = PathBuf::from(path);
    if !file.is_file() {
        return Err(format!("文件不存在：{}", file.display()));
    }
    let extension = file
        .extension()
        .and_then(|value| value.to_str())
        .map(str::to_ascii_lowercase)
        .ok_or("文件缺少扩展名。")?;
    if !allowed_extensions.contains(&extension.as_str()) {
        return Err(format!("不支持的文件格式：.{extension}"));
    }
    Ok(path.to_string())
}

fn validate_scene_source(source: &str) -> Result<String, String> {
    if let Some(job_path) = source.strip_prefix(GENERATOR_BASE_URL)
        && let Some(job_id) = job_path.strip_suffix("/result")
        && job_id.len() == 32
        && job_id.bytes().all(|value| value.is_ascii_hexdigit())
    {
        return Ok(source.to_string());
    }
    validate_file(source, &["sog", "spz", "ply"])
}

fn find_generator_root() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(root) = env::var_os("IMAGE_FAKE_3D_ROOT") {
        candidates.push(PathBuf::from(root));
    }
    candidates.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    if let Ok(path) = env::current_dir() {
        candidates.push(path);
    }
    if let Ok(path) = env::current_exe() {
        candidates.push(path);
    }

    candidates.into_iter().find_map(|candidate| {
        candidate.ancestors().find_map(|ancestor| {
            let root = ancestor.to_path_buf();
            (root.join("sharp_server.py").is_file()
                && root.join(".venv/bin/python").is_file()
                && root.join(".venv/bin/sharp").is_file())
            .then_some(root)
        })
    })
}

fn fit_wallpaper_to_primary_monitor(app: &AppHandle, window: &WebviewWindow) -> Result<(), String> {
    let monitor = app
        .primary_monitor()
        .map_err(|error| error.to_string())?
        .ok_or("没有检测到主显示器。")?;
    window
        .set_position(Position::Physical(PhysicalPosition::new(
            monitor.position().x,
            monitor.position().y,
        )))
        .map_err(|error| error.to_string())?;
    window
        .set_size(Size::Physical(PhysicalSize::new(
            monitor.size().width,
            monitor.size().height,
        )))
        .map_err(|error| error.to_string())
}

fn wallpaper_surface_description(
    app: &AppHandle,
    window: &WebviewWindow,
) -> Result<String, String> {
    let monitor = app
        .primary_monitor()
        .map_err(|error| error.to_string())?
        .ok_or("没有检测到主显示器。")?;
    let size = window.inner_size().map_err(|error| error.to_string())?;
    Ok(format!(
        "全屏画布 {}×{} px · Retina {:.1}x；移动鼠标可改变整个画面的视角。",
        size.width,
        size.height,
        monitor.scale_factor(),
    ))
}

fn start_cursor_loop(app: AppHandle, generation: u64, current_generation: Arc<AtomicU64>) {
    thread::spawn(move || {
        let mut previous = CursorPayload { x: 9.0, y: 9.0 };
        while current_generation.load(Ordering::SeqCst) == generation {
            let Some(window) = app.get_webview_window(WALLPAPER_LABEL) else {
                break;
            };
            if let (Ok(cursor), Ok(position), Ok(size)) = (
                window.cursor_position(),
                window.outer_position(),
                window.outer_size(),
            ) {
                let payload = normalize_cursor(cursor, position, size);
                if (payload.x - previous.x).abs() > 0.001 || (payload.y - previous.y).abs() > 0.001
                {
                    let _ = window.emit("wallpaper-cursor", payload.clone());
                    previous = payload;
                }
            }
            thread::sleep(Duration::from_millis(16));
        }
    });
}

fn normalize_cursor(
    cursor: PhysicalPosition<f64>,
    window_position: PhysicalPosition<i32>,
    window_size: PhysicalSize<u32>,
) -> CursorPayload {
    let width = f64::from(window_size.width.max(1));
    let height = f64::from(window_size.height.max(1));
    CursorPayload {
        x: (((cursor.x - f64::from(window_position.x)) / width) * 2.0 - 1.0).clamp(-1.0, 1.0),
        y: (((cursor.y - f64::from(window_position.y)) / height) * 2.0 - 1.0).clamp(-1.0, 1.0),
    }
}

#[cfg(target_os = "macos")]
fn configure_desktop_window(window: &WebviewWindow) -> Result<(), String> {
    use objc2_app_kit::{NSWindow, NSWindowAnimationBehavior, NSWindowCollectionBehavior};
    use objc2_core_graphics::{CGWindowLevelForKey, CGWindowLevelKey};

    let native_window = window.ns_window().map_err(|error| error.to_string())? as usize;
    window
        .run_on_main_thread(move || unsafe {
            let native_window = &*(native_window as *const NSWindow);
            let desktop_level = CGWindowLevelForKey(CGWindowLevelKey::DesktopWindowLevelKey);
            native_window.setLevel((desktop_level + 1) as isize);
            native_window.setIgnoresMouseEvents(true);
            native_window.setHasShadow(false);
            native_window.setCanHide(false);
            native_window.setAnimationBehavior(NSWindowAnimationBehavior::None);
            native_window.setCollectionBehavior(
                NSWindowCollectionBehavior::CanJoinAllSpaces
                    | NSWindowCollectionBehavior::Stationary
                    | NSWindowCollectionBehavior::IgnoresCycle
                    | NSWindowCollectionBehavior::Transient,
            );
        })
        .map_err(|error| error.to_string())
}

#[cfg(not(target_os = "macos"))]
fn configure_desktop_window(_window: &WebviewWindow) -> Result<(), String> {
    Err("当前原型只实现了 macOS 桌面宿主。".into())
}

#[cfg(target_os = "macos")]
fn order_desktop_window_back(window: &WebviewWindow) -> Result<(), String> {
    use objc2_app_kit::NSWindow;

    let native_window = window.ns_window().map_err(|error| error.to_string())? as usize;
    window
        .run_on_main_thread(move || unsafe {
            let native_window = &*(native_window as *const NSWindow);
            native_window.orderBack(None);
        })
        .map_err(|error| error.to_string())
}

#[cfg(not(target_os = "macos"))]
fn order_desktop_window_back(_window: &WebviewWindow) -> Result<(), String> {
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .manage(GeneratorState::default())
        .manage(WallpaperState::default())
        .invoke_handler(tauri::generate_handler![
            apply_wallpaper,
            get_wallpaper_config,
            start_scene_generator,
            stop_wallpaper,
            update_wallpaper_settings,
            wallpaper_failed,
            wallpaper_ready,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Spatial Wallpaper Lab");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_cursor_around_window_center() {
        let payload = normalize_cursor(
            PhysicalPosition::new(600.0, 350.0),
            PhysicalPosition::new(100, -50),
            PhysicalSize::new(1000, 800),
        );
        assert_eq!(payload, CursorPayload { x: 0.0, y: 0.0 });
    }

    #[test]
    fn clamps_cursor_outside_wallpaper_bounds() {
        let payload = normalize_cursor(
            PhysicalPosition::new(-500.0, 2000.0),
            PhysicalPosition::new(0, 0),
            PhysicalSize::new(1920, 1080),
        );
        assert_eq!(payload, CursorPayload { x: -1.0, y: 1.0 });
    }

    #[test]
    fn rejects_missing_scene_files() {
        let result = validate_file("/definitely/missing/scene.sog", &["sog", "spz", "ply"]);
        assert!(result.unwrap_err().contains("文件不存在"));
    }

    #[test]
    fn accepts_only_local_generator_result_urls() {
        let valid = "http://127.0.0.1:4173/api/jobs/0123456789abcdef0123456789abcdef/result";
        assert_eq!(validate_scene_source(valid).unwrap(), valid);
        assert!(validate_scene_source("https://example.com/scene.sog").is_err());
        assert!(validate_scene_source("http://127.0.0.1:4173/api/health").is_err());
    }
}
