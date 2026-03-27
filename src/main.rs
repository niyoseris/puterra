use actix_web::{web, App, HttpServer, HttpResponse, Responder, get, post, delete};
use actix_files as fs;
use actix_multipart::Multipart;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream;
use std::sync::Mutex;
use std::collections::HashMap;
use sha2::{Sha256, Digest};
use uuid::Uuid;
use std::process::Command;
use printpdf::*;
use std::io::{BufWriter, Read, Write, Seek};
use regex::Regex;
// Import from the external image crate (same version as printpdf uses)
use ::image::DynamicImage;
use resvg::usvg::{Options, Tree};
use resvg::tiny_skia::Pixmap;

// ============================================================
// TYPES
// ============================================================

#[derive(Clone, Serialize, Deserialize)]
struct User {
    id: String,
    username: String,
    password_hash: String,
}

const USERS_PATH: &str = "data/users.json";

fn load_users() -> HashMap<String, User> {
    std::fs::read_to_string(USERS_PATH)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_users(users: &HashMap<String, User>) {
    std::fs::create_dir_all("data").ok();
    std::fs::write(USERS_PATH, serde_json::to_string_pretty(users).unwrap_or_default()).ok();
}

/// A share key lets a user share their LLM API access without exposing the real key
#[derive(Clone, Serialize, Deserialize)]
struct ShareKey {
    id: String,            // The token (sk_...)
    owner: String,         // Username who created it
    label: String,         // Human-readable description
    api_url: String,       // LLM endpoint URL
    model: String,         // Model name
    api_key: String,       // The actual API key (never returned to non-owners)
    active: bool,
    uses: u64,
    created_at: u64,
    #[serde(default)]
    max_uses: Option<u64>,
}

const SHARE_KEYS_PATH: &str = "data/share_keys.json";

fn load_share_keys() -> HashMap<String, ShareKey> {
    std::fs::read_to_string(SHARE_KEYS_PATH)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_share_keys(keys: &HashMap<String, ShareKey>) {
    std::fs::create_dir_all("data").ok();
    std::fs::write(SHARE_KEYS_PATH, serde_json::to_string_pretty(keys).unwrap_or_default()).ok();
}

fn default_timeout_agent() -> u64 { 300 }
fn default_timeout_web_fetch() -> u64 { 15 }
fn default_timeout_web_search() -> u64 { 15 }
fn default_timeout_image() -> u64 { 30 }
fn default_timeout_llm_test() -> u64 { 60 }
fn default_llm_temperature() -> f64 { 0.7 }
fn default_llm_max_tokens() -> u64 { 4096 }
fn default_chat_context_limit() -> usize { 200 }

/// Persistent settings stored in data/settings.json
#[derive(Clone, Serialize, Deserialize)]
struct Settings {
    // Local Ollama configuration
    llm_api_url_local: String,
    llm_model_local: String,
    llm_api_key_local: String,

    // Cloud Ollama configuration
    llm_api_url_cloud: String,
    llm_model_cloud: String,
    llm_api_key_cloud: String,

    // Active source selection
    llm_active_source: String,  // "local" or "cloud"

    // Legacy fields (kept for backward compatibility)
    #[serde(default)]
    llm_api_url: String,
    #[serde(default)]
    llm_model: String,
    #[serde(default)]
    llm_api_key: String,
    #[serde(default)]
    llm_provider: String,

    search_engine: String,
    max_agent_iterations: usize,
    shell_enabled: bool,
    admin_password: String,

    // Timeouts (seconds)
    #[serde(default = "default_timeout_agent")]
    timeout_agent: u64,
    #[serde(default = "default_timeout_web_fetch")]
    timeout_web_fetch: u64,
    #[serde(default = "default_timeout_web_search")]
    timeout_web_search: u64,
    #[serde(default = "default_timeout_image")]
    timeout_image: u64,
    #[serde(default = "default_timeout_llm_test")]
    timeout_llm_test: u64,

    // LLM generation parameters
    #[serde(default = "default_llm_temperature")]
    llm_temperature: f64,
    #[serde(default = "default_llm_max_tokens")]
    llm_max_tokens: u64,
    #[serde(default)]
    llm_think: bool,

    // Chat context window
    #[serde(default = "default_chat_context_limit")]
    chat_context_limit: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            // Local Ollama (default)
            llm_api_url_local: std::env::var("LLM_API_URL_LOCAL")
                .unwrap_or_else(|_| "http://localhost:11434/api".to_string()),
            llm_model_local: std::env::var("LLM_MODEL_LOCAL")
                .unwrap_or_else(|_| "llama3.2".to_string()),
            llm_api_key_local: std::env::var("LLM_API_KEY_LOCAL")
                .unwrap_or_default(),

            // Cloud Ollama
            llm_api_url_cloud: std::env::var("LLM_API_URL_CLOUD")
                .unwrap_or_else(|_| "https://ollama.com/api".to_string()),
            llm_model_cloud: std::env::var("LLM_MODEL_CLOUD")
                .unwrap_or_else(|_| "glm-5:cloud".to_string()),
            llm_api_key_cloud: std::env::var("LLM_API_KEY_CLOUD")
                .unwrap_or_default(),

            // Active source (default to local)
            llm_active_source: std::env::var("LLM_ACTIVE_SOURCE")
                .unwrap_or_else(|_| "local".to_string()),

            // Legacy fields (for backward compatibility)
            llm_api_url: std::env::var("LLM_API_URL")
                .or_else(|_| std::env::var("OLLAMA_API_URL"))
                .unwrap_or_else(|_| "http://localhost:11434/api".to_string()),
            llm_model: std::env::var("LLM_MODEL")
                .unwrap_or_else(|_| "llama3.2".to_string()),
            llm_api_key: std::env::var("LLM_API_KEY")
                .or_else(|_| std::env::var("OLLAMA_API_KEY"))
                .unwrap_or_default(),
            llm_provider: "ollama".to_string(),

            search_engine: "duckduckgo".to_string(),
            max_agent_iterations: 100,
            shell_enabled: true,
            admin_password: "changeme".to_string(),

            timeout_agent: 120,
            timeout_web_fetch: 15,
            timeout_web_search: 15,
            timeout_image: 30,
            timeout_llm_test: 60,

            llm_temperature: 0.7,
            llm_max_tokens: 4096,
            llm_think: false,

            chat_context_limit: 200,
        }
    }
}

const SETTINGS_PATH: &str = "data/settings.json";

fn load_settings() -> Settings {
    std::fs::read_to_string(SETTINGS_PATH)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_settings(settings: &Settings) {
    std::fs::create_dir_all("data").ok();
    std::fs::write(SETTINGS_PATH, serde_json::to_string_pretty(settings).unwrap_or_default()).ok();
}

#[derive(Clone)]
struct Session {
    token: String,
    username: String,
}

struct AppState {
    users: Mutex<HashMap<String, User>>,
    settings: Mutex<Settings>,
    sessions: Mutex<HashMap<String, Session>>,  // token -> Session
    share_keys: Mutex<HashMap<String, ShareKey>>,
}

#[derive(Serialize)]
struct FileEntry {
    name: String,
    is_dir: bool,
    size: u64,
}

#[derive(Deserialize)]
struct SignupRequest { username: String, password: String }
#[derive(Deserialize)]
struct LoginRequest { username: String, password: String }
#[derive(Deserialize)]
struct WebSearchRequest { query: String, max_results: Option<usize> }
#[derive(Deserialize)]
struct ShellRequest { command: String, cwd: Option<String>, #[serde(default)] username: String }
#[derive(Deserialize)]
struct FetchRequest { url: String }
#[derive(Deserialize)]
struct MemoryRequest { action: String, key: Option<String>, value: Option<String>, query: Option<String>, username: Option<String> }
#[derive(Deserialize)]
struct ChatRequest { message: String, model: Option<String> }
#[derive(Deserialize)]
struct CreateRequest { username: String, name: String, r#type: String }
#[derive(Deserialize)]
struct DeleteRequest { username: String, names: Vec<String> }
#[derive(Deserialize)]
struct RenameRequest { username: String, old_name: String, new_name: String }
#[derive(Deserialize)]
struct ReadRequest { username: String, name: String }
#[derive(Deserialize)]
struct WriteRequest { username: String, name: String, content: String }

// Agent types
#[derive(Deserialize)]
struct AgentRequest {
    message: String,
    history: Option<Vec<AgentChatMessage>>,
    model: Option<String>,
    username: Option<String>,
    #[serde(default)]
    share_key: Option<String>,
    #[serde(default)]
    conv_id: Option<String>,
}

#[derive(Deserialize)]
struct CreateShareKeyRequest {
    token: String,
    label: String,
    api_url: String,
    model: String,
    api_key: String,
    #[serde(default)]
    max_uses: Option<u64>,
}

#[derive(Deserialize, Serialize, Clone)]
struct AgentChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize, Clone)]
struct AgentStep {
    thinking: Option<String>,
    thought: String,
    action: Option<String>,
    action_input: Option<String>,
    observation: Option<String>,
}

// Search result from DuckDuckGo
#[derive(Serialize, Clone)]
struct SearchResult {
    title: String,
    url: String,
    snippet: String,
}

// ============================================================
// HELPERS
// ============================================================

fn hash_password(password: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(password.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn get_session_username(data: &web::Data<AppState>, token: &str) -> Option<String> {
    let sessions = data.sessions.lock().unwrap();
    sessions.get(token).map(|s| s.username.clone())
}

fn get_user_dir(username: &str) -> String {
    format!("data/users/{}", username)
}

/// Safe string truncation that respects UTF-8 char boundaries
fn safe_truncate(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Get a snapshot of current settings (reads from file each time for freshness)
fn current_settings() -> Settings {
    load_settings()
}

/// Convert HTML to plain text
fn html_to_text(html: &str) -> String {
    // Remove script, style, noscript blocks (no backreferences - Rust regex doesn't support them)
    let re_script = regex::Regex::new(r"(?is)<script[^>]*>.*?</script>").unwrap();
    let text = re_script.replace_all(html, "");
    let re_style = regex::Regex::new(r"(?is)<style[^>]*>.*?</style>").unwrap();
    let text = re_style.replace_all(&text, "");
    let re_noscript = regex::Regex::new(r"(?is)<noscript[^>]*>.*?</noscript>").unwrap();
    let text = re_noscript.replace_all(&text, "");

    let re_block = regex::Regex::new(r"(?i)</(p|div|h[1-6]|li|tr|br|hr)[^>]*>").unwrap();
    let text = re_block.replace_all(&text, "\n");

    let re_br = regex::Regex::new(r"(?i)<br\s*/?>").unwrap();
    let text = re_br.replace_all(&text, "\n");

    let re_tags = regex::Regex::new(r"<[^>]+>").unwrap();
    let text = re_tags.replace_all(&text, "");

    let text = text
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
        .replace("&#x27;", "'")
        .replace("&#x2F;", "/");

    let re_ws = regex::Regex::new(r"\n{3,}").unwrap();
    let text = re_ws.replace_all(&text, "\n\n");

    let re_spaces = regex::Regex::new(r"[ \t]+").unwrap();
    let text = re_spaces.replace_all(&text, " ");

    text.trim().to_string()
}

/// Load an image from a local path or URL
/// Convert SVG bytes to a DynamicImage by rendering to raster
fn svg_to_image(svg_bytes: &[u8], width: u32) -> Result<DynamicImage, String> {
    // Parse SVG
    let opt = Options::default();
    let tree = Tree::from_data(svg_bytes, &opt)
        .map_err(|e| format!("Failed to parse SVG: {}", e))?;

    // Calculate height to maintain aspect ratio
    let size = tree.size();
    let aspect = size.height() / size.width();
    let height = (width as f32 * aspect) as u32;

    // Create pixmap and render
    let mut pixmap = Pixmap::new(width, height)
        .ok_or("Failed to create pixmap for SVG rendering")?;

    resvg::render(&tree, resvg::usvg::Transform::identity(), &mut pixmap.as_mut());

    // Convert pixmap to RGBA bytes
    let rgba_data = pixmap.data();

    // Create image from RGBA bytes
    let img = ::image::RgbaImage::from_raw(width, height, rgba_data.to_vec())
        .ok_or("Failed to create image from SVG render")?;

    Ok(DynamicImage::ImageRgba8(img))
}

/// Load an image from a local path or URL (supports SVG, PNG, JPEG, GIF, WebP, BMP)
fn load_image(path: &str) -> Result<DynamicImage, String> {
    // Check if it's a URL
    if path.starts_with("http://") || path.starts_with("https://") {
        // Download image from URL in a separate thread to avoid blocking runtime issues
        let path_owned = path.to_string();
        let handle = std::thread::spawn(move || {
            // Create a client that looks like a real browser (required by many sites)
            let client = reqwest::blocking::Client::builder()
                .user_agent("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
                .timeout(std::time::Duration::from_secs(current_settings().timeout_image))
                .redirect(reqwest::redirect::Policy::limited(10))  // Follow redirects
                .build()
                .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

            let response = client.get(&path_owned)
                .header("Accept", "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8")
                .header("Accept-Language", "en-US,en;q=0.9")
                .header("Referer", "https://www.google.com/search?q=cyprus+map")
                .header("Cache-Control", "no-cache")
                .header("Pragma", "no-cache")
                .send()
                .map_err(|e| format!("Failed to download image: {}", e))?;

            // Check for HTTP errors
            let status = response.status();
            if !status.is_success() {
                return Err(format!("HTTP error {} - The image URL returned an error. Status: {}. Try a different image source.", status.as_u16(), status.as_str()));
            }

            // Check content type for SVG
            let is_svg = if let Some(content_type) = response.headers().get("content-type") {
                if let Ok(ct) = content_type.to_str() {
                    ct.contains("svg") || ct.contains("image/svg")
                } else {
                    false
                }
            } else {
                // Fallback to extension check
                path_owned.to_lowercase().contains(".svg")
            };

            let bytes = response.bytes()
                .map_err(|e| format!("Failed to read image bytes: {}", e))?;

            if is_svg {
                // Convert SVG to PNG (render at 800px width)
                eprintln!("DEBUG: Detected SVG from Content-Type header");
                svg_to_image(&bytes, 800)
            } else {
                // Check if content is HTML error page
                let is_html = bytes.starts_with(b"<!DOCTYPE") ||
                              bytes.starts_with(b"<html") ||
                              bytes.windows(50).any(|w| w.starts_with(b"<html") || w.starts_with(b"<title>"));

                if is_html {
                    Err("Server returned HTML error page (possibly blocked or not found). Try a different image URL.".to_string())
                } else {
                    // Check if content looks like SVG - search for <svg in first 2000 bytes
                    let search_len = bytes.len().min(2000);
                    let search_area = &bytes[..search_len];
                    let is_svg_content = search_area.windows(4).any(|w| w == b"<svg");

                    if is_svg_content {
                        eprintln!("DEBUG: Detected SVG from content (found <svg tag)");
                        svg_to_image(&bytes, 800)
                    } else {
                        eprintln!("DEBUG: Not SVG, trying as regular image. First 100 bytes: {:?}", &bytes[..bytes.len().min(100)]);
                        ::image::load_from_memory(&bytes)
                            .map_err(|e| format!("Failed to decode image (not SVG): {}", e))
                    }
                }
            }
        });
        handle.join().unwrap_or_else(|_| Err("Thread panicked while downloading image".to_string()))
    } else {
        // Try as local file path
        let expanded = if path.starts_with("~") {
            // Expand ~ to home directory
            if let Ok(home) = std::env::var("HOME") {
                path.replacen("~", &home, 1)
            } else {
                path.to_string()
            }
        } else {
            path.to_string()
        };

        // Try the path as-is first, then relative to data/ directory
        let paths_to_try = vec![
            expanded.clone(),
            format!("data/{}", expanded),
            format!("./{}", expanded),
        ];

        for try_path in paths_to_try {
            if let Ok(bytes) = std::fs::read(&try_path) {
                // Check if it's SVG by extension or content
                let is_svg = try_path.to_lowercase().ends_with(".svg") ||
                            try_path.to_lowercase().ends_with(".svgz") ||
                            bytes.starts_with(b"<svg") ||
                            (bytes.starts_with(b"<?xml") && bytes.windows(200).any(|w| w.starts_with(b"<svg")));

                if is_svg {
                    return svg_to_image(&bytes, 800);
                } else {
                    return ::image::load_from_memory(&bytes)
                        .map_err(|e| format!("Failed to decode image '{}': {}", try_path, e));
                }
            }
        }

        Err(format!("Image file not found: {}", path))
    }
}

/// Generate a PDF document from title and text content, save to given path
/// Supports markdown-style images: ![alt](path) where path can be local or URL
fn generate_pdf(title: &str, content: &str, output_path: &str) -> Result<String, String> {
    let (doc, page1, layer1) = PdfDocument::new(title, Mm(210.0), Mm(297.0), "Layer 1");

    // Try to find a system TrueType font that supports Unicode
    let font_paths = vec![
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ];

    let font_result = font_paths.iter().find_map(|path| {
        std::fs::read(path).ok().and_then(|bytes| {
            doc.add_external_font(bytes.as_slice()).ok()
        })
    });

    // Bold font paths
    let bold_font_paths = vec![
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ];

    let bold_font_result = bold_font_paths.iter().find_map(|path| {
        std::fs::read(path).ok().and_then(|bytes| {
            doc.add_external_font(bytes.as_slice()).ok()
        })
    });

    let font = match font_result {
        Some(f) => f,
        None => {
            // Fallback to built-in font (limited Unicode support)
            doc.add_builtin_font(BuiltinFont::Helvetica)
                .map_err(|e| format!("Font error: {}", e))?
        }
    };

    let bold_font = match bold_font_result {
        Some(f) => f,
        None => {
            doc.add_builtin_font(BuiltinFont::HelveticaBold)
                .map_err(|e| format!("Bold font error: {}", e))?
        }
    };

    let font_size_title: f32 = 18.0;
    let font_size_heading: f32 = 14.0;
    let font_size_body: f32 = 11.0;
    let line_height: f32 = 5.0; // mm between lines
    let margin_left: f32 = 20.0;
    let margin_right: f32 = 20.0;
    let margin_top: f32 = 25.0;
    let margin_bottom: f32 = 20.0;
    let page_width: f32 = 210.0;
    let page_height: f32 = 297.0;
    let max_text_width: f32 = page_width - margin_left - margin_right;
    let max_image_width: f32 = 170.0; // Max image width in mm
    let max_image_height: f32 = 200.0; // Max image height in mm

    // Approximate chars per line (rough estimate based on font size)
    let chars_per_line = (max_text_width / (font_size_body * 0.22)) as usize;

    let mut current_page = page1;
    let mut current_layer = layer1;
    let mut y_position: f32 = page_height - margin_top;

    // Helper: start new page
    let new_page = |doc: &PdfDocumentReference, y: &mut f32| -> (PdfPageIndex, PdfLayerIndex) {
        let (page, layer) = doc.add_page(Mm(210.0), Mm(297.0), "Layer 1");
        *y = page_height - margin_top;
        (page, layer)
    };

    // Helper: add image to PDF
    let add_image = |doc: &PdfDocumentReference,
                     page_idx: PdfPageIndex,
                     layer_idx: PdfLayerIndex,
                     img: &DynamicImage,
                     x: f32, y: f32,
                     max_w: f32, max_h: f32| -> f32 {
        let layer = doc.get_page(page_idx).get_layer(layer_idx);

        // Get image dimensions
        let (img_w, img_h) = (img.width() as f32, img.height() as f32);

        // Calculate physical size at 96 DPI (how the image was likely created)
        // 1 inch = 25.4 mm, 96 DPI means 1 pixel = 25.4/96 mm
        let px_to_mm = 25.4 / 96.0;
        let phys_w = img_w * px_to_mm;  // width in mm
        let phys_h = img_h * px_to_mm;  // height in mm

        // Scale to fit within max dimensions while preserving aspect ratio
        let scale = if phys_w > max_w || phys_h > max_h {
            (max_w / phys_w).min(max_h / phys_h).min(1.0)
        } else {
            1.0  // Don't upscale small images
        };

        let _display_w = phys_w * scale;
        let display_h = phys_h * scale;

        // Create image object
        let image_obj = Image::from_dynamic_image(img);

        // Add image to layer with transformation
        // Set DPI to 96 so printpdf interprets pixels correctly
        image_obj.add_to_layer(
            layer.clone(),
            ImageTransform {
                translate_x: Some(Mm(x)),
                translate_y: Some(Mm(y - display_h)),
                scale_x: Some(scale),
                scale_y: Some(scale),
                dpi: Some(96.0),  // Tell printpdf to interpret pixels at 96 DPI
                ..Default::default()
            },
        );

        display_h // Return the height used
    };

    // Write title
    {
        let layer = doc.get_page(current_page).get_layer(current_layer);
        layer.use_text(title, font_size_title, Mm(margin_left), Mm(y_position), &bold_font);
        y_position -= font_size_title * 0.5 + line_height;

        y_position -= line_height;
    }

    // Regex to match markdown image syntax: ![alt](path)
    let img_regex = Regex::new(r"!\[([^\]]*)\]\(([^)]+)\)").unwrap();

    // Process content line by line
    let lines: Vec<&str> = content.lines().collect();

    for line_text in &lines {
        let trimmed = line_text.trim();

        // Check if we need a new page
        if y_position < margin_bottom + line_height {
            let (p, l) = new_page(&doc, &mut y_position);
            current_page = p;
            current_layer = l;
        }

        // Empty line = paragraph break
        if trimmed.is_empty() {
            y_position -= line_height;
            continue;
        }

        // Check for image markdown syntax
        if let Some(caps) = img_regex.captures(trimmed) {
            let _alt = caps.get(1).map(|m| m.as_str()).unwrap_or("");
            let img_path = caps.get(2).map(|m| m.as_str()).unwrap_or("");

            // Try to load the image
            match load_image(img_path) {
                Ok(img) => {
                    // Check if image fits on current page
                    let estimated_height = (img.height() as f32 * 0.264583).min(max_image_height);
                    if y_position - estimated_height < margin_bottom {
                        let (p, l) = new_page(&doc, &mut y_position);
                        current_page = p;
                        current_layer = l;
                    }

                    // Add image
                    let img_height = add_image(&doc, current_page, current_layer, &img,
                                               margin_left, y_position,
                                               max_image_width, max_image_height);

                    y_position -= img_height + line_height * 2.0; // Space after image
                }
                Err(e) => {
                    // If image fails, show placeholder text
                    let layer = doc.get_page(current_page).get_layer(current_layer);
                    layer.use_text(&format!("[Image: {}]", img_path), font_size_body,
                                   Mm(margin_left), Mm(y_position), &font);
                    y_position -= line_height;
                    eprintln!("Warning: {}", e);
                }
            }
            continue;
        }

        // Detect markdown-style headings
        let (text, is_heading, is_bold) = if trimmed.starts_with("### ") {
            (&trimmed[4..], false, true)
        } else if trimmed.starts_with("## ") {
            (&trimmed[3..], true, false)
        } else if trimmed.starts_with("# ") {
            (&trimmed[2..], true, false)
        } else if trimmed.starts_with("**") && trimmed.ends_with("**") && trimmed.len() > 4 {
            (&trimmed[2..trimmed.len()-2], false, true)
        } else {
            (trimmed, false, false)
        };

        // Strip remaining markdown formatting (*, **, |, -)
        let clean_text = text
            .replace("**", "")
            .replace("*", "")
            .replace("---", "")
            .replace("___", "");

        if clean_text.trim().is_empty() {
            y_position -= line_height * 0.5;
            continue;
        }

        let (current_font, current_size) = if is_heading {
            (&bold_font, font_size_heading)
        } else if is_bold {
            (&bold_font, font_size_body)
        } else {
            (&font, font_size_body)
        };

        // Word wrap
        let words: Vec<&str> = clean_text.split_whitespace().collect();
        let mut current_line = String::new();

        for word in words {
            let test_line = if current_line.is_empty() {
                word.to_string()
            } else {
                format!("{} {}", current_line, word)
            };

            if test_line.len() > chars_per_line && !current_line.is_empty() {
                // Flush current line
                if y_position < margin_bottom + line_height {
                    let (p, l) = new_page(&doc, &mut y_position);
                    current_page = p;
                    current_layer = l;
                }
                let layer = doc.get_page(current_page).get_layer(current_layer);
                layer.use_text(&current_line, current_size, Mm(margin_left), Mm(y_position), current_font);
                y_position -= line_height;
                current_line = word.to_string();
            } else {
                current_line = test_line;
            }
        }

        // Flush remaining text
        if !current_line.is_empty() {
            if y_position < margin_bottom + line_height {
                let (p, l) = new_page(&doc, &mut y_position);
                current_page = p;
                current_layer = l;
            }
            let layer = doc.get_page(current_page).get_layer(current_layer);
            layer.use_text(&current_line, current_size, Mm(margin_left), Mm(y_position), current_font);
            y_position -= line_height;
        }

        // Extra spacing after headings
        if is_heading {
            y_position -= line_height * 0.5;
        }
    }

    // Save to file
    let file = std::fs::File::create(output_path)
        .map_err(|e| format!("Could not create file: {}", e))?;
    let mut writer = BufWriter::new(file);
    doc.save(&mut writer)
        .map_err(|e| format!("Could not save PDF: {}", e))?;

    let file_size = std::fs::metadata(output_path)
        .map(|m| m.len())
        .unwrap_or(0);

    Ok(format!("PDF created: '{}' ({} bytes, {} pages estimated)",
        output_path.rsplit('/').next().unwrap_or(output_path),
        file_size,
        ((lines.len() as f32 / 50.0).ceil() as usize).max(1)
    ))
}

/// Web search - dispatches to Ollama Cloud or DuckDuckGo based on settings
async fn do_web_search(client: &reqwest::Client, query: &str, max_results: usize) -> Vec<SearchResult> {
    let settings = current_settings();

    // Use Ollama Cloud web search if configured
    if settings.search_engine == "ollama" {
        return do_ollama_web_search(client, query, max_results, &settings).await;
    }

    // Default: DuckDuckGo
    do_duckduckgo_search(client, query, max_results).await
}

/// Ollama Cloud web search API (requires API key from ollama.com)
async fn do_ollama_web_search(client: &reqwest::Client, query: &str, max_results: usize, settings: &Settings) -> Vec<SearchResult> {
    // Use API key from active source
    let api_key = match settings.llm_active_source.as_str() {
        "cloud" => &settings.llm_api_key_cloud,
        _ => &settings.llm_api_key_local,
    };

    if api_key.is_empty() {
        eprintln!("[Search] Ollama web search requires API key. Falling back to DuckDuckGo.");
        return do_duckduckgo_search(client, query, max_results).await;
    }

    let body = serde_json::json!({
        "query": query,
        "max_results": max_results.min(10)
    });

    let res = client
        .post("https://ollama.com/api/web_search")
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await;

    match res {
        Ok(response) => {
            if let Ok(text) = response.text().await {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                    // Check for error (some queries fail, e.g. non-English)
                    if json.get("error").is_some() {
                        eprintln!("[Search] Ollama web search error for query '{}', falling back to DuckDuckGo", query);
                        return do_duckduckgo_search(client, query, max_results).await;
                    }
                    // Parse Ollama web search results
                    if let Some(results) = json.get("results").and_then(|r| r.as_array()) {
                        let parsed: Vec<SearchResult> = results.iter().take(max_results).filter_map(|r| {
                            Some(SearchResult {
                                title: r.get("title").and_then(|t| t.as_str()).unwrap_or("").to_string(),
                                url: r.get("url").and_then(|u| u.as_str()).unwrap_or("").to_string(),
                                snippet: r.get("content").and_then(|c| c.as_str())
                                    .map(|s| safe_truncate(s, 500).to_string())
                                    .unwrap_or_default(),
                            })
                        }).collect();
                        if !parsed.is_empty() {
                            return parsed;
                        }
                    }
                }
            }
            eprintln!("[Search] Ollama web search returned no results, falling back to DuckDuckGo");
            do_duckduckgo_search(client, query, max_results).await
        }
        Err(e) => {
            eprintln!("[Search] Ollama web search error: {}, falling back to DuckDuckGo", e);
            do_duckduckgo_search(client, query, max_results).await
        }
    }
}

/// DuckDuckGo HTML search
async fn do_duckduckgo_search(client: &reqwest::Client, query: &str, max_results: usize) -> Vec<SearchResult> {
    let res = client
        .get("https://html.duckduckgo.com/html/")
        .query(&[("q", query)])
        .header("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        .send()
        .await;

    let html = match res {
        Ok(r) => match r.text().await {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        },
        Err(_) => return Vec::new(),
    };

    let mut results = Vec::new();

    let link_re = regex::Regex::new(r#"class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>"#).unwrap();
    let snippet_re = regex::Regex::new(r#"class="result__snippet"[^>]*>(.*?)</a>"#).unwrap();

    let links: Vec<(String, String)> = link_re.captures_iter(&html)
        .map(|cap| {
            let url = cap[1].to_string();
            let title = html_to_text(&cap[2]);
            let actual_url = if url.contains("uddg=") {
                let decoded = url.split("uddg=").nth(1).unwrap_or(&url);
                urlencoding_decode(decoded)
            } else {
                url
            };
            (actual_url, title)
        })
        .collect();

    let snippets: Vec<String> = snippet_re.captures_iter(&html)
        .map(|cap| html_to_text(&cap[1]))
        .collect();

    for (i, (url, title)) in links.iter().enumerate() {
        if i >= max_results { break; }
        results.push(SearchResult {
            title: title.clone(),
            url: url.clone(),
            snippet: snippets.get(i).cloned().unwrap_or_default(),
        });
    }

    results
}

/// Simple URL decode
fn urlencoding_decode(s: &str) -> String {
    let mut result = String::new();
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '%' {
            let hex: String = chars.by_ref().take(2).collect();
            if let Ok(byte) = u8::from_str_radix(&hex, 16) {
                result.push(byte as char);
            }
        } else if c == '+' {
            result.push(' ');
        } else if c == '&' {
            break;
        } else {
            result.push(c);
        }
    }
    result
}

/// Image search result
#[derive(Clone, Serialize, Deserialize)]
struct ImageResult {
    url: String,
    source: String,
    width: Option<u32>,
    height: Option<u32>,
}

/// Search for images using multiple sources
async fn do_image_search(client: &reqwest::Client, query: &str, max_results: usize) -> Vec<ImageResult> {
    eprintln!("[ImageSearch] Starting search for: {}", query);
    let mut all_results = Vec::new();

    // 1. Try Wikipedia Commons first (most permissive, good for maps/diagrams)
    let wiki_results = search_wikipedia_commons(client, query, max_results).await;
    eprintln!("[ImageSearch] Wikimedia: {} results", wiki_results.len());
    all_results.extend(wiki_results);

    if all_results.len() >= max_results {
        return all_results.into_iter().take(max_results).collect();
    }

    // 2. Try Unsplash API (free, high quality photos)
    let unsplash_results = search_unsplash(client, query, max_results - all_results.len()).await;
    eprintln!("[ImageSearch] Unsplash: {} results", unsplash_results.len());
    all_results.extend(unsplash_results);

    if all_results.len() >= max_results {
        return all_results.into_iter().take(max_results).collect();
    }

    // 3. Try DuckDuckGo image search
    let ddg_results = search_ddg_images(client, query, max_results - all_results.len()).await;
    eprintln!("[ImageSearch] DuckDuckGo: {} results", ddg_results.len());
    all_results.extend(ddg_results);

    // 4. Try Pixabay API (free)
    if all_results.len() < max_results {
        let pixabay_results = search_pixabay(client, query, max_results - all_results.len()).await;
        eprintln!("[ImageSearch] Pixabay: {} results", pixabay_results.len());
        all_results.extend(pixabay_results);
    }

    eprintln!("[ImageSearch] Total: {} results", all_results.len());
    all_results.into_iter().take(max_results).collect()
}

/// Search Unsplash for images
async fn search_unsplash(client: &reqwest::Client, query: &str, max_results: usize) -> Vec<ImageResult> {
    let url = format!("https://unsplash.com/napi/search/photos?query={}&per_page={}&order_by=relevance",
        urlencoding::encode(query),
        max_results.min(10)
    );

    let res = client
        .get(&url)
        .header("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")
        .header("Accept", "application/json")
        .send()
        .await;

    match res {
        Ok(response) => {
            if let Ok(text) = response.text().await {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                    if let Some(results) = json.get("results").and_then(|r| r.as_array()) {
                        return results.iter().take(max_results).filter_map(|item| {
                            let urls = item.get("urls")?;
                            let raw_url = urls.get("regular").or_else(|| urls.get("small"))?;
                            let url = raw_url.as_str()?.to_string();
                            let width = item.get("width").and_then(|w| w.as_u64()).map(|w| w as u32);
                            let height = item.get("height").and_then(|h| h.as_u64()).map(|h| h as u32);
                            Some(ImageResult {
                                url,
                                source: "unsplash".to_string(),
                                width,
                                height,
                            })
                        }).collect();
                    }
                }
            }
            Vec::new()
        }
        Err(_) => Vec::new()
    }
}

/// Search DuckDuckGo for images
async fn search_ddg_images(client: &reqwest::Client, query: &str, max_results: usize) -> Vec<ImageResult> {
    let url = format!("https://duckduckgo.com/?q={}&iar=images&iax=images&ia=images",
        urlencoding::encode(query)
    );

    let res = client
        .get(&url)
        .header("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        .header("Accept", "text/html")
        .send()
        .await;

    match res {
        Ok(response) => {
            if let Ok(html) = response.text().await {
                // Extract image URLs from vqd token and then fetch actual images
                let vqd_re = regex::Regex::new(r#"vqd\s*=\s*'([^']+)'"#).unwrap();
                if let Some(caps) = vqd_re.captures(&html) {
                    let vqd = &caps[1];

                    // Now fetch the actual image results
                    let images_url = format!("https://duckduckgo.com/i.js?l=wt-wt&o=json&q={}&vqd={}&f=,,,,,,",
                        urlencoding::encode(query),
                        vqd
                    );

                    let img_res = client
                        .get(&images_url)
                        .header("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")
                        .header("Accept", "application/json")
                        .header("Referer", "https://duckduckgo.com/")
                        .send()
                        .await;

                    if let Ok(img_response) = img_res {
                        if let Ok(img_text) = img_response.text().await {
                            if let Ok(img_json) = serde_json::from_str::<serde_json::Value>(&img_text) {
                                if let Some(results) = img_json.get("results").and_then(|r| r.as_array()) {
                                    return results.iter().take(max_results).filter_map(|item| {
                                        let url = item.get("image").and_then(|i| i.as_str())?.to_string();
                                        let width = item.get("width").and_then(|w| w.as_u64()).map(|w| w as u32);
                                        let height = item.get("height").and_then(|h| h.as_u64()).map(|h| h as u32);
                                        Some(ImageResult {
                                            url,
                                            source: "duckduckgo".to_string(),
                                            width,
                                            height,
                                        })
                                    }).collect();
                                }
                            }
                        }
                    }
                }
            }
            Vec::new()
        }
        Err(_) => Vec::new()
    }
}

/// Search Pixabay for images (requires API key, fallback to free)
async fn search_pixabay(client: &reqwest::Client, query: &str, max_results: usize) -> Vec<ImageResult> {
    // Pixabay has a free API that returns sample results without key
    let url = format!("https://pixabay.com/api/?q={}&per_page={}&image_type=photo&safesearch=true",
        urlencoding::encode(query),
        max_results.min(10)
    );

    let res = client
        .get(&url)
        .header("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")
        .send()
        .await;

    match res {
        Ok(response) => {
            if let Ok(text) = response.text().await {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                    if let Some(hits) = json.get("hits").and_then(|h| h.as_array()) {
                        return hits.iter().take(max_results).filter_map(|item| {
                            let url = item.get("largeImageURL").or_else(|| item.get("webformatURL"))?;
                            let url_str = url.as_str()?.to_string();
                            let width = item.get("imageWidth").and_then(|w| w.as_u64()).map(|w| w as u32);
                            let height = item.get("imageHeight").and_then(|h| h.as_u64()).map(|h| h as u32);
                            Some(ImageResult {
                                url: url_str,
                                source: "pixabay".to_string(),
                                width,
                                height,
                            })
                        }).collect();
                    }
                }
            }
            Vec::new()
        }
        Err(_) => Vec::new()
    }
}

/// Search Wikipedia Commons for images (more permissive, good for maps/diagrams)
async fn search_wikipedia_commons(client: &reqwest::Client, query: &str, max_results: usize) -> Vec<ImageResult> {
    eprintln!("[ImageSearch] Searching Wikimedia Commons for: {}", query);

    // Use Wikipedia API to search for images on Wikimedia Commons
    // Using generator approach with imageinfo to get URLs directly
    let url = format!(
        "https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrnamespace=6&gsrlimit={}&gsrsearch={}&prop=imageinfo&iiprop=url|mime&format=json",
        max_results.min(20),
        urlencoding::encode(query)
    );

    let res = client
        .get(&url)
        .header("User-Agent", "PuterraAgent/1.1 (https://github.com/puterra)")
        .header("Accept", "application/json")
        .send()
        .await;

    match res {
        Ok(response) => {
            if let Ok(text) = response.text().await {
                eprintln!("[ImageSearch] Wikimedia response length: {} bytes", text.len());

                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                    if let Some(pages) = json.get("query").and_then(|q| q.get("pages")) {
                        if let Some(pages_obj) = pages.as_object() {
                            let mut image_results = Vec::new();

                            for (_, page) in pages_obj.iter() {
                                if image_results.len() >= max_results {
                                    break;
                                }

                                // Get the image URL from imageinfo
                                if let Some(imageinfo) = page.get("imageinfo").and_then(|ii| ii.as_array()) {
                                    if let Some(info) = imageinfo.first() {
                                        if let Some(url) = info.get("url").and_then(|u| u.as_str()) {
                                            let mime = info.get("mime").and_then(|m| m.as_str()).unwrap_or("");
                                            let is_svg = mime.contains("svg") || url.to_lowercase().ends_with(".svg");

                                            // Prefer PNG/JPEG over SVG for better compatibility
                                            if !is_svg || image_results.len() < max_results {
                                                eprintln!("[ImageSearch] Found image: {} (mime: {})", url, mime);
                                                image_results.push(ImageResult {
                                                    url: url.to_string(),
                                                    source: "wikimedia".to_string(),
                                                    width: None,
                                                    height: None,
                                                });
                                            }
                                        }
                                    }
                                }
                            }

                            eprintln!("[ImageSearch] Found {} images from Wikimedia Commons", image_results.len());
                            return image_results;
                        }
                    }
                }
            }
            Vec::new()
        }
        Err(e) => {
            eprintln!("[ImageSearch] Wikimedia Commons error: {}", e);
            Vec::new()
        }
    }
}

/// Web fetch - dispatches to Ollama Cloud or direct fetch based on settings
async fn do_web_fetch(client: &reqwest::Client, url: &str) -> Result<String, String> {
    let settings = current_settings();

    // Use Ollama Cloud web fetch if configured
    let api_key = match settings.llm_active_source.as_str() {
        "cloud" => &settings.llm_api_key_cloud,
        _ => &settings.llm_api_key_local,
    };

    if settings.search_engine == "ollama" && !api_key.is_empty() {
        match do_ollama_web_fetch(client, url, &settings).await {
            Ok(content) => return Ok(content),
            Err(e) => eprintln!("[Fetch] Ollama fetch failed ({}), trying direct fetch", e),
        }
    }

    do_direct_web_fetch(client, url).await
}

/// Ollama Cloud web fetch API
async fn do_ollama_web_fetch(client: &reqwest::Client, url: &str, settings: &Settings) -> Result<String, String> {
    // Use API key from active source
    let api_key = match settings.llm_active_source.as_str() {
        "cloud" => &settings.llm_api_key_cloud,
        _ => &settings.llm_api_key_local,
    };

    let body = serde_json::json!({"url": url});

    let res = client
        .post("https://ollama.com/api/web_fetch")
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&body)
        .timeout(std::time::Duration::from_secs(current_settings().timeout_web_fetch))
        .send()
        .await
        .map_err(|e| format!("Ollama fetch error: {}", e))?;

    let text = res.text().await.map_err(|e| format!("Read error: {}", e))?;
    let json: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("Parse error: {}", e))?;

    if let Some(content) = json.get("content").and_then(|c| c.as_str()) {
        Ok(content.to_string())
    } else {
        Err("No content in Ollama fetch response".to_string())
    }
}

/// Direct web fetch - GET a URL and return text content
async fn do_direct_web_fetch(client: &reqwest::Client, url: &str) -> Result<String, String> {
    let res = client
        .get(url)
        .header("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        .header("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
        .header("Accept-Language", "en-US,en;q=0.5,tr;q=0.3")
        .timeout(std::time::Duration::from_secs(current_settings().timeout_web_fetch))
        .send()
        .await
        .map_err(|e| format!("Fetch failed: {}", e))?;

    let content_type = res.headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    let text = res.text().await.map_err(|e| format!("Read failed: {}", e))?;

    if content_type.contains("html") || text.trim_start().starts_with("<!") || text.trim_start().starts_with("<html") {
        Ok(html_to_text(&text))
    } else {
        Ok(text)
    }
}

/// Call LLM with messages array and optional tools (native Ollama/OpenAI tool calling)
/// Pass `share_key_override` to use a share key's LLM config instead of global settings
async fn call_llm_chat(
    client: &reqwest::Client,
    messages: &[serde_json::Value],
    tools: Option<&[serde_json::Value]>,
    model: Option<&str>,
    share_key_override: Option<&ShareKey>,
) -> Result<serde_json::Value, String> {
    let settings = current_settings();

    // Select config: share key overrides global settings
    let (api_url, llm_model, api_key): (String, String, String) = if let Some(sk) = share_key_override {
        (
            sk.api_url.clone(),
            model.unwrap_or(&sk.model).to_string(),
            sk.api_key.clone(),
        )
    } else {
        match settings.llm_active_source.as_str() {
            "cloud" => (
                settings.llm_api_url_cloud.clone(),
                model.unwrap_or(&settings.llm_model_cloud).to_string(),
                settings.llm_api_key_cloud.clone(),
            ),
            _ => (
                settings.llm_api_url_local.clone(),
                model.unwrap_or(&settings.llm_model_local).to_string(),
                settings.llm_api_key_local.clone(),
            ),
        }
    };

    // Build chat endpoint URL
    let base = api_url.trim_end_matches('/');
    let chat_url = if base.ends_with("/chat") || base.ends_with("/chat/completions") {
        base.to_string()
    } else if settings.llm_provider == "openai" && !base.contains("ollama.com") {
        format!("{}/chat/completions", base)
    } else {
        format!("{}/chat", base)
    };

    let mut body = serde_json::json!({
        "model": llm_model,
        "messages": messages,
        "stream": false,
    });

    // OpenAI-compat: URL ends with /chat/completions (Gemini, OpenAI, Groq, Together, etc.)
    // Ollama native: URL ends with /chat or /api/chat
    let is_openai_compat = chat_url.ends_with("/chat/completions");
    let is_ollama_native = !is_openai_compat;
    let is_cloud = share_key_override.map(|sk| sk.api_url.contains("ollama.com")).unwrap_or(settings.llm_active_source == "cloud");

    // Think mode: Ollama-only field, never send to OpenAI-compat APIs
    if is_ollama_native {
        if settings.llm_think {
            body["think"] = serde_json::json!(true);
        } else if !is_cloud {
            body["think"] = serde_json::json!(false);
        }
    }

    // Generation params: Ollama uses num_predict, OpenAI-compat uses max_tokens
    if (settings.llm_temperature - 0.7).abs() > 0.001 {
        body["temperature"] = serde_json::json!(settings.llm_temperature);
    }
    if settings.llm_max_tokens != 4096 {
        if is_ollama_native {
            body["num_predict"] = serde_json::json!(settings.llm_max_tokens);
        } else {
            body["max_tokens"] = serde_json::json!(settings.llm_max_tokens);
        }
    }

    if let Some(tools) = tools {
        if !tools.is_empty() {
            // Ollama Cloud requires property types as arrays ["string"] instead of "string"
            // OpenAI-compat APIs (Gemini, OpenAI, Groq) require standard string format
            let tools_to_send = if is_ollama_native {
                normalize_tool_property_types(tools.to_vec())
            } else {
                tools.to_vec()
            };
            body["tools"] = serde_json::json!(tools_to_send);
        }
    }

    eprintln!("[LLM] Request body: {}", serde_json::to_string(&body).unwrap_or_default().chars().take(500).collect::<String>());

    // Retry loop: handles 429 rate-limit with API-provided retry delay (up to 3 attempts)
    for attempt in 0..3u32 {
        let result = client
            .post(&chat_url)
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&body)
            .send()
            .await;

        let text = match result {
            Ok(resp) => match resp.text().await {
                Ok(t) => t,
                Err(e) => return Err(format!("Error reading LLM response: {}", e)),
            },
            Err(e) => return Err(format!("Error calling LLM: {}", e)),
        };

        eprintln!("[LLM] Response ({} bytes): {}", text.len(), &text);

        // Try to parse as JSON
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
            // --- 429 Rate Limit ---
            let status_code = json.get("error")
                .and_then(|e| e.get("code"))
                .and_then(|c| c.as_u64())
                .unwrap_or(0);
            if status_code == 429 {
                // Extract retryDelay from details[].retryDelay (e.g. "51s")
                let retry_secs = json.get("error")
                    .and_then(|e| e.get("details"))
                    .and_then(|d| d.as_array())
                    .and_then(|arr| arr.iter().find_map(|item| {
                        item.get("retryDelay").and_then(|rd| rd.as_str()).map(|s| {
                            s.trim_end_matches('s').parse::<u64>().unwrap_or(60)
                        })
                    }))
                    .unwrap_or(60);
                // Cap at 120s to avoid hanging the agent forever
                let wait = retry_secs.min(120) + 2;
                eprintln!("[LLM] ⚠️  Rate limited (429). Waiting {}s before retry {}/3...", wait, attempt + 1);
                tokio::time::sleep(std::time::Duration::from_secs(wait)).await;
                continue; // retry
            }

            // --- Error response ---
            if let Some(err_obj) = json.get("error") {
                let msg = err_obj.get("message").and_then(|m| m.as_str())
                    .unwrap_or_else(|| err_obj.as_str().unwrap_or("unknown error"));
                return Err(format!("LLM error: {}", msg));
            }

            // --- Ollama native format ---
            if json.get("message").is_some() {
                return Ok(json);
            }

            // --- OpenAI-compat format → normalize to Ollama shape ---
            if let Some(msg) = json.get("choices")
                .and_then(|c| c.get(0))
                .and_then(|c| c.get("message")) {
                return Ok(serde_json::json!({
                    "message": msg,
                    "done": true
                }));
            }

            return Err(format!("Unexpected LLM response format: {}", safe_truncate(&text, 300)));
        }

        // Streaming fallback (newline-delimited JSON from Ollama local)
        let mut content = String::new();
        let mut found_any = false;
        for line in text.lines() {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(line) {
                found_any = true;
                if let Some(c) = json.get("message")
                    .and_then(|m| m.get("content"))
                    .and_then(|c| c.as_str()) {
                    content.push_str(c);
                }
            }
        }
        if found_any && !content.is_empty() {
            return Ok(serde_json::json!({"message": {"role": "assistant", "content": content}}));
        }
        return Err(format!("Could not parse LLM response: {}", safe_truncate(&text, 200)));
    }

    Err("LLM rate limited after 3 retries".to_string())
}

/// Simple LLM call (for chat endpoint) - backwards compatible
async fn call_llm(client: &reqwest::Client, prompt: &str, model: Option<&str>) -> String {
    let messages = vec![serde_json::json!({"role": "user", "content": prompt})];
    match call_llm_chat(client, &messages, None, model, None).await {
        Ok(json) => {
            json.get("message")
                .and_then(|m| m.get("content"))
                .and_then(|c| c.as_str())
                .unwrap_or("")
                .to_string()
        }
        Err(e) => e,
    }
}

// ============================================================
// AUTH ENDPOINTS
// ============================================================

#[get("/")]
async fn index() -> impl Responder {
    match std::fs::read_to_string("public/index.html") {
        Ok(html) => HttpResponse::Ok()
            .content_type("text/html; charset=utf-8")
            .insert_header(("Cache-Control", "no-cache, no-store, must-revalidate"))
            .body(html),
        Err(_) => HttpResponse::NotFound().body("index.html not found"),
    }
}

#[post("/api/signup")]
async fn signup(data: web::Data<AppState>, body: web::Json<SignupRequest>) -> impl Responder {
    let mut users = data.users.lock().unwrap();
    if users.contains_key(&body.username) {
        return HttpResponse::BadRequest().json(serde_json::json!({
            "success": false, "error": "Username exists"
        }));
    }
    users.insert(body.username.clone(), User {
        id: Uuid::new_v4().to_string(),
        username: body.username.clone(),
        password_hash: hash_password(&body.password),
    });
    save_users(&users);
    std::fs::create_dir_all(get_user_dir(&body.username)).ok();
    HttpResponse::Ok().json(serde_json::json!({ "success": true }))
}

#[post("/api/login")]
async fn login(data: web::Data<AppState>, body: web::Json<LoginRequest>) -> impl Responder {
    let users = data.users.lock().unwrap();
    match users.get(&body.username) {
        Some(user) if user.password_hash == hash_password(&body.password) => {
            let username = user.username.clone();
            drop(users);  // Release lock before acquiring sessions lock

            // Generate token
            let token = Uuid::new_v4().to_string();
            let session = Session {
                token: token.clone(),
                username: username.clone(),
            };

            // Store session
            let mut sessions = data.sessions.lock().unwrap();
            sessions.insert(token.clone(), session);

            HttpResponse::Ok().json(serde_json::json!({
                "success": true,
                "token": token,
                "user": { "username": username }
            }))
        }
        _ => HttpResponse::Unauthorized().json(serde_json::json!({
            "success": false, "error": "Invalid credentials"
        }))
    }
}

#[post("/api/logout")]
async fn logout(data: web::Data<AppState>, req: actix_web::HttpRequest) -> impl Responder {
    // Extract token from Authorization header
    if let Some(auth_header) = req.headers().get("Authorization") {
        if let Ok(auth_str) = auth_header.to_str() {
            if let Some(token) = auth_str.strip_prefix("Bearer ") {
                let mut sessions = data.sessions.lock().unwrap();
                sessions.remove(token);
            }
        }
    }
    HttpResponse::Ok().json(serde_json::json!({ "success": true }))
}

#[post("/api/validate-token")]
async fn validate_token(data: web::Data<AppState>, req: actix_web::HttpRequest) -> impl Responder {
    // Extract token from Authorization header
    if let Some(auth_header) = req.headers().get("Authorization") {
        if let Ok(auth_str) = auth_header.to_str() {
            if let Some(token) = auth_str.strip_prefix("Bearer ") {
                let sessions = data.sessions.lock().unwrap();
                if sessions.contains_key(token) {
                    return HttpResponse::Ok().json(serde_json::json!({ "valid": true }));
                }
            }
        }
    }
    HttpResponse::Unauthorized().json(serde_json::json!({ "valid": false }))
}

// ============================================================
// FILE ENDPOINTS
// ============================================================

#[get("/api/files/{username}")]
async fn list_files(path: web::Path<String>, query: web::Query<std::collections::HashMap<String, String>>) -> impl Responder {
    let username = path.into_inner();
    let user_dir = get_user_dir(&username);
    std::fs::create_dir_all(&user_dir).ok();

    // Optional subfolder path from query param ?path=folder/subfolder
    let sub_path = query.get("path").map(|p| p.trim_matches('/')).unwrap_or("");
    let dir = if sub_path.is_empty() || sub_path.contains("..") {
        user_dir.clone()
    } else {
        format!("{}/{}", user_dir, sub_path)
    };

    let files: Vec<FileEntry> = std::fs::read_dir(&dir)
        .map(|entries| {
            entries.filter_map(|e| e.ok())
                .filter(|e| !e.file_name().to_string_lossy().starts_with("._"))
                .map(|e| FileEntry {
                    name: e.file_name().to_string_lossy().to_string(),
                    is_dir: e.path().is_dir(),
                    size: e.metadata().map(|m| m.len()).unwrap_or(0),
                }).collect()
        }).unwrap_or_default();

    HttpResponse::Ok().json(serde_json::json!({ "success": true, "files": files }))
}

#[post("/api/files/create")]
async fn create_item(body: web::Json<CreateRequest>) -> impl Responder {
    let user_dir = get_user_dir(&body.username);
    std::fs::create_dir_all(&user_dir).ok();
    if body.name.contains("..") {
        return HttpResponse::BadRequest().json(serde_json::json!({ "success": false, "error": "Invalid name" }));
    }
    let path = format!("{}/{}", user_dir, body.name);
    if let Some(parent) = std::path::Path::new(&path).parent() { std::fs::create_dir_all(parent).ok(); }
    let result = if body.r#type == "folder" { std::fs::create_dir(&path) } else { std::fs::write(&path, "").map(|_| ()) };
    match result {
        Ok(_) => HttpResponse::Ok().json(serde_json::json!({ "success": true })),
        Err(e) => HttpResponse::InternalServerError().json(serde_json::json!({ "success": false, "error": e.to_string() }))
    }
}

#[post("/api/files/delete")]
async fn delete_items(body: web::Json<DeleteRequest>) -> impl Responder {
    let user_dir = get_user_dir(&body.username);
    let mut deleted = 0;
    for name in &body.names {
        if name.contains("..") { continue; }
        let path = format!("{}/{}", user_dir, name);
        if std::fs::remove_file(&path).is_ok() || std::fs::remove_dir_all(&path).is_ok() { deleted += 1; }
    }
    HttpResponse::Ok().json(serde_json::json!({ "success": true, "deleted": deleted }))
}

#[post("/api/files/rename")]
async fn rename_item(body: web::Json<RenameRequest>) -> impl Responder {
    let user_dir = get_user_dir(&body.username);
    if body.old_name.contains("..") || body.new_name.contains("..") {
        return HttpResponse::BadRequest().json(serde_json::json!({ "success": false, "error": "Invalid name" }));
    }
    match std::fs::rename(format!("{}/{}", user_dir, body.old_name), format!("{}/{}", user_dir, body.new_name)) {
        Ok(_) => HttpResponse::Ok().json(serde_json::json!({ "success": true })),
        Err(e) => HttpResponse::InternalServerError().json(serde_json::json!({ "success": false, "error": e.to_string() }))
    }
}

#[post("/api/files/read")]
async fn read_file(body: web::Json<ReadRequest>) -> impl Responder {
    if body.name.contains("..") {
        return HttpResponse::BadRequest().json(serde_json::json!({ "success": false, "error": "Invalid name" }));
    }
    let path = format!("{}/{}", get_user_dir(&body.username), body.name);
    match std::fs::read_to_string(&path) {
        Ok(content) => HttpResponse::Ok().json(serde_json::json!({ "success": true, "content": content })),
        Err(e) => HttpResponse::InternalServerError().json(serde_json::json!({ "success": false, "error": e.to_string() }))
    }
}

#[post("/api/files/write")]
async fn write_file(body: web::Json<WriteRequest>) -> impl Responder {
    if body.name.contains("..") {
        return HttpResponse::BadRequest().json(serde_json::json!({ "success": false, "error": "Invalid name" }));
    }
    let user_dir = get_user_dir(&body.username);
    let full_path = format!("{}/{}", user_dir, body.name);
    if let Some(parent) = std::path::Path::new(&full_path).parent() { std::fs::create_dir_all(parent).ok(); }
    match std::fs::write(&full_path, &body.content) {
        Ok(_) => HttpResponse::Ok().json(serde_json::json!({ "success": true })),
        Err(e) => HttpResponse::InternalServerError().json(serde_json::json!({ "success": false, "error": e.to_string() }))
    }
}

/// Download/serve a file (for binary files like PDF, images)
#[get("/api/files/download/{username}/{filename:.*}")]
async fn download_file(path: web::Path<(String, String)>) -> impl Responder {
    let (username, filename) = path.into_inner();
    if filename.contains("..") {
        return HttpResponse::BadRequest().body("Invalid filename");
    }
    let file_path = format!("{}/{}", get_user_dir(&username), filename);
    match std::fs::read(&file_path) {
        Ok(bytes) => {
            let ext = filename.rsplit('.').next().unwrap_or("").to_lowercase();
            let content_type = match ext.as_str() {
                "pdf" => "application/pdf",
                "png" => "image/png",
                "jpg" | "jpeg" => "image/jpeg",
                "gif" => "image/gif",
                "bmp" => "image/bmp",
                "webp" => "image/webp",
                "svg" => "image/svg+xml",
                "mp3" => "audio/mpeg",
                "wav" => "audio/wav",
                "ogg" => "audio/ogg",
                "flac" => "audio/flac",
                "m4a" => "audio/mp4",
                "mp4" | "m4v" => "video/mp4",
                "webm" => "video/webm",
                "mov" => "video/quicktime",
                "mkv" => "video/x-matroska",
                "avi" => "video/x-msvideo",
                "zip" => "application/zip",
                "json" => "application/json",
                "txt" => "text/plain; charset=utf-8",
                "html" | "htm" => "text/html; charset=utf-8",
                "css" => "text/css",
                "js" => "application/javascript",
                _ => "application/octet-stream",
            };
            let is_video = ["mp4","webm","mov","mkv","avi","m4v","ogg"].contains(&ext.as_str());
            HttpResponse::Ok()
                .content_type(content_type)
                .insert_header(("Content-Disposition", format!("inline; filename=\"{}\"", filename)))
                .insert_header(("Accept-Ranges", "bytes"))
                .insert_header(("Cache-Control", if is_video { "no-cache" } else { "public, max-age=3600" }))
                .body(bytes)
        }
        Err(_) => HttpResponse::NotFound().body("File not found"),
    }
}

/// Serve user files for the HTML viewer (auth via token query param for iframe compatibility)
/// Relative URLs in HTML files resolve naturally: ./style.css → /api/files/view/user/dir/style.css
#[get("/api/files/view/{username}/{path:.*}")]
async fn view_file(
    req_path: web::Path<(String, String)>,
    query: web::Query<std::collections::HashMap<String, String>>,
    data: web::Data<AppState>,
) -> impl Responder {
    let (username, file_path) = req_path.into_inner();
    if file_path.contains("..") {
        return HttpResponse::Forbidden().body("Forbidden");
    }

    // Auth: token from query param (iframes can't send Authorization headers)
    let token = query.get("token").cloned().unwrap_or_default();
    let valid = {
        let sessions = data.sessions.lock().unwrap();
        sessions.get(&token).map(|s| s.username == username).unwrap_or(false)
    };
    if !valid {
        return HttpResponse::Unauthorized()
            .content_type("text/html")
            .body("<html><body style='font-family:sans-serif;color:#ef4444;padding:2rem'><h2>401 – Unauthorized</h2><p>Invalid or expired session token.</p></body></html>");
    }

    let full_path = format!("{}/{}", get_user_dir(&username), file_path);
    match std::fs::read(&full_path) {
        Ok(bytes) => {
            let ext = std::path::Path::new(&full_path)
                .extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();

            let mime = match ext.as_str() {
                "html" | "htm" | "xhtml" => "text/html; charset=utf-8",
                "js"  | "mjs"  | "cjs"  => "application/javascript; charset=utf-8",
                "css"                    => "text/css; charset=utf-8",
                "wasm"                   => "application/wasm",
                "json"                   => "application/json; charset=utf-8",
                "svg"                    => "image/svg+xml",
                "png"                    => "image/png",
                "jpg" | "jpeg"           => "image/jpeg",
                "gif"                    => "image/gif",
                "webp"                   => "image/webp",
                "ico"                    => "image/x-icon",
                "txt"                    => "text/plain; charset=utf-8",
                "xml"                    => "application/xml",
                _                        => "application/octet-stream",
            };

            // For HTML files: inject a <base> tag so relative assets resolve via this endpoint
            if matches!(ext.as_str(), "html" | "htm" | "xhtml") {
                if let Ok(mut html) = String::from_utf8(bytes.clone()) {
                    // Build base URL: everything up to (but not including) the filename
                    let dir_part = if let Some(pos) = file_path.rfind('/') {
                        &file_path[..=pos]
                    } else {
                        ""
                    };
                    let token_val = token.as_str();
                    // Keep the token propagating through relative fetches by rewriting script/link srcs is hard,
                    // so we just set <base href> pointing to the view endpoint directory.
                    // Note: assets must use ?token= for auth – we inject a tiny bootstrap script for that.
                    let base_url = format!("/api/files/view/{}/{}", username, dir_part);
                    let inject = format!(
                        r#"<base href="{base_url}"><script>
window.__PUTERRA_TOKEN__="{token_val}";
// Rewrite fetch/XHR so relative requests carry the token automatically
const _origFetch=window.fetch;
window.fetch=function(url,...args){{
  const u=typeof url==='string'&&!url.startsWith('http')&&!url.startsWith('//')?url+(url.includes('?')?'&':'?')+'token={token_val}':url;
  return _origFetch(u,...args);
}};
</script>"#,
                        base_url = base_url,
                        token_val = token_val,
                    );
                    // Insert after <head> or before </head> or prepend
                    let patched = if let Some(pos) = html.find("<head>").or_else(|| html.find("<HEAD>")) {
                        html.insert_str(pos + 6, &inject);
                        html
                    } else {
                        format!("{}{}", inject, html)
                    };
                    return HttpResponse::Ok().content_type(mime).body(patched);
                }
            }

            let is_media = matches!(ext.as_str(), "mp4"|"webm"|"mov"|"mkv"|"avi"|"m4v"|"ogg"|"mp3"|"wav"|"flac"|"m4a");
            let mut resp = HttpResponse::Ok();
            resp.content_type(mime);
            if is_media {
                resp.insert_header(("Accept-Ranges", "bytes"));
                resp.insert_header(("Cache-Control", "no-cache"));
            }
            resp.body(bytes)
        }
        Err(_) => HttpResponse::NotFound()
            .content_type("text/html")
            .body(format!(
                "<html><body style='font-family:sans-serif;color:#ef4444;padding:2rem'><h2>404 – Not Found</h2><p>{}</p></body></html>",
                file_path
            )),
    }
}

/// Upload file(s) via multipart form data
#[post("/api/files/upload/{username}")]
async fn upload_file(path: web::Path<String>, mut payload: Multipart) -> impl Responder {
    let username = path.into_inner();
    let user_dir = get_user_dir(&username);
    std::fs::create_dir_all(&user_dir).ok();

    let mut uploaded: Vec<String> = Vec::new();
    let max_size: usize = 50 * 1024 * 1024; // 50 MB per file

    while let Some(item) = payload.next().await {
        let mut field = match item {
            Ok(f) => f,
            Err(e) => {
                return HttpResponse::BadRequest().json(serde_json::json!({
                    "success": false, "error": format!("Multipart error: {}", e)
                }));
            }
        };

        // Get filename from content disposition
        let filename = field.content_disposition()
            .get_filename()
            .map(|f| f.to_string())
            .unwrap_or_else(|| format!("upload_{}", Uuid::new_v4()));

        // Sanitize filename
        let safe_name: String = filename.chars()
            .map(|c| if c == '/' || c == '\\' || c == '\0' { '_' } else { c })
            .collect();
        if safe_name.contains("..") || safe_name.is_empty() {
            continue;
        }

        // Read file bytes
        let mut bytes = Vec::new();
        while let Some(chunk) = field.next().await {
            match chunk {
                Ok(data) => {
                    if bytes.len() + data.len() > max_size {
                        return HttpResponse::BadRequest().json(serde_json::json!({
                            "success": false,
                            "error": format!("File '{}' exceeds 50MB limit", safe_name)
                        }));
                    }
                    bytes.extend_from_slice(&data);
                }
                Err(e) => {
                    return HttpResponse::BadRequest().json(serde_json::json!({
                        "success": false, "error": format!("Read error: {}", e)
                    }));
                }
            }
        }

        // Write file
        let file_path = format!("{}/{}", user_dir, safe_name);
        match std::fs::write(&file_path, &bytes) {
            Ok(_) => uploaded.push(safe_name),
            Err(e) => {
                return HttpResponse::InternalServerError().json(serde_json::json!({
                    "success": false, "error": format!("Write error: {}", e)
                }));
            }
        }
    }

    HttpResponse::Ok().json(serde_json::json!({
        "success": true,
        "uploaded": uploaded,
        "count": uploaded.len()
    }))
}

// ============================================================
// INTEGRATED TOOL ENDPOINTS (no external API proxy)
// ============================================================

/// Web Search - uses DuckDuckGo directly
#[post("/api/web_search")]
async fn web_search(body: web::Json<WebSearchRequest>) -> impl Responder {
    let max_results = body.max_results.unwrap_or(8);
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(current_settings().timeout_web_search))
        .build()
        .unwrap_or_default();

    let results = do_web_search(&client, &body.query, max_results).await;

    HttpResponse::Ok().json(serde_json::json!({
        "success": true,
        "results": results
    }))
}

/// Web Fetch - fetches URL directly and returns text content
#[post("/api/web_fetch")]
async fn web_fetch(body: web::Json<FetchRequest>) -> impl Responder {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(current_settings().timeout_web_fetch))
        .build()
        .unwrap_or_default();

    match do_web_fetch(&client, &body.url).await {
        Ok(content) => HttpResponse::Ok().json(serde_json::json!({
            "success": true,
            "content": safe_truncate(&content, 50000)
        })),
        Err(e) => HttpResponse::InternalServerError().json(serde_json::json!({
            "success": false,
            "error": e
        }))
    }
}

/// Shell Execute
#[post("/api/shell")]
async fn shell_exec(body: web::Json<ShellRequest>) -> impl Responder {
    let settings = current_settings();
    if !settings.shell_enabled {
        return HttpResponse::Forbidden().json(serde_json::json!({ "success": false, "error": "Shell is disabled in settings" }));
    }
    // Default to user's storage dir so relative paths in scripts save to user storage
    let user_dir = if body.username.is_empty() { "/tmp".to_string() } else { get_user_dir(&body.username) };
    let cwd = body.cwd.as_deref().unwrap_or(&user_dir);
    let dangerous = ["rm -rf", "mkfs", "dd if=", "> /dev/", "chmod 777", "chown root"];
    for d in dangerous {
        if body.command.contains(d) {
            return HttpResponse::BadRequest().json(serde_json::json!({ "success": false, "error": "Command not allowed" }));
        }
    }
    match Command::new("sh").arg("-c").arg(&body.command).current_dir(cwd).output() {
        Ok(output) => {
            HttpResponse::Ok().json(serde_json::json!({
                "success": output.status.success(),
                "stdout": String::from_utf8_lossy(&output.stdout),
                "stderr": String::from_utf8_lossy(&output.stderr),
                "code": output.status.code()
            }))
        }
        Err(e) => HttpResponse::InternalServerError().json(serde_json::json!({ "success": false, "error": format!("Execution failed: {}", e) }))
    }
}

/// Memory operations
#[post("/api/memory")]
async fn memory(body: web::Json<MemoryRequest>) -> impl Responder {
    let username = body.username.as_deref().unwrap_or("guest");
    let mem_path = format!("{}/._memories.json", get_user_dir(username));
    std::fs::create_dir_all(get_user_dir(username)).ok();

    match body.action.as_str() {
        "store" => {
            let key = match &body.key { Some(k) => k.clone(), None => return HttpResponse::BadRequest().json(serde_json::json!({ "success": false, "error": "Key required" })) };
            let value = match &body.value { Some(v) => v.clone(), None => return HttpResponse::BadRequest().json(serde_json::json!({ "success": false, "error": "Value required" })) };
            let mut memories: HashMap<String, String> = std::fs::read_to_string(&mem_path).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
            memories.insert(key, value);
            std::fs::write(&mem_path, serde_json::to_string_pretty(&memories).unwrap_or_default()).ok();
            HttpResponse::Ok().json(serde_json::json!({ "success": true }))
        }
        "get" => {
            let key = match &body.key { Some(k) => k.clone(), None => return HttpResponse::BadRequest().json(serde_json::json!({ "success": false, "error": "Key required" })) };
            let memories: HashMap<String, String> = std::fs::read_to_string(&mem_path).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
            let value = memories.get(&key).cloned();
            HttpResponse::Ok().json(serde_json::json!({ "success": true, "value": value }))
        }
        "search" => {
            let query = match &body.query { Some(q) => q.clone(), None => return HttpResponse::BadRequest().json(serde_json::json!({ "success": false, "error": "Query required" })) };
            let memories: HashMap<String, String> = std::fs::read_to_string(&mem_path).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
            let results: HashMap<&String, &String> = memories.iter().filter(|(k, v)| k.contains(&query) || v.contains(&query)).collect();
            HttpResponse::Ok().json(serde_json::json!({ "success": true, "results": results }))
        }
        _ => HttpResponse::BadRequest().json(serde_json::json!({ "success": false, "error": "Invalid action" }))
    }
}

/// Chat - direct LLM call
#[post("/api/chat")]
async fn chat(body: web::Json<ChatRequest>) -> impl Responder {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(current_settings().timeout_agent))
        .build()
        .unwrap_or_default();

    let model = body.model.as_deref();
    let response = call_llm(&client, &body.message, model).await;

    HttpResponse::Ok().json(serde_json::json!({
        "success": true,
        "response": response
    }))
}

// ============================================================
// AGENT: Native Tool Calling (Ollama / OpenAI compatible)
// ============================================================

/// Build the system message for the agent
fn build_system_prompt(username: &str, conv_id: &str) -> String {
    let now = chrono::Utc::now();
    format!(
        r#"You are Puterra AI, an advanced agentic assistant inside Puterra Cloud OS.
Current date: {}. User: {}. Conversation ID: {}.

You have access to tools for web search, file operations, code execution, memory, and more.
Use tools when you need real-time data, file operations, or system actions.
NEVER say "I don't have access to real-time data" - use your tools instead!

Rules:
- For current events, weather, prices, news -> use web_search
- You can chain tools: search -> web_fetch -> analyze
- Respond in the SAME LANGUAGE as the user's message
- Use markdown formatting in your answers
- Be helpful and detailed
- If run_python fails due to a missing library, tell the user which library is needed and ASK if they want you to install it. If they agree, install it with shell_exec (pip install <library>). Do NOT keep retrying with workarounds.
- To create PDF files, ALWAYS use the create_pdf tool. NEVER use Python/reportlab/fpdf for PDFs. The create_pdf tool is built-in, reliable, and supports markdown.

## File Storage — CRITICAL RULES
ALL files you create MUST end up in the user's cloud storage so they can download them.

**For text files:** Use `file_write` tool. Supports subpaths: `file_write(name="tasks/folder/file.txt", ...)`

**For binary files created with run_python (videos, images, zip, etc.):**
- `run_python` executes with the user's storage as the CURRENT WORKING DIRECTORY
- ALWAYS use RELATIVE paths: `open('output.mp4', 'wb')` → saves to user storage ✅
- NEVER use absolute paths: `open('/Users/.../output.mp4', 'wb')` → saves to server ❌
- NEVER use `/tmp/`, `~/Desktop/`, or any absolute path
- After creating the file with run_python, tell the user its name so they can find it in Files

**For shell_exec:** shell runs in /tmp by default. If you must create files with shell_exec, pipe output through run_python instead, or explicitly use the user storage path returned by checking `shell_exec(command="pwd")` first.
- The create_pdf tool supports images: ![alt](path). Use image_search to find reliable image URLs before creating PDFs.
- For PDFs with images: FIRST use image_search to find working image URLs, THEN create the PDF. This avoids broken images from blocked URLs.
- image_search tool: Use this to find direct image URLs for maps, diagrams, photos, etc. It returns URLs you can use directly in create_pdf.
- When user says "bunu PDF olarak kaydet" or "save as PDF", use the content from your PREVIOUS answer in this conversation. Do NOT search again.

Tool Reuse Strategy:
- When you write code to accomplish a repeatable task, save it as a custom tool using file_write to the tools/ directory as a JSON file.
- Before writing code from scratch, check existing custom tools with file_list to see if a relevant tool already exists.
- Tools should be GENERAL PURPOSE and REUSABLE, not single-use scripts. Example: write a general "web_scraper" tool with URL+selector params, not a "scrape_kktc_cinemas" tool.
- When an existing tool almost fits, read it with file_read and improve it rather than creating a new one.

## Task Persistence (REQUIRED for multi-step tasks)

For ANY task with 3+ steps, or that involves research, writing, data collection, or file creation:

**FIRST — always check for an existing task folder:**
Before starting anything, run `file_list` and look for `tasks/{}-*` folders.
- If a matching folder exists: read `plan.md` and `progress.md` IMMEDIATELY.
  - The user's new message is almost always a CONTINUATION or UPDATE of the existing task (e.g. "now make a video", "also add X", "change Y"). Treat it as an UPDATE unless it is completely unrelated.
  - On UPDATE: reuse the folder, update `plan.md` with the new goal (keep ✅ completed steps), use `findings.md` to avoid re-doing research, add new steps, continue.
  - Only create a new folder if the topic is genuinely unrelated to existing tasks.
- If no matching folder exists: create one and start fresh.

**At the START of a new task:**
1. Create a task folder: `tasks/CONV_ID-short-task-slug/`  (e.g. `tasks/{}-istanbul-guide/`)
2. Write `tasks/FOLDER/plan.md`:
   - ## Goal: what the user wants
   - ## Steps: numbered list of every step
   - ## Status: 🔄 In Progress

**When the user UPDATES or CHANGES the task:**
1. Read existing `plan.md` and `progress.md`
2. Update `plan.md` with the new goal (keep ✅ completed steps)
3. Add new steps as needed — do NOT discard previous findings
4. Use `findings.md` to avoid repeating already-done research
5. Continue from the first incomplete step

**After EACH completed step:**
- Update `tasks/FOLDER/progress.md`: ✅ done / 🔄 current / ⏳ remaining
- Append new findings to `tasks/FOLDER/findings.md`
- Save created files into `tasks/FOLDER/`

**On ERROR or unexpected stop:**
1. Read `tasks/FOLDER/progress.md` → resume from last incomplete step
2. Read `findings.md` → use existing research, NEVER redo completed work

**At task COMPLETION:** Set `plan.md` Status to ✅ Complete."#,
        now.format("%Y-%m-%d %H:%M UTC"),
        username,
        conv_id,
        conv_id,
        conv_id
    )
}

/// Some APIs (e.g. GLM/Zhipu) require property `type` as ["string"] instead of "string".
/// Walk through tool definitions and wrap string type values inside `properties` as arrays.
fn normalize_tool_property_types(tools: Vec<serde_json::Value>) -> Vec<serde_json::Value> {
    tools.into_iter().map(|mut tool| {
        if let Some(props) = tool
            .get_mut("function")
            .and_then(|f| f.get_mut("parameters"))
            .and_then(|p| p.get_mut("properties"))
            .and_then(|p| p.as_object_mut())
        {
            for prop in props.values_mut() {
                if let Some(t) = prop.get("type").and_then(|t| t.as_str()).map(|s| s.to_string()) {
                    prop["type"] = serde_json::json!([t]);
                }
            }
        }
        tool
    }).collect()
}

/// Build native tool definitions for Ollama/OpenAI tool calling API
fn build_tool_definitions(username: &str) -> Vec<serde_json::Value> {
    let mut tools = vec![
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web using DuckDuckGo for current information, news, prices, events, etc.",
                "parameters": {
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    }
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "web_fetch",
                "description": "Fetch and read the text content of a web page URL",
                "parameters": {
                    "type": "object",
                    "required": ["url"],
                    "properties": {
                        "url": {"type": "string", "description": "URL to fetch"}
                    }
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "image_search",
                "description": "Search for images and get direct image URLs. Use this when you need images for PDFs, presentations, or visual content. Returns image URLs that can be used directly in create_pdf.",
                "parameters": {
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": {"type": "string", "description": "Image search query (e.g., 'Cyprus map', 'sunset beach', 'city skyline')"}
                    }
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "file_list",
                "description": "List all files in the user's cloud storage",
                "parameters": {"type": "object", "properties": {}}
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "file_read",
                "description": "Read the content of a file from user's storage",
                "parameters": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                        "name": {"type": "string", "description": "File name to read"}
                    }
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "file_write",
                "description": "Write content to a file (creates if not exists, overwrites if exists)",
                "parameters": {
                    "type": "object",
                    "required": ["name", "content"],
                    "properties": {
                        "name": {"type": "string", "description": "File name"},
                        "content": {"type": "string", "description": "Content to write"}
                    }
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "file_create",
                "description": "Create an empty file or folder",
                "parameters": {
                    "type": "object",
                    "required": ["name", "kind"],
                    "properties": {
                        "name": {"type": "string", "description": "File or folder name"},
                        "kind": {"type": "string", "description": "Either 'file' or 'folder'", "enum": ["file", "folder"]}
                    }
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "file_delete",
                "description": "Delete a file or folder from user's storage",
                "parameters": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                        "name": {"type": "string", "description": "File or folder name to delete. Can include path like 'folder/file.txt'"}
                    }
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "file_move",
                "description": "Move or rename a file or folder. Use this to move files between folders.",
                "parameters": {
                    "type": "object",
                    "required": ["source", "destination"],
                    "properties": {
                        "source": {"type": "string", "description": "Source path, e.g. 'file.pdf' or 'folder/file.pdf'"},
                        "destination": {"type": "string", "description": "Destination path, e.g. 'travel/file.pdf' or 'new_name.pdf'"}
                    }
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "file_edit",
                "description": "Edit a file by replacing an exact string with a new string. Use this instead of file_write when making small changes to existing files — safer and more efficient.",
                "parameters": {
                    "type": "object",
                    "required": ["name", "old_string", "new_string"],
                    "properties": {
                        "name": {"type": "string", "description": "File name, can include path like 'folder/file.txt'"},
                        "old_string": {"type": "string", "description": "The exact text to find and replace. Must be unique in the file."},
                        "new_string": {"type": "string", "description": "The text to replace it with."}
                    }
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "shell_exec",
                "description": "Execute a shell command on the server",
                "parameters": {
                    "type": "object",
                    "required": ["command"],
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to execute"}
                    }
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "zip_create",
                "description": "Create a ZIP archive from a file or folder in the user's storage",
                "parameters": {
                    "type": "object",
                    "required": ["source"],
                    "properties": {
                        "source": {"type": "string", "description": "File or folder name to zip (e.g. 'report.pdf' or 'travel/')"},
                        "output": {"type": "string", "description": "Output zip filename (optional, defaults to source.zip)"}
                    }
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "unzip",
                "description": "Extract a ZIP archive in the user's storage",
                "parameters": {
                    "type": "object",
                    "required": ["archive"],
                    "properties": {
                        "archive": {"type": "string", "description": "Name of the .zip file to extract"},
                        "destination": {"type": "string", "description": "Destination folder name (optional, defaults to archive name without .zip)"}
                    }
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "cron_create",
                "description": "Schedule a recurring task using a cron expression. The task will run a shell command on schedule.",
                "parameters": {
                    "type": "object",
                    "required": ["name", "schedule", "command"],
                    "properties": {
                        "name": {"type": "string", "description": "Human-readable name for this job"},
                        "schedule": {"type": "string", "description": "5-field cron expression, e.g. '0 9 * * *' = daily at 9am, '*/30 * * * *' = every 30 min"},
                        "command": {"type": "string", "description": "Shell command to run on schedule"},
                        "description": {"type": "string", "description": "What this job does (optional)"}
                    }
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "cron_list",
                "description": "List all scheduled cron jobs",
                "parameters": {"type": "object", "properties": {}}
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "cron_delete",
                "description": "Delete a scheduled cron job by id or name",
                "parameters": {
                    "type": "object",
                    "required": ["id"],
                    "properties": {
                        "id": {"type": "string", "description": "Job id (first 8 chars) or exact name"}
                    }
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "memory_store",
                "description": "Store information in persistent memory for later retrieval",
                "parameters": {
                    "type": "object",
                    "required": ["key", "value"],
                    "properties": {
                        "key": {"type": "string", "description": "Memory key/topic"},
                        "value": {"type": "string", "description": "Information to store"}
                    }
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "memory_search",
                "description": "Search previously stored memories",
                "parameters": {
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": {"type": "string", "description": "Search terms"}
                    }
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "run_python",
                "description": "Execute Python code on the server and return output. If a library is missing, ask the user for permission to install it via shell_exec before retrying.",
                "parameters": {
                    "type": "object",
                    "required": ["code"],
                    "properties": {
                        "code": {"type": "string", "description": "Python code to execute"}
                    }
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "run_javascript",
                "description": "Execute JavaScript/Node.js code on the server and return output",
                "parameters": {
                    "type": "object",
                    "required": ["code"],
                    "properties": {
                        "code": {"type": "string", "description": "JavaScript code to execute"}
                    }
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "create_pdf",
                "description": "Create a PDF document and save it to the user's cloud storage. Supports markdown-style formatting (# headings, **bold**, bullet lists). Perfect for reports, summaries, and documents. Content should be plain text or simple markdown.",
                "parameters": {
                    "type": "object",
                    "required": ["filename", "title", "content"],
                    "properties": {
                        "filename": {"type": "string", "description": "Output filename (e.g. 'report.pdf')"},
                        "title": {"type": "string", "description": "Document title shown at the top of the PDF"},
                        "content": {"type": "string", "description": "Document content as plain text or simple markdown. Use # for headings, **text** for bold, - for bullet points."}
                    }
                }
            }
        }),
    ];

    // Add custom tools from user's tools directory
    let tools_dir = format!("{}/tools", get_user_dir(username));
    if let Ok(entries) = std::fs::read_dir(&tools_dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            if !entry.file_name().to_string_lossy().ends_with(".json") { continue; }
            if let Ok(tool_json) = std::fs::read_to_string(entry.path()) {
                if let Ok(tool_def) = serde_json::from_str::<serde_json::Value>(&tool_json) {
                    let name = tool_def.get("name").and_then(|n| n.as_str()).unwrap_or("unknown");
                    let desc = tool_def.get("description").and_then(|d| d.as_str()).unwrap_or("Custom tool");
                    // Extract properties and required from the tool's parameters object
                    let params_val = tool_def.get("parameters");
                    let props = params_val
                        .and_then(|p| p.get("properties"))
                        .cloned()
                        .unwrap_or(serde_json::json!({}));
                    let required = params_val
                        .and_then(|p| p.get("required"))
                        .cloned()
                        .unwrap_or(serde_json::json!([]));
                    tools.push(serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": format!("custom_{}", name),
                            "description": desc,
                            "parameters": {
                                "type": "object",
                                "properties": props,
                                "required": required
                            }
                        }
                    }));
                }
            }
        }
    }

    tools  // normalization is applied in call_llm_chat based on endpoint type
}

/// Execute a tool directly (no HTTP calls to self)
async fn execute_tool(
    tool: &str,
    input_json: &str,
    username: &str,
    client: &reqwest::Client,
) -> String {
    let input: serde_json::Value = serde_json::from_str(input_json).unwrap_or(serde_json::json!({}));

    match tool.to_lowercase().as_str() {
        "web_search" => {
            let query = input.get("query").and_then(|q| q.as_str()).unwrap_or("");
            if query.is_empty() { return "Error: query is required".to_string(); }

            let results = do_web_search(client, query, 8).await;
            if results.is_empty() {
                return "No search results found. Try a different query.".to_string();
            }

            let mut out = format!("Found {} results:\n\n", results.len());
            for (i, r) in results.iter().enumerate() {
                out += &format!("{}. **{}**\n   URL: {}\n   {}\n\n", i + 1, r.title, r.url, r.snippet);
            }
            out
        }

        "web_fetch" => {
            let url = input.get("url").and_then(|u| u.as_str()).unwrap_or("");
            if url.is_empty() { return "Error: url is required".to_string(); }

            match do_web_fetch(client, url).await {
                Ok(content) => safe_truncate(&content, 5000).to_string(),
                Err(e) => format!("Fetch error: {}", e),
            }
        }

        "image_search" => {
            let query = input.get("query").and_then(|q| q.as_str()).unwrap_or("");
            if query.is_empty() { return "Error: query is required".to_string(); }

            let results = do_image_search(client, query, 5).await;
            if results.is_empty() {
                return "No images found. Try a different search query.".to_string();
            }

            let mut out = format!("Found {} images for '{}':\n\n", results.len(), query);
            for (i, r) in results.iter().enumerate() {
                out += &format!("{}. {}\n   Source: {}\n   Usage: ![image]({})\n\n", i + 1, r.url, r.source, r.url);
            }
            out += "\nCopy any URL above to use in create_pdf with ![alt](URL)";
            out
        }

        "file_list" => {
            let user_dir = get_user_dir(username);
            std::fs::create_dir_all(&user_dir).ok();
            match std::fs::read_dir(&user_dir) {
                Ok(entries) => {
                    let files: Vec<String> = entries
                        .filter_map(|e| e.ok())
                        .filter(|e| !e.file_name().to_string_lossy().starts_with("._"))
                        .map(|e| {
                            let name = e.file_name().to_string_lossy().to_string();
                            if e.path().is_dir() { format!("[folder] {}", name) }
                            else { format!("[file] {} ({} bytes)", name, e.metadata().map(|m| m.len()).unwrap_or(0)) }
                        })
                        .collect();
                    if files.is_empty() { "No files found.".to_string() }
                    else { format!("Files:\n{}", files.join("\n")) }
                }
                Err(e) => format!("Error: {}", e),
            }
        }

        "file_read" => {
            let name = input.get("name").and_then(|n| n.as_str()).unwrap_or("");
            if name.is_empty() || name.contains("..") {
                return "Error: valid file name is required".to_string();
            }
            match std::fs::read_to_string(format!("{}/{}", get_user_dir(username), name)) {
                Ok(content) if content.is_empty() => format!("File '{}' is empty.", name),
                Ok(content) => format!("Content of '{}':\n{}", name, safe_truncate(&content, 5000)),
                Err(e) => format!("Error reading '{}': {}", name, e),
            }
        }

        "file_write" => {
            let name = input.get("name").and_then(|n| n.as_str()).unwrap_or("");
            let content = input.get("content").and_then(|c| c.as_str()).unwrap_or("");
            if name.is_empty() || name.contains("..") {
                return "Error: valid file name is required".to_string();
            }
            let user_dir = get_user_dir(username);
            let path = format!("{}/{}", user_dir, name);
            if let Some(parent) = std::path::Path::new(&path).parent() {
                std::fs::create_dir_all(parent).ok();
            }
            match std::fs::write(&path, content) {
                Ok(_) => format!("Written {} bytes to '{}'", content.len(), name),
                Err(e) => format!("Error writing '{}': {}", name, e),
            }
        }

        "file_create" => {
            let name = input.get("name").and_then(|n| n.as_str()).unwrap_or("");
            let item_type = input.get("kind").or_else(|| input.get("type")).and_then(|t| t.as_str()).unwrap_or("file");
            if name.is_empty() || name.contains("..") {
                return "Error: valid name is required".to_string();
            }
            let user_dir = get_user_dir(username);
            let path = format!("{}/{}", user_dir, name);
            if let Some(parent) = std::path::Path::new(&path).parent() {
                std::fs::create_dir_all(parent).ok();
            }
            match if item_type == "folder" { std::fs::create_dir(&path) } else { std::fs::write(&path, "").map(|_| ()) } {
                Ok(_) => format!("Created {} '{}'", item_type, name),
                Err(e) => format!("Error: {}", e),
            }
        }

        "file_edit" => {
            let name = input.get("name").and_then(|n| n.as_str()).unwrap_or("");
            let old_string = input.get("old_string").and_then(|s| s.as_str()).unwrap_or("");
            let new_string = input.get("new_string").and_then(|s| s.as_str()).unwrap_or("");
            if name.is_empty() || name.contains("..") {
                return "Error: valid file name is required".to_string();
            }
            if old_string.is_empty() {
                return "Error: old_string cannot be empty".to_string();
            }
            let path = format!("{}/{}", get_user_dir(username), name);
            match std::fs::read_to_string(&path) {
                Err(e) => format!("Error reading '{}': {}", name, e),
                Ok(content) => {
                    let count = content.matches(old_string).count();
                    if count == 0 {
                        return format!("Error: old_string not found in '{}'", name);
                    }
                    if count > 1 {
                        return format!("Error: old_string matches {} times in '{}' — make it more specific", count, name);
                    }
                    let new_content = content.replacen(old_string, new_string, 1);
                    match std::fs::write(&path, &new_content) {
                        Ok(_) => format!("Edited '{}': replaced {} chars with {} chars", name, old_string.len(), new_string.len()),
                        Err(e) => format!("Error writing '{}': {}", name, e),
                    }
                }
            }
        }

        "file_delete" => {
            let name = input.get("name").and_then(|n| n.as_str()).unwrap_or("");
            if name.is_empty() || name.contains("..") {
                return "Error: valid name is required".to_string();
            }
            let path = format!("{}/{}", get_user_dir(username), name);
            if std::fs::remove_file(&path).is_ok() || std::fs::remove_dir_all(&path).is_ok() {
                format!("Deleted '{}'", name)
            } else {
                format!("'{}' not found", name)
            }
        }

        "file_move" => {
            let source = input.get("source").and_then(|s| s.as_str()).unwrap_or("");
            let destination = input.get("destination").and_then(|d| d.as_str()).unwrap_or("");
            if source.is_empty() || destination.is_empty() {
                return "Error: source and destination are required".to_string();
            }
            if source.contains("..") || destination.contains("..") {
                return "Error: invalid path".to_string();
            }
            let user_dir = get_user_dir(username);
            let src_path = format!("{}/{}", user_dir, source);
            let dst_path = format!("{}/{}", user_dir, destination);
            // Create parent directories for destination if needed
            if let Some(parent) = std::path::Path::new(&dst_path).parent() {
                std::fs::create_dir_all(parent).ok();
            }
            match std::fs::rename(&src_path, &dst_path) {
                Ok(_) => format!("Moved '{}' to '{}'", source, destination),
                Err(e) => format!("Error moving file: {}", e),
            }
        }

        "shell_exec" => {
            let command = input.get("command").and_then(|c| c.as_str()).unwrap_or("");
            if command.is_empty() { return "Error: command is required".to_string(); }
            let dangerous = ["rm -rf", "mkfs", "dd if=", "> /dev/", "chmod 777", "chown root"];
            if dangerous.iter().any(|d| command.contains(d)) {
                return "Error: command blocked for safety".to_string();
            }
            let shell_cwd = get_user_dir(username);
            std::fs::create_dir_all(&shell_cwd).ok();
            match Command::new("sh").arg("-c").arg(command).current_dir(&shell_cwd).output() {
                Ok(output) => {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    let mut result = String::new();
                    if !stdout.is_empty() { result += &format!("stdout:\n{}\n", safe_truncate(&stdout, 3000)); }
                    if !stderr.is_empty() { result += &format!("stderr:\n{}\n", safe_truncate(&stderr, 1000)); }
                    if result.is_empty() { format!("OK (exit {})", output.status.code().unwrap_or(-1)) } else { result }
                }
                Err(e) => format!("Failed: {}", e),
            }
        }

        "memory_store" => {
            let key = input.get("key").and_then(|k| k.as_str()).unwrap_or("");
            let value = input.get("value").and_then(|v| v.as_str()).unwrap_or("");
            if key.is_empty() || value.is_empty() { return "Error: key and value required".to_string(); }
            let user_dir = get_user_dir(username);
            std::fs::create_dir_all(&user_dir).ok();
            let mem_path = format!("{}/._memories.json", user_dir);
            let mut memories: HashMap<String, String> = std::fs::read_to_string(&mem_path).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
            memories.insert(key.to_string(), value.to_string());
            std::fs::write(&mem_path, serde_json::to_string_pretty(&memories).unwrap_or_default()).ok();
            format!("Stored: '{}' = '{}'", key, safe_truncate(value, 100))
        }

        "memory_search" => {
            let query = input.get("query").and_then(|q| q.as_str()).unwrap_or("");
            let mem_path = format!("{}/._memories.json", get_user_dir(username));
            let memories: HashMap<String, String> = std::fs::read_to_string(&mem_path).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
            let results: Vec<String> = memories.iter()
                .filter(|(k, v)| k.contains(query) || v.contains(query))
                .map(|(k, v)| format!("- {}: {}", k, safe_truncate(v, 200)))
                .collect();
            if results.is_empty() { format!("No memories matching '{}'", query) }
            else { format!("Found {}:\n{}", results.len(), results.join("\n")) }
        }

        "run_python" => {
            let code = input.get("code").and_then(|c| c.as_str()).unwrap_or("");
            if code.is_empty() { return "Error: code is required".to_string(); }
            let settings = current_settings();
            if !settings.shell_enabled { return "Error: code execution disabled".to_string(); }

            let user_dir = get_user_dir(username);
            std::fs::create_dir_all(&user_dir).ok();
            let tmp = format!("{}/._agent_run.py", user_dir);
            eprintln!("[run_python] Writing {} bytes to {}", code.len(), tmp);
            if let Err(e) = std::fs::write(&tmp, code) {
                return format!("Error writing temp file: {}", e);
            }
            let result = match Command::new("python3")
                .arg(&tmp)
                .current_dir(&user_dir)
                .env("PYTHONIOENCODING", "utf-8")
                .output() {
                Ok(o) => {
                    let mut r = String::new();
                    let stdout = String::from_utf8_lossy(&o.stdout);
                    let stderr = String::from_utf8_lossy(&o.stderr);
                    eprintln!("[run_python] exit={} stdout={} stderr={}", o.status.code().unwrap_or(-1), stdout.len(), stderr.len());
                    if !stdout.is_empty() { r += &stdout; }
                    if !stderr.is_empty() { r += &format!("\nSTDERR: {}", stderr); }
                    if r.is_empty() { "OK (no output)".to_string() } else { safe_truncate(&r, 3000).to_string() }
                }
                Err(e) => format!("Python not available: {}", e),
            };
            std::fs::remove_file(&tmp).ok();
            result
        }

        "run_javascript" | "run_js" => {
            let code = input.get("code").and_then(|c| c.as_str()).unwrap_or("");
            if code.is_empty() { return "Error: code is required".to_string(); }
            let settings = current_settings();
            if !settings.shell_enabled { return "Error: code execution disabled".to_string(); }

            let user_dir = get_user_dir(username);
            std::fs::create_dir_all(&user_dir).ok();
            let tmp = format!("{}/._agent_run.js", user_dir);
            std::fs::write(&tmp, code).ok();
            let result = match Command::new("node").arg(&tmp).current_dir(&user_dir).output() {
                Ok(o) => {
                    let mut r = String::new();
                    let stdout = String::from_utf8_lossy(&o.stdout);
                    let stderr = String::from_utf8_lossy(&o.stderr);
                    if !stdout.is_empty() { r += &stdout; }
                    if !stderr.is_empty() { r += &format!("\nSTDERR: {}", stderr); }
                    if r.is_empty() { "OK (no output)".to_string() } else { safe_truncate(&r, 3000).to_string() }
                }
                Err(e) => format!("Node.js not available: {}", e),
            };
            std::fs::remove_file(&tmp).ok();
            result
        }

        "create_pdf" => {
            let filename = input.get("filename").and_then(|f| f.as_str()).unwrap_or("document.pdf");
            let title = input.get("title").and_then(|t| t.as_str()).unwrap_or("Document");
            let content = input.get("content").and_then(|c| c.as_str()).unwrap_or("");
            if content.is_empty() { return "Error: content is required".to_string(); }
            if filename.contains("..") || filename.contains('/') {
                return "Error: invalid filename (no paths, just a name like report.pdf)".to_string();
            }
            let filename = if !filename.ends_with(".pdf") { format!("{}.pdf", filename) } else { filename.to_string() };
            let user_dir = get_user_dir(username);
            std::fs::create_dir_all(&user_dir).ok();
            let output_path = format!("{}/{}", user_dir, filename);
            match generate_pdf(title, content, &output_path) {
                Ok(msg) => msg,
                Err(e) => format!("PDF creation failed: {}", e),
            }
        }

        // Handle custom tools (custom_toolname)
        tool_name if tool_name.starts_with("custom_") => {
            let custom_name = &tool_name[7..];
            let tool_path = format!("{}/tools/{}.json", get_user_dir(username), custom_name);
            match std::fs::read_to_string(&tool_path) {
                Ok(tool_json) => {
                    if let Ok(tool_def) = serde_json::from_str::<serde_json::Value>(&tool_json) {
                        let language = tool_def.get("language").and_then(|l| l.as_str()).unwrap_or("python");
                        let tool_code = tool_def.get("code").and_then(|c| c.as_str()).unwrap_or("");
                        let full_code = match language {
                            "python" => format!("import json\ninput_data = json.loads('{}')\n{}", input_json.replace('\'', "\\'"), tool_code),
                            _ => format!("const input_data = {};\n{}", input_json, tool_code),
                        };
                        let user_dir = get_user_dir(username);
                        std::fs::create_dir_all(&user_dir).ok();
                        let (cmd, ext) = if language == "python" { ("python3", "py") } else { ("node", "js") };
                        let tmp = format!("{}/._custom_run.{}", user_dir, ext);
                        std::fs::write(&tmp, &full_code).ok();
                        let result = match Command::new(cmd).arg(&tmp).current_dir(&user_dir).output() {
                            Ok(o) => {
                                let stdout = String::from_utf8_lossy(&o.stdout);
                                let stderr = String::from_utf8_lossy(&o.stderr);
                                let mut r = stdout.to_string();
                                if !stderr.is_empty() { r += &format!("\nSTDERR: {}", stderr); }
                                if r.trim().is_empty() { "OK".to_string() } else { safe_truncate(&r, 3000).to_string() }
                            }
                            Err(e) => format!("{} not available: {}", cmd, e),
                        };
                        std::fs::remove_file(&tmp).ok();
                        result
                    } else {
                        format!("Error parsing tool: {}", custom_name)
                    }
                }
                Err(_) => format!("Custom tool not found: {}", custom_name),
            }
        }

        // ── zip_create ──────────────────────────────────────────────────────
        "zip_create" => {
            let source = input.get("source").and_then(|s| s.as_str()).unwrap_or("");
            let output = input.get("output").and_then(|s| s.as_str()).unwrap_or("");
            if source.is_empty() { return "Error: source is required".to_string(); }
            if source.contains("..") || output.contains("..") { return "Error: path traversal not allowed".to_string(); }
            let user_dir = get_user_dir(username);
            let src_path = std::path::Path::new(&user_dir).join(source);
            let zip_name = if output.is_empty() {
                format!("{}.zip", source.trim_end_matches('/'))
            } else {
                output.to_string()
            };
            let zip_path = std::path::Path::new(&user_dir).join(&zip_name);

            let zip_file = match std::fs::File::create(&zip_path) {
                Ok(f) => f,
                Err(e) => return format!("Error creating zip file: {}", e),
            };
            let mut zip = zip::ZipWriter::new(zip_file);
            let options: zip::write::SimpleFileOptions = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated);

            let mut file_count = 0u32;
            if src_path.is_file() {
                let fname = src_path.file_name().unwrap_or_default().to_string_lossy().to_string();
                if let Ok(mut f) = std::fs::File::open(&src_path) {
                    let mut buf = Vec::new();
                    if f.read_to_end(&mut buf).is_ok() {
                        if zip.start_file(&fname, options).is_ok() {
                            zip.write_all(&buf).ok();
                            file_count += 1;
                        }
                    }
                }
            } else if src_path.is_dir() {
                fn add_dir_to_zip<W: Write + Seek>(
                    zip: &mut zip::ZipWriter<W>,
                    dir: &std::path::Path,
                    base: &std::path::Path,
                    options: zip::write::SimpleFileOptions,
                    count: &mut u32,
                ) {
                    if let Ok(entries) = std::fs::read_dir(dir) {
                        for entry in entries.flatten() {
                            let path = entry.path();
                            let rel = path.strip_prefix(base).unwrap_or(&path);
                            if path.is_file() {
                                if let Ok(mut f) = std::fs::File::open(&path) {
                                    let mut buf = Vec::new();
                                    if f.read_to_end(&mut buf).is_ok() {
                                        let name = rel.to_string_lossy().to_string();
                                        if zip.start_file(&name, options).is_ok() {
                                            zip.write_all(&buf).ok();
                                            *count += 1;
                                        }
                                    }
                                }
                            } else if path.is_dir() {
                                let name = format!("{}/", rel.to_string_lossy());
                                zip.add_directory(&name, options).ok();
                                add_dir_to_zip(zip, &path, base, options, count);
                            }
                        }
                    }
                }
                add_dir_to_zip(&mut zip, &src_path, &src_path, options, &mut file_count);
            } else {
                return format!("Error: '{}' not found", source);
            }

            match zip.finish() {
                Ok(_) => format!("Created '{}' with {} file(s)", zip_name, file_count),
                Err(e) => format!("Error finalizing zip: {}", e),
            }
        }

        // ── unzip ────────────────────────────────────────────────────────────
        "unzip" => {
            let archive = input.get("archive").and_then(|s| s.as_str()).unwrap_or("");
            let dest = input.get("destination").and_then(|s| s.as_str()).unwrap_or("");
            if archive.is_empty() { return "Error: archive is required".to_string(); }
            if archive.contains("..") || dest.contains("..") { return "Error: path traversal not allowed".to_string(); }
            let user_dir = get_user_dir(username);
            let archive_path = std::path::Path::new(&user_dir).join(archive);
            let dest_dir = if dest.is_empty() {
                std::path::Path::new(&user_dir).join(archive.trim_end_matches(".zip"))
            } else {
                std::path::Path::new(&user_dir).join(dest)
            };

            let zip_file = match std::fs::File::open(&archive_path) {
                Ok(f) => f,
                Err(e) => return format!("Error opening archive: {}", e),
            };
            let mut archive = match zip::ZipArchive::new(zip_file) {
                Ok(a) => a,
                Err(e) => return format!("Error reading zip: {}", e),
            };

            std::fs::create_dir_all(&dest_dir).ok();
            let mut extracted = 0u32;
            for i in 0..archive.len() {
                let mut file = match archive.by_index(i) {
                    Ok(f) => f,
                    Err(_) => continue,
                };
                let out_path = dest_dir.join(file.name());
                if file.name().ends_with('/') {
                    std::fs::create_dir_all(&out_path).ok();
                } else {
                    if let Some(parent) = out_path.parent() {
                        std::fs::create_dir_all(parent).ok();
                    }
                    if let Ok(mut out) = std::fs::File::create(&out_path) {
                        let mut buf = Vec::new();
                        if file.read_to_end(&mut buf).is_ok() {
                            out.write_all(&buf).ok();
                            extracted += 1;
                        }
                    }
                }
            }
            format!("Extracted {} file(s) to '{}'", extracted, dest_dir.file_name().unwrap_or_default().to_string_lossy())
        }

        // ── cron_create ──────────────────────────────────────────────────────
        "cron_create" => {
            let name    = input.get("name").and_then(|s| s.as_str()).unwrap_or("").trim().to_string();
            let schedule = input.get("schedule").and_then(|s| s.as_str()).unwrap_or("").trim().to_string();
            let command = input.get("command").and_then(|s| s.as_str()).unwrap_or("").trim().to_string();
            let description = input.get("description").and_then(|s| s.as_str()).unwrap_or("").to_string();
            if name.is_empty() || schedule.is_empty() || command.is_empty() {
                return "Error: name, schedule, and command are required".to_string();
            }
            // Validate cron expression (5 fields)
            let parts: Vec<&str> = schedule.split_whitespace().collect();
            if parts.len() != 5 {
                return "Error: schedule must be a 5-field cron expression (e.g. '0 9 * * *' = daily at 9am)".to_string();
            }
            let cron_dir = format!("data/cron/{}", username);
            std::fs::create_dir_all(&cron_dir).ok();
            let jobs_path = format!("{}/jobs.json", cron_dir);
            let mut jobs: Vec<serde_json::Value> = std::fs::read_to_string(&jobs_path)
                .ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
            let id = Uuid::new_v4().to_string();
            let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
            jobs.push(serde_json::json!({
                "id": id, "name": name, "schedule": schedule,
                "command": command, "description": description,
                "created_at": now, "last_run": null, "enabled": true
            }));
            match std::fs::write(&jobs_path, serde_json::to_string_pretty(&jobs).unwrap_or_default()) {
                Ok(_) => format!("Cron job '{}' created (id: {}). Schedule: {} — runs: {}", name, &id[..8], schedule, command),
                Err(e) => format!("Error saving cron job: {}", e),
            }
        }

        // ── cron_list ────────────────────────────────────────────────────────
        "cron_list" => {
            let jobs_path = format!("data/cron/{}/jobs.json", username);
            let jobs: Vec<serde_json::Value> = std::fs::read_to_string(&jobs_path)
                .ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
            if jobs.is_empty() {
                return "No cron jobs scheduled.".to_string();
            }
            let mut out = format!("{} scheduled job(s):\n", jobs.len());
            for j in &jobs {
                let enabled = if j.get("enabled").and_then(|e| e.as_bool()).unwrap_or(true) { "✅" } else { "⏸️" };
                let last = j.get("last_run").and_then(|l| l.as_u64()).map(|t| {
                    chrono::DateTime::<chrono::Utc>::from(std::time::UNIX_EPOCH + std::time::Duration::from_secs(t))
                        .format("%Y-%m-%d %H:%M UTC").to_string()
                }).unwrap_or("never".to_string());
                out.push_str(&format!(
                    "{} [{}] {} | schedule: {} | last: {} | cmd: {}\n",
                    enabled,
                    &j.get("id").and_then(|i| i.as_str()).unwrap_or("?")[..8],
                    j.get("name").and_then(|n| n.as_str()).unwrap_or("?"),
                    j.get("schedule").and_then(|s| s.as_str()).unwrap_or("?"),
                    last,
                    j.get("command").and_then(|c| c.as_str()).unwrap_or("?"),
                ));
            }
            out.trim_end().to_string()
        }

        // ── cron_delete ──────────────────────────────────────────────────────
        "cron_delete" => {
            let id_or_name = input.get("id").and_then(|s| s.as_str())
                .or_else(|| input.get("name").and_then(|s| s.as_str()))
                .unwrap_or("");
            if id_or_name.is_empty() { return "Error: id or name is required".to_string(); }
            let jobs_path = format!("data/cron/{}/jobs.json", username);
            let mut jobs: Vec<serde_json::Value> = std::fs::read_to_string(&jobs_path)
                .ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
            let before = jobs.len();
            jobs.retain(|j| {
                let jid = j.get("id").and_then(|i| i.as_str()).unwrap_or("");
                let jname = j.get("name").and_then(|n| n.as_str()).unwrap_or("");
                !jid.starts_with(id_or_name) && jname != id_or_name
            });
            let removed = before - jobs.len();
            if removed == 0 { return format!("No job found matching '{}'", id_or_name); }
            std::fs::write(&jobs_path, serde_json::to_string_pretty(&jobs).unwrap_or_default()).ok();
            format!("Deleted {} cron job(s) matching '{}'", removed, id_or_name)
        }

        _ => format!("Unknown tool: '{}'", tool),
    }
}

/// Build a ReAct-style system prompt (fallback for models without native tool calling)
fn build_react_system_prompt(username: &str, conv_id: &str) -> String {
    let now = chrono::Utc::now();
    format!(r#"You are Puterra AI, an advanced agentic assistant inside Puterra Cloud OS.
Current date: {}. User: {}. Conversation ID: {}.

You have access to powerful tools. You MUST use tools when you need real-time information, file operations, or any action.

## Available Tools

1. **web_search** - Search the web for current information
   Input: {{"query": "search terms"}}
2. **web_fetch** - Fetch and read content from a specific URL
   Input: {{"url": "https://example.com"}}
3. **file_list** - List all files in the user's storage
   Input: {{}}
4. **file_read** - Read content of a file
   Input: {{"name": "filename.txt"}}
5. **file_write** - Write content to a file
   Input: {{"name": "filename.txt", "content": "file content"}}
6. **file_create** - Create an empty file or folder
   Input: {{"name": "filename.txt", "kind": "file"}}
7. **file_edit** - Edit a file by replacing an exact string (safer than file_write for small changes)
   Input: {{"name": "filename.txt", "old_string": "text to replace", "new_string": "replacement text"}}
8. **file_delete** - Delete a file
   Input: {{"name": "filename.txt"}}
9. **shell_exec** - Execute a shell command
   Input: {{"command": "ls -la"}}
10. **memory_store** - Store information for later retrieval
    Input: {{"key": "topic", "value": "information"}}
11. **memory_search** - Search stored memories
    Input: {{"query": "search terms"}}
12. **run_python** - Execute Python code
    Input: {{"code": "print('hello')"}}
13. **run_javascript** - Execute JavaScript/Node.js code
    Input: {{"code": "console.log('hello')"}}
14. **create_pdf** - Create a PDF document and save to user's storage
    Input: {{"filename": "report.pdf", "title": "My Report", "content": "Heading\n\nBody text here..."}}
15. **zip_create** - Create a ZIP archive from a file or folder
    Input: {{"source": "travel/", "output": "travel_backup.zip"}}
16. **unzip** - Extract a ZIP archive
    Input: {{"archive": "files.zip", "destination": "extracted/"}}
17. **cron_create** - Schedule a recurring shell command
    Input: {{"name": "daily backup", "schedule": "0 9 * * *", "command": "echo backup done >> backup.log"}}
18. **cron_list** - List all scheduled jobs
    Input: {{}}
19. **cron_delete** - Delete a scheduled job
    Input: {{"id": "abc12345"}}

## How to respond

**When you need to use a tool:**
Thought: [your reasoning]
Action: [tool_name]
Action Input: [valid JSON input]

**When you have the final answer:**
Thought: [brief reasoning]
Final Answer: [your complete answer with markdown formatting]

## Rules
- For current events, weather, prices, news, schedules -> use web_search FIRST
- You CAN chain multiple tools: search -> fetch -> analyze -> answer
- Respond in the SAME LANGUAGE as the user message
- NEVER say you do not have access to real-time data - USE the tools!
- To create PDF files, ALWAYS use the create_pdf tool. Do NOT use Python for PDF creation.
- After receiving an Observation, you MUST either use another tool OR give a Final Answer

## Task Persistence (REQUIRED for multi-step tasks)

For ANY task with 3+ steps (research, writing, data collection, file creation):

**FIRST — check existing task folder:** Look for `tasks/{}-*` folders.
- Found → read plan.md & progress.md. The new message is almost always a CONTINUATION (e.g. "now make a video", "add X"). Treat as UPDATE: reuse folder, update plan.md, keep ✅ steps, use findings.md. Only create a new folder if completely unrelated.
- Not found → create new folder.

**When user UPDATES the task:** Read plan.md/progress.md → update goal → keep ✅ completed steps → add new steps → use findings.md (NEVER redo research already done).

**New task folder:** `tasks/CONV_ID-slug/plan.md` (e.g. `tasks/{}-guide/`)

**After EACH step:** Update `tasks/FOLDER/progress.md` (✅/🔄/⏳), append to findings.md.

**On ERROR:** Read progress.md → resume from last incomplete step. NEVER restart.

**On COMPLETION:** Mark plan.md as ✅ Complete."#,
        now.format("%Y-%m-%d %H:%M UTC"), username, conv_id, conv_id, conv_id)
}

/// Parse ReAct-style response (fallback)
fn parse_react_response(response: &str) -> (String, Option<String>, Option<String>, Option<String>) {
    let thought_re = regex::Regex::new(r"(?i)Thought:\s*(.+?)(?:\n|$)").unwrap();
    let thought = thought_re.captures(response)
        .map(|c| c[1].trim().to_string())
        .unwrap_or_default();

    let final_re = regex::Regex::new(r"(?is)Final\s*Answer:\s*(.+)$").unwrap();
    if let Some(caps) = final_re.captures(response) {
        return (thought, None, None, Some(caps[1].trim().to_string()));
    }

    let action_re = regex::Regex::new(r"(?i)Action:\s*(\w+)").unwrap();
    let action = action_re.captures(response).map(|c| c[1].trim().to_string());

    let input_re = regex::Regex::new(r"(?is)Action\s*Input:\s*(\{.*?\})").unwrap();
    let mut action_input = input_re.captures(response).map(|c| c[1].trim().to_string());

    if action.is_some() && action_input.is_none() {
        let input_line_re = regex::Regex::new(r"(?i)Action\s*Input:\s*(.+?)(?:\n|$)").unwrap();
        if let Some(caps) = input_line_re.captures(response) {
            let raw = caps[1].trim();
            action_input = Some(if raw.starts_with('{') {
                raw.to_string()
            } else {
                format!(r#"{{"query": "{}"}}"#, raw)
            });
        }
    }

    (thought, action, action_input, None)
}

/// Agent with ReAct text-based fallback — streams thinking/tool events via tx in real-time
async fn agent_react_fallback(
    client: &reqwest::Client,
    message: &str,
    history: &Option<Vec<AgentChatMessage>>,
    username: &str,
    conv_id: &str,
    model: Option<&str>,
    share_key: Option<&ShareKey>,
    tx: &mpsc::Sender<String>,
    step_num: &mut usize,
) -> (bool, String) {
    let system_prompt = build_react_system_prompt(username, conv_id);
    let settings = current_settings();
    let max_iterations = settings.max_agent_iterations;

    let mut conversation = String::new();
    if let Some(hist) = history {
        let recent: Vec<&AgentChatMessage> = hist.iter().rev().take(10).collect::<Vec<_>>().into_iter().rev().collect();
        let total = recent.len();
        for (i, msg) in recent.iter().enumerate() {
            let prefix = if msg.role == "user" { "User" } else { "Assistant" };
            let is_recent = i >= total.saturating_sub(4);
            let content = if is_recent || msg.content.len() <= 2000 {
                msg.content.clone()
            } else {
                format!("{}...", safe_truncate(&msg.content, 1000))
            };
            conversation += &format!("{}: {}\n", prefix, content);
        }
    }
    conversation += &format!("User: {}\n\n", message);

    let mut scratchpad = String::new();

    for _iteration in 0..max_iterations {
        let full_prompt = format!(
            "{}\n\n## Conversation\n{}\n## Agent Scratchpad\n{}\n\nContinue with your next Thought and Action, or provide a Final Answer.",
            system_prompt, conversation, scratchpad
        );

        let _ = tx.send(sse_event(&serde_json::json!({"type": "status", "message": format!("Thinking... (step {})", *step_num + 1)}))).await;

        let messages = vec![serde_json::json!({"role": "user", "content": full_prompt})];
        let (llm_response, step_thinking) = match call_llm_chat(client, &messages, None, model, share_key).await {
            Ok(json) => {
                let msg = json.get("message").cloned().unwrap_or_default();
                let (content, thinking) = extract_content(&msg);
                (content, thinking)
            },
            Err(e) => return (false, format!("LLM Error: {}", e)),
        };

        // Emit thinking immediately
        if let Some(ref t) = step_thinking {
            let _ = tx.send(sse_event(&serde_json::json!({"type": "thinking", "thinking": t, "step": *step_num}))).await;
        }

        if llm_response.is_empty() {
            return (false, "LLM returned empty response.".to_string());
        }

        let (thought, action, action_input, final_answer) = parse_react_response(&llm_response);

        if let Some(answer) = final_answer {
            *step_num += 1;
            return (true, answer);
        }

        if let Some(ref act) = action {
            let input_str = action_input.as_deref().unwrap_or("{}");

            // Emit tool_call before execution
            let _ = tx.send(sse_event(&serde_json::json!({"type": "tool_call", "action": act, "action_input": input_str, "step": *step_num}))).await;

            let observation = execute_tool(act, input_str, username, client).await;

            // Emit tool_result after execution
            let _ = tx.send(sse_event(&serde_json::json!({"type": "tool_result", "action": act, "observation": safe_truncate(&observation, 500), "step": *step_num}))).await;

            *step_num += 1;

            scratchpad += &format!(
                "Thought: {}\nAction: {}\nAction Input: {}\nObservation: {}\n\n",
                thought, act, input_str, safe_truncate(&observation, 3000)
            );
            continue;
        }

        // No structured output - return as-is
        let answer = if thought.is_empty() { llm_response.trim().to_string() } else { thought };
        return (true, answer);
    }

    // Max iterations summary
    let _ = tx.send(sse_event(&serde_json::json!({"type": "status", "message": "Summarizing findings..."}))).await;
    let summary_prompt = format!(
        "{}\n\n## Conversation\n{}\n## Research Done\n{}\n\nProvide a Final Answer based on all observations above.",
        system_prompt, conversation, scratchpad
    );
    let messages = vec![serde_json::json!({"role": "user", "content": summary_prompt})];
    let (final_response, summary_thinking) = match call_llm_chat(client, &messages, None, model, share_key).await {
        Ok(json) => {
            let msg = json.get("message").cloned().unwrap_or_default();
            let (content, thinking) = extract_content(&msg);
            (content, thinking)
        },
        Err(e) => return (false, format!("Error: {}", e)),
    };
    if let Some(ref t) = summary_thinking {
        let _ = tx.send(sse_event(&serde_json::json!({"type": "thinking", "thinking": t, "step": *step_num}))).await;
    }
    let (_, _, _, final_answer) = parse_react_response(&final_response);
    let answer = final_answer.unwrap_or_else(|| final_response.trim().to_string());

    (true, answer)
}

/// Extract content and thinking from LLM message.
/// Handles both dedicated `thinking` field (Ollama native) and <think>...</think> tags in content.
fn extract_content(message: &serde_json::Value) -> (String, Option<String>) {
    let raw_content = message.get("content").and_then(|c| c.as_str()).unwrap_or("").to_string();

    // 1. Dedicated `thinking` field (Ollama native think mode)
    let native_thinking = message.get("thinking").and_then(|t| t.as_str())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_string());

    // 2. <think>...</think> tags embedded in content (DeepSeek R1, QwQ, etc.)
    let (tag_thinking, clean_content) = {
        let re = regex::Regex::new(r"(?s)<think>(.*?)</think>").unwrap();
        let mut thoughts = Vec::new();
        for cap in re.captures_iter(&raw_content) {
            let t = cap[1].trim().to_string();
            if !t.is_empty() { thoughts.push(t); }
        }
        let cleaned = re.replace_all(&raw_content, "").trim().to_string();
        if thoughts.is_empty() {
            (None, raw_content.clone())
        } else {
            (Some(thoughts.join("\n\n")), cleaned)
        }
    };

    let thinking = native_thinking.or(tag_thinking);
    let content = if clean_content.is_empty() {
        thinking.clone().unwrap_or_default()
    } else {
        clean_content
    };

    (content, thinking)
}

/// Helper: format SSE event
fn sse_event(data: &serde_json::Value) -> String {
    format!("data: {}\n\n", serde_json::to_string(data).unwrap_or_default())
}

/// Agent endpoint - SSE streaming with native tool calling and ReAct fallback
#[post("/api/agent")]
async fn agent_chat(app_data: web::Data<AppState>, body: web::Json<AgentRequest>) -> HttpResponse {
    let message = body.message.clone();
    let history = body.history.clone();
    let model = body.model.clone();
    let username = body.username.clone().unwrap_or_else(|| "guest".to_string());
    let conv_id = body.conv_id.clone().unwrap_or_default();

    // Look up share key if provided — increment uses, validate active
    let share_key_opt: Option<ShareKey> = body.share_key.as_deref().and_then(|key_id| {
        let mut keys = app_data.share_keys.lock().unwrap();
        let k = keys.get_mut(key_id)?;
        if !k.active { return None; }
        k.uses += 1;
        if let Some(max) = k.max_uses {
            if k.uses >= max { k.active = false; }
        }
        let cloned = k.clone();
        drop(keys);
        // Save updated usage count (best-effort, don't block)
        let all_keys = app_data.share_keys.lock().unwrap().clone();
        save_share_keys(&all_keys);
        Some(cloned)
    });

    let (tx, rx) = mpsc::channel::<String>(64);

    // Spawn the agent loop in a background task
    tokio::spawn(async move {
        let model_ref = model.as_deref();
        let system_prompt = build_system_prompt(&username, &conv_id);
        let tool_defs = build_tool_definitions(&username);
        let settings = current_settings();
        let max_iterations = settings.max_agent_iterations;
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(settings.timeout_agent))
            .build()
            .unwrap_or_default();
        let mut step_num: usize = 0;

        // Build messages array
        let mut messages: Vec<serde_json::Value> = vec![
            serde_json::json!({"role": "system", "content": system_prompt}),
        ];

        if let Some(hist) = &history {
            let recent: Vec<&AgentChatMessage> = hist.iter().rev().take(10).collect::<Vec<_>>().into_iter().rev().collect();
            let total = recent.len();
            for (i, msg) in recent.iter().enumerate() {
                let is_recent = i >= total.saturating_sub(4);
                let content = if is_recent || msg.content.len() <= 2000 {
                    msg.content.clone()
                } else {
                    format!("{}...", safe_truncate(&msg.content, 1000))
                };
                messages.push(serde_json::json!({"role": msg.role, "content": content}));
            }
        }

        messages.push(serde_json::json!({"role": "user", "content": message}));

        // --- First LLM call ---
        let _ = tx.send(sse_event(&serde_json::json!({"type": "status", "message": "Sending query to LLM..."}))).await;
        let first_result = call_llm_chat(&client, &messages, Some(&tool_defs), model_ref, share_key_opt.as_ref()).await;

        let native_works = match &first_result {
            Ok(json) => {
                let msg = json.get("message");
                if let Some(m) = msg {
                    let has_content = m.get("content").and_then(|c| c.as_str()).map_or(false, |c| !c.is_empty());
                    let has_tools = m.get("tool_calls").and_then(|tc| tc.as_array()).map_or(false, |tc| !tc.is_empty());
                    has_content || has_tools
                } else { false }
            }
            Err(_) => false,
        };

        // ReAct fallback — streams events in real-time via tx
        if !native_works {
            let (success, answer) =
                agent_react_fallback(&client, &message, &history, &username, &conv_id, model_ref, share_key_opt.as_ref(), &tx, &mut step_num).await;
            let _ = tx.send(sse_event(&serde_json::json!({"type": "answer", "answer": answer, "success": success}))).await;
            let _ = tx.send(sse_event(&serde_json::json!({"type": "done"}))).await;
            return;
        }

        let response_json = first_result.unwrap();
        let msg = response_json.get("message").unwrap().clone();
        let (content, thinking) = extract_content(&msg);
        let tool_calls = msg.get("tool_calls").and_then(|tc| tc.as_array()).cloned();

        // Send thinking event
        if let Some(ref t) = thinking {
            let _ = tx.send(sse_event(&serde_json::json!({"type": "thinking", "thinking": t, "step": step_num}))).await;
        }

        // No tool calls - direct answer
        if tool_calls.is_none() || tool_calls.as_ref().map_or(true, |tc| tc.is_empty()) {
            let answer = if content.is_empty() { "I couldn't generate a response.".to_string() } else { content };
            let _ = tx.send(sse_event(&serde_json::json!({"type": "answer", "answer": answer, "success": true}))).await;
            let _ = tx.send(sse_event(&serde_json::json!({"type": "done"}))).await;
            return;
        }

        // Process first round of tool calls
        let tool_calls = tool_calls.unwrap();
        messages.push(msg.clone());

        for tc in &tool_calls {
            let func = tc.get("function").unwrap_or(tc);
            let tool_name = func.get("name").and_then(|n| n.as_str()).unwrap_or("unknown");
            let arguments = func.get("arguments").cloned().unwrap_or(serde_json::json!({}));
            let args_str = if arguments.is_string() {
                arguments.as_str().unwrap_or("{}").to_string()
            } else {
                serde_json::to_string(&arguments).unwrap_or_else(|_| "{}".to_string())
            };
            // tool_call_id required by OpenAI-compat APIs (Gemini, OpenAI, Groq)
            let tool_call_id = tc.get("id").and_then(|i| i.as_str())
                .unwrap_or(tool_name).to_string();

            // Send tool_call event
            let _ = tx.send(sse_event(&serde_json::json!({
                "type": "tool_call", "action": tool_name, "action_input": args_str, "step": step_num
            }))).await;

            let observation = execute_tool(tool_name, &args_str, &username, &client).await;

            // Send tool_result event
            let _ = tx.send(sse_event(&serde_json::json!({
                "type": "tool_result", "action": tool_name, "observation": safe_truncate(&observation, 500), "step": step_num
            }))).await;

            messages.push(serde_json::json!({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": safe_truncate(&observation, 4000)
            }));
            step_num += 1;
        }

        // Continue tool calling loop
        for iteration in 1..max_iterations {
            eprintln!("[Agent] Iteration {}/{}: Calling LLM...", iteration, max_iterations - 1);
            let _ = tx.send(sse_event(&serde_json::json!({"type": "status", "message": format!("LLM thinking... (iteration {})", iteration)}))).await;

            let llm_result = call_llm_chat(&client, &messages, Some(&tool_defs), model_ref, share_key_opt.as_ref()).await;

            let response_json = match llm_result {
                Ok(json) => json,
                Err(e) => {
                    eprintln!("[Agent] ❌ LLM Error: {}", e);
                    let _ = tx.send(sse_event(&serde_json::json!({"type": "error", "error": format!("LLM Error: {}", e)}))).await;
                    let _ = tx.send(sse_event(&serde_json::json!({"type": "done"}))).await;
                    return;
                }
            };

            let msg = match response_json.get("message") {
                Some(m) => m.clone(),
                None => {
                    eprintln!("[Agent] No message in response, breaking loop");
                    break;
                }
            };

            let (content, thinking) = extract_content(&msg);
            let tool_calls = msg.get("tool_calls").and_then(|tc| tc.as_array()).cloned();

            // Send thinking event
            if let Some(ref t) = thinking {
                eprintln!("[Agent] 💭 Thinking: {}", safe_truncate(t, 100));
                let _ = tx.send(sse_event(&serde_json::json!({"type": "thinking", "thinking": t, "step": step_num}))).await;
            }

            // No tool calls - final answer
            if tool_calls.is_none() || tool_calls.as_ref().map_or(true, |tc| tc.is_empty()) {
                let answer = if content.is_empty() { "I couldn't generate a response.".to_string() } else { content };
                let _ = tx.send(sse_event(&serde_json::json!({"type": "answer", "answer": answer, "success": true}))).await;
                let _ = tx.send(sse_event(&serde_json::json!({"type": "done"}))).await;
                return;
            }

            let tool_calls = tool_calls.unwrap();
            eprintln!("[Agent] 🔧 Calling {} tool(s)...", tool_calls.len());
            messages.push(msg.clone());

            for (idx, tc) in tool_calls.iter().enumerate() {
                let func = tc.get("function").unwrap_or(tc);
                let tool_name = func.get("name").and_then(|n| n.as_str()).unwrap_or("unknown");
                let arguments = func.get("arguments").cloned().unwrap_or(serde_json::json!({}));
                let args_str = if arguments.is_string() {
                    arguments.as_str().unwrap_or("{}").to_string()
                } else {
                    serde_json::to_string(&arguments).unwrap_or_else(|_| "{}".to_string())
                };

                let tool_call_id = tc.get("id").and_then(|i| i.as_str())
                    .unwrap_or(tool_name).to_string();
                eprintln!("[Agent]   [{}/{}] Executing: {} {}", idx + 1, tool_calls.len(), tool_name, safe_truncate(&args_str, 80));

                // Send tool_call event
                let _ = tx.send(sse_event(&serde_json::json!({
                    "type": "tool_call", "action": tool_name, "action_input": args_str, "step": step_num
                }))).await;

                let observation = execute_tool(tool_name, &args_str, &username, &client).await;
                eprintln!("[Agent]   ✓ Result: {}", safe_truncate(&observation, 100));

                // Send tool_result event
                let _ = tx.send(sse_event(&serde_json::json!({
                    "type": "tool_result", "action": tool_name, "observation": safe_truncate(&observation, 500), "step": step_num
                }))).await;

                messages.push(serde_json::json!({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": safe_truncate(&observation, 4000)
                }));
                step_num += 1;
            }
        }

        // Max iterations - ask for summary
        messages.push(serde_json::json!({
            "role": "user",
            "content": "Please provide your final answer now based on all the information you've gathered. Respond in the same language as the original question."
        }));
        let final_result = call_llm_chat(&client, &messages, None, model_ref, share_key_opt.as_ref()).await;
        let answer = match final_result {
            Ok(json) => {
                let m = json.get("message");
                let c = m.and_then(|m| m.get("content")).and_then(|c| c.as_str()).unwrap_or("");
                if c.is_empty() {
                    m.and_then(|m| m.get("thinking")).and_then(|t| t.as_str()).unwrap_or("Max iterations reached.").to_string()
                } else { c.to_string() }
            }
            Err(e) => format!("Error: {}", e),
        };

        let _ = tx.send(sse_event(&serde_json::json!({"type": "answer", "answer": answer, "success": true}))).await;
        let _ = tx.send(sse_event(&serde_json::json!({"type": "done"}))).await;
    });

    // Create SSE streaming response from channel
    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    let body_stream = stream.map(|s| Ok::<_, actix_web::Error>(actix_web::web::Bytes::from(s)));

    HttpResponse::Ok()
        .content_type("text/event-stream")
        .insert_header(("Cache-Control", "no-cache"))
        .insert_header(("X-Accel-Buffering", "no"))
        .streaming(body_stream)
}

// ============================================================
// META ENDPOINTS
// ============================================================

#[get("/api/tools")]
async fn tools_list() -> impl Responder {
    HttpResponse::Ok().json(serde_json::json!({
        "success": true,
        "tools": [
            {"name": "web_search", "description": "Search the web (DuckDuckGo)"},
            {"name": "web_fetch", "description": "Fetch content from a URL"},
            {"name": "image_search", "description": "Search for images and get direct URLs"},
            {"name": "shell_exec", "description": "Execute shell commands"},
            {"name": "memory_store", "description": "Store data in memory"},
            {"name": "memory_search", "description": "Search stored memories"},
            {"name": "run_python", "description": "Execute Python code"},
            {"name": "run_javascript", "description": "Execute JavaScript code"},
            {"name": "file_list", "description": "List files"},
            {"name": "file_read", "description": "Read file content"},
            {"name": "file_write", "description": "Write file content"},
            {"name": "file_edit", "description": "Edit file by replacing exact string"},
            {"name": "file_create", "description": "Create file or folder"},
            {"name": "file_delete", "description": "Delete file or folder"},
            {"name": "create_pdf", "description": "Create PDF documents (built-in Rust, no external deps)"},
            {"name": "zip_create", "description": "Create ZIP archive from file or folder"},
            {"name": "unzip", "description": "Extract ZIP archive"},
            {"name": "cron_create", "description": "Schedule a recurring shell command"},
            {"name": "cron_list", "description": "List scheduled cron jobs"},
            {"name": "cron_delete", "description": "Delete a scheduled cron job"},
            {"name": "agent", "description": "Agentic AI with native tool calling"}
        ]
    }))
}

// ============================================================
// CODE EXECUTION
// ============================================================

#[derive(Deserialize)]
struct RunCodeRequest {
    language: String,
    code: String,
    username: Option<String>,
}

/// Server-side code execution (Python, Node.js, shell)
#[post("/api/run")]
async fn run_code(body: web::Json<RunCodeRequest>) -> impl Responder {
    let settings = current_settings();
    if !settings.shell_enabled {
        return HttpResponse::Forbidden().json(serde_json::json!({
            "success": false, "error": "Code execution is disabled in settings"
        }));
    }

    let username = body.username.as_deref().unwrap_or("guest");
    let user_dir = get_user_dir(username);
    std::fs::create_dir_all(&user_dir).ok();

    // Helper: get absolute path so current_dir + relative arg doesn't double-up
    let abs_user_dir = std::fs::canonicalize(&user_dir)
        .unwrap_or_else(|_| std::path::PathBuf::from(&user_dir));

    match body.language.as_str() {
        "python" => {
            let tmp_file = abs_user_dir.join("._run_tmp.py");
            std::fs::write(&tmp_file, &body.code).ok();

            match Command::new("python3")
                .arg(&tmp_file)
                .current_dir(&abs_user_dir)
                .output()
            {
                Ok(output) => {
                    std::fs::remove_file(&tmp_file).ok();
                    HttpResponse::Ok().json(serde_json::json!({
                        "success": output.status.success(),
                        "stdout": String::from_utf8_lossy(&output.stdout),
                        "stderr": String::from_utf8_lossy(&output.stderr),
                        "code": output.status.code()
                    }))
                }
                Err(e) => {
                    std::fs::remove_file(&tmp_file).ok();
                    HttpResponse::Ok().json(serde_json::json!({
                        "success": false,
                        "error": format!("Python not available: {}", e),
                        "hint": "Python runs in-browser via Pyodide. Install Python 3 on server for server-side execution."
                    }))
                }
            }
        }

        "javascript" | "js" => {
            let tmp_file = abs_user_dir.join("._run_tmp.js");
            std::fs::write(&tmp_file, &body.code).ok();

            match Command::new("node")
                .arg(&tmp_file)
                .current_dir(&abs_user_dir)
                .output()
            {
                Ok(output) => {
                    std::fs::remove_file(&tmp_file).ok();
                    HttpResponse::Ok().json(serde_json::json!({
                        "success": output.status.success(),
                        "stdout": String::from_utf8_lossy(&output.stdout),
                        "stderr": String::from_utf8_lossy(&output.stderr),
                        "code": output.status.code()
                    }))
                }
                Err(e) => {
                    std::fs::remove_file(&tmp_file).ok();
                    HttpResponse::Ok().json(serde_json::json!({
                        "success": false,
                        "error": format!("Node.js not available: {}", e),
                        "hint": "JavaScript can run in-browser. Install Node.js for server-side execution."
                    }))
                }
            }
        }

        "shell" | "bash" | "sh" => {
            let tmp_file = abs_user_dir.join("._run_tmp.sh");
            std::fs::write(&tmp_file, &body.code).ok();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&tmp_file, std::fs::Permissions::from_mode(0o755)).ok();
            }
            match Command::new("bash")
                .arg(&tmp_file)
                .current_dir(&abs_user_dir)
                .output()
            {
                Ok(output) => {
                    std::fs::remove_file(&tmp_file).ok();
                    HttpResponse::Ok().json(serde_json::json!({
                        "success": output.status.success(),
                        "stdout": String::from_utf8_lossy(&output.stdout),
                        "stderr": String::from_utf8_lossy(&output.stderr),
                        "code": output.status.code()
                    }))
                }
                Err(e) => {
                    std::fs::remove_file(&tmp_file).ok();
                    HttpResponse::Ok().json(serde_json::json!({
                        "success": false,
                        "error": format!("bash not available: {}", e)
                    }))
                }
            }
        }

        "typescript" | "ts" => {
            let tmp_file = abs_user_dir.join("._run_tmp.ts");
            std::fs::write(&tmp_file, &body.code).ok();
            // Try ts-node first, fall back to node (strips types)
            let ts_node = Command::new("npx")
                .args(["--yes", "ts-node", "--transpile-only"])
                .arg(&tmp_file)
                .current_dir(&abs_user_dir)
                .output();
            let output = match ts_node {
                Ok(o) if o.status.success() || !o.stdout.is_empty() || !o.stderr.is_empty() => o,
                _ => {
                    match Command::new("node")
                        .arg(&tmp_file)
                        .current_dir(&abs_user_dir)
                        .output()
                    {
                        Ok(o) => o,
                        Err(e) => {
                            std::fs::remove_file(&tmp_file).ok();
                            return HttpResponse::Ok().json(serde_json::json!({
                                "success": false,
                                "error": format!("Node.js not available: {}", e),
                                "hint": "Install Node.js or ts-node to run TypeScript files."
                            }));
                        }
                    }
                }
            };
            std::fs::remove_file(&tmp_file).ok();
            HttpResponse::Ok().json(serde_json::json!({
                "success": output.status.success(),
                "stdout": String::from_utf8_lossy(&output.stdout),
                "stderr": String::from_utf8_lossy(&output.stderr),
                "code": output.status.code()
            }))
        }

        _ => HttpResponse::BadRequest().json(serde_json::json!({
            "success": false,
            "error": format!("Unsupported language: {}", body.language)
        }))
    }
}

/// List custom tools (stored as .tool.json files in user dir)
#[get("/api/tools/custom/{username}")]
async fn list_custom_tools(path: web::Path<String>) -> impl Responder {
    let username = path.into_inner();
    let user_dir = get_user_dir(&username);
    let tools_dir = format!("{}/tools", user_dir);
    std::fs::create_dir_all(&tools_dir).ok();

    let tools: Vec<serde_json::Value> = std::fs::read_dir(&tools_dir)
        .map(|entries| {
            entries.filter_map(|e| e.ok())
                .filter(|e| e.file_name().to_string_lossy().ends_with(".json"))
                .filter_map(|e| {
                    std::fs::read_to_string(e.path()).ok()
                        .and_then(|s| serde_json::from_str(&s).ok())
                })
                .collect()
        })
        .unwrap_or_default();

    HttpResponse::Ok().json(serde_json::json!({ "success": true, "tools": tools }))
}

/// Save a custom tool
#[post("/api/tools/custom")]
async fn save_custom_tool(body: web::Json<serde_json::Value>) -> impl Responder {
    let username = body.get("username").and_then(|u| u.as_str()).unwrap_or("guest");
    let name = body.get("name").and_then(|n| n.as_str()).unwrap_or("");
    if name.is_empty() || name.contains("..") || name.contains('/') {
        return HttpResponse::BadRequest().json(serde_json::json!({ "success": false, "error": "Invalid tool name" }));
    }

    let tools_dir = format!("{}/tools", get_user_dir(username));
    std::fs::create_dir_all(&tools_dir).ok();
    let path = format!("{}/{}.json", tools_dir, name);

    match std::fs::write(&path, serde_json::to_string_pretty(&body.0).unwrap_or_default()) {
        Ok(_) => HttpResponse::Ok().json(serde_json::json!({ "success": true })),
        Err(e) => HttpResponse::InternalServerError().json(serde_json::json!({ "success": false, "error": e.to_string() }))
    }
}

// ============================================================
// SHARE KEYS
// ============================================================

/// Create a share key — caller must provide their session token for auth
#[post("/api/share-keys")]
async fn create_share_key(data: web::Data<AppState>, body: web::Json<CreateShareKeyRequest>) -> impl Responder {
    let owner = {
        let sessions = data.sessions.lock().unwrap();
        match sessions.get(&body.token) {
            Some(s) => s.username.clone(),
            None => return HttpResponse::Unauthorized().json(serde_json::json!({"success":false,"error":"Unauthorized"})),
        }
    };
    if body.label.trim().is_empty() || body.api_url.trim().is_empty() || body.model.trim().is_empty() {
        return HttpResponse::BadRequest().json(serde_json::json!({"success":false,"error":"label, api_url and model are required"}));
    }
    let key_id = format!("sk_{}", uuid::Uuid::new_v4().to_string().replace('-', ""));
    let share_key = ShareKey {
        id: key_id.clone(),
        owner,
        label: body.label.trim().to_string(),
        api_url: body.api_url.trim().to_string(),
        model: body.model.trim().to_string(),
        api_key: body.api_key.clone(),
        active: true,
        uses: 0,
        created_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        max_uses: body.max_uses,
    };
    let mut keys = data.share_keys.lock().unwrap();
    keys.insert(key_id.clone(), share_key);
    save_share_keys(&keys);
    HttpResponse::Ok().json(serde_json::json!({"success":true,"key":key_id}))
}

/// List share keys owned by the authenticated user
#[get("/api/share-keys")]
async fn list_share_keys(data: web::Data<AppState>, req: actix_web::HttpRequest) -> impl Responder {
    let token = req.headers()
        .get("Authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .unwrap_or("")
        .to_string();
    let owner = {
        let sessions = data.sessions.lock().unwrap();
        match sessions.get(&token) {
            Some(s) => s.username.clone(),
            None => return HttpResponse::Unauthorized().json(serde_json::json!({"success":false,"error":"Unauthorized"})),
        }
    };
    let keys = data.share_keys.lock().unwrap();
    let mut user_keys: Vec<serde_json::Value> = keys.values()
        .filter(|k| k.owner == owner)
        .map(|k| serde_json::json!({
            "id": k.id,
            "label": k.label,
            "api_url": k.api_url,
            "model": k.model,
            "active": k.active,
            "uses": k.uses,
            "created_at": k.created_at,
            "max_uses": k.max_uses,
        }))
        .collect();
    user_keys.sort_by(|a, b| b["created_at"].as_u64().cmp(&a["created_at"].as_u64()));
    HttpResponse::Ok().json(serde_json::json!({"success":true,"keys":user_keys}))
}

/// Revoke (delete) a share key
#[delete("/api/share-keys/{key_id}")]
async fn revoke_share_key(data: web::Data<AppState>, path: web::Path<String>, req: actix_web::HttpRequest) -> impl Responder {
    let key_id = path.into_inner();
    let token = req.headers()
        .get("Authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .unwrap_or("")
        .to_string();
    let owner = {
        let sessions = data.sessions.lock().unwrap();
        match sessions.get(&token) {
            Some(s) => s.username.clone(),
            None => return HttpResponse::Unauthorized().json(serde_json::json!({"success":false,"error":"Unauthorized"})),
        }
    };
    let mut keys = data.share_keys.lock().unwrap();
    match keys.get(&key_id) {
        Some(k) if k.owner == owner => {
            keys.remove(&key_id);
            save_share_keys(&keys);
            HttpResponse::Ok().json(serde_json::json!({"success":true}))
        }
        Some(_) => HttpResponse::Forbidden().json(serde_json::json!({"success":false,"error":"Not your key"})),
        None => HttpResponse::NotFound().json(serde_json::json!({"success":false,"error":"Key not found"})),
    }
}

/// Validate a share key — returns public info without the API key (used by recipients)
#[get("/api/share-keys/validate/{key_id}")]
async fn validate_share_key(data: web::Data<AppState>, path: web::Path<String>) -> impl Responder {
    let key_id = path.into_inner();
    let keys = data.share_keys.lock().unwrap();
    match keys.get(&key_id) {
        Some(k) if k.active => HttpResponse::Ok().json(serde_json::json!({
            "success": true,
            "owner": k.owner,
            "label": k.label,
            "model": k.model,
            "api_url": k.api_url,
            "uses": k.uses,
        })),
        Some(_) => HttpResponse::Ok().json(serde_json::json!({"success":false,"error":"Key is inactive or revoked"})),
        None => HttpResponse::Ok().json(serde_json::json!({"success":false,"error":"Invalid share key"})),
    }
}

// ============================================================
// SETTINGS
// ============================================================

/// Get settings
#[get("/api/settings")]
async fn get_settings() -> impl Responder {
    let settings = current_settings();

    // Mask local API key
    let masked_key_local = if settings.llm_api_key_local.is_empty() {
        String::new()
    } else if settings.llm_api_key_local.len() <= 8 {
        "--------".to_string()
    } else {
        format!("{}--------{}", &settings.llm_api_key_local[..4], &settings.llm_api_key_local[settings.llm_api_key_local.len()-4..])
    };

    // Mask cloud API key
    let masked_key_cloud = if settings.llm_api_key_cloud.is_empty() {
        String::new()
    } else if settings.llm_api_key_cloud.len() <= 8 {
        "--------".to_string()
    } else {
        format!("{}--------{}", &settings.llm_api_key_cloud[..4], &settings.llm_api_key_cloud[settings.llm_api_key_cloud.len()-4..])
    };

    HttpResponse::Ok().json(serde_json::json!({
        "success": true,
        "settings": {
            "llm_api_url_local": settings.llm_api_url_local,
            "llm_model_local": settings.llm_model_local,
            "llm_api_key_local_masked": masked_key_local,
            "llm_api_key_local_set": !settings.llm_api_key_local.is_empty(),

            "llm_api_url_cloud": settings.llm_api_url_cloud,
            "llm_model_cloud": settings.llm_model_cloud,
            "llm_api_key_cloud_masked": masked_key_cloud,
            "llm_api_key_cloud_set": !settings.llm_api_key_cloud.is_empty(),

            "llm_active_source": settings.llm_active_source,

            "search_engine": settings.search_engine,
            "max_agent_iterations": settings.max_agent_iterations,
            "shell_enabled": settings.shell_enabled,

            "timeout_agent": settings.timeout_agent,
            "timeout_web_fetch": settings.timeout_web_fetch,
            "timeout_web_search": settings.timeout_web_search,
            "timeout_image": settings.timeout_image,
            "timeout_llm_test": settings.timeout_llm_test,

            "llm_temperature": settings.llm_temperature,
            "llm_max_tokens": settings.llm_max_tokens,
            "llm_think": settings.llm_think,

            "chat_context_limit": settings.chat_context_limit,
        }
    }))
}

/// Update settings
#[post("/api/settings")]
async fn update_settings(data: web::Data<AppState>, body: web::Json<serde_json::Value>) -> impl Responder {
    let mut settings = data.settings.lock().unwrap();

    // Local Ollama configuration
    if let Some(v) = body.get("llm_api_url_local").and_then(|v| v.as_str()) {
        if !v.is_empty() { settings.llm_api_url_local = v.to_string(); }
    }
    if let Some(v) = body.get("llm_model_local").and_then(|v| v.as_str()) {
        if !v.is_empty() { settings.llm_model_local = v.to_string(); }
    }
    if let Some(v) = body.get("llm_api_key_local").and_then(|v| v.as_str()) {
        if !v.contains("--") { settings.llm_api_key_local = v.to_string(); }
    }

    // Cloud Ollama configuration
    if let Some(v) = body.get("llm_api_url_cloud").and_then(|v| v.as_str()) {
        if !v.is_empty() { settings.llm_api_url_cloud = v.to_string(); }
    }
    if let Some(v) = body.get("llm_model_cloud").and_then(|v| v.as_str()) {
        if !v.is_empty() { settings.llm_model_cloud = v.to_string(); }
    }
    if let Some(v) = body.get("llm_api_key_cloud").and_then(|v| v.as_str()) {
        if !v.contains("--") { settings.llm_api_key_cloud = v.to_string(); }
    }

    // Active source selection
    if let Some(v) = body.get("llm_active_source").and_then(|v| v.as_str()) {
        if v == "local" || v == "cloud" {
            settings.llm_active_source = v.to_string();
        }
    }

    // Legacy fields (kept for backward compatibility)
    if let Some(v) = body.get("llm_api_url").and_then(|v| v.as_str()) {
        if !v.is_empty() { settings.llm_api_url = v.to_string(); }
    }
    if let Some(v) = body.get("llm_model").and_then(|v| v.as_str()) {
        if !v.is_empty() { settings.llm_model = v.to_string(); }
    }
    if let Some(v) = body.get("llm_api_key").and_then(|v| v.as_str()) {
        if !v.contains("--") { settings.llm_api_key = v.to_string(); }
    }
    if let Some(v) = body.get("llm_provider").and_then(|v| v.as_str()) {
        settings.llm_provider = v.to_string();
    }
    if let Some(v) = body.get("search_engine").and_then(|v| v.as_str()) {
        settings.search_engine = v.to_string();
    }
    if let Some(v) = body.get("max_agent_iterations").and_then(|v| v.as_u64()) {
        settings.max_agent_iterations = (v as usize).clamp(1, 500);
    }
    if let Some(v) = body.get("shell_enabled").and_then(|v| v.as_bool()) {
        settings.shell_enabled = v;
    }
    if let Some(v) = body.get("admin_password").and_then(|v| v.as_str()) {
        if !v.is_empty() && v.len() >= 6 {
            settings.admin_password = v.to_string();
            let mut users = data.users.lock().unwrap();
            if let Some(admin) = users.get_mut("admin") {
                admin.password_hash = hash_password(v);
            }
            save_users(&users);
        }
    }

    // Timeouts
    if let Some(v) = body.get("timeout_agent").and_then(|v| v.as_u64()) {
        settings.timeout_agent = v.clamp(10, 600);
    }
    if let Some(v) = body.get("timeout_web_fetch").and_then(|v| v.as_u64()) {
        settings.timeout_web_fetch = v.clamp(5, 120);
    }
    if let Some(v) = body.get("timeout_web_search").and_then(|v| v.as_u64()) {
        settings.timeout_web_search = v.clamp(5, 120);
    }
    if let Some(v) = body.get("timeout_image").and_then(|v| v.as_u64()) {
        settings.timeout_image = v.clamp(5, 120);
    }
    if let Some(v) = body.get("timeout_llm_test").and_then(|v| v.as_u64()) {
        settings.timeout_llm_test = v.clamp(10, 300);
    }

    // LLM generation
    if let Some(v) = body.get("llm_temperature").and_then(|v| v.as_f64()) {
        settings.llm_temperature = (v * 10.0).round() / 10.0; // round to 1 decimal
        settings.llm_temperature = settings.llm_temperature.clamp(0.0, 2.0);
    }
    if let Some(v) = body.get("llm_max_tokens").and_then(|v| v.as_u64()) {
        settings.llm_max_tokens = v.clamp(256, 32768);
    }
    if let Some(v) = body.get("llm_think").and_then(|v| v.as_bool()) {
        settings.llm_think = v;
    }

    // Chat context
    if let Some(v) = body.get("chat_context_limit").and_then(|v| v.as_u64()) {
        settings.chat_context_limit = (v as usize).clamp(1, 50);
    }

    save_settings(&settings);

    HttpResponse::Ok().json(serde_json::json!({ "success": true, "message": "Settings saved" }))
}

/// Test LLM connection
#[post("/api/settings/test_llm")]
async fn test_llm() -> impl Responder {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(current_settings().timeout_llm_test))
        .build()
        .unwrap_or_default();

    let response = call_llm(&client, "Say 'OK' in one word.", None).await;
    let is_error = response.starts_with("Error");

    HttpResponse::Ok().json(serde_json::json!({
        "success": !is_error,
        "response": safe_truncate(&response, 500),
    }))
}

#[get("/api/health")]
async fn health() -> impl Responder {
    let settings = current_settings();
    HttpResponse::Ok().json(serde_json::json!({
        "status": "ok",
        "version": "1.1.0",
        "llm_url": settings.llm_api_url,
        "llm_model": settings.llm_model,
        "llm_provider": settings.llm_provider,
        "features": ["agent", "native_tool_calling", "web_search", "web_fetch", "shell", "memory", "files", "code_execution"],
        "search_engine": settings.search_engine
    }))
}

// ============================================================
// MAIN
// ============================================================

// ============================================================
// CRON RUNNER
// ============================================================

/// Returns true if the 5-field cron expression fires at the current UTC minute.
fn matches_schedule(schedule_str: &str) -> bool {
    use std::str::FromStr;
    use chrono::Timelike;
    // cron crate uses 6 fields (sec min hour dom month dow); prepend "0 "
    let expr = format!("0 {}", schedule_str);
    match cron::Schedule::from_str(&expr) {
        Err(e) => { eprintln!("[CRON] Invalid expression '{}': {}", schedule_str, e); false }
        Ok(schedule) => {
            let now: chrono::DateTime<chrono::Utc> = chrono::Utc::now();
            // Truncate to current minute
            let current_minute = match now.with_second(0).and_then(|t| t.with_nanosecond(0)) {
                Some(t) => t,
                None => return false,
            };
            // One second before current minute → next occurrence should be current_minute
            let just_before = current_minute - chrono::Duration::seconds(1);
            match schedule.after(&just_before).next() {
                Some(next) => next == current_minute,
                None => false,
            }
        }
    }
}

/// Spawn a background task that runs cron jobs every minute.
fn start_cron_runner() {
    tokio::spawn(async move {
        loop {
            // Sleep until start of next minute
            let now: chrono::DateTime<chrono::Utc> = chrono::Utc::now();
            use chrono::Timelike;
            let secs_to_next = 60u64.saturating_sub(now.second() as u64);
            tokio::time::sleep(std::time::Duration::from_secs(secs_to_next.max(1))).await;

            let now_ts = chrono::Utc::now().timestamp() as u64;

            // Walk data/cron/<username>/jobs.json for all users
            let cron_base = std::path::Path::new("data/cron");
            if !cron_base.exists() { continue; }
            let entries = match std::fs::read_dir(cron_base) {
                Ok(e) => e,
                Err(_) => continue,
            };

            for user_entry in entries.flatten() {
                let username = user_entry.file_name().to_string_lossy().to_string();
                let jobs_path = format!("data/cron/{}/jobs.json", username);
                let mut jobs: Vec<serde_json::Value> = std::fs::read_to_string(&jobs_path)
                    .ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
                if jobs.is_empty() { continue; }

                let user_dir = get_user_dir(&username);
                std::fs::create_dir_all(&user_dir).ok();

                let mut changed = false;
                for job in jobs.iter_mut() {
                    let enabled = job.get("enabled").and_then(|e| e.as_bool()).unwrap_or(true);
                    if !enabled { continue; }

                    // Avoid double-firing within the same minute
                    if let Some(last) = job.get("last_run").and_then(|l| l.as_u64()) {
                        if now_ts.saturating_sub(last) < 55 { continue; }
                    }

                    let schedule_str = job.get("schedule").and_then(|s| s.as_str()).unwrap_or("").to_string();
                    if !matches_schedule(&schedule_str) { continue; }

                    let command = job.get("command").and_then(|c| c.as_str()).unwrap_or("").to_string();
                    let job_name = job.get("name").and_then(|n| n.as_str()).unwrap_or("?").to_string();
                    let dir = user_dir.clone();

                    eprintln!("[CRON] Firing '{}' for user {} | cmd: {}", job_name, username, command);

                    // Run in a blocking thread so we don't block the async runtime
                    let output = tokio::task::spawn_blocking(move || {
                        std::process::Command::new("sh")
                            .arg("-c")
                            .arg(&command)
                            .current_dir(&dir)
                            .output()
                    }).await;

                    match output {
                        Ok(Ok(out)) => {
                            let stdout = String::from_utf8_lossy(&out.stdout).to_string();
                            let stderr = String::from_utf8_lossy(&out.stderr).to_string();
                            if !stdout.is_empty() { eprintln!("[CRON] stdout: {}", stdout.trim()); }
                            if !stderr.is_empty() { eprintln!("[CRON] stderr: {}", stderr.trim()); }
                        }
                        Err(e) => eprintln!("[CRON] spawn error: {}", e),
                        Ok(Err(e)) => eprintln!("[CRON] exec error: {}", e),
                    }

                    job["last_run"] = serde_json::json!(now_ts);
                    changed = true;
                }

                if changed {
                    std::fs::write(&jobs_path, serde_json::to_string_pretty(&jobs).unwrap_or_default()).ok();
                }
            }
        }
    });
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    dotenv::dotenv().ok();
    std::fs::create_dir_all("data/users").ok();

    let settings = load_settings();
    save_settings(&settings);

    let app_state = web::Data::new(AppState {
        users: Mutex::new(HashMap::new()),
        settings: Mutex::new(settings.clone()),
        sessions: Mutex::new(HashMap::new()),
        share_keys: Mutex::new(load_share_keys()),
    });

    // Load persisted users, then ensure admin is up-to-date
    {
        let mut persisted = load_users();
        // Always sync admin password from settings (source of truth)
        persisted.insert("admin".to_string(), User {
            id: persisted.get("admin").map(|u| u.id.clone()).unwrap_or_else(|| Uuid::new_v4().to_string()),
            username: "admin".to_string(),
            password_hash: hash_password(&settings.admin_password),
        });
        save_users(&persisted);
        let user_count = persisted.len();
        *app_state.users.lock().unwrap() = persisted;
        println!("Users: {} loaded from disk", user_count);
    }

    std::fs::create_dir_all("data/users/guest").ok();

    // Start cron job runner (checks every minute)
    start_cron_runner();

    println!("Puterra Cloud OS v1.1.0");
    println!("LLM: {} [{}] (model: {})", settings.llm_provider, settings.llm_api_url, settings.llm_model);
    println!("Search: {} (integrated)", settings.search_engine);
    println!("Agent: Native tool calling, max {} iterations", settings.max_agent_iterations);
    println!("Shell: {}", if settings.shell_enabled { "enabled" } else { "disabled" });
    println!("Cron: background runner active");
    println!("http://0.0.0.0:3707");

    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .service(index)
            .service(signup)
            .service(login)
            .service(logout)
            .service(validate_token)
            .service(list_files)
            .service(create_item)
            .service(delete_items)
            .service(rename_item)
            .service(read_file)
            .service(write_file)
            .service(download_file)
            .service(view_file)
            .service(upload_file)
            .service(web_search)
            .service(web_fetch)
            .service(shell_exec)
            .service(memory)
            .service(chat)
            .service(agent_chat)
            .service(run_code)
            .service(list_custom_tools)
            .service(save_custom_tool)
            .service(get_settings)
            .service(update_settings)
            .service(test_llm)
            .service(tools_list)
            .service(health)
            .service(validate_share_key)
            .service(create_share_key)
            .service(list_share_keys)
            .service(revoke_share_key)
            .service(fs::Files::new("/static", "public"))
            .default_service(web::route().to(|| async {
                fs::NamedFile::open("public/index.html")
            }))
    })
    .bind("0.0.0.0:3707")?
    .run()
    .await
}
