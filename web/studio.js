"use strict";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const assets = "/assets/";
const descriptions = {
  bicubic: "Classical interpolation and reproducible baseline.",
  srcnn: "Three convolution layers refine a bicubic image.",
  srresnet: "Residual blocks and sub-pixel upsampling.",
  srgan: "Adversarial fine-tuning of the SRResNet generator.",
  esrgan: "Pretrained restoration for real-world images.",
};
const pageText = {
  image: ["Super-resolution studio", "Restore detail in images and short videos with classical and learned methods.", "Upscale image"],
  video: ["Super-resolution studio", "Upscale individual video frames and export a playable H.264 MP4.", "Upscale video"],
  experiment: ["Reconstruction experiment", "Downsample a known original and evaluate the reconstructed image against ground truth.", "Run experiment"],
};
const labels = {
  input: "Low-resolution input", baseline: "Bicubic baseline", reference: "Ground truth",
  jpeg_deblock: "JPEG deblocking", denoise: "Spatial smoothing", deblur: "Deblurring",
  enhance: "Intensity transformation", frequency_filter: "Frequency filtering",
};
const form = $("#settings-form");
const fileInput = $("#file");
const video = $("#video-player");
const demo = {output: assets + "demo-after.png", input: assets + "demo-input.png", input_size: [240, 150], output_size: [960, 600]};
// Keep separate settings and results for image, video, and experiment tabs.
const sessions = Object.fromEntries(["image", "video", "experiment"].map((name) => [
  name, {
    file: null, url: null, result: name === "image" ? demo : null, message: "", status: "",
    view: name === "experiment" ? "reference" : "input", videoView: "result", diagnostic: "edges",
    settings: {
      model: name === "video" ? "bicubic" : "esrgan", scale: name === "video" ? 2 : 4,
      denoise: "off", intensity: "off", sharpening: "off", frequency: "off",
      cutoff: 0.12, input_quality: 30, simulate: false, crf: 20,
      deblock: false, refinement: false,
    },
  },
]));
let inventory = [
  {id: "bicubic", label: "Bicubic", scales: [2, 4]},
  {id: "esrgan", label: "Real-ESRGAN", scales: [2, 4]},
];
let mode = "image";
let studioMode = "image";
let pending = false;

function press(selector, match) {
  $$(selector).forEach((button) => button.setAttribute("aria-pressed", String(match(button))));
}

function setStatus(state, message, kind = "") {
  state.message = message;
  state.status = kind;
  if (state === sessions[mode]) {
    const status = $("#status");
    status.textContent = message;
    status.className = "status " + kind;
  }
}

function fieldOptions() {
  const settings = sessions[mode].settings;
  // Filter the model picker by the scale each checkpoint supports.
  const compatible = inventory.filter((entry) => entry.scales.includes(settings.scale));
  if (!compatible.some((entry) => entry.id === settings.model)) settings.model = "bicubic";
  $("#model").replaceChildren(...compatible.map((entry) => new Option(entry.label, entry.id)));
  for (const [name, value] of Object.entries(settings)) {
    const field = form.elements.namedItem(name);
    if (!field) continue;
    if (field.type === "checkbox") field.checked = value;
    else field.value = value;
  }
  $("#model-note").textContent = descriptions[settings.model] || "";
  press("[data-scale]", (button) => Number(button.dataset.scale) === settings.scale);
  $("#experiment-options").hidden = mode !== "experiment";
  $("#video-options").hidden = mode !== "video";
  $("#cutoff-control").hidden = settings.frequency === "off";
  $("#input-quality-control").hidden = !settings.simulate;
  $("#cutoff-value").textContent = Number(settings.cutoff).toFixed(2);
  $("#input-quality-value").textContent = settings.input_quality;
  $("#crf-value").textContent = settings.crf;
}

function syncUpload() {
  const {file} = sessions[mode];
  const isVideo = mode === "video";
  const title = mode === "experiment" ? "Choose a reference image" : isVideo ? "Choose a video" : "Choose an image";
  $("#upload-guide strong").textContent = file?.name || title;
  $("#upload-guide small").textContent = file
    ? (file.size / 1024 ** 2).toFixed(2) + " MiB · click to replace"
    : isVideo ? "MP4, MOV, AVI, MKV or WebM" : "PNG, JPEG, WebP or BMP";
  $("#sample").textContent = isVideo ? "Use sample clip" : "Use sample image";
  fileInput.accept = isVideo ? ".mp4,.mov,.avi,.mkv,.webm" : ".png,.jpg,.jpeg,.webp,.bmp";
}

function syncExport() {
  const {result} = sessions[mode];
  $("#export").hidden = !result;
  $("#export-format").hidden = mode === "video";
  const jpeg = mode !== "video" && $("#export-format").value === "jpeg";
  $("#jpeg-control").hidden = !result || !jpeg;
  $("#download").textContent = mode === "video" ? "Download MP4" : jpeg ? "Download JPEG" : "Download PNG";
  $("#download").disabled = pending;
}

function measure(value, places, suffix = "") {
  return value == null ? "—" : typeof value === "number" ? value.toFixed(places) + suffix : String(value);
}

function syncMetrics() {
  const metrics = sessions[mode].result?.metrics;
  $("#scores").hidden = !metrics;
  if (!metrics) return;
  for (const [id, key, places, suffix] of [["psnr", "PSNR (dB)", 2, " dB"], ["ssim", "SSIM", 4, ""]]) {
    $("#" + id + "-value").textContent = measure(metrics.processed[key], places, suffix);
    $("#" + id + "-baseline").textContent = "Bicubic " + measure(metrics.bicubic[key], places, suffix);
  }
}

function dimensions(values) {
  return Array.isArray(values) ? values.join(" × ") : String(values || "").replaceAll("×", " × ");
}

function syncViewer() {
  const state = sessions[mode];
  const {file, result, url} = state;
  const isVideo = mode === "video";
  const compare = !isVideo && Boolean(result);
  $("#compare").hidden = !compare;
  $("#still-preview").hidden = compare || isVideo;
  $("#video-preview").hidden = !isVideo;
  $("#zoom").hidden = !compare;
  $("#video-views").hidden = !isVideo || !result;
  press("[data-video-view]", (button) => button.dataset.videoView === state.videoView);
  $("#viewer-title").textContent = isVideo ? "Video preview" : compare ? "Image comparison"
    : mode === "experiment" ? "Reference preview" : "Input preview";
  $("#viewer-note").textContent = isVideo
    ? (result ? "Play the original and enhanced clips to compare them." : "Review the source clip before upscaling.")
    : compare ? "Drag to compare the two views." : "Run processing to compare the reconstruction.";
  $("#sample-credit span:last-child").textContent = file?.name || (isVideo ? "DIV2K sample clip · 0807" : "DIV2K sample · 0807");
  $("#image-dimensions").textContent = result
    ? dimensions(result.input_size) + " to " + dimensions(result.output_size) + (isVideo ? " · " + result.fps + " fps" : "")
    : "";
  if (isVideo) {
    // Use a browser object URL for uploaded and processed clips, or serve the built-in sample.
    const source = (state.videoView === "result" && result?.url) || url || assets + "demo-clip.mp4";
    if (video.getAttribute("src") !== source) video.src = source;
  } else {
    video.pause();
    if (!compare) {
      $("#uploaded-preview").src = url || assets + (mode === "experiment" ? "demo-reference.png" : "demo-input.png");
      $("#uploaded-preview").alt = mode === "experiment" ? "High-resolution reference" : "Selected image";
    } else {
      const selection = ["input", "baseline", "reference"].find((key) => key === state.view && result[key]) || "input";
      $("#before").src = result[selection];
      $("#before").alt = labels[selection];
      $("#before-label").textContent = labels[selection];
      $("#after").src = result.output;
      $("#reference-view").replaceChildren(
        ...["input", "baseline", "reference"].filter((key) => result[key]).map((key) => new Option(labels[key], key)),
      );
      $("#reference-view").value = selection;
    }
  }
  $("#reference-control").hidden = !compare || !result?.baseline;
  $("#analysis").hidden = !result?.diagnostics;
  syncMetrics();
  syncDiagnostic();
  syncExport();
}

function syncWorkspace() {
  const [title, description, action] = pageText[mode];
  $("#page-title").textContent = title;
  $("#page-description").textContent = description;
  $("#run").textContent = action;
  $("#run").disabled = pending;
  fileInput.disabled = pending;
  $("#sample").disabled = pending;
  $("#mode-tabs").hidden = mode === "experiment";
  press("[data-mode]", (button) => button.dataset.mode === mode);
  fieldOptions();
  syncUpload();
  syncViewer();
  setStatus(sessions[mode], sessions[mode].message, sessions[mode].status);
}

function route() {
  const next = location.hash.slice(1);
  const page = ["experiment", "models"].includes(next) ? next : "studio";
  $$("nav a").forEach((link) => {
    if (link.dataset.page === page) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  $("#workspace").hidden = page === "models";
  $("#models-page").hidden = page !== "models";
  if (page === "models") {
    video.pause();
    loadInventory();
  } else {
    mode = page === "experiment" ? "experiment" : studioMode;
    fileInput.value = "";
    syncWorkspace();
  }
}

function clearResult(state) {
  // Release the previous video blob when its result is replaced.
  if (state.result?.url) URL.revokeObjectURL(state.result.url);
  state.result = null;
}

function setFile(file) {
  if (pending) return;
  const max = mode === "video" ? 50 : 32;
  const state = sessions[mode];
  if (file && (file.size === 0 || file.size > max * 1024 ** 2)) {
    setStatus(state, "Choose a nonempty file under " + max + " MiB.", "error");
    return;
  }
  if (state.url) URL.revokeObjectURL(state.url);
  clearResult(state);
  state.file = file || null;
  state.url = file ? URL.createObjectURL(file) : null;
  const isInitialDemo = !file && mode === "image" && state.settings.model === "esrgan"
    && state.settings.scale === 4 && ["denoise", "intensity", "sharpening", "frequency"].every((key) => state.settings[key] === "off")
    && !state.settings.deblock && !state.settings.refinement;
  if (isInitialDemo) state.result = demo;
  setStatus(state, "");
  syncUpload();
  syncViewer();
}

function markSettingsChanged() {
  const state = sessions[mode];
  if (state.result) setStatus(state, "Run again to apply the new settings.");
}

for (const button of $$("[data-mode]")) {
  button.addEventListener("click", () => {
    mode = studioMode = button.dataset.mode;
    fileInput.value = "";
    syncWorkspace();
  });
}
for (const button of $$("[data-scale]")) {
  button.addEventListener("click", () => {
    sessions[mode].settings.scale = Number(button.dataset.scale);
    markSettingsChanged();
    fieldOptions();
  });
}
form.addEventListener("change", (event) => {
  const {name, value, type, checked} = event.target;
  const settings = sessions[mode].settings;
  if (!Object.hasOwn(settings, name)) return;
  settings[name] = type === "checkbox" ? checked : type === "range" ? Number(value) : value;
  markSettingsChanged();
  fieldOptions();
});
form.addEventListener("input", (event) => {
  const {name, value, type} = event.target;
  if (type !== "range" || !Object.hasOwn(sessions[mode].settings, name)) return;
  const display = {cutoff: "#cutoff-value", input_quality: "#input-quality-value", crf: "#crf-value"}[name];
  $(display).textContent = name === "cutoff" ? Number(value).toFixed(2) : value;
});

fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
$("#sample").addEventListener("click", () => {
  fileInput.value = "";
  setFile(null);
});
const drop = $("#dropzone");
for (const event of ["dragenter", "dragover"]) {
  drop.addEventListener(event, (e) => { e.preventDefault(); drop.classList.add("dragging"); });
}
for (const event of ["dragleave", "drop"]) {
  drop.addEventListener(event, (e) => { e.preventDefault(); drop.classList.remove("dragging"); });
}
drop.addEventListener("drop", (event) => {
  fileInput.value = "";
  setFile(event.dataTransfer.files[0]);
});
// The slider only changes a CSS variable. It never reruns image inference.
$("#reveal").addEventListener("input", (e) => $("#compare").style.setProperty("--split", e.target.value + "%"));
$("#zoom").addEventListener("click", () => {
  const zoom = $("#compare").classList.toggle("zoom");
  $("#zoom").textContent = zoom ? "Reset zoom" : "Zoom 2×";
  $("#zoom").setAttribute("aria-pressed", String(zoom));
});
for (const button of $$("[data-video-view]")) {
  button.addEventListener("click", () => {
    sessions.video.videoView = button.dataset.videoView;
    syncViewer();
  });
}
video.addEventListener("error", () => {
  if (mode === "video" && !sessions.video.result && sessions.video.file) {
    $("#viewer-note").textContent = "The source codec cannot be previewed here. Upscale to create a playable MP4.";
  }
});
$("#reference-view").addEventListener("change", (event) => {
  sessions[mode].view = event.target.value;
  syncViewer();
});

async function assertResponse(response) {
  if (response.ok) return;
  let detail;
  try { detail = (await response.json()).detail; } catch { detail = null; }
  throw new Error(typeof detail === "string" ? detail : "The server could not complete the request.");
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (pending) return;
  const requestMode = mode;
  const state = sessions[requestMode];
  // Snapshot the settings used for this run so later edits do not relabel the result.
  const requestSettings = {...state.settings};
  const {simulate, input_quality, ...settings} = requestSettings;
  const body = new FormData();
  // Send settings as a JSON string inside multipart form data, alongside the optional media file.
  body.set("settings", JSON.stringify({
    ...settings, experiment: requestMode === "experiment",
    input_quality: requestMode === "experiment" && simulate ? input_quality : null,
  }));
  if (state.file) body.set("upload", state.file);
  pending = true;
  $("#run").disabled = true;
  const applyRestoration = $("#apply-restoration");
  applyRestoration.disabled = true;
  applyRestoration.textContent = "Processing…";
  fileInput.disabled = true;
  $("#sample").disabled = true;
  syncExport();
  setStatus(state, requestMode === "video" ? "Processing frames and encoding the clip…" : "Processing image…", "busy");
  try {
    // Both media modes use the same settings and dispatch to different FastAPI routes.
    const response = await fetch("/api/" + (requestMode === "video" ? "video" : "image"), {method: "POST", body});
    await assertResponse(response);
    let result = null;
    if (requestMode === "video") {
      // MP4 is binary. The API sends frame count and dimensions in response headers.
      const blob = await response.blob();
      result = {
        url: URL.createObjectURL(blob), size: blob.size, fps: response.headers.get("X-FPS"),
        frames: response.headers.get("X-Frame-Count"),
        input_size: response.headers.get("X-Input-Size"), output_size: response.headers.get("X-Output-Size"),
      };
    } else {
      // Images and diagnostic previews are PNG data URLs in the JSON response.
      result = await response.json();
    }
    clearResult(state);
    state.result = result;
    state.videoView = "result";
    state.view = requestMode === "experiment" ? "reference" : "input";
    const runLabel = requestMode === "video" ? result.frames + " frames processed · " + (result.size / 1024).toFixed(1) + " KiB"
      : "Processed in " + Number(result.seconds).toFixed(2) + " s";
    // Keep the result attached to the settings that actually produced it.
    const current = Object.keys(requestSettings).every((key) => requestSettings[key] === state.settings[key]);
    setStatus(state, runLabel + (current ? "" : " · run again to apply new settings"));
  } catch (error) {
    setStatus(state, error.message, "error");
  } finally {
    pending = false;
    $("#run").disabled = false;
    applyRestoration.disabled = false;
    applyRestoration.textContent = "Apply sharpening";
    fileInput.disabled = false;
    $("#sample").disabled = false;
    if (mode === requestMode) syncViewer();
    else syncExport();
  }
});

function downloadUrl(url, filename) {
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
}
$("#export-format").addEventListener("change", syncExport);
$("#jpeg-quality").addEventListener("input", (event) => { $("#jpeg-value").textContent = event.target.value; });
$("#download").addEventListener("click", async () => {
  const state = sessions[mode];
  const result = state.result;
  if (!result) return;
  if (mode === "video") {
    downloadUrl(result.url, "enhanced-video.mp4");
    return;
  }
  if ($("#export-format").value === "png") {
    downloadUrl(result.output, "enhanced-image.png");
    return;
  }
  $("#download").disabled = true;
  setStatus(state, "Encoding JPEG…", "busy");
  try {
    const body = new FormData();
    // JPEG export re-encodes the existing output and does not run the network again.
    body.set("upload", await (await fetch(result.output)).blob(), "image.png");
    body.set("quality", $("#jpeg-quality").value);
    const response = await fetch("/api/jpeg", {method: "POST", body});
    await assertResponse(response);
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    downloadUrl(url, "enhanced-image.jpg");
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    const psnr = response.headers.get("X-PSNR");
    setStatus(state, (blob.size / 1024).toFixed(1) + " KiB · " + (psnr === "inf" ? "∞" : psnr) + " dB vs PNG");
  } catch (error) {
    setStatus(state, error.message, "error");
  } finally {
    $("#download").disabled = pending;
  }
});

function drawHistogram(series) {
  const canvas = $("#histogram");
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const left = 40;
  const right = width - 12;
  const base = height - 30;
  // Use one vertical scale for the two luma histograms so their counts are comparable.
  const maximum = Math.max(1, ...series.flat());
  ctx.clearRect(0, 0, width, height);
  ctx.font = "13px system-ui";
  ctx.strokeStyle = "#dce4df";
  ctx.fillStyle = "#4f625a";
  for (let step = 0; step <= 4; step++) {
    const y = 10 + step * (base - 10) / 4;
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(right, y);
    ctx.stroke();
    ctx.fillText(String(Math.round(maximum * (1 - step / 4))), 1, y + 4);
  }
  for (const value of [0, 64, 128, 192, 255]) {
    ctx.fillText(String(value), left + (right - left) * value / 255 - 10, height - 9);
  }
  for (const [i, points] of series.entries()) {
    ctx.beginPath();
    ctx.strokeStyle = ["#8b9b96", "#146455"][i];
    ctx.lineWidth = 2;
    points.forEach((value, x) => {
      const coord = [left + (right - left) * x / 255, base - (base - 10) * value / maximum];
      if (x === 0) ctx.moveTo(...coord);
      else ctx.lineTo(...coord);
    });
    ctx.stroke();
  }
}

const diagnosticNotes = {
  edges: "Compare the same center crop. Sobel shows gradient strength. Canny marks detected edges.",
  channels: "These channels show the processed center crop. Y carries brightness, while Cb and Cr carry color differences.",
  histogram: "Compare counts for each luma level on the same center crop, from dark on the left to bright on the right.",
  spectrum: "The center shows low spatial frequencies. Points farther out represent fine detail, edges, or noise. Each view has its own brightness scale.",
  stages: "See the intermediate images from the restoration filters used for this result.",
};
const stageLabels = {
  ...labels,
  deblur: "Sharpening or deblurring",
  before_refinement: "Upscaled before refinement",
  refinement: "After refinement",
};

function syncDiagnostic() {
  const state = sessions[mode];
  if (!state.result?.diagnostics) return;
  const {diagnostics: d, stages, input, output} = state.result;
  const panel = state.diagnostic;
  const restorationStages = Object.entries(stages);
  const showRestorationAction = panel === "stages" && restorationStages.length === 0;
  press("[data-diagnostic]", (button) => button.dataset.diagnostic === panel);
  $("#edge-control").hidden = panel !== "edges";
  $("#histogram-panel").hidden = panel !== "histogram";
  $("#diagnostic-note").textContent = showRestorationAction
    ? "Compare the input with the upscaled output. Apply sharpening to preview a restoration step."
    : diagnosticNotes[panel];
  $("#restoration-actions").hidden = !showRestorationAction;
  const container = $("#diagnostic-images");
  container.replaceChildren();
  if (panel === "histogram") {
    drawHistogram(d.histogram);
    return;
  }
  const method = $("#edge-method").value;
  const figures = {
    edges: d.edges[method].map((url, i) => [[ "Bicubic · " + method, "Processed · " + method][i], url]),
    channels: d.channels.map((url, i) => [["Y · Luminance", "Cb · Blue chroma", "Cr · Red chroma"][i], url]),
    spectrum: d.spectrum.map((url, i) => [["Bicubic · DFT", "Processed · DFT"][i], url]),
    stages: restorationStages.map(([name, url]) => [stageLabels[name] || name, url]),
  }[panel];
  if (panel === "stages") {
    // Use the real input and output previews when a run had no intermediate filters.
    if (showRestorationAction) figures.push([labels.input, input]);
    if (!stages.refinement) figures.push(["Upscaled output", output]);
  }
  for (const [label, url] of figures) {
    const figure = document.createElement("figure");
    const image = new Image();
    image.alt = label;
    image.src = url;
    const caption = document.createElement("figcaption");
    caption.textContent = label;
    figure.append(image, caption);
    container.append(figure);
  }
}

for (const button of $$("[data-diagnostic]")) {
  button.addEventListener("click", () => {
    sessions[mode].diagnostic = button.dataset.diagnostic;
    syncDiagnostic();
  });
}
$("#edge-method").addEventListener("change", syncDiagnostic);
$("#apply-restoration").addEventListener("click", () => {
  if (pending) return;
  sessions[mode].settings.sharpening = "unsharp";
  fieldOptions();
  form.requestSubmit();
});
$("#open-restoration").addEventListener("click", () => {
  $("#restoration").open = true;
  form.elements.namedItem("denoise").focus();
});

function addRow(body, cells) {
  const row = body.insertRow();
  for (const text of cells) row.insertCell().textContent = text;
}

async function loadInventory() {
  try {
    // Read actual checkpoint availability, training provenance, and measured benchmark rows.
    const response = await fetch("/api/models", {cache: "no-store"});
    await assertResponse(response);
    const status = await response.json();
    inventory = status.models;
    $("#device").textContent = status.device;
    $("#device").hidden = false;
    const models = $("#model-rows");
    models.replaceChildren();
    for (const entry of inventory) {
      const training = entry.epochs + (entry.epochs === 1 ? " epoch" : " epochs")
        + (entry.training_images ? " · " + entry.training_images + " DIV2K training images" : "");
      const label = entry.id === "bicubic" ? "Interpolation" : entry.id === "esrgan" ? "Bundled pretrained weights"
        : entry.epochs ? training
        : entry.scales.length ? "Local checkpoint" : "Training required";
      addRow(models, [entry.label, entry.scales.map((scale) => scale + "×").join(", ") || "—", label]);
    }
    const benchmark = $("#benchmark-rows");
    benchmark.replaceChildren();
    $("#benchmark-table").hidden = !status.benchmark.length;
    if (status.benchmark.length) {
      const first = status.benchmark[0];
      $("#benchmark-note").textContent = first.images + " held-out DIV2K crops · " + first.scale + "× · aligned RGB metrics. Higher PSNR and SSIM indicate closer agreement.";
      for (const row of status.benchmark) {
        const name = inventory.find((entry) => entry.id === row.model)?.label || row.model;
        addRow(benchmark, [name, measure(row.psnr_hr, 2), measure(row.ssim_hr, 4), measure(row.latency_ms, 1, " ms")]);
      }
    } else {
      $("#benchmark-note").textContent = "Benchmark results will appear after a DIV2K evaluation.";
    }
    fieldOptions();
  } catch (error) {
    $("#benchmark-note").textContent = "Model inventory could not be loaded. Refresh after starting the server.";
  }
}

window.addEventListener("hashchange", route);
route();
loadInventory();
