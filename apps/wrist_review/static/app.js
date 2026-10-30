const appState = {
  views: [],
  selectedIndex: 0,
  result: null,
  toastTimer: null,
};

const MAX_STUDY_VIEWS = 8;

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const filePicker = $("#filePicker");
const folderPicker = $("#folderPicker");
const workspace = $("#workspace");
const uploadState = $("#uploadState");
const reviewResult = $("#reviewResult");

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function filenameStem(name) {
  return name.replace(/\.[^.]+$/, "");
}

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("show");
  window.clearTimeout(appState.toastTimer);
  appState.toastTimer = window.setTimeout(() => toast.classList.remove("show"), 3600);
}

function updateProgress(step) {
  const items = $$(".progress-step");
  items.forEach((item, index) => item.classList.toggle("current", index <= step));
}

async function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error(`Could not read ${file.name}.`));
    reader.readAsDataURL(file);
  });
}

async function imageDimensions(blobUrl) {
  return new Promise((resolve) => {
    const image = new Image();
    image.onload = () => resolve({ width: image.naturalWidth, height: image.naturalHeight, previewable: true });
    image.onerror = () => resolve({ width: null, height: null, previewable: false });
    image.src = blobUrl;
  });
}

async function normaliseFiles(fileList) {
  const files = [...fileList].filter((file) => file.size > 0);
  if (!files.length) throw new Error("No readable image files were selected.");
  if (files.length > MAX_STUDY_VIEWS) {
    throw new Error(
      `This selection contains ${files.length} files. Select 1–${MAX_STUDY_VIEWS} views from one study and try again.`,
    );
  }
  const names = new Set();
  const views = [];
  for (const file of files) {
    if (names.has(file.name)) throw new Error("Each selected image must have a unique filename.");
    names.add(file.name);
    const blobUrl = URL.createObjectURL(file);
    const dimensions = await imageDimensions(blobUrl);
    views.push({
      file,
      name: file.name,
      relativePath: file.webkitRelativePath || "",
      blobUrl,
      type: file.type || "Image file",
      size: file.size,
      ...dimensions,
    });
  }
  return views;
}

async function selectFiles(fileList, sourceInput = null) {
  try {
    const views = await normaliseFiles(fileList);
    appState.views.forEach((view) => URL.revokeObjectURL(view.blobUrl));
    appState.views = views;
    appState.selectedIndex = 0;
    appState.result = null;
    reviewResult.classList.add("hidden");
    const firstPath = views[0]?.relativePath || "";
    const folder = firstPath.includes("/") ? firstPath.split("/")[0] : "Uploaded study";
    $("#folderName").textContent = folder;
    renderWorkspace();
    uploadState.classList.add("hidden");
    workspace.classList.remove("hidden");
    updateProgress(1);
  } catch (error) {
    // Reset the picker after a rejected selection so the same files can be
    // chosen again after the user corrects the selection.
    if (sourceInput) sourceInput.value = "";
    showToast(error instanceof Error ? error.message : "Could not add those files.");
  }
}

function renderWorkspace() {
  if (!appState.views.length) {
    workspace.classList.add("hidden");
    uploadState.classList.remove("hidden");
    updateProgress(0);
    return;
  }
  const selected = appState.views[appState.selectedIndex] || appState.views[0];
  appState.selectedIndex = appState.views.indexOf(selected);
  $("#viewCount").textContent = appState.views.length;
  $("#fileRail").innerHTML = appState.views.map((view, index) => `
    <button type="button" class="file-item ${index === appState.selectedIndex ? "selected" : ""}" data-index="${index}">
      <img class="file-item-thumb" src="${view.blobUrl}" alt="" />
      <span class="file-item-text"><span>${escapeHtml(filenameStem(view.name))}</span><small>View ${index + 1}</small></span>
    </button>
  `).join("");
  $$(".file-item").forEach((button) => {
    button.addEventListener("click", () => {
      appState.selectedIndex = Number(button.dataset.index);
      renderWorkspace();
    });
  });

  const preview = $("#imagePreview");
  preview.src = selected.blobUrl;
  preview.alt = `Preview of ${selected.name}`;
  $("#imageIndex").textContent = `View ${appState.selectedIndex + 1} of ${appState.views.length}`;
  $("#selectedFileName").textContent = selected.name;
  $("#selectedPath").textContent = selected.relativePath || "Local upload · not stored";
  $("#detailFormat").textContent = (selected.type.split("/").pop() || "Image").toUpperCase();
  $("#detailSize").textContent = formatBytes(selected.size);
  $("#detailDimensions").textContent = selected.previewable ? `${selected.width} × ${selected.height}` : "Preview unavailable";
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  }[character]));
}

function removeSelected() {
  if (!appState.views.length) return;
  const [removed] = appState.views.splice(appState.selectedIndex, 1);
  URL.revokeObjectURL(removed.blobUrl);
  appState.selectedIndex = Math.min(appState.selectedIndex, appState.views.length - 1);
  appState.result = null;
  reviewResult.classList.add("hidden");
  renderWorkspace();
}

function resultTone(outcome) {
  if (outcome === "RAISE_PRIORITY") return "priority";
  if (outcome === "UNSUPPORTED") return "unsupported";
  if (outcome === "NEEDS_HUMAN_REVIEW") return "review";
  return "standard";
}

function outcomeTitle(outcome) {
  return {
    RAISE_PRIORITY: "Raise priority",
    STANDARD_HUMAN_REVIEW: "Standard human review",
    NEEDS_HUMAN_REVIEW: "Needs human review",
    UNSUPPORTED: "Unsupported input",
  }[outcome] || "Review outcome";
}

function humanReason(result) {
  const reason = String(result.reason || "").replaceAll("_", " ");
  const readable = reason.replace(/\b\w/g, (letter) => letter.toUpperCase());
  if (result.deterministic_outcome === "RAISE_PRIORITY") {
    return result.model_predicted_abnormal
      ? "The frozen study-level policy flagged the study. This raises workflow priority; it is not a diagnosis."
      : "The view-level evidence crossed the frozen threshold differently across views, so review priority was raised.";
  }
  if (result.deterministic_outcome === "STANDARD_HUMAN_REVIEW") {
    return "No model flag was raised by the fixed policy. Standard human review remains required.";
  }
  return `Automatic classification was not completed: ${readable}.`;
}

function renderResult(result) {
  const evidence = result.view_evidence || {};
  const threshold = Number(evidence.threshold ?? result.model_metadata?.frozen_threshold);
  const probability = Number(evidence.mean_probability);
  const canShowScore = Boolean(result.classification_performed) && Number.isFinite(probability);
  const qc = Array.isArray(result.qc_results) ? result.qc_results : [];
  const qcStatus = qc.length ? (qc.some((item) => item.status === "FAIL") ? "Check required" : "Passed") : "Not run";
  const perView = Array.isArray(result.per_view_probabilities) ? result.per_view_probabilities : [];
  const tone = resultTone(result.deterministic_outcome);
  reviewResult.innerHTML = `
    <article class="result-card">
      <header class="result-header ${tone}">
        <p>DETERMINISTIC WORKFLOW OUTCOME</p>
        <h3>${escapeHtml(outcomeTitle(result.deterministic_outcome))}</h3>
      </header>
      <div class="result-body">
        <p class="result-reason">${escapeHtml(humanReason(result))}</p>
        ${canShowScore ? `<div class="score-row">
          <div class="score-tile"><span>STUDY SCORE</span><strong>${(probability * 100).toFixed(1)}%</strong></div>
          <div class="score-tile"><span>FROZEN THRESHOLD</span><strong>${(threshold * 100).toFixed(1)}%</strong></div>
        </div>` : ""}
        <ul class="result-facts">
          <li><span>Human review</span><b>Required</b></li>
          <li><span>Technical QC</span><b>${escapeHtml(qcStatus)}</b></li>
          <li><span>Real views reviewed</span><b>${escapeHtml(String(appState.views.length))}</b></li>
          ${canShowScore ? `<li><span>View disagreement</span><b>${evidence.threshold_disagreement ? "Present" : "Not present"}</b></li>` : ""}
          ${perView.length ? `<li><span>Per-view scores</span><b>${perView.map((value) => `${(value * 100).toFixed(0)}%`).join(" · ")}</b></li>` : ""}
        </ul>
        <p class="result-footnote">This local prototype applies the fixed study-level Model B policy. It does not diagnose a fracture or remove a case from standard clinical care.</p>
      </div>
    </article>
  `;
  reviewResult.classList.remove("hidden");
  updateProgress(2);
  reviewResult.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function runReview() {
  if (!appState.views.length) return showToast("Add at least one study image first.");
  const button = $("#reviewButton");
  button.disabled = true;
  button.querySelector("strong").textContent = "Reviewing study…";
  button.querySelector("small").textContent = "Loading the fixed local model if needed";
  try {
    const views = await Promise.all(appState.views.map(async (view) => ({
      name: view.name,
      data_url: await fileToDataUrl(view.file),
    })));
    const response = await fetch("/api/review", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        case_alias: $("#caseAlias").value.trim() || "LOCAL-CASE",
        declared_anatomy: $("#anatomy").value,
        views,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "The review could not be completed.");
    appState.result = payload;
    renderResult(payload);
  } catch (error) {
    showToast(error instanceof Error ? error.message : "The review could not be completed.");
  } finally {
    button.disabled = false;
    button.querySelector("strong").textContent = "Review this study";
    button.querySelector("small").textContent = "Run the verified deterministic model";
  }
}

function resetReview() {
  appState.views.forEach((view) => URL.revokeObjectURL(view.blobUrl));
  appState.views = [];
  appState.selectedIndex = 0;
  appState.result = null;
  filePicker.value = "";
  folderPicker.value = "";
  $("#caseAlias").value = "LOCAL-CASE";
  $("#anatomy").value = "XR_WRIST";
  reviewResult.classList.add("hidden");
  workspace.classList.add("hidden");
  uploadState.classList.remove("hidden");
  updateProgress(0);
  showScreen("reviewScreen");
}

function showScreen(targetId) {
  $$(".screen").forEach((screen) => screen.classList.toggle("active", screen.id === targetId));
  $$(".nav-item").forEach((button) => button.classList.toggle("active", button.dataset.target === targetId));
  window.scrollTo({ top: 0, behavior: "smooth" });
}

async function loadModelStatus() {
  const target = $("#modelStatus");
  try {
    const response = await fetch("/api/status");
    const status = await response.json();
    if (!response.ok || !status.ready) {
      const issue = Array.isArray(status.issues) ? status.issues.join(", ") : "local bundle unavailable";
      target.innerHTML = `<div class="info-icon blue">!</div><p class="status-problem">Model bundle not ready: ${escapeHtml(issue)}.</p>`;
      return;
    }
    const threshold = (Number(status.threshold) * 100).toFixed(1);
    target.innerHTML = `<div class="info-icon teal">✓</div><p class="status-ready"><b>${escapeHtml(status.model || "Model B")}</b> verified locally · frozen threshold ${threshold}% · ${escapeHtml(status.study_aggregation || "mean probability")}</p>`;
  } catch (_error) {
    target.innerHTML = `<div class="info-icon blue">!</div><p class="status-problem">Could not verify the local model bundle.</p>`;
  }
}

filePicker.addEventListener("change", (event) => selectFiles(event.target.files, event.target));
folderPicker.addEventListener("change", (event) => selectFiles(event.target.files, event.target));
$("#changeFiles").addEventListener("click", () => filePicker.click());
$("#removeSelected").addEventListener("click", removeSelected);
$("#reviewButton").addEventListener("click", runReview);
$("#resetButton").addEventListener("click", resetReview);
$$(".nav-item").forEach((button) => button.addEventListener("click", () => showScreen(button.dataset.target)));

loadModelStatus();
