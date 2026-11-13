const MAX_STUDY_VIEWS = 8;

const appState = {
  queueName: "",
  patientFolders: [],
  studies: [],
  selectedStudyKey: null,
  running: false,
  toastTimer: null,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  }[character]));
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function filenameStem(name) {
  return String(name).replace(/\.[^.]+$/, "");
}

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("show");
  window.clearTimeout(appState.toastTimer);
  appState.toastTimer = window.setTimeout(() => toast.classList.remove("show"), 3800);
}

function updateProgress(step) {
  $$(".progress-step").forEach((item, index) => item.classList.toggle("current", index <= step));
}

async function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error(`Could not read ${file.name}.`));
    reader.readAsDataURL(file);
  });
}

async function imageDetails(file, relativePath) {
  const blobUrl = URL.createObjectURL(file);
  const dimensions = await new Promise((resolve) => {
    const image = new Image();
    image.onload = () => resolve({ width: image.naturalWidth, height: image.naturalHeight, previewable: true });
    image.onerror = () => resolve({ width: null, height: null, previewable: false });
    image.src = blobUrl;
  });
  return {
    file,
    name: file.name,
    relativePath,
    blobUrl,
    type: file.type || "Image file",
    size: file.size,
    ...dimensions,
  };
}

function studyFolderParts(relativePath, fallbackName) {
  const raw = String(relativePath || fallbackName).split("/").filter(Boolean);
  const root = raw.length > 1 ? raw[0] : "Selected worklist";
  const folders = raw.length > 2 ? raw.slice(1, -1) : [];
  return { root, folders, relativePath: raw.join("/") || fallbackName };
}

function createInputAttentionResult(reason) {
  return {
    deterministic_outcome: "NEEDS_HUMAN_REVIEW",
    classification_performed: false,
    reason,
    human_review_required: true,
    worklist: {
      bucket: "input_attention",
      rank: 1,
      study_score: null,
      ambiguous_view_evidence: false,
      human_review_required: true,
      agent_status: "not_run",
      agent_note: "The study was not sent to the model. Correct the input selection first.",
    },
  };
}

function createRequestAttentionResult(reason) {
  return {
    deterministic_outcome: "NEEDS_HUMAN_REVIEW",
    classification_performed: false,
    reason,
    human_review_required: true,
    worklist: {
      bucket: "input_attention",
      rank: 1,
      study_score: null,
      ambiguous_view_evidence: false,
      human_review_required: true,
      agent_status: "not_run",
      agent_note: "The local request did not complete. Human review remains required.",
    },
  };
}

function validateStudy(study) {
  const names = new Set();
  for (const view of study.views) {
    if (names.has(view.name)) return "duplicate_file_name_in_study";
    names.add(view.name);
  }
  if (study.views.length > MAX_STUDY_VIEWS) return `too_many_views_maximum_${MAX_STUDY_VIEWS}`;
  if (!study.views.length) return "no_readable_images";
  return null;
}

function localWorklistEntries(fileList) {
  return [...fileList].map((file) => ({
    file,
    relativePath: file.webkitRelativePath || file.name,
  }));
}

function fileFromDataUrl(dataUrl, name, mimeType) {
  const separator = String(dataUrl).indexOf(",");
  if (separator < 0) throw new Error(`Could not read ${name} from the ZIP archive.`);
  const binary = window.atob(String(dataUrl).slice(separator + 1));
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return new File([bytes], name, { type: mimeType || "application/octet-stream" });
}

async function buildWorklist(entries) {
  const files = entries.filter(({ file }) => file.size > 0 && (file.type.startsWith("image/") || /\.(png|jpe?g|bmp|gif|webp)$/i.test(file.name)));
  if (!files.length) throw new Error("No readable image files were found in this selection.");

  const groups = new Map();
  for (const { file, relativePath } of files) {
    const path = studyFolderParts(relativePath, file.name);
    const folderKey = path.folders.length > 1
      ? path.folders.join("/")
      : path.folders.length === 1
        ? [path.root, path.folders[0]].join("/")
        : path.root;
    if (!groups.has(folderKey)) {
      const patientFolder = path.folders.length > 1 ? path.folders[0] : (path.folders.length === 1 ? path.root : "");
      const studyFolder = path.folders.length > 1 ? path.folders.slice(1).join(" / ") : (path.folders[0] || path.root || "Selected study");
      groups.set(folderKey, {
        key: folderKey,
        originalIndex: groups.size,
        patientFolder,
        studyFolder,
        folderPath: folderKey,
        views: [],
        selectedViewIndex: 0,
        inputIssue: null,
        result: null,
        processing: false,
      });
    }
    groups.get(folderKey).views.push(await imageDetails(file, path.relativePath));
  }

  const studies = [...groups.values()].sort((left, right) => left.folderPath.localeCompare(right.folderPath));
  studies.forEach((study, index) => {
    study.originalIndex = index;
    study.inputIssue = validateStudy(study);
    if (study.inputIssue) study.result = createInputAttentionResult(study.inputIssue);
  });
  const patientFolders = [...new Set(studies.map((study) => study.patientFolder || study.studyFolder))];
  return { root: studyFolderParts(files[0].relativePath, files[0].file.name).root, patientFolders, studies };
}

function releaseViews(studies) {
  studies.forEach((study) => study.views.forEach((view) => URL.revokeObjectURL(view.blobUrl)));
}

async function appendWorklist(entries, sourceInput, sourceLabel = "patient folder") {
  try {
    const queue = await buildWorklist(entries);
    const existingKeys = new Set(appState.studies.map((study) => study.key));
    const newStudies = queue.studies.filter((study) => !existingKeys.has(study.key));
    const duplicateStudies = queue.studies.filter((study) => existingKeys.has(study.key));
    const skippedStudies = duplicateStudies.length;
    releaseViews(duplicateStudies);
    if (!newStudies.length) {
      sourceInput.value = "";
      showToast("Those studies are already in the queue.");
      return;
    }
    newStudies.forEach((study, index) => { study.originalIndex = appState.studies.length + index; });
    appState.patientFolders = [...new Set([...appState.patientFolders, ...queue.patientFolders])];
    appState.queueName = appState.patientFolders.length === 1 ? appState.patientFolders[0] : "Selected patient folders";
    appState.studies.push(...newStudies);
    appState.selectedStudyKey = appState.selectedStudyKey || newStudies[0]?.key || null;
    $("#queueEmpty").classList.add("hidden");
    $("#worklistWorkspace").classList.remove("hidden");
    renderWorklist();
    updateProgress(1);
    const blocked = newStudies.filter((study) => study.inputIssue).length;
    sourceInput.value = "";
    const skippedNote = skippedStudies ? ` ${skippedStudies} duplicate ${skippedStudies === 1 ? "study was" : "studies were"} skipped.` : "";
    showToast(blocked ? `${newStudies.length} studies added from ${sourceLabel}. ${blocked} need an input check.${skippedNote}` : `${newStudies.length} studies added from ${sourceLabel}.${skippedNote}`);
  } catch (error) {
    sourceInput.value = "";
    showToast(error instanceof Error ? error.message : "Could not read that local worklist.");
  }
}

async function selectWorklist(fileList, sourceInput) {
  return appendWorklist(localWorklistEntries(fileList), sourceInput, "patient folder");
}

async function selectZipWorklist(file, sourceInput) {
  if (!file) return;
  try {
    const response = await fetch("/api/worklist-zip", {
      method: "POST",
      headers: { "Content-Type": "application/zip" },
      body: file,
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not read that ZIP worklist.");
    const entries = payload.files.map((entry) => ({
      file: fileFromDataUrl(entry.data_url, entry.name, entry.mime_type),
      relativePath: entry.relative_path,
    }));
    await appendWorklist(entries, sourceInput, file.name);
  } catch (error) {
    sourceInput.value = "";
    showToast(error instanceof Error ? error.message : "Could not read that ZIP worklist.");
  }
}

function worklistMeta(study) {
  return study.result?.worklist || { rank: 3, study_score: null, bucket: "pending" };
}

function orderedStudies() {
  return [...appState.studies].sort((left, right) => {
    const leftMeta = worklistMeta(left);
    const rightMeta = worklistMeta(right);
    if (leftMeta.rank !== rightMeta.rank) return leftMeta.rank - rightMeta.rank;
    const leftScore = Number.isFinite(leftMeta.study_score) ? leftMeta.study_score : -1;
    const rightScore = Number.isFinite(rightMeta.study_score) ? rightMeta.study_score : -1;
    if (leftScore !== rightScore) return rightScore - leftScore;
    return left.originalIndex - right.originalIndex;
  });
}

function outcomeTitle(outcome) {
  return {
    RAISE_PRIORITY: "Priority review",
    STANDARD_HUMAN_REVIEW: "Routine human review",
    NEEDS_HUMAN_REVIEW: "Input attention",
    UNSUPPORTED: "Unsupported input",
  }[outcome] || "Ready for review";
}

function outcomeTone(outcome) {
  if (outcome === "RAISE_PRIORITY") return "priority";
  if (outcome === "STANDARD_HUMAN_REVIEW") return "routine";
  if (outcome === "UNSUPPORTED") return "unsupported";
  return "attention";
}

function readableAgentStatus(status) {
  return {
    completed_bounded_investigation: "Evidence check completed",
    not_called_no_optional_tools: "No optional tool authorized",
    not_called_deterministic_route: "Blocked by the safety route",
    safe_failure: "Evidence check unavailable",
    not_run: "Not run",
  }[String(status)] || "Deterministic route retained";
}

function readableResearchAgentStatus(status) {
  return {
    completed_research_explanation: "Research explanation completed",
    not_configured: "Optional research explanation is off",
    not_called: "Not eligible for research explanation",
    safe_failure: "Research explanation unavailable",
    not_run: "Not run",
  }[String(status)] || "Deterministic route retained";
}

function readableAgentReason(reason) {
  return {
    deterministic_policy_authorized_tta_and_gradcam: "The fixed evidence policy authorized local TTA and Grad-CAM summaries. The review route did not change.",
    consistent_normal_without_optional_tool_trigger: "This was a consistent normal study outside the configured borderline band, so no optional evidence tool was authorized.",
    deterministic_route_not_eligible_for_agent: "The study was already routed by a deterministic safety rule before the model workflow.",
    optional_tool_policy_not_authorized: "The fixed optional-evidence policy did not authorize tool use for this study.",
    agent_tool_unavailable: "An authorized optional evidence tool was unavailable, so the case was escalated safely.",
    agent_tool_failure: "An authorized optional evidence tool did not return valid evidence, so the case was escalated safely.",
  }[String(reason)] || "The deterministic route was retained for human review.";
}

function readableToolName(tool) {
  return {
    input_qc: "Technical QC",
    support_check: "Wrist scope check",
    primary_model: "Fixed Model B",
    view_evidence: "View evidence",
    uncertainty_tta: "Local TTA stability check",
    gradcam_explainer: "Local Grad-CAM attribution summary",
  }[String(tool)] || String(tool).replaceAll("_", " ");
}

function agentEvidencePanel(study) {
  const report = study.result?.agent_report;
  const researchReport = study.result?.research_agent_report;
  const meta = worklistMeta(study);
  if (!study.result) {
    return `<section class="agent-evidence-card pending"><div class="agent-evidence-heading"><span class="agent-icon" aria-hidden="true">↗</span><div><p class="section-overline">OPTIONAL EVIDENCE</p><h4>Available after review</h4></div></div><p>After Model B and the deterministic rules run, optional evidence is considered only when the fixed policy finds an abnormal mean, conflicting views, or a borderline study score.</p></section>`;
  }

  const status = String(report?.agent_status || meta.agent_status || "not_run");
  const reason = String(report?.reason_for_escalation_or_rejection || meta.agent_note || "");
  const tools = Array.isArray(report?.allowed_tools_called) ? report.allowed_tools_called : [];
  const tta = tools.find((tool) => tool?.tool === "uncertainty_tta" && tool?.status === "OK")?.payload;
  const gradcam = tools.find((tool) => tool?.tool === "gradcam_explainer" && tool?.status === "OK")?.payload;
  const triggers = Array.isArray(report?.triggers_fired) ? report.triggers_fired : [];
  const tone = status === "completed_bounded_investigation" ? "completed" : status === "safe_failure" ? "attention" : "quiet";
  const toolsMarkup = tools.length
    ? `<div class="agent-tool-list">${tools.map((tool) => `<span class="agent-tool ${tool?.status === "FAILED" ? "failed" : ""}">${escapeHtml(readableToolName(tool?.tool))}</span>`).join("")}</div>`
    : "";
  const ttaMarkup = tta
    ? `<div class="tta-facts"><span>TTA range<b>${(Number(tta.tta_range) * 100).toFixed(1)} pts</b></span><span>Variation<b>${(Number(tta.tta_std) * 100).toFixed(1)} pts</b></span><span>Changed route?<b>No</b></span></div>`
    : "";
  const gradcamMarkup = gradcam
    ? `<div class="tta-facts"><span>Grad-CAM views<b>${escapeHtml(String(gradcam.views_processed || 0))}</b></span><span>Maps saved<b>No</b></span><span>Localization claim?<b>No</b></span></div>`
    : "";
  const triggerLabels = {
    abnormal_mean_prediction: "abnormal mean study score",
    view_threshold_disagreement: "conflicting view scores around the frozen threshold",
    borderline_mean_probability: "mean score inside the configured borderline band",
  };
  const triggerMarkup = triggers.length
    ? `<p class="agent-trigger">Trigger: ${triggers.map((trigger) => triggerLabels[trigger] || String(trigger)).join(" · ")}.</p>`
    : "";
  const researchStatus = String(researchReport?.agent_status || meta.research_agent_status || "not_run");
  const researchSummary = typeof researchReport?.summary === "string" ? researchReport.summary : "";
  const researchLimitations = Array.isArray(researchReport?.limitations) ? researchReport.limitations : [];
  const researchTone = researchStatus === "completed_research_explanation" ? "completed" : researchStatus === "safe_failure" ? "attention" : "quiet";
  const researchMarkup = `<div class="research-explanation ${researchTone}">
    <div><p class="section-overline">OPTIONAL RESEARCH EXPLANATION</p><h5>${escapeHtml(readableResearchAgentStatus(researchStatus))}</h5></div>
    ${researchSummary ? `<p>${escapeHtml(researchSummary)}</p>` : `<p>${researchStatus === "not_configured" ? "The deterministic route is complete. CrewAI is opt-in and was not enabled for this local session." : "This optional layer is available after a valid supported deterministic route."}</p>`}
    ${researchLimitations.length ? `<ul>${researchLimitations.slice(0, 3).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>` : ""}
    <small>${researchStatus === "completed_research_explanation" ? "It received structured, de-identified evidence only. It cannot change the route or review requirement." : "Raw images, paths, labels, and patient identifiers are never sent to this layer."}</small>
  </div>`;

  return `<section class="agent-evidence-card ${tone}">
    <div class="agent-evidence-heading"><span class="agent-icon" aria-hidden="true">${status === "completed_bounded_investigation" ? "✓" : "↗"}</span><div><p class="section-overline">OPTIONAL EVIDENCE</p><h4>${escapeHtml(readableAgentStatus(status))}</h4></div></div>
    <p>${escapeHtml(readableAgentReason(reason))}</p>
    ${triggerMarkup}
    ${toolsMarkup}
    ${ttaMarkup}
    ${gradcamMarkup}
    <small>Evidence only — it cannot change the model score, frozen threshold, queue route, or human-review requirement.</small>
    ${researchMarkup}
  </section>`;
}

function visibleStatus(study) {
  if (study.processing) return { label: "Reviewing…", tone: "processing" };
  if (study.result) return { label: outcomeTitle(study.result.deterministic_outcome), tone: outcomeTone(study.result.deterministic_outcome) };
  return { label: "Ready", tone: "ready" };
}

function selectedStudy() {
  return appState.studies.find((study) => study.key === appState.selectedStudyKey) || appState.studies[0] || null;
}

function renderStudyList() {
  const ordered = orderedStudies();
  $("#studyCountBadge").textContent = String(ordered.length);
  $("#studyList").innerHTML = ordered.map((study, index) => {
    const status = visibleStatus(study);
    const score = worklistMeta(study).study_score;
    const scoreText = Number.isFinite(score) ? `${(score * 100).toFixed(1)}%` : `${study.views.length} views`;
    return `<button type="button" class="study-row ${study.key === appState.selectedStudyKey ? "selected" : ""}" data-study-key="${escapeHtml(study.key)}">
      <span class="study-order">${study.result ? index + 1 : "·"}</span>
      <span class="study-row-copy"><b>${escapeHtml(study.studyFolder)}</b><small>${escapeHtml(study.patientFolder || "Local study folder")} · ${escapeHtml(scoreText)}</small></span>
      <span class="route-pill ${status.tone}">${escapeHtml(status.label)}</span>
    </button>`;
  }).join("");
  $$(".study-row").forEach((button) => button.addEventListener("click", () => {
    appState.selectedStudyKey = button.dataset.studyKey;
    renderWorklist();
  }));
}

function renderInspector() {
  const study = selectedStudy();
  const target = $("#studyInspector");
  if (!study) {
    target.innerHTML = "";
    return;
  }
  const currentView = study.views[study.selectedViewIndex] || study.views[0];
  const status = visibleStatus(study);
  const score = worklistMeta(study).study_score;
  const scoreLine = Number.isFinite(score) ? `<span class="detail-score">${(score * 100).toFixed(1)}% study score</span>` : "";
  target.innerHTML = `
    <div class="inspector-header"><div><p class="section-overline">SELECTED STUDY</p><h3>${escapeHtml(study.studyFolder)}</h3><p>${escapeHtml(study.patientFolder || "Local study folder")} · ${escapeHtml(study.folderPath)}</p></div><span class="route-pill ${status.tone}">${escapeHtml(status.label)}</span></div>
    ${study.inputIssue ? `<div class="input-warning"><b>Not submitted to the model.</b><span>${escapeHtml(study.inputIssue.replaceAll("_", " "))}</span></div>` : ""}
    <div class="study-preview-grid">
      <div class="view-rail">${study.views.map((view, index) => `<button type="button" class="view-thumb ${index === study.selectedViewIndex ? "selected" : ""}" data-view-index="${index}"><img src="${view.blobUrl}" alt=""/><span>View ${index + 1}</span></button>`).join("")}</div>
      <div class="preview-panel">
        <div class="image-frame"><img src="${currentView.blobUrl}" alt="Preview of ${escapeHtml(currentView.name)}"/><span>View ${study.selectedViewIndex + 1} of ${study.views.length}</span></div>
        <div class="file-caption"><b>${escapeHtml(currentView.name)}</b><small>${escapeHtml(currentView.relativePath)}</small></div>
        <div class="file-facts"><span>FORMAT<b>${escapeHtml((currentView.type.split("/").pop() || "image").toUpperCase())}</b></span><span>SIZE<b>${escapeHtml(formatBytes(currentView.size))}</b></span><span>DIMENSIONS<b>${currentView.previewable ? `${currentView.width} × ${currentView.height}` : "Preview unavailable"}</b></span></div>
      </div>
    </div>
    ${agentEvidencePanel(study)}
    <div class="inspector-footer"><span>${study.views.length} real views · ${scoreLine || "Model not run"}</span><span>Human review required</span></div>
  `;
  $$(".view-thumb").forEach((button) => button.addEventListener("click", () => {
    study.selectedViewIndex = Number(button.dataset.viewIndex);
    renderInspector();
  }));
}

function renderQueueSummary() {
  const runnable = appState.studies.filter((study) => !study.inputIssue).length;
  const needsInput = appState.studies.length - runnable;
  const folderCount = appState.patientFolders.length;
  $("#queueName").textContent = appState.queueName || "Local worklist";
  $("#queueSummary").textContent = `${folderCount} patient ${folderCount === 1 ? "folder" : "folders"} · ${runnable} ${runnable === 1 ? "study" : "studies"} ready${needsInput ? ` · ${needsInput} need input attention` : ""}`;
}

function resultReason(result) {
  if (result.deterministic_outcome === "RAISE_PRIORITY") {
    return result.model_predicted_abnormal ? "The fixed model policy flagged this study for priority human review." : "Views crossed the fixed threshold differently, so priority review was raised.";
  }
  if (result.deterministic_outcome === "STANDARD_HUMAN_REVIEW") return "No model priority flag was raised. Human review remains required.";
  return `Automatic classification was not completed: ${String(result.reason || "input check required").replaceAll("_", " ")}.`;
}

function renderResults() {
  const target = $("#resultsPanel");
  const completed = appState.studies.filter((study) => study.result);
  if (!completed.length) {
    target.classList.add("hidden");
    return;
  }
  const ordered = orderedStudies().filter((study) => study.result);
  const priority = ordered.filter((study) => worklistMeta(study).bucket === "priority").length;
  const attention = ordered.filter((study) => worklistMeta(study).bucket === "input_attention").length;
  const routine = ordered.filter((study) => worklistMeta(study).bucket === "routine").length;
  const investigated = ordered.filter((study) => String(worklistMeta(study).agent_status) === "completed_bounded_investigation").length;
  target.innerHTML = `
    <div class="results-heading"><div><p class="section-overline">HUMAN-REVIEW ORDER</p><h3>Prioritized worklist</h3><p>Higher-priority studies appear first. This is a workflow order, not a diagnosis.</p></div><div class="result-counts"><span class="priority">${priority} priority</span><span class="attention">${attention} attention</span><span class="routine">${routine} routine</span></div></div>
    <p class="agent-run-summary">${investigated ? `${investigated} ${investigated === 1 ? "study used" : "studies used"} the bounded optional-evidence check.` : "No study was authorized for an optional evidence tool."} It adds evidence only; every study remains in human review.</p>
    <div class="ranked-results">${ordered.map((study, index) => {
      const result = study.result;
      const meta = worklistMeta(study);
      const score = Number.isFinite(meta.study_score) ? `${(meta.study_score * 100).toFixed(1)}%` : "—";
      const agentStatus = String(meta.agent_status || "not_run");
      const agentText = readableAgentStatus(agentStatus);
      return `<article class="result-row ${outcomeTone(result.deterministic_outcome)}"><span class="result-rank">${index + 1}</span><div class="result-copy"><div><h4>${escapeHtml(study.studyFolder)}</h4><span>${escapeHtml(study.patientFolder || "Local study folder")} · ${study.views.length} views</span></div><p>${escapeHtml(resultReason(result))}</p><small>${escapeHtml(agentText)} · Human review required</small></div><div class="result-score"><b>${escapeHtml(score)}</b><span>${escapeHtml(outcomeTitle(result.deterministic_outcome))}</span></div></article>`;
    }).join("")}</div>`;
  target.classList.remove("hidden");
}

function renderWorklist() {
  renderQueueSummary();
  renderStudyList();
  renderInspector();
  renderResults();
}

async function runWorklist() {
  if (appState.running) return;
  const candidates = appState.studies.filter((study) => !study.inputIssue && !study.result);
  if (!candidates.length) {
    showToast("There are no unreviewed studies ready for the model. Choose another folder to start again.");
    return;
  }
  appState.running = true;
  const button = $("#runWorklistButton");
  button.disabled = true;
  for (let index = 0; index < candidates.length; index += 1) {
    const study = candidates[index];
    study.processing = true;
    $("#runButtonNote").textContent = `Reviewing study ${index + 1} of ${candidates.length}`;
    renderWorklist();
    try {
      const views = await Promise.all(study.views.map(async (view) => ({ name: view.name, data_url: await fileToDataUrl(view.file) })));
      const response = await fetch("/api/review", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ case_alias: study.folderPath, declared_anatomy: $("#anatomy").value, views }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "The local review did not complete.");
      study.result = payload;
    } catch (error) {
      study.result = createRequestAttentionResult(error instanceof Error ? error.message : "local_review_failed");
    } finally {
      study.processing = false;
      renderWorklist();
    }
  }
  appState.running = false;
  button.disabled = false;
  $("#runButtonNote").textContent = "Worklist review complete";
  updateProgress(2);
  $("#resultsPanel").scrollIntoView({ behavior: "smooth", block: "start" });
  showToast("Worklist review complete. All studies remain in human review.");
}

function resetWorklist() {
  releaseViews(appState.studies);
  appState.queueName = "";
  appState.patientFolders = [];
  appState.studies = [];
  appState.selectedStudyKey = null;
  appState.running = false;
  $("#worklistPicker").value = "";
  $("#zipWorklistPicker").value = "";
  $("#anatomy").value = "XR_WRIST";
  $("#queueEmpty").classList.remove("hidden");
  $("#worklistWorkspace").classList.add("hidden");
  $("#resultsPanel").classList.add("hidden");
  $("#runButtonNote").textContent = "Run the fixed local model study by study";
  updateProgress(0);
  showScreen("queueScreen");
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
    const research = status.research_agent || {};
    const optionalPolicy = status.optional_tool_policy || {};
    const policyCopy = optionalPolicy.ready
      ? `Optional evidence: abnormal score, conflicting views, or within ±${(Number(optionalPolicy.borderline_margin) * 100).toFixed(0)} points of the frozen threshold.`
      : "Optional evidence policy is unavailable, so its tools will remain blocked safely.";
    const researchCopy = research.ready
      ? `CrewAI research explanation is enabled · max ${escapeHtml(String(research.max_llm_calls || 3))} LLM calls per eligible study · no raw images shared.`
      : "CrewAI research explanation is optional and off for this local session; local safety rules and optional evidence remain active.";
    target.innerHTML = `<div class="info-icon teal">✓</div><div><p class="status-ready"><b>${escapeHtml(status.model || "Model B")}</b> verified locally · frozen threshold ${(Number(status.threshold) * 100).toFixed(1)}% · ${escapeHtml(status.study_aggregation || "mean probability")}</p><p class="status-note">${policyCopy}</p><p class="status-note">${researchCopy}</p></div>`;
  } catch (_error) {
    target.innerHTML = `<div class="info-icon blue">!</div><p class="status-problem">Could not verify the local model bundle.</p>`;
  }
}

$("#worklistPicker").addEventListener("change", (event) => selectWorklist(event.target.files, event.target));
$("#zipWorklistPicker").addEventListener("change", (event) => selectZipWorklist(event.target.files[0], event.target));
$("#addPatientFolder").addEventListener("click", () => $("#worklistPicker").click());
$("#addZipWorklist").addEventListener("click", () => $("#zipWorklistPicker").click());
$("#resetButton").addEventListener("click", resetWorklist);
$("#runWorklistButton").addEventListener("click", runWorklist);
$$(".nav-item").forEach((button) => button.addEventListener("click", () => showScreen(button.dataset.target)));

loadModelStatus();
