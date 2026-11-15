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
    not_called_policy_blocked: "Optional tools blocked by policy",
    safe_failure: "Evidence check unavailable",
    not_run: "Not run",
  }[String(status)] || "Deterministic route retained";
}

function readableResearchAgentStatus(status) {
  return {
    completed_research_explanation: "Research explanation completed",
    not_configured: "Optional research explanation is off",
    google_api_key_not_configured: "Google API key is not configured",
    crewai_dependency_not_installed: "CrewAI runtime is not installed",
    not_called: "Not eligible for research explanation",
    safe_failure: "Research explanation unavailable",
    not_run: "Not run",
  }[String(status)] || "Deterministic route retained";
}

function readableResearchAgentReason(reason) {
  const value = String(reason || "");
  const knownReasons = {
    disabled_by_operator: "Start the app with the optional CrewAI explanation enabled to request a research-only report.",
    llm_model_not_configured: "No LLM model was configured for this local session.",
    google_api_key_not_configured: "The local session has no Google API key, so no external explanation was requested.",
    crewai_dependency_not_installed: "The optional CrewAI runtime is not installed in this Python environment.",
    deterministic_route_not_eligible_for_agent: "The deterministic safety route ended before a research explanation could be requested.",
  };
  if (knownReasons[value]) return knownReasons[value];
  if (value.startsWith("research_explanation_unavailable:")) {
    const category = value.slice("research_explanation_unavailable:".length) || "unknown runtime error";
    const validationReasons = {
      agent_response_not_object: "CrewAI returned a response outside the required report format, so it was rejected safely.",
      invalid_research_summary: "CrewAI returned an incomplete summary, so it was rejected safely.",
      invalid_evidence_used: "CrewAI returned an invalid evidence list, so it was rejected safely.",
      invalid_limitations: "CrewAI returned an invalid limitations list, so it was rejected safely.",
      required_safety_flags_missing: "CrewAI omitted required research-safety flags, so it was rejected safely.",
      unsupported_clinical_claim: "CrewAI returned an unsupported clinical claim, so it was rejected safely.",
    };
    if (validationReasons[category]) {
      return `${validationReasons[category]} No PDF was created; the deterministic route remains unchanged.`;
    }
    return `CrewAI stopped safely before it returned an explanation (${category}). No PDF was created; the deterministic route remains unchanged.`;
  }
  if (value.startsWith("context_validation_failed:")) {
    return "The local evidence record did not satisfy the research-report contract, so no external explanation was requested.";
  }
  return "The optional research explanation was not completed. The deterministic route remains unchanged.";
}

function readableAgentReason(reason) {
  return {
    deterministic_policy_authorized_tta_and_gradcam: "The fixed evidence policy authorized local TTA and Grad-CAM summaries. The review route did not change.",
    consistent_normal_without_optional_tool_trigger: "This was a consistent normal study outside the configured borderline band, so no optional evidence tool was authorized.",
    deterministic_route_not_eligible_for_agent: "The study was already routed by a deterministic safety rule before the model workflow.",
    optional_tool_policy_not_authorized: "The fixed optional-evidence policy did not authorize tool use for this study.",
    agent_tool_unavailable: "An authorized optional evidence tool was unavailable, so the case was escalated safely.",
    agent_tool_failure: "An authorized optional evidence tool did not return valid evidence, so the case was escalated safely.",
    agent_tool_budget_exceeded: "The optional-evidence tool budget was exceeded, so the case was escalated safely.",
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

function optionalToolSummary(study) {
  const report = study.result?.agent_report;
  const policy = study.result?.optional_tool_policy;
  const completedTools = Array.isArray(report?.allowed_tools_called)
    ? report.allowed_tools_called.filter((tool) => tool?.status === "OK").map((tool) => String(tool?.tool || ""))
    : [];
  const authorizedTools = Array.isArray(policy?.authorized_tools) ? policy.authorized_tools.map(String) : [];
  // Once an execution report exists, show only tools that actually completed.
  // The policy allowlist is useful before execution, but is not evidence of a tool call.
  const allTools = [...new Set(report ? completedTools : authorizedTools)];
  const names = [];
  if (allTools.includes("uncertainty_tta") || allTools.includes("tta_uncertainty")) names.push("TTA");
  if (allTools.includes("gradcam_explainer")) names.push("Grad-CAM");
  return names;
}

function studyDecision(study) {
  if (!study.result) {
    return { label: "Awaiting review", detail: "The fixed model has not run yet.", tools: [], tone: "pending" };
  }
  const result = study.result;
  const policy = result.optional_tool_policy || {};
  const report = result.agent_report || {};
  const triggers = Array.isArray(policy.triggers) ? policy.triggers : (Array.isArray(report.triggers_fired) ? report.triggers_fired : []);
  const tools = optionalToolSummary(study);

  if (result.deterministic_outcome === "UNSUPPORTED") {
    return { label: "Unsupported anatomy", detail: "Stopped before Model B inference.", tools: [], tone: "blocked" };
  }
  if (result.classification_performed !== true) {
    return { label: "Input attention", detail: "The deterministic safety route blocked classification.", tools: [], tone: "blocked" };
  }
  if (triggers.includes("view_threshold_disagreement")) {
    return { label: "Ambiguous views", detail: "View scores fall on both sides of the frozen threshold.", tools, tone: "ambiguous" };
  }
  if (triggers.includes("borderline_mean_probability")) {
    return { label: "Borderline score", detail: "The mean score is inside the fixed borderline band.", tools, tone: "ambiguous" };
  }
  if (triggers.includes("abnormal_mean_prediction")) {
    return { label: "Priority signal", detail: "The mean score crossed the frozen threshold.", tools, tone: "priority" };
  }
  return { label: "Routine case", detail: "Consistent score with no optional evidence trigger.", tools: [], tone: "routine" };
}

function readableAgentTrigger(trigger) {
  return {
    abnormal_mean_prediction: "mean score was abnormal",
    view_threshold_disagreement: "views disagreed around the frozen threshold",
    borderline_mean_probability: "mean score was inside the borderline band",
  }[String(trigger)] || String(trigger).replaceAll("_", " ");
}

function agentWorkflowSummary(study) {
  if (!study.result) {
    return {
      state: "Waiting for review",
      detail: "Model, safety rules, and the optional-evidence policy have not run yet.",
      tools: [],
      triggers: [],
      tone: "pending",
      research: "Not run",
    };
  }

  const result = study.result;
  const report = result.agent_report || {};
  const policy = result.optional_tool_policy || {};
  const status = String(report.agent_status || "not_run");
  const reason = String(report.reason_for_escalation_or_rejection || policy.reason || "");
  const triggers = Array.isArray(report.triggers_fired) && report.triggers_fired.length
    ? report.triggers_fired
    : (Array.isArray(policy.triggers) ? policy.triggers : []);
  const tools = optionalToolSummary(study);
  const researchStatus = String(result.research_agent_report?.agent_status || "not_configured");
  const research = readableResearchAgentStatus(researchStatus);

  if (result.deterministic_outcome === "UNSUPPORTED") {
    return {
      state: "Safety gate — agent not used",
      detail: "The declared anatomy is outside the wrist scope, so Model B and optional evidence tools are not called.",
      tools: [], triggers: [], tone: "blocked", research,
    };
  }
  if (result.classification_performed !== true) {
    return {
      state: "Safety gate — agent not used",
      detail: "Input safety checks stopped model classification before any optional evidence could run.",
      tools: [], triggers: [], tone: "blocked", research,
    };
  }
  if (status === "completed_bounded_investigation") {
    return {
      state: "Extra evidence ran",
      detail: "The fixed policy authorized local evidence. The deterministic route did not change.",
      tools, triggers, tone: "active", research,
    };
  }
  if (status === "not_called_no_optional_tools") {
    return {
      state: "No extra tools needed",
      detail: "Consistent normal result outside the borderline band. The policy kept TTA and Grad-CAM off.",
      tools: [], triggers: [], tone: "quiet", research,
    };
  }
  if (status === "safe_failure") {
    return {
      state: "Extra evidence unavailable",
      detail: readableAgentReason(reason),
      tools, triggers, tone: "attention", research,
    };
  }
  return {
    state: readableAgentStatus(status),
    detail: readableAgentReason(reason),
    tools, triggers, tone: "quiet", research,
  };
}

function agentReportRow(study, index) {
  const summary = agentWorkflowSummary(study);
  const decision = studyDecision(study);
  const triggerText = summary.triggers.length
    ? summary.triggers.map(readableAgentTrigger).join(" · ")
    : "No trigger";
  const toolsText = summary.tools.length ? summary.tools.join(" + ") : "None";
  return `<article class="agent-report-row ${escapeHtml(summary.tone)}">
    <span class="agent-report-rank">${index + 1}</span>
    <div class="agent-report-main">
      <div><h4>${escapeHtml(study.studyFolder)}</h4><span>${escapeHtml(decision.label)}</span></div>
      <b>${escapeHtml(summary.state)}</b>
      <p>${escapeHtml(summary.detail)}</p>
      <div class="agent-report-tags"><span><em>TRIGGER</em>${escapeHtml(triggerText)}</span><span><em>LOCAL TOOLS</em>${escapeHtml(toolsText)}</span></div>
    </div>
    <div class="agent-report-research"><em>CREWAI</em><b>${escapeHtml(summary.research)}</b><small>Separate, opt-in explanation</small></div>
  </article>`;
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
  const researchReason = String(researchReport?.stop_reason || "");
  const researchTone = researchStatus === "completed_research_explanation" ? "completed" : researchStatus === "safe_failure" ? "attention" : "quiet";
  const pdfAction = researchStatus === "completed_research_explanation"
    ? `<button class="research-pdf-button" type="button" data-research-pdf="${escapeHtml(study.key)}"><span aria-hidden="true">↓</span> Download PDF report</button>`
    : "";
  const researchMarkup = `<div class="research-explanation ${researchTone}">
    <div><p class="section-overline">LLM RESEARCH EXPLANATION</p><h5>${escapeHtml(readableResearchAgentStatus(researchStatus))}</h5></div>
    ${researchSummary ? `<p>${escapeHtml(researchSummary)}</p>` : `<p>${escapeHtml(readableResearchAgentReason(researchReason))}</p>`}
    ${researchLimitations.length ? `<ul>${researchLimitations.slice(0, 3).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>` : ""}
    <small>${researchStatus === "completed_research_explanation" ? "It received structured, de-identified evidence only. It cannot change the route or review requirement." : "Raw images, paths, labels, and patient identifiers are never sent to this layer."}</small>
    ${pdfAction}
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

async function downloadResearchReport(studyKey) {
  const study = appState.studies.find((item) => item.key === studyKey);
  if (!study?.result) {
    showToast("Review the study before requesting its report.");
    return;
  }
  const button = document.querySelector(`[data-research-pdf="${CSS.escape(studyKey)}"]`);
  if (button) {
    button.disabled = true;
    button.textContent = "Creating PDF…";
  }
  try {
    const response = await fetch("/api/research-report.pdf", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ review: study.result }),
    });
    if (!response.ok) {
      const errorPayload = await response.json().catch(() => ({}));
      throw new Error(errorPayload.error || "The PDF report could not be created.");
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const download = document.createElement("a");
    const disposition = response.headers.get("Content-Disposition") || "";
    const match = disposition.match(/filename="?([^";]+)"?/i);
    download.href = url;
    download.download = match?.[1] || "wrist_research_explanation.pdf";
    document.body.appendChild(download);
    download.click();
    download.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1_000);
    showToast("Research report PDF downloaded. It contains no radiographs.");
  } catch (error) {
    showToast(error instanceof Error ? error.message : "The PDF report could not be created.");
  } finally {
    if (button) {
      button.disabled = false;
      button.innerHTML = "<span aria-hidden=\"true\">↓</span> Download PDF report";
    }
  }
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
    const decision = studyDecision(study);
    const toolText = decision.tools.length ? `Tools: ${decision.tools.join(" + ")}` : decision.label === "Routine case" ? "No optional tools" : decision.label === "Awaiting review" ? "Not run" : "Tools blocked";
    return `<button type="button" class="study-row ${study.key === appState.selectedStudyKey ? "selected" : ""}" data-study-key="${escapeHtml(study.key)}">
      <span class="study-order">${study.result ? index + 1 : "·"}</span>
      <span class="study-row-copy"><b>${escapeHtml(study.studyFolder)}</b><small>${escapeHtml(study.patientFolder || "Local study folder")} · ${escapeHtml(scoreText)}</small><em class="study-decision ${escapeHtml(decision.tone)}">${escapeHtml(decision.label)} · ${escapeHtml(toolText)}</em></span>
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
  const decision = studyDecision(study);
  const toolsLine = decision.tools.length ? decision.tools.join(" + ") : decision.label === "Routine case" ? "No optional tools" : decision.label === "Awaiting review" ? "Pending" : "Tools blocked";
  target.innerHTML = `
    <div class="inspector-header"><div><p class="section-overline">SELECTED STUDY</p><h3>${escapeHtml(study.studyFolder)}</h3><p>${escapeHtml(study.patientFolder || "Local study folder")} · ${escapeHtml(study.folderPath)}</p></div><span class="route-pill ${status.tone}">${escapeHtml(status.label)}</span></div>
    ${study.inputIssue ? `<div class="input-warning"><b>Not submitted to the model.</b><span>${escapeHtml(study.inputIssue.replaceAll("_", " "))}</span></div>` : ""}
    <section class="decision-card ${escapeHtml(decision.tone)}"><div><p>CASE STATE</p><b>${escapeHtml(decision.label)}</b><small>${escapeHtml(decision.detail)}</small></div><div><p>OPTIONAL TOOLS</p><b>${escapeHtml(toolsLine)}</b><small>${decision.tools.length ? "Evidence only — it cannot change the route." : "The deterministic route stays in control."}</small></div></section>
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
  $$(".research-pdf-button").forEach((button) => button.addEventListener("click", () => {
    downloadResearchReport(button.dataset.researchPdf);
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
  const ambiguous = ordered.filter((study) => studyDecision(study).tone === "ambiguous").length;
  const noOptionalTools = ordered.filter((study) => String(worklistMeta(study).agent_status) === "not_called_no_optional_tools").length;
  const safetyBlocked = ordered.filter((study) => String(worklistMeta(study).agent_status) === "not_called_deterministic_route").length;
  target.innerHTML = `
    <div class="results-heading"><div><p class="section-overline">HUMAN-REVIEW ORDER</p><h3>Prioritized worklist</h3><p>Higher-priority studies appear first. This is a workflow order, not a diagnosis.</p></div><div class="result-counts"><span class="priority">${priority} priority</span><span class="attention">${attention} attention</span><span class="routine">${routine} routine</span><span class="ambiguous">${ambiguous} ambiguous</span></div></div>
    <p class="agent-run-summary">${investigated ? `${investigated} ${investigated === 1 ? "study used" : "studies used"} the bounded optional-evidence check.` : "No study was authorized for an optional evidence tool."} It adds evidence only; every study remains in human review.</p>
    <div class="ranked-results">${ordered.map((study, index) => {
      const result = study.result;
      const meta = worklistMeta(study);
      const score = Number.isFinite(meta.study_score) ? `${(meta.study_score * 100).toFixed(1)}%` : "—";
      const agentStatus = String(meta.agent_status || "not_run");
      const agentText = readableAgentStatus(agentStatus);
      const decision = studyDecision(study);
      const toolsText = decision.tools.length ? decision.tools.join(" + ") : decision.label === "Routine case" ? "No optional tools" : "Tools blocked";
      return `<article class="result-row ${outcomeTone(result.deterministic_outcome)}"><span class="result-rank">${index + 1}</span><div class="result-copy"><div><h4>${escapeHtml(study.studyFolder)}</h4><span>${escapeHtml(study.patientFolder || "Local study folder")} · ${study.views.length} views</span></div><p>${escapeHtml(resultReason(result))}</p><div class="result-decision ${escapeHtml(decision.tone)}"><b>${escapeHtml(decision.label)}</b><span>${escapeHtml(toolsText)}</span></div><small>${escapeHtml(agentText)} · Human review required</small></div><div class="result-score"><b>${escapeHtml(score)}</b><span>${escapeHtml(outcomeTitle(result.deterministic_outcome))}</span></div></article>`;
    }).join("")}</div>`;
  target.innerHTML += `
    <section class="agent-report-card">
      <div class="agent-report-heading"><div><p class="section-overline">AGENT & EVIDENCE REPORT</p><h3>When extra tools were used</h3><p>The deterministic model and safety rules always decide the review route first. The local bounded evidence workflow can only add evidence; it cannot change the score, threshold, route, or need for human review.</p></div><div class="agent-report-counts"><span>${investigated} evidence run${investigated === 1 ? "" : "s"}</span><span>${noOptionalTools} no tools</span><span>${safetyBlocked} safety block${safetyBlocked === 1 ? "" : "s"}</span></div></div>
      <div class="agent-policy-guide">
        <div><b>1</b><span><strong>View disagreement</strong>Some views are above and some below the frozen threshold.</span></div>
        <div><b>2</b><span><strong>Borderline mean</strong>The study score is within ±5 percentage points of the frozen threshold.</span></div>
        <div><b>3</b><span><strong>Abnormal mean</strong>The study mean score crosses the frozen threshold.</span></div>
      </div>
      <p class="agent-policy-note"><b>What happens then?</b> The app runs local TTA stability and Grad-CAM evidence summaries. A consistent normal case receives a trace but no optional tools. CrewAI is a separate, opt-in research explanation and is shown independently for each case.</p>
      <div class="agent-report-list">${ordered.map(agentReportRow).join("")}</div>
    </section>`;
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
    renderWorklist();
    showToast("This worklist is already reviewed. Its report is shown below the prioritized queue.");
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
