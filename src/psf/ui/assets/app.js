const state = {
  view: "benchmarks",
  benchmarks: [],
  benchmarkDetail: null,
  systems: [],
  systemPath: null,
  systemDraft: null,
  graph: null,
  selectedNodeId: null,
  selectedEdgeId: null,
  connecting: false,
  connectSource: null,
  positions: {},
};

const els = {
  statusText: document.getElementById("status-text"),
  pathInfo: document.getElementById("path-info"),
  benchmarksList: document.getElementById("benchmarks-list"),
  benchmarkDetail: document.getElementById("benchmark-detail"),
  systemSelect: document.getElementById("system-select"),
  graph: document.getElementById("graph"),
  inspector: document.getElementById("inspector"),
  yamlEditor: document.getElementById("yaml-editor"),
  yamlTextarea: document.getElementById("yaml-textarea"),
};

const navButtons = Array.from(document.querySelectorAll(".nav-btn"));

const EDGE_COLORS = {
  on_success: "#1f7a8c",
  on_failure: "#e0684f",
  on_loop_exhausted: "#d1a62b",
  loop: "#6e655a",
};

function setStatus(text, tone = "idle") {
  els.statusText.textContent = text;
  const dot = document.querySelector(".status-dot");
  if (tone === "error") {
    dot.style.background = "#e0684f";
  } else if (tone === "busy") {
    dot.style.background = "#d1a62b";
  } else {
    dot.style.background = "#1f7a8c";
  }
}

async function fetchJSON(url, options = {}) {
  const res = await fetch(url, options);
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.error || "Request failed");
  }
  return data;
}

function escapeHtml(text) {
  return String(text || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function formatPercent(value) {
  if (typeof value !== "number") return "-";
  return `${(value * 100).toFixed(1)}%`;
}

function prettyJSON(value) {
  return JSON.stringify(value ?? {}, null, 2);
}

function parseMaybeJson(value) {
  const trimmed = String(value || "").trim();
  if (!trimmed) return "";
  try {
    return JSON.parse(trimmed);
  } catch {
    return trimmed;
  }
}

function showView(name) {
  state.view = name;
  navButtons.forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.view === name);
  });
  document.querySelectorAll(".view").forEach((view) => {
    view.classList.toggle("active", view.id === `view-${name}`);
  });
  if (name === "builder" && state.graph?.network) {
    requestAnimationFrame(() => {
      state.graph.network.setSize();
      state.graph.network.redraw();
      if (state.graph.nodes.get().length) {
        state.graph.network.fit({ animation: { duration: 300, easingFunction: "easeInOutQuad" } });
      }
    });
  }
}

async function loadInfo() {
  try {
    const data = await fetchJSON("/api/info");
    els.pathInfo.textContent = `runs: ${data.runs_dir}\nsystems: ${data.systems_dir}`;
  } catch (err) {
    setStatus(err.message, "error");
  }
}

async function loadBenchmarks() {
  setStatus("Loading benchmarks...", "busy");
  try {
    const data = await fetchJSON("/api/benchmarks");
    state.benchmarks = data.benchmarks || [];
    renderBenchmarksList();
    setStatus("Benchmarks loaded", "idle");
  } catch (err) {
    setStatus(err.message, "error");
  }
}

function renderBenchmarksList() {
  if (!state.benchmarks.length) {
    els.benchmarksList.innerHTML = '<div class="empty">No benchmarks found.</div>';
    return;
  }
  els.benchmarksList.innerHTML = "";
  state.benchmarks.forEach((bench) => {
    const item = document.createElement("div");
    item.className = "list-item";
    if (state.benchmarkDetail && state.benchmarkDetail.path === bench.dir) {
      item.classList.add("active");
    }
    item.innerHTML = `
      <div><strong>${escapeHtml(bench.benchmark_id)}</strong> <span class="badge">${escapeHtml(bench.type)}</span></div>
      <div class="meta">${escapeHtml(bench.started_at || bench.created_at || "-")}</div>
    `;
    item.addEventListener("click", () => loadBenchmarkDetail(bench.dir));
    els.benchmarksList.appendChild(item);
  });
}

async function loadBenchmarkDetail(path) {
  setStatus("Loading benchmark...", "busy");
  try {
    const data = await fetchJSON(`/api/benchmark?path=${encodeURIComponent(path)}`);
    state.benchmarkDetail = data;
    renderBenchmarksList();
    renderBenchmarkDetail();
    setStatus("Benchmark loaded", "idle");
  } catch (err) {
    setStatus(err.message, "error");
  }
}

function renderBenchmarkDetail() {
  if (!state.benchmarkDetail) {
    els.benchmarkDetail.innerHTML = '<div class="empty">Select a benchmark to inspect.</div>';
    return;
  }
  const detail = state.benchmarkDetail;
  const suite = detail.suite_summary;
  const summary = detail.summary;

  let html = "";
  if (suite) {
    const systems = suite.systems || [];
    const systemCards = systems
      .map(
        (sys) => `
          <div class="detail-card">
          <div class="label">${escapeHtml(sys.system_label || sys.system_id)}</div>
          <div>${formatPercent(sys.overall_accuracy)} accuracy</div>
        </div>
      `
      )
      .join("");
    html += `
      <div class="detail-section">
        <div class="detail-grid">${systemCards}</div>
        <div>
          <div class="label">Suite Summary</div>
          <pre class="code-block">${escapeHtml(prettyJSON(suite))}</pre>
        </div>
      </div>
    `;
  }

  if (summary) {
    const generatorStats = summary.generators || {};
    const validatorStats = summary.validators || {};
    const renderStats = (stats) =>
      Object.entries(stats)
        .map(
          ([id, info]) => `
        <div class="detail-card">
          <div class="label">${escapeHtml(id)}</div>
          <div>legal rate: ${formatPercent(info.oracle_legal_rate)}</div>
          <div>parse rate: ${formatPercent(info.oracle_parse_rate)}</div>
        </div>
      `
        )
        .join("");
    html += `
      <div class="detail-section">
        <div class="detail-grid">${renderStats(generatorStats)}</div>
        <div class="detail-grid">${renderStats(validatorStats)}</div>
        <div>
          <div class="label">Random Suite Summary</div>
          <pre class="code-block">${escapeHtml(prettyJSON(summary))}</pre>
        </div>
      </div>
    `;
  }

  if (detail.systems && detail.systems.length) {
    const systemsList = detail.systems
      .map((system) => {
        const runCount = (system.runs || []).length;
        return `
        <div class="detail-card">
          <div class="label">${escapeHtml(system.system_id)}</div>
          <div>${system.benchmark_summary ? "summary available" : "no summary"}</div>
          <div>${runCount} runs</div>
        </div>
      `;
      })
      .join("");
    html += `
      <div class="detail-section">
        <div class="detail-grid">${systemsList}</div>
      </div>
    `;
  }

  els.benchmarkDetail.innerHTML = html || '<div class="empty">No benchmark details.</div>';
}

async function loadSystems() {
  try {
    const data = await fetchJSON("/api/systems");
    state.systems = data.systems || [];
    renderSystemSelect();
  } catch (err) {
    setStatus(err.message, "error");
  }
}

function renderSystemSelect() {
  els.systemSelect.innerHTML = "";
  if (!state.systems.length) {
    const opt = document.createElement("option");
    opt.textContent = "No systems found";
    els.systemSelect.appendChild(opt);
    return;
  }
  state.systems.forEach((system) => {
    const opt = document.createElement("option");
    opt.value = system.path;
    opt.textContent = system.path;
    els.systemSelect.appendChild(opt);
  });
  if (state.systemPath) {
    els.systemSelect.value = state.systemPath;
  }
}

function normalizeSystem(system) {
  const draft = JSON.parse(JSON.stringify(system || {}));
  if (!draft.system_input) draft.system_input = { fields: {} };
  if (!draft.system_input.fields) draft.system_input.fields = {};
  if (!draft.system_output) draft.system_output = { fields: {} };
  if (!draft.system_output.fields) draft.system_output.fields = {};
  if (!draft.nodes) draft.nodes = [];
  draft.nodes.forEach((node) => {
    if (!node.input_bindings) node.input_bindings = {};
    if (!node.routing) node.routing = { on_success: [], on_failure: [], on_loop_exhausted: [] };
    if (!node.routing.on_success) node.routing.on_success = [];
    if (!node.routing.on_failure) node.routing.on_failure = [];
    if (!node.routing.on_loop_exhausted) node.routing.on_loop_exhausted = [];
  });
  return draft;
}

function initGraph() {
  if (state.graph) return;
  const nodes = new vis.DataSet();
  const edges = new vis.DataSet();
  const network = new vis.Network(
    els.graph,
    { nodes, edges },
    {
      nodes: {
        shape: "box",
        font: { face: "Space Grotesk", size: 14 },
        color: {
          background: "#f9f3ea",
          border: "#1f7a8c",
        },
        margin: 10,
      },
      edges: {
        arrows: "to",
        font: { align: "top", size: 11, face: "IBM Plex Mono" },
        color: { inherit: false },
      },
      physics: { stabilization: false },
    }
  );

  network.on("selectNode", (params) => {
    state.selectedNodeId = params.nodes[0];
    state.selectedEdgeId = null;
    renderInspector();
  });

  network.on("selectEdge", (params) => {
    state.selectedEdgeId = params.edges[0];
    state.selectedNodeId = null;
    renderInspector();
  });

  network.on("click", (params) => {
    if (!state.connecting) return;
    if (!params.nodes.length) return;
    const target = params.nodes[0];
    if (!state.connectSource) {
      state.connectSource = target;
      setStatus(`Select target for ${target}`, "busy");
      return;
    }
    if (state.connectSource === target) {
      return;
    }
    addRoutingEdge(state.connectSource, target, document.getElementById("edge-type").value);
    state.connecting = false;
    state.connectSource = null;
    document.getElementById("connect-mode").classList.remove("active");
    setStatus("Edge created", "idle");
  });

  network.on("click", (params) => {
    if (state.connecting) {
      return;
    }
    if (!params.nodes.length && !params.edges.length) {
      showSystemInspector();
    }
  });

  network.on("dragEnd", () => {
    if (!state.systemPath) return;
    const positions = network.getPositions();
    state.positions = positions;
    const key = `psf-ui-positions:${state.systemPath}`;
    localStorage.setItem(key, JSON.stringify(positions));
  });

  state.graph = { network, nodes, edges };
}

function loadPositions() {
  if (!state.systemPath) {
    state.positions = {};
    return;
  }
  const key = `psf-ui-positions:${state.systemPath}`;
  try {
    state.positions = JSON.parse(localStorage.getItem(key) || "{}");
  } catch {
    state.positions = {};
  }
}

function refreshGraph() {
  if (!state.systemDraft) return;
  initGraph();
  loadPositions();
  const { nodes, edges } = state.graph;
  nodes.clear();
  edges.clear();

  state.systemDraft.nodes.forEach((node) => {
    const pos = state.positions[node.id];
    nodes.add({
      id: node.id,
      label: node.label ? `${node.id}\n${node.label}` : node.id,
      borderWidth: node.id === state.systemDraft.entry_node_id ? 3 : 1,
      color: {
        border: node.id === state.systemDraft.entry_node_id ? "#e0684f" : "#1f7a8c",
        background: node.terminal ? "#f4ede0" : "#f9f3ea",
      },
      x: pos ? pos.x : undefined,
      y: pos ? pos.y : undefined,
    });

    const routing = node.routing || {};
    ["on_success", "on_failure", "on_loop_exhausted"].forEach((type) => {
      (routing[type] || []).forEach((rule, idx) => {
        edges.add({
          id: `${node.id}:${type}:${idx}`,
          from: node.id,
          to: rule.next,
          label: rule.condition ? `${type}` : type,
          color: { color: EDGE_COLORS[type] },
          data: { type, index: idx, from: node.id, to: rule.next },
        });
      });
    });

    if (node.loop_control && node.loop_control.loop_target_node) {
      edges.add({
        id: `${node.id}:loop`,
        from: node.id,
        to: node.loop_control.loop_target_node,
        label: "loop",
        dashes: true,
        color: { color: EDGE_COLORS.loop },
        data: { type: "loop", from: node.id, to: node.loop_control.loop_target_node },
      });
    }
  });

  state.graph.network.redraw();
  state.graph.network.setSize();
  if (nodes.get().length) {
    state.graph.network.fit({ animation: { duration: 300, easingFunction: "easeInOutQuad" } });
  }
}

function addRoutingEdge(from, to, type) {
  const node = state.systemDraft.nodes.find((n) => n.id === from);
  if (!node) return;
  if (!node.routing) node.routing = { on_success: [], on_failure: [], on_loop_exhausted: [] };
  node.routing[type] = node.routing[type] || [];
  node.routing[type].push({ next: to });
  refreshGraph();
  renderInspector();
}

function removeRoutingEdge(edge) {
  const meta = edge.data || {};
  const node = state.systemDraft.nodes.find((n) => n.id === meta.from);
  if (!node || !node.routing || !meta.type || meta.type === "loop") return;
  node.routing[meta.type].splice(meta.index, 1);
  refreshGraph();
  renderInspector();
}

function renderInspector() {
  if (!state.systemDraft) {
    els.inspector.innerHTML = '<div class="empty">Load a system to start editing.</div>';
    return;
  }
  const systemButton = document.getElementById("show-system");
  if (systemButton) {
    systemButton.classList.toggle("active", !state.selectedNodeId && !state.selectedEdgeId);
  }
  if (state.selectedEdgeId) {
    const edge = state.graph.edges.get(state.selectedEdgeId);
    renderEdgeInspector(edge);
    return;
  }
  if (state.selectedNodeId) {
    const node = state.systemDraft.nodes.find((n) => n.id === state.selectedNodeId);
    renderNodeInspector(node);
    return;
  }
  renderSystemInspector();
}

function showSystemInspector() {
  state.selectedNodeId = null;
  state.selectedEdgeId = null;
  if (state.graph?.network) {
    state.graph.network.unselectAll();
  }
  renderInspector();
}

function renderSystemInspector() {
  const sys = state.systemDraft;
  const inputFields = Object.entries(sys.system_input.fields || {});
  const outputFields = Object.entries(sys.system_output.fields || {});

  const inputRows = inputFields
    .map(
      ([name, field]) => `
      <div class="table-row" data-input-key="${escapeHtml(name)}">
        <input data-input-prop="name" value="${escapeHtml(name)}" />
        <select data-input-prop="type">
          ${["string", "number", "boolean", "object", "array"]
            .map((opt) => `<option value="${opt}" ${field.type === opt ? "selected" : ""}>${opt}</option>`)
            .join("")}
        </select>
        <select data-input-prop="required">
          <option value="true" ${field.required ? "selected" : ""}>required</option>
          <option value="false" ${!field.required ? "selected" : ""}>optional</option>
        </select>
        <input data-input-prop="description" value="${escapeHtml(field.description || "")}" placeholder="description" />
        <button class="icon-btn" data-remove-input>x</button>
      </div>
    `
    )
    .join("");

  const outputRows = outputFields
    .map(
      ([name, field]) => `
      <div class="table-row" data-output-key="${escapeHtml(name)}">
        <input data-output-prop="name" value="${escapeHtml(name)}" />
        <input data-output-prop="from" value="${escapeHtml(field.from || "")}" placeholder="node.final.output" />
        <input data-output-prop="description" value="${escapeHtml(field.description || "")}" placeholder="description" />
        <div></div>
        <button class="icon-btn" data-remove-output>x</button>
      </div>
    `
    )
    .join("");

  els.inspector.innerHTML = `
    <div class="inspector-section">
      <h3>System Details</h3>
      <div class="row">
        <div class="field"><label>ID</label><input id="sys-id" value="${escapeHtml(sys.id || "")}" /></div>
        <div class="field"><label>Label</label><input id="sys-label" value="${escapeHtml(sys.label || "")}" /></div>
      </div>
      <div class="row">
        <div class="field"><label>Version</label><input id="sys-version" value="${escapeHtml(sys.version || "")}" /></div>
        <div class="field"><label>Entry Node</label><input id="sys-entry" value="${escapeHtml(sys.entry_node_id || "")}" /></div>
      </div>
    </div>
    <div class="inspector-section">
      <h3>System Input</h3>
      <div class="table">${inputRows || '<div class="empty">No inputs defined.</div>'}</div>
      <button class="btn" id="add-input">Add Input</button>
    </div>
    <div class="inspector-section">
      <h3>System Output</h3>
      <div class="table">${outputRows || '<div class="empty">No outputs defined.</div>'}</div>
      <button class="btn" id="add-output">Add Output</button>
    </div>
    <div class="inspector-section">
      <h3>Test Runner</h3>
      <div class="field"><label>Input JSON</label><textarea id="test-input">{}</textarea></div>
      <div class="mini-row">
        <button class="btn" id="validate-system">Validate</button>
        <button class="btn" id="dry-run">Dry Run</button>
        <button class="btn primary" id="run-system">Run</button>
      </div>
      <div class="field"><label>Benchmark Inputs Dir</label><input id="bench-inputs" placeholder="examples/input" /></div>
      <div class="mini-row">
        <input id="bench-iterations" type="number" value="5" />
        <input id="bench-expect" value="expected_legality" />
        <input id="bench-output" value="is_legal" />
      </div>
      <button class="btn" id="run-benchmark">Run Benchmark</button>
      <div id="test-status" class="meta"></div>
    </div>
  `;

  document.getElementById("sys-id").addEventListener("input", (e) => {
    sys.id = e.target.value;
  });
  document.getElementById("sys-label").addEventListener("input", (e) => {
    sys.label = e.target.value || undefined;
  });
  document.getElementById("sys-version").addEventListener("input", (e) => {
    sys.version = e.target.value || undefined;
  });
  document.getElementById("sys-entry").addEventListener("input", (e) => {
    sys.entry_node_id = e.target.value;
    refreshGraph();
  });

  document.getElementById("add-input").addEventListener("click", () => {
    const name = `input_${Object.keys(sys.system_input.fields).length + 1}`;
    sys.system_input.fields[name] = { type: "string", required: false };
    renderInspector();
  });

  document.getElementById("add-output").addEventListener("click", () => {
    const name = `output_${Object.keys(sys.system_output.fields).length + 1}`;
    sys.system_output.fields[name] = { from: "node.final.output" };
    renderInspector();
  });

  els.inspector.querySelectorAll("[data-remove-input]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const row = btn.closest(".table-row");
      const key = row.dataset.inputKey;
      delete sys.system_input.fields[key];
      renderInspector();
    });
  });

  els.inspector.querySelectorAll("[data-remove-output]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const row = btn.closest(".table-row");
      const key = row.dataset.outputKey;
      delete sys.system_output.fields[key];
      renderInspector();
    });
  });

  els.inspector.querySelectorAll("[data-input-prop]").forEach((input) => {
    input.addEventListener("input", () => {
      const row = input.closest(".table-row");
      const key = row.dataset.inputKey;
      const prop = input.dataset.inputProp;
      if (prop === "name") {
        const next = input.value.trim();
        if (!next || next === key) return;
        sys.system_input.fields[next] = sys.system_input.fields[key];
        delete sys.system_input.fields[key];
        row.dataset.inputKey = next;
      } else if (prop === "required") {
        sys.system_input.fields[key].required = input.value === "true";
      } else {
        sys.system_input.fields[key][prop] = input.value;
      }
    });
  });

  els.inspector.querySelectorAll("[data-output-prop]").forEach((input) => {
    input.addEventListener("input", () => {
      const row = input.closest(".table-row");
      const key = row.dataset.outputKey;
      const prop = input.dataset.outputProp;
      if (prop === "name") {
        const next = input.value.trim();
        if (!next || next === key) return;
        sys.system_output.fields[next] = sys.system_output.fields[key];
        delete sys.system_output.fields[key];
        row.dataset.outputKey = next;
      } else {
        sys.system_output.fields[key][prop] = input.value;
      }
    });
  });

  document.getElementById("validate-system").addEventListener("click", async () => {
    const status = document.getElementById("test-status");
    if (!state.systemPath) {
      status.textContent = "Save the system before validating.";
      return;
    }
    status.textContent = "Validating...";
    try {
      const res = await fetchJSON("/api/validate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: state.systemPath }),
      });
      status.textContent = res.ok ? "Valid" : `Error: ${res.error}`;
    } catch (err) {
      status.textContent = err.message;
    }
  });

  document.getElementById("dry-run").addEventListener("click", async () => {
    const status = document.getElementById("test-status");
    if (!state.systemPath) {
      status.textContent = "Save the system before running.";
      return;
    }
    status.textContent = "Dry run...";
    try {
      await fetchJSON("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: state.systemPath, dry_run: true }),
      });
      status.textContent = "Dry run complete";
    } catch (err) {
      status.textContent = err.message;
    }
  });

  document.getElementById("run-system").addEventListener("click", async () => {
    const status = document.getElementById("test-status");
    if (!state.systemPath) {
      status.textContent = "Save the system before running.";
      return;
    }
    status.textContent = "Running...";
    const inputText = document.getElementById("test-input").value;
    let payload = {};
    try {
      payload = JSON.parse(inputText);
    } catch {
      try {
        payload = jsyaml.load(inputText) || {};
      } catch (err) {
        status.textContent = `Invalid input: ${err.message}`;
        return;
      }
    }
    try {
      const res = await fetchJSON("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: state.systemPath, input: payload }),
      });
      if (!res.ok) {
        status.textContent = `Error: ${res.error}`;
        return;
      }
      status.textContent = `Run ${res.status || "done"} saved to ${res.run_dir}.`;
    } catch (err) {
      status.textContent = err.message;
    }
  });

  document.getElementById("run-benchmark").addEventListener("click", async () => {
    const status = document.getElementById("test-status");
    if (!state.systemPath) {
      status.textContent = "Save the system before benchmarking.";
      return;
    }
    status.textContent = "Running benchmark...";
    const inputsDir = document.getElementById("bench-inputs").value;
    const iterations = Number(document.getElementById("bench-iterations").value || 5);
    const expectField = document.getElementById("bench-expect").value || "expected_legality";
    const outputField = document.getElementById("bench-output").value || "is_legal";
    try {
      const res = await fetchJSON("/api/benchmark", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          systems: [state.systemPath],
          inputs_dir: inputsDir,
          iterations,
          expect_field: expectField,
          output_field: outputField,
        }),
      });
      status.textContent = res.ok ? "Benchmark complete" : `Benchmark error: ${res.error}`;
    } catch (err) {
      status.textContent = err.message;
    }
  });
}

function renderNodeInspector(node) {
  if (!node) {
    els.inspector.innerHTML = '<div class="empty">Node not found.</div>';
    return;
  }
  const isSystemNode = Boolean(node.use_system);
  const bindings = Object.entries(node.input_bindings || {});
  const outputSchemaText = node.output_schema ? JSON.stringify(node.output_schema, null, 2) : "";
  const outputMode = node.output_mode || "raw";

  const bindingRows = bindings
    .map(
      ([name, binding]) => `
      <div class="table-row" data-bind-key="${escapeHtml(name)}">
        <input data-bind-prop="name" value="${escapeHtml(name)}" />
        <select data-bind-prop="kind">
          <option value="from" ${binding.from ? "selected" : ""}>from</option>
          <option value="value" ${binding.value !== undefined ? "selected" : ""}>value</option>
        </select>
        <input data-bind-prop="value" value="${escapeHtml(binding.from || binding.value || "")}" placeholder="system.input" />
        <select data-bind-prop="required">
          <option value="true" ${binding.required !== false ? "selected" : ""}>required</option>
          <option value="false" ${binding.required === false ? "selected" : ""}>optional</option>
        </select>
        <button class="icon-btn" data-remove-bind>x</button>
      </div>
    `
    )
    .join("");

  els.inspector.innerHTML = `
    <div class="inspector-section">
      <h3>Node Details</h3>
      <div class="row">
        <div class="field"><label>ID</label><input id="node-id" value="${escapeHtml(node.id)}" /></div>
        <div class="field"><label>Label</label><input id="node-label" value="${escapeHtml(node.label || "")}" /></div>
      </div>
      <div class="row">
        <div class="field"><label>Node Kind</label>
          <select id="node-kind">
            <option value="llm" ${!isSystemNode ? "selected" : ""}>llm</option>
            <option value="system" ${isSystemNode ? "selected" : ""}>system</option>
          </select>
        </div>
        <div class="field"><label>Terminal</label>
          <select id="node-terminal">
            <option value="false" ${!node.terminal ? "selected" : ""}>false</option>
            <option value="true" ${node.terminal ? "selected" : ""}>true</option>
          </select>
        </div>
      </div>
      <div class="field"><label>Max Retries</label><input id="node-retries" type="number" value="${node.max_retries || 0}" /></div>
    </div>
    <div class="inspector-section">
      <h3>Node Inputs</h3>
      <div class="table">${bindingRows || '<div class="empty">No bindings.</div>'}</div>
      <button class="btn" id="add-binding">Add Binding</button>
    </div>
    <div class="inspector-section" id="node-system" style="${isSystemNode ? "" : "display:none"}">
      <h3>System Node</h3>
      <div class="field"><label>use_system Path</label><input id="node-use-system" value="${escapeHtml(node.use_system || "")}" /></div>
      <div class="meta">Output uses the subsystem final_output mapping.</div>
    </div>
    <div class="inspector-section" id="node-prompt" style="${isSystemNode ? "display:none" : ""}">
      <h3>Prompt</h3>
      <div class="field"><label>Instructions</label><textarea id="node-instructions">${escapeHtml(node.instructions || "")}</textarea></div>
      <div class="field"><label>Prompt Template</label><textarea id="node-prompt-template">${escapeHtml(node.prompt_template || "")}</textarea></div>
    </div>
    <div class="inspector-section" id="node-output" style="${isSystemNode ? "display:none" : ""}">
      <h3>Output</h3>
      <div class="field"><label>Output Mode</label>
        <select id="node-output-mode">
          ${["raw", "structured", "both"].map((mode) => `<option value="${mode}" ${outputMode === mode ? "selected" : ""}>${mode}</option>`).join("")}
        </select>
      </div>
      <div class="field"><label>Output Schema (JSON)</label><textarea id="node-output-schema">${escapeHtml(outputSchemaText)}</textarea></div>
    </div>
    <div class="inspector-section" id="node-llm" style="${isSystemNode ? "display:none" : ""}">
      <h3>Model</h3>
      <div class="row">
        <div class="field"><label>Provider</label><input id="node-provider" value="${escapeHtml(node.provider || "openai")}" /></div>
        <div class="field"><label>Model</label><input id="node-model" value="${escapeHtml(node.model || "")}" /></div>
      </div>
      <div class="row">
        <div class="field"><label>Temperature</label><input id="node-temp" type="number" step="0.1" value="${node.temperature ?? ""}" /></div>
        <div class="field"><label>Max Tokens</label><input id="node-max" type="number" value="${node.max_tokens ?? ""}" /></div>
      </div>
    </div>
    <div class="inspector-section">
      <h3>Loop Control</h3>
      <div class="row">
        <div class="field"><label>Counter Key</label><input id="loop-key" value="${escapeHtml(node.loop_control?.counter_key || "")}" /></div>
        <div class="field"><label>Max Iterations</label><input id="loop-max" type="number" value="${node.loop_control?.max_iterations || ""}" /></div>
      </div>
      <div class="row">
        <div class="field"><label>Loop Target</label><input id="loop-target" value="${escapeHtml(node.loop_control?.loop_target_node || "")}" /></div>
        <div class="field"><label>Exit On Success</label>
          <select id="loop-exit">
            <option value="true" ${node.loop_control?.exit_on_success !== false ? "selected" : ""}>true</option>
            <option value="false" ${node.loop_control?.exit_on_success === false ? "selected" : ""}>false</option>
          </select>
        </div>
      </div>
    </div>
  `;

  document.getElementById("node-id").addEventListener("input", (e) => {
    const next = e.target.value.trim();
    if (!next || next === node.id) return;
    renameNode(node.id, next);
  });
  document.getElementById("node-label").addEventListener("input", (e) => {
    node.label = e.target.value || undefined;
    refreshGraph();
  });
  document.getElementById("node-kind").addEventListener("change", (e) => {
    setNodeKind(node, e.target.value);
    refreshGraph();
    renderInspector();
  });
  document.getElementById("node-terminal").addEventListener("change", (e) => {
    node.terminal = e.target.value === "true";
    refreshGraph();
  });
  document.getElementById("node-retries").addEventListener("input", (e) => {
    node.max_retries = Number(e.target.value || 0);
  });

  if (!isSystemNode) {
    document.getElementById("node-provider").addEventListener("input", (e) => {
      node.provider = e.target.value || undefined;
    });
    document.getElementById("node-model").addEventListener("input", (e) => {
      node.model = e.target.value || undefined;
    });
    document.getElementById("node-temp").addEventListener("input", (e) => {
      node.temperature = e.target.value === "" ? undefined : Number(e.target.value);
    });
    document.getElementById("node-max").addEventListener("input", (e) => {
      node.max_tokens = e.target.value === "" ? undefined : Number(e.target.value);
    });
    document.getElementById("node-instructions").addEventListener("input", (e) => {
      node.instructions = e.target.value;
    });
    document.getElementById("node-prompt-template").addEventListener("input", (e) => {
      node.prompt_template = e.target.value;
    });
    document.getElementById("node-output-mode").addEventListener("change", (e) => {
      node.output_mode = e.target.value;
    });
    document.getElementById("node-output-schema").addEventListener("input", (e) => {
      try {
        node.output_schema = e.target.value.trim() ? JSON.parse(e.target.value) : undefined;
      } catch {
        // keep previous schema if invalid
      }
    });
  } else {
    document.getElementById("node-use-system").addEventListener("input", (e) => {
      node.use_system = e.target.value;
    });
  }

  document.getElementById("loop-key").addEventListener("input", (e) => {
    ensureLoop(node);
    node.loop_control.counter_key = e.target.value;
    refreshGraph();
  });
  document.getElementById("loop-max").addEventListener("input", (e) => {
    ensureLoop(node);
    node.loop_control.max_iterations = Number(e.target.value || 1);
    refreshGraph();
  });
  document.getElementById("loop-target").addEventListener("input", (e) => {
    ensureLoop(node);
    node.loop_control.loop_target_node = e.target.value;
    refreshGraph();
  });
  document.getElementById("loop-exit").addEventListener("change", (e) => {
    ensureLoop(node);
    node.loop_control.exit_on_success = e.target.value === "true";
  });

  document.getElementById("add-binding").addEventListener("click", () => {
    const name = `input_${Object.keys(node.input_bindings).length + 1}`;
    node.input_bindings[name] = { from: "system.input" };
    renderInspector();
  });

  els.inspector.querySelectorAll("[data-remove-bind]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const row = btn.closest(".table-row");
      const key = row.dataset.bindKey;
      delete node.input_bindings[key];
      renderInspector();
    });
  });

  els.inspector.querySelectorAll("[data-bind-prop]").forEach((input) => {
    input.addEventListener("input", () => {
      const row = input.closest(".table-row");
      const key = row.dataset.bindKey;
      const prop = input.dataset.bindProp;
      if (prop === "name") {
        const next = input.value.trim();
        if (!next || next === key) return;
        node.input_bindings[next] = node.input_bindings[key];
        delete node.input_bindings[key];
        row.dataset.bindKey = next;
      } else if (prop === "kind") {
        const current = node.input_bindings[key];
        if (input.value === "from") {
          current.from = current.from || "system.input";
          delete current.value;
        } else {
          current.value = current.value ?? "";
          delete current.from;
        }
      } else if (prop === "required") {
        node.input_bindings[key].required = input.value === "true";
      } else if (prop === "value") {
        const current = node.input_bindings[key];
        if (current.from !== undefined) {
          current.from = input.value;
        } else {
          current.value = parseMaybeJson(input.value);
        }
      }
    });
  });
}

function renderEdgeInspector(edge) {
  if (!edge) {
    els.inspector.innerHTML = '<div class="empty">Edge not found.</div>';
    return;
  }
  const meta = edge.data || {};
  const node = state.systemDraft.nodes.find((n) => n.id === meta.from);
  if (meta.type === "loop") {
    els.inspector.innerHTML = `
      <div class="inspector-section">
        <h3>Loop Edge</h3>
        <div class="detail-grid">
          <div class="detail-card"><div class="label">From</div><div>${escapeHtml(meta.from || "-")}</div></div>
          <div class="detail-card"><div class="label">To</div><div>${escapeHtml(meta.to || "-")}</div></div>
          <div class="detail-card"><div class="label">Type</div><div>loop</div></div>
        </div>
        <div class="meta">Loop routing is controlled in the node loop settings.</div>
        <button class="btn" id="loop-remove">Remove Loop</button>
      </div>
    `;
    document.getElementById("loop-remove").addEventListener("click", () => {
      if (node && node.loop_control) {
        node.loop_control = null;
        refreshGraph();
        renderInspector();
      }
    });
    return;
  }

  const rule = node?.routing?.[meta.type]?.[meta.index];
  const condition = rule?.condition || "";

  els.inspector.innerHTML = `
    <div class="inspector-section">
      <h3>Edge</h3>
      <div class="detail-grid">
        <div class="detail-card"><div class="label">From</div><div>${escapeHtml(meta.from || "-")}</div></div>
        <div class="detail-card"><div class="label">To</div><div>${escapeHtml(meta.to || "-")}</div></div>
        <div class="detail-card"><div class="label">Type</div><div>${escapeHtml(meta.type || "-")}</div></div>
      </div>
      <div class="field"><label>Condition</label><textarea id="edge-condition">${escapeHtml(condition)}</textarea></div>
      <div class="mini-row">
        <button class="btn primary" id="edge-save">Save Condition</button>
        <button class="btn" id="edge-delete">Delete Edge</button>
      </div>
    </div>
  `;

  document.getElementById("edge-save").addEventListener("click", () => {
    const value = document.getElementById("edge-condition").value.trim();
    if (rule) {
      rule.condition = value || undefined;
      refreshGraph();
    }
  });

  document.getElementById("edge-delete").addEventListener("click", () => {
    removeRoutingEdge(edge);
  });
}

function setNodeKind(node, kind) {
  if (kind === "system") {
    node.use_system = node.use_system || "";
    delete node.provider;
    delete node.model;
    delete node.temperature;
    delete node.max_tokens;
    delete node.extra_params;
    node.instructions = "";
    node.prompt_template = "";
    node.output_mode = "raw";
    delete node.output_schema;
  } else {
    delete node.use_system;
    node.provider = node.provider || "openai";
    node.model = node.model || "gpt-4o-mini";
    node.instructions = node.instructions || "Provide the next step.";
    node.prompt_template = node.prompt_template || "{{input}}";
    node.output_mode = node.output_mode || "raw";
  }
}

function ensureLoop(node) {
  if (!node.loop_control) {
    node.loop_control = {
      counter_key: "loop_counter",
      max_iterations: 2,
      loop_target_node: node.id,
      exit_on_success: true,
    };
  }
}

function renameNode(oldId, newId) {
  const node = state.systemDraft.nodes.find((n) => n.id === oldId);
  if (!node) return;
  node.id = newId;
  if (state.systemDraft.entry_node_id === oldId) {
    state.systemDraft.entry_node_id = newId;
  }
  state.systemDraft.nodes.forEach((n) => {
    const routing = n.routing || {};
    ["on_success", "on_failure", "on_loop_exhausted"].forEach((type) => {
      (routing[type] || []).forEach((rule) => {
        if (rule.next === oldId) rule.next = newId;
      });
    });
    if (n.loop_control?.loop_target_node === oldId) {
      n.loop_control.loop_target_node = newId;
    }
    Object.values(n.input_bindings || {}).forEach((binding) => {
      if (binding.from && binding.from.includes(`node.${oldId}`)) {
        binding.from = binding.from.replaceAll(`node.${oldId}`, `node.${newId}`);
      }
    });
  });
  Object.values(state.systemDraft.system_output.fields || {}).forEach((field) => {
    if (field.from && field.from.includes(`node.${oldId}`)) {
      field.from = field.from.replaceAll(`node.${oldId}`, `node.${newId}`);
    }
  });
  refreshGraph();
  renderInspector();
}

function addNode() {
  if (!state.systemDraft) {
    setStatus("Load or create a system before adding nodes.", "error");
    return;
  }
  const base = `node_${state.systemDraft.nodes.length + 1}`;
  let id = base;
  let idx = 1;
  while (state.systemDraft.nodes.find((n) => n.id === id)) {
    id = `${base}_${idx++}`;
  }
  const node = {
    id,
    label: "",
    provider: "openai",
    model: "gpt-4o-mini",
    instructions: "Describe the step output as JSON.",
    prompt_template: "{{input}}",
    input_bindings: {},
    output_mode: "raw",
    routing: { on_success: [], on_failure: [], on_loop_exhausted: [] },
    terminal: false,
    max_retries: 0,
  };
  state.systemDraft.nodes.push(node);
  if (!state.systemDraft.entry_node_id) state.systemDraft.entry_node_id = node.id;
  refreshGraph();
  renderInspector();
}

function deleteSelected() {
  if (state.selectedEdgeId) {
    const edge = state.graph.edges.get(state.selectedEdgeId);
    removeRoutingEdge(edge);
    state.selectedEdgeId = null;
    return;
  }
  if (state.selectedNodeId) {
    const id = state.selectedNodeId;
    state.systemDraft.nodes = state.systemDraft.nodes.filter((n) => n.id !== id);
    state.systemDraft.nodes.forEach((n) => {
      const routing = n.routing || {};
      ["on_success", "on_failure", "on_loop_exhausted"].forEach((type) => {
        routing[type] = (routing[type] || []).filter((rule) => rule.next !== id);
      });
      if (n.loop_control?.loop_target_node === id) {
        n.loop_control.loop_target_node = "";
      }
    });
    if (state.systemDraft.entry_node_id === id) {
      state.systemDraft.entry_node_id = state.systemDraft.nodes[0]?.id || "";
    }
    state.selectedNodeId = null;
    refreshGraph();
    renderInspector();
  }
}

async function loadSystem(path) {
  if (!path) return;
  setStatus("Loading system...", "busy");
  try {
    const data = await fetchJSON(`/api/system?path=${encodeURIComponent(path)}`);
    state.systemPath = data.path;
    state.systemDraft = normalizeSystem(data.system);
    state.selectedNodeId = null;
    state.selectedEdgeId = null;
    renderSystemSelect();
    refreshGraph();
    renderInspector();
    setStatus("System loaded", "idle");
  } catch (err) {
    setStatus(err.message, "error");
  }
}

async function saveSystem(path, system) {
  setStatus("Saving system...", "busy");
  try {
    const payload = { path, system };
    await fetchJSON("/api/system/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    setStatus("System saved", "idle");
    await loadSystems();
    state.systemPath = path;
  } catch (err) {
    setStatus(err.message, "error");
  }
}

function toggleYamlEditor(show) {
  els.yamlEditor.classList.toggle("hidden", !show);
  if (show) {
    els.yamlTextarea.value = jsyaml.dump(state.systemDraft, { sortKeys: false });
  }
}

function bindControls() {
  navButtons.forEach((btn) => {
    btn.addEventListener("click", () => showView(btn.dataset.view));
  });

  document.getElementById("refresh-benchmarks").addEventListener("click", loadBenchmarks);

  document.getElementById("system-load").addEventListener("click", () => {
    loadSystem(els.systemSelect.value);
  });

  document.getElementById("system-new").addEventListener("click", () => {
    const draft = {
      id: "new_system",
      entry_node_id: "",
      system_input: { fields: {} },
      system_output: { fields: {} },
      nodes: [],
    };
    state.systemPath = null;
    state.systemDraft = normalizeSystem(draft);
    refreshGraph();
    renderInspector();
  });

  document.getElementById("system-save").addEventListener("click", () => {
    if (!state.systemPath) {
      const next = prompt("Save system to path (relative to repo)", "systems/new_system.yaml");
      if (!next) return;
      saveSystem(next, state.systemDraft);
      return;
    }
    saveSystem(state.systemPath, state.systemDraft);
  });

  document.getElementById("system-save-as").addEventListener("click", () => {
    const next = prompt("Save system to path (relative to repo)", state.systemPath || "systems/new_system.yaml");
    if (!next) return;
    saveSystem(next, state.systemDraft);
  });

  document.getElementById("add-node").addEventListener("click", addNode);

  document.getElementById("connect-mode").addEventListener("click", (e) => {
    state.connecting = !state.connecting;
    state.connectSource = null;
    e.target.classList.toggle("active", state.connecting);
    setStatus(state.connecting ? "Select source node" : "Connect mode off", "busy");
  });

  document.getElementById("show-system").addEventListener("click", showSystemInspector);

  document.getElementById("delete-selected").addEventListener("click", deleteSelected);

  document.getElementById("toggle-yaml").addEventListener("click", () => {
    toggleYamlEditor(els.yamlEditor.classList.contains("hidden"));
  });

  document.getElementById("yaml-apply").addEventListener("click", () => {
    try {
      const next = jsyaml.load(els.yamlTextarea.value);
      state.systemDraft = normalizeSystem(next);
      refreshGraph();
      renderInspector();
      toggleYamlEditor(false);
    } catch (err) {
      setStatus(`YAML error: ${err.message}`, "error");
    }
  });

  document.getElementById("yaml-close").addEventListener("click", () => {
    toggleYamlEditor(false);
  });
}

async function init() {
  bindControls();
  await loadInfo();
  await Promise.all([loadBenchmarks(), loadSystems()]);
  showView("benchmarks");
  window.addEventListener("resize", () => {
    if (state.graph?.network) {
      state.graph.network.setSize();
    }
  });
}

init();
