/* ═══════════════════════════════════════════════════
   app.js — Requirement Ambiguity Checker
   Sections:
   1. Config
   2. State
   3. Navigation
   4. Single check
   5. Compare
   6. History
   7. Stats
   8. CSV Batch
   9. Result rendering
   10. Utilities
   11. Event listeners
════════════════════════════════════════════════════ */


/* ── 1. CONFIG ───────────────────────────────────────
   Update these URLs when your backends change.
   LLM backend is on port 3000
   ML backend is on port 3001 — plug it in when ready.
─────────────────────────────────────────────────── */
const BACKENDS = {
  llm: 'http://127.0.0.1:3000/check',
  ml:  'http://127.0.0.1:3001/check'
};


/* ── 2. STATE ────────────────────────────────────────
   All runtime data lives here — one object per tool.
─────────────────────────────────────────────────── */
const state = {
  llm: { history: [], stats: { total: 0, amb: 0, clear: 0 }, csvRows: [], batchResults: [], running: false },
  ml:  { history: [], stats: { total: 0, amb: 0, clear: 0 }, csvRows: [], batchResults: [], running: false }
};


/* ── 3. NAVIGATION ───────────────────────────────────
   goTo()     — show a page, hide the rest
   switchTab()— toggle single/batch tabs within a tool
─────────────────────────────────────────────────── */
function goTo(page) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active-nav'));
  document.getElementById('page-' + page).classList.add('active');
  document.getElementById('nav-' + page).classList.add('active-nav');
  window.scrollTo(0, 0);
}

function switchTab(tool, tab) {
  const prefix = tool + '-tab-';
  document.querySelectorAll(`[id^="${prefix}"]`).forEach(el => el.classList.remove('active'));
  document.getElementById(prefix + tab).classList.add('active');

  const tabs = document.querySelectorAll(`#page-${tool} .tab`);
  tabs.forEach((t, i) =>
    t.classList.toggle('active', (i === 0 && tab === 'single') || (i === 1 && tab === 'batch'))
  );
}


/* ── 4. SINGLE CHECK ─────────────────────────────────
   checkTool() — sends one requirement to the backend
   clearTool() — resets the input and result
─────────────────────────────────────────────────── */
async function checkTool(tool) {
  const req = document.getElementById(tool + '-input').value.trim();
  if (!req) return;

  setLoading(tool, true);
  hide(tool + '-result');
  hide(tool + '-error');

  try {
    const data = await callBackend(tool, req);
    showResult(tool + '-result', data, tool);
    addHistory(tool, req, data);
    updateStats(tool, data.verdict);
  } catch (e) {
    showError(tool + '-error', `Could not reach backend at ${BACKENDS[tool]}\nIs your server running?`);
  } finally {
    setLoading(tool, false);
  }
}

function clearTool(tool) {
  document.getElementById(tool + '-input').value = '';
  hide(tool + '-result');
  hide(tool + '-error');
}

function setLoading(tool, on) {
  document.getElementById(tool + '-loading').classList.toggle('visible', on);
  document.getElementById(tool + '-check-btn').disabled = on;
}


/* ── 5. COMPARE ──────────────────────────────────────
   Fires both backends in parallel via Promise.allSettled
   so one failure doesn't block the other.
─────────────────────────────────────────────────── */
async function runCompare() {
  const req = document.getElementById('compare-input').value.trim();
  if (!req) return;

  const btn = document.getElementById('compare-btn');
  btn.disabled = true;
  btn.textContent = 'Running...';

  document.getElementById('compare-llm-result').innerHTML = '<div class="compare-empty"><span class="spinner"></span> Analysing...</div>';
  document.getElementById('compare-ml-result').innerHTML  = '<div class="compare-empty"><span class="spinner ml"></span> Analysing...</div>';
  hide('agree-badge');

  const [llmRes, mlRes] = await Promise.allSettled([
    callBackend('llm', req),
    callBackend('ml',  req)
  ]);

  const llmData = llmRes.status === 'fulfilled' ? llmRes.value : null;
  const mlData  = mlRes.status  === 'fulfilled' ? mlRes.value  : null;

  document.getElementById('compare-llm-result').innerHTML =
    llmData ? buildResultHTML(llmData, 'llm') : errorHTML('LLM backend unreachable (localhost:3000)');
  document.getElementById('compare-ml-result').innerHTML =
    mlData  ? buildResultHTML(mlData, 'ml')   : errorHTML('ML backend unreachable (localhost:3001)');

  // Show agreement badge only if both responded
  if (llmData && mlData) {
    const badge = document.getElementById('agree-badge');
    const agree = llmData.verdict === mlData.verdict;
    badge.textContent = agree
      ? `✅ Both tools agree — ${llmData.verdict.toUpperCase()}`
      : `⚠️ Tools disagree — LLM says ${llmData.verdict.toUpperCase()}, ML says ${mlData.verdict.toUpperCase()}`;
    badge.className = 'agree-badge visible ' + (agree ? 'agree' : 'disagree');
  }

  btn.disabled = false;
  btn.textContent = 'Run both tools';
}

function clearCompare() {
  document.getElementById('compare-input').value = '';
  document.getElementById('compare-llm-result').innerHTML = '<div class="compare-empty">Result will appear here</div>';
  document.getElementById('compare-ml-result').innerHTML  = '<div class="compare-empty">Result will appear here</div>';
  hide('agree-badge');
}


/* ── 6. HISTORY ──────────────────────────────────────
   addHistory()     — prepend a result to the sidebar list
   renderHistory()  — re-renders the full history list
   restoreHistory() — click a history item to reload it
─────────────────────────────────────────────────── */
function addHistory(tool, req, data) {
  state[tool].history.unshift({ req, data, time: new Date() });
  renderHistory(tool);
}

function renderHistory(tool) {
  const list = document.getElementById(tool + '-history');
  const h = state[tool].history;

  if (!h.length) {
    list.innerHTML = '<div class="empty-state">No checks yet.</div>';
    return;
  }

  list.innerHTML = h.map((item, i) => {
    const t = item.time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    return `
      <div class="history-item" onclick="restoreHistory('${tool}', ${i})">
        <div class="h-text">${esc(item.req)}</div>
        <div class="history-meta">
          <span class="h-pill ${item.data.verdict}">${item.data.verdict.toUpperCase()}</span>
          <span class="h-time">${t}</span>
        </div>
      </div>`;
  }).join('');
}

function restoreHistory(tool, i) {
  const item = state[tool].history[i];
  document.getElementById(tool + '-input').value = item.req;
  showResult(tool + '-result', item.data, tool);
  hide(tool + '-error');
  switchTab(tool, 'single');
}


/* ── 7. STATS ────────────────────────────────────────
   Updates the three counter tiles at the top of each
   tool page after every successful check.
─────────────────────────────────────────────────── */
function updateStats(tool, verdict) {
  state[tool].stats.total++;
  verdict === 'ambiguous' ? state[tool].stats.amb++ : state[tool].stats.clear++;

  document.getElementById(tool + '-stat-total').textContent = state[tool].stats.total;
  document.getElementById(tool + '-stat-amb').textContent   = state[tool].stats.amb;
  document.getElementById(tool + '-stat-clear').textContent = state[tool].stats.clear;
}


/* ── 8. CSV BATCH ────────────────────────────────────
   handleCSV()   — reads uploaded file, populates column picker
   startBatch()  — loops through rows, calls backend per row
   resetCSV()    — clears upload state
   exportCSV()   — downloads results as a CSV file
─────────────────────────────────────────────────── */
function handleCSV(tool, file) {
  if (!file) return;

  const reader = new FileReader();
  reader.onload = e => {
    try {
      state[tool].csvRows = parseCSV(e.target.result);
      if (!state[tool].csvRows.length) throw new Error('No data rows found.');

      const headers = Object.keys(state[tool].csvRows[0]);
      const sel = document.getElementById(tool + '-col-select');
      sel.innerHTML = headers.map(h => `<option value="${esc(h)}">${esc(h)}</option>`).join('');

      document.getElementById(tool + '-csv-preview').textContent =
        `Loaded: ${file.name}  ·  ${state[tool].csvRows.length} rows  ·  ${headers.length} columns`;
      document.getElementById(tool + '-picker').classList.add('visible');
      resetBatchUI(tool);
      hide(tool + '-batch-error');
    } catch (e) {
      showError(tool + '-batch-error', 'Could not parse CSV: ' + e.message);
    }
  };
  reader.readAsText(file);
}

async function startBatch(tool) {
  if (state[tool].running) return;

  const col  = document.getElementById(tool + '-col-select').value;
  const rows = state[tool].csvRows.filter(r => r[col] && r[col].trim());
  if (!rows.length) { showError(tool + '-batch-error', 'No values found in that column.'); return; }

  state[tool].running = true;
  state[tool].batchResults = [];
  resetBatchUI(tool);

  const tbody = document.getElementById(tool + '-tbody');
  const total = rows.length;

  document.getElementById(tool + '-progress').classList.add('visible');
  document.getElementById(tool + '-table-wrap').classList.add('visible');
  document.getElementById(tool + '-export-row').style.display = 'none';

  for (let i = 0; i < total; i++) {
    const req = rows[i][col];

    // Update progress bar
    document.getElementById(tool + '-prog-text').textContent  = `Processing ${i + 1} of ${total}...`;
    document.getElementById(tool + '-prog-count').textContent = `${i} / ${total}`;
    document.getElementById(tool + '-prog-fill').style.width  = ((i / total) * 100) + '%';

    // Placeholder row while waiting
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="tbl-conf">${i + 1}</td>
      <td class="tbl-req">${esc(req)}</td>
      <td><span class="spinner" style="width:12px;height:12px;border-width:1.5px"></span></td>
      <td>—</td>
      <td>—</td>`;
    tbody.appendChild(tr);

    try {
      const data = await callBackend(tool, req);
      state[tool].batchResults.push({ req, data });
      updateStats(tool, data.verdict);

      const conf = Math.round((data.confidence || 0) * 100);
      const exp  = (data.reasons || [])[0] || '';
      tr.innerHTML = `
        <td class="tbl-conf">${i + 1}</td>
        <td class="tbl-req">${esc(req)}</td>
        <td><span class="tbl-pill ${data.verdict}">${data.verdict.toUpperCase()}</span></td>
        <td class="tbl-conf">${conf}%</td>
        <td style="font-size:12px;color:var(--muted);max-width:260px">${esc(exp)}</td>`;
    } catch (e) {
      state[tool].batchResults.push({ req, data: null });
      tr.innerHTML = `
        <td class="tbl-conf">${i + 1}</td>
        <td class="tbl-req">${esc(req)}</td>
        <td colspan="3" style="color:var(--accent);font-family:var(--font-mono);font-size:11px">Error — backend unreachable</td>`;
    }

    await sleep(300); // small delay to avoid overwhelming the backend
  }

  // Done
  document.getElementById(tool + '-prog-fill').style.width   = '100%';
  document.getElementById(tool + '-prog-text').textContent   = `Done — ${total} requirements analysed`;
  document.getElementById(tool + '-prog-count').textContent  = `${total} / ${total}`;
  document.getElementById(tool + '-export-row').style.display = 'flex';
  state[tool].running = false;
}

function resetCSV(tool) {
  state[tool].csvRows = [];
  state[tool].batchResults = [];
  state[tool].running = false;
  document.getElementById(tool + '-picker').classList.remove('visible');
  document.getElementById(tool + '-csv-preview').textContent = '';
  document.getElementById(tool + '-csv-input').value = '';
  resetBatchUI(tool);
  hide(tool + '-batch-error');
}

function resetBatchUI(tool) {
  document.getElementById(tool + '-tbody').innerHTML = '';
  document.getElementById(tool + '-table-wrap').classList.remove('visible');
  document.getElementById(tool + '-progress').classList.remove('visible');
  document.getElementById(tool + '-export-row').style.display = 'none';
}

function exportCSV(tool) {
  const results = state[tool].batchResults;
  if (!results.length) return;

  const headers = ['requirement', 'verdict', 'confidence', 'explanation'];
  const rows = results.map(r => r.data
    ? [r.req, r.data.verdict || '', Math.round((r.data.confidence || 0) * 100) + '%', (r.data.reasons || [])[0] || '']
    : [r.req, 'error', '', '']
  );

  const csv = [headers, ...rows]
    .map(row => row.map(cell => `"${String(cell).replace(/"/g, '""')}"`).join(','))
    .join('\n');

  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
  a.download = `${tool}_ambiguity_results.csv`;
  a.click();
}


/* ── 9. RESULT RENDERING ─────────────────────────────
   showResult()     — renders result into a panel element
   buildResultHTML()— builds the HTML string for a result
   errorHTML()      — small error message for compare panel
─────────────────────────────────────────────────── */
function showResult(id, data, tool) {
  const box = document.getElementById(id);
  box.innerHTML = buildResultHTML(data, tool);
  box.classList.add('visible');
}

function buildResultHTML(data, tool) {
  const conf      = Math.round((data.confidence || 0) * 100);
  const fillClass = data.verdict === 'ambiguous' ? 'ambiguous' : (tool === 'ml' ? 'ml' : '');
  const reasons   = (data.reasons || [])
    .map(r => `<li style="font-size:14px;color:var(--muted);margin-bottom:4px;line-height:1.5">${esc(r)}</li>`)
    .join('');

  return `
    <span class="verdict-pill ${data.verdict}">${data.verdict === 'ambiguous' ? 'AMBIGUOUS' : 'CLEAR'}</span>
    <div class="conf-label"><span>Confidence</span><span>${conf}%</span></div>
    <div class="conf-track"><div class="conf-fill ${fillClass}" style="width:${conf}%"></div></div>
    ${reasons   ? `<div class="section-label">Reasoning</div><ul style="padding-left:16px;margin-top:4px">${reasons}</ul>` : ''}
    ${data.suggested_rewrite ? `<div class="section-label">Suggested rewrite</div><div class="rewrite-box">${esc(data.suggested_rewrite)}</div>` : ''}
  `;
}

function errorHTML(msg) {
  return `<div style="font-family:var(--font-mono);font-size:12px;color:var(--accent);padding:20px;text-align:center">${msg}</div>`;
}


/* ── 10. UTILITIES ───────────────────────────────────
   callBackend() — fetch wrapper for the /check endpoint
   parseCSV()    — parse CSV text into array of objects
   splitLine()   — handle quoted commas in CSV correctly
   showError()   — display error message in any error div
   hide()        — remove .visible class from any element
   esc()         — escape HTML to prevent XSS
   sleep()       — promise-based delay for batch pacing
─────────────────────────────────────────────────── */
async function callBackend(tool, requirement) {
  const res = await fetch(BACKENDS[tool], {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ requirement })
  });
  if (!res.ok) throw new Error('Server error ' + res.status);
  return res.json();
}

function parseCSV(text) {
  text = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
  const records = [];
  let fields = [], cur = '', inQ = false, i = 0;
  while (i < text.length) {
    const c = text[i];
    if (inQ) {
      if (c === '"') {
        if (text[i + 1] === '"') { cur += '"'; i += 2; continue; } // escaped quote
        inQ = false;
      } else {
        cur += (c === '\n') ? ' ' : c; // newlines inside quotes become spaces
      }
    } else {
      if      (c === '"')  { inQ = true; }
      else if (c === ',')  { fields.push(cur.trim()); cur = ''; }
      else if (c === '\n') {
        fields.push(cur.trim()); cur = '';
        if (fields.some(f => f !== '')) records.push(fields);
        fields = [];
      } else { cur += c; }
    }
    i++;
  }
  // Handle last row
  if (cur.trim() || fields.length) {
    fields.push(cur.trim());
    if (fields.some(f => f !== '')) records.push(fields);
  }
  if (records.length < 2) return [];
  const headers = records[0];
  return records.slice(1).map(vals => {
    const row = {};
    headers.forEach((h, idx) => row[h.trim()] = (vals[idx] || '').trim());
    return row;
  });
}

function splitLine(line) {
  const result = [];
  let cur = '', inQ = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if      (c === '"')              inQ = !inQ;
    else if (c === ',' && !inQ)      { result.push(cur); cur = ''; }
    else                             cur += c;
  }
  result.push(cur);
  return result;
}

function showError(id, msg) {
  const el = document.getElementById(id);
  el.textContent = msg;
  el.classList.add('visible');
}

function hide(id) {
  document.getElementById(id).classList.remove('visible');
}

function esc(s) {
  return String(s || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}


/* ── 11. COMPARE BATCH ───────────────────────────────
   CSV batch for the compare page — runs both backends
   per row and shows results in a single merged table.
─────────────────────────────────────────────────── */

// Tab switcher for compare page
function switchCompareTab(tab) {
  document.querySelectorAll('[id^="compare-tab-"]').forEach(el => el.classList.remove('active'));
  document.getElementById('compare-tab-' + tab).classList.add('active');
  const tabs = document.querySelectorAll('#page-compare .tab');
  tabs.forEach((t, i) => t.classList.toggle('active', (i === 0 && tab === 'single') || (i === 1 && tab === 'batch')));
}

// Compare batch state
const compareState = { csvRows: [], batchResults: [], running: false };

function handleCompareCSV(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    try {
      compareState.csvRows = parseCSV(e.target.result);
      if (!compareState.csvRows.length) throw new Error('No data rows found.');
      const headers = Object.keys(compareState.csvRows[0]);
      const sel = document.getElementById('compare-col-select');
      sel.innerHTML = headers.map(h => `<option value="${esc(h)}">${esc(h)}</option>`).join('');
      document.getElementById('compare-csv-preview').textContent =
        `Loaded: ${file.name}  ·  ${compareState.csvRows.length} rows  ·  ${headers.length} columns`;
      document.getElementById('compare-picker').classList.add('visible');
      resetCompareBatchUI();
      hide('compare-batch-error');
    } catch (e) {
      showError('compare-batch-error', 'Could not parse CSV: ' + e.message);
    }
  };
  reader.readAsText(file);
}

async function startCompareBatch() {
  if (compareState.running) return;

  const col  = document.getElementById('compare-col-select').value;
  const rows = compareState.csvRows.filter(r => r[col] && r[col].trim());
  if (!rows.length) { showError('compare-batch-error', 'No values found in that column.'); return; }

  compareState.running = true;
  compareState.batchResults = [];
  resetCompareBatchUI();

  const tbody = document.getElementById('compare-tbody');
  const total = rows.length;

  document.getElementById('compare-progress').classList.add('visible');
  document.getElementById('compare-table-wrap').classList.add('visible');
  document.getElementById('compare-export-row').style.display = 'none';
  
  // ── Phase 1: Run all rows through LLM ──────────────────
  const llmResults = [];
  for (let i = 0; i < total; i++) {
    document.getElementById('compare-prog-text').textContent  = `LLM: processing ${i + 1} of ${total}...`;
    document.getElementById('compare-prog-count').textContent = `${i} / ${total}`;
    document.getElementById('compare-prog-fill').style.width  = ((i / total) * 50) + '%'; // 0-50%
    try {
      llmResults.push(await callBackend('llm', rows[i][col]));
    } catch(e) {
      llmResults.push(null);
    }
    await sleep(100);
  }

  // ── Phase 2: Run all rows through ML (fast) ────────────
  const mlResults = [];
  for (let i = 0; i < total; i++) {
    document.getElementById('compare-prog-text').textContent  = `ML: processing ${i + 1} of ${total}...`;
    document.getElementById('compare-prog-count').textContent = `${i} / ${total}`;
    document.getElementById('compare-prog-fill').style.width  = (50 + (i / total) * 50) + '%'; // 50-100%
    try {
      mlResults.push(await callBackend('ml', rows[i][col]));
    } catch(e) {
      mlResults.push(null);
    }
    await sleep(50); // ML is fast, shorter delay
  }

  // ── Phase 3: Merge and render table ───────────────────
  for (let i = 0; i < total; i++) {
    const req     = rows[i][col];
    const llmData = llmResults[i];
    const mlData  = mlResults[i];

    compareState.batchResults.push({ req, llmData, mlData });

    const llmVerdict = llmData ? llmData.verdict : 'error';
    const mlVerdict  = mlData  ? mlData.verdict  : 'error';
    const llmConf    = llmData ? Math.round((llmData.confidence || 0) * 100) + '%' : '—';
    const mlConf     = mlData  ? Math.round((mlData.confidence  || 0) * 100) + '%' : '—';
    const agree      = llmData && mlData && llmVerdict === mlVerdict;
    const agreeCell  = llmData && mlData
      ? (agree
          ? `<span style="color:var(--llm);font-weight:600">✓ Agree</span>`
          : `<span style="color:var(--amber);font-weight:600">✗ Disagree</span>`)
      : '—';
    const llmPill = llmData
      ? `<span class="tbl-pill ${llmVerdict}">${llmVerdict.toUpperCase()}</span>`
      : `<span style="color:var(--accent);font-family:var(--font-mono);font-size:10px">ERROR</span>`;
    const mlPill = mlData
      ? `<span class="tbl-pill ${mlVerdict}">${mlVerdict.toUpperCase()}</span>`
      : `<span style="color:var(--accent);font-family:var(--font-mono);font-size:10px">ERROR</span>`;

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="tbl-conf">${i + 1}</td>
      <td class="tbl-req">${esc(req)}</td>
      <td>${llmPill}</td>
      <td class="tbl-conf">${llmConf}</td>
      <td>${mlPill}</td>
      <td class="tbl-conf">${mlConf}</td>
      <td>${agreeCell}</td>`;
    tbody.appendChild(tr);
  }
  // for (let i = 0; i < total; i++) {
  //   const req = rows[i][col];

  //   // Progress
  //   document.getElementById('compare-prog-text').textContent  = `Processing ${i + 1} of ${total}...`;
  //   document.getElementById('compare-prog-count').textContent = `${i} / ${total}`;
  //   document.getElementById('compare-prog-fill').style.width  = ((i / total) * 100) + '%';

  //   // Placeholder row
  //   const tr = document.createElement('tr');
  //   tr.innerHTML = `
  //     <td class="tbl-conf">${i + 1}</td>
  //     <td class="tbl-req">${esc(req)}</td>
  //     <td colspan="5" style="text-align:center">
  //       <span class="spinner" style="width:12px;height:12px;border-width:1.5px"></span>
  //       Running both tools...
  //     </td>`;
  //   tbody.appendChild(tr);

  //   // Fire both backends in parallel
  //   const [llmRes, mlRes] = await Promise.allSettled([
  //     callBackend('llm', req),
  //     callBackend('ml',  req)
  //   ]);

  //   const llmData = llmRes.status === 'fulfilled' ? llmRes.value : null;
  //   const mlData  = mlRes.status  === 'fulfilled' ? mlRes.value  : null;

  //   compareState.batchResults.push({ req, llmData, mlData });

  //   // Build table row
  //   const llmVerdict  = llmData ? llmData.verdict : 'error';
  //   const mlVerdict   = mlData  ? mlData.verdict  : 'error';
  //   const llmConf     = llmData ? Math.round((llmData.confidence || 0) * 100) + '%' : '—';
  //   const mlConf      = mlData  ? Math.round((mlData.confidence  || 0) * 100) + '%' : '—';

  //   const agree       = llmData && mlData && llmVerdict === mlVerdict;
  //   const agreeCell   = llmData && mlData
  //     ? (agree
  //         ? `<span style="color:var(--llm);font-weight:600">✓ Agree</span>`
  //         : `<span style="color:var(--amber);font-weight:600">✗ Disagree</span>`)
  //     : '—';

  //   const llmPill = llmData
  //     ? `<span class="tbl-pill ${llmVerdict}">${llmVerdict.toUpperCase()}</span>`
  //     : `<span style="color:var(--accent);font-family:var(--font-mono);font-size:10px">BACKEND ERROR</span>`;

  //   const mlPill = mlData
  //     ? `<span class="tbl-pill ${mlVerdict}">${mlVerdict.toUpperCase()}</span>`
  //     : `<span style="color:var(--accent);font-family:var(--font-mono);font-size:10px">BACKEND ERROR</span>`;

  //   tr.innerHTML = `
  //     <td class="tbl-conf">${i + 1}</td>
  //     <td class="tbl-req">${esc(req)}</td>
  //     <td>${llmPill}</td>
  //     <td class="tbl-conf">${llmConf}</td>
  //     <td>${mlPill}</td>
  //     <td class="tbl-conf">${mlConf}</td>
  //     <td>${agreeCell}</td>`;

  //   await sleep(300);
  // }

  // Done
  document.getElementById('compare-prog-fill').style.width   = '100%';
  document.getElementById('compare-prog-text').textContent   = `Done — ${total} requirements analysed`;
  document.getElementById('compare-prog-count').textContent  = `${total} / ${total}`;
  document.getElementById('compare-export-row').style.display = 'flex';
  compareState.running = false;
}

function resetCompareBatchUI() {
  document.getElementById('compare-tbody').innerHTML = '';
  document.getElementById('compare-table-wrap').classList.remove('visible');
  document.getElementById('compare-progress').classList.remove('visible');
  document.getElementById('compare-export-row').style.display = 'none';
}

function resetCompareCSV() {
  compareState.csvRows = [];
  compareState.batchResults = [];
  compareState.running = false;
  document.getElementById('compare-picker').classList.remove('visible');
  document.getElementById('compare-csv-preview').textContent = '';
  document.getElementById('compare-csv-input').value = '';
  resetCompareBatchUI();
  hide('compare-batch-error');
}

function exportCompareCSV() {
  if (!compareState.batchResults.length) return;

  const headers = [
    'requirement',
    'llm_verdict', 'llm_confidence', 'llm_explanation',
    'ml_verdict',  'ml_confidence',
    'agreement'
  ];

  const rows = compareState.batchResults.map(r => {
    const llm  = r.llmData;
    const ml   = r.mlData;
    const agree = llm && ml ? (llm.verdict === ml.verdict ? 'agree' : 'disagree') : 'error';
    return [
      r.req,
      llm ? llm.verdict    : 'error',
      llm ? Math.round((llm.confidence || 0) * 100) + '%' : '',
      llm ? (llm.reasons || [])[0] || '' : '',
      ml  ? ml.verdict     : 'error',
      ml  ? Math.round((ml.confidence  || 0) * 100) + '%' : '',
      agree
    ];
  });

  const csv = [headers, ...rows]
    .map(row => row.map(cell => `"${String(cell).replace(/"/g, '""')}"`).join(','))
    .join('\n');

  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
  a.download = 'compare_results.csv';
  a.click();
}


/* ── 12. EVENT LISTENERS ─────────────────────────────
   Set up drag-and-drop and keyboard shortcuts for
   both tools and the compare page at page load.
─────────────────────────────────────────────────── */

// Ctrl+Enter to analyse in both tool textareas
['llm', 'ml'].forEach(tool => {
  document.getElementById(tool + '-input').addEventListener('keydown', e => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) checkTool(tool);
  });
});

// Drag-and-drop CSV for both tools
['llm', 'ml'].forEach(tool => {
  const dz = document.getElementById(tool + '-drop');
  dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('dragover'); });
  dz.addEventListener('dragleave', ()  => dz.classList.remove('dragover'));
  dz.addEventListener('drop', e => {
    e.preventDefault();
    dz.classList.remove('dragover');
    const f = e.dataTransfer.files[0];
    if (f && f.name.endsWith('.csv')) handleCSV(tool, f);
    else showError(tool + '-batch-error', 'Please drop a .csv file.');
  });
});

// Drag-and-drop CSV for compare page
const compareDZ = document.getElementById('compare-drop');
compareDZ.addEventListener('dragover', e => { e.preventDefault(); compareDZ.classList.add('dragover'); });
compareDZ.addEventListener('dragleave', () => compareDZ.classList.remove('dragover'));
compareDZ.addEventListener('drop', e => {
  e.preventDefault();
  compareDZ.classList.remove('dragover');
  const f = e.dataTransfer.files[0];
  if (f && f.name.endsWith('.csv')) handleCompareCSV(f);
  else showError('compare-batch-error', 'Please drop a .csv file.');
});
