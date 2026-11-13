const $ = id => document.getElementById(id);

async function api(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Something went wrong. Please try again.');
  return data;
}

function setActiveDataset(filename) {
  $('active-dataset').textContent = filename || 'No file selected';
}

async function selectFile(filename) {
  const data = await api('/api/active', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({filename})
  });
  setActiveDataset(data.active_dataset);
  await refreshFiles();
}

async function refreshFiles() {
  const data = await api('/api/files');
  setActiveDataset(data.active_dataset);
  if (!data.active_dataset) $('dataset-metadata').textContent = '';
  $('files').replaceChildren(...data.files.map(name => {
    const li = document.createElement('li');
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'file-button' + (name === data.active_dataset ? ' active' : '');
    button.textContent = name;
    button.addEventListener('click', () => selectFile(name));
    li.append(button);
    return li;
  }));
}

function appendInlineFormatting(container, text) {
  const pattern = /(\*\*[^*]+\*\*|\[[^\]]+\]\(https?:\/\/[^)\s]+\))/g;
  let position = 0;
  for (const match of text.matchAll(pattern)) {
    container.append(document.createTextNode(text.slice(position, match.index)));
    const token = match[0];
    if (token.startsWith('**')) {
      const strong = document.createElement('strong');
      strong.textContent = token.slice(2, -2);
      container.append(strong);
    } else {
      const parts = token.match(/^\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)$/);
      const link = document.createElement('a');
      link.textContent = parts[1];
      link.href = parts[2];
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      container.append(link);
    }
    position = match.index + token.length;
  }
  container.append(document.createTextNode(text.slice(position)));
}

function markdownCells(line) {
  return line.trim().replace(/^\||\|$/g, '').split('|').map(cell => cell.trim());
}

function isTableDivider(line) {
  const cells = markdownCells(line);
  return cells.length > 1 && cells.every(cell => /^:?-{3,}:?$/.test(cell));
}

function renderMarkdownTable(lines, start, container) {
  if (start + 1 >= lines.length || !lines[start].includes('|') || !isTableDivider(lines[start + 1])) {
    return 0;
  }
  const headers = markdownCells(lines[start]);
  const rows = [];
  let position = start + 2;
  while (position < lines.length && lines[position].includes('|') && lines[position].trim()) {
    rows.push(markdownCells(lines[position]));
    position += 1;
  }
  renderTable(rows.map(row => Object.fromEntries(headers.map((header, index) =>
    [header, row[index] || '']))), container);
  return position - start;
}

function renderAssistantText(text, container) {
  const lines = text.split(/\r?\n/);
  let list = null;
  for (let index = 0; index < lines.length;) {
    const line = lines[index].trim();
    if (!line) {
      list = null;
      index += 1;
      continue;
    }
    const tableLength = renderMarkdownTable(lines, index, container);
    if (tableLength) {
      list = null;
      index += tableLength;
      continue;
    }
    const bullet = line.match(/^[-*]\s+(.+)/);
    if (bullet) {
      if (!list) {
        list = document.createElement('ul');
        list.className = 'answer-list';
        container.append(list);
      }
      const item = document.createElement('li');
      appendInlineFormatting(item, bullet[1]);
      list.append(item);
    } else {
      list = null;
      const paragraph = document.createElement('p');
      appendInlineFormatting(paragraph, line);
      container.append(paragraph);
    }
    index += 1;
  }
}

function message(text, role = '') {
  const div = document.createElement('div');
  div.className = `message ${role}`;
  if (role === 'assistant') renderAssistantText(text, div);
  else div.textContent = text;
  $('messages').append(div);
  div.scrollIntoView({block: 'nearest'});
}

function readableLabel(value) {
  return String(value)
    .replace(/^normalized_/, '')
    .replace(/[_/]+/g, ' ')
    .replace(/\b\w/g, letter => letter.toUpperCase());
}

function readableValue(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return 'Missing';
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

function renderTable(rows, container, vertical = false) {
  if (!rows.length || typeof rows[0] !== 'object' || Array.isArray(rows[0])) return false;
  const displayedRows = vertical
    ? Object.entries(rows[0]).map(([detail, value]) => ({Detail: readableLabel(detail), Value: value}))
    : rows;
  const columns = [...new Set(displayedRows.flatMap(row => Object.keys(row)))];
  const wrapper = document.createElement('div');
  wrapper.className = 'table-wrapper';
  const table = document.createElement('table');
  table.className = 'result-table';
  const head = document.createElement('thead');
  const headerRow = document.createElement('tr');
  columns.forEach(column => {
    const th = document.createElement('th');
    th.textContent = readableLabel(column);
    headerRow.append(th);
  });
  head.append(headerRow);
  table.append(head);
  const body = document.createElement('tbody');
  displayedRows.slice(0, 100).forEach(row => {
    const tr = document.createElement('tr');
    columns.forEach(column => {
      const td = document.createElement('td');
      td.textContent = readableValue(row[column]);
      tr.append(td);
    });
    body.append(tr);
  });
  table.append(body);
  wrapper.append(table);
  container.append(wrapper);
  return true;
}

function resultRows(data) {
  if (!data || typeof data !== 'object') return null;
  if (Array.isArray(data.result)) return data.result;
  if (data.result && typeof data.result === 'object') {
    return Object.entries(data.result).map(([group, value]) => ({Group: group, Value: value}));
  }
  if (data.summary && typeof data.summary === 'object') return [data.summary];
  if (Array.isArray(data.results)) return data.results;
  if (data.rankings_by_currency && typeof data.rankings_by_currency === 'object') {
    return Object.entries(data.rankings_by_currency).flatMap(([currency, ranking]) =>
      (ranking.result || []).map(row => ({Currency: currency, ...row})));
  }
  return null;
}

function resultTitle(tool) {
  if (/inspect_dataset|list_datasets|get_dataset_info|list_documents/.test(tool)) return null;
  if (/column_summary/.test(tool)) return 'About this column';
  if (/groupby/.test(tool)) return 'Results by group';
  if (/aggregate/.test(tool)) return 'Calculated results';
  if (/filter/.test(tool)) return 'Matching rows';
  if (/rank/.test(tool)) return 'Top results';
  if (/rag_|retrieve|search_dataset/.test(tool)) return 'From your files';
  if (/web_|tavily/.test(tool)) return 'Sources from the web';
  if (/analyze/.test(tool)) return 'A closer look';
  return 'Supporting details';
}

function renderResults(results = []) {
  results.forEach(result => {
    const title = resultTitle(result.tool);
    const rows = resultRows(result.data);
    if (!title || !rows || !rows.length) return;
    const card = document.createElement('section');
    card.className = 'result-card';
    const heading = document.createElement('h3');
    heading.textContent = title;
    card.append(heading);
    renderTable(rows, card, rows.length === 1);
    $('messages').append(card);
  });
}

$('chat').addEventListener('submit', async event => {
  event.preventDefault();
  const question = $('question').value.trim();
  if (!question) return;
  message(question, 'user');
  $('send').disabled = true;
  try {
    const data = await api('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: question})
    });
    message(data.answer, 'assistant');
    renderResults(data.results);
    $('question').value = '';
  } catch (error) {
    message(error.message, 'error');
  } finally {
    $('send').disabled = false;
  }
});

$('upload').addEventListener('submit', async event => {
  event.preventDefault();
  const button = event.target.querySelector('button');
  button.disabled = true;
  try {
    const data = await api('/api/upload', {method: 'POST', body: new FormData(event.target)});
    const metadata = data.metadata || {};
    const details = [metadata.rows !== undefined ? `${metadata.rows} rows` : null,
                     metadata.columns ? `${metadata.columns.length} columns` : null,
                     metadata.chunks !== undefined ? `${metadata.chunks} sections` : null]
                    .filter(Boolean).join(' · ');
    $('upload-status').textContent = `Added ${data.filename}${details ? ` · ${details}` : ''}`;
    $('dataset-metadata').textContent = metadata.columns ? `Columns: ${metadata.columns.join(', ')}` : '';
    setActiveDataset(data.active_dataset);
    await refreshFiles();
  } catch (error) {
    $('upload-status').textContent = error.message;
  } finally {
    button.disabled = false;
  }
});

$('health').addEventListener('click', async () => {
  $('health-status').textContent = 'Checking connections…';
  try {
    const data = await api('/api/health');
    const connected = data.connections.filter(connection => connection.connected).map(connection => connection.name);
    const missing = data.connections.filter(connection => !connection.connected).map(connection => connection.name);
    $('health-status').textContent = `${data.connected} of ${data.total} connections are working` +
      (connected.length ? ` · ${connected.join(', ')}` : '') +
      (missing.length ? `. Not connected: ${missing.join(', ')}` : '.');
  } catch (error) {
    $('health-status').textContent = error.message;
  }
});

refreshFiles().catch(error => {$('upload-status').textContent = error.message;});
