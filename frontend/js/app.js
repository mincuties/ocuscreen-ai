import { fetchSymptoms, screenImage } from './api.js';
import { byId as $, renderSymptoms, renderResult, filterSymptoms } from './components.js';
import { applyTranslations, getLang, setLang, t } from './i18n.js';
import { matchRules, ruleReason } from './validation.js';
import { prepareUpload } from './resize.js';

import { createUpload } from './upload.js';

const state = { file: null, loaded: false, busy: false, rules: [], labels: {}, loadToken: 0 };
// Remember the last status so a language switch can translate it; server errors stay as sent.
let lastStatus = { key: 'loadingSymptoms', text: null, error: false };
const selected = () => [...$('checks').querySelectorAll('input:checked')].map(input => input.value);
const triage = () => ({ age_group: $('ageGroup').value, duration: $('duration').value });
/** Warn before the upload when a red rule already matches the answers. */
function updateUrgentHint() {
  const ids = selected();
  const matched = matchRules(state.rules, ids, triage());
  $('urgentReasons').textContent = matched.map(rule => ruleReason(rule, ids, state.labels)).join('. ');
  $('urgentHint').hidden = !matched.length;
}
const langButtons = () => [...document.querySelectorAll('.lang-switch [data-lang]')];
function renderStatus() {
  $('status').textContent = lastStatus.text ?? t(lastStatus.key);
  $('status').dataset.error = String(lastStatus.error);
}
function status(key, error = false) {
  lastStatus = { key, text: null, error };
  renderStatus();
}
function statusText(text, error = false) {
  lastStatus = { key: null, text, error };
  renderStatus();
}
function updateControls() {
  $('submit').disabled = state.busy || !state.loaded || !state.file;
  $('submit').textContent = t(state.busy ? 'submitBusy' : 'submit');
  $('image').disabled = state.busy;
  $('dropzone').classList.toggle('disabled', state.busy);
  $('symptomSearch').disabled = state.busy || !state.loaded;
  $('symptomField').disabled = state.busy || !state.loaded;
  for (const id of ['ageGroup', 'duration', 'confirmed']) $(id).disabled = state.busy || !state.loaded;
  $('screenForm').querySelector('[type=reset]').disabled = state.busy;
  $('screenForm').setAttribute('aria-busy', String(state.busy));
  $('selectedCount').textContent = t('selectedCount', { count: selected().length });
  for (const button of langButtons()) button.disabled = state.busy;
  const active = !$('result').hidden ? 'stepResult' : state.file ? 'stepSymptoms' : 'stepImage';
  for (const id of ['stepImage', 'stepSymptoms', 'stepResult']) {
    if (id === active) $(id).setAttribute('aria-current', 'step');
    else $(id).removeAttribute('aria-current');
  }
}
function clearResult() { $('result').hidden = true; }
function applyLanguage() {
  document.documentElement.lang = getLang();
  applyTranslations(document);
  for (const button of langButtons()) button.setAttribute('aria-pressed', String(button.dataset.lang === getLang()));
  renderStatus();
  updateControls();
}
const upload = createUpload({
  input: $('image'), dropzone: $('dropzone'), preview: $('preview'), meta: $('fileMeta'),
  isBusy: () => state.busy,
  onError: message => statusText(message, true),
  onChange: file => {
    state.file = file;
    clearResult();
    status(file ? 'imageReady' : 'chooseImage');
    updateControls();
  },
});
function updateSearch() {
  const visible = filterSymptoms($('checks'), $('symptomSearch').value);
  $('emptySearch').hidden = visible > 0;
}
$('symptomSearch').addEventListener('input', updateSearch);
async function init() {
  const token = ++state.loadToken;
  $('retry').hidden = true;
  status('loadingSymptoms');
  try {
    const { data } = await fetchSymptoms();
    if (token !== state.loadToken) return; // A newer load (e.g. another language) won.
    const keep = new Set(selected());
    state.rules = data.red_rules || [];
    state.labels = Object.fromEntries(data.symptoms.map(s => [s.id, s.label]));
    renderSymptoms(data.symptoms, $('checks'));
    for (const input of $('checks').querySelectorAll('input')) input.checked = keep.has(input.value);
    updateUrgentHint();
    updateSearch();
    state.loaded = true;
    status(state.file ? 'readyToAnalyze' : 'chooseImage');
  } catch (error) {
    if (token !== state.loadToken) return;
    statusText(error.message, true);
    $('retry').hidden = false;
  }
  updateControls();
}
async function changeLanguage(lang) {
  if (state.busy || lang === getLang()) return;
  setLang(lang);
  const hadResult = !$('result').hidden;
  clearResult();
  applyLanguage();
  await init();
  // Results come from the server in one language; ask for a new run instead of mixing languages.
  if (hadResult && state.loaded && !lastStatus.error) status('languageChanged');
}
for (const button of langButtons()) button.addEventListener('click', () => changeLanguage(button.dataset.lang));
$('retry').addEventListener('click', init);
for (const id of ['checks', 'ageGroup', 'duration', 'confirmed']) {
  $(id).addEventListener('change', () => {
    // A confirmation given before the checklist changed no longer covers it.
    if (id === 'checks') $('confirmed').checked = false;
    clearResult();
    updateUrgentHint();
    updateControls();
  });
}
$('screenForm').addEventListener('reset', () => {
  clearResult();
  upload.clear();
  state.file = null;
  $('fileMeta').textContent = '';
  $('urgentHint').hidden = true;
  status(state.loaded ? 'chooseImage' : 'reloadSymptoms');
  queueMicrotask(() => { updateSearch(); updateControls(); });
});
$('screenForm').addEventListener('submit', async event => {
  event.preventDefault();
  if (state.busy || !state.loaded || !state.file) return;
  const ids = selected();
  state.busy = true;
  clearResult();
  updateControls();
  status('analyzing');
  try {
    let photo;
    try {
      photo = await prepareUpload(state.file);
    } catch {
      throw new Error(t('fileUnreadable'));
    }
    const { data, ok } = await screenImage(photo, ids, { ...triage(), confirmed: $('confirmed').checked });
    renderResult(data);
    status(ok ? 'analyzed' : 'symptomsOnly');
  } catch (error) {
    statusText(error.message, true);
  } finally {
    state.busy = false;
    updateControls();
  }
});
applyLanguage();
init();
