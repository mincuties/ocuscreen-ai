import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { validateFile, hasUrgentSymptoms, MAX_BYTES, MAX_INPUT_BYTES } from '../frontend/js/validation.js';
import { targetSize, MAX_EDGE } from '../frontend/js/resize.js';
import { requestJSON, fetchSymptoms, screenImage } from '../frontend/js/api.js';
import { MESSAGES, getLang, setLang, t, applyTranslations } from '../frontend/js/i18n.js';
const kb = JSON.parse(readFileSync(new URL('../data/disease_knowledge.json', import.meta.url)));

test('upload validation rejects unsupported, empty and oversized images', () => {
  assert.equal(validateFile({ size: MAX_BYTES, type: 'image/png' }), null);
  // Photos above the 5 MB upload limit are accepted here and downscaled before upload.
  assert.equal(validateFile({ size: MAX_BYTES + 1, type: 'image/jpeg' }), null);
  for (const file of [null, {size: 0}, {size: MAX_INPUT_BYTES + 1, type: 'image/jpeg'}, {size: 20, type: 'text/html'}]) {
    assert.ok(validateFile(file));
  }
});
test('large photos are downscaled to the upload size, small ones are left alone', () => {
  assert.equal(targetSize(1200, 900, 400_000), null);
  assert.deepEqual(targetSize(4032, 3024, 3_000_000), { width: MAX_EDGE, height: 1200 });
  assert.deepEqual(targetSize(3024, 4032, 3_000_000), { width: 1200, height: MAX_EDGE });
  // Small in pixels but too heavy: re-encode at the same size.
  assert.deepEqual(targetSize(1500, 1000, MAX_BYTES + 1), { width: 1500, height: 1000 });
});
test('pre-inference warnings match image-independent knowledge rules', () => {
  const warn = ids => hasUrgentSymptoms(ids, kb.urgent_symptoms, kb.urgent_combinations);
  assert.equal(warn([]), false);
  assert.equal(warn(['itch']), false);
  assert.equal(warn(['contacts']), false);
  for (const ids of [['vision_loss'], ['contacts', 'red_eye'], ['fever', 'pain_movement'], ['yellow_skin'], ['bulging']]) {
    assert.equal(warn(ids), true);
  }
});
test('API retains symptom-only 503 results and handles non-JSON failures', async () => {
  const original = globalThis.fetch;
  try {
    globalThis.fetch = async () => new Response(JSON.stringify({ risk: { level: 'high' } }), {status: 503});
    assert.equal((await requestJSON('/api/screen')).data.risk.level, 'high');
    globalThis.fetch = async () => new Response('<html>Bad gateway</html>', {status: 502});
    await assert.rejects(requestJSON('/api/screen'), /Please try again/);
    setLang('vi');
    await assert.rejects(requestJSON('/api/screen'), /Vui lòng thử lại/);
  } finally { globalThis.fetch = original; setLang('en'); }
});

test('Vietnamese symptom search ignores accents and preserves choices', async () => {
  const { filterSymptoms } = await import('../frontend/js/components.js');
  const choices = [{textContent: 'Đỏ mắt', checked: true}, {textContent: 'Ngứa mắt'}];
  assert.equal(filterSymptoms({children: choices}, 'do mat'), 1);
  assert.equal(choices[0].hidden, false);
  assert.equal(choices[1].hidden, true);
  assert.equal(choices[0].checked, true);
  assert.equal(filterSymptoms({children: choices}, ''), 2);
});

test('upload synchronizes native files, releases previews and preserves cancellation', async () => {
  const { createUpload } = await import('../frontend/js/upload.js');
  const original = { window: globalThis.window, transfer: globalThis.DataTransfer,
    create: URL.createObjectURL, revoke: URL.revokeObjectURL };
  const element = () => Object.assign(new EventTarget(), {
    value: '', files: [], hidden: true, textContent: '',
    classList: {remove() {}, toggle() {}}, removeAttribute() {},
  });
  const input = element(), dropzone = element(), preview = element(), meta = element();
  let chosen, error, busy = false;
  const revoked = [];
  try {
    globalThis.window = new EventTarget();
    globalThis.DataTransfer = class { constructor() { this.files=[]; this.items={add: f=>this.files.push(f)}; } };
    URL.createObjectURL = file => `blob:${file.name}`;
    URL.revokeObjectURL = url => revoked.push(url);
    const upload = createUpload({input, dropzone, preview, meta, isBusy:()=>busy,
      onChange:f=>{chosen=f;}, onError:m=>{error=m;}});
    const first={name:'first.png',type:'image/png',size:100};
    input.files=[first];input.dispatchEvent(new Event('change'));
    assert.equal(chosen,first);
    input.files=[];input.dispatchEvent(new Event('change'));
    assert.equal(input.files[0],first);
    const second={...first,name:'second.png'};
    const drop=new Event('drop');drop.dataTransfer={files:[second,first]};
    dropzone.dispatchEvent(drop);
    assert.deepEqual(input.files,[second]);
    assert.deepEqual(revoked,['blob:first.png']);
    busy=true;input.dispatchEvent(new Event('change'));
    assert.equal(chosen,second);
    busy=false;
    preview.dispatchEvent(new Event('error'));
    assert.equal(chosen,null);
    assert.match(error,/could not be read/);
    assert.deepEqual(revoked,['blob:first.png','blob:second.png']);
    upload.clear();
    assert.equal(preview.hidden,true);
  } finally {
    globalThis.window=original.window;globalThis.DataTransfer=original.transfer;
    URL.createObjectURL=original.create;URL.revokeObjectURL=original.revoke;
  }
});

test('English is the default and both languages define the same UI keys', () => {
  assert.equal(getLang(), 'en');
  assert.deepEqual(Object.keys(MESSAGES.vi).sort(), Object.keys(MESSAGES.en).sort());
  for (const lang of ['en', 'vi']) {
    for (const [key, text] of Object.entries(MESSAGES[lang])) assert.ok(text.trim(), `${lang}.${key}`);
  }
  const html = readFileSync(new URL('../frontend/index.html', import.meta.url), 'utf8');
  const keys = [...html.matchAll(/data-i18n="(\w+)"/g)].map(m => m[1]);
  for (const pairs of html.matchAll(/data-i18n-attr="([^"]+)"/g)) {
    keys.push(...pairs[1].split(';').map(pair => pair.split(':')[1]));
  }
  assert.ok(keys.length > 20);
  for (const key of keys) assert.ok(key in MESSAGES.en, key);
});

test('translations interpolate values and switch language', () => {
  try {
    assert.equal(t('selectedCount', { count: 3 }), '3 selected');
    assert.equal(setLang('vi'), 'vi');
    assert.equal(t('selectedCount', { count: 3 }), '3 đã chọn');
    assert.equal(validateFile(null), 'Hãy chọn một ảnh mắt.');
    assert.equal(setLang('fr'), 'en');
    assert.equal(validateFile(null), 'Please choose an eye photo.');
    assert.equal(t('missingKey'), 'missingKey');
  } finally { setLang('en'); }
});

test('applyTranslations fills text and attributes', () => {
  const text = { dataset: { i18n: 'submit' }, textContent: '' };
  const attrs = {};
  const input = { dataset: { i18nAttr: 'placeholder:searchPlaceholder;aria-label:searchLabel' },
    setAttribute: (name, value) => { attrs[name] = value; } };
  const root = { querySelectorAll: selector => (selector === '[data-i18n]' ? [text] : [input]) };
  try {
    setLang('vi');
    applyTranslations(root);
    assert.equal(text.textContent, 'Phân tích sàng lọc');
    assert.deepEqual(attrs, { placeholder: 'Ví dụ: đỏ mắt, ngứa, chảy nước mắt', 'aria-label': 'Tìm triệu chứng' });
  } finally { setLang('en'); }
});

test('API requests send the selected language', async () => {
  const original = globalThis.fetch;
  const urls = [];
  try {
    globalThis.fetch = async url => { urls.push(url); return new Response('{"symptoms": []}'); };
    await fetchSymptoms();
    setLang('vi');
    await fetchSymptoms();
    await screenImage(new Blob(['x']), ['itch']);
    assert.deepEqual(urls, ['/api/symptoms?lang=en', '/api/symptoms?lang=vi', '/api/screen?lang=vi']);
  } finally { globalThis.fetch = original; setLang('en'); }
});
