import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { validateFile, matchRules, ruleReason, MAX_BYTES, MAX_INPUT_BYTES } from '../frontend/js/validation.js';
import { targetSize, MAX_EDGE } from '../frontend/js/resize.js';
import { requestJSON, fetchSymptoms, screenImage } from '../frontend/js/api.js';
import { MESSAGES, getLang, setLang, t, applyTranslations } from '../frontend/js/i18n.js';
const kb = JSON.parse(readFileSync(new URL('../data/disease_knowledge.json', import.meta.url)));
const cases = JSON.parse(readFileSync(new URL('./triage_cases.json', import.meta.url))).cases;
// The same subset /api/symptoms sends as red_rules.
const redRules = kb.triage_rules.filter(rule => rule.level === 'red' && !rule.image)
  .map(rule => ({ ...rule, reason: rule.reason.en }));

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
test('pre-upload warnings agree with the backend triage cases', () => {
  for (const c of cases) {
    const warned = matchRules(redRules, c.symptoms, c).length > 0;
    // Never warn when the backend would not say red; without an image sign, agree exactly.
    if (warned) assert.equal(c.level, 'red', c.id);
    if (['Normal', 'Back Ground'].includes(c.image.label)) assert.equal(warned, c.level === 'red', c.id);
  }
  assert.equal(matchRules(redRules, []).length, 0);
  assert.equal(matchRules(redRules, ['contacts']).length, 0);
  assert.deepEqual(matchRules(redRules, ['blurred']).map(r => r.id), ['blurred_new']);
  assert.equal(matchRules(redRules, ['blurred'], { duration: '1_7d' }).length, 0);
  assert.deepEqual(matchRules(redRules, ['itch'], { age_group: 'infant' }).map(r => r.id), ['infant_symptoms']);
});
test('rule reasons name the selected symptoms', () => {
  const labels = { vision_loss: 'Vision loss', photophobia: 'Light pain', itch: 'Itch' };
  const ids = ['itch', 'vision_loss', 'photophobia'];
  const [rule] = matchRules(redRules, ids);
  assert.equal(ruleReason(rule, ids, labels), 'Danger signs: Vision loss, Light pain');
});
test('API retains symptom-only 503 results and handles non-JSON failures', async () => {
  const original = globalThis.fetch;
  try {
    globalThis.fetch = async () => new Response(JSON.stringify({ risk: { level: 'red' } }), {status: 503});
    assert.equal((await requestJSON('/api/screen')).data.risk.level, 'red');
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

test('other possibilities come only from a valid image, without background or weak scores', async () => {
  const { otherPossibilities } = await import('../frontend/js/components.js');
  const top3 = [{ label: 'Redness', confidence: 0.62 }, { label: 'Back Ground', confidence: 0.2 },
    { label: 'Discharge', confidence: 0.16 }];
  assert.deepEqual(otherPossibilities({ status: 'valid', top3 }).map(i => i.label), ['Discharge']);
  assert.deepEqual(otherPossibilities({ status: 'uncertain', top3 }), []);
  assert.deepEqual(otherPossibilities({ status: 'valid', top3: [top3[0], { label: 'Watery', confidence: 0.1 }] }), []);
  assert.deepEqual(otherPossibilities({ status: 'invalid', top3: null }), []);
});

test('attention maps are only taken from the JPEG data URL the API builds', async () => {
  const { attentionSource } = await import('../frontend/js/components.js');
  const ok = 'data:image/jpeg;base64,AAAA';
  assert.equal(attentionSource({ attention: { image: ok } }), ok);
  for (const image of [{}, { attention: null }, { attention: { image: 'javascript:alert(1)' } },
    { attention: { image: 'https://example.com/x.jpg' } }, { attention: { image: 42 } }]) {
    assert.equal(attentionSource(image), null);
  }
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
    const bodies = [];
    globalThis.fetch = async (url, options) => { urls.push(url); bodies.push(options?.body); return new Response('{"symptoms": []}'); };
    await fetchSymptoms();
    setLang('vi');
    await fetchSymptoms();
    await screenImage(new Blob(['x']), ['itch'], { age_group: 'adult', duration: '', confirmed: true, explain: false });
    assert.deepEqual(urls, ['/api/symptoms?lang=en', '/api/symptoms?lang=vi', '/api/screen?lang=vi']);
    const form = bodies[2];
    assert.equal(form.get('age_group'), 'adult');
    assert.equal(form.get('confirmed'), 'true');
    assert.equal(form.has('duration'), false);
    assert.equal(form.get('explain'), 'false');
  } finally { globalThis.fetch = original; setLang('en'); }
});
