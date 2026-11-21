import { t } from './i18n.js';

/** Small DOM components. API text is always inserted as text, never HTML. */
export const byId = id => document.getElementById(id);
export function textNode(tag, text, parent, className) {
  const node = document.createElement(tag);
  node.textContent = text;
  if (className) node.className = className;
  parent.append(node);
  return node;
}

export function renderSymptoms(symptoms, container) {
  const fragment = document.createDocumentFragment();
  for (const symptom of symptoms) {
    const label = document.createElement('label');
    label.className = 'choice';
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.value = symptom.id;
    label.append(input);
    textNode('span', symptom.label, label);
    fragment.append(label);
  }
  container.replaceChildren(fragment);
}

/** Model scores below this are close to the 12-class junk baseline; not worth showing. */
export const OTHER_MIN = 0.15;
/** Runner-up labels of a valid image, shown for transparency only. */
export function otherPossibilities(image) {
  if (image?.status !== 'valid' || !Array.isArray(image.top3)) return [];
  return image.top3.slice(1).filter(item => item.label !== 'Back Ground' && item.confidence >= OTHER_MIN);
}
const percent = score => (score * 100).toFixed(1);

/** Only accept the JPEG data URL the API builds, never another URL scheme. */
export function attentionSource(image) {
  const src = image?.attention?.image;
  return typeof src === 'string' && src.startsWith('data:image/jpeg;base64,') ? src : null;
}

export function renderResult({ image, risk }, { explain = false } = {}) {
  byId('level').textContent = t('priority', { label: risk.label });
  byId('level').className = 'badge ' + (['red', 'yellow', 'green'].includes(risk.level) ? risk.level : '');
  // An uncertain image has no label worth showing; its message asks for a retake.
  byId('imageResult').textContent = image.status === 'valid' && Number.isFinite(image.confidence)
    ? t('imageLabel', { label: image.label, score: percent(image.confidence) })
    : t('noImageLabel');
  const src = attentionSource(image);
  byId('attention').hidden = !explain;
  byId('attentionImage').hidden = !src;
  if (src) byId('attentionImage').src = src;
  else byId('attentionImage').removeAttribute('src');
  byId('attentionNote').textContent = src ? image.attention.note : t('attentionMissing');
  const others = otherPossibilities(image);
  byId('otherLabels').hidden = !others.length;
  byId('otherLabels').textContent = others.length ? t('otherLabels', {
    items: others.map(item => `${item.label} ${percent(item.confidence)}%`).join(', '),
  }) : '';
  byId('imageMessage').textContent = image.message;
  const advice = image.quality?.advice || [];
  byId('qualityAdvice').replaceChildren();
  advice.forEach(tip => textNode('li', tip, byId('qualityAdvice')));
  byId('quality').hidden = !advice.length;
  byId('advice').textContent = risk.recommendation;
  byId('destination').hidden = !risk.destination_text;
  byId('destination').textContent = risk.destination_text ? t('goTo', { place: risk.destination_text }) : '';
  byId('firstAidList').replaceChildren();
  (risk.first_aid || []).forEach(step => textNode('li', step, byId('firstAidList')));
  byId('firstAid').hidden = !risk.first_aid?.length;
  byId('warnings').replaceChildren();
  (risk.warnings || []).forEach(warning => textNode('li', warning, byId('warnings')));
  byId('reasons').replaceChildren();
  risk.reasons.forEach(reason => textNode('li', reason, byId('reasons')));
  const candidates = byId('candidates');
  candidates.replaceChildren();
  if (risk.candidates.length) {
    textNode('h3', t('candidatesTitle'), candidates);
    textNode('p', t('candidatesHint'), candidates, 'hint');
    for (const candidate of risk.candidates) {
      const article = document.createElement('article');
      textNode('strong', candidate.name, article);
      textNode('p', candidate.note, article);
      const matches = [candidate.matched_image_sign, ...candidate.matched_symptoms].filter(Boolean);
      textNode('small', t('matchedInfo', { items: matches.join(', ') }), article);
      candidates.append(article);
    }
  }
  byId('disclaimer').textContent = risk.disclaimer;
  byId('printMeta').textContent = t('versions', {
    model: image.model_version ? image.model_version.slice(0, 12) : t('noModel'),
    rules: risk.knowledge_version,
    time: new Date().toLocaleString(document.documentElement.lang || undefined),
  });
  byId('result').hidden = false;
  byId('result').focus({ preventScroll: true });
  byId('result').scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth', block: 'start' });
}

/** Accent-insensitive search preserves checked inputs, including hidden matches. */
export function normalizeSearch(value) {
  return value.normalize('NFD').replace(/[\u0300-\u036f]/g, '').replace(/[đĐ]/g, 'd').toLowerCase().trim();
}
export function filterSymptoms(container, query) {
  const term = normalizeSearch(query);
  let count = 0;
  for (const choice of container.children) {
    choice.hidden = !normalizeSearch(choice.textContent).includes(term);
    if (!choice.hidden) count++;
  }
  return count;
}
