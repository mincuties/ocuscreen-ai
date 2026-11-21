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

export function renderResult({ image, risk }) {
  byId('level').textContent = t('priority', { label: risk.label });
  byId('level').className = 'badge ' + (['red', 'yellow', 'green'].includes(risk.level) ? risk.level : '');
  byId('imageResult').textContent = image.label && Number.isFinite(image.confidence)
    ? t('imageLabel', { label: image.label, score: (image.confidence * 100).toFixed(1) })
    : t('noImageLabel');
  byId('imageMessage').textContent = image.message;
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
