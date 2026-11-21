import { t } from './i18n.js';

/** Server upload limit. Larger photos are downscaled in the browser before upload. */
export const MAX_BYTES = 5 * 1024 * 1024;
/** Largest photo accepted for downscaling (a 50 MP JPEG is usually below this). */
export const MAX_INPUT_BYTES = 30 * 1024 * 1024;
export function validateFile(file) {
  if (!file || !file.size) return t('noFile');
  if (file.size > MAX_INPUT_BYTES) return t('fileTooLarge');
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) return t('fileType');
  return null;
}

/**
 * Red rules that need no image (sent by /api/symptoms) can warn before the upload.
 * Mirrors backend/risk_engine.py; an empty age or duration means "unknown".
 */
export function matchRules(rules, ids, { age_group = '', duration = '' } = {}) {
  const selected = new Set(ids);
  return rules.filter(rule =>
    !rule.image &&
    (rule.all || []).every(id => selected.has(id)) &&
    (!rule.any || rule.any.some(id => selected.has(id))) &&
    (!rule.age || rule.age.includes(age_group || 'unknown')) &&
    (!rule.duration || rule.duration.includes(duration || 'unknown')));
}

/** Fill a rule reason with the labels of the symptoms that made it match. */
export function ruleReason(rule, ids, labels) {
  const named = rule.any || rule.all || [];
  const list = ids.filter(id => named.includes(id)).map(id => labels[id] ?? id);
  return rule.reason.replace('{symptoms}', list.join(', '));
}
