import { t } from './i18n.js';

export const MAX_BYTES = 5 * 1024 * 1024;
export function validateFile(file) {
  if (!file || !file.size) return t('noFile');
  if (file.size > MAX_BYTES) return t('fileTooLarge');
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) return t('fileType');
  return null;
}

/** Only image-independent warning rules can be evaluated before inference. */
export function hasUrgentSymptoms(ids, urgentIds, combinations) {
  const selected = new Set(ids);
  return urgentIds.some(id => selected.has(id)) || combinations.some(rule =>
    !rule.image_signs.length && rule.all_symptoms.every(id => selected.has(id)));
}
