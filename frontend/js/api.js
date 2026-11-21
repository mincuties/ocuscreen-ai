import { getLang, t } from './i18n.js';

/** JSON transport, including symptom-only results returned with HTTP 503. */
export async function requestJSON(url, options = {}, timeout = 90000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    const data = await response.json().catch(() => null);
    if (!data || (!response.ok && !data.risk)) {
      throw new Error(typeof data?.detail === 'string' ? data.detail :
        t('serverNoResult'));
    }
    return { data, ok: response.ok };
  } catch (error) {
    if (error.name === 'AbortError') throw new Error(t('serverTimeout'));
    if (error instanceof TypeError) throw new Error(t('serverOffline'));
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

export function fetchSymptoms(lang = getLang()) {
  return requestJSON(`/api/symptoms?lang=${encodeURIComponent(lang)}`, {}, 15000);
}

/** triage: { age_group, duration, confirmed }; empty values are left out. */
export function screenImage(file, symptoms, triage = {}, lang = getLang()) {
  const body = new FormData();
  body.append('image', file);
  body.append('symptoms', JSON.stringify(symptoms));
  for (const [key, value] of Object.entries(triage)) {
    if (value !== '' && value !== undefined && value !== null) body.append(key, String(value));
  }
  return requestJSON(`/api/screen?lang=${encodeURIComponent(lang)}`, { method: 'POST', body });
}
