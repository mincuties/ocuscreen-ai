import { MAX_BYTES } from './validation.js';

/** Long edge sent to the server. The model only sees 224×224, so this keeps detail to spare. */
export const MAX_EDGE = 1600;

/** Size to send, or null when the original can be uploaded as is. */
export function targetSize(width, height, bytes) {
  const edge = Math.max(width, height);
  if (edge <= MAX_EDGE && bytes <= MAX_BYTES) return null;
  const scale = Math.min(1, MAX_EDGE / edge);
  return { width: Math.max(1, Math.round(width * scale)), height: Math.max(1, Math.round(height * scale)) };
}

/**
 * Downscale large phone photos to a JPEG before upload, so 12-50 MP photos fit the
 * server's 5 MB / 25 MP limits. If the browser cannot decode the file here (old
 * browser, out of memory), a file within the upload limit is sent unchanged and the
 * server decides; only an oversized file fails.
 */
export async function prepareUpload(file) {
  let bitmap;
  try {
    // Ask for the EXIF orientation explicitly: the canvas output drops EXIF.
    bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
  } catch (error) {
    if (file.size <= MAX_BYTES) return file;
    throw error;
  }
  try {
    const size = targetSize(bitmap.width, bitmap.height, file.size);
    if (!size) return file;
    const canvas = document.createElement('canvas');
    canvas.width = size.width;
    canvas.height = size.height;
    const context = canvas.getContext('2d');
    context.fillStyle = '#fff'; // JPEG has no alpha; keep transparent PNG areas white.
    context.fillRect(0, 0, size.width, size.height);
    context.drawImage(bitmap, 0, 0, size.width, size.height);
    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.92));
    if (!blob) throw new Error('toBlob failed');
    return new File([blob], file.name.replace(/\.[^.]*$/, '') + '.jpg', { type: 'image/jpeg' });
  } finally {
    bitmap.close();
  }
}
