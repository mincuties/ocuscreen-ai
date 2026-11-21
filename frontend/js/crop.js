/**
 * Square crop around the eye, chosen by the user before upload.
 *
 * The server centre-crops every photo to a square, so an eye near the edge of a
 * wide photo can be cut off. Letting the user place the square sends exactly the
 * area they mean. Coordinates are pixels of the upright (EXIF-rotated) photo.
 */

/** Smallest crop side as a share of the photo's short side. */
export const MIN_SHARE = 0.2;
/** The server rejects images under 64 pixels a side. */
export const MIN_PIXELS = 64;

/** The square the server would take on its own: centred, as large as possible. */
export function defaultCrop(width, height) {
  const size = Math.min(width, height);
  return { x: (width - size) / 2, y: (height - size) / 2, size, width, height };
}

/** Keep the square inside the photo and between MIN_SHARE and the short side. */
export function clampCrop(crop) {
  const { width, height } = crop;
  const short = Math.min(width, height);
  const smallest = Math.min(short, Math.max(short * MIN_SHARE, MIN_PIXELS));
  const size = Math.min(short, Math.max(smallest, crop.size));
  return {
    x: Math.min(Math.max(crop.x, 0), width - size),
    y: Math.min(Math.max(crop.y, 0), height - size),
    size,
    width,
    height,
  };
}

/** Resize around the centre of the current square. */
export function resizeCrop(crop, size) {
  const cx = crop.x + crop.size / 2;
  const cy = crop.y + crop.size / 2;
  return clampCrop({ ...crop, x: cx - size / 2, y: cy - size / 2, size });
}

/**
 * Map a crop onto a decoded bitmap of the same photo. Returns null when the bitmap
 * is not the same upright photo (different aspect ratio, e.g. a browser that did not
 * apply EXIF rotation in one of the two places): then nothing is cropped.
 */
export function cropToSource(crop, bitmapWidth, bitmapHeight) {
  if (!crop) return null;
  const scale = bitmapWidth / crop.width;
  if (Math.abs(bitmapHeight - crop.height * scale) > 2) return null;
  return {
    sx: Math.round(crop.x * scale),
    sy: Math.round(crop.y * scale),
    side: Math.max(1, Math.min(Math.round(crop.size * scale), bitmapWidth, bitmapHeight)),
  };
}

/**
 * Drag (pointer or arrow keys) and resize (slider) a square box over the preview.
 * The box is positioned in percentages so it follows the preview's displayed size.
 */
export function createCropper({ preview, box, sizeInput, resetButton, isBusy, onChange = () => {} }) {
  let crop = null;
  function render() {
    if (!crop) return;
    box.style.left = `${(crop.x / crop.width) * 100}%`;
    box.style.top = `${(crop.y / crop.height) * 100}%`;
    box.style.width = `${(crop.size / crop.width) * 100}%`;
    box.style.height = `${(crop.size / crop.height) * 100}%`;
    sizeInput.value = String(Math.round((crop.size / Math.min(crop.width, crop.height)) * 100));
  }
  function reset() {
    if (!preview.naturalWidth) return;
    crop = defaultCrop(preview.naturalWidth, preview.naturalHeight);
    render();
  }
  preview.addEventListener('load', reset);
  resetButton.addEventListener('click', () => { reset(); onChange(); });
  sizeInput.min = String(MIN_SHARE * 100);
  sizeInput.addEventListener('input', () => {
    if (!crop || isBusy()) return;
    crop = resizeCrop(crop, (Number(sizeInput.value) / 100) * Math.min(crop.width, crop.height));
    render();
    onChange();
  });
  let drag = null;
  box.addEventListener('pointerdown', event => {
    if (!crop || isBusy()) return;
    event.preventDefault();
    box.setPointerCapture(event.pointerId);
    drag = { pointerX: event.clientX, pointerY: event.clientY, x: crop.x, y: crop.y };
  });
  box.addEventListener('pointermove', event => {
    if (!drag) return;
    // Displayed pixels to photo pixels.
    const scale = crop.width / preview.getBoundingClientRect().width;
    crop = clampCrop({
      ...crop,
      x: drag.x + (event.clientX - drag.pointerX) * scale,
      y: drag.y + (event.clientY - drag.pointerY) * scale,
    });
    render();
  });
  for (const type of ['pointerup', 'pointercancel']) {
    box.addEventListener(type, () => {
      if (drag) onChange();
      drag = null;
    });
  }
  box.addEventListener('keydown', event => {
    const steps = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
    if (!crop || isBusy() || !(event.key in steps)) return;
    event.preventDefault();
    const [dx, dy] = steps[event.key];
    const step = Math.min(crop.width, crop.height) * 0.02;
    crop = clampCrop({ ...crop, x: crop.x + dx * step, y: crop.y + dy * step });
    render();
    onChange();
  });
  return {
    getCrop: () => crop,
    clear() { crop = null; },
  };
}
