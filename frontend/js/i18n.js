/** UI strings for the English (default) and Vietnamese versions of the page. */
export const LANGUAGES = ['en', 'vi'];
export const DEFAULT_LANG = 'en';
const STORAGE_KEY = 'ocuscreen-lang';

export const MESSAGES = {
  en: {
    pageTitle: 'OcuScreen AI · Eye screening',
    noscript: 'Please enable JavaScript to upload a photo and use the screening.',
    skipLink: 'Skip to form',
    langGroup: 'Language',
    tag: 'PROTOTYPE',
    eyebrow: 'PRELIMINARY SCREENING · TWO STAGES',
    headline1: 'Understand eye signs.',
    headline2: 'Take the next step with confidence.',
    intro: 'Combines an eye photo with the symptoms you report to suggest how urgently to see a doctor. Results are for reference only and have not been clinically validated.',
    emergencyNotice: 'If you have severe eye pain, sudden loss of vision or a chemical in the eye, seek care immediately. Do not wait for the AI result.',
    stepperLabel: 'Screening progress',
    stepImage: '01 · Photo',
    stepSymptoms: '02 · Symptoms',
    stepResult: '03 · Result',
    photoTitle: 'Choose an eye photo',
    dropTitle: 'Choose or drag and drop an eye photo',
    dropHint: 'JPEG, PNG, WebP · up to 30 MB',
    previewAlt: 'Selected eye photo',
    photoGuide: 'Use good lighting and keep the eye centered. Do not use color filters. The image must be at least 64×64 pixels.',
    privacy: 'The photo is sent to the server for processing. The app does not save photos or checklists as records; do not enter identifying information.',
    symptomsTitle: 'Your symptoms',
    selectedCount: '{count} selected',
    symptomsHint: 'Select everything that applies. Leave it empty if you have no symptoms.',
    urgentTitle: 'Some signs need immediate care.',
    urgentTail: 'Do not wait for the image analysis to seek medical help.',
    searchLabel: 'Search symptoms',
    searchPlaceholder: 'For example: red eye, itchy, watery',
    searchHint: 'Search with or without accents. Selected items stay selected.',
    symptomLegend: 'Current symptoms',
    emptySearch: 'No symptoms found. Try another keyword.',
    ageLabel: 'Who is this for?',
    choose: 'Choose…',
    ageInfant: 'Baby under 3 months',
    ageChild: 'Child (3 months to 15 years)',
    ageAdult: 'Adult',
    durationLabel: 'How long has it lasted?',
    durationUnknown: 'Not sure / no symptoms',
    durationDay: 'Less than 1 day',
    durationWeek: '1 to 7 days',
    durationLong: 'More than 7 days',
    confirmLabel: 'I have read the whole list and ticked every symptom I have.',
    firstAidTitle: 'Do this now',
    goTo: 'Where: {place}',
    submit: 'Run screening',
    submitBusy: 'Analyzing…',
    reset: 'Start over',
    retry: 'Reload symptoms',
    resultTitle: 'Preliminary screening result',
    printResult: 'Print / save as PDF',
    footer: 'OcuScreen AI · Photo → Signs → Symptoms → Care suggestion',
    footerLink: 'Signs that need an eye examination (NHS)',
    loadingSymptoms: 'Loading the symptom list…',
    chooseImage: 'Choose an eye photo to begin.',
    imageReady: 'Photo ready. Check your symptoms, then run the screening.',
    readyToAnalyze: 'Ready to analyze.',
    reloadSymptoms: 'Please reload the symptom list.',
    analyzing: 'Analyzing the photo and symptoms…',
    analyzed: 'Analysis complete.',
    symptomsOnly: 'Symptom-only result; the model is not ready.',
    languageChanged: 'Language changed. Run the screening again to see the result in English.',
    priority: 'Level: {label}',
    imageLabel: 'Image label: {label} · Model score: {score}%',
    noImageLabel: 'No valid image label',
    candidatesTitle: 'Symptom patterns with similarities',
    candidatesHint: 'For reference only; ordered by match points, not by probability of disease.',
    matchedInfo: 'Matched: {items}',
    noFile: 'Please choose an eye photo.',
    fileTooLarge: 'The photo is larger than 30 MB. Please choose a smaller one.',
    fileType: 'Only JPEG, PNG or WebP images are accepted.',
    fileUnreadable: 'The photo could not be read. Please choose another one.',
    serverNoResult: 'The server did not return a result. Please try again.',
    serverTimeout: 'The server took too long to respond. Please try again.',
    serverOffline: 'Could not reach the server. Check your connection and try again.',
  },
  vi: {
    pageTitle: 'OcuScreen AI · Sàng lọc mắt',
    noscript: 'Vui lòng bật JavaScript để tải ảnh và sử dụng sàng lọc.',
    skipLink: 'Chuyển đến biểu mẫu',
    langGroup: 'Ngôn ngữ',
    tag: 'BẢN THỬ NGHIỆM',
    eyebrow: 'SÀNG LỌC SƠ BỘ · HAI TẦNG',
    headline1: 'Hiểu dấu hiệu mắt.',
    headline2: 'Chủ động bước tiếp theo.',
    intro: 'Kết hợp ảnh mắt và triệu chứng bạn cung cấp để gợi ý mức độ ưu tiên đi khám. Kết quả mang tính tham khảo, chưa được thẩm định lâm sàng.',
    emergencyNotice: 'Nếu đau mắt dữ dội, giảm thị lực đột ngột hoặc hóa chất vào mắt, hãy đi khám ngay. Đừng chờ kết quả AI.',
    stepperLabel: 'Tiến trình sàng lọc',
    stepImage: '01 · Chọn ảnh',
    stepSymptoms: '02 · Triệu chứng',
    stepResult: '03 · Kết quả',
    photoTitle: 'Chọn ảnh mắt',
    dropTitle: 'Chọn hoặc kéo thả ảnh mắt',
    dropHint: 'JPEG, PNG, WebP · tối đa 30 MB',
    previewAlt: 'Ảnh mắt đã chọn',
    photoGuide: 'Chụp đủ sáng, giữ mắt ở giữa ảnh. Không dùng bộ lọc màu. Ảnh cần ít nhất 64×64 điểm ảnh.',
    privacy: 'Ảnh được gửi đến máy chủ để xử lý. Ứng dụng không lưu ảnh hoặc checklist thành hồ sơ; không nhập thông tin định danh.',
    symptomsTitle: 'Triệu chứng của bạn',
    selectedCount: '{count} đã chọn',
    symptomsHint: 'Chọn tất cả mục phù hợp. Nếu không có triệu chứng, để trống.',
    urgentTitle: 'Có dấu hiệu cần khám ngay.',
    urgentTail: 'Đừng chờ phân tích ảnh để tìm trợ giúp y tế.',
    searchLabel: 'Tìm triệu chứng',
    searchPlaceholder: 'Ví dụ: đỏ mắt, ngứa, chảy nước mắt',
    searchHint: 'Tìm có dấu hoặc không dấu. Các mục đã chọn vẫn được giữ lại.',
    symptomLegend: 'Triệu chứng hiện tại',
    emptySearch: 'Không tìm thấy triệu chứng. Thử từ khóa khác.',
    ageLabel: 'Người cần sàng lọc là ai?',
    choose: 'Chọn…',
    ageInfant: 'Trẻ dưới 3 tháng tuổi',
    ageChild: 'Trẻ em (3 tháng đến 15 tuổi)',
    ageAdult: 'Người lớn',
    durationLabel: 'Triệu chứng kéo dài bao lâu?',
    durationUnknown: 'Không rõ / không có triệu chứng',
    durationDay: 'Dưới 1 ngày',
    durationWeek: '1 đến 7 ngày',
    durationLong: 'Hơn 7 ngày',
    confirmLabel: 'Tôi đã đọc hết danh sách và chọn mọi triệu chứng mình đang có.',
    firstAidTitle: 'Làm ngay',
    goTo: 'Nơi khám: {place}',
    submit: 'Phân tích sàng lọc',
    submitBusy: 'Đang phân tích…',
    reset: 'Làm lại',
    retry: 'Tải lại triệu chứng',
    resultTitle: 'Kết quả sàng lọc sơ bộ',
    printResult: 'In / lưu thành PDF',
    footer: 'OcuScreen AI · Ảnh → Dấu hiệu → Triệu chứng → Gợi ý đi khám',
    footerLink: 'Tham khảo dấu hiệu cần khám mắt (NHS)',
    loadingSymptoms: 'Đang tải danh sách triệu chứng…',
    chooseImage: 'Chọn ảnh mắt để bắt đầu.',
    imageReady: 'Ảnh đã sẵn sàng. Kiểm tra triệu chứng rồi phân tích.',
    readyToAnalyze: 'Sẵn sàng phân tích.',
    reloadSymptoms: 'Hãy tải lại danh sách triệu chứng.',
    analyzing: 'Đang phân tích ảnh và triệu chứng…',
    analyzed: 'Đã phân tích.',
    symptomsOnly: 'Chỉ có kết quả triệu chứng; model chưa sẵn sàng.',
    languageChanged: 'Đã đổi ngôn ngữ. Hãy phân tích lại để xem kết quả bằng tiếng Việt.',
    priority: 'Mức độ: {label}',
    imageLabel: 'Nhãn ảnh: {label} · Điểm model: {score}%',
    noImageLabel: 'Chưa có nhãn ảnh hợp lệ',
    candidatesTitle: 'Nhóm biểu hiện có điểm tương đồng',
    candidatesHint: 'Danh sách tham khảo; thứ tự theo điểm khớp, không phải xác suất bệnh.',
    matchedInfo: 'Thông tin khớp: {items}',
    noFile: 'Hãy chọn một ảnh mắt.',
    fileTooLarge: 'Ảnh lớn hơn 30 MB. Hãy chọn ảnh nhỏ hơn.',
    fileType: 'Chỉ nhận ảnh JPEG, PNG hoặc WebP.',
    fileUnreadable: 'Không đọc được ảnh. Hãy chọn ảnh khác.',
    serverNoResult: 'Máy chủ chưa trả được kết quả. Vui lòng thử lại.',
    serverTimeout: 'Máy chủ phản hồi quá lâu. Vui lòng thử lại.',
    serverOffline: 'Không kết nối được máy chủ. Kiểm tra kết nối rồi thử lại.',
  },
};

function storedLang() {
  try {
    return globalThis.localStorage?.getItem(STORAGE_KEY);
  } catch {
    return null; // Storage can be blocked (private mode, sandboxed previews).
  }
}

const saved = storedLang();
let current = LANGUAGES.includes(saved) ? saved : DEFAULT_LANG;

export function getLang() {
  return current;
}

export function setLang(lang) {
  current = LANGUAGES.includes(lang) ? lang : DEFAULT_LANG;
  try {
    globalThis.localStorage?.setItem(STORAGE_KEY, current);
  } catch {
    // The choice still applies for this page view.
  }
  return current;
}

export function t(key, params = {}) {
  const text = MESSAGES[current][key] ?? MESSAGES[DEFAULT_LANG][key] ?? key;
  return text.replace(/\{(\w+)\}/g, (match, name) => (name in params ? String(params[name]) : match));
}

/** Fill [data-i18n] text and [data-i18n-attr="attr:key;attr:key"] attributes. */
export function applyTranslations(root) {
  for (const node of root.querySelectorAll('[data-i18n]')) node.textContent = t(node.dataset.i18n);
  for (const node of root.querySelectorAll('[data-i18n-attr]')) {
    for (const pair of node.dataset.i18nAttr.split(';')) {
      const [attr, key] = pair.split(':');
      node.setAttribute(attr, t(key));
    }
  }
}
