"""User-facing message catalog for the API (English default, Vietnamese optional)."""

SUPPORTED = ("en", "vi")
DEFAULT = "en"

MESSAGES = {
    "busy": {
        "en": "The server is analyzing another image. Please try again in a few seconds.",
        "vi": "Máy chủ đang phân tích ảnh khác. Vui lòng thử lại sau vài giây.",
    },
    "single_fields": {
        "en": "Send exactly one image and one symptom list.",
        "vi": "Chỉ gửi một ảnh và một danh sách triệu chứng.",
    },
    "missing_image": {
        "en": "Missing image in the 'image' field.",
        "vi": "Thiếu ảnh ở trường image.",
    },
    "model_unavailable": {
        "en": "The model is not ready. Only symptoms were assessed; "
        "an administrator needs to check the server.",
        "vi": "Model chưa sẵn sàng. Chỉ đánh giá triệu chứng; "
        "quản trị viên cần kiểm tra máy chủ.",
    },
    "not_multipart": {
        "en": "The request must be multipart/form-data.",
        "vi": "Yêu cầu phải là multipart/form-data.",
    },
    "bad_content_length": {
        "en": "Invalid Content-Length.",
        "vi": "Content-Length không hợp lệ.",
    },
    "too_large": {
        "en": "The image exceeds the 5 MB limit.",
        "vi": "Ảnh vượt quá giới hạn 5 MB.",
    },
    "upload_timeout": {
        "en": "The upload took too long. Please try again.",
        "vi": "Upload quá lâu. Vui lòng thử lại.",
    },
    "symptoms_not_json": {
        "en": "symptoms must be a JSON list string.",
        "vi": "symptoms phải là chuỗi JSON dạng danh sách.",
    },
    "symptoms_invalid": {
        "en": "Invalid symptom list.",
        "vi": "Danh sách triệu chứng không hợp lệ.",
    },
    "image_size": {
        "en": "The image must be between 1 byte and 5 MB.",
        "vi": "Ảnh phải có dung lượng từ 1 byte đến 5 MB.",
    },
    "image_format": {
        "en": "Only JPEG, PNG or WebP images are accepted.",
        "vi": "Chỉ nhận ảnh JPEG, PNG hoặc WebP.",
    },
    "image_animated": {
        "en": "Please use a still image.",
        "vi": "Vui lòng dùng ảnh tĩnh.",
    },
    "image_dimensions": {
        "en": "The image must be at least 64×64 and no larger than 25 megapixels.",
        "vi": "Ảnh cần ít nhất 64×64 và không quá 25 megapixel.",
    },
    "image_unreadable": {
        "en": "The image could not be read. Please choose another clear eye photo.",
        "vi": "Không đọc được ảnh. Hãy chọn ảnh mắt rõ nét khác.",
    },
    "image_invalid": {
        "en": "The image was recognized as Back Ground. Retake a close-up of the eye "
        "in good light, with the eye centered.",
        "vi": "Ảnh được nhận diện là Back Ground. Hãy chụp lại cận cảnh mắt, đủ sáng, "
        "mắt nằm giữa ảnh.",
    },
    "image_uncertain": {
        "en": "The model is not confident enough. Please retake the photo; "
        "the symptom warnings below still apply.",
        "vi": "Model chưa đủ chắc chắn. Hãy chụp lại ảnh; "
        "cảnh báo triệu chứng bên dưới vẫn có giá trị.",
    },
    "image_valid": {
        "en": "Image analyzed. The model score is not a probability of disease.",
        "vi": "Đã phân tích ảnh. Điểm model không phải xác suất mắc bệnh.",
    },
    "level_high": {"en": "High", "vi": "Cao"},
    "level_medium": {"en": "Medium", "vi": "Vừa"},
    "level_low": {"en": "Low", "vi": "Thấp"},
    "level_none": {"en": "Not assessed", "vi": "Chưa đánh giá"},
    "advice_urgent": {
        "en": "Go to a medical facility or eye specialist now to be assessed. "
        "Do not wait to retake the photo or for the AI result.",
        "vi": "Đến cơ sở y tế hoặc chuyên khoa mắt ngay để được đánh giá. "
        "Không chờ chụp lại ảnh hay kết quả AI.",
    },
    "advice_high": {
        "en": "A doctor should assess this soon, preferably today. "
        "If there is severe pain or reduced vision, seek care immediately.",
        "vi": "Cần được bác sĩ đánh giá sớm, ưu tiên trong ngày. "
        "Nếu đau nhiều hoặc giảm thị lực, đi khám ngay.",
    },
    "advice_medium": {
        "en": "Arrange an eye examination if symptoms persist, recur or worsen. "
        "If severe pain or vision changes appear, seek care immediately.",
        "vi": "Sắp xếp khám mắt nếu triệu chứng còn tiếp diễn, tái phát hoặc tăng lên. "
        "Nếu xuất hiện đau nhiều hay thay đổi thị lực, đi khám ngay.",
    },
    "advice_low": {
        "en": "No priority signs were found in the information provided. This does not "
        "rule out eye disease; get examined if you feel discomfort or are worried.",
        "vi": "Chưa ghi nhận dấu hiệu ưu tiên từ thông tin đã nhập. Điều này không loại "
        "trừ bệnh mắt; khám nếu có khó chịu hoặc lo lắng.",
    },
    "advice_none": {
        "en": "Not enough information to classify. Retake the eye photo and review "
        "the symptom checklist. If you are worried or the problem persists, see an "
        "eye doctor.",
        "vi": "Chưa đủ thông tin để phân loại. Hãy chụp lại ảnh mắt và kiểm tra "
        "checklist triệu chứng. Nếu lo lắng hoặc tình trạng kéo dài, hãy đi khám mắt.",
    },
    "reason_uncertain_hint": {
        "en": "The photo may show: {sign}. The image result is uncertain, so have it "
        "checked by a doctor.",
        "vi": "Ảnh có thể có: {sign}. Kết quả ảnh chưa chắc chắn, nên để bác sĩ "
        "kiểm tra.",
    },
}


def resolve_lang(value) -> str:
    """Map an optional ?lang= value to a supported language, defaulting to English."""
    value = (value or "").strip().lower()
    return value if value in SUPPORTED else DEFAULT


def t(key: str, lang: str = DEFAULT) -> str:
    return MESSAGES[key][resolve_lang(lang)]


def localize(value, lang: str = DEFAULT):
    """Pick one language from a knowledge-base {"en": ..., "vi": ...} text object."""
    if isinstance(value, dict):
        return value.get(resolve_lang(lang)) or value[DEFAULT]
    return value
