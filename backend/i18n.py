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
        "en": "This does not look like a photo of an eye. Retake a close-up of the eye "
        "in good light, with the eye centered.",
        "vi": "Ảnh này có vẻ không phải ảnh chụp mắt. Hãy chụp lại cận cảnh mắt, đủ sáng, "
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
    "image_unassessed": {
        "en": "This model does not check photos for: {signs}. Their absence here is not a normal result.",
        "vi": "Model này chưa kiểm tra ảnh về: {signs}. Không thấy ở đây không có nghĩa là bình thường.",
    },
    "attention_note": {
        "en": "Coloured areas show which parts of the photo raised the score of this label "
        "(red = most, compared within this photo only). It is not the location of a "
        "lesion and is not a diagnosis.",
        "vi": "Vùng tô màu cho thấy phần nào của ảnh làm tăng điểm của nhãn này (đỏ là "
        "nhiều nhất, chỉ so sánh trong ảnh này). Đây không phải vị trí tổn thương và "
        "không phải chẩn đoán.",
    },
    "quality_blurry": {
        "en": "The photo looks blurry. Hold the phone steady and tap the eye to focus.",
        "vi": "Ảnh có vẻ bị mờ. Giữ máy chắc tay và chạm vào mắt trên màn hình để lấy nét.",
    },
    "quality_dark": {
        "en": "The photo looks too dark. Move to a brighter place, facing the light.",
        "vi": "Ảnh có vẻ quá tối. Hãy ra chỗ sáng hơn, quay mặt về phía nguồn sáng.",
    },
    "quality_bright": {
        "en": "The photo looks too bright. Avoid direct sunlight and turn off the flash.",
        "vi": "Ảnh có vẻ quá sáng. Tránh nắng chiếu thẳng và tắt đèn flash.",
    },
    "quality_glare": {
        "en": "There is strong glare on the photo. Turn off the flash and avoid lights "
        "reflecting in the eye.",
        "vi": "Ảnh bị lóa mạnh. Tắt đèn flash và tránh đèn phản chiếu vào mắt.",
    },
    "level_red": {"en": "Red · go now", "vi": "Đỏ · đi khám ngay"},
    "level_yellow": {"en": "Yellow · see a doctor", "vi": "Vàng · cần đi khám"},
    "level_green": {"en": "Green · self-care and watch", "vi": "Xanh · tự theo dõi"},
    "level_none": {"en": "Not assessed", "vi": "Chưa đánh giá"},
    "advice_red": {
        "en": "Seek care now. Do not wait to retake the photo or for the AI result.",
        "vi": "Đi khám ngay. Không chờ chụp lại ảnh hay kết quả AI.",
    },
    "advice_yellow": {
        "en": "Arrange an examination at the place shown below. If severe pain, "
        "vision changes or any danger sign appears, seek care immediately.",
        "vi": "Sắp xếp đi khám tại nơi gợi ý bên dưới. Nếu đau nhiều, thay đổi thị lực "
        "hoặc có dấu hiệu nguy hiểm, đi khám ngay.",
    },
    "advice_green": {
        "en": "No sign needing an examination was found in your answers. Rest your "
        "eyes and watch. See an eye doctor if it is not better within a few days, "
        "gets worse, or you are worried. This does not rule out eye disease.",
        "vi": "Chưa thấy dấu hiệu cần đi khám từ thông tin đã nhập. Cho mắt nghỉ ngơi và "
        "theo dõi. Đi khám mắt nếu không đỡ sau vài ngày, nặng hơn hoặc bạn lo lắng. "
        "Điều này không loại trừ bệnh mắt.",
    },
    "advice_none": {
        "en": "Not enough information to classify. Retake the eye photo and review "
        "the symptom checklist. If you are worried or the problem persists, see an "
        "eye doctor.",
        "vi": "Chưa đủ thông tin để phân loại. Hãy chụp lại ảnh mắt và kiểm tra "
        "checklist triệu chứng. Nếu lo lắng hoặc tình trạng kéo dài, hãy đi khám mắt.",
    },
    "advice_confirm": {
        "en": "Please choose who this is for, review the whole symptom list and confirm "
        "that you have ticked everything that applies, then run the screening again.",
        "vi": "Hãy chọn người cần sàng lọc, xem hết danh sách triệu chứng và xác nhận "
        "đã chọn mọi mục phù hợp, rồi phân tích lại.",
    },
    "dest_eye_emergency": {
        "en": "Eye emergency service or hospital emergency department, now",
        "vi": "Cấp cứu mắt hoặc khoa cấp cứu bệnh viện, ngay bây giờ",
    },
    "dest_emergency": {
        "en": "Hospital emergency department (children's emergency for a child), now",
        "vi": "Khoa cấp cứu bệnh viện (cấp cứu nhi nếu là trẻ em), ngay bây giờ",
    },
    "dest_eye_today": {
        "en": "Eye doctor today",
        "vi": "Bác sĩ mắt trong hôm nay",
    },
    "dest_doctor_today": {
        "en": "General doctor today (ask for an urgent appointment)",
        "vi": "Bác sĩ đa khoa trong hôm nay (xin khám sớm)",
    },
    "dest_eye_doctor": {
        "en": "Eye doctor within the next few days",
        "vi": "Bác sĩ mắt trong vài ngày tới",
    },
    "dest_doctor": {
        "en": "General doctor within the next few days, for a general check-up",
        "vi": "Bác sĩ đa khoa trong vài ngày tới, để kiểm tra toàn thân",
    },
    "dest_self_care": {
        "en": "Self-care at home; see an eye doctor if it does not improve",
        "vi": "Tự chăm sóc tại nhà; đi khám mắt nếu không đỡ",
    },
    "first_aid_rinse": {
        "en": "Rinse the eye straight away with plenty of clean running water for "
        "15 to 20 minutes, holding the eyelids open, then go to emergency care.",
        "vi": "Rửa mắt ngay bằng nhiều nước sạch đang chảy trong 15 đến 20 phút, giữ "
        "mí mắt mở, rồi đi cấp cứu.",
    },
    "first_aid_injury_care": {
        "en": "Do not rub or press on the eye and do not try to remove an object stuck "
        "in it.",
        "vi": "Không dụi, không ấn vào mắt và không tự lấy dị vật đang mắc trong mắt ra.",
    },
    "first_aid_remove_lenses": {
        "en": "Take your contact lenses out now and keep them out until a doctor has "
        "checked your eye. Bring the lenses and their case with you.",
        "vi": "Tháo kính áp tròng ngay và không đeo lại cho tới khi bác sĩ kiểm tra mắt. "
        "Mang theo kính và hộp đựng kính khi đi khám.",
    },
    "warning_steroids": {
        "en": "Do not use eye drops or ointments containing corticosteroids (for "
        "example dexamethasone or prednisolone) unless an eye doctor has examined you "
        "and prescribed them.",
        "vi": "Không tự dùng thuốc nhỏ hoặc mỡ tra mắt có corticoid (ví dụ dexamethason, "
        "prednisolon) khi chưa được bác sĩ mắt khám và kê đơn.",
    },
    "reason_mild": {
        "en": "Only mild symptoms: {symptoms}",
        "vi": "Chỉ có triệu chứng nhẹ: {symptoms}",
    },
    "reason_image_not_used": {
        "en": "The photo could not be used, so this result is based on your answers "
        "only.",
        "vi": "Không dùng được ảnh nên kết quả này chỉ dựa trên câu trả lời của bạn.",
    },
    "triage_field_invalid": {
        "en": "Invalid age group, duration or true/false value.",
        "vi": "Nhóm tuổi, thời gian kéo dài hoặc giá trị true/false không hợp lệ.",
    },
    "reason_image_display_only": {
        "en": "The photo may show: {sign}. This model is not reliable enough for that sign "
        "to change the result on its own. If you notice it, tick the matching symptom and "
        "run the check again.",
        "vi": "Ảnh có thể có: {sign}. Model chưa đủ tin cậy để dấu hiệu này tự thay đổi kết "
        "quả. Nếu bạn thấy có dấu hiệu đó, hãy đánh dấu triệu chứng tương ứng rồi kiểm tra lại.",
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
