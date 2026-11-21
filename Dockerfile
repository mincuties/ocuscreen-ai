FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=-1 \
    TF_NUM_INTRAOP_THREADS=2 \
    TF_NUM_INTEROP_THREADS=1 \
    OMP_NUM_THREADS=2 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=7860
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=appuser:appuser backend ./backend
COPY --chown=appuser:appuser data ./data
COPY --chown=appuser:appuser frontend ./frontend
COPY --chown=appuser:appuser model ./model
# model/ also holds labels.txt, so COPY succeeds without the weights; fail the build instead
# of shipping an image that answers 503 forever.
RUN test -s model/keras_model.h5 || { echo "model/keras_model.h5 is missing; download it as described in README.md" >&2; exit 1; }
USER appuser
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/health' % os.environ.get('PORT', '7860'), timeout=4)"
CMD ["sh", "-c", "exec uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1 --limit-concurrency 16 --timeout-keep-alive 5 --no-access-log"]
