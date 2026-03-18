from __future__ import annotations

import logging
import tempfile
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any

from ..config import get_settings

logger = logging.getLogger(__name__)


class SttError(RuntimeError):
    """Base STT error."""


class SttUnavailableError(SttError):
    """Raised when local STT runtime is unavailable."""


class SttNoSpeechError(SttError):
    """Raised when audio is valid but no speech is detected."""


@lru_cache(maxsize=1)
def _get_whisper_model() -> Any:
    settings = get_settings()
    if not settings.stt_enabled:
        raise SttUnavailableError("Local STT is disabled by configuration")
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as exc:
        raise SttUnavailableError("faster-whisper is not installed on this runtime") from exc

    logger.info(
        "stt.model_loading model=%s device=%s compute_type=%s",
        settings.stt_model_size,
        settings.stt_device,
        settings.stt_compute_type,
    )
    return WhisperModel(
        settings.stt_model_size,
        device=settings.stt_device,
        compute_type=settings.stt_compute_type,
    )


def transcribe_audio_bytes(
    *,
    audio_bytes: bytes,
    suffix: str = ".webm",
    language: str | None = "en",
) -> tuple[str, int]:
    """
    Transcribe audio bytes with local Faster-Whisper.

    Returns transcript and elapsed latency_ms.
    """
    started_at = perf_counter()
    if not audio_bytes:
        raise SttNoSpeechError("No audio data was captured")

    model = _get_whisper_model()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(audio_bytes)
        temp_path = Path(handle.name)

    try:
        settings = get_settings()
        segments, _ = model.transcribe(
            str(temp_path),
            language=language,
            vad_filter=True,
            no_speech_threshold=settings.stt_no_speech_threshold,
            condition_on_previous_text=False,
            beam_size=1,
        )
        transcript_parts = [segment.text.strip() for segment in segments if str(segment.text or "").strip()]
        transcript = " ".join(transcript_parts).strip()
        if not transcript:
            raise SttNoSpeechError("No speech detected in recording")
        latency_ms = int((perf_counter() - started_at) * 1000)
        return transcript, latency_ms
    except SttNoSpeechError:
        raise
    except Exception as exc:
        raise SttError("Transcription failed") from exc
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("stt.tempfile_cleanup_failed path=%s", temp_path)
