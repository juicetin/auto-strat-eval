"""
DSPy BaseLM implementation that bridges to on-device Gemini Nano via adb.

Flow:
  1. DSPy calls forward() with messages containing text + image
  2. We extract the image (base64 data URI → file), push to device
  3. Trigger VlmEvalReceiver via adb broadcast
  4. Poll for result sentinel file
  5. Read result, return OpenAI-compatible response

Usage:
    import dspy
    from gemini_nano_lm import GeminiNanoLM

    lm = GeminiNanoLM(serial="2A281FDH3002TN")
    dspy.configure(lm=lm)
"""

import base64
import json
import os
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Minimal OpenAI-compatible response objects (no openai dependency needed)
# ---------------------------------------------------------------------------

@dataclass
class _Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __iter__(self):
        yield "prompt_tokens", self.prompt_tokens
        yield "completion_tokens", self.completion_tokens
        yield "total_tokens", self.total_tokens


@dataclass
class _Message:
    content: str = ""
    role: str = "assistant"


@dataclass
class _Choice:
    message: _Message = field(default_factory=_Message)
    index: int = 0
    finish_reason: str = "stop"


@dataclass
class _Response:
    choices: list = field(default_factory=list)
    usage: _Usage = field(default_factory=_Usage)
    model: str = "gemini-nano"
    id: str = field(default_factory=lambda: f"nano-{uuid.uuid4().hex[:8]}")


# ---------------------------------------------------------------------------
# ADB helpers
# ---------------------------------------------------------------------------

def _adb(serial: str, *args: str, timeout: int = 30) -> str:
    """Run an adb command targeting a specific device."""
    cmd = ["adb", "-s", serial] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"adb failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _adb_push(serial: str, local_path: str, remote_path: str) -> None:
    _adb(serial, "push", local_path, remote_path)


def _adb_pull(serial: str, remote_path: str, local_path: str) -> None:
    _adb(serial, "pull", remote_path, local_path)


def _adb_shell(serial: str, cmd: str, timeout: int = 30) -> str:
    return _adb(serial, "shell", cmd, timeout=timeout)


def _adb_file_exists(serial: str, path: str) -> bool:
    result = _adb_shell(serial, f'[ -f "{path}" ] && echo YES || echo NO')
    return result.strip() == "YES"


def _adb_rm(serial: str, path: str) -> None:
    _adb_shell(serial, f'rm -f "{path}"')


def _adb_read_file(serial: str, remote_path: str) -> str:
    """Pull a file from device and read its contents."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        local_path = f.name
    try:
        _adb_pull(serial, remote_path, local_path)
        return Path(local_path).read_text()
    finally:
        os.unlink(local_path)


# ---------------------------------------------------------------------------
# Image extraction from DSPy messages
# ---------------------------------------------------------------------------

def _extract_image_and_text(messages: list[dict]) -> tuple[str | None, str]:
    """
    Extract image (as local file path) and text prompt from DSPy messages.

    DSPy sends images as base64 data URIs in message content blocks:
    [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}]
    or as a plain file path.
    """
    text_parts = []
    image_path = None

    for msg in (messages or []):
        content = msg.get("content", "")
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, str):
                    text_parts.append(block)
                elif isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block["text"])
                    elif block.get("type") == "image_url":
                        url = block.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            # base64 data URI → decode to temp file
                            header, b64data = url.split(",", 1)
                            ext = ".jpg"
                            if "png" in header:
                                ext = ".png"
                            tmp = tempfile.NamedTemporaryFile(
                                suffix=ext, delete=False, dir="/tmp"
                            )
                            tmp.write(base64.b64decode(b64data))
                            tmp.close()
                            image_path = tmp.name
                        elif os.path.isfile(url):
                            image_path = url
                        else:
                            # Might be a regular URL — not supported on-device
                            text_parts.append(f"[image: {url}]")

    return image_path, "\n".join(text_parts)


# ---------------------------------------------------------------------------
# GeminiNanoLM
# ---------------------------------------------------------------------------

DEVICE_EVAL_DIR = "/sdcard/Android/data/com.jtingexpo.mobile/files/eval"
DEVICE_IMAGE_PATH = f"{DEVICE_EVAL_DIR}/input.jpg"
DEVICE_PROMPT_PATH = f"{DEVICE_EVAL_DIR}/prompt.txt"
DEVICE_RESULT_PATH = f"{DEVICE_EVAL_DIR}/result.txt"
DEVICE_DONE_PATH = f"{DEVICE_RESULT_PATH}.done"

# Receiver coordinates
RECEIVER_PACKAGE = "com.jtingexpo.mobile"
RECEIVER_CLASS = "expo.modules.geminanano.VlmEvalReceiver"
BROADCAST_ACTION = "com.foodtracker.IDENTIFY_FOOD"


class GeminiNanoLM:
    """
    DSPy-compatible LM that runs inference on Gemini Nano via a connected Android device.

    Subclasses dspy.BaseLM by duck-typing (inheriting would require dspy as a dependency
    at import time — we defer the import so the module works standalone for testing).
    At runtime, set this as the LM via dspy.configure(lm=lm).
    """

    def __init__(
        self,
        serial: str = "48181FDAP00A1U",
        poll_interval: float = 0.5,
        max_wait: float = 30.0,
        temperature: float = 0.2,
        max_tokens: int = 256,
        cooldown: float = 3.0,
        max_retries: int = 3,
        retry_backoff: float = 30.0,
        **kwargs: Any,
    ):
        self.model = "gemini-nano"
        self.model_type = "chat"
        self.cache = False  # No caching — each device call is unique
        self.kwargs = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.history: list[dict] = []

        self.serial = serial
        self.poll_interval = poll_interval
        self.max_wait = max_wait
        self.cooldown = cooldown          # Seconds between calls to avoid rate limiting
        self.max_retries = max_retries    # Retry on rate limit (ErrorCode 9)
        self.retry_backoff = retry_backoff  # Seconds to wait on rate limit before retry

        # Track timing stats
        self._call_count = 0
        self._total_time = 0.0
        self._last_call_time = 0.0

        # Ensure eval dir exists on device
        _adb_shell(serial, f'mkdir -p "{DEVICE_EVAL_DIR}"')

    def __call__(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> list[str]:
        response = self.forward(prompt=prompt, messages=messages, **kwargs)
        return self._process(response)

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> "_Response":
        # Cooldown between calls to avoid rate limiting
        since_last = time.time() - self._last_call_time
        if since_last < self.cooldown and self._last_call_time > 0:
            time.sleep(self.cooldown - since_last)

        start = time.time()
        self._call_count += 1

        # Extract image and text from messages
        local_image, text_prompt = _extract_image_and_text(messages or [])

        # Use prompt string as fallback if messages didn't have text
        if not text_prompt and prompt:
            text_prompt = prompt

        if not text_prompt:
            return _Response(
                choices=[_Choice(message=_Message(content="ERROR:no_prompt"))],
            )

        # Push image to device (if we have one)
        if local_image:
            _adb_push(self.serial, local_image, DEVICE_IMAGE_PATH)
            device_image = DEVICE_IMAGE_PATH
        else:
            device_image = ""

        # Write prompt to file on device (avoids shell escaping issues)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as pf:
            pf.write(text_prompt)
            prompt_local = pf.name

        _adb_push(self.serial, prompt_local, DEVICE_PROMPT_PATH)
        os.unlink(prompt_local)

        # Retry loop for rate limiting (ErrorCode 9)
        result_text = "ERROR:max_retries_exceeded"
        for attempt in range(self.max_retries):
            # Clean previous result
            _adb_rm(self.serial, DEVICE_RESULT_PATH)
            _adb_rm(self.serial, DEVICE_DONE_PATH)

            # Trigger broadcast — receiver reads prompt from prompt_path file
            # Run in background (&) because am broadcast blocks until receiver's goAsync() finishes
            _adb_shell(
                self.serial,
                f'nohup am broadcast -a {BROADCAST_ACTION} '
                f'--es image_path "{device_image}" '
                f'--es prompt_path "{DEVICE_PROMPT_PATH}" '
                f'--es result_path "{DEVICE_RESULT_PATH}" '
                f'{RECEIVER_PACKAGE}/{RECEIVER_CLASS} > /dev/null 2>&1 &',
                timeout=5,
            )

            # Poll for completion
            poll_elapsed = 0.0
            while poll_elapsed < self.max_wait:
                if _adb_file_exists(self.serial, DEVICE_DONE_PATH):
                    break
                time.sleep(self.poll_interval)
                poll_elapsed += self.poll_interval
            else:
                result_text = "ERROR:timeout"
                break

            # Read result
            result_text = _adb_read_file(self.serial, DEVICE_RESULT_PATH)

            # Check for rate limit error — retry with backoff
            if "ErrorCode 9" in result_text or "usage quota" in result_text:
                if attempt < self.max_retries - 1:
                    wait = self.retry_backoff * (attempt + 1)
                    print(f"  [rate-limited] Waiting {wait:.0f}s before retry {attempt + 2}/{self.max_retries}...")
                    time.sleep(wait)
                    continue
            # Success or non-retryable error
            break

        self._last_call_time = time.time()
        duration = time.time() - start
        self._total_time += duration

        # Estimate token counts (rough: 4 chars ≈ 1 token)
        prompt_tokens = len(text_prompt) // 4
        completion_tokens = len(result_text) // 4

        return _Response(
            choices=[_Choice(message=_Message(content=result_text))],
            usage=_Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    def _process(self, response: "_Response") -> list[str]:
        outputs = [c.message.content for c in response.choices]

        entry = {
            "prompt": None,
            "messages": None,
            "kwargs": {},
            "response": response,
            "outputs": outputs,
            "usage": dict(response.usage),
            "cost": None,
            "model": self.model,
            "response_model": response.model,
            "model_type": self.model_type,
        }
        self.history.append(entry)
        if len(self.history) > 100:
            self.history.pop(0)

        return outputs

    def copy(self, **kwargs):
        import copy as copy_mod
        new = copy_mod.deepcopy(self)
        new.history = []
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(new, k, v)
            if k in self.kwargs:
                new.kwargs[k] = v
        return new

    def inspect_history(self, n: int = 1):
        for entry in self.history[-n:]:
            print(f"--- {entry['model']} ---")
            for o in entry["outputs"]:
                print(o[:200])

    @property
    def avg_latency(self) -> float:
        return self._total_time / max(self._call_count, 1)

    def stats(self) -> dict:
        return {
            "calls": self._call_count,
            "total_time": round(self._total_time, 2),
            "avg_latency": round(self.avg_latency, 2),
        }


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    serial = sys.argv[1] if len(sys.argv) > 1 else "2A281FDH3002TN"
    image = sys.argv[2] if len(sys.argv) > 2 else None

    lm = GeminiNanoLM(serial=serial)
    print(f"GeminiNanoLM ready — device {serial}")
    print(f"Stats: {lm.stats()}")

    if image:
        # Quick smoke test
        from pathlib import Path
        img_bytes = Path(image).read_bytes()
        b64 = base64.b64encode(img_bytes).decode()
        mime = "image/jpeg" if image.endswith(".jpg") else "image/png"

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            'Identify all food in this image. Return only valid JSON:\n'
                            '{"dishes":[{"name":string,"cuisine":string,"ingredients":[{"name":string,"amount_g":number}]}]}'
                        ),
                    },
                ],
            }
        ]
        result = lm(messages=messages)
        print(f"\nResult: {result}")
        print(f"Stats: {lm.stats()}")
    else:
        print("Pass an image path as second arg for a smoke test")
