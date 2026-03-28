"""
DSPy-compatible LM that runs Gemini Nano via Chrome's Built-in AI API.

Flow:
  1. Launch Chrome Canary with flags pre-set in Local State
  2. Use Playwright to execute JS calling the LanguageModel global API
  3. Return OpenAI-compatible response for DSPy consumption

Advantages over the adb bridge:
  - No Android device needed
  - No rate limiting from ML Kit
  - No app crashes from repeated inference
  - Direct JS API — lower latency, more reliable

First run downloads the ~1.5GB Nano model (cached in persistent profile).

Usage:
    from gemini_nano_chrome_lm import GeminiNanoChromeLM

    lm = GeminiNanoChromeLM()
    lm.ensure_model_ready()  # Downloads model if needed (first run only)

    # Use standalone
    result = lm(messages=[{"role": "user", "content": "What food is this?"}])

    # Or with DSPy
    import dspy
    dspy.configure(lm=lm)
"""

import base64
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Minimal OpenAI-compatible response objects (same as gemini_nano_lm.py)
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
    model: str = "gemini-nano-chrome"
    id: str = field(default_factory=lambda: f"nano-chrome-{uuid.uuid4().hex[:8]}")


# ---------------------------------------------------------------------------
# Chrome configuration
# ---------------------------------------------------------------------------

CHROME_CANARY_BIN = "/usr/bin/google-chrome-canary"
PROFILE_DIR = Path.home() / "media" / "chrome-nano-canary-profile"

# Flags are set via Local State file, not --enable-features
# (Playwright's default args conflict with --enable-features)
REQUIRED_FLAGS = [
    "optimization-guide-on-device-model@1",
    "prompt-api-for-gemini-nano@1",
    "prompt-api-for-gemini-nano-multimodal-input@1",
]

# ---------------------------------------------------------------------------
# JS snippets — use the new LanguageModel global (not window.ai)
# ---------------------------------------------------------------------------

JS_CHECK_AVAILABILITY = """
async () => {
    if (typeof LanguageModel === 'undefined') {
        return JSON.stringify({status: "unsupported", error: "LanguageModel global not available"});
    }
    try {
        const avail = await LanguageModel.availability();
        return JSON.stringify({status: avail});
    } catch (e) {
        return JSON.stringify({status: "error", error: e.message});
    }
}
"""

JS_TRIGGER_DOWNLOAD = """
async () => {
    try {
        const session = await LanguageModel.create();
        session.destroy();
        return JSON.stringify({status: "downloaded"});
    } catch (e) {
        return JSON.stringify({status: "error", error: e.message});
    }
}
"""

JS_TEXT_PROMPT = """
async ([promptText]) => {
    try {
        const session = await LanguageModel.create();
        const result = await session.prompt(promptText);
        session.destroy();
        return JSON.stringify({result: result, error: null});
    } catch (e) {
        return JSON.stringify({result: null, error: e.message});
    }
}
"""

JS_MULTIMODAL_PROMPT = """
async ([promptText, imageBase64, mimeType]) => {
    try {
        const binary = atob(imageBase64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {
            bytes[i] = binary.charCodeAt(i);
        }
        const blob = new Blob([bytes], {type: mimeType});
        const imageBitmap = await createImageBitmap(blob);

        const session = await LanguageModel.create({
            expectedInputs: [{type: "image"}],
        });
        const result = await session.prompt([{
            role: "user",
            content: [
                {type: "image", value: imageBitmap},
                {type: "text", value: promptText},
            ],
        }]);
        session.destroy();
        return JSON.stringify({result: result, error: null});
    } catch (e) {
        return JSON.stringify({result: null, error: e.message});
    }
}
"""


# ---------------------------------------------------------------------------
# Chrome flag setup
# ---------------------------------------------------------------------------

def _ensure_flags_in_local_state(profile_dir: Path) -> None:
    """Write required chrome://flags overrides into Chrome's Local State file."""
    profile_dir.mkdir(parents=True, exist_ok=True)
    local_state_path = profile_dir / "Local State"

    state = {}
    if local_state_path.exists():
        state = json.loads(local_state_path.read_text())

    if "browser" not in state:
        state["browser"] = {}

    existing = set(state["browser"].get("enabled_labs_experiments", []))
    needed = set(REQUIRED_FLAGS)

    if not needed.issubset(existing):
        state["browser"]["enabled_labs_experiments"] = list(existing | needed)
        local_state_path.write_text(json.dumps(state))


# ---------------------------------------------------------------------------
# Image extraction from DSPy messages
# ---------------------------------------------------------------------------

def _extract_image_and_text(messages: list[dict]) -> tuple[tuple[str, str] | None, str]:
    """
    Extract image (as base64 + mime) and text prompt from DSPy messages.

    Returns:
        (image_b64, mime_type) or None, text_prompt
    """
    text_parts = []
    image_data = None

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
                            header, b64data = url.split(",", 1)
                            mime = "image/jpeg"
                            if "png" in header:
                                mime = "image/png"
                            image_data = (b64data, mime)
                        elif os.path.isfile(url):
                            img_bytes = Path(url).read_bytes()
                            b64 = base64.b64encode(img_bytes).decode()
                            ext = Path(url).suffix.lower()
                            mime = "image/png" if ext == ".png" else "image/jpeg"
                            image_data = (b64, mime)

    return image_data, "\n".join(text_parts)


# ---------------------------------------------------------------------------
# GeminiNanoChromeLM
# ---------------------------------------------------------------------------

class GeminiNanoChromeLM:
    """
    DSPy-compatible LM that runs inference on Gemini Nano via Chrome Built-in AI.

    Uses Playwright to control a Chrome Canary instance with the LanguageModel
    API enabled via chrome://flags. The Chrome profile (including the downloaded
    model) persists across runs at ~/media/chrome-nano-canary-profile/.
    """

    def __init__(
        self,
        chrome_bin: str = CHROME_CANARY_BIN,
        profile_dir: str | Path = PROFILE_DIR,
        headless: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 256,
        timeout: float = 60.0,
        **kwargs: Any,
    ):
        self.model = "gemini-nano-chrome"
        self.model_type = "chat"
        self.cache = False
        self.kwargs = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.history: list[dict] = []

        self.chrome_bin = chrome_bin
        self.profile_dir = Path(profile_dir)
        self.headless = headless
        self.timeout = timeout

        self._playwright = None
        self._browser = None
        self._page = None

        self._call_count = 0
        self._total_time = 0.0

    def _launch_browser(self):
        """Launch Chrome Canary with Built-in AI flags via Playwright."""
        if self._page is not None:
            return

        from playwright.sync_api import sync_playwright

        _ensure_flags_in_local_state(self.profile_dir)

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch_persistent_context(
            user_data_dir=str(self.profile_dir),
            executable_path=self.chrome_bin,
            headless=self.headless,
            ignore_default_args=True,
            args=[
                "--remote-debugging-pipe",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-background-timer-throttling",
                "--disable-renderer-backgrounding",
                "--disable-backgrounding-occluded-windows",
                f"--user-data-dir={self.profile_dir}",
            ],
        )
        self._page = self._browser.new_page()
        self._page.goto("https://example.com", wait_until="domcontentloaded")
        self._page.wait_for_timeout(2000)

    def check_availability(self) -> dict:
        """Check if Gemini Nano model is available in Chrome."""
        self._launch_browser()
        result = self._page.evaluate(JS_CHECK_AVAILABILITY)
        return json.loads(result)

    def ensure_model_ready(self, max_wait: float = 600.0) -> bool:
        """
        Ensure the Nano model is downloaded and ready.
        On first run, this triggers the ~1.5GB download and waits.
        Returns True if model is ready.
        """
        self._launch_browser()

        status = self.check_availability()
        print(f"Model availability: {status}")

        if status.get("status") == "available":
            print("Gemini Nano model is ready.")
            return True

        if status.get("status") == "unsupported":
            print(f"ERROR: Built-in AI not available: {status.get('error')}")
            return False

        if status.get("status") in ("after-download", "downloadable", "downloading"):
            print("Triggering model download... (this may take a few minutes)")
            try:
                result = self._page.evaluate(JS_TRIGGER_DOWNLOAD)
                dl_status = json.loads(result)
                print(f"Download trigger result: {dl_status}")
            except Exception as e:
                print(f"Download trigger error (expected during download): {e}")

            start = time.time()
            while time.time() - start < max_wait:
                avail = self.check_availability()
                if avail.get("status") == "available":
                    print("Model download complete — ready.")
                    return True
                print(f"  Still downloading... ({time.time() - start:.0f}s)")
                time.sleep(15)

            print(f"Timed out waiting for model download after {max_wait}s")
            return False

        print(f"Unexpected status: {status}")
        return False

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
        self._launch_browser()
        start = time.time()
        self._call_count += 1

        image_data, text_prompt = _extract_image_and_text(messages or [])

        if not text_prompt and prompt:
            text_prompt = prompt

        if not text_prompt:
            return _Response(
                choices=[_Choice(message=_Message(content="ERROR:no_prompt"))],
            )

        try:
            if image_data:
                b64, mime = image_data
                result_json = self._page.evaluate(
                    JS_MULTIMODAL_PROMPT,
                    [text_prompt, b64, mime],
                )
            else:
                result_json = self._page.evaluate(
                    JS_TEXT_PROMPT,
                    [text_prompt],
                )

            result = json.loads(result_json)
            if result.get("error"):
                result_text = f"ERROR:{result['error']}"
            else:
                result_text = result.get("result", "")

        except Exception as e:
            result_text = f"ERROR:{type(e).__name__}:{e}"

        duration = time.time() - start
        self._total_time += duration

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
        new._playwright = None
        new._browser = None
        new._page = None
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

    def close(self):
        """Shut down the browser."""
        if self._browser:
            self._browser.close()
            self._browser = None
            self._page = None
        if self._playwright:
            self._playwright.stop()
            self._playwright = None

    def __del__(self):
        self.close()


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    lm = GeminiNanoChromeLM(headless="--headless" in sys.argv)
    print("Checking model availability...")

    if not lm.ensure_model_ready():
        print("Model not ready — exiting")
        sys.exit(1)

    # Quick text-only test
    print("\n--- Text-only test ---")
    result = lm(prompt="What is 2 + 2? Answer with just the number.")
    print(f"Result: {result}")

    # Image test if provided
    image_args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if image_args:
        image_path = image_args[0]
        print(f"\n--- Image test: {image_path} ---")
        img_bytes = Path(image_path).read_bytes()
        b64 = base64.b64encode(img_bytes).decode()
        ext = Path(image_path).suffix.lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": "What food is in this image? List the items."},
                ],
            }
        ]
        result = lm(messages=messages)
        print(f"Result: {result}")

    print(f"\nStats: {lm.stats()}")
    lm.close()
