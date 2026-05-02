import logging
import time

import requests


logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        num_ctx: int,
        num_predict: int,
        timeout: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        try:
            start = time.perf_counter()
            logger.info("ollama_generate_started model=%s prompt_chars=%s", self.model, len(prompt))
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_ctx": self.num_ctx,
                        "num_predict": self.num_predict,
                    },
                },
                timeout=self.timeout,
            )
        except requests.RequestException:
            logger.exception("ollama_generate_request_failed base_url=%s model=%s", self.base_url, self.model)
            raise
        if not response.ok:
            detail = response.text.strip()
            logger.error(
                "ollama_generate_failed status_code=%s model=%s detail=%s",
                response.status_code,
                self.model,
                detail,
            )
            raise RuntimeError(f"Ollama generate failed with HTTP {response.status_code}: {detail}")
        payload = response.json()
        answer = str(payload.get("response", "")).strip()
        logger.info(
            "ollama_generate_completed model=%s response_chars=%s duration_ms=%.2f",
            self.model,
            len(answer),
            (time.perf_counter() - start) * 1000,
        )
        return answer
