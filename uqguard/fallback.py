"""Sticky model fallback: when the primary model fails with a provider-side
error (rate limit, quota, timeout, 5xx, connection), swap in the configured
fallback model for the rest of the run. Anything that looks like a
programming error is re-raised -- a fallback model can't fix a bug and must
not mask one.

Place FIRST in the middleware list (outermost), so a failure inside
CaptureMiddleware's k-loop rewinds the whole step onto the fallback model
instead of mixing models within one step's candidates.

`active` is deliberately process-global, not per-conversation: a provider
outage affects every conversation on this agent.
"""

import logging

from langchain.agents.middleware import AgentMiddleware

log = logging.getLogger("uqguard")

# programming errors: never mask with a fallback
_BUGS = (TypeError, AttributeError, KeyError, IndexError, NameError, AssertionError)

# provider/network failure fingerprints, matched against class name + message
_PROVIDER = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "rate limit",
    "ratelimit",
    "quota",
    "resource_exhausted",
    "overloaded",
    "timeout",
    "timed out",
    "connection",
    "connect",
    "unavailable",
    "temporarily",
    "server error",
    "apierror",
    "apistatuserror",
    "apiconnection",
    "internalservererror",
    "servicetier",
)


def _provider_error(e: Exception) -> bool:
    if isinstance(e, _BUGS):
        return False
    text = f"{type(e).__name__} {e}".lower()
    return any(m in text for m in _PROVIDER)


class ModelFallbackMiddleware(AgentMiddleware):
    def __init__(self, fallback_model):
        super().__init__()
        self._fallback = fallback_model  # BaseChatModel or init_chat_model string
        self.active = False

    def _model(self):
        if isinstance(self._fallback, str):
            from langchain.chat_models import init_chat_model

            # temperature must stay > 0 after the swap or k-sampling collapses;
            # pass a configured model instance instead of a string to control it
            self._fallback = init_chat_model(self._fallback, temperature=0.7)
        return self._fallback

    def _trip(self, e: Exception) -> None:
        """Activate the fallback for a provider error; re-raise anything else."""
        if not _provider_error(e):
            raise e
        log.warning(
            "primary model failed (%s: %s), switching to fallback model",
            type(e).__name__,
            str(e)[:120],
        )
        self.active = True

    def wrap_model_call(self, request, handler):
        if self.active:
            return handler(request.override(model=self._model()))
        try:
            return handler(request)
        except Exception as e:
            self._trip(e)
            return handler(request.override(model=self._model()))

    async def awrap_model_call(self, request, handler):
        if self.active:
            return await handler(request.override(model=self._model()))
        try:
            return await handler(request)
        except Exception as e:
            self._trip(e)
            return await handler(request.override(model=self._model()))
