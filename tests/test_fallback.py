import asyncio

import pytest
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

from uqguard import ModelFallbackMiddleware


def _req(model="primary"):
    return ModelRequest(model=model, messages=[HumanMessage("hi")])


def _ok():
    return ModelResponse(result=[AIMessage("ok")])


@pytest.mark.parametrize(
    "err",
    [
        RuntimeError("429 RESOURCE_EXHAUSTED"),
        ConnectionError("connection refused"),
        TimeoutError("request timed out"),
        RuntimeError("503 Service Unavailable"),
    ],
)
def test_provider_errors_trigger_fallback(err):
    mw = ModelFallbackMiddleware("sentinel")
    mw._fallback = fallback = object()
    seen = []

    def flaky(request):
        seen.append(request.model)
        if len(seen) == 1:
            raise err
        return _ok()

    mw.wrap_model_call(_req(), flaky)
    assert seen == ["primary", fallback] and mw.active


@pytest.mark.parametrize(
    "err",
    [
        TypeError("'NoneType' object is not subscriptable"),
        AttributeError("no attribute 'tool_calls'"),
        KeyError("args"),
        ValueError("some downstream bug"),
    ],
)
def test_programming_errors_reraise(err):
    mw = ModelFallbackMiddleware("sentinel")

    def buggy(request):
        raise err

    with pytest.raises(type(err)):
        mw.wrap_model_call(_req(), buggy)
    assert not mw.active  # bug must not flip the sticky fallback


def test_async_fallback():
    mw = ModelFallbackMiddleware("sentinel")
    mw._fallback = fallback = object()
    seen = []

    async def flaky(request):
        seen.append(request.model)
        if len(seen) == 1:
            raise RuntimeError("rate limit exceeded")
        return _ok()

    asyncio.run(mw.awrap_model_call(_req(), flaky))
    assert seen == ["primary", fallback] and mw.active
