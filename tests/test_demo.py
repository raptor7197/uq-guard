from demo_agent import FLIGHTS, TASKS, FakeTravelAPI, check, default_model, resolve_model


def test_openrouter_model_resolution(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.delenv("UQGUARD_MODEL", raising=False)
    assert default_model() == "openrouter:openai/gpt-4o-mini"
    m = resolve_model("openrouter:qwen/qwen3-14b")
    assert type(m).__name__ == "ChatOpenAI"
    assert m.model_name == "qwen/qwen3-14b"
    assert "openrouter.ai" in str(m.openai_api_base)

    monkeypatch.setenv("OLLAMA_API_KEY", "olk-test")
    m = resolve_model("ollama_cloud:gpt-oss:120b")
    assert type(m).__name__ == "ChatOllama"
    assert m.model == "gpt-oss:120b"
    assert m.base_url == "https://ollama.com"
    assert m.client_kwargs["headers"]["Authorization"] == "Bearer olk-test"

    m = resolve_model("ollama:qwen3:4b")  # plain prefixes go through init_chat_model
    assert type(m).__name__ == "ChatOllama"
    assert m.model == "qwen3:4b" and m.temperature == 0.7


def test_search_filters():
    api = FakeTravelAPI()
    assert [f["id"] for f in api.search_flights("nyc", "lon", "2026-03-03")] == ["F1", "F2", "F3"]
    assert [f["id"] for f in api.search_flights("SFO", "TYO")] == ["F6", "F7"]
    assert api.search_flights("NYC", "TYO") == "No flights found."


def test_book_and_refund_mutate_state():
    api = FakeTravelAPI()
    assert "B1" in api.book_flight("F4")
    assert api.bookings == {"B1": "F4"}
    assert "Refunded" in api.refund("B1")
    assert api.refunded == {"B1"}
    assert "Error" in api.book_flight("F99")
    assert "Error" in api.refund("B99")


def test_reset_seeds_bookings():
    api = FakeTravelAPI()
    api.reset(seed_bookings=("F1", "F6"))
    assert api.bookings == {"B1": "F1", "B2": "F6"}
    assert api.refunded == set()


def test_task_ground_truth_well_formed():
    flight_ids = {f["id"] for f in FLIGHTS}
    for t in TASKS:
        targets = t.get("book") or t["refund"]
        if "book" in t:
            assert targets <= flight_ids
        # clear task = exactly one acceptable outcome; ambiguous = several
        assert (len(targets) == 1) == (not t["ambiguous"]), t["id"]


def test_check_accepts_and_rejects():
    api = FakeTravelAPI()
    api.book_flight("F3")
    ok, done = check(TASKS[0], api)  # task 1 expects F3
    assert ok and done == {"F3"}

    api.reset()
    api.book_flight("F1")
    ok, _ = check(TASKS[0], api)
    assert not ok

    api.reset()
    api.book_flight("F3")
    api.book_flight("F1")  # two actions -> not acceptable even if one matches
    ok, _ = check(TASKS[0], api)
    assert not ok

    api.reset(seed_bookings=("F1", "F6"))
    api.refund("B2")
    ok, done = check(TASKS[9], api)  # task 10: either booking acceptable
    assert ok and done == {"B2"}
