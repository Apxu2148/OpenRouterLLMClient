from logger import preview_text, sanitize_text


def test_sanitize_masks_openrouter_key() -> None:
    secret = "sk-or-v1-abcdefghijklmnopqrstuvwxyz123456"
    sanitized = sanitize_text(f"Authorization: Bearer {secret}")

    assert secret not in sanitized
    assert "[REDACTED]" in sanitized


def test_sanitize_masks_known_environment_secret(monkeypatch) -> None:
    secret = "secret-value-from-env"
    monkeypatch.setenv("OPENROUTER_API_KEY", secret)

    sanitized = sanitize_text(f"The key is {secret}")

    assert secret not in sanitized
    assert "[REDACTED]" in sanitized


def test_preview_truncates_and_sanitizes() -> None:
    secret = "sk-or-v1-abcdefghijklmnopqrstuvwxyz123456"
    preview = preview_text(secret + " " + ("x" * 400), limit=60)

    assert secret not in preview
    assert len(preview) <= 60
    assert preview.endswith("...")
