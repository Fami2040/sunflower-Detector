# Security

## Secrets

- **Telegram bot token** must be set only via environment variables (e.g. Railway **Variables**, Render, or a local `.env` file). The application reads it with `os.getenv("BOT_TOKEN")`.
- **Never** commit real tokens, API keys, or Railway tokens to this repository.
- If a token was ever exposed (git history, chat, screenshot, or public file), **revoke it in [@BotFather](https://t.me/BotFather)** immediately and set a new value in your host’s secret store only.

## Repository hygiene

- Do not add deploy helper scripts or markdown that embed secret values. Keep runbooks local or redact all placeholders.
- Enable [GitHub secret scanning](https://docs.github.com/code-security/secret-scanning) on the repository if available.
