"""Interactive Telegram login check.

Prompts for the phone number and the login code Telegram sends.
The auth session is stored at ~/.osi/telegram.session.
"""

import os
from pathlib import Path

# Makes start(), get_me(), and disconnect() block instead of returning coroutines.
from telethon.sync import TelegramClient


def load_dotenv(path: Path) -> None:
    """Read KEY=VALUE lines from path into the process environment."""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        # Keep a value already exported in the shell.
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def main() -> None:
    env_file = Path(__file__).resolve().parent / ".env"
    load_dotenv(env_file)

    api_id_raw = os.environ.get("TELEGRAM_API_ID", "")
    api_hash = os.environ.get("TELEGRAM_API_HASH", "")
    # The committed .env holds placeholders until real app credentials are filled in.
    if not api_id_raw.isdigit() or not api_hash or api_hash.startswith("<"):
        raise SystemExit(
            "Fill TELEGRAM_API_ID and TELEGRAM_API_HASH in .env "
            "(https://my.telegram.org/apps), then run this again."
        )

    # Telethon does not expand ~, and it only appends .session when the path lacks that suffix.
    session_path = Path.home() / ".osi" / "telegram.session"
    # sqlite opens the file directly and will not create missing parent directories.
    session_path.parent.mkdir(parents=True, exist_ok=True)

    client = TelegramClient(str(session_path), int(api_id_raw), api_hash)
    # Port 443 on Telegram DCs is answered with HTTP here, which breaks the handshake.
    # 5222 is an MTProto port, including after a phone-number DC migration.
    original_set_dc = client.session.set_dc

    def set_dc(dc_id, server_address, port):
        return original_set_dc(dc_id, server_address, 5222)

    client.session.set_dc = set_dc
    if client.session.server_address and client.session.port != 5222:
        original_set_dc(client.session.dc_id, client.session.server_address, 5222)
    # First run asks for the phone, the login code, and the 2FA password if one is set.
    client.start()
    me = client.get_me()
    # Accounts can be logged in without a public @username.
    username = f"@{me.username}" if me.username else "(no username)"
    print(f"Logged in as {username} (id={me.id})")
    client.disconnect()


if __name__ == "__main__":
    main()
