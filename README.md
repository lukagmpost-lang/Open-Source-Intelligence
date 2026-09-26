# Open-Source-Intelligence

Local multi-layer graph of public social relations. Accounts are nodes, public relations are weighted edges, and each relation type is a layer. The viewer groups accounts into communities and lists bridges: accounts whose neighbors fall in more than one community.

## What it will collect

| Source | Relation | Access |
| --- | --- | --- |
| GitHub | followers, following, mutual | Public REST API, at most 2 pages of 100 |
| Reddit | public comments to subreddits | Official OAuth API, at most 100 comments |
| Steam | friends and owned games | Web API, profile must be public, caps 300 friends and 200 games |
| Spotify | the token owner's public playlists | User token only, caps 20 playlists and 100 tracks |
| Telegram | gifts displayed on a profile | Bot API `getUserGifts`, at most 2 pages of 50. Hidden gifts and hidden senders are dropped |
| File | any of the above, plus `same_as` links you assert | JSON you already have |

Private Telegram channels, gifts that are not displayed, and anonymous page scraping are not implemented. There is no score for exploiting a person or a community.

## Run

```bash
PYTHONPATH=src python3 -m osi.app
```

Open http://127.0.0.1:8765. Load the sample, paste a graph file, or fetch one public account. Tokens are read from the environment, not from the page:

- `GITHUB_TOKEN` optional, raises the GitHub rate limit
- `REDDIT_TOKEN` bearer token from a Reddit app you own
- `STEAM_API_KEY`
- `SPOTIFY_TOKEN` for the account whose playlists you are reading
- `TELEGRAM_BOT_TOKEN`

## Graph file

```json
{
  "nodes": [{"platform": "github", "account_id": "ada", "label": "ada"}],
  "edges": [{"source": "github:ada", "target": "github:grace", "layer": "mutual", "weight": 1}],
  "same_as": [{"platform": "github", "account_id": "ada", "same_platform": "reddit", "same_account_id": "ada"}]
}
```

`same_as` is stored only when you assert it. The tool does not guess that two accounts are the same person.

## Tests

```bash
PYTHONPATH=src python3 -m pytest
```
