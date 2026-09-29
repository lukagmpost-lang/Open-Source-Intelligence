# Open-Source-Intelligence

Local multi-layer graph of public social relations. Accounts are nodes, public relations are weighted edges, and each relation type is a layer.

## Try it

`examples/` holds the manual test inputs.

- `examples/simple.csv` is a weighted edge list. Alice, Bob, and Carol form a triangle, and Dave is tied to Alice.
- `examples/simple.json` is NetworkX node-link JSON: a path from `a` to `c` through `b`.
- `examples/simple.graphml` has four nodes in two communities. Alice and Bob are east, Carol and Dave are west, and the edge from Bob to Carol is the bridge.

```bash
python3 main.py --source file --path examples/simple.csv --save-run simple-v1
python3 -m osi.ask --run simple-v1 "top 5 by pagerank"
```

The same two commands work with `examples/simple.json` and `examples/simple.graphml`.

## Web

```bash
pip install osi[web]
uvicorn osi.server:app --host 0.0.0.0 --port 7860
```

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

## LLM Provider

The tool uses Groq's free tier by default. No credit card required.

1. Get a free key at https://console.groq.com/keys
2. Add it to .env:
       LLM_API_KEY=your_key_here

The default model is llama-3.3-70b-versatile. If you hit the 1,000 requests/day limit, switch to llama-3.1-8b-instant in .env for 14,400 requests/day.
