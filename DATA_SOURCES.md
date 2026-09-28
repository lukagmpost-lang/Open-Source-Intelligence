# Data sources

Exact constants are the ones in the code. Counts after a crawl are the numbers that crawl printed.

## Reddit

Stored user graphs: `r2008-v1` and `r2008-v2`, layer `reddit_user`, month `2008-01`. `r2012-v2`, layer `reddit_2012_user`, month `2012-08`. Both months use `layers/reddit_archive.py`.

Archive dataset: `Dk587/arctic`. Shard list:

`https://huggingface.co/api/datasets/Dk587/arctic/tree/main/data/comments/{year}/{mon}`

Shard file:

`https://huggingface.co/datasets/Dk587/arctic/resolve/main/{path}`

The DuckDB loader in `src/osi/loaders/reddit_duckdb.py` reads a different dataset path and does not rebuild those stored runs:

`hf://datasets/open-index/arctic/data/comments/{year}/{month:02d}/*.parquet`

It selects `author`, `subreddit`, and `link_id AS thread_id`, then `WHERE subreddit IN (SELECT UNNEST(?))`. The archive loader keeps the column name `link_id` and uses that value as the thread key. The rename to `thread_id` is only in the DuckDB loader.

Subreddit filter, both paths: `programming`, `science`, `entertainment`, `business`, `gaming`, `gadgets`, `sports`, `netsec`.

`THREAD_AUTHOR_CAP = 30`. `DROPPED_AUTHORS = {"[deleted]", "[removed]", ""}`.

User-layer rule: two accounts share an edge when both commented on the same `link_id`. Weight is the number of such threads. A thread with fewer than 2 authors, or more than 30, adds no pairs.

Subreddit layer (`reddit_subreddit`, `reddit_2012_subreddit`): two subreddits share an edge when the same account commented in both. Weight is the number of such accounts. This layer is built with the user layer and is not the saved run graph.

## GitHub follows

Saved as `github-multi-ego`, layer `github`, `max_pages = 2`. Copied later as `github-v1`, layer `github_multi_ego`.

Seed logins, from the saved config, sorted: `antirez`, `chromakode`, `defunkt`, `hadley`, `mojombo`, `rtomayko`, `spez`.

`GITHUB_PAGE_SIZE = 100`, so `per_page=100`. `max_pages = 2` stops each list at 200 accounts.

`https://api.github.com/users/{handle}`

`https://api.github.com/users/{username}/followers?per_page=100&page={page}`

`https://api.github.com/users/{username}/following?per_page=100&page={page}`

No token is stored on the run. `_headers()` sends `Authorization: Bearer` only when `GITHUB_TOKEN` is set. The saved config does not record one.

Edge rule: an undirected edge joins the login to each account in the follower list or the following list. `relation` is `follow` or `mutual`. `direction` is `follower`, `following`, or `mutual`. `weight` is `1.0`. A login with `followers == 0` is omitted. A neighbor of two seeds is one node with two edges.

## GitHub co-contribution

`github-organic-v1`, layer `github_organic`.

Seed owners: `torvalds`, `antirez`, `rtomayko`, `hadley`, `chromakode`, `defunkt`, `mojombo`.

Named repos, added even when the owner list omits them: `torvalds/linux`, `antirez/redis`, `rtomayko/rack`, `hadley/ggplot2`.

Owned-repo pages, at most `_MAX_OWNER_PAGES = 10`:

`https://api.github.com/users/{login}/repos?type=owner&per_page=100&page={page}&sort=full_name&direction=asc`

Contributors, one page:

`https://api.github.com/repos/{owner}/{repo}/contributors?per_page=100`

Forks are dropped. A page of 100 contributors is over the cap and is not paired. After bots are removed, a shorter page with more than `GROUP_CAP = 30` humans is also skipped. Fewer than 2 humans adds no edge.

Bots removed before the cap: login ending in `[bot]`, API `type` `Bot`, or one of `dependabot`, `dependabot-preview`, `renovate`, `renovate-bot`, `github-actions`, `pyup-bot`, `greenkeeper`, `greenkeeperio-bot`, `imgbot`, `allcontributors`, `codecov`, `codecov-io`, `whitesource-bolt`, `snyk-bot`.

An `OSError`, `RuntimeError`, or `ValueError` skips that repo, counts it failed, and is not cached. `get_json` turns HTTP 404 and HTTP 403 into `RuntimeError`. Empty bodies raise `ValueError` (`Expecting value`). The crawl's six failures: `defunkt/my-fun-repo` HTTP 403 (repository access blocked), `hadley/chequer` empty body, `hadley/test4` empty body, `mojombo/pyberry` empty body, `rtomayko/rack` HTTP 404, `torvalds/linux` HTTP 403 (contributor list too large).

Cache replay of the saved responses: 520 repos considered, 195 used, 21 over the cap, 298 under 2 contributors, 6 failed, 733 nodes, 5400 edges. Over the cap: `antirez/disque`, `antirez/ds4`, `antirez/llama.cpp-deepseek-v4-flash`, `antirez/redis`, `defunkt/coffee-mode`, `defunkt/gist`, `defunkt/github-gem`, `defunkt/jquery-pjax`, `defunkt/markdown-mode`, `hadley/adv-r`, `hadley/ggplot2`, `hadley/ggplot2-book`, `hadley/mastering-shiny`, `hadley/r-pkgs`, `hadley/r4ds`, `hadley/r4ds-1e`, `mojombo/chronic`, `mojombo/god`, `mojombo/grit`, `rtomayko/rack-cache`, `rtomayko/tilt`.

Edge rule: two logins share an edge when both are human contributors of the same kept repository. Weight is the number of shared repositories. Case-insensitive spellings are one node; the first spelling is kept.

## Bluesky follows

`bluesky-v1`, layer `bluesky`. Handles from `identity.json`: `jay.bsky.team`, `bsky.app`, `pfrazee.com`, `emilyliu.me`, `danabra.mov`.

Base URL `https://public.api.bsky.app`. HTTP 403 switches to `https://api.bsky.app`. No API key is sent.

`/xrpc/app.bsky.actor.getProfile?actor={handle}`

`/xrpc/app.bsky.graph.getFollows?actor={handle}&limit={page}`

`/xrpc/app.bsky.graph.getFollowers?actor={handle}&limit={page}`

`fetch_follows` and `fetch_followers` default to `limit=100`. Each page asks for `min(100, limit - len(collected))`. `fetch_posts` default is `limit=50` and those posts are stored on the seed node; they are not edges.

Edge rule: undirected edge from the seed to each followed account (`direction=following`) and each follower (`direction=follower`). The opposite direction on an existing edge becomes `mutual`. `weight` is `1.0`. Node ids are `bluesky:{handle}`.

## Bluesky replies

`bluesky-organic-v1`, layer `bluesky_organic`. Same five handles.

`/xrpc/app.bsky.feed.getAuthorFeed?actor={handle}&limit=100` (`_FEED_LIMIT = 100`).

`/xrpc/app.bsky.feed.getPostThread?uri={root}&depth=6` (`_THREAD_DEPTH = 6`).

The crawl printed: 5 seeds, 250 threads, 157 used, 71 skipped over the cap, 19 skipped under 2 authors, 3 failed, 1218 nodes, 12735 edges.

Authors are the handles on the thread view, including `parent` and nested `replies`. Dropped before the cap: blank handles, a `[bot]` suffix, and `handle.invalid`. `GROUP_CAP = 30`.

Edge rule: two handles share an edge when both appear in the same kept thread. Weight is the number of shared threads. This layer does not copy follow edges.

## SNAP Facebook

`snap-v1`, layer `snap_facebook`.

URL: `https://snap.stanford.edu/data/facebook_combined.txt.gz`

Gzip size: 218576 bytes. sha256 of those bytes: `125e84db872eeba443d270c70315c256b0af43a502fcfe51f50621166ad035d7`. SNAP does not publish a checksum in this repo. Decompressed size: 854362 bytes, sha256 `f41c026ed8af3cc3359f1ca5573d0605fb09ae0eefa34544b820fd8c6e2ef296`, 88234 lines. The loader rejects a download over `_MAX_BYTES = 5000000`.

Edge rule: each non-comment line with two integer ids is an undirected edge of `weight` `1.0`. Self-loops are skipped.

## Derived layers

Identity supra-graphs (`cross-v1` layers `github,reddit`; `cross-v2` the same with `normalize_layers`; `cross-v4` adds `bluesky` and normalizes). `normalize_layers` scales every layer so its edge weights sum to `1.0`. `merge_identity_layers` rewrites mapped handles onto one person id and adds an interlayer edge of weight `1.0` between that person's layer nodes. Nodes are then labeled `both`, `github-only`, `reddit-only`, or `bluesky-only`.

## Retrieval dates

First saved row for each layer, UTC:

| Layer | Run | created_at |
|---|---|---|
| Reddit 2008 user | `r2008-v1` | 2026-09-26T18:38:25+00:00 |
| Reddit 2012 user | `r2012-v2` | 2026-09-26T19:34:44+00:00 |
| Reddit 2008 user, later save | `r2008-v2` | 2026-09-26T21:45:09+00:00 |
| GitHub follows | `github-multi-ego` | 2026-09-27T01:10:13+00:00 |
| Bluesky follows | `bluesky-v1` | 2026-09-27T02:05:52+00:00 |
| GitHub follows, later save | `github-v1` | 2026-09-27T02:13:30+00:00 |
| SNAP Facebook | `snap-v1` | 2026-09-27T02:13:57+00:00 |
| GitHub co-contribution | `github-organic-v1` | 2026-09-27T02:44:38+00:00 |
| Bluesky replies | `bluesky-organic-v1` | 2026-09-27T02:55:05+00:00 |

## Known limitations

GitHub follow lists stop at 2 pages of 100, which is 200 accounts per direction. Bluesky follows and followers stop at 100. A GitHub login with zero followers is omitted. A profile the API will not return is skipped (`GitHub {relation} for {username} was not public`, or an empty Bluesky profile). Unauthenticated GitHub allows 60 requests per hour. The multi-ego and organic clients wait 2.0 seconds between live calls. Bluesky follows wait 1.0 second. HTTP 429 raises `rate limited by the upstream API; retry later`. Empty GitHub JSON is a failed repo, not a cached empty list. `rtomayko/rack` is HTTP 404 and `torvalds/linux` is HTTP 403, so those named repos contributed no edges. Threads, repos, and reply threads over 30 accounts are dropped, so the largest rooms are absent. The Arctic archive loader only has comment shards from 2005-12 through 2012.
