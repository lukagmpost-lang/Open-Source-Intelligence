# Findings

## Finding 5: Robustness to node removal (complete)

### Giant component size after removal, by strategy and ratio

| Ratio | 2008 random | 2008 degree | 2008 betweenness | 2012 random | 2012 degree | 2012 betweenness |
|-------|-------------|-------------|------------------|-------------|-------------|------------------|
| 1%    | 0.948       | 0.942       | 0.937            | 0.938       | 0.905       | 0.900            |
| 2%    | 0.938       | 0.926       | 0.917            | 0.928       | 0.877       | 0.870            |
| 5%    | 0.907       | 0.879       | 0.863            | 0.896       | 0.817       | 0.791            |
| 10%   | 0.854       | 0.807       | 0.768            | 0.843       | 0.703       | 0.647            |
| 20%   | 0.751       | 0.632       | 0.598            | 0.738       | 0.300       | 0.132            |
| 30%   | 0.649       | 0.300       | 0.280            | 0.633       | 0.022       | 0.001            |

Key observation: the 2012 graph is as robust as 2008 to random removal (0.633 vs 0.649 at 30%) but far more fragile to targeted removal (0.022 vs 0.300 for degree; 0.001 vs 0.280 for betweenness).

After 30% betweenness-targeted removal, the 2012 graph fragments into 15,989 components — 15× more than 2008's 1,029.

This is the signature of a scale-free network: random removal is tolerated, targeted hub removal causes catastrophic collapse. It matches the health metrics in Finding 6 (collapsed rich club, power-law degree distribution, high max degree).

## Steam layer: verification failure

Attempted to add Steam as a third layer using SteamGPT (steamgpt.net), a third-party scraper that does not require a Valve API key.

Verification: opened steamcommunity.com/id/gabelogannewell in a browser. Steam reports "This profile is private." SteamGPT returned 79 friends for the same profile.

Conclusion: the SteamGPT data is either stale (cached before the profile went private) or fabricated. It is not usable. The Steam layer was removed.

This is why official APIs and verified research archives are preferable to third-party scrapers: provenance cannot be verified.

## Finding 9: Structural fingerprinting fails for cross-platform identity resolution

Compared `r2008-v2` with `github_multi_ego.graphml` (top 100 Reddit users by PageRank, all 1,247 GitHub nodes). Each node got seven features: degree, PageRank, betweenness, closeness, community size, mean neighbor degree, and neighbor-degree Gini. Cosine similarity did the ranking. A shared name only adjusted confidence afterward. The structural cutoff stayed at 0.8.

Three implementations:

1. Within-graph z-scoring. Degenerate. `identity_v4.json` stored 495 STRUCTURAL_ONLY rows (similarity 0.8014–0.988) covering all 100 Reddit hubs. 494 of them were the seven GitHub ego centers: defunkt 100, rtomayko 98, antirez 98, hadley 82, mojombo 62, chromakode 39, spez 15. The remaining row was KeyserSosa.

2. Pooled z-scoring of the raw scores. Still degenerate. The GitHub centers' pooled z-scores on PageRank and betweenness ran from 10.26 to 44.97, while Reddit nodes on those axes sat near zero. Cosine among those seven centers was 0.966–1.000. No cross-platform pair cleared 0.8.

3. Rank percentiles inside each graph. The scale is shared, and the same person still sits in a different place on each platform. 116,574 of 124,700 pairs (93.5%) scored above 0.8. Out of 1,247 GitHub nodes, the five same-name pairs ranked antirez 266, hadley 898, chromakode 1115, rtomayko 1179, spez 1183. The best GitHub match for antirez on Reddit was marcel at 0.9839, not antirez at 0.9251.

Conclusion: structural fingerprinting cannot resolve identity across platforms with different edge semantics. The same person occupies different positions on GitHub (an asymmetric follow network) and Reddit (a co-participation network). Fingerprint similarity measures that difference, which is why it fails as an identity method. Cross-platform identity resolution needs content, timing, or manual verification. Structure alone is insufficient.

## Status

- [x] Modularity comparison
- [x] Elite turnover
- [x] Community dissolution
- [x] Individual trajectories
- [x] Robustness
- [ ] Cross-platform integration
