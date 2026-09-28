# Findings

## Finding 1: Modularity rose 72% (Reddit 2008 → 2012)

Louvain modularity of the stored partition, weight `weight`:

| Run | Layer | Nodes | Edges | Communities | Modularity |
|---|---|---:|---:|---:|---:|
| r2008-v2 | reddit_user | 5110 | 86268 | 96 | 0.3246767820 |
| r2012-v2 | reddit_2012_user | 54817 | 556415 | 1141 | 0.5582584329 |

(0.5582584329 - 0.3246767820) / 0.3246767820 = 0.7194282555751091.

## Finding 2: Elite turnover 99%

`compare.py --a r2008-v2 --b r2012-v2 --mode cohort --metric pagerank --top 100`:

```
COHORT_A 100
present 36  gone 64  still_top 1
median_delta +7.1227%
COHORT_B 100
from_a 1  new 99  were_top 1
```

99 of the 100 highest-PageRank accounts in r2012-v2 are absent from r2008-v2. Of the r2008-v2 top 100, 1 is still in the r2012-v2 top 100. That account is grauenwolf, rank 31 to rank 62:

```
grauenwolf            31      0.6067%  0.001665    62      0.1131%   0.000370    -0.4936%
```

## Finding 3: Community dissolution

Louvain containment of r2008-v2 inside r2012-v2. Null containment is 0.000876 (1/1141). A destination counts when lift is above 10. Two or more such destinations are SPLIT. One destination with lift below 10 is DISSOLVED. One destination that also absorbs another earlier community is MERGED. One destination with lift above 50 is STABLE.

96 communities: DISSOLVED 78, SPLIT 14, MERGED 3, STABLE 1.

| a_community | size | b_community | containment | lift | destinations | classification |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 1210 | 3 | 0.095041 | 108.44 | 2 | SPLIT |
| 1 | 761 | 4 | 0.018397 | 20.99 | 4 | SPLIT |
| 2 | 665 | 1 | 0.024060 | 27.45 | 4 | SPLIT |
| 3 | 495 | 4 | 0.030303 | 34.58 | 4 | SPLIT |
| 4 | 318 | 3 | 0.047170 | 53.82 | 4 | SPLIT |
| 5 | 307 | 4 | 0.045603 | 52.03 | 3 | SPLIT |
| 6 | 302 | 4 | 0.029801 | 34.00 | 3 | SPLIT |
| 7 | 224 | 4 | 0.022321 | 25.47 | 4 | SPLIT |
| 8 | 146 | 3 | 0.041096 | 46.89 | 3 | SPLIT |
| 9 | 133 | 3 | 0.075188 | 85.79 | 2 | SPLIT |
| 10 | 106 | 3 | 0.056604 | 64.58 | 2 | SPLIT |
| 11 | 74 | 1 | 0.040541 | 46.26 | 2 | SPLIT |
| 12 | 70 | 3 | 0.042857 | 48.90 | 3 | SPLIT |
| 13 | 40 | 0 | 0.050000 | 57.05 | 2 | SPLIT |
| 23 | 3 | 5 | 0.333333 | 380.33 | 1 | MERGED |
| 46 | 2 | 4 | 0.500000 | 570.50 | 1 | MERGED |
| 54 | 2 | 35 | 0.500000 | 570.50 | 1 | STABLE |
| 92 | 2 | 5 | 0.500000 | 570.50 | 1 | MERGED |

The other 78 communities have containment 0.000000, lift 0.00, destinations 0, classification DISSOLVED. Their ids and sizes: 14/35, 15/24, 16/12, 17/10, 18/8, 19/7, 20/4, 21/3, 22/3, 24/3, 25/3, 26/3, 27/3, 28/3, 29/3, 30/2, 31/2, 32/2, 33/2, 34/2, 35/2, 36/2, 37/2, 38/2, 39/2, 40/2, 41/2, 42/2, 43/2, 44/2, 45/2, 47/2, 48/2, 49/2, 50/2, 51/2, 52/2, 53/2, 55/2, 56/2, 57/2, 58/2, 59/2, 60/2, 61/2, 62/2, 63/2, 64/2, 65/2, 66/2, 67/2, 68/2, 69/2, 70/2, 71/2, 72/2, 73/2, 74/2, 75/2, 76/2, 77/2, 78/2, 79/2, 80/2, 81/2, 82/2, 83/2, 84/2, 85/2, 86/2, 87/2, 88/2, 89/2, 90/2, 91/2, 93/2, 94/2, 95/2.

## Finding 4: Individual trajectories (akdas, grauenwolf, incredible-ninja)

Stored scores from `report.py --runs r2008-v2,r2012-v2`.

akdas is in r2008-v2 and absent from r2012-v2.

```
PageRank:     0.004217  (rank 1 / 5110)
Degree:       0.148561  (rank 1)
Betweenness:  0.028257  (rank 1)
Closeness:    0.485747  (rank 1)
Louvain:      0
Leiden:       1
```

grauenwolf:

```
r2012-v2  PageRank 0.000370 (rank 62 / 54817)  Degree 0.008903 (rank 54)  Betweenness 0.003769 (rank 81)  Closeness 0.289038 (rank 5443)  Louvain 3  Leiden 3
r2008-v2  PageRank 0.001665 (rank 31 / 5110)  Degree 0.063026 (rank 33)  Betweenness 0.003498 (rank 106)  Closeness 0.433624 (rank 57)  Louvain 0  Leiden 1
PageRank:  r2012-v2 -> r2008-v2  0.000370 -> 0.001665 (delta +0.001295)
Degree:  r2012-v2 -> r2008-v2  0.008903 -> 0.063026 (delta +0.054124)
Betweenness:  r2012-v2 -> r2008-v2  0.003769 -> 0.003498 (delta -0.000270)
Closeness:  r2012-v2 -> r2008-v2  0.289038 -> 0.433624 (delta +0.144586)
```

incredible-ninja is in r2012-v2 and absent from r2008-v2.

```
PageRank:     0.001657  (rank 1 / 54817)
Degree:       0.033913  (rank 2)
Betweenness:  0.034073  (rank 1)
Closeness:    0.396632  (rank 1)
Louvain:      0
Leiden:       0
```

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

## Finding 6: Structural phase transition (small-world → scale-free)

Stored `health` payloads.

r2008-v2:

```
assortativity -0.016717553605958634
avg_clustering 0.6401807425046512
transitivity 0.20005060075430686
num_components 77
avg_degree 33.76438356164383
max_degree 759
best_fit power_law
power_law alpha 1.3470966437968435 xmin 1.0 r_squared 0.863577291676034
lognormal sigma 1.3805841974765207 mu 2.671948284602693 r_squared 0.6774477273108284
exponential rate 0.008221262857075693 r_squared 0.5246267564366679
rich_club 10 0.014804127170539385
rich_club 20 0.026138681710970866
rich_club 50 0.08923894763697328
rich_club 100 0.21477572559366753
```

r2012-v2:

```
assortativity 0.0006093685391545562
avg_clustering 0.7401053405065873
transitivity 0.1692050724069002
num_components 1082
avg_degree 20.30081908896875
max_degree 1890
best_fit power_law
power_law alpha 1.8664661261573763 xmin 1.0 r_squared 0.8917107623581217
lognormal sigma 1.2239076132657458 mu 2.2957505649310845 r_squared 0.6449072674711174
exponential rate 0.005419504019258511 r_squared 0.39298142018626137
rich_club 10 0.0011088400418329738
rich_club 20 0.002226628821753049
rich_club 50 0.01125501295596993
rich_club 100 0.038446553565701834
```

## Finding 7: Reach is flat

Closeness is the stored reach score. Every node in both runs has a stored value.

| Run | n | min | median | mean | max |
|---|---:|---:|---:|---:|---:|
| r2008-v2 | 5110 | 0.0001957330 | 0.3361043755 | 0.3222961113 | 0.4857467911 |
| r2012-v2 | 54817 | 0.0000182428 | 0.2563272273 | 0.2428832907 | 0.3966320935 |

The maximum is akdas at 0.485747 in r2008-v2 and incredible-ninja at 0.396632 in r2012-v2.

## Steam layer: verification failure

Attempted to add Steam as a third layer using SteamGPT (steamgpt.net), a third-party scraper that does not require a Valve API key.

Verification: opened steamcommunity.com/id/gabelogannewell in a browser. Steam reports "This profile is private." SteamGPT returned 79 friends for the same profile.

Conclusion: the SteamGPT data is either stale (cached before the profile went private) or fabricated. It is not usable. The Steam layer was removed.

This is why official APIs and verified research archives are preferable to third-party scrapers: provenance cannot be verified.

## Finding 8: Platform role divergence (5-person fingerprint study)

`person_compare.py` on `github-multi-ego` layer `github` (1247 nodes) and `r2008-v2` (5110 nodes). Class uses PageRank rank. Above the 90th percentile means rank/nodes <= 0.10.

```
antirez
  degree        203 (rank 5 / 1247)  110 (rank 336 / 5110)
  pagerank      0.060576 (rank 5)    0.000479 (rank 377)
  betweenness   0.244800 (rank 4)    0.000721 (rank 557)
  closeness     0.397068 (rank 4)    0.401721 (rank 323)
  community     2 (size 198)         0 (size 1210)
  fingerprint   0.9251
  class         UNIVERSAL

spez
  degree        66 (rank 7 / 1247)    86 (rank 462 / 5110)
  pagerank      0.024363 (rank 7)     0.000387 (rank 516)
  betweenness   0.100130 (rank 7)     0.000449 (rank 735)
  closeness     0.228792 (rank 1183)  0.386129 (rank 566)
  community     6 (size 67)           5 (size 307)
  fingerprint   0.7326
  class         GITHUB-DOMINANT

rtomayko
  degree        266 (rank 2 / 1247)  40 (rank 1106 / 5110)
  pagerank      0.069264 (rank 4)    0.000174 (rank 1615)
  betweenness   0.244769 (rank 5)    0.000044 (rank 1817)
  closeness     0.413953 (rank 2)    0.377651 (rank 720)
  community     4 (size 177)         0 (size 1210)
  fingerprint   0.8109
  class         GITHUB-DOMINANT

hadley
  degree        201 (rank 6 / 1247)  30 (rank 1444 / 5110)
  pagerank      0.070747 (rank 3)    0.000171 (rank 1637)
  betweenness   0.282443 (rank 3)    0.000410 (rank 792)
  closeness     0.360116 (rank 12)   0.362996 (rank 1143)
  community     1 (size 202)         0 (size 1210)
  fingerprint   0.8497
  class         GITHUB-DOMINANT

chromakode
  degree        258 (rank 3 / 1247)  11 (rank 3202 / 5110)
  pagerank      0.091735 (rank 2)    0.000074 (rank 3790)
  betweenness   0.436470 (rank 1)    0.000000 (rank 3342)
  closeness     0.387438 (rank 7)    0.328395 (rank 2942)
  community     0 (size 247)         0 (size 1210)
  fingerprint   0.6873
  class         GITHUB-DOMINANT
```

## Finding 9: Structural fingerprinting fails for cross-platform identity resolution

Compared `r2008-v2` with `github_multi_ego.graphml` (top 100 Reddit users by PageRank, all 1,247 GitHub nodes). Each node got seven features: degree, PageRank, betweenness, closeness, community size, mean neighbor degree, and neighbor-degree Gini. Cosine similarity did the ranking. A shared name only adjusted confidence afterward. The structural cutoff stayed at 0.8.

Three implementations:

1. Within-graph z-scoring. Degenerate. `identity_v4.json` stored 495 STRUCTURAL_ONLY rows (similarity 0.8014–0.988) covering all 100 Reddit hubs. 494 of them were the seven GitHub ego centers: defunkt 100, rtomayko 98, antirez 98, hadley 82, mojombo 62, chromakode 39, spez 15. The remaining row was KeyserSosa.

2. Pooled z-scoring of the raw scores. Still degenerate. The GitHub centers' pooled z-scores on PageRank and betweenness ran from 10.26 to 44.97, while Reddit nodes on those axes sat near zero. Cosine among those seven centers was 0.966–1.000. No cross-platform pair cleared 0.8.

3. Rank percentiles inside each graph. The scale is shared, and the same person still sits in a different place on each platform. 116,574 of 124,700 pairs (93.5%) scored above 0.8. Out of 1,247 GitHub nodes, the five same-name pairs ranked antirez 266, hadley 898, chromakode 1115, rtomayko 1179, spez 1183. The best GitHub match for antirez on Reddit was marcel at 0.9839, not antirez at 0.9251.

Conclusion: structural fingerprinting cannot resolve identity across platforms with different edge semantics. The same person occupies different positions on GitHub (an asymmetric follow network) and Reddit (a co-participation network). Fingerprint similarity measures that difference, which is why it fails as an identity method. Cross-platform identity resolution needs content, timing, or manual verification. Structure alone is insufficient.

## Finding 10: Edge semantics determines network structure (6-platform table)

Louvain modularity is the stored partition. Assortativity, clustering, and power-law R² are the stored `health` payload. The halving point is the smallest tested degree-removal ratio whose largest component is at most half the intact graph. Those ratios are the ones recorded for the comparison page: GitHub follows 0.01, Bluesky follows 0.01, SNAP Facebook 0.30, Reddit 2008 0.30, GitHub co-contribution 0.02, Bluesky replies 0.10.

| platform | nodes | edges | modularity | assortativity | clustering | power-law R² | halving point |
|---|---:|---:|---:|---:|---:|---:|---:|
| GitHub follows | 1247 | 1542 | 0.6489554893 | -0.8647454014896636 | 0.13118320762376973 | 0.7041682971997252 | 0.01 |
| Bluesky follows | 833 | 915 | 0.6987064409 | -0.95334050812306 | 0.02101115093398593 | 0.5409539362800933 | 0.01 |
| SNAP Facebook | 4039 | 88234 | 0.8349209912 | 0.06357722918564943 | 0.6055467186200862 | 0.8091782885710821 | 0.30 |
| Reddit 2008 | 5110 | 86268 | 0.3246767820 | -0.016717553605958634 | 0.6401807425046512 | 0.863577291676034 | 0.30 |
| GitHub co-contribution | 733 | 5400 | 0.7839194696 | -0.10246224195867357 | 0.8794333679156484 | 0.592321643542107 | 0.02 |
| Bluesky replies | 1218 | 12735 | 0.7379022935 | -0.059616652774097334 | 0.9213323257263784 | 0.5263247371505854 | 0.10 |

## Status

- [x] Modularity comparison
- [x] Elite turnover
- [x] Community dissolution
- [x] Individual trajectories
- [x] Robustness
- [ ] Cross-platform integration
