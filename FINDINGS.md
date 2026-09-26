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

## Status

- [x] Modularity comparison
- [x] Elite turnover
- [x] Community dissolution
- [x] Individual trajectories
- [x] Robustness
- [ ] Cross-platform integration
