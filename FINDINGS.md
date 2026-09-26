# Findings

## Finding 5: Robustness to node removal

### 2008 graph (5,110 nodes, 77 components, giant = 95.9%)

Three removal strategies tested at ratios 1% to 30%. The largest-component column is the giant component as a fraction of the original 5,110 nodes.

| Ratio | Random (largest CC) | Degree | Betweenness |
|---|---|---|---|
| 0%    | 0.959 | 0.959 | 0.959 |
| 1%    | 0.948 | 0.942 | 0.937 |
| 2%    | 0.938 | 0.926 | 0.917 |
| 5%    | 0.907 | 0.879 | 0.863 |
| 10%   | 0.854 | 0.807 | 0.768 |
| 20%   | 0.751 | 0.632 | 0.598 |
| 30%   | 0.649 | 0.300 | 0.280 |

Efficiency at 10% removal: random 0.329, degree 0.207, betweenness 0.196.

Key observation: random removal preserves the giant component even at 30% (0.649). Targeted removal of high-betweenness nodes halves it at ~25%: the giant is still 0.598 at 20% removal and 0.280 at 30%, and half of the intact giant (0.959) is 0.479.

### 2012 graph

Pending. The `sim2012` robustness run is still going, so these rows are not filled in yet.

## Status

- [x] Modularity comparison
- [x] Elite turnover
- [x] Community dissolution
- [x] Individual trajectories
- [~] Robustness (2008 done, 2012 pending)
- [ ] Cross-platform integration
