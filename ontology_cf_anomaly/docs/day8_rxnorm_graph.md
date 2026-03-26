# Day 8  RxNorm Directed Knowledge Graph (Build + Audit)

## Inputs
- ontologies/processed/rxnorm_concepts.csv
- ontologies/processed/rxnorm_relationships.csv

## Output
- ontologies/processed/rxnorm_graph.gpickle

## Validation Results (audited)
- Runtime: ~8.3 seconds (end-to-end loadbuildsave on local hardware)
- Nodes: 406,601 (unique RxCUI)
- Edges: 1,661,222 (directed)
- Output size: 89,993,560 bytes (~85.8 MiB)
- Sanity: successors + predecessors verified for RxCUI 1049630

## How to rebuild
python scripts/build_rxnorm_graph.py
