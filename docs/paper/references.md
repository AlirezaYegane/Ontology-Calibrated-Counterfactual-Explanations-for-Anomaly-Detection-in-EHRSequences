# References

This file collects the references used by the manuscript. Prior-work entries are drawn
from the repository's own literature survey (`docs/research/summary.md`,
`docs/one_pagers/`); the source URLs below are the ones recorded during that survey.
Author lists, exact years, venues, page numbers, and DOIs that are **not** unambiguously
determined by these URLs are deliberately left for verification in
[`references_todo.md`](references_todo.md) rather than guessed.

## Prior work (from the project literature survey)

1. **Med-BERT: pretrained contextualized embeddings on large-scale structured EHRs.**
   npj Digital Medicine. https://www.nature.com/articles/s41746-021-00455-y
2. **CEHR-BERT: incorporating temporal information from structured EHR data.**
   Proceedings of Machine Learning Research (ML4H). https://proceedings.mlr.press/v158/pang21a.html
3. **GRAM: graph-based attention model for healthcare representation learning.**
   ACM SIGKDD. https://dl.acm.org/doi/10.1145/3097983.3098126
4. **G-BERT: pre-training of graph-augmented transformers for medication recommendation.**
   IJCAI. https://www.ijcai.org/proceedings/2019/825
5. **DeepAnT: a deep learning approach for unsupervised anomaly detection in time series.**
   IEEE Access. https://www.dfki.de/fileadmin/user_upload/import/10175_DeepAnt.pdf
6. **EHR-BERT: a BERT-based model for effective anomaly detection in EHRs.**
   Journal of Biomedical Informatics. https://www.sciencedirect.com/science/article/pii/S1532046424000236
7. **MedSeqCF: style-transfer counterfactual explanations for ICU mortality (event
   sequences).** https://pubmed.ncbi.nlm.nih.gov/36628793/ —
   code: https://github.com/zhendong3wang/counterfactuals-for-event-sequences —
   dissertation write-up: https://www.diva-portal.org/smash/get/diva2:1906268/FULLTEXT03.pdf
8. **EHRDiff: exploring realistic EHR synthesis with diffusion models.**
   arXiv preprint. https://arxiv.org/abs/2303.05656
9. **SHy: self-explaining hypergraph neural networks for disease diagnosis prediction.**
   Proceedings of Machine Learning Research. https://proceedings.mlr.press/v287/yu25a.html
10. **CONAN: complementary pattern augmentation for rare-event detection.**
    AAAI. https://ojs.aaai.org/index.php/AAAI/article/view/5401

## Data and ontology resources

- **MIMIC-IV** — freely accessible ICU electronic health record dataset, distributed via
  PhysioNet under credentialed access. https://physionet.org/content/mimiciv/
- **eICU Collaborative Research Database / GOSSIS** — multi-center ICU dataset, PhysioNet
  credentialed access. https://physionet.org/content/gossis/
- **UMLS Metathesaurus** — U.S. National Library of Medicine. Used for ICD→SNOMED and
  drug→RxNorm crosswalks (release 2026AA). https://www.nlm.nih.gov/research/umls/
- **SNOMED CT** — SNOMED International, distributed via the UMLS/UTS (US Edition).
  https://www.nlm.nih.gov/healthit/snomedct/
- **RxNorm** — U.S. National Library of Medicine. https://www.nlm.nih.gov/research/umls/rxnorm/

## Note on citation discipline

Consistent with the project's honesty constraints, DOIs and full author/year strings are
**not** fabricated. The entries above are cited by title, venue (where the URL makes it
unambiguous), and the survey URL. Before external submission, complete the bibliographic
details tracked in [`references_todo.md`](references_todo.md).
