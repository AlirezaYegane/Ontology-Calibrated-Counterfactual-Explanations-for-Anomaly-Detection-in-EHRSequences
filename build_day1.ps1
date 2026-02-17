$ErrorActionPreference = "Stop"

$baseDir     = "D:\Article"
$excelPath   = Join-Path $baseDir "papers.xlsx"
$sheetName   = "Papers"
$summaryPath = Join-Path $baseDir "summary.md"

# Backup (so you don't cry later)
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
if (Test-Path $excelPath)   { Copy-Item $excelPath   ($excelPath   + ".bak_" + $stamp) -Force }
if (Test-Path $summaryPath) { Copy-Item $summaryPath ($summaryPath + ".bak_" + $stamp) -Force }

Import-Module ImportExcel -ErrorAction Stop

# Core 10 (as PSCustomObjects) - no JSON drama
$papers = @(
  [pscustomobject]@{ ID=1; Title='Med-BERT: Pre-trained Contextualized Embeddings on Large-scale Structured EHRs'; Authors='Rasmy et al.'; Venue='NPJ Digital Medicine'; Year=2021; Bucket='EHR Representation Learning'; 'Data/Modality'='Structured EHR tokens'; 'Method/Model'='Transformer/BERT; self-supervised pretraining; time encoding'; Evaluation='Large-scale structured EHR; downstream prediction tasks'; 'Key Contribution'='Backbone encoder; demonstrates viability of massive pretraining for robust clinical representations'; Strengths='Strong transferable embeddings; scalable pretraining; supports irregular timing with explicit encoding'; Limitations='Needs very large data; tokenization/time encoding choices affect stability'; 'Relevance to My Framework'='Candidate backbone encoder for our pipeline; aligns with representation learning foundation'; Tags='#representation-learning #transformer #pretraining #time-encoding'; Link=''; Status='To Read'; Notes='' }
  [pscustomobject]@{ ID=2; Title='CEHR-BERT: Incorporating Temporal Information from Structured EHR Data'; Authors='Pang et al.'; Venue='PMLR (ML4H)'; Year=2021; Bucket='EHR Representation Learning'; 'Data/Modality'='Structured EHR + time'; 'Method/Model'='Transformer with explicit temporal encoding for irregular intervals'; Evaluation='EHR predictive tasks'; 'Key Contribution'='Temporal encoding to model irregular gaps; critical for ICU temporal signals'; Strengths='Better temporal fidelity than token-only modeling'; Limitations='Temporal assumptions may not generalize across hospitals/granularities'; 'Relevance to My Framework'='Supports temporal anomaly detection component in ICU setting'; Tags='#representation-learning #transformer #temporal-encoding'; Link=''; Status='To Read'; Notes='' }
  [pscustomobject]@{ ID=3; Title='GRAM: Graph-based Attention Model for Healthcare Representation Learning'; Authors='Choi et al.'; Venue='KDD'; Year=2017; Bucket='Medical Ontologies / Knowledge Graph'; 'Data/Modality'='ICD/clinical code hierarchies'; 'Method/Model'='Ontology-aware attention over code ancestors'; Evaluation='EHR tasks on large datasets'; 'Key Contribution'='Ontology calibration to enforce semantic consistency in representations'; Strengths='Strong clinical semantics; interpretable hierarchy attention'; Limitations='Depends on ontology quality/coverage; hierarchy can be coarse'; 'Relevance to My Framework'='Core for ontology-calibrated embeddings and anomaly/counterfactual plausibility constraints'; Tags='#clinical-knowledge #ontology #graph #attention'; Link=''; Status='To Read'; Notes='' }
  [pscustomobject]@{ ID=4; Title='G-BERT: Pre-training of Graph Augmented Transformers for Medication Recommendation'; Authors='Shang et al.'; Venue='IJCAI'; Year=2019; Bucket='Medical Ontologies / Knowledge Graph'; 'Data/Modality'='Ontology graph + EHR sequences'; 'Method/Model'='Graph + Transformer fusion (GNN for ontology, BERT for sequence)'; Evaluation='Medication recommendation'; 'Key Contribution'='Graph-augmented transformer framework for clinically aware sequence modeling'; Strengths='Good semantic fusion; aligns with ontology-calibrated representations'; Limitations='Graph construction choices matter; sensitive to missing edges'; 'Relevance to My Framework'='Blueprint for fusing KG/ontology into the encoder stage before anomaly/counterfactual modules'; Tags='#clinical-knowledge #ontology #graph #transformer #pretraining'; Link=''; Status='To Read'; Notes='' }
  [pscustomobject]@{ ID=5; Title='DeepAnT: A Deep Learning Approach for Unsupervised Anomaly Detection in Time Series'; Authors='Munir et al.'; Venue='IEEE Access'; Year=2019; Bucket='Anomaly Detection in EHR/ICU'; 'Data/Modality'='Continuous time-series (vitals/waveforms)'; 'Method/Model'='Unsupervised CNN-based forecasting/anomaly scoring'; Evaluation='Time-series anomaly benchmarks'; 'Key Contribution'='Baseline for continuous-signal anomaly module'; Strengths='Practical baseline; handles continuous channels'; Limitations='Weak clinical grounding; can over-flag noise without constraints'; 'Relevance to My Framework'='Baseline comparator for vitals/waveform anomaly detection stream'; Tags='#anomaly-detection #time-series #cnn #unsupervised'; Link=''; Status='To Read'; Notes='' }
  [pscustomobject]@{ ID=6; Title='EHR-BERT: A BERT-based Model for Effective Anomaly Detection in EHRs'; Authors='Niu et al.'; Venue='Journal of Biomedical Informatics'; Year=2024; Bucket='Anomaly Detection in EHR/ICU'; 'Data/Modality'='Discrete EHR event logs'; 'Method/Model'='Transformer anomaly detection via masked token prediction objective'; Evaluation='EHR anomaly detection via event prediction'; 'Key Contribution'='Transformer-based anomaly objective tailored to event sequences'; Strengths='Strong sequence modeling; direct anomaly objective'; Limitations='Prediction-error anomaly definition needs calibration (rare-but-valid vs truly abnormal)'; 'Relevance to My Framework'='Anchor method for discrete-event anomaly stream before ontology calibration and counterfactuals'; Tags='#anomaly-detection #transformer #masked-token-prediction'; Link=''; Status='To Read'; Notes='' }
  [pscustomobject]@{ ID=7; Title='MedSeqCF: Style-transfer Counterfactual Explanations for ICU Mortality'; Authors='Wang et al.'; Venue='Artificial Intelligence in Medicine'; Year=2023; Bucket='Generative Counterfactuals'; 'Data/Modality'='ICU sequences'; 'Method/Model'='Counterfactual generation via style-transfer framing'; Evaluation='ICU mortality counterfactual explanations'; 'Key Contribution'='Clinically oriented counterfactual framework tailored to ICU sequences'; Strengths='Actionable explanation lens; aligns with clinically meaningful counterfactual requirement'; Limitations='Hard plausibility constraints; unrealistic edits possible without strong priors'; 'Relevance to My Framework'='Direct anchor for counterfactual module design and evaluation'; Tags='#counterfactuals #explainability #icu #sequence'; Link=''; Status='To Read'; Notes='' }
  [pscustomobject]@{ ID=8; Title='EHRDiff: Exploring Realistic EHR Synthesis with Diffusion Models'; Authors='Li et al.'; Venue='TMLR'; Year=2023; Bucket='Generative Modeling for Clinical Data'; 'Data/Modality'='EHR records'; 'Method/Model'='Diffusion models for EHR synthesis'; Evaluation='EHR synthesis quality comparisons'; 'Key Contribution'='Stable diffusion-based generator for synthetic EHR and counterfactual sampling'; Strengths='Training stability; strong generative fidelity'; Limitations='Compute heavy; privacy requires auditing; conditioning design matters'; 'Relevance to My Framework'='Primary candidate generative backbone for counterfactual generation'; Tags='#generative #diffusion #privacy #augmentation'; Link=''; Status='To Read'; Notes='' }
  [pscustomobject]@{ ID=9; Title='SHy: Self-Explaining Hypergraph Neural Networks for Disease Diagnosis Prediction'; Authors='Yu et al.'; Venue='CHIL'; Year=2025; Bucket='Hypergraph / Interpretable Modeling'; 'Data/Modality'='Higher-order clinical interactions'; 'Method/Model'='Hypergraph model with self-explaining components'; Evaluation='Diagnosis prediction with interpretable hyperedges'; 'Key Contribution'='Interpretable higher-order structure useful for explanation and constraints'; Strengths='Interpretability baked in; captures higher-order relations'; Limitations='Hypergraph construction complexity; sensitive to sparsity/definitions'; 'Relevance to My Framework'='Supports interpretable, ontology-aware anomaly and counterfactual explanations'; Tags='#representation-learning #hypergraph #interpretability'; Link=''; Status='To Read'; Notes='' }
  [pscustomobject]@{ ID=10; Title='CONAN: Complementary Pattern Augmentation for Rare Event Detection'; Authors='Xiao et al.'; Venue='AAAI'; Year=2020; Bucket='Anomaly Detection (Rare Events)'; 'Data/Modality'='Event sequences / rare patterns'; 'Method/Model'='Augmentation-driven rare event detection'; Evaluation='Rare event detection benchmarks'; 'Key Contribution'='Couples augmentation with rare-event detection to boost sensitivity'; Strengths='Addresses imbalance; relevant for safety-critical rare anomalies'; Limitations='Augmentation can inject artifacts; needs clinical constraints'; 'Relevance to My Framework'='Motivates coupling generative module with anomaly detection for rare anomaly sensitivity'; Tags='#anomaly-detection #rare-events #augmentation'; Link=''; Status='To Read'; Notes='' }
)

# Write Excel (clean, table, freeze, autosize)
$papers | Export-Excel -Path $excelPath -WorksheetName $sheetName -TableName 'PapersTable' -TableStyle Medium9 -FreezeTopRow -BoldTopRow -AutoSize -ClearSheet

# Post-format: wrap + vertical top for long columns
$pkg = Open-ExcelPackage -Path $excelPath
$ws  = $pkg.Workbook.Worksheets[$sheetName]

# Wrap text for columns G to Q (7..17)
Set-ExcelRange -Worksheet $ws -Range 'G:Q' -WrapText -VerticalAlignment Top
Set-ExcelRange -Worksheet $ws -Range 'A:Q' -VerticalAlignment Top

Close-ExcelPackage $pkg

# summary.md
$lines = New-Object System.Collections.Generic.List[string]
$lines.Add('# Day 1 - Literature Survey Summary')
$lines.Add('')
$lines.Add('Generated: ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
$lines.Add('')
$lines.Add('## Core Papers (10)')
$lines.Add('')

foreach ($p in $papers) {
  $lines.Add("### [$($p.ID)] $($p.Title) ($($p.Year))")
  $lines.Add("- **Authors:** $($p.Authors)")
  $lines.Add("- **Venue:** $($p.Venue)")
  $lines.Add("- **Bucket:** $($p.Bucket)")
  $lines.Add("- **Data/Modality:** $($p.'Data/Modality')")
  $lines.Add("- **Method/Model:** $($p.'Method/Model')")
  $lines.Add("- **Evaluation:** $($p.Evaluation)")
  $lines.Add("- **Key Contribution:** $($p.'Key Contribution')")
  $lines.Add("- **Strengths:** $($p.Strengths)")
  $lines.Add("- **Limitations:** $($p.Limitations)")
  $lines.Add("- **Relevance to Framework:** $($p.'Relevance to My Framework')")
  $lines.Add("- **Tags:** $($p.Tags)")
  $lines.Add('')
}

$lines.Add('## Gaps / Action Items (for Implementation)')
$lines.Add('- Decide final backbone encoder (Med-BERT vs CEHR-BERT vs graph-augmented transformer).')
$lines.Add('- Define ontology calibration mechanism (hierarchy attention vs KG embeddings vs hypergraph).')
$lines.Add('- Specify anomaly targets (event-log objective vs time-series forecasting, plus rare-event augmentation).')
$lines.Add('- Lock counterfactual evaluation criteria (plausibility, minimality, clinical actionability).')
$lines.Add('')

$lines | Set-Content -Path $summaryPath -Encoding UTF8

"OK. Updated: $excelPath | Created: $summaryPath"
