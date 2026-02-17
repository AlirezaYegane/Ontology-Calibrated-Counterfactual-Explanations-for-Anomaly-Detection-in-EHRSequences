#requires -Version 5.1
[CmdletBinding()]
param(
  [Parameter(Mandatory=$false)]
  [string]$Root = "D:\Article",

  [Parameter(Mandatory=$false)]
  [string]$ProjectDirName = "ontology_cf_anomaly",

  # For RTX 50xx you already used cu128; keep as default.
  [ValidateSet("cu128","cu121","cpu")]
  [string]$TorchVariant = "cu128"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-Command([string]$Name) {
  if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
    throw "Missing dependency: '$Name' is not installed or not in PATH."
  }
}

function Ensure-Dir([string]$Path) {
  if (-not (Test-Path $Path)) {
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
  }
}

function Write-Step([string]$Msg) {
  Write-Host ("`n==> " + $Msg) -ForegroundColor Cyan
}

Assert-Command git
Assert-Command python

if (-not (Test-Path $Root)) { throw "Root folder not found: $Root" }
Set-Location $Root

$ProjectDir = Join-Path $Root $ProjectDirName

Write-Step "Ensure project folders"
$dirs = @(
  (Join-Path $ProjectDir "data\mimiciii"),
  (Join-Path $ProjectDir "data\mimiciv"),
  (Join-Path $ProjectDir "data\eicu"),
  (Join-Path $ProjectDir "data\synthetic"),
  (Join-Path $ProjectDir "data\processed"),
  (Join-Path $ProjectDir "ontologies\umls_maps"),
  (Join-Path $ProjectDir "ontologies\snomed"),
  (Join-Path $ProjectDir "ontologies\rxnorm"),
  (Join-Path $ProjectDir "ontologies\embeddings"),
  (Join-Path $ProjectDir "notebooks"),
  (Join-Path $ProjectDir "src\preprocessing"),
  (Join-Path $ProjectDir "src\models"),
  (Join-Path $ProjectDir "src\training"),
  (Join-Path $ProjectDir "src\evaluation"),
  (Join-Path $ProjectDir "src\utils"),
  (Join-Path $ProjectDir "config"),
  (Join-Path $ProjectDir "tests"),
  (Join-Path $ProjectDir "scripts")
)
$dirs | ForEach-Object { Ensure-Dir $_ }

Write-Step "Ensure .gitkeep for empty tracked dirs"
$gitkeeps = @(
  (Join-Path $ProjectDir "data\mimiciii\.gitkeep"),
  (Join-Path $ProjectDir "data\mimiciv\.gitkeep"),
  (Join-Path $ProjectDir "data\eicu\.gitkeep"),
  (Join-Path $ProjectDir "data\synthetic\.gitkeep"),
  (Join-Path $ProjectDir "data\processed\.gitkeep"),
  (Join-Path $ProjectDir "ontologies\umls_maps\.gitkeep"),
  (Join-Path $ProjectDir "ontologies\snomed\.gitkeep"),
  (Join-Path $ProjectDir "ontologies\rxnorm\.gitkeep"),
  (Join-Path $ProjectDir "ontologies\embeddings\.gitkeep")
)
foreach ($f in $gitkeeps) { if (-not (Test-Path $f)) { New-Item -ItemType File -Force -Path $f | Out-Null } }

Write-Step "Ensure requirements.txt"
$reqPath = Join-Path $Root "requirements.txt"
if (-not (Test-Path $reqPath)) {
@"
pandas
networkx
numpy
tqdm
pyyaml
scikit-learn
matplotlib
"@ | Set-Content -Encoding UTF8 $reqPath
}

Write-Step "Ensure Torch check script"
$torchCheckPath = Join-Path $ProjectDir "scripts\check_torch_gpu.py"
if (-not (Test-Path $torchCheckPath)) {
@'
import torch

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("gpu_name:", torch.cuda.get_device_name(0))
    x = torch.randn((2048, 2048), device="cuda")
    y = torch.randn((2048, 2048), device="cuda")
    z = (x @ y).mean()
    torch.cuda.synchronize()
    print("GPU matmul ok, mean:", float(z))
else:
    print("gpu_name: CPU")
    x = torch.randn((512, 512))
    y = torch.randn((512, 512))
    z = (x @ y).mean()
    print("CPU matmul ok, mean:", float(z))
'@ | Set-Content -Encoding UTF8 $torchCheckPath
}

Write-Step "Ensure venv"
$venvPy = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
  python -m venv .venv
}

Write-Step "Upgrade pip tooling"
& $venvPy -m pip install -U pip setuptools wheel

Write-Step "Install PyTorch ($TorchVariant)"
if ($TorchVariant -eq "cu128") {
  & $venvPy -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
}
elseif ($TorchVariant -eq "cu121") {
  & $venvPy -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
}
else {
  & $venvPy -m pip install torch torchvision torchaudio
}

Write-Step "Install core requirements"
& $venvPy -m pip install -r $reqPath

Write-Step "Run Torch/CUDA checkpoint"
& $venvPy $torchCheckPath

Write-Host "`n✅ Day3 setup finished OK." -ForegroundColor Green
