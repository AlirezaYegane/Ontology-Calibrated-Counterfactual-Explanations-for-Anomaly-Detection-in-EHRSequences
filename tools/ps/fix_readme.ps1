#requires -Version 5.1
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$readme = "README.md"
if (-not (Test-Path $readme)) { throw "README.md not found in current folder." }

$txt = Get-Content $readme -Raw

# Fix the broken literal "rn" sequences that got appended earlier
$txt = $txt -replace "rn## Docsrn- Day 2 System Architecture:\s*docs/day2_system_architecture/", "`r`n## Docs`r`n- Day 2 System Architecture: docs/day2_system_architecture/"

# If Docs section is missing entirely, append it neatly
if ($txt -notmatch "## Docs") {
  $txt = $txt.TrimEnd() + "`r`n`r`n## Docs`r`n- Day 2 System Architecture: docs/day2_system_architecture/`r`n"
}

Set-Content -Encoding UTF8 $readme $txt
Write-Host "✅ README.md cleaned." -ForegroundColor Green
