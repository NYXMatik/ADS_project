# Run from the extracted ann_hdf5_to_csv directory.
$ErrorActionPreference = 'Stop'
if (-not (Test-Path '.venv\Scripts\python.exe')) {
  py -m venv .venv
  if ($LASTEXITCODE -ne 0) { throw 'Python venv creation failed' }
}
$python = '.\.venv\Scripts\python.exe'
& $python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
$datasets = @(
  'nytimes-256-angular',
  'glove-200-angular',
  'mnist-784-euclidean',
  'coco-nomic-768-normalized'
)
foreach ($ds in $datasets) {
  Write-Host "=== Processing $ds ==="
  & $python convert.py --dataset $ds --download --estimate --convert
  if ($LASTEXITCODE -ne 0) { throw "Failed on $ds" }
}
Write-Host 'Done. Inspect data/<dataset>/manifest.json for the measured CSV sizes.'
