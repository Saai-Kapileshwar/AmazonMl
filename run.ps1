param(
    [ValidateSet('prepare','train','predict','all','test','validate','package')]
    [string]$Command = 'all'
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Python environment missing. Create .venv and install code/business_entity_resolution/requirements.txt as described in README.md.'
}
if ($Command -eq 'test') {
    & $projectPython -X utf8 -m unittest discover -s code/business_entity_resolution/src -p test_pipeline.py -v
} elseif ($Command -eq 'validate') {
    & $projectPython -X utf8 code/business_entity_resolution/src/validate_outputs.py --data student_resource/dataset/test --output output
} elseif ($Command -eq 'package') {
    & $projectPython -X utf8 code/business_entity_resolution/src/package_submission.py --root . --team MLG4
} else {
    & $projectPython -u -X utf8 code/business_entity_resolution/src/pipeline.py $Command
}
if ($LASTEXITCODE -ne 0) { throw "Pipeline exited with code $LASTEXITCODE" }
if ($Command -eq 'all') {
    & $projectPython -X utf8 code/business_entity_resolution/src/validate_outputs.py --data student_resource/dataset/test --output output
    if ($LASTEXITCODE -ne 0) { throw 'Output validation failed' }
}
