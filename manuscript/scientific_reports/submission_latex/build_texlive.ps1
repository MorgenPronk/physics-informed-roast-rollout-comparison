$ErrorActionPreference = "Stop"

$texliveBin = "C:\texlive\2024\bin\windows"
$pdflatex = Join-Path $texliveBin "pdflatex.exe"
$bibtex = Join-Path $texliveBin "bibtex.exe"

if (!(Test-Path $pdflatex)) {
    throw "TeX Live pdflatex not found at $pdflatex"
}
if (!(Test-Path $bibtex)) {
    throw "TeX Live bibtex not found at $bibtex"
}

& $pdflatex -interaction=nonstopmode main.tex
& $bibtex main
& $pdflatex -interaction=nonstopmode main.tex
& $pdflatex -interaction=nonstopmode main.tex

Write-Host "Built manuscript with TeX Live:" (Join-Path (Get-Location) "main.pdf")
