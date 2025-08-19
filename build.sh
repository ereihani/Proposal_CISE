#!/usr/bin/env bash
# build.sh — compile LaTeX with optional cleanup + BibTeX
# Usage: ./build.sh [main_CISE.tex]

set -Eeuo pipefail

TEX_FILE="${1:-main_CISE.tex}"
STEM="${TEX_FILE%.tex}"

# Optional cleanup with latexmk -C if available
if command -v latexmk >/dev/null 2>&1; then
  echo "Cleaning aux files with: latexmk -C"
  latexmk -C || true
else
  echo "latexmk not found; skipping optional cleanup."
fi

# Check required tools
for tool in pdflatex bibtex; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Error: '$tool' not found in PATH." >&2
    exit 1
  fi
done

# Check source file
if [[ ! -f "$TEX_FILE" ]]; then
  echo "Error: '$TEX_FILE' not found." >&2
  exit 1
fi

echo "1) pdflatex (first pass)…"
pdflatex -interaction=nonstopmode "$TEX_FILE"

echo "2) bibtex…"
bibtex "$STEM"

echo "3) pdflatex (second pass)…"
pdflatex -interaction=nonstopmode "$TEX_FILE"

echo "4) pdflatex (third pass)…"
pdflatex -interaction=nonstopmode "$TEX_FILE"

echo "✅ Done: ${STEM}.pdf"
