#!/bin/bash
# Clean build script to force fresh bibliography compilation

# stop on error
set -e

echo "🧹 Cleaning ALL build files..."
rm -f main_CISE.aux main_CISE.bbl main_CISE.blg main_CISE.log main_CISE.out main_CISE.toc main_CISE.pdf main_CISE.fls main_CISE.fdb_latexmk

echo "📝 First LaTeX pass..."
pdflatex -interaction=nonstopmode main_CISE.tex

echo "📚 Running BibTeX..."
bibtex main_CISE

# Check if BibTeX found entries
if [ -f "main_CISE.bbl" ]; then
    echo "✅ BibTeX created .bbl file"
    echo "📖 Bibliography entries found:"
    grep "\\bibitem" main_CISE.bbl || echo "⚠️  No bibitem entries found"
else
    echo "❌ BibTeX failed to create .bbl file"
    echo "📋 BibTeX log:"
    cat main_CISE.blg
    exit 1
fi

echo "📝 Second LaTeX pass..."
pdflatex -interaction=nonstopmode main_CISE.tex

echo "📝 Third LaTeX pass..."
pdflatex -interaction=nonstopmode main_CISE.tex

echo "🔍 Checking for citation issues..."
if grep -q "Citation.*undefined" main_CISE.log; then
    echo "⚠️  Undefined citations found:"
    grep "Citation.*undefined" main_CISE.log
fi

if grep -q "?" main_CISE.aux; then
    echo "⚠️  Question marks found in aux file - citations may be broken"
fi

if [ -f "main_CISE.pdf" ]; then
    echo "✅ Build complete: main_CISE.pdf"
else
    echo "❌ Build failed"
    exit 1
fi
