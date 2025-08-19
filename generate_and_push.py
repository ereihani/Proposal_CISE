#!/usr/bin/env python3
import subprocess
import sys
import os
import argparse
from datetime import datetime

def run_command(cmd, description, ignore_failure=False):
    \"\"\"Run a command and handle errors.\"\"\"
    print(f\"Running: {description}\")
    try:
        result = subprocess.run(cmd, shell=True, check=True,
                                capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            # Some TeX tools write useful info to STDERR even on success
            print(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f\"Error {description}: {e}\")
        if e.stdout:
            print(f\"STDOUT: {e.stdout}\")
        if e.stderr:
            print(f\"STDERR: {e.stderr}\")
        if ignore_failure:
            print(\"Continuing despite the error (ignore_failure=True).\" )
            return False
        sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description=\"Compile LaTeX, run bibtex, and push to git.\")
    parser.add_argument(\"--clean\", action=\"store_true\",
                        help=\"Optionally run 'latexmk -C' to clean aux files before building.\")
    parser.add_argument(\"--file\", default=\"main_CISE.tex\",
                        help=\"LaTeX entrypoint .tex file (default: main_CISE.tex)\")
    return parser.parse_args()

def main():
    args = parse_args()

    tex_file = args.file
    tex_stem = os.path.splitext(tex_file)[0]
    pdf_file = f\"{tex_stem}.pdf\"

    # Check if .tex exists
    if not os.path.exists(tex_file):
        print(f\"Error: {tex_file} not found in current directory\")
        sys.exit(1)

    print(f\"Generating PDF from {tex_file}...\")

    # Optional cleanup (latexmk -C)
    if args.clean:
        # Run latexmk -C but don't hard-fail if tool is missing/not installed
        run_command(\"latexmk -C\", \"latexmk cleanup (optional)\", ignore_failure=True)

    # Exact LaTeX/BibTeX sequence requested:
    # latexmk -C                      # optional: cleans aux files
    # pdflatex main_CISE.tex
    # bibtex   main_CISE              # <-- this is the missing step
    # pdflatex main_CISE.tex
    # pdflatex main_CISE.tex

    # First LaTeX pass (to create .aux)
    run_command(f\"pdflatex -interaction=nonstopmode {tex_file}\", \"first LaTeX compilation\")

    # BibTeX
    run_command(f\"bibtex {tex_stem}\", \"bibtex compilation\")

    # Two more LaTeX passes
    run_command(f\"pdflatex -interaction=nonstopmode {tex_file}\", \"second LaTeX compilation\")
    run_command(f\"pdflatex -interaction=nonstopmode {tex_file}\", \"third LaTeX compilation\")

    # Verify PDF was generated
    if not os.path.exists(pdf_file):
        print(\"Error: PDF was not generated successfully\")
        sys.exit(1)

    print(\"PDF generated successfully!\")

    # Git operations
    timestamp = datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")
    commit_message = f\"Update {pdf_file} - {timestamp}\"

    print(\"Adding files to git...\")
    run_command(\"git add .\", \"git add\")

    print(\"Committing changes...\")
    try:
        run_command(f'git commit -m \"{commit_message}\"', \"git commit\")
    except SystemExit:
        # If there were no changes to commit, surface a friendly message instead of failing
        result = subprocess.run(\"git status --porcelain\", shell=True,
                                capture_output=True, text=True)
        if not result.stdout.strip():
            print(\"No changes to commit\")
        else:
            raise

    print(\"Pushing to remote...\")
    run_command(\"git push\", \"git push\")

    print(\"All operations completed successfully!\")

if __name__ == \"__main__\":
    main()
