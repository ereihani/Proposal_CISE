#!/usr/bin/env python3
import subprocess
import sys
import os
from datetime import datetime

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error {description}: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False

def main():
    # Check if main_CISE.tex exists
    if not os.path.exists("main_CISE.tex"):
        print("Error: main_CISE.tex not found in current directory")
        sys.exit(1)
    
    print("Generating PDF from main_CISE.tex...")
    
    # Complete LaTeX compilation sequence with bibtex
    if not run_command("pdflatex -interaction=nonstopmode main_CISE.tex", "first LaTeX compilation"):
        sys.exit(1)
    
    if not run_command("bibtex main_CISE", "bibtex compilation"):
        sys.exit(1)
    
    if not run_command("pdflatex -interaction=nonstopmode main_CISE.tex", "second LaTeX compilation"):
        sys.exit(1)
    
    if not run_command("pdflatex -interaction=nonstopmode main_CISE.tex", "third LaTeX compilation"):
        sys.exit(1)
    
    # Check if PDF was generated
    if not os.path.exists("main_CISE.pdf"):
        print("Error: PDF was not generated successfully")
        sys.exit(1)
    
    print("PDF generated successfully!")
    
    # Git operations
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    commit_message = f"Update main_CISE.pdf - {timestamp}"
    
    print("Adding files to git...")
    if not run_command("git add .", "git add"):
        sys.exit(1)
    
    print("Committing changes...")
    if not run_command(f'git commit -m "{commit_message}"', "git commit"):
        # Check if there were no changes to commit
        result = subprocess.run("git status --porcelain", shell=True, 
                              capture_output=True, text=True)
        if not result.stdout.strip():
            print("No changes to commit")
        else:
            sys.exit(1)
    
    print("Pushing to remote...")
    if not run_command("git push", "git push"):
        sys.exit(1)
    
    print("All operations completed successfully!")

if __name__ == "__main__":
    main()