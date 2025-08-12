#!/usr/bin/env python3

# Read the main document
with open('main_CISE.tex', 'r') as f:
    lines = f.readlines()

# Read the replacement content
with open('intellectual_merit_condensed.tex', 'r') as f:
    replacement_content = f.read()

# Find line numbers for replacement (0-indexed)
# Section starts at line 91 (index 90), ends at line 347 (index 346)
start_line = 90  # Line 91 in 1-indexed
end_line = 347   # Line 348 in 1-indexed (exclusive)

# Replace the section
new_lines = lines[:start_line] + [replacement_content + '\n'] + lines[end_line:]

# Write back to file
with open('main_CISE.tex', 'w') as f:
    f.writelines(new_lines)

print(f"Replaced lines {start_line+1} to {end_line} with condensed content")