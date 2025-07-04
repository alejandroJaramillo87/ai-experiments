#!/bin/bash

echo "--- Searching for Python Installations ---"
echo "This script will look for Python executables in common locations."
echo "You may see 'Permission denied' errors; these are normal for system directories."
echo "------------------------------------------------------------------"
echo ""

# 1. Search common PATH directories
echo "1. Python executables in your system's PATH:"
which -a python python2 python3 python3.? || echo "  No Python executables found directly in PATH."
echo ""

# 2. Search for pyenv-managed Python versions (if pyenv is installed)
echo "2. Python versions managed by pyenv (if installed):"
if command -v pyenv &> /dev/null; then
    pyenv versions | while read -r line; do
        version_name=$(echo "$line" | sed -e 's/^[ *]*//' -e 's/ (.*)//')
        if [[ "$version_name" != "system" ]]; then
            py_path="$HOME/.pyenv/versions/$version_name/bin/python"
            if [ -f "$py_path" ]; then
                echo "  $py_path"
                "$py_path" --version 2>/dev/null
            fi
        fi
    done
    if [ -z "$(pyenv versions | grep -v 'system')" ]; then
        echo "  No non-system pyenv-managed Python versions found."
    fi
else
    echo "  pyenv command not found. pyenv-managed Python versions will not be listed."
fi
echo ""

# 3. Search for virtual environments (.venv, venv, .env) in your home directory
echo "3. Python executables within virtual environments in your home directory:"
FOUND_VENV=false
find "$HOME" -maxdepth 4 -type d \( -name ".venv" -o -name "venv" -o -name ".env" -o -name ".pyenv" \) 2>/dev/null | while read -r venv_dir; do
    python_exec="$venv_dir/bin/python"
    if [ -f "$python_exec" ]; then
        echo "  $python_exec"
        "$python_exec" --version 2>/dev/null
        FOUND_VENV=true
    fi
done
if ! $FOUND_VENV; then
    echo "  No common virtual environments (.venv, venv, .env) found in your home directory."
fi
echo ""

echo "--- Search Complete ---"
echo "The above lists show various Python executables found on your system."
echo "The 'system Python' is typically located in /usr/bin/python3."
echo "-----------------------"