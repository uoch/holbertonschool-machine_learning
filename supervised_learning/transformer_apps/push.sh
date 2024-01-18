#!/bin/bash

# Check if the correct number of arguments (2) is provided
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <file(s)> <commit_message>"
  exit 1
fi

files="$1"
commit_message="$2"

# Check if the files/directories exist
if [ ! -e "$files" ]; then
  echo "Error: File(s) not found: $files"
  exit 1
fi

# Add the specified files/directories to the staging area
git add "$files"

# Commit the changes with the provided commit message
git commit -m "$commit_message"

# Push the changes to the remote repository
git push origin
