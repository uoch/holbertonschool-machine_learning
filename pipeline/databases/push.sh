#!/bin/bash

usage() {
    echo "Usage: $0 [-m <commit_message>] <file1> <file2> ..."
    exit 1
}

# Check if the number of arguments is less than 2
if [ "$#" -lt 1 ]; then
    usage
fi

# Parse options
while getopts ":m:" opt; do
    case ${opt} in
        m )
            commit_message="$OPTARG"
            ;;
        \? )
            echo "Invalid option: $OPTARG" 1>&2
            usage
            ;;
        : )
            echo "Invalid option: $OPTARG requires an argument" 1>&2
            usage
            ;;
    esac
done
shift $((OPTIND -1))

# Check if commit message is provided, otherwise use default
if [ -z "$commit_message" ]; then
    commit_message="done"
fi

# Check if there are files to commit
if [ "$#" -lt 1 ]; then
    echo "No files provided. Exiting."
    usage
fi

# Commit changes
echo "Changes detected. Proceeding with commit and push."
git add "$@"
git commit -m "$commit_message"
git push
echo "Push successful."
