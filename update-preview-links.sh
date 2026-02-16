#!/bin/bash
# 快捷脚本：更新预览链接
# Usage: ./update-preview-links.sh

cd "$(dirname "$0")"

if [ ! -f try_git_pat/.env ]; then
    echo "Error: try_git_pat/.env not found"
    exit 1
fi

source try_git_pat/.env
export GITHUB_READ_TOKEN

echo "Updating preview links..."
python3 .tools/sync_yellow_page.py

echo ""
echo "Done."
