#!/usr/bin/env python3
"""
系统地管理 GitHub Pages 预览链接
- 扫描所有 repos 的 html 目录
- 检测特殊入口文件
- 更新 index.html 中的预览链接
"""

import json
import re
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_config():
    """加载配置文件"""
    config_path = os.path.join(os.path.dirname(__file__), 'preview_config.json')
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"special_previews": {}, "entry_file_priority": ["index.html"]}

def save_config(config):
    """保存配置文件"""
    config_path = os.path.join(os.path.dirname(__file__), 'preview_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def scan_repo_html_files(repo_name, token):
    """扫描单个 repo 的 html 目录文件"""
    import urllib.request
    import urllib.error
    
    url = f"https://api.github.com/repos/zsc/{repo_name}/contents/html?ref=main"
    headers = {"Authorization": f"token {token}"}
    
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            import json
            data = json.loads(response.read().decode())
            return {item['name'] for item in data if item['type'] == 'file'}
    except Exception as e:
        return set()

def detect_entry_file(files, priority_list):
    """根据优先级检测入口文件"""
    for entry in priority_list:
        if entry in files:
            return entry
    return "index.html"  # 默认

def scan_all_repos(token, repo_list_file='/tmp/repo_names.txt'):
    """扫描所有 repos 的入口文件"""
    with open(repo_list_file, 'r') as f:
        repos = [line.strip() for line in f if line.strip()]
    
    config = load_config()
    priority = config.get('entry_file_priority', ['index.html'])
    
    special_previews = {}
    
    print(f"Scanning {len(repos)} repositories...")
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_repo = {
            executor.submit(scan_repo_html_files, repo, token): repo 
            for repo in repos
        }
        
        for future in as_completed(future_to_repo):
            repo = future_to_repo[future]
            try:
                files = future.result()
                if files:
                    entry = detect_entry_file(files, priority)
                    if entry != "index.html":
                        special_previews[repo] = {
                            "entry_file": entry,
                            "description": f"Entry file: {entry}"
                        }
                        print(f"  Found special entry: {repo} -> {entry}")
            except Exception as e:
                print(f"  Error scanning {repo}: {e}")
    
    return special_previews

def update_config_from_scan(special_previews):
    """从扫描结果更新配置文件"""
    config = load_config()
    
    # 合并新旧配置（保留手动配置，添加新发现）
    for repo, info in special_previews.items():
        if repo not in config['special_previews']:
            config['special_previews'][repo] = info
            print(f"Added to config: {repo} -> {info['entry_file']}")
    
    save_config(config)
    return config

def apply_preview_links_to_index(index_file='index.html'):
    """应用配置到 index.html"""
    config = load_config()
    special_previews = config.get('special_previews', {})
    
    with open(index_file, 'r') as f:
        content = f.read()
    
    updated_count = 0
    
    for repo, info in special_previews.items():
        entry_file = info['entry_file']
        new_url = f"https://zsc.github.io/{repo}/html/{entry_file}"
        old_url_pattern = f"https://zsc.github.io/{repo}/html\""
        
        # 检查是否需要更新
        if old_url_pattern in content:
            # 构建替换模式（保留 class 和 target 属性）
            pattern = rf'(href=")https://zsc\.github\.io/{repo}/html(\".*?class="link-preview")'
            replacement = rf'\g<1>{new_url}\g<2>'
            
            new_content = re.sub(pattern, replacement, content)
            if new_content != content:
                content = new_content
                updated_count += 1
                print(f"Updated: {repo} -> {entry_file}")
    
    with open(index_file, 'w') as f:
        f.write(content)
    
    return updated_count

def main():
    """主函数"""
    import sys
    
    # 获取 GitHub token
    token = os.environ.get('GITHUB_READ_TOKEN')
    if not token:
        # 尝试从 .env 文件读取
        env_file = os.path.join(os.path.dirname(__file__), '..', 'try_git_pat', '.env')
        if os.path.exists(env_file):
            with open(env_file, 'r') as f:
                for line in f:
                    if 'GITHUB_READ_TOKEN' in line:
                        token = line.split('=')[1].strip().strip('"')
                        break
    
    if not token:
        print("Error: GITHUB_READ_TOKEN not found")
        sys.exit(1)
    
    print("=== Step 1: Scanning repositories ===")
    special_previews = scan_all_repos(token)
    
    print(f"\n=== Step 2: Updating config ({len(special_previews)} found) ===")
    config = update_config_from_scan(special_previews)
    
    print("\n=== Step 3: Applying to index.html ===")
    updated = apply_preview_links_to_index()
    
    print(f"\nDone! Updated {updated} preview links.")
    print(f"Config file: .tools/preview_config.json")

if __name__ == '__main__':
    main()
