#!/usr/bin/env python3
"""
系统化处理'编程/系统'类别下的预览链接
- 扫描所有带有'编程/系统'标签的 repos
- 检测 html/目录结构
- 自动修复预览链接
"""

import json
import re
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_prog_system_repos():
    """获取所有带有'编程/系统'标签的 repos"""
    with open('index.html', 'r') as f:
        content = f.read()
    
    repos = []
    pattern = r'<div class="repo-card"[^>]*data-name="([^"]+)"[^>]*data-tags="([^"]*)"'
    for match in re.finditer(pattern, content):
        name = match.group(1)
        tags = match.group(2)
        if '编程/系统' in tags:
            repos.append(name)
    return repos

def check_repo_structure(repo, token):
    """检查 repo 的 html 目录结构"""
    url = f"https://api.github.com/repos/zsc/{repo}/contents/html?ref=main"
    headers = {"Authorization": f"token {token}"}
    
    result = {
        'repo': repo,
        'has_html_dir': False,
        'has_index_html': False,
        'entry_file': None,
        'alternative_files': []
    }
    
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            files = [item['name'] for item in data if item['type'] == 'file']
            
            result['has_html_dir'] = True
            result['has_index_html'] = 'index.html' in files
            
            # 查找其他可能的入口文件
            priority = ['visualize_all.html', 'main.html', 'demo.html', 'app.html', 'home.html']
            for entry in priority:
                if entry in files:
                    result['entry_file'] = entry
                    break
            
            # 记录所有 html 文件
            result['alternative_files'] = [f for f in files if f.endswith('.html')]
            
    except urllib.error.HTTPError as e:
        if e.code == 404:
            # html 目录不存在，检查根目录
            result['has_html_dir'] = False
            result = check_root_directory(repo, token, result)
    except Exception as e:
        result['error'] = str(e)
    
    return result

def check_root_directory(repo, token, result):
    """检查 repo 根目录是否有 HTML 文件"""
    url = f"https://api.github.com/repos/zsc/{repo}/contents/?ref=main"
    headers = {"Authorization": f"token {token}"}
    
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            files = [item['name'] for item in data if item['type'] == 'file']
            
            # 查找根目录下的 html 文件
            html_files = [f for f in files if f.endswith('.html')]
            result['alternative_files'] = html_files
            
            if html_files:
                # 按优先级选择
                priority = ['demo.html', 'visualize_all.html', 'app.html', 'main.html', 'home.html']
                for entry in priority:
                    if entry in html_files:
                        result['entry_file'] = entry
                        result['in_root'] = True
                        break
                
                # 如果没有优先级的，选第一个
                if not result['entry_file']:
                    result['entry_file'] = html_files[0]
                    result['in_root'] = True
                    
    except Exception as e:
        result['error'] = str(e)
    
    return result

def update_preview_links(results):
    """更新 index.html 中的预览链接"""
    with open('index.html', 'r') as f:
        content = f.read()
    
    updated = []
    
    for result in results:
        repo = result['repo']
        entry = result.get('entry_file')
        in_root = result.get('in_root', False)
        
        if not entry:
            continue
        
        # 构建正确的 URL
        if in_root:
            new_url = f"https://zsc.github.io/{repo}/{entry}"
        else:
            new_url = f"https://zsc.github.io/{repo}/html/{entry}"
        
        # 检查当前链接
        old_pattern = rf'href="https://zsc\.github\.io/{repo}/html(/?)"'
        
        if re.search(old_pattern, content):
            content = re.sub(old_pattern, f'href="{new_url}"', content)
            updated.append(f"{repo} -> {new_url}")
    
    with open('index.html', 'w') as f:
        f.write(content)
    
    return updated

def update_config(results):
    """更新 preview_config.json"""
    config_path = '.tools/preview_config.json'
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        config = {'special_previews': {}, 'entry_file_priority': []}
    
    for result in results:
        repo = result['repo']
        entry = result.get('entry_file')
        
        if entry and (not result['has_index_html'] or result.get('in_root')):
            config['special_previews'][repo] = {
                'entry_file': entry,
                'entry_path': '' if result.get('in_root') else 'html/',
                'has_index_html': result['has_index_html'],
                'description': f"Auto-detected entry for {repo}"
            }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    return config

def main():
    token = os.environ.get('GITHUB_READ_TOKEN')
    if not token:
        print("Error: GITHUB_READ_TOKEN not found")
        return
    
    repos = get_prog_system_repos()
    print(f"Found {len(repos)} repos with '编程/系统' tag")
    print("\nScanning repositories...")
    
    results = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        future_to_repo = {executor.submit(check_repo_structure, repo, token): repo for repo in repos}
        
        for future in as_completed(future_to_repo):
            result = future.result()
            results.append(result)
            
            # 打印需要特殊处理的 repos
            if not result.get('has_index_html') and result.get('entry_file'):
                location = "root" if result.get('in_root') else "html/"
                print(f"  ⚠️  {result['repo']}: using {location}{result['entry_file']}")
    
    print(f"\nUpdating config...")
    config = update_config(results)
    
    print(f"Updating index.html...")
    updated = update_preview_links(results)
    
    print(f"\n✅ Updated {len(updated)} preview links:")
    for item in updated:
        print(f"  - {item}")

if __name__ == '__main__':
    main()
