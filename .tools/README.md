# 黄页同步/预览链接工具

## 文件说明

- `sync_yellow_page.py` - **推荐**：同步 `github.com/zsc` 全部仓库到 `index.html`，并修复/标注预览链接、补齐标签
- `preview_config.json` / `update_previews.py` - 旧版预览链接扫描工具（如需可继续用）

## 使用方法

### 1. 一键同步（推荐）
```bash
cd /Users/georgezhou/Downloads/ai_book_yellow_page
source try_git_pat/.env
python3 .tools/sync_yellow_page.py
```

### 2. （旧版）手动添加特殊预览链接
编辑 `preview_config.json`:
```json
{
  "special_previews": {
    "dtw_cvx_demo": {
      "entry_file": "visualize_all.html",
      "description": "DTW CVX visualization demo"
    },
    "new_repo": {
      "entry_file": "app.html",
      "description": "Single page app"
    }
  }
}
```

然后运行脚本应用更改。

### 3. （旧版）入口文件优先级
在 `preview_config.json` 中配置检测优先级：
```json
{
  "entry_file_priority": [
    "visualize_all.html",
    "main.html",
    "demo.html",
    "app.html",
    "home.html",
    "index.html"
  ]
}
```

脚本会按优先级检测，使用第一个存在的文件作为入口。

## 工作流程

1. **自动发现**: 脚本扫描所有 repos 的 html 目录
2. **配置更新**: 将新发现的特殊入口添加到配置文件
3. **应用更改**: 更新 index.html 中的预览链接

## 当前配置

{
  "special_previews": {
    "dtw_cvx_demo": {
      "entry_file": "visualize_all.html",
      "description": "DTW CVX visualization demo"
    }
  },
  "entry_file_priority": [
    "visualize_all.html",
    "main.html",
    "demo.html",
    "app.html",
    "home.html",
    "index.html"
  ]
}
