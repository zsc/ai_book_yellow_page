#!/usr/bin/env python3
"""
Sync zsc's GitHub repos into this yellow page (index.html).

Features:
- Add missing repos that have GitHub Pages enabled (or html/ directory).
- Fix dead preview links by selecting a working GitHub Pages URL; otherwise link to GitHub and mark as broken.
- Fix HTML directory links to use the repo's default branch and available source path.
- Fill missing tags (allow multi-tags, max 3) with lightweight heuristics.
- Update stats and tag counts in index.html.

Usage:
  source try_git_pat/.env
  python3 .tools/sync_yellow_page.py
"""

from __future__ import annotations

import html as html_lib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


OWNER = "zsc"
INDEX_FILE = "index.html"

TAG_ORDER: List[str] = [
    "智能体",
    "AI/大模型",
    "其他",
    "编程/系统",
    "芯片/硬件",
    "自动驾驶/汽车",
    "视频/音频",
    "公司历史",
    "商业/管理",
    "游戏",
    "图形学/3D",
    "机器人",
]

TAG_BG_COLOR: Dict[str, str] = {
    "AI/大模型": "#5a7a9c",
    "智能体": "#8b5a9f",
    "其他": "#7a808c",
    "编程/系统": "#6ca8d0",
    "芯片/硬件": "#b0a060",
    "自动驾驶/汽车": "#b8a0c0",
    "视频/音频": "#c08090",
    "公司历史": "#b08070",
    "商业/管理": "#7a6aa8",
    "游戏": "#b8a0b8",
    "图形学/3D": "#6ab88a",
    "机器人": "#5a9cc8",
}


ENTRY_PRIORITY: List[str] = [
    "index.html",
    "visualization.html",
    "visualize_all.html",
    "main.html",
    "demo.html",
    "app.html",
    "home.html",
]


@dataclass(frozen=True)
class RepoInfo:
    name: str
    default_branch: str
    description: Optional[str]
    fork: bool

    @property
    def key(self) -> str:
        return self.name.lower()


@dataclass
class CardData:
    data_name: str
    desc: str
    keywords: str
    tags: List[str]
    github_url: str
    html_url: str
    preview_url: str
    preview_broken: bool
    preview_reason: str

    @property
    def tags_str(self) -> str:
        return " ".join(self.tags)


def load_token() -> str:
    token = os.environ.get("GITHUB_READ_TOKEN")
    if token:
        return token.strip()

    env_file = os.path.join(os.path.dirname(__file__), "..", "try_git_pat", ".env")
    if os.path.exists(env_file):
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "GITHUB_READ_TOKEN" in line and "=" in line:
                    token = line.split("=", 1)[1].strip().strip('"')
                    if token:
                        return token

    raise RuntimeError("GITHUB_READ_TOKEN not found (set env or try_git_pat/.env)")


def github_request(url: str, token: str) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        headers={
            "Authorization": f"token {token}",
            "User-Agent": "ai_book_yellow_page",
            "Accept": "application/vnd.github+json",
        },
    )


def fetch_json(url: str, token: str, timeout_s: int = 30, retries: int = 3) -> Any:
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = github_request(url, token)
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            # Respect rate limiting (very lightweight).
            if e.code == 403 and e.headers.get("X-RateLimit-Remaining") == "0":
                reset = e.headers.get("X-RateLimit-Reset")
                if reset and reset.isdigit():
                    sleep_s = max(1, int(reset) - int(time.time()) + 1)
                    print(f"Rate limited. Sleeping {sleep_s}s...", file=sys.stderr)
                    time.sleep(sleep_s)
                    continue
            last_err = e
            if e.code in (500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(0.5 * (attempt + 1))
                continue
            raise
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(0.5 * (attempt + 1))
                continue
            raise
    raise last_err or RuntimeError("fetch_json failed")


def list_repos(token: str) -> List[RepoInfo]:
    repos: List[RepoInfo] = []
    page = 1
    while True:
        url = (
            f"https://api.github.com/users/{OWNER}/repos"
            f"?per_page=100&page={page}&type=owner"
        )
        data = fetch_json(url, token)
        if not data:
            break
        for r in data:
            repos.append(
                RepoInfo(
                    name=r["name"],
                    default_branch=r.get("default_branch") or "main",
                    description=r.get("description"),
                    fork=bool(r.get("fork")),
                )
            )
        if len(data) < 100:
            break
        page += 1
    return repos


def parse_existing_cards(index_html: str) -> Dict[str, Tuple[str, str, List[str]]]:
    """
    Return mapping: repo_key -> (desc, keywords, tags_list).
    """
    mapping: Dict[str, Tuple[str, str, List[str]]] = {}
    pattern = re.compile(
        r'<div class="repo-card"[^>]*data-name="([^"]+)"[^>]*data-desc="([^"]*)"'  # name, desc
        r'[^>]*data-keywords="([^"]*)"'  # keywords
        r'[^>]*data-tags="([^"]*)"',  # tags
        re.I,
    )
    for name, desc, keywords, tags in pattern.findall(index_html):
        tag_list = [t for t in tags.split() if t]
        mapping[name.lower()] = (html_unescape(desc), html_unescape(keywords), tag_list)
    return mapping


def html_escape(text: str) -> str:
    return html_lib.escape(text, quote=True)


def html_unescape(text: str) -> str:
    return html_lib.unescape(text)


def head_status(url: str, timeout_s: int = 10) -> Optional[int]:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return int(getattr(resp, "status", 200))
    except urllib.error.HTTPError as e:
        return int(e.code)
    except Exception:
        return None


def pages_info(repo: RepoInfo, token: str) -> Optional[Dict[str, Any]]:
    url = f"https://api.github.com/repos/{OWNER}/{repo.name}/pages"
    try:
        return fetch_json(url, token, timeout_s=20, retries=2)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        return None
    except Exception:
        return None


def list_dir(repo: RepoInfo, token: str, path: str) -> Optional[List[Dict[str, Any]]]:
    path = path.strip("/")
    url = (
        f"https://api.github.com/repos/{OWNER}/{repo.name}/contents"
        + (f"/{path}" if path else "")
        + f"?ref={repo.default_branch}"
    )
    try:
        data = fetch_json(url, token, timeout_s=20, retries=2)
        if isinstance(data, list):
            return data
        return None
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        return None
    except Exception:
        return None


def detect_entry(files: Sequence[str], priority: Sequence[str]) -> Optional[str]:
    lower_to_real = {f.lower(): f for f in files}
    for want in priority:
        if want.lower() in lower_to_real:
            return lower_to_real[want.lower()]
    # fallback: first html file
    for f in files:
        if f.lower().endswith(".html"):
            return f
    return None


def choose_preview_url(
    repo: RepoInfo,
    *,
    pages: Optional[Dict[str, Any]],
    html_files: Optional[Sequence[str]],
    root_html_files: Optional[Sequence[str]],
) -> Tuple[str, bool, str]:
    """
    Returns: (url, broken, reason)
    - broken=False means url is a working GitHub Pages URL (HEAD 200).
    - broken=True means we couldn't find a working Pages URL; caller should link somewhere non-dead.
    """
    repo_url_segment = repo.name
    base = f"https://{OWNER}.github.io/{repo_url_segment}/"

    if pages is None:
        return "", True, "GitHub Pages 未启用"

    status = pages.get("status")
    if status and status != "built" and status != "building":
        return "", True, f"GitHub Pages 状态: {status}"

    candidates: List[str] = []

    # If we know an entry in html/, prefer that.
    if html_files:
        entry = detect_entry(html_files, ENTRY_PRIORITY)
        if entry:
            candidates.append(base + "html/" + entry)
        # common default
        candidates.append(base + "html/index.html")

    # Root index / base (works for repos built from root/docs output).
    candidates.extend([base + "index.html", base])

    # If base is 404 but there are root html files, try one.
    if root_html_files:
        entry = detect_entry(root_html_files, ENTRY_PRIORITY)
        if entry:
            candidates.insert(0, base + entry)

    # Deduplicate while preserving order.
    seen = set()
    uniq: List[str] = []
    for u in candidates:
        if u in seen:
            continue
        seen.add(u)
        uniq.append(u)

    for u in uniq:
        status_code = head_status(u, timeout_s=6)
        if status_code == 200:
            return u, False, ""

    return "", True, "未找到可用预览入口"


def normalize_tags(tags: Sequence[str]) -> List[str]:
    clean: List[str] = []
    for t in tags:
        t = t.strip()
        if not t:
            continue
        if t not in clean:
            clean.append(t)
    return clean


def classify_tags(name: str, desc: str, keywords: str) -> List[str]:
    text = f"{name} {desc} {keywords}".lower()

    tags: List[str] = []

    def add(tag: str) -> None:
        if tag not in tags:
            tags.append(tag)

    # Agents / LLM
    if any(k in text for k in ["agent", "智能体", "omni_agent", "cruiserbot"]):
        add("智能体")

    if any(
        k in text
        for k in [
            "llm",
            "mllm",
            "rag",
            "diffusion",
            "transformer",
            "vllm",
            "sft",
            "dpo",
            "pretrain",
            "大模型",
            "语言模型",
            "多模态",
            "视觉语言",
            "扩散",
            "生成",
        ]
    ):
        add("AI/大模型")

    # Chips / hardware
    if any(k in text for k in ["npu", "chip", "芯片", "硬件", "orin", "orinx", "asic", "fpga"]):
        add("芯片/硬件")

    # Auto / vehicle
    if any(
        k in text
        for k in [
            "auto",
            "autonomous",
            "drive",
            "driving",
            "vehicle",
            "car",
            "fsd",
            "自动驾驶",
            "智驾",
            "车辆",
            "导航",
            "cockpit",
        ]
    ):
        add("自动驾驶/汽车")

    # Robotics
    if any(k in text for k in ["robot", "机器人", "unitree", "四足", "manipulator"]):
        add("机器人")

    # Graphics / 3D / viz
    if any(
        k in text
        for k in [
            "3d",
            "mesh",
            "render",
            "graphics",
            "x3d",
            "svg",
            "ui",
            "vision",
            "可视化",
            "图形",
            "图形学",
            "建模",
            "动画",
        ]
    ):
        add("图形学/3D")

    # Audio / video
    if any(
        k in text
        for k in [
            "audio",
            "video",
            "tts",
            "asr",
            "voice",
            "sound",
            "music",
            "语音",
            "音频",
            "视频",
            "合成器",
            "发音",
        ]
    ):
        add("视频/音频")

    # Games
    if any(k in text for k in ["game", "minecraft", "shooter", "tanks", "genshin", "twine", "游戏"]):
        add("游戏")

    # Programming / systems / security
    if any(
        k in text
        for k in [
            "os",
            "linux",
            "compiler",
            "cuda",
            "cli",
            "tool",
            "script",
            "coq",
            "lean",
            "cryptography",
            "crypto",
            "hacker",
            "program",
            "system",
            "系统",
            "编程",
            "密码学",
            "安全",
            "自动化",
        ]
    ):
        add("编程/系统")

    # Company / biz
    if any(k in text for k in ["公司", "history", "历史", "meituan", "ali_", "tesla"]):
        add("公司历史")

    if any(
        k in text
        for k in [
            "finance",
            "经济",
            "商业",
            "管理",
            "seo",
            "公关",
            "marketing",
            "会计",
            "cpa",
            "推荐",
        ]
    ):
        add("商业/管理")

    # Default
    if not tags:
        tags = ["其他"]

    # Max 3 tags, prefer earlier inferred ones.
    tags = tags[:3]

    # Ensure only known tags, otherwise fallback.
    tags = [t for t in tags if t in TAG_ORDER] or ["其他"]
    return tags


def build_card_html(card: CardData) -> str:
    repo_name_escaped = html_escape(card.data_name)
    desc_attr = html_escape(card.desc)
    keywords_attr = html_escape(card.keywords)
    tags_attr = html_escape(card.tags_str)
    desc_text = html_escape(card.desc)

    preview_class = "link-preview" + (" broken" if card.preview_broken else "")
    preview_title = html_escape(card.preview_reason) if card.preview_broken and card.preview_reason else ""
    preview_title_attr = f' title="{preview_title}"' if preview_title else ""

    tags_html = "".join(
        f'<span class="repo-tag" style="background: {TAG_BG_COLOR.get(t, TAG_BG_COLOR["其他"])};">{html_escape(t)}</span>'
        for t in card.tags
    )

    return (
        f'            <div class="repo-card" data-name="{repo_name_escaped}" '
        f'data-desc="{desc_attr}" data-keywords="{keywords_attr}" data-tags="{tags_attr}">\n'
        f'                <div class="repo-header">\n'
        f'                    <span class="repo-name">{repo_name_escaped}</span>\n'
        f"                </div>\n"
        f'                <div class="repo-tags">\n'
        f"                    {tags_html}\n"
        f"                </div>\n"
        f'                <div class="repo-desc">{desc_text}</div>\n'
        f'                <div class="repo-links">\n'
        f'                    <a href="{html_escape(card.github_url)}" class="link-github" target="_blank">GitHub</a>\n'
        f'                    <a href="{html_escape(card.html_url)}" class="link-html" target="_blank">HTML</a>\n'
        f'                    <a href="{html_escape(card.preview_url)}" class="{preview_class}" target="_blank"{preview_title_attr}>预览</a>\n'
        f"                </div>\n"
        f"            </div>"
    )


def replace_repo_grid(index_html: str, new_cards_html: str) -> str:
    marker = '<div class="repo-grid" id="repoGrid">'
    start = index_html.find(marker)
    if start == -1:
        raise RuntimeError("repoGrid not found in index.html")

    # Find matching </div> for repoGrid.
    div_start = start + index_html[start:].find("<div")
    depth = 0
    pos = div_start
    while True:
        next_open = index_html.find("<div", pos)
        next_close = index_html.find("</div>", pos)
        if next_close == -1:
            raise RuntimeError("Unbalanced divs while locating repoGrid end")
        if next_open != -1 and next_open < next_close:
            depth += 1
            pos = next_open + 4
        else:
            depth -= 1
            pos = next_close + len("</div>")
            if depth == 0:
                end = pos
                break

    # Split into: opening tag line + inner + closing </div>
    open_end = index_html.find(">", start) + 1
    before = index_html[:open_end]
    after = index_html[end - len("</div>") :]

    # Keep existing indentation style: blank line then cards.
    inner = "\n\n" + new_cards_html.rstrip() + "\n\n        "
    return before + inner + after


def update_total_repo_count(index_html: str, total: int) -> str:
    # Header stat: 总仓库数
    pattern = re.compile(
        r'(<div class="stat-item">\s*<div class="number">)(\d+)(</div>\s*<div class="label">总仓库数</div>)',
        re.S,
    )
    index_html, n = pattern.subn(rf"\g<1>{total}\g<3>", index_html, count=1)
    if n == 0:
        print("Warning: failed to update header total repo count", file=sys.stderr)

    # Result info
    index_html = re.sub(r"显示 <strong>\d+</strong> 个仓库", f"显示 <strong>{total}</strong> 个仓库", index_html)
    return index_html


def update_tag_counts(index_html: str, tag_counts: Dict[str, int]) -> str:
    for tag in TAG_ORDER:
        count = tag_counts.get(tag, 0)
        pattern = re.compile(
            rf'(data-tag="{re.escape(tag)}"[^>]*>{re.escape(tag)}<span class="count">)(\d+)(</span>)'
        )
        index_html, _ = pattern.subn(rf"\g<1>{count}\g<3>", index_html)
    return index_html


def compute_tag_counts(cards: Sequence[CardData]) -> Dict[str, int]:
    counts: Dict[str, int] = {t: 0 for t in TAG_ORDER}
    for c in cards:
        for t in set(c.tags):
            if t in counts:
                counts[t] += 1
    return counts


def main() -> int:
    token = load_token()
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        index_html = f.read()

    existing = parse_existing_cards(index_html)

    repos = list_repos(token)
    repo_map: Dict[str, RepoInfo] = {r.key: r for r in repos}

    # Render ALL repos under github.com/zsc (keeps existing descriptions/keywords/tags when present).
    all_keys = sorted(repo_map.keys())

    # Fetch Pages info / directory listings for preview & html links.
    pages_cache: Dict[str, Optional[Dict[str, Any]]] = {}
    html_files_cache: Dict[str, Optional[List[str]]] = {}
    root_html_files_cache: Dict[str, Optional[List[str]]] = {}
    html_dir_exists_cache: Dict[str, bool] = {}

    def collect_repo_assets(key: str) -> Tuple[str, Optional[Dict[str, Any]], Optional[List[str]], Optional[List[str]]]:
        repo = repo_map.get(key)
        if repo is None:
            return key, None, None, None

        pages = pages_info(repo, token)

        html_dir = list_dir(repo, token, "html")
        html_files: Optional[List[str]] = None
        if html_dir is not None:
            html_dir_exists_cache[key] = True
            html_files = [i["name"] for i in html_dir if i.get("type") == "file" and i["name"].lower().endswith(".html")]
        else:
            html_dir_exists_cache[key] = False

        root_dir = list_dir(repo, token, "")
        root_html_files: Optional[List[str]] = None
        if root_dir is not None:
            root_html_files = [i["name"] for i in root_dir if i.get("type") == "file" and i["name"].lower().endswith(".html")]

        return key, pages, html_files, root_html_files

    with ThreadPoolExecutor(max_workers=25) as ex:
        futures = [ex.submit(collect_repo_assets, key) for key in all_keys]
        for fut in as_completed(futures):
            key, pages, html_files, root_html_files = fut.result()
            pages_cache[key] = pages
            html_files_cache[key] = html_files
            root_html_files_cache[key] = root_html_files

    # Build base card fields (tags, desc, links).
    base: Dict[str, Dict[str, Any]] = {}
    filled_tags = 0
    reduced_tags = 0

    for key in all_keys:
        repo = repo_map.get(key)
        if repo is None:
            continue

        old_desc, old_keywords, old_tags = existing.get(key, ("", "", []))
        desc = old_desc or (repo.description or repo.name)
        keywords = old_keywords or f"{repo.name} {desc}"

        tags = normalize_tags(old_tags)
        if not tags:
            tags = classify_tags(repo.name, desc, keywords)
            filled_tags += 1
        else:
            if len(tags) > 3:
                tags = tags[:3]
                reduced_tags += 1

        # Ensure only known tags and max 3.
        tags = [t for t in tags if t in TAG_ORDER]
        if not tags:
            tags = ["其他"]
        if len(tags) > 3:
            tags = tags[:3]

        # Special: reduce overly broad tag sets (keep relevant 3).
        if key == "create_auto_vehicle_sys":
            preferred = ["自动驾驶/汽车", "AI/大模型", "编程/系统"]
            tags = [t for t in preferred if t in tags] + [t for t in tags if t not in preferred]
            tags = tags[:3]

        github_url = f"https://github.com/{OWNER}/{repo.name}"

        pages = pages_cache.get(key)
        html_dir_exists = html_dir_exists_cache.get(key, False)

        # HTML link: prefer html/ when available; else prefer Pages source path; else repo root.
        html_url = github_url
        if html_dir_exists:
            html_url = f"https://github.com/{OWNER}/{repo.name}/tree/{repo.default_branch}/html"
        elif pages and isinstance(pages.get("source"), dict):
            source = pages["source"]
            src_branch = source.get("branch") or repo.default_branch
            src_path = (source.get("path") or "/").lstrip("/")  # '' means root
            if src_path:
                html_url = f"https://github.com/{OWNER}/{repo.name}/tree/{src_branch}/{src_path}"
            else:
                html_url = f"https://github.com/{OWNER}/{repo.name}/tree/{src_branch}"
        else:
            html_url = github_url

        base[key] = {
            "repo": repo,
            "desc": desc,
            "keywords": keywords,
            "tags": tags,
            "github_url": github_url,
            "html_url": html_url,
            "pages": pages,
            "html_dir_exists": html_dir_exists,
        }

    # Compute preview URLs concurrently (HEAD checks can be slow if done serially).
    def compute_preview(key: str) -> Tuple[str, str, bool, str]:
        info = base.get(key)
        if info is None:
            return key, "", True, "缺少仓库信息"

        repo: RepoInfo = info["repo"]
        pages = info.get("pages")

        preview_url, broken, reason = choose_preview_url(
            repo,
            pages=pages,
            html_files=html_files_cache.get(key),
            root_html_files=root_html_files_cache.get(key),
        )

        # Explicit override from user request.
        if key == "2dfft_video_demo":
            return key, f"https://{OWNER}.github.io/2dfft_video_demo/visualization.html", False, ""

        # Special case: tolerate wikidata_tutorial build errors.
        if key == "wikidata_tutorial" and pages and pages.get("status") == "errored":
            return key, "", True, "GitHub Pages 构建失败（special case）"

        return key, preview_url, broken, reason

    preview_result: Dict[str, Tuple[str, bool, str]] = {}
    with ThreadPoolExecutor(max_workers=60) as ex:
        futures = [ex.submit(compute_preview, key) for key in base.keys()]
        for fut in as_completed(futures):
            key, url, broken, reason = fut.result()
            preview_result[key] = (url, broken, reason)

    # Build CardData list.
    cards: List[CardData] = []
    preview_fixed_or_set = 0
    preview_broken_count = 0
    for key in all_keys:
        info = base.get(key)
        if info is None:
            continue

        preview_url, broken, reason = preview_result.get(key, ("", True, "缺少预览结果"))
        if broken:
            preview_broken_count += 1
            preview_url = info["html_url"] if info["html_dir_exists"] else info["github_url"]
        else:
            preview_fixed_or_set += 1

        cards.append(
            CardData(
                data_name=key,
                desc=info["desc"],
                keywords=info["keywords"],
                tags=info["tags"],
                github_url=info["github_url"],
                html_url=info["html_url"],
                preview_url=preview_url,
                preview_broken=broken,
                preview_reason=reason,
            )
        )

    # Order: keep original order for existing cards as much as possible, append new ones sorted.
    existing_order: List[str] = []
    for m in re.finditer(r'data-name="([^"]+)"', index_html):
        existing_order.append(m.group(1).lower())

    order_rank: Dict[str, int] = {name: i for i, name in enumerate(existing_order)}
    cards.sort(key=lambda c: (order_rank.get(c.data_name, 10**9), c.data_name))

    tag_counts = compute_tag_counts(cards)

    new_cards_html = "\n\n".join(build_card_html(c) for c in cards)
    index_html = replace_repo_grid(index_html, new_cards_html)
    index_html = update_total_repo_count(index_html, len(cards))
    index_html = update_tag_counts(index_html, tag_counts)

    # Visual hint for broken preview links.
    if ".link-preview.broken" not in index_html:
        index_html = index_html.replace(
            ".link-preview {",
            ".link-preview.broken {\n"
            "            background: rgba(176, 181, 188, 0.22);\n"
            "            color: #7a808c;\n"
            "            border-color: rgba(176, 181, 188, 0.35);\n"
            "        }\n"
            "        .link-preview.broken:hover {\n"
            "            opacity: 1;\n"
            "            transform: none;\n"
            "        }\n\n        .link-preview {",
        )

    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        f.write(index_html)

    existing_names = set(existing.keys())
    added = len([k for k in repo_map.keys() if k not in existing_names])
    print("=== Sync done ===")
    print(f"Repos in index: {len(cards)} (added {added})")
    print(f"Tags filled: {filled_tags}, tags reduced: {reduced_tags}")
    print(f"Preview working: {preview_fixed_or_set}, preview broken: {preview_broken_count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
