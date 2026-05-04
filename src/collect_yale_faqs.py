"""Collect FAQ-like lines from public Yale HTML pages (course project, polite crawl).

This script is intended to help grow ``collected_questions.csv`` toward thousands of
rows. It only fetches URLs under hostnames mapped to the six RouteRight labels,
checks ``robots.txt`` when available, rate-limits requests, and skips PDFs.

Review output before training: automated extraction is noisy; delete bad rows or
tune heuristics as needed.
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.preprocess import clean_text

USER_AGENT = (
    "RouteRight/0.1 (Yale CPSC 381/581 course project; polite FAQ text collection)"
)

# Hosts we are willing to crawl for each label (keep tight to reduce mis-routing).
HOSTS_BY_CATEGORY: dict[str, frozenset[str]] = {
    "Registrar": frozenset({"registrar.yale.edu"}),
    "Financial_Aid": frozenset({"finaid.yale.edu", "financialaid.yale.edu"}),
    "Housing": frozenset({"housing.yale.edu"}),
    "IT": frozenset({"help.canvas.yale.edu", "canvas.yale.edu"}),
    "Dining": frozenset({"hospitality.yale.edu"}),
    "Health_Wellness": frozenset({"yalehealth.yale.edu", "campushealth.yale.edu"}),
}

BOILERPLATE_SUBSTRINGS = (
    "cookie",
    "javascript",
    "skip to",
    "sign in",
    "log in",
    "privacy policy",
    "facebook",
    "instagram",
    "twitter",
    "youtube",
)


@dataclass
class CrawlRow:
    text: str
    label: str
    source_name: str
    school: str
    url: str


def can_fetch(url: str, robots_by_host: dict[str, RobotFileParser]) -> bool:
    parts = urlparse(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return False
    host = parts.netloc.lower()
    robots_url = f"{parts.scheme}://{host}/robots.txt"
    if host not in robots_by_host:
        rp = RobotFileParser()
        rp.set_url(robots_url)
        try:
            rp.read()
        except Exception:
            # If robots cannot be read, be conservative but allow small academic fetches.
            robots_by_host[host] = rp
            return True
        robots_by_host[host] = rp
    rp = robots_by_host[host]
    try:
        return rp.can_fetch(USER_AGENT, url)
    except Exception:
        return True


def _in_boilerplate_nav(tag) -> bool:
    for anc in tag.parents:
        if anc is None or not getattr(anc, "name", None):
            continue
        if anc.name in {"nav", "footer", "header"}:
            return True
        cid = (anc.get("id") or "").lower()
        classes = " ".join(anc.get("class") or []).lower()
        if any(
            x in classes or x in cid
            for x in ("footer", "navbar", "menu", "breadcrumb", "site-header")
        ):
            return True
    return False


def extract_candidates(soup: BeautifulSoup, page_url: str) -> list[str]:
    out: list[str] = []
    for tag in soup.find_all(["h2", "h3", "h4"]):
        if _in_boilerplate_nav(tag):
            continue
        t = tag.get_text(" ", strip=True)
        t = " ".join(t.split())
        if 18 <= len(t) <= 700:
            out.append(t)
    for tag in soup.find_all("li"):
        if _in_boilerplate_nav(tag):
            continue
        t = tag.get_text(" ", strip=True)
        t = " ".join(t.split())
        if 28 <= len(t) <= 520:
            out.append(t)
    for tag in soup.find_all("p"):
        if _in_boilerplate_nav(tag):
            continue
        t = tag.get_text(" ", strip=True)
        t = " ".join(t.split())
        if 45 <= len(t) <= 900:
            low = t.lower()
            if "?" in t or low.startswith(
                (
                    "if ",
                    "you ",
                    "when ",
                    "once ",
                    "students ",
                    "student ",
                    "please ",
                    "note:",
                    "note ",
                    "for ",
                    "the ",
                    "this ",
                )
            ):
                out.append(t)
    # De-dupe within page while preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for t in out:
        ct = clean_text(t)
        if len(ct) < 18:
            continue
        if any(b in ct for b in BOILERPLATE_SUBSTRINGS):
            continue
        if ct in seen:
            continue
        seen.add(ct)
        uniq.append(t)
    return uniq


def normalize_url(base: str, href: str) -> str | None:
    joined = urljoin(base, href)
    p = urlparse(joined)
    if p.scheme not in ("http", "https"):
        return None
    if p.path.lower().endswith(".pdf"):
        return None
    fragless = p._replace(fragment="").geturl()
    return fragless


def enqueue_links(
    soup: BeautifulSoup,
    page_url: str,
    category: str,
    allowed: frozenset[str],
    queue: deque[tuple[str, str, str]],
    seen_urls: set[str],
    max_links_per_page: int,
) -> None:
    n = 0
    for a in soup.find_all("a", href=True):
        if n >= max_links_per_page:
            break
        nu = normalize_url(page_url, a["href"])
        if not nu:
            continue
        host = urlparse(nu).netloc.lower()
        if host not in allowed:
            continue
        if nu in seen_urls:
            continue
        seen_urls.add(nu)
        queue.append((nu, category, "crawl"))
        n += 1


def fetch_html(session: requests.Session, url: str, timeout: float) -> str | None:
    try:
        r = session.get(url, timeout=timeout, allow_redirects=True)
        if r.status_code != 200:
            return None
        ctype = (r.headers.get("content-type") or "").lower()
        if "pdf" in ctype:
            return None
        return r.text
    except requests.RequestException:
        return None


def load_seeds(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"category", "url", "source_name"}
    if not need.issubset(df.columns):
        raise ValueError(f"Seed CSV must have columns: {sorted(need)}")
    return df


def collect(
    seeds: pd.DataFrame,
    *,
    max_pages: int,
    delay_s: float,
    max_links_per_page: int,
    request_timeout: float,
) -> list[CrawlRow]:
    robots_by_host: dict[str, RobotFileParser] = {}
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    queue: deque[tuple[str, str, str]] = deque()
    for _, row in seeds.iterrows():
        cat = str(row["category"]).strip()
        url = str(row["url"]).strip()
        src = str(row["source_name"]).strip()
        if cat not in HOSTS_BY_CATEGORY:
            continue
        queue.append((url, cat, src))

    visited: set[str] = set()
    queued: set[str] = {str(row["url"]).strip() for _, row in seeds.iterrows()}
    rows: list[CrawlRow] = []

    while queue and len(visited) < max_pages:
        url, category, source_name = queue.popleft()
        if url in visited:
            continue
        allowed = HOSTS_BY_CATEGORY.get(category)
        if not allowed or urlparse(url).netloc.lower() not in allowed:
            visited.add(url)
            continue
        if not can_fetch(url, robots_by_host):
            visited.add(url)
            continue

        visited.add(url)
        time.sleep(delay_s)
        html = fetch_html(session, url, request_timeout)
        if not html:
            continue

        soup = BeautifulSoup(html, "html.parser")
        for chunk in extract_candidates(soup, url):
            ct = clean_text(chunk)
            if len(ct) < 18:
                continue
            rows.append(
                CrawlRow(
                    text=chunk.strip(),
                    label=category,
                    source_name=source_name,
                    school="Yale",
                    url=url,
                )
            )
        enqueue_links(
            soup,
            url,
            category,
            allowed,
            queue,
            queued,
            max_links_per_page=max_links_per_page,
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Yale FAQ-like snippets into CSV")
    parser.add_argument(
        "--seed-csv",
        default="data/raw/yale_faq_seed_urls.csv",
        help="CSV with columns: category,url,source_name",
    )
    parser.add_argument(
        "--output",
        default="data/raw/collected_questions_yale_scraped.csv",
        help="Output CSV (text,label,source_name,school,url)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=350,
        help="Maximum distinct HTML pages to fetch (BFS from seeds)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.25,
        help="Seconds to sleep between HTTP requests (be polite)",
    )
    parser.add_argument(
        "--max-links-per-page",
        type=int,
        default=40,
        help="Cap on discovered same-domain links enqueued per page",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=25.0,
        help="HTTP timeout seconds",
    )
    parser.add_argument(
        "--merge-into",
        default=None,
        help="Optional path to append/dedupe into (e.g. data/raw/collected_questions.csv)",
    )
    args = parser.parse_args()

    seeds = load_seeds(args.seed_csv)
    rows = collect(
        seeds,
        max_pages=args.max_pages,
        delay_s=args.delay,
        max_links_per_page=args.max_links_per_page,
        request_timeout=args.timeout,
    )

    df = pd.DataFrame([r.__dict__ for r in rows])
    if df.empty:
        print("No rows collected (check network, robots, or seed URLs).")
        return

    df["_c"] = df["text"].map(clean_text)
    df = df.drop_duplicates(subset=["_c"], keep="first").drop(columns=["_c"])

    out_path = args.output
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(df)} rows from {args.max_pages} max pages crawl.")

    if args.merge_into:
        base = pd.read_csv(args.merge_into)
        merged = pd.concat([base, df], ignore_index=True)
        merged["_c"] = merged["text"].map(clean_text)
        merged = merged.drop_duplicates(subset=["_c"], keep="first").drop(columns=["_c"])
        merged.to_csv(args.merge_into, index=False)
        print(f"Merged into {args.merge_into} -> {len(merged)} total rows (deduped).")


if __name__ == "__main__":
    main()
