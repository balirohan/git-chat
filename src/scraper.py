"""
GitLab Handbook Scraper
- handbook.gitlab.com: requests + link extraction (static HTML)
- about.gitlab.com: Playwright with targeted link extraction (JS-rendered)
"""

import requests
from bs4 import BeautifulSoup
from typing import Optional, Set
from pathlib import Path
import json
import logging

from playwright.sync_api import sync_playwright

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_FILE = Path("data/scraped_content_final.json")


def save_content(content: dict, filepath: Path = CACHE_FILE):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(content, f, indent=2)
    logger.info(f"Cached {len(content)} pages to {filepath}")


def load_content(filepath: Path = CACHE_FILE) -> dict:
    if filepath.exists():
        with open(filepath) as f:
            cached = json.load(f)
        logger.info(f"Loaded {len(cached)} pages from cache")
        return cached
    return {}


class GitLabScraper:
    """Handles scraping of GitLab handbook and direction pages."""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })

    # ===== Handbook (static HTML) =====

    def fetch_page_requests(self, url: str) -> Optional[str]:
        """Fetch using requests (for static HTML pages)."""
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)
            lines = [line for line in text.split("\n") if len(line.strip()) > 50]
            return "\n".join(lines)

        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None

    def extract_links_requests(self, url: str) -> Set[str]:
        """Extract links using requests."""
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")
            links = set()

            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]

                if href.startswith("#") or not href:
                    continue

                if href.startswith("/"):
                    href = f"https://handbook.gitlab.com{href}"

                if "handbook.gitlab.com" in href:
                    links.add(href)

            return links

        except requests.RequestException as e:
            logger.error(f"Failed to extract links from {url}: {e}")
            return set()

    def scrape_handbook(self) -> dict[str, str]:
        """Scrape handbook.gitlab.com - extract links from base page and fetch all."""
        content = {}

        # Extract links from handbook base
        base_url = "https://handbook.gitlab.com"
        logger.info(f"Extracting links from: {base_url}")
        links = self.extract_links_requests(base_url)
        logger.info(f"  Found {len(links)} links")

        # Fetch base page
        text = self.fetch_page_requests(base_url)
        if text:
            content[base_url] = text

        # Fetch all discovered links
        for url in links:
            logger.info(f"Fetching: {url}")
            text = self.fetch_page_requests(url)
            if text:
                content[url] = text

        return content

    # ===== About GitLab (JS-rendered with Playwright) =====

    def fetch_whats_new_playwright(self) -> tuple[dict[str, str], list[dict]]:
        """
        Fetch about.gitlab.com/releases/whats-new/ using Playwright.
        Returns (content dict, roadmap items list)
        """
        content = {}
        roadmap_items = []

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )
                page = context.new_page()

                logger.info("Navigating to about.gitlab.com/releases/whats-new/...")
                page.goto(
                    "https://about.gitlab.com/releases/whats-new/#whats-coming",
                    wait_until="domcontentloaded",
                    timeout=60000
                )
                page.wait_for_timeout(5000)

                # Click "What's coming" tab
                try:
                    tab_locator = page.locator("button", has_text="What's coming")
                    if tab_locator.is_visible():
                        tab_locator.click()
                        page.wait_for_timeout(5000)
                        logger.info("Clicked 'What's Coming' tab")
                except Exception as e:
                    logger.warning(f"Could not click tab: {e}")

                # Extract content from page
                html_content = page.content()
                browser.close()

            soup = BeautifulSoup(html_content, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)
            lines = [line for line in text.split("\n") if len(line.strip()) > 50]
            content["https://about.gitlab.com/releases/whats-new/"] = "\n".join(lines)

            # Extract roadmap items (blogs, issues, epics)
            for el in soup.find_all("a", href=True):
                try:
                    href = el.get("href")
                    text = el.text.strip()

                    if not href or len(text) < 5:
                        continue

                    full_url = href if href.startswith("http") else f"https://about.gitlab.com{href}"
                    clean_url = full_url.split("#")[0].split("?")[0].rstrip("/")

                    source_type = "other"
                    if "/blog/" in clean_url:
                        source_type = "blog"
                    elif "/issues/" in clean_url or "/work_items/" in clean_url:
                        source_type = "technical_issue"
                    elif "/epics/" in clean_url:
                        source_type = "epic_roadmap"

                    if source_type != "other":
                        roadmap_items.append({
                            "title": text.replace("\n", " "),
                            "url": clean_url,
                            "type": source_type
                        })

                except Exception:
                    continue

            # Deduplicate
            unique_items = {item["url"]: item for item in roadmap_items}.values()
            roadmap_items = sorted(unique_items, key=lambda x: x["type"])

        except Exception as e:
            logger.error(f"Failed to fetch whats-new with Playwright: {e}")

        return content, roadmap_items

    def fetch_about_page(self, url: str) -> Optional[str]:
        """Fetch a single about.gitlab.com page using Playwright."""
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=60000)
                page.wait_for_timeout(3000)

                content = page.content()
                browser.close()

            soup = BeautifulSoup(content, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)
            lines = [line for line in text.split("\n") if len(line.strip()) > 50]
            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Failed to fetch {url} with Playwright: {e}")
            return None

    def scrape_about(self) -> dict[str, str]:
        """Scrape about.gitlab.com using Playwright."""
        content = {}
        roadmap_items = []

        # Fetch whats-new page
        whats_new_content, roadmap_items = self.fetch_whats_new_playwright()
        content.update(whats_new_content)
        logger.info(f"Found {len(roadmap_items)} roadmap items")

        # Fetch roadmap item pages (blogs, issues, epics)
        for item in roadmap_items:
            logger.info(f"Fetching roadmap item: {item['url']}")
            text = self.fetch_about_page(item["url"])
            if text:
                content[item["url"]] = text

        return content

    # ===== Main scraping =====

    def scrape_all(self) -> dict[str, str]:
        """Scrape both handbook.gitlab.com and about.gitlab.com."""
        all_content = {}

        logger.info("=" * 50)
        logger.info("Scraping handbook.gitlab.com...")
        handbook_content = self.scrape_handbook()
        all_content.update(handbook_content)
        logger.info(f"Handbook complete: {len(handbook_content)} pages")

        logger.info("=" * 50)
        logger.info("Scraping about.gitlab.com...")
        about_content = self.scrape_about()
        all_content.update(about_content)
        logger.info(f"About complete: {len(about_content)} pages")

        return all_content


if __name__ == "__main__":
    scraper = GitLabScraper()
    content = scraper.scrape_all()

    save_content(content)

    logger.info("=" * 50)
    logger.info(f"TOTAL pages scraped: {len(content)}")
    for url, text in list(content.items())[:3]:
        print(f"\n{url} ({len(text)} chars)")
