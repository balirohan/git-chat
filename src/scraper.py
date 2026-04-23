"""
GitLab Handbook Scraper
Fetches content from two main base domains - 'https://handbook.gitlab.com' & 'https://about.gitlab.com'
"""

import requests
from bs4 import BeautifulSoup
from typing import Optional
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache file path
CACHE_FILE = Path("data/scraped_content.json")


def save_content(content: dict, filepath: Path = CACHE_FILE):
    """
    Saves scraped content to a JSON file for caching.
    This prevents re-scraping on every run.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(content, f, indent=2)
    logger.info(f"Cached {len(content)} pages to {filepath}")


def load_content(filepath: Path = CACHE_FILE) -> dict:
    """
    Loads cached content if available.
    Returns empty dict if cache doesn't exist.
    """
    if filepath.exists():
        with open(filepath) as f:
            cached = json.load(f)
        logger.info(f"Loaded {len(cached)} pages from cache")
        return cached
    return {}


class GitLabScraper:
    """Handles scraping of GitLab handbook and direction pages."""

    BASE_DOMAINS = ["handbook.gitlab.com", "about.gitlab.com"]

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; GitLabChatbot/1.0; Educational Project)"
        })

    def fetch_page(self, url: str) -> Optional[str]:
        """Fetch a single page and return cleaned text content."""
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

    def extract_links(self, url: str) -> set[str]:
        """Extract all same-domain links from a single page (no recursion)."""
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")
            links = set()

            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]

                if href.startswith("#") or not href:
                    continue

                # Handle relative URLs
                if href.startswith("/"):
                    for domain in self.BASE_DOMAINS:
                        href = f"https://{domain}{href}"
                        break

                # Only include our domains
                for domain in self.BASE_DOMAINS:
                    if domain in href:
                        links.add(href)
                        break

            return links

        except requests.RequestException as e:
            logger.error(f"Failed to extract links from {url}: {e}")
            return set()

    def fetch_with_links(self, base_urls: list[str]) -> dict[str, str]:
        """
        Fetch base URLs + all links found in those pages (depth 1 only).

        Example:
            scraper = GitLabScraper()
            content = scraper.fetch_with_links([
                "https://handbook.gitlab.com",
                "https://about.gitlab.com/releases/whats-new/"
            ])
        """
        all_urls = set(base_urls)

        # Extract links from each base page
        for url in base_urls:
            logger.info(f"Extracting links from: {url}")
            links = self.extract_links(url)
            all_urls.update(links)
            logger.info(f"  Found {len(links)} links")

        # Fetch all collected URLs
        logger.info(f"\nTotal URLs to fetch: {len(all_urls)}")
        content = {}
        for url in all_urls:
            logger.info(f"Fetching: {url}")
            text = self.fetch_page(url)
            if text:
                content[url] = text

        return content

    def fetch_pages(self, urls: list[str]) -> dict[str, str]:
        """Fetch multiple URLs (legacy method for compatibility)."""
        content = {}
        for url in urls:
            text = self.fetch_page(url)
            if text:
                content[url] = text
        return content


if __name__ == "__main__":
    scraper = GitLabScraper()

    # Fetch base pages + all links found in them (depth 1 only)
    base_urls = [
        "https://handbook.gitlab.com",
        "https://about.gitlab.com/releases/whats-new/"
    ]

    content = scraper.fetch_with_links(base_urls)

    # Save to cache
    save_content(content)

    logger.info(f"\nTotal pages scraped: {len(content)}")
    for url, text in list(content.items())[:3]:
        print(f"\n{'='*50}")
        print(f"URL: {url}")
        print(f"Characters: {len(text)}")