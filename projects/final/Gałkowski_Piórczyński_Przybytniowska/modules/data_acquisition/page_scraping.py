import argparse
import os
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


def save_file(url, content, extension, data_path):
    # Parse the URL to create a valid file path
    parsed_url = urlparse(url)
    path = parsed_url.path.strip("/")  # Convert URL path to a valid filename
    if not path:
        path = "index"  # Default for homepage

    # Use the original file name from the URL with appropriate extension
    filename = f"{data_path}/{path}"
    if not filename.endswith(extension):
        filename += extension

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "wb") as file:
        file.write(content)
    print(f"Saved: {filename}")


def log_failed_url(url, failed_urls_file):
    with open(failed_urls_file, "a", encoding="utf-8") as file:
        file.write(url + "\n")


def log_successful_url(url, successful_urls_file):
    with open(successful_urls_file, "a", encoding="utf-8") as file:
        file.write(url + "\n")


def load_failed_urls(failed_urls_file):
    if os.path.exists(failed_urls_file):
        with open(failed_urls_file, "r", encoding="utf-8") as file:
            return set(line.strip() for line in file.readlines())
    return set()


def load_successful_urls(successful_urls_file):
    if os.path.exists(successful_urls_file):
        with open(successful_urls_file, "r", encoding="utf-8") as file:
            return set(line.strip() for line in file.readlines())
    return set()


def crawl_page(
    url, visited_urls, data_path, base_url, failed_urls_file, successful_urls_file
):
    if url in visited_urls:
        return
    visited_urls.add(url)

    print(f"Crawling: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()  # check if the request was successful
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch {url}: {e}")
        log_failed_url(url, failed_urls_file)  # log failed URL for retry later
        return

    content_type = response.headers.get("Content-Type", "")

    if "text/html" in content_type:
        soup = BeautifulSoup(response.text, "html.parser")
        page_content = soup.prettify()  # Get pretty HTML content
        page_title = soup.title.string if soup.title else "No Title"
        print(f"Page title: {page_title}")

        # Save the HTML to a file
        save_file(url, page_content.encode("utf-8"), ".html", data_path)
        log_successful_url(url, successful_urls_file)

        # Find all links on the page
        links = soup.find_all("a", href=True)
        for link in links:
            link_url = link["href"]

            # Convert relative URLs to absolute URLs
            full_url = urljoin(url, link_url)

            # Check if the full_url starts with the base domain (ensure it's within the same site)
            if base_url and full_url.startswith(base_url):
                crawl_page(
                    full_url,
                    visited_urls,
                    data_path,
                    base_url,
                    failed_urls_file,
                    successful_urls_file,
                )

    elif "application/pdf" in content_type:
        # Save PDF file with original path and filename
        print(f"Saving PDF from: {url}")
        save_file(url, response.content, ".pdf", data_path)
        log_successful_url(url, successful_urls_file)

    else:
        print(f"Skipping unsupported content type: {content_type} for URL: {url}")
        log_failed_url(
            url, failed_urls_file
        )  # Mark it as failed for retry or review later

    time.sleep(2)  # To avoid overloading the server


def main(start_url: str, data_path: str):
    """
    Crawl a website and download HTML and PDF content.

    Usage:
    python page_scraping.py <start_url> <data_path>

    Example:
    python page_scraping.py https://ww2.mini.pw.edu.pl/ ./data
    """

    os.makedirs(data_path, exist_ok=True)

    failed_urls_file = os.path.join(data_path, "failed_urls.txt")
    successful_urls_file = os.path.join(data_path, "successful_urls.txt")

    visited_urls = load_successful_urls(successful_urls_file)

    failed_urls = load_failed_urls(failed_urls_file)
    visited_urls.update(
        failed_urls
    )  # Add failed URLs to the visited set to avoid re-crawling

    # Get the base domain from the start URL to ensure we only crawl links within the same domain
    base_url = urlparse(start_url).scheme + "://" + urlparse(start_url).hostname

    crawl_page(
        start_url,
        visited_urls,
        data_path,
        base_url,
        failed_urls_file,
        successful_urls_file,
    )

    # Clearing the failed URLs file after the crawl is complete
    open(failed_urls_file, "w").close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crawl a website and download HTML and PDF content."
    )
    parser.add_argument(
        "--start_url",
        type=str,
        required=False,
        default="https://ww2.mini.pw.edu.pl",
        help="The URL to start crawling from",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=False,
        default="./data",
        help="Directory where the files will be saved",
    )

    args = parser.parse_args()
    main(args.start_url, args.data_path)
