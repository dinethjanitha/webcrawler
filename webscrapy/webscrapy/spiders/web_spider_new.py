from pathlib import Path
from bson.objectid import ObjectId
from bs4 import BeautifulSoup
import scrapy
import pymongo
from dotenv import load_dotenv
import os
import datetime
from urllib.parse import urlparse, urljoin, urlunparse

load_dotenv("./.env")

CONNECTION_STRING = os.getenv("CONNECTION_STRING")


class WebCrawSpider(scrapy.Spider):
    name = "WebSpider"
    
    # Custom settings
    custom_settings = {
        'ROBOTSTXT_OBEY': False,  # Bypass robots.txt blocking
        'CONCURRENT_REQUESTS': 1,  # Process URLs one at a time
        'DOWNLOAD_DELAY': 1,  # 1 second between requests
        'DOWNLOAD_TIMEOUT': 30,  # 30 second timeout per URL
        'RETRY_ENABLED': True,
        'RETRY_TIMES': 2,
        'LOG_LEVEL': 'INFO',
        'CLOSESPIDER_TIMEOUT': 0,
        'CLOSESPIDER_PAGECOUNT': 5,  # Stop after 100 pages
    }

    def __init__(self, start_urls=None, keywordId=None, *args, **kwargs):
        super(WebCrawSpider, self).__init__(*args, **kwargs)

        if not start_urls:
            raise Exception("start_urls cannot be empty!")
        
        if not keywordId:
            raise Exception("keywordId is required!")

        # Ensure URLs have https://
        for i in range(len(start_urls)):
            if not start_urls[i].startswith("https://") and not start_urls[i].startswith("http://"):
                start_urls[i] = "https://" + start_urls[i]

        self.keywordId = keywordId
        self.client = pymongo.MongoClient(CONNECTION_STRING)
        self.db = self.client['webcrawl']
        self.collection = self.db['sitesData']
        self.collection2 = self.db['keyword']
        
        # Get previously crawled URLs using aggregate
        aggregate = [
            {
                '$match': {
                    '_id': ObjectId(keywordId)
                }
            }, {
                '$lookup': {
                    'from': 'sitesData', 
                    'localField': '_id', 
                    'foreignField': 'keywordId', 
                    'as': 'sitesInfo'
                }
            }, {
                '$project': {
                    '_id': 1, 
                    'keyword': 1, 
                    'urls': '$sitesInfo.siteUrl'
                }
            }
        ]

        self.previous_crawled_urls = set()
        
        try:
            result = list(self.collection2.aggregate(aggregate))

            if result and len(result) > 0:
                previous_crawled_urls_list = result[0].get("urls", [])
                self.previous_crawled_urls = set(previous_crawled_urls_list)
                print(f"Loaded {len(self.previous_crawled_urls)} previously crawled URLs")
            else:
                print("No previous crawled URLs found (new keyword)")
                
        except Exception as e:
            print(f"Error loading previous URLs: {e}")
            import traceback
            traceback.print_exc()
            self.previous_crawled_urls = set()
        
        # remember original provided urls (before filtering)
        original_count = len(start_urls)
        original_start_urls = start_urls.copy()

        # Filter out already crawled URLs from start_urls
        self.start_urls = [url for url in start_urls if url not in self.previous_crawled_urls]
        skipped_count = original_count - len(self.start_urls)

        # === FIX: if user provided exactly ONE URL, don't silently drop it just because it exists in DB.
        # We want single-url mode to run and follow internal links even if that URL was previously crawled.
        if original_count == 1 and len(self.start_urls) == 0:
            # restore the original single URL so spider runs in single_url_with_links mode
            self.start_urls = [original_start_urls[0]]
            # adjust skipped_count accordingly (we still mark it as previously crawled)
            skipped_count = 1 if original_start_urls[0] in self.previous_crawled_urls else 0
        # === end fix

        # Track progress and visited URLs
        self.processed_count = 0
        self.success_count = 0
        self.fail_count = 0
        self.skipped_count = skipped_count
        self.visited_urls = set(self.previous_crawled_urls)
        self.queued_urls = set()
        self.max_pages = 100
        
        # Determine crawl mode
        if len(self.start_urls) == 0:
            self.crawl_mode = "none"
        elif len(self.start_urls) == 1:
            self.crawl_mode = "single_url_with_links"
        else:
            self.crawl_mode = "multiple_urls"
        
        # Extract allowed domains (include www and non-www versions)
        # NOTE: use original_start_urls so domain checks align with the provided URL(s),
        # even if we restored a single URL above.
        self.allowed_domains = set()
        for url in original_start_urls:
            parsed = urlparse(url)
            domain = parsed.netloc
            if domain:
                self.allowed_domains.add(domain)
                if domain.startswith('www.'):
                    self.allowed_domains.add(domain[4:])
                else:
                    self.allowed_domains.add('www.' + domain)
        
        print("=" * 80)
        print("WebSpider Initialized")
        print(f"Keyword ID: {self.keywordId}")
        print(f"Crawl Mode: {self.crawl_mode}")
        
        if self.crawl_mode == "multiple_urls":
            print(f"   Multiple URLs ({len(self.start_urls)}): Parse only provided URLs")
        elif self.crawl_mode == "single_url_with_links":
            print(f"   Single URL: Follow internal links (max {self.max_pages} pages)")
        else:
            print(f"   No URLs to crawl (all already processed)")
        
        print(f"Allowed domains: {', '.join(sorted(self.allowed_domains))}")
        print(f"Total URLs provided: {original_count}")
        print(f"Already crawled (skipped): {skipped_count}")
        print(f"New URLs to crawl: {len(self.start_urls)}")
        
        if len(self.start_urls) > 1 : 
            if skipped_count > 0:
                print(f"\nSkipped URLs (already crawled):")
                for url in original_start_urls:
                    if url in self.previous_crawled_urls:
                        print(f"   {url}")
            
            if len(self.start_urls) > 0:
                print(f"\nNew URLs to crawl:")
                for i, url in enumerate(self.start_urls, 1):
                    print(f"   [{i}] {url}")
            else:
                print("\nNo new URLs to crawl - all URLs have been previously processed!")            
            
        print("=" * 80)

    def normalize_url(self, url):
        """Normalize URL by removing fragments and trailing slashes"""
        parsed = urlparse(url)
        path = parsed.path.rstrip('/') if parsed.path != '/' else parsed.path
        
        normalized = urlunparse((
            parsed.scheme,
            parsed.netloc,
            path,
            parsed.params,
            parsed.query,
            ''  # Remove fragment
        ))
        
        return normalized

    def parse(self, response):
        """Parse each URL - called automatically for each URL in start_urls"""
        
        normalized_url = self.normalize_url(response.url)
        
        # Skip if already visited
        # if normalized_url in self.visited_urls:
        #     print(f"[SKIP] Already visited: {normalized_url}")
        #     return
        # === FIX: allow single-url-with-links mode to crawl even if URL was previously crawled.
        # if normalized_url in self.previous_crawled_urls and self.crawl_mode != "single_url_with_links":
        #     print(f"[SKIP] Already visited (previously crawled): {normalized_url}")
        #     return
        # === end fix
        
        # Check if max pages reached
        if self.crawl_mode == "single_url_with_links" and self.processed_count >= self.max_pages:
            print(f"\n[STOP] Max pages limit ({self.max_pages}) reached.")
            print(f"\n[STOP] Max pages limit ({self.max_pages}) reached.")
            print(f"\n[STOP] Max pages limit ({self.max_pages}) reached.")
            print(f"\n[STOP] Max pages limit ({self.max_pages}) reached.")
            print(f"\n[STOP] Max pages limit ({self.max_pages}) reached.")
            print(f"\n[STOP] Max pages limit ({self.max_pages}) reached.")
            return
        
        self.visited_urls.add(normalized_url)
        self.processed_count += 1
        
        print(f"\n[{self.processed_count}/{self.max_pages if self.crawl_mode == 'single_url_with_links' else len(self.start_urls)}] Processing: {normalized_url}")
        
        try:
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract image URLs
            image_urls = []
            for img in soup.find_all('img'):
                img_src = img.get('src') or img.get('data-src')
                if img_src:
                    absolute_img_url = urljoin(response.url, img_src)
                    image_urls.append(absolute_img_url)
            
            image_urls = list(set(image_urls))

            # Remove unwanted tags
            for tag in soup(["script", "style", "noscript", "header", "footer", "svg", "meta"]):
                tag.decompose()

            # Extract clean text
            body_text = " ".join(soup.get_text(separator=" ").split())

            # Prepare data
            data = {
                "keywordId": ObjectId(self.keywordId),
                "siteUrl": normalized_url,
                "content": body_text,
                "imageUrls": image_urls,
                "createdAt" : datetime.datetime.utcnow()
            }

            # Save to MongoDB
            result = self.collection.insert_one(data)
            self.success_count += 1
            
            print(f"[SUCCESS] Saved to MongoDB")
            print(f"   Document ID: {result.inserted_id}")
            print(f"   Content: {len(body_text)} chars")
            print(f"   Images: {len(image_urls)}")
            
            # Follow links ONLY if single URL mode

            print("Before start here")

            if len(self.start_urls) == 1 :
                print("Start here")
                if self.processed_count < self.max_pages:
                    links_found = 0
                    links_queued = 0
                    links_skipped_invalid = 0
                    links_skipped_external = 0
                    links_skipped_duplicate = 0
                    
                    print(f"\n   [EXTRACTING LINKS]")
                    
                    for link in soup.find_all('a', href=True):
                        href = link['href'].strip()
                        
                        print("Href")
                        print(href)
                        # Skip invalid links
                        if (not href or 
                            href.startswith('#') or 
                            href.startswith('javascript:') or
                            href.startswith('mailto:') or
                            href.startswith('tel:')):
                            links_skipped_invalid += 1
                            continue
                        
                        absolute_url = urljoin(response.url, href)
                        absolute_url = self.normalize_url(absolute_url)
                        
                        if not absolute_url.startswith(('http://', 'https://')):
                            links_skipped_invalid += 1
                            continue
                        
                        link_domain = urlparse(absolute_url).netloc
                        
                        links_found += 1
                        
                        # Check if domain is allowed
                        if link_domain not in self.allowed_domains:
                            links_skipped_external += 1
                            continue
                        
                        # Check if already visited or queued
                        if absolute_url in self.visited_urls or absolute_url in self.queued_urls:
                            links_skipped_duplicate += 1
                            continue
                        
                        # Queue the link
                        links_queued += 1
                        self.queued_urls.add(absolute_url)
                        
                        print(f"   [QUEUE {links_queued}] {absolute_url}")
                  
                        yield scrapy.Request(
                            absolute_url, 
                            callback=self.parse,
                        )
                    
                    print(f"\n   [LINK SUMMARY]")
                    print(f"      Total <a> tags: {len(soup.find_all('a', href=True))}")
                    print(f"      Valid links: {links_found}")
                    print(f"      Queued: {links_queued}")
                    print(f"      Skipped invalid: {links_skipped_invalid}")
                    print(f"      Skipped external: {links_skipped_external}")
                    print(f"      Skipped duplicate: {links_skipped_duplicate}")
                else:
                    print(f"   [LIMIT] Max page limit reached")
            else:
                print(f"   [MODE] Multiple URL mode - not following links")
            
        except Exception as e:
            self.fail_count += 1
            print(f"[FAILED] {response.url}")
            print(f"   Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def handle_error(self, failure):
        """Handle request errors"""
        self.fail_count += 1
        print(f"[ERROR] Request failed: {failure.request.url}")
        print(f"   Reason: {failure.value}")
    
    def closed(self, reason):
        """Called when spider finishes"""
        print("\n" + "=" * 80)
        print(f"Spider Finished: {reason}")
        print(f"Crawl Mode: {self.crawl_mode}")
        print("\nResults:")
        print(f"   Initial URLs: {len(self.start_urls) + self.skipped_count}")
        print(f"   Previously crawled: {self.skipped_count}")
        print(f"   New URLs: {len(self.start_urls)}")
        if self.crawl_mode == "single_url_with_links":
            print(f"   Max pages limit: {self.max_pages}")
        print(f"   Total Processed: {self.processed_count}")
        print(f"   Success: {self.success_count}")
        print(f"   Failed: {self.fail_count}")
        print(f"   Unique URLs visited: {len(self.visited_urls)}")
        print(f"   URLs queued: {len(self.queued_urls)}")
        print("=" * 80)
        
        if hasattr(self, 'client'):
            self.client.close()
