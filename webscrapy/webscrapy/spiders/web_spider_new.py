from pathlib import Path
from bson.objectid import ObjectId
from bs4 import BeautifulSoup
import scrapy
import pymongo
from dotenv import load_dotenv
import os
import datetime
from urllib.parse import urlparse, urljoin, urlunparse

"""
WebCrawSpider - Simple Web Crawler with Social Media Link Extraction

Single URL Mode:
- Starts from the exact URL you provide
- Follows internal links found on that page (up to 100 pages)
- Extracts social media links but does NOT crawl them

Multiple URL Mode:
- Crawls only the URLs you provide
- Does NOT follow any links
"""

load_dotenv("./.env")

CONNECTION_STRING = os.getenv("CONNECTION_STRING")


class WebCrawSpider(scrapy.Spider):
    name = "WebSpider"
    
    # Custom settings
    custom_settings = {
        'ROBOTSTXT_OBEY': False,
        'CONCURRENT_REQUESTS': 1,
        'DOWNLOAD_DELAY': 1,
        'DOWNLOAD_TIMEOUT': 30,
        'RETRY_ENABLED': True,
        'RETRY_TIMES': 2,
        'LOG_LEVEL': 'INFO',
        'CLOSESPIDER_TIMEOUT': 0,
        'CLOSESPIDER_PAGECOUNT': 5,  # Disabled, manual limit
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

        self.start_urls = start_urls
        self.keywordId = keywordId
        self.client = pymongo.MongoClient(CONNECTION_STRING)
        self.db = self.client['webcrawl']
        self.collection = self.db['sitesData']
        
        # Track progress and visited URLs
        self.processed_count = 0
        self.success_count = 0
        self.fail_count = 0
        self.visited_urls = set()  # URLs visited in THIS session only
        self.queued_urls = set()
        self.max_pages = 100
        
        # Determine crawl mode
        if len(self.start_urls) == 1:
            self.crawl_mode = "single_url_with_links"
        else:
            self.crawl_mode = "multiple_urls"
        
        # Extract allowed domains (include www and non-www versions)
        self.allowed_domains = set()
        for url in self.start_urls:
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
            print(f"   Multiple URLs: Parse only provided URLs")
        
        print(f"Allowed domains: {', '.join(sorted(self.allowed_domains))}")
        print(f"Maximum pages to crawl: {self.max_pages if self.crawl_mode == 'single_url_with_links' else len(self.start_urls)}")
        print(f"Initial URLs to crawl: {len(self.start_urls)}")
        for i, url in enumerate(self.start_urls, 1):
            print(f"   [{i}] {url}")
        print(f"\nNote: Will check database for each URL before saving")
        print(f"      Already crawled URLs will be skipped but links extracted")
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
        
        # Check if max pages reached
        if self.crawl_mode == "single_url_with_links" and self.processed_count >= self.max_pages:
            print(f"\n[STOP] Max pages limit ({self.max_pages}) reached.")
            return
        
        # Skip if already visited in this session
        if normalized_url in self.visited_urls:
            print(f"[SKIP] Already visited in this session: {normalized_url}")
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

            # Extract social media links
            social_media_domains = {
                'facebook.com', 'fb.com', 'fb.me', 'www.facebook.com',
                'twitter.com', 'x.com', 'www.twitter.com', 'www.x.com',
                'instagram.com', 'www.instagram.com',
                'linkedin.com', 'www.linkedin.com',
                'youtube.com', 'youtu.be', 'www.youtube.com',
                'tiktok.com', 'www.tiktok.com',
                'pinterest.com', 'www.pinterest.com',
                'reddit.com', 'www.reddit.com',
                'snapchat.com', 'www.snapchat.com',
                'whatsapp.com', 'wa.me',
                'telegram.org', 't.me'
            }
            
            social_media_links = []
            
            for link in soup.find_all('a', href=True):
                href = link['href'].strip()
                
                if not href or href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                    continue
                
                try:
                    absolute_url = urljoin(response.url, href)
                    link_domain = urlparse(absolute_url).netloc.lower()
                    
                    # Check if it's a social media link
                    for sm_domain in social_media_domains:
                        if sm_domain in link_domain:
                            social_media_links.append(absolute_url)
                            break
                except:
                    continue
            
            social_media_links = list(set(social_media_links))

            # Remove unwanted tags
            for tag in soup(["script", "style", "noscript", "header", "footer", "svg", "meta"]):
                tag.decompose()

            # Extract clean text
            body_text = " ".join(soup.get_text(separator=" ").split())

            # Check if URL already exists in database
            existing_doc = self.collection.find_one({
                "keywordId": ObjectId(self.keywordId),
                "siteUrl": response.url
            })
            
            if existing_doc:
                print(f"[SKIP] Already in database - not saving again")
                print(f"   Existing Document ID: {existing_doc['_id']}")
                print(f"   Will continue to extract links from this page")
            else:
                # Prepare data
                data = {
                    "keywordId": ObjectId(self.keywordId),
                    "siteUrl": response.url,
                    "content": body_text,
                    "imageUrls": image_urls,
                    # "socialMediaLinks": social_media_links,
                    "createdAt" : datetime.datetime.utcnow()
                }

                # Save to MongoDB
                result = self.collection.insert_one(data)
                self.success_count += 1
                
                print(f"[SUCCESS] Saved to MongoDB")
                print(f"   Document ID: {result.inserted_id}")
                print(f"   URL Saved: {response.url}")
                print(f"   Content: {len(body_text)} chars")
                print(f"   Images: {len(image_urls)}")
                print(f"   Social Media Links: {len(social_media_links)}")
                if social_media_links:
                    print(f"   Social Media Found:")
                    for sm_link in social_media_links[:3]:
                        print(f"      - {sm_link}")
                    if len(social_media_links) > 3:
                        print(f"      ... and {len(social_media_links) - 3} more")
            
            # Follow links ONLY if single URL mode
            if self.crawl_mode == "single_url_with_links":
                if self.processed_count < self.max_pages:
                    links_found = 0
                    links_queued = 0
                    links_skipped_invalid = 0
                    links_skipped_external = 0
                    links_skipped_duplicate = 0
                    
                    print(f"\n   [EXTRACTING LINKS]")
                    
                    # Social media domains to skip (don't crawl these)
                    social_media_domains = {
                        'facebook.com', 'fb.com', 'fb.me', 'www.facebook.com',
                        'twitter.com', 'x.com', 'www.twitter.com', 'www.x.com',
                        'instagram.com', 'www.instagram.com',
                        'linkedin.com', 'www.linkedin.com',
                        'youtube.com', 'youtu.be', 'www.youtube.com',
                        'tiktok.com', 'www.tiktok.com',
                        'pinterest.com', 'www.pinterest.com',
                        'reddit.com', 'www.reddit.com',
                        'snapchat.com', 'www.snapchat.com',
                        'whatsapp.com', 'wa.me',
                        'telegram.org', 't.me'
                    }
                    
                    for link in soup.find_all('a', href=True):
                        href = link['href'].strip()
                        
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
                        
                        link_domain = urlparse(absolute_url).netloc.lower()
                        
                        # Skip social media links (already collected separately)
                        is_social_media = any(sm_domain in link_domain for sm_domain in social_media_domains)
                        if is_social_media:
                            links_skipped_external += 1
                            continue
                        
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
        print(f"   New Documents Saved: {self.success_count}")
        print(f"   Already in DB (skipped): {self.processed_count - self.success_count - self.fail_count}")
        print(f"   Failed: {self.fail_count}")
        print(f"   Unique URLs visited: {len(self.visited_urls)}")
        print("=" * 80)
        
        if hasattr(self, 'client'):
            self.client.close()
