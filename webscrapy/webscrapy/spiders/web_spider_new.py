from pathlib import Path
from bson.objectid import ObjectId
from bs4 import BeautifulSoup
import scrapy
import pymongo
from dotenv import load_dotenv
import os
from urllib.parse import urlparse, urljoin

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
        'CLOSESPIDER_PAGECOUNT': 5,
    }

    def __init__(self, start_urls=None, keywordId=None, *args, **kwargs):
        super(WebCrawSpider, self).__init__(*args, **kwargs)

        # Ensure URLs have https://
        for i in range(len(start_urls)):
            if not start_urls[i].startswith("https://") and not start_urls[i].startswith("http://"):
                start_urls[i] = "https://" + start_urls[i]

        self.keywordId = keywordId or ""
        self.client = pymongo.MongoClient(CONNECTION_STRING)
        self.db = self.client['webcrawl']
        self.collection = self.db['sitesData']
        self.collection2 = self.db['keyword']
        
        # Get previously crawled URLs using aggregate
        aggregate = [
            {
                '$match': {
                    '_id': ObjectId(keywordId)  # ✅ Convert to ObjectId
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

                print("previous_crawled_urls_list")
                print(previous_crawled_urls_list)
                self.previous_crawled_urls = set(previous_crawled_urls_list)
                print(f"✓ Loaded {len(self.previous_crawled_urls)} previously crawled URLs")
            else:
                print("No previous crawled URLs found (new keyword)")
                
        except Exception as e:
            print(f"Error loading previous URLs: {e}")
            self.previous_crawled_urls = set()
        
        # Filter out already crawled URLs from start_urls
        original_count = len(start_urls)
        original_start_urls = start_urls.copy()
        self.start_urls = [url for url in start_urls if url not in self.previous_crawled_urls]
        skipped_count = original_count - len(self.start_urls)
        
        # Track progress and visited URLs
        self.processed_count = 0
        self.success_count = 0
        self.fail_count = 0
        self.skipped_count = skipped_count
        self.visited_urls = set(self.previous_crawled_urls)  # Initialize with previous URLs
        self.max_pages = 100
        
        # IMPORTANT: Determine crawl mode based on number of URLs
        self.crawl_mode = "multiple_urls" if len(self.start_urls) > 1 else "single_url_with_links"
        
        # Extract allowed domains from start_urls
        self.allowed_domains = set()
        for url in self.start_urls:
            domain = urlparse(url).netloc
            self.allowed_domains.add(domain)
        
        print("=" * 80)
        print("WebSpider Initialized")
        print(f"Keyword ID: {self.keywordId}")
        print(f"Crawl Mode: {self.crawl_mode}")
        print(f"   > Multiple URLs ({len(self.start_urls)}): Parse only provided URLs" if self.crawl_mode == "multiple_urls" 
              else f"   > Single URL: Follow internal links (max {self.max_pages} pages)")
        print(f"Allowed domains: {', '.join(self.allowed_domains)}")
        print(f"Total URLs provided: {original_count}")
        print(f"Already crawled (skipped): {skipped_count}")
        print(f"New URLs to crawl: {len(self.start_urls)}")
        
        # Show skipped URLs
        if skipped_count > 0:
            print(f"\n  Skipped URLs (already crawled):")
            for url in original_start_urls:
                if url in self.previous_crawled_urls:
                    print(f"   ✓ {url}")
        
        # Show new URLs to crawl
        if len(self.start_urls) > 0:
            print(f"\n New URLs to crawl:")
            for i, url in enumerate(self.start_urls, 1):
                print(f"   [{i}] {url}")
        else:
            print("\ No new URLs to crawl - all URLs have been previously processed!")
        
        print("=" * 80)
        
        if not self.keywordId:
            raise Exception("Keyword id not found!")
        
        if not self.start_urls and original_count > 0:
            print("\n✓ All provided URLs were already crawled. Spider will exit gracefully.")

    def parse(self, response):
        """Parse each URL - called automatically for each URL in start_urls"""
        # Check if max pages reached (only for single URL mode)
        if self.crawl_mode == "single_url_with_links" and self.processed_count >= self.max_pages:
            print(f"\nMax pages limit ({self.max_pages}) reached. Stopping crawler.")
            return
        
        # Skip if already visited (includes previously crawled URLs)
        if response.url in self.visited_urls:
            print(f" Skipping already visited URL: {response.url}")
            return
        
        self.visited_urls.add(response.url)
        self.processed_count += 1
        
        print(f"\n[{self.processed_count}] Processing: {response.url}")
        
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
                "siteUrl": response.url,
                "content": body_text,
                "imageUrls": image_urls
            }

            # Save to MongoDB
            result = self.collection.insert_one(data)
            self.success_count += 1
            
            print(f"✓ SUCCESS - Saved to MongoDB!")
            print(f"   Document ID: {result.inserted_id}")
            print(f"   Content length: {len(body_text)} characters")
            print(f"   Images found: {len(image_urls)}")
            
            # LOGIC: Follow links ONLY if single URL mode
            if self.crawl_mode == "single_url_with_links":
                if self.processed_count < self.max_pages:
                    links_found = 0
                    
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        absolute_url = urljoin(response.url, href)
                        link_domain = urlparse(absolute_url).netloc
                        
                        # Only follow links from same domain AND not already visited
                        if (link_domain in self.allowed_domains and 
                            absolute_url not in self.visited_urls):
                            links_found += 1
                            print(f"   → Queuing: {absolute_url}")
                            yield scrapy.Request(absolute_url, callback=self.parse)
                    
                    if links_found > 0:
                        print(f"   Queued {links_found} new internal links")
                    else:
                        print(f"   No new internal links found")
                else:
                    print(f"   Max page limit reached - not queuing more links")
            else:
                # Multiple URLs mode - don't follow links
                print(f"   [Multiple URL Mode] - Not following internal links")
            
        except Exception as e:
            self.fail_count += 1
            print(f"✗ FAILED: {response.url}")
            print(f"   Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def closed(self, reason):
        """Called when spider finishes"""
        print("\n" + "=" * 80)
        print(f"Spider Finished: {reason}")
        print(f"Crawl Mode: {self.crawl_mode}")
        print("Results:")
        print(f"   Initial URLs provided: {len(self.start_urls) + self.skipped_count}")
        print(f"   Previously crawled (skipped): {self.skipped_count}")
        print(f"   New URLs attempted: {len(self.start_urls)}")
        if self.crawl_mode == "single_url_with_links":
            print(f"   Maximum pages limit: {self.max_pages}")
        print(f"   Total Processed: {self.processed_count}")
        print(f"   Success: {self.success_count}")
        print(f"   Failed: {self.fail_count}")
        print(f"   Total unique URLs in DB: {len(self.visited_urls)}")
        print("=" * 80)
        
        if hasattr(self, 'client'):
            self.client.close()