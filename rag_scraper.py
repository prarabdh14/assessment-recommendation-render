from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import time
import json
import re
import os
import logging
import platform

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
BASE_URL = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"
OUTPUT_FILE = "shl_assessments_rag.csv"
CACHE_FILE = "assessment_cache.json"

# Setup Selenium
def setup_driver():
    """Setup and return a configured Chrome WebDriver."""
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--start-maximized")
    
    # Add user agent to appear more like a regular browser
    options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
    
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        options.binary_location = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    
    service = Service()
    driver = webdriver.Chrome(service=service, options=options)
    
    # Set page load timeout
    driver.set_page_load_timeout(30)
    
    return driver

def get_assessment_links(driver):
    """Retrieve all assessment links from the catalog page using Selenium."""
    logger.info("Fetching assessment links from catalog...")
    try:
        driver.get(CATALOG_URL)
        time.sleep(5)  # Give more time for JavaScript to load
        
        # Try different possible class names and selectors
        selectors = [
            "a.product-card",
            "a[href*='/solutions/products/']",
            ".product-card",
            ".assessment-card",
            "a[class*='card']",
            "a[href*='assessment']"
        ]
        
        links = []
        for selector in selectors:
            try:
                # Wait for elements with current selector
                elements = WebDriverWait(driver, 5).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, selector))
                )
                if elements:
                    logger.info(f"Found elements with selector: {selector}")
                    for element in elements:
                        href = element.get_attribute('href')
                        if href and '/solutions/products/' in href:
                            links.append(href)
            except Exception as e:
                logger.debug(f"Selector {selector} failed: {e}")
                continue
        
        # Remove duplicates while preserving order
        links = list(dict.fromkeys(links))
        
        if not links:
            # If no links found, try to get all links from the page
            logger.info("No links found with selectors, trying to get all links...")
            all_links = driver.find_elements(By.TAG_NAME, "a")
            for link in all_links:
                href = link.get_attribute('href')
                if href and '/solutions/products/' in href:
                    links.append(href)
        
        logger.info(f"Found {len(links)} assessment links")
        if links:
            logger.info("Sample links:")
            for link in links[:3]:
                logger.info(f"- {link}")
        else:
            # Save page source for debugging
            with open("debug_page.html", "w", encoding="utf-8") as f:
                f.write(driver.page_source)
            logger.info("Saved page source to debug_page.html for inspection")
        
        return links
    except Exception as e:
        logger.error(f"Error fetching assessment links: {e}")
        return []

def extract_features_with_rag(text):
    """Extract features using pattern matching approach."""
    features = {
        "duration": "N/A",
        "remote": "No",
        "adaptive": "No"
    }
    
    try:
        # Extract duration using regex patterns
        duration_patterns = [
            r'Approximate Completion Time in minutes\s*=\s*(\d+)',
            r'completion time.*?(\d+)\s*minutes',
            r'duration.*?(\d+)\s*minutes',
            r'takes.*?(\d+)\s*minutes',
            r'(\d+)\s*minutes to complete',
            r'(\d+)\s*min',
            r'approximately\s*(\d+)\s*minutes'
        ]
        
        for pattern in duration_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                features["duration"] = f"{match.group(1)} minutes"
                break
        
        # Check for remote testing indicators
        remote_patterns = [
            r'remote\s+testing',
            r'online\s+assessment',
            r'virtual\s+assessment',
            r'web-based\s+test'
        ]
        
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in remote_patterns):
            features["remote"] = "Yes"
        
        # Check for adaptive testing indicators
        adaptive_patterns = [
            r'adaptive\s+testing',
            r'irt(\s+|\s*-\s*)based',
            r'item\s+response\s+theory',
            r'computer\s+adaptive'
        ]
        
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in adaptive_patterns):
            features["adaptive"] = "Yes"
        
        return features
    except Exception as e:
        logger.error(f"Error in feature extraction: {e}")
        return features

def parse_assessment_page(driver, url):
    """Parse an assessment page and extract detailed information."""
    logger.info(f"Parsing assessment page: {url}")
    try:
        driver.get(url)
        # Wait for the content to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Get the page source after JavaScript has rendered
        soup = BeautifulSoup(driver.page_source, "html.parser")
        
        # Extract title
        name = soup.find("h1").text.strip() if soup.find("h1") else "N/A"
        
        # Initialize assessment data
        assessment_data = {
            "Assessment Name": name,
            "URL": url,
            "Duration": "N/A",
            "Remote Testing": "No",
            "Adaptive Support": "No",
            "Job Level": "N/A"
        }
        
        # Look for duration in the page content
        # First try to find the exact completion time text
        completion_time_pattern = re.compile(r'Approximate Completion Time in minutes\s*=\s*(\d+)', re.IGNORECASE)
        
        # Search in all text elements
        for element in soup.find_all(text=True):
            if element.strip():  # Skip empty text nodes
                match = completion_time_pattern.search(element.strip())
                if match:
                    minutes = match.group(1)
                    assessment_data["Duration"] = f"{minutes} minutes"
                    break
        
        # If duration not found, try alternative patterns
        if assessment_data["Duration"] == "N/A":
            # Look in specific sections
            detail_sections = soup.find_all("div", class_="product-detail__key-feature")
            for section in detail_sections:
                text = section.get_text(strip=True)
                # Try different duration patterns
                duration_patterns = [
                    r'completion time.*?(\d+)\s*minutes',
                    r'duration.*?(\d+)\s*minutes',
                    r'takes.*?(\d+)\s*minutes',
                    r'(\d+)\s*minutes to complete',
                    r'(\d+)\s*min'
                ]
                
                for pattern in duration_patterns:
                    match = re.search(pattern, text.lower())
                    if match:
                        assessment_data["Duration"] = f"{match.group(1)} minutes"
                        break
                if assessment_data["Duration"] != "N/A":
                    break
        
        # Look for job level in different places
        # 1. Look in the breadcrumb navigation
        breadcrumbs = soup.find_all("nav", class_="breadcrumb")
        if breadcrumbs:
            for crumb in breadcrumbs:
                crumb_text = crumb.get_text().lower()
                if any(level in crumb_text for level in ["entry", "professional", "manager", "executive", "graduate"]):
                    for level in ["entry", "professional", "manager", "executive", "graduate"]:
                        if level in crumb_text:
                            assessment_data["Job Level"] = level.title()
                            break
        
        # 2. Look in specific sections
        detail_sections = soup.find_all("div", class_="product-detail__key-feature")
        for section in detail_sections:
            text = section.get_text(strip=True).lower()
            
            # Check for job level in the text
            if "level" in text or "seniority" in text or "position" in text:
                # Look for specific job level keywords
                job_levels = {
                    "entry": ["entry", "junior", "beginner", "graduate"],
                    "professional": ["professional", "intermediate", "experienced"],
                    "manager": ["manager", "management", "supervisor", "lead"],
                    "executive": ["executive", "senior", "director", "leadership"]
                }
                
                for level, keywords in job_levels.items():
                    if any(keyword in text.lower() for keyword in keywords):
                        assessment_data["Job Level"] = level.title()
                        break
        
        # 3. Look in the page URL
        url_lower = url.lower()
        job_level_patterns = {
            "Entry": ["/entry-", "entry/", "graduate-", "graduate/"],
            "Professional": ["/professional-", "professional/", "experienced-"],
            "Manager": ["/manager-", "manager/", "management-", "supervisor-"],
            "Executive": ["/executive-", "executive/", "leadership-", "senior-"]
        }
        
        for level, patterns in job_level_patterns.items():
            if any(pattern in url_lower for pattern in patterns):
                assessment_data["Job Level"] = level
                break
        
        # 4. Look in the page content
        content = soup.get_text().lower()
        job_level_phrases = {
            "Entry": ["entry level position", "entry-level role", "graduate position", "junior role"],
            "Professional": ["professional level", "experienced professional", "mid-level position"],
            "Manager": ["managerial position", "management role", "supervisory position"],
            "Executive": ["executive level", "senior position", "leadership role"]
        }
        
        for level, phrases in job_level_phrases.items():
            if any(phrase in content for phrase in phrases):
                assessment_data["Job Level"] = level
                break
        
        # Check for remote testing (look for green dot)
        remote_indicators = soup.find_all("div", class_="remote-indicator")
        for indicator in remote_indicators:
            # Check if there's a green dot or checkmark
            if indicator.find("span", class_="green-dot") or indicator.find("i", class_="check-icon"):
                assessment_data["Remote Testing"] = "Yes"
                break
        
        # If no explicit remote indicator found, check text content
        if assessment_data["Remote Testing"] == "No":
            remote_text = soup.find_all(string=re.compile(r'remote|online|virtual', re.IGNORECASE))
            if remote_text:
                assessment_data["Remote Testing"] = "Yes"
        
        # Check for adaptive support
        adaptive_indicators = soup.find_all(string=re.compile(r'adaptive|irt|item response theory', re.IGNORECASE))
        if adaptive_indicators:
            assessment_data["Adaptive Support"] = "Yes"
        
        # If RAG features are still needed, use them as a fallback
        if assessment_data["Duration"] == "N/A":
            # Extract all text content for RAG processing
            content = soup.get_text()
            
            # Extract features using RAG
            features = extract_features_with_rag(content)
            
            # Use RAG features as fallback
            if assessment_data["Duration"] == "N/A":
                assessment_data["Duration"] = features["duration"]
            if assessment_data["Remote Testing"] == "No" and features["remote"] == "Yes":
                assessment_data["Remote Testing"] = "Yes"
            if assessment_data["Adaptive Support"] == "No" and features["adaptive"] == "Yes":
                assessment_data["Adaptive Support"] = "Yes"
        
        logger.info(f"Successfully parsed assessment: {name}")
        return assessment_data
    except Exception as e:
        logger.error(f"Error parsing assessment page {url}: {e}")
        return None

def load_cache():
    """Load cached assessment data if available."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
    return {}

def save_cache(cache_data):
    """Save assessment data to cache."""
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache_data, f)
        logger.info("Cache saved successfully")
    except Exception as e:
        logger.error(f"Error saving cache: {e}")

def scrape_catalog():
    """Main function to scrape the assessment catalog using RAG."""
    logger.info("Starting catalog scraping with RAG...")
    
    # Initialize the driver
    driver = setup_driver()
    
    try:
        # Load cache
        cache = load_cache()
        
        # Get assessment links
        links = get_assessment_links(driver)
        logger.info(f"Found {len(links)} assessments to process")
        
        data = []
        for i, link in enumerate(links):
            # Check cache first
            if link in cache:
                logger.info(f"[{i+1}/{len(links)}] Using cached data for: {link}")
                data.append(cache[link])
                continue
            
            try:
                logger.info(f"[{i+1}/{len(links)}] Scraping: {link}")
                item = parse_assessment_page(driver, link)
                if item:
                    data.append(item)
                    cache[link] = item  # Update cache
                    save_cache(cache)  # Save cache after each successful scrape
                time.sleep(2)  # Polite delay between requests
            except Exception as e:
                logger.error(f"Failed to scrape {link}: {e}")
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(data)
        df.to_csv(OUTPUT_FILE, index=False)
        logger.info(f"✅ Saved {len(data)} assessments to {OUTPUT_FILE}")
    
    finally:
        # Always close the driver
        driver.quit()

if __name__ == "__main__":
    scrape_catalog() 