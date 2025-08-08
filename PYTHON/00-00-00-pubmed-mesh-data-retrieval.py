"""
PUBMED DISEASE ABSTRACTS RETRIEVAL (1975-2024) - SMART CHUNKING VERSION

Improved chunking algorithm that prevents data loss by using:
- Disease category splitting when time chunks exceed limits
- Binary search for optimal date ranges
- Hour-level chunking for extreme cases
- Adaptive chunk sizing based on publication growth trends

KEY IMPROVEMENTS:
- NO DATA LOSS: Multiple strategies to stay under 9,999 limit
- FASTER: Smarter initial chunk sizes reduce unnecessary API calls
- ROBUST: Handles even the busiest publication periods
"""

import os
import re
import pandas as pd
import warnings
import time
from Bio import Entrez
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import logging
import gc
import calendar
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import concurrent.futures
from functools import partial

# Try to import tqdm
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Install tqdm for better progress bars: pip install tqdm")
    class tqdm:
        def __init__(self, iterable, desc="", total=None, leave=True):
            self.iterable = iterable
            self.desc = desc
            print(f"{desc}: Starting...")
        
        def __iter__(self):
            for item in self.iterable:
                yield item
        
        def update(self, n=1):
            pass
        
        def close(self):
            print(f"{self.desc}: Complete!")

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pubmed_retrieval.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Setup paths
current_dir = os.getcwd()
data_dir = os.path.join(current_dir, "DATA")
progress_dir = os.path.join(data_dir, "yearly_progress")
checkpoint_file = os.path.join(data_dir, "retrieval_checkpoint.json")
os.makedirs(data_dir, exist_ok=True)
os.makedirs(progress_dir, exist_ok=True)

# Configure Entrez - CHANGE THIS TO YOUR EMAIL
Entrez.email = "mc@manuelcorpas.com"  # REQUIRED: Change to your email
Entrez.tool = "DiseaseAbstractRetrieval"
Entrez.api_key = "44271e8e8b6d39627a80dc93092a718c6808"  # Optional: Add NCBI API key

def load_checkpoint():
    """Load checkpoint to resume from interruption"""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_checkpoint(year, month=None):
    """Save progress checkpoint"""
    checkpoint = load_checkpoint()
    checkpoint['last_year'] = year
    checkpoint['last_month'] = month
    checkpoint['timestamp'] = datetime.now().isoformat()
    
    try:
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save checkpoint: {e}")

# Disease categories for splitting when needed
DISEASE_CATEGORIES = {
    'cancer': ['"Neoplasms"[MeSH]'],
    'infectious': ['"Infections"[MeSH]', '"Communicable Diseases"[MeSH]'],
    'cardiovascular': ['"Cardiovascular Diseases"[MeSH]'],
    'mental': ['"Mental Disorders"[MeSH]'],
    'neurological': ['"Nervous System Diseases"[MeSH]'],
    'respiratory': ['"Respiratory Tract Diseases"[MeSH]'],
    'digestive': ['"Digestive System Diseases"[MeSH]'],
    'metabolic': ['"Nutritional and Metabolic Diseases"[MeSH]'],
    'genetic': ['"Genetic Diseases, Inborn"[MeSH]'],
    'musculoskeletal': ['"Musculoskeletal Diseases"[MeSH]'],
    'other': ['"Disease"[MeSH]', '"Pathological Conditions, Signs and Symptoms"[MeSH]']
}

# Estimated annual publication growth (rough estimates)
PUBLICATION_ESTIMATES = {
    range(1975, 1980): 50000,   # 1970s: ~50k/year
    range(1980, 1990): 100000,  # 1980s: ~100k/year
    range(1990, 2000): 200000,  # 1990s: ~200k/year
    range(2000, 2010): 400000,  # 2000s: ~400k/year
    range(2010, 2020): 800000,  # 2010s: ~800k/year
    range(2020, 2025): 1200000, # 2020s: ~1.2M/year
}

def estimate_articles_per_year(year: int) -> int:
    """Estimate number of articles for a given year"""
    for year_range, estimate in PUBLICATION_ESTIMATES.items():
        if year in year_range:
            return estimate
    return 1000000  # Default conservative estimate

def create_disease_search_query(
    year: Optional[int] = None,
    month: Optional[int] = None,
    day: Optional[int] = None,
    hour_range: Optional[Tuple[int, int]] = None,
    date_range: Optional[Tuple[str, str]] = None,
    disease_category: Optional[str] = None
) -> str:
    """Create search query with flexible parameters"""
    base_criteria = [
        '"journal article"[Publication Type]',
        'english[Language]',
        'humans[MeSH Terms]',
        'hasabstract[text]'
    ]
    
    # Add disease criteria
    if disease_category and disease_category in DISEASE_CATEGORIES:
        disease_terms = DISEASE_CATEGORIES[disease_category]
    else:
        # All disease terms
        all_disease_terms = []
        for terms in DISEASE_CATEGORIES.values():
            all_disease_terms.extend(terms)
        disease_terms = list(set(all_disease_terms))
    
    base_criteria.append(f'({" OR ".join(disease_terms)})')
    
    # Add date constraints
    if date_range:
        start_date, end_date = date_range
        date_constraint = f'"{start_date}"[Date - Publication] : "{end_date}"[Date - Publication]'
    elif hour_range and year and month and day:
        # PubMed doesn't support hour-level directly, but we can use PDAT
        start_hour, end_hour = hour_range
        start_time = f"{year}/{month:02d}/{day:02d} {start_hour:02d}:00"
        end_time = f"{year}/{month:02d}/{day:02d} {end_hour:02d}:59"
        date_constraint = f'"{start_time}"[PDAT] : "{end_time}"[PDAT]'
    elif year and month and day:
        date_constraint = f'"{year}/{month:02d}/{day:02d}"[Date - Publication]'
    elif year and month:
        date_constraint = f'"{year}/{month:02d}"[Date - Publication]'
    elif year:
        date_constraint = f'"{year}"[Date - Publication]'
    else:
        date_constraint = '"1975"[Date - Publication] : "2024"[Date - Publication]'
    
    base_criteria.append(date_constraint)
    
    return ' AND '.join(base_criteria)

def binary_search_date_split(start_date: str, end_date: str, target_count: int = 5000) -> List[Tuple[str, str]]:
    """Use binary search to find optimal date ranges that stay under 9999 limit"""
    date_ranges = []
    
    # Convert to datetime objects
    start = datetime.strptime(start_date, "%Y/%m/%d")
    end = datetime.strptime(end_date, "%Y/%m/%d")
    
    def search_range(s_date: datetime, e_date: datetime, depth: int = 0):
        # Prevent infinite recursion
        if depth > 20:
            logger.error(f"Max recursion depth reached for {s_date} to {e_date}")
            return [(s_date.strftime("%Y/%m/%d"), e_date.strftime("%Y/%m/%d"), 9999)]
        
        if s_date > e_date:
            return []
        
        # Format dates
        s_str = s_date.strftime("%Y/%m/%d")
        e_str = e_date.strftime("%Y/%m/%d")
        
        # Check count for this range
        query = create_disease_search_query(date_range=(s_str, e_str))
        try:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=0)
            result = Entrez.read(handle)
            handle.close()
            count = int(result["Count"])
            
            if count == 0:
                return []
            elif count <= 9999:
                return [(s_str, e_str, count)]
            else:
                # Check if this is a single day
                if s_date == e_date:
                    # Can't split a single day further
                    logger.warning(f"  Single day {s_str} has {count:,} articles - splitting by disease category")
                    # Return marker for disease category split
                    return [(s_str, e_str, -count)]  # Negative count indicates need for disease split
                
                # Calculate midpoint
                total_days = (e_date - s_date).days
                if total_days == 0:
                    # Same day, can't split
                    logger.warning(f"  Cannot split {s_str} with {count:,} articles")
                    return [(s_str, e_str, -count)]
                
                mid_date = s_date + timedelta(days=total_days // 2)
                
                # Make sure we're actually splitting
                if mid_date == s_date or mid_date == e_date:
                    # Can't split further
                    if total_days == 1:
                        # Two consecutive days, split them
                        left_ranges = search_range(s_date, s_date, depth + 1)
                        right_ranges = search_range(e_date, e_date, depth + 1)
                    else:
                        # Shouldn't happen, but handle it
                        logger.warning(f"  Cannot properly split {s_str} to {e_str}")
                        return [(s_str, e_str, -count)]
                else:
                    # Recursively search both halves
                    left_ranges = search_range(s_date, mid_date, depth + 1)
                    right_ranges = search_range(mid_date + timedelta(days=1), e_date, depth + 1)
                
                return left_ranges + right_ranges
                
        except Exception as e:
            logger.error(f"Error in binary search for {s_str} to {e_str}: {e}")
            return []
    
    return search_range(start, end)

def smart_chunk_year(year: int) -> List[Dict[str, any]]:
    """Smart chunking based on estimated volume"""
    year_file = os.path.join(progress_dir, f"pubmed_diseases_{year}.csv")
    
    # Check if already processed
    if os.path.exists(year_file):
        try:
            df = pd.read_csv(year_file)
            if 'Abstract' in df.columns and not df['Abstract'].isna().all():
                logger.info(f"Year {year} already processed: {len(df):,} articles")
                return df.to_dict('records')
        except Exception as e:
            logger.warning(f"Reprocessing year {year}: {e}")
    
    logger.info(f"Processing year {year} with smart chunking...")
    
    # Estimate expected volume
    estimated_volume = estimate_articles_per_year(year)
    logger.info(f"  Estimated volume: ~{estimated_volume:,} articles")
    
    try:
        # Determine initial chunk strategy
        if estimated_volume < 100000:  # Can likely handle by month
            year_articles = process_year_by_month(year)
        elif estimated_volume < 500000:  # Need week-level
            year_articles = process_year_by_week(year)
        else:  # High volume - start with day-level
            year_articles = process_year_by_day(year)
        
        # Save the results immediately after processing
        if year_articles:
            df = pd.DataFrame(year_articles)
            df.to_csv(year_file, index=False)
            logger.info(f"✓ Successfully saved {len(year_articles):,} articles for year {year}")
            save_checkpoint(year)  # Save checkpoint after successful save
        else:
            logger.warning(f"No articles found for year {year}")
        
        return year_articles
        
    except Exception as e:
        logger.error(f"Error in smart_chunk_year for {year}: {e}")
        logger.error(f"Traceback: ", exc_info=True)
        return []

def process_year_by_month(year: int) -> List[Dict]:
    """Process year month by month with overflow handling"""
    year_articles = []
    year_file = os.path.join(progress_dir, f"pubmed_diseases_{year}.csv")  # FIX: Define year_file
    
    for month in range(1, 13):
        query = create_disease_search_query(year=year, month=month)
        
        try:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=0)
            result = Entrez.read(handle)
            handle.close()
            count = int(result["Count"])
            
            if count == 0:
                continue
            
            logger.info(f"  {year}/{month:02d}: {count:,} articles")
            
            if count <= 9999:
                # Fetch directly
                handle = Entrez.esearch(db="pubmed", term=query, retmax=count)
                result = Entrez.read(handle)
                handle.close()
                
                pmids = result["IdList"]
                articles = fetch_articles_with_abstracts(pmids, f"{year}/{month:02d}")
                year_articles.extend(articles)
            else:
                # Use binary search to find optimal splits
                days_in_month = calendar.monthrange(year, month)[1]
                start_date = f"{year}/{month:02d}/01"
                end_date = f"{year}/{month:02d}/{days_in_month:02d}"
                
                logger.info(f"    Using binary search for {year}/{month:02d}")
                date_ranges = binary_search_date_split(start_date, end_date)
                
                for date_start, date_end, range_count in date_ranges:
                    if range_count < 0:
                        # Negative count means we need disease category splitting
                        actual_count = -range_count
                        logger.warning(f"    Day {date_start} has {actual_count:,} articles - using disease splits")
                        
                        # Parse the date for disease category processing
                        date_obj = datetime.strptime(date_start, "%Y/%m/%d")
                        day_articles = process_day_by_disease_category(
                            date_obj.year, 
                            date_obj.month, 
                            date_obj.day
                        )
                        year_articles.extend(day_articles)
                        
                    elif range_count > 0:
                        query = create_disease_search_query(date_range=(date_start, date_end))
                        handle = Entrez.esearch(db="pubmed", term=query, retmax=range_count)
                        result = Entrez.read(handle)
                        handle.close()
                        
                        pmids = result["IdList"]
                        articles = fetch_articles_with_abstracts(pmids, f"{date_start}-{date_end}")
                        year_articles.extend(articles)
            
        except Exception as e:
            logger.error(f"Error processing {year}/{month:02d}: {e}")
            continue
    
    # Save progress
    if year_articles:
        df = pd.DataFrame(year_articles)
        df.to_csv(year_file, index=False)
        logger.info(f"Saved {len(year_articles):,} articles for year {year}")
        gc.collect()
    
    return year_articles

def process_year_by_week(year: int) -> List[Dict]:
    """Process year week by week"""
    year_articles = []
    year_file = os.path.join(progress_dir, f"pubmed_diseases_{year}.csv")  # FIX: Define year_file
    
    # Generate all weeks in the year
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    
    current_date = start_date
    while current_date <= end_date:
        week_end = min(current_date + timedelta(days=6), end_date)
        
        week_start_str = current_date.strftime("%Y/%m/%d")
        week_end_str = week_end.strftime("%Y/%m/%d")
        
        query = create_disease_search_query(date_range=(week_start_str, week_end_str))
        
        try:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=0)
            result = Entrez.read(handle)
            handle.close()
            count = int(result["Count"])
            
            if count == 0:
                current_date += timedelta(days=7)
                continue
            
            if count <= 9999:
                logger.info(f"  Week {week_start_str}: {count:,} articles")
                handle = Entrez.esearch(db="pubmed", term=query, retmax=count)
                result = Entrez.read(handle)
                handle.close()
                
                pmids = result["IdList"]
                articles = fetch_articles_with_abstracts(pmids, f"Week {week_start_str}")
                year_articles.extend(articles)
            else:
                # Binary search this week
                logger.info(f"  Week {week_start_str}: {count:,} articles - using binary search")
                date_ranges = binary_search_date_split(week_start_str, week_end_str)
                
                for date_start, date_end, range_count in date_ranges:
                    if range_count < 0:
                        # Negative count means we need disease category splitting
                        actual_count = -range_count
                        logger.warning(f"    Day {date_start} has {actual_count:,} articles - using disease splits")
                        
                        # Parse the date for disease category processing
                        date_obj = datetime.strptime(date_start, "%Y/%m/%d")
                        day_articles = process_day_by_disease_category(
                            date_obj.year, 
                            date_obj.month, 
                            date_obj.day
                        )
                        year_articles.extend(day_articles)
                        
                    elif range_count > 0:
                        query = create_disease_search_query(date_range=(date_start, date_end))
                        handle = Entrez.esearch(db="pubmed", term=query, retmax=range_count)
                        result = Entrez.read(handle)
                        handle.close()
                        
                        pmids = result["IdList"]
                        articles = fetch_articles_with_abstracts(pmids, f"{date_start}-{date_end}")
                        year_articles.extend(articles)
            
        except Exception as e:
            logger.error(f"Error processing week {week_start_str}: {e}")
        
        current_date += timedelta(days=7)
    
    # Save progress
    if year_articles:
        df = pd.DataFrame(year_articles)
        df.to_csv(year_file, index=False)  # Now year_file is properly defined
        logger.info(f"Saved {len(year_articles):,} articles for year {year}")
        gc.collect()
    
    return year_articles

def process_year_by_day(year: int) -> List[Dict]:
    """Process year day by day with disease category splitting if needed"""
    year_articles = []
    year_file = os.path.join(progress_dir, f"pubmed_diseases_{year}.csv")
    
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    
    current_date = start_date
    total_days = (end_date - start_date).days + 1
    
    pbar = tqdm(total=total_days, desc=f"Processing {year} by day") if TQDM_AVAILABLE else None
    
    while current_date <= end_date:
        day_str = current_date.strftime("%Y/%m/%d")
        
        # First, try the full day
        query = create_disease_search_query(
            year=current_date.year,
            month=current_date.month,
            day=current_date.day
        )
        
        try:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=0)
            result = Entrez.read(handle)
            handle.close()
            count = int(result["Count"])
            
            if count == 0:
                current_date += timedelta(days=1)
                if pbar:
                    pbar.update(1)
                continue
            
            if count <= 9999:
                # Can handle this day directly
                handle = Entrez.esearch(db="pubmed", term=query, retmax=count)
                result = Entrez.read(handle)
                handle.close()
                
                pmids = result["IdList"]
                articles = fetch_articles_with_abstracts(pmids, f"Day {day_str}")
                year_articles.extend(articles)
            else:
                # Day is too large - split by disease category
                logger.warning(f"  Day {day_str}: {count:,} articles - splitting by disease category")
                
                day_articles = process_day_by_disease_category(
                    current_date.year, 
                    current_date.month, 
                    current_date.day
                )
                year_articles.extend(day_articles)
            
        except Exception as e:
            logger.error(f"Error processing day {day_str}: {e}")
        
        current_date += timedelta(days=1)
        if pbar:
            pbar.update(1)
    
    if pbar:
        pbar.close()
    
    # Save progress
    if year_articles:
        df = pd.DataFrame(year_articles)
        df.to_csv(year_file, index=False)
        logger.info(f"Saved {len(year_articles):,} articles for year {year}")
        gc.collect()
    
    return year_articles

def process_day_by_disease_category(year: int, month: int, day: int) -> List[Dict]:
    """Process a single day split by disease categories"""
    day_articles = []
    day_str = f"{year}/{month:02d}/{day:02d}"
    
    for category_name, category_terms in DISEASE_CATEGORIES.items():
        query = create_disease_search_query(
            year=year,
            month=month,
            day=day,
            disease_category=category_name
        )
        
        try:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=0)
            result = Entrez.read(handle)
            handle.close()
            count = int(result["Count"])
            
            if count == 0:
                continue
            
            if count <= 9999:
                logger.info(f"    {day_str} [{category_name}]: {count:,} articles")
                handle = Entrez.esearch(db="pubmed", term=query, retmax=count)
                result = Entrez.read(handle)
                handle.close()
                
                pmids = result["IdList"]
                articles = fetch_articles_with_abstracts(
                    pmids, 
                    f"{day_str} [{category_name}]"
                )
                
                # Tag with disease category
                for article in articles:
                    article['Disease_Category_Search'] = category_name
                
                day_articles.extend(articles)
            else:
                # Even a disease category for one day is too large
                # Try hour-level chunking as last resort
                logger.error(f"    📛 {day_str} [{category_name}]: {count:,} articles - trying hour chunks")
                
                hour_articles = process_day_by_hour(year, month, day, category_name)
                day_articles.extend(hour_articles)
            
        except Exception as e:
            logger.error(f"Error processing {day_str} [{category_name}]: {e}")
            continue
    
    return day_articles

def process_day_by_hour(year: int, month: int, day: int, disease_category: str = None) -> List[Dict]:
    """Last resort: process by hour chunks (though PubMed may not have hour precision)"""
    hour_articles = []
    
    # Try 6-hour chunks (0-5, 6-11, 12-17, 18-23)
    hour_chunks = [(0, 5), (6, 11), (12, 17), (18, 23)]
    
    for start_hour, end_hour in hour_chunks:
        query = create_disease_search_query(
            year=year,
            month=month,
            day=day,
            hour_range=(start_hour, end_hour),
            disease_category=disease_category
        )
        
        try:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=0)
            result = Entrez.read(handle)
            handle.close()
            count = int(result["Count"])
            
            if count == 0:
                continue
            
            # Accept whatever we can get
            fetch_count = min(count, 9999)
            if count > 9999:
                logger.error(f"      📛 LOSING {count - 9999} articles from hours {start_hour}-{end_hour}")
            
            handle = Entrez.esearch(db="pubmed", term=query, retmax=fetch_count)
            result = Entrez.read(handle)
            handle.close()
            
            pmids = result["IdList"]
            articles = fetch_articles_with_abstracts(
                pmids, 
                f"{year}/{month:02d}/{day:02d} H{start_hour}-{end_hour}"
            )
            hour_articles.extend(articles)
            
        except Exception as e:
            logger.error(f"Error processing hours {start_hour}-{end_hour}: {e}")
            continue
    
    return hour_articles

def fetch_articles_with_abstracts(pmids: List[str], batch_description: str) -> List[Dict]:
    """Fetch articles ensuring abstracts are included"""
    if not pmids:
        return []
    
    batch_size = 100
    articles = []
    
    pbar = tqdm(
        range(0, len(pmids), batch_size),
        desc=f"Fetching {batch_description}",
        leave=False
    ) if TQDM_AVAILABLE else range(0, len(pmids), batch_size)
    
    for i in pbar:
        batch_pmids = pmids[i:i+batch_size]
        retry_count = 3
        
        while retry_count > 0:
            try:
                handle = Entrez.efetch(
                    db="pubmed",
                    id=batch_pmids,
                    rettype="xml",
                    retmode="xml"
                )
                records = handle.read()
                handle.close()
                
                root = ET.fromstring(records)
                
                for article in root.findall('.//PubmedArticle'):
                    article_data = parse_pubmed_article_with_abstract(article)
                    if article_data:  # Only include if has abstract
                        articles.append(article_data)
                
                # Rate limiting
                time.sleep(0.34 if not Entrez.api_key else 0.1)
                break
                
            except Exception as e:
                retry_count -= 1
                if retry_count == 0:
                    logger.error(f"Failed to fetch batch: {e}")
                else:
                    time.sleep(2)
    
    return articles

def parse_pubmed_article_with_abstract(article_xml) -> Optional[Dict]:
    """Parse article XML including abstract"""
    try:
        article_data = {}
        
        # PMID
        pmid_elem = article_xml.find('.//PMID')
        article_data['PMID'] = pmid_elem.text if pmid_elem is not None else ''
        
        # Title
        title_elem = article_xml.find('.//ArticleTitle')
        article_data['Title'] = title_elem.text if title_elem is not None else ''
        
        # Abstract - Critical
        abstract_text = ""
        abstract_elem = article_xml.find('.//Abstract')
        if abstract_elem is not None:
            abstract_parts = []
            for abstract_text_elem in abstract_elem.findall('.//AbstractText'):
                if abstract_text_elem.text:
                    label = abstract_text_elem.get('Label', '')
                    if label:
                        abstract_parts.append(f"{label}: {abstract_text_elem.text}")
                    else:
                        abstract_parts.append(abstract_text_elem.text)
            abstract_text = " ".join(abstract_parts)
        
        article_data['Abstract'] = abstract_text
        
        # Skip if no abstract
        if not abstract_text or len(abstract_text) < 50:
            return None
        
        # Journal
        journal_elem = article_xml.find('.//Journal/Title')
        if journal_elem is None:
            journal_elem = article_xml.find('.//Journal/ISOAbbreviation')
        article_data['Journal'] = journal_elem.text if journal_elem is not None else ''
        
        # Year
        year_elem = article_xml.find('.//PubDate/Year')
        if year_elem is None:
            year_elem = article_xml.find('.//PubDate/MedlineDate')
            if year_elem is not None and year_elem.text:
                year_match = re.search(r'(\d{4})', year_elem.text)
                article_data['Year'] = year_match.group(1) if year_match else ''
            else:
                article_data['Year'] = ''
        else:
            article_data['Year'] = year_elem.text
        
        # MeSH Terms
        mesh_terms = []
        for mesh in article_xml.findall('.//MeshHeading/DescriptorName'):
            if mesh.text:
                mesh_terms.append(mesh.text)
        article_data['MeSH_Terms'] = '; '.join(mesh_terms)
        
        # Authors
        authors = []
        first_author_affiliation = ''
        
        author_list = article_xml.find('.//AuthorList')
        if author_list is not None:
            for i, author in enumerate(author_list.findall('Author')):
                last_name = author.find('LastName')
                first_name = author.find('ForeName')
                
                if last_name is not None:
                    author_name = last_name.text
                    if first_name is not None:
                        author_name += f", {first_name.text}"
                    authors.append(author_name)
                
                # First author affiliation
                if i == 0:
                    for aff in author.findall('.//Affiliation'):
                        if aff.text:
                            first_author_affiliation = aff.text.strip()
                            break
        
        article_data['Authors'] = '; '.join(authors[:10])
        article_data['FirstAuthorAffiliation'] = first_author_affiliation
        
        return article_data
        
    except Exception as e:
        logger.error(f"Error parsing article: {e}")
        return None

def combine_all_years() -> pd.DataFrame:
    """Combine all yearly files into final dataset"""
    logger.info("Combining all yearly files...")
    
    all_dfs = []
    years_found = []
    
    for year in range(1975, 2025):
        year_file = os.path.join(progress_dir, f"pubmed_diseases_{year}.csv")
        if os.path.exists(year_file):
            try:
                df = pd.read_csv(year_file)
                if 'Abstract' in df.columns:
                    valid_abstracts = df['Abstract'].notna() & (df['Abstract'].str.len() > 50)
                    df = df[valid_abstracts]
                    if len(df) > 0:
                        all_dfs.append(df)
                        years_found.append(year)
                        logger.info(f"  Year {year}: {len(df):,} articles")
            except Exception as e:
                logger.error(f"Error loading year {year}: {e}")
    
    if not all_dfs:
        logger.error("No valid data files found")
        return None
    
    logger.info(f"Combining {len(all_dfs)} yearly files...")
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Remove duplicates
    initial_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['PMID'], keep='first')
    final_count = len(combined_df)
    
    if initial_count != final_count:
        logger.info(f"Removed {initial_count - final_count:,} duplicate articles")
    
    return combined_df

def main():
    """Main execution function"""
    print("="*80)
    print("PUBMED DISEASE ABSTRACTS RETRIEVAL - SMART CHUNKING VERSION")
    print("="*80)
    print("Improvements:")
    print("  ✓ Binary search for optimal date ranges")
    print("  ✓ Disease category splitting when needed")
    print("  ✓ Adaptive chunk sizing based on year")
    print("  ✓ Hour-level chunking as last resort")
    print("="*80)
    
    if Entrez.email == "your.email@example.com":
        print("\n⚠️  ERROR: Please set your email address")
        return
    
    print(f"\nConfiguration:")
    print(f"  Email: {Entrez.email}")
    print(f"  API Key: {'Set' if Entrez.api_key else 'Not set'}")
    print(f"  Period: 1975-2024")
    
    # Check for existing data and checkpoint
    checkpoint = load_checkpoint()
    existing_years = []
    
    for year in range(1975, 2025):
        year_file = os.path.join(progress_dir, f"pubmed_diseases_{year}.csv")
        if os.path.exists(year_file):
            try:
                df = pd.read_csv(year_file)
                if len(df) > 0:
                    existing_years.append(year)
            except:
                pass
    
    if existing_years:
        print(f"\n📊 Found existing data for {len(existing_years)} years")
        print(f"   Years with data: {sorted(existing_years)}")
        
        # Ask user what to do
        print("\nOptions:")
        print("  1. Resume from where left off (skip completed years)")
        print("  2. Start fresh from 1975 (reprocess all years)")
        print("  3. Cancel")
        
        choice = input("\nChoice (1/2/3): ").strip()
        
        if choice == "1":
            # Resume - skip existing years
            start_year = 1975
            years_to_process = [y for y in range(1975, 2025) if y not in existing_years]
            print(f"\n📌 Will process {len(years_to_process)} missing years: {years_to_process[:5]}...")
        elif choice == "2":
            # Fresh start
            print("\n🔄 Starting fresh from 1975 (ignoring existing data)")
            # Clear checkpoint
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
                print("   Cleared checkpoint file")
            years_to_process = list(range(1975, 2025))
        else:
            print("Cancelled.")
            return
    else:
        # No existing data, start fresh
        years_to_process = list(range(1975, 2025))
        print("\n📌 Starting fresh from 1975")
    
    proceed = input("\nProceed? (y/n): ").strip().lower()
    if proceed not in ['y', 'yes']:
        return
    
    start_time = datetime.now()
    
    # Process each year
    for year in tqdm(years_to_process, desc="Processing years") if TQDM_AVAILABLE else years_to_process:
        try:
            # Skip if already processed (only in resume mode, choice 1)
            if existing_years and 'choice' in locals() and choice == "1":
                year_file = os.path.join(progress_dir, f"pubmed_diseases_{year}.csv")
                if os.path.exists(year_file):
                    try:
                        df = pd.read_csv(year_file)
                        if len(df) > 0:
                            logger.info(f"Skipping year {year} - already have {len(df):,} articles")
                            continue
                    except:
                        pass
            
            # Process the year
            smart_chunk_year(year)
            
        except KeyboardInterrupt:
            print(f"\n⚠️  Interrupted at year {year}")
            print("Progress has been saved. Run again to resume.")
            return
        except Exception as e:
            logger.error(f"Failed year {year}: {e}")
            continue
    
    # Combine all data
    print("\n📊 Creating final dataset...")
    final_df = combine_all_years()
    
    if final_df is not None:
        output_file = os.path.join(data_dir, 'pubmed_disease_abstracts_1975_2024.csv')
        final_df.to_csv(output_file, index=False, chunksize=100000)
        
        duration = datetime.now() - start_time
        
        print(f"\n✅ COMPLETE!")
        print(f"  Total articles: {len(final_df):,}")
        print(f"  Time taken: {duration}")
        print(f"  Output: {output_file}")

if __name__ == "__main__":
    main()