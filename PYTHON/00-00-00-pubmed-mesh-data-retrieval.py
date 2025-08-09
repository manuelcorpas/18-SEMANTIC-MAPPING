"""
PUBMED CONDITION-SPECIFIC ABSTRACTS RETRIEVAL (1975-2024)

Retrieves disease-related research articles by specific conditions with intelligent chunking:
- Searches 200+ individual disease conditions across 50 years
- Implements minute-level chunking when needed (Year → Month → Week → Day → Hour → Minute)
- Tracks completeness for each condition-year combination
- Saves data organized by condition
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
        def __init__(self, iterable=None, desc="", total=None, leave=True):
            self.iterable = iterable
            self.desc = desc
            self.total = total
            print(f"{desc}: Starting...")
        
        def __iter__(self):
            if self.iterable:
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
        logging.FileHandler('pubmed_condition_retrieval.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Setup paths
current_dir = os.getcwd()
data_dir = os.path.join(current_dir, "CONDITION_DATA")
progress_dir = os.path.join(data_dir, "condition_progress")
checkpoint_file = os.path.join(data_dir, "condition_retrieval_checkpoint.json")
completeness_file = os.path.join(data_dir, "retrieval_completeness.json")
os.makedirs(data_dir, exist_ok=True)
os.makedirs(progress_dir, exist_ok=True)

# Configure Entrez - CHANGE THIS TO YOUR EMAIL
Entrez.email = "mc@manuelcorpas.com"  # REQUIRED: Change to your email
Entrez.tool = "ConditionAbstractRetrieval"
Entrez.api_key = "44271e8e8b6d39627a80dc93092a718c6808"  # Optional: Add NCBI API key

# Complete list of conditions to search
CONDITIONS = [
    "Breast cancer", "Cervical cancer", "Uterine cancer", "Prostate cancer", "Colon and rectum cancer",
    "Lip and oral cavity cancer", "Nasopharynx cancer", "Other pharynx cancer", "Gallbladder and biliary tract cancer",
    "Pancreatic cancer", "Malignant skin melanoma", "Non-melanoma skin cancer", "Ovarian cancer", "Testicular cancer",
    "Kidney cancer", "Bladder cancer", "Brain and central nervous system cancer", "Thyroid cancer", "Mesothelioma",
    "Hodgkin lymphoma", "Non-Hodgkin lymphoma", "Multiple myeloma", "Leukemia", "Other malignant neoplasms",
    "Other neoplasms", "Rheumatic heart disease", "Endocrine metabolic blood and immune disorders",
    "Rheumatoid arthritis", "Other musculoskeletal disorders", "Congenital birth defects", "Typhoid and paratyphoid",
    "Invasive Non-typhoidal Salmonella (iNTS)", "Bacterial skin diseases", "Upper digestive system diseases",
    "Pulmonary Arterial Hypertension", "Malaria", "Chagas disease", "Leishmaniasis", "African trypanosomiasis",
    "Schistosomiasis", "Cysticercosis", "Cystic echinococcosis", "Dengue", "Yellow fever", "Rabies",
    "Intestinal nematode infections", "Other neglected tropical diseases", "Maternal disorders", "Neonatal disorders",
    "Tuberculosis", "HIV/AIDS", "Diarrheal diseases", "Other intestinal infectious diseases", "Lower respiratory infections",
    "Upper respiratory infections", "Otitis media", "Meningitis", "Encephalitis", "Diphtheria", "Pertussis", "Tetanus",
    "Protein-energy malnutrition", "Appendicitis", "Paralytic ileus and intestinal obstruction",
    "Inguinal femoral and abdominal hernia", "Inflammatory bowel disease", "Vascular intestinal disorders",
    "Gallbladder and biliary diseases", "Pancreatitis", "Measles", "Varicella and herpes zoster",
    "Ischemic heart disease", "Stroke", "Hypertensive heart disease", "Cardiomyopathy and myocarditis",
    "Atrial fibrillation and flutter", "Aortic aneurysm", "Lower extremity peripheral arterial disease",
    "Endocarditis", "Non-rheumatic valvular heart disease", "Other cardiovascular and circulatory diseases",
    "Chronic obstructive pulmonary disease", "Other digestive diseases", "Alzheimer's disease and other dementias",
    "Parkinson's disease", "Idiopathic epilepsy", "Multiple sclerosis", "Motor neuron disease",
    "Other neurological disorders", "Adverse effects of medical treatment", "Animal contact", "Foreign body",
    "Pneumoconiosis", "Asthma", "Interstitial lung disease and pulmonary sarcoidosis",
    "Other chronic respiratory diseases", "Cirrhosis and other chronic liver diseases", "Decubitus ulcer",
    "Other skin and subcutaneous diseases", "Sudden infant death syndrome", "Road injuries", "Self-harm",
    "Interpersonal violence", "Exposure to forces of nature", "Environmental heat and cold exposure", "Ebola",
    "Other transport injuries", "Falls", "Drowning", "Fire heat and hot substances", "Poisonings",
    "Exposure to mechanical forces", "Eye cancer", "Soft tissue and other extraosseous sarcomas",
    "Malignant neoplasm of bone and articular cartilage", "Other nutritional deficiencies",
    "Sexually transmitted infections excluding HIV", "Acute hepatitis", "Other unspecified infectious diseases",
    "Esophageal cancer", "Stomach cancer", "Liver cancer", "Larynx cancer", "Tracheal bronchus and lung cancer",
    "Alcohol use disorders", "Drug use disorders", "Eating disorders", "Diabetes mellitus", "Acute glomerulonephritis",
    "Chronic kidney disease", "Urinary diseases and male infertility", "Gynecological diseases",
    "Hemoglobinopathies and hemolytic anemias", "Police conflict and executions", "Zika virus",
    "Conflict and terrorism", "Neuroblastoma and other peripheral nervous cell tumors", "COVID-19",
    "Other unintentional injuries", "Leprosy", "Schizophrenia", "Depressive disorders", "Acne vulgaris",
    "Alopecia areata", "Pruritus", "Urticaria", "Age-related and other hearing loss", "Other sense organ diseases",
    "Oral disorders", "Headache disorders", "Bipolar disorder", "Anxiety disorders", "Autism spectrum disorders",
    "Attention-deficit/hyperactivity disorder", "Conduct disorder", "Idiopathic developmental intellectual disability",
    "Other mental disorders", "Blindness and vision loss", "Lymphatic filariasis", "Onchocerciasis", "Trachoma",
    "Food-borne trematodiases", "Iodine deficiency", "Vitamin A deficiency", "Dietary iron deficiency",
    "Osteoarthritis", "Low back pain", "Neck pain", "Gout", "Dermatitis", "Psoriasis", "Scabies",
    "Fungal skin diseases", "Viral skin diseases", "Guinea worm disease"
]

# Mapping of conditions to MeSH terms and search strategies
CONDITION_SEARCH_MAPPING = {
    # Cancers
    "Breast cancer": '"Breast Neoplasms"[MeSH] OR "breast cancer"[Title/Abstract]',
    "Cervical cancer": '"Uterine Cervical Neoplasms"[MeSH] OR "cervical cancer"[Title/Abstract]',
    "Uterine cancer": '"Uterine Neoplasms"[MeSH] OR "uterine cancer"[Title/Abstract] OR "endometrial cancer"[Title/Abstract]',
    "Prostate cancer": '"Prostatic Neoplasms"[MeSH] OR "prostate cancer"[Title/Abstract]',
    "Colon and rectum cancer": '"Colorectal Neoplasms"[MeSH] OR "colorectal cancer"[Title/Abstract] OR "colon cancer"[Title/Abstract] OR "rectal cancer"[Title/Abstract]',
    "Lip and oral cavity cancer": '"Mouth Neoplasms"[MeSH] OR "oral cancer"[Title/Abstract] OR "lip cancer"[Title/Abstract]',
    "Nasopharynx cancer": '"Nasopharyngeal Neoplasms"[MeSH] OR "nasopharyngeal cancer"[Title/Abstract]',
    "Other pharynx cancer": '"Pharyngeal Neoplasms"[MeSH] OR "pharyngeal cancer"[Title/Abstract]',
    "Gallbladder and biliary tract cancer": '"Gallbladder Neoplasms"[MeSH] OR "Biliary Tract Neoplasms"[MeSH] OR "gallbladder cancer"[Title/Abstract]',
    "Pancreatic cancer": '"Pancreatic Neoplasms"[MeSH] OR "pancreatic cancer"[Title/Abstract]',
    "Malignant skin melanoma": '"Melanoma"[MeSH] OR "malignant melanoma"[Title/Abstract]',
    "Non-melanoma skin cancer": '"Skin Neoplasms"[MeSH] NOT "Melanoma"[MeSH] OR "basal cell carcinoma"[Title/Abstract] OR "squamous cell carcinoma"[Title/Abstract]',
    "Ovarian cancer": '"Ovarian Neoplasms"[MeSH] OR "ovarian cancer"[Title/Abstract]',
    "Testicular cancer": '"Testicular Neoplasms"[MeSH] OR "testicular cancer"[Title/Abstract]',
    "Kidney cancer": '"Kidney Neoplasms"[MeSH] OR "renal cell carcinoma"[Title/Abstract]',
    "Bladder cancer": '"Urinary Bladder Neoplasms"[MeSH] OR "bladder cancer"[Title/Abstract]',
    "Brain and central nervous system cancer": '"Brain Neoplasms"[MeSH] OR "Central Nervous System Neoplasms"[MeSH] OR "brain cancer"[Title/Abstract]',
    "Thyroid cancer": '"Thyroid Neoplasms"[MeSH] OR "thyroid cancer"[Title/Abstract]',
    "Mesothelioma": '"Mesothelioma"[MeSH] OR "mesothelioma"[Title/Abstract]',
    "Hodgkin lymphoma": '"Hodgkin Disease"[MeSH] OR "hodgkin lymphoma"[Title/Abstract]',
    "Non-Hodgkin lymphoma": '"Lymphoma, Non-Hodgkin"[MeSH] OR "non-hodgkin lymphoma"[Title/Abstract]',
    "Multiple myeloma": '"Multiple Myeloma"[MeSH] OR "multiple myeloma"[Title/Abstract]',
    "Leukemia": '"Leukemia"[MeSH] OR "leukemia"[Title/Abstract] OR "leukaemia"[Title/Abstract]',
    "Esophageal cancer": '"Esophageal Neoplasms"[MeSH] OR "esophageal cancer"[Title/Abstract]',
    "Stomach cancer": '"Stomach Neoplasms"[MeSH] OR "gastric cancer"[Title/Abstract]',
    "Liver cancer": '"Liver Neoplasms"[MeSH] OR "hepatocellular carcinoma"[Title/Abstract]',
    "Larynx cancer": '"Laryngeal Neoplasms"[MeSH] OR "laryngeal cancer"[Title/Abstract]',
    "Tracheal bronchus and lung cancer": '"Lung Neoplasms"[MeSH] OR "lung cancer"[Title/Abstract]',
    
    # Infectious diseases
    "Tuberculosis": '"Tuberculosis"[MeSH] OR "tuberculosis"[Title/Abstract]',
    "HIV/AIDS": '"HIV Infections"[MeSH] OR "Acquired Immunodeficiency Syndrome"[MeSH] OR "HIV"[Title/Abstract] OR "AIDS"[Title/Abstract]',
    "Malaria": '"Malaria"[MeSH] OR "malaria"[Title/Abstract]',
    "Dengue": '"Dengue"[MeSH] OR "dengue"[Title/Abstract]',
    "COVID-19": '"COVID-19"[MeSH] OR "SARS-CoV-2"[MeSH] OR "covid-19"[Title/Abstract] OR "coronavirus disease 2019"[Title/Abstract]',
    "Ebola": '"Hemorrhagic Fever, Ebola"[MeSH] OR "ebola"[Title/Abstract]',
    "Measles": '"Measles"[MeSH] OR "measles"[Title/Abstract]',
    "Hepatitis": '"Hepatitis"[MeSH] OR "hepatitis"[Title/Abstract]',
    
    # Cardiovascular diseases
    "Ischemic heart disease": '"Myocardial Ischemia"[MeSH] OR "ischemic heart disease"[Title/Abstract]',
    "Stroke": '"Stroke"[MeSH] OR "stroke"[Title/Abstract] OR "cerebrovascular accident"[Title/Abstract]',
    "Hypertensive heart disease": '"Hypertensive Heart Disease"[MeSH] OR "hypertensive heart disease"[Title/Abstract]',
    "Atrial fibrillation and flutter": '"Atrial Fibrillation"[MeSH] OR "Atrial Flutter"[MeSH] OR "atrial fibrillation"[Title/Abstract]',
    
    # Neurological conditions
    "Alzheimer's disease and other dementias": '"Alzheimer Disease"[MeSH] OR "Dementia"[MeSH] OR "alzheimer"[Title/Abstract]',
    "Parkinson's disease": '"Parkinson Disease"[MeSH] OR "parkinson"[Title/Abstract]',
    "Multiple sclerosis": '"Multiple Sclerosis"[MeSH] OR "multiple sclerosis"[Title/Abstract]',
    "Idiopathic epilepsy": '"Epilepsy"[MeSH] OR "epilepsy"[Title/Abstract]',
    
    # Mental health
    "Schizophrenia": '"Schizophrenia"[MeSH] OR "schizophrenia"[Title/Abstract]',
    "Depressive disorders": '"Depressive Disorder"[MeSH] OR "depression"[Title/Abstract]',
    "Bipolar disorder": '"Bipolar Disorder"[MeSH] OR "bipolar"[Title/Abstract]',
    "Anxiety disorders": '"Anxiety Disorders"[MeSH] OR "anxiety disorder"[Title/Abstract]',
    "Autism spectrum disorders": '"Autism Spectrum Disorder"[MeSH] OR "autism"[Title/Abstract]',
    "Attention-deficit/hyperactivity disorder": '"Attention Deficit Disorder with Hyperactivity"[MeSH] OR "ADHD"[Title/Abstract]',
    
    # Respiratory diseases
    "Chronic obstructive pulmonary disease": '"Pulmonary Disease, Chronic Obstructive"[MeSH] OR "COPD"[Title/Abstract]',
    "Asthma": '"Asthma"[MeSH] OR "asthma"[Title/Abstract]',
    "Pneumoconiosis": '"Pneumoconiosis"[MeSH] OR "pneumoconiosis"[Title/Abstract]',
    
    # Metabolic and endocrine
    "Diabetes mellitus": '"Diabetes Mellitus"[MeSH] OR "diabetes"[Title/Abstract]',
    "Chronic kidney disease": '"Renal Insufficiency, Chronic"[MeSH] OR "chronic kidney disease"[Title/Abstract]',
    
    # Add remaining conditions with appropriate search terms...
    # This is a partial mapping - you'll need to complete it for all 200+ conditions
}

def get_condition_search_query(condition: str) -> str:
    """Get the appropriate search query for a condition"""
    # Use mapping if available, otherwise create a basic search
    if condition in CONDITION_SEARCH_MAPPING:
        return CONDITION_SEARCH_MAPPING[condition]
    else:
        # Fallback: use the condition name as a search term
        # Clean the condition name for search
        clean_name = condition.lower().replace(" and ", " ").replace("/", " ")
        return f'"{condition}"[Title/Abstract] OR "{clean_name}"[Title/Abstract]'

def create_condition_search_query(
    condition: str,
    year: Optional[int] = None,
    month: Optional[int] = None,
    day: Optional[int] = None,
    hour_range: Optional[Tuple[int, int]] = None,
    minute_range: Optional[Tuple[int, int]] = None,
    date_range: Optional[Tuple[str, str]] = None
) -> str:
    """Create search query for a specific condition with temporal constraints"""
    base_criteria = [
        '"journal article"[Publication Type]',
        'english[Language]',
        'humans[MeSH Terms]',
        'hasabstract[text]'
    ]
    
    # Add condition-specific search terms
    condition_query = get_condition_search_query(condition)
    base_criteria.append(f'({condition_query})')
    
    # Add date constraints
    if date_range:
        start_date, end_date = date_range
        date_constraint = f'"{start_date}"[Date - Publication] : "{end_date}"[Date - Publication]'
    elif minute_range and year and month and day:
        # Minute-level precision (if supported by PubMed)
        start_min, end_min = minute_range
        if hour_range:
            start_hour, end_hour = hour_range
        else:
            start_hour = end_hour = 0
        start_time = f"{year}/{month:02d}/{day:02d} {start_hour:02d}:{start_min:02d}"
        end_time = f"{year}/{month:02d}/{day:02d} {end_hour:02d}:{end_min:02d}"
        date_constraint = f'"{start_time}"[PDAT] : "{end_time}"[PDAT]'
    elif hour_range and year and month and day:
        # Hour-level precision
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

def load_checkpoint():
    """Load checkpoint to resume from interruption"""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_checkpoint(condition: str, year: int, status: str = "processing"):
    """Save progress checkpoint"""
    checkpoint = load_checkpoint()
    if condition not in checkpoint:
        checkpoint[condition] = {}
    checkpoint[condition][str(year)] = {
        'status': status,
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save checkpoint: {e}")

def load_completeness_tracking():
    """Load completeness tracking data"""
    if os.path.exists(completeness_file):
        try:
            with open(completeness_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_completeness_tracking(condition: str, year: int, total_found: int, total_retrieved: int):
    """Save completeness tracking information"""
    tracking = load_completeness_tracking()
    if condition not in tracking:
        tracking[condition] = {}
    tracking[condition][str(year)] = {
        'total_found': total_found,
        'total_retrieved': total_retrieved,
        'completeness': (total_retrieved / total_found * 100) if total_found > 0 else 100,
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        with open(completeness_file, 'w') as f:
            json.dump(tracking, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save completeness tracking: {e}")

def process_condition_year(condition: str, year: int) -> Tuple[List[Dict], int, int]:
    """Process a specific condition for a specific year"""
    condition_dir = os.path.join(progress_dir, condition.replace("/", "_").replace(" ", "_"))
    os.makedirs(condition_dir, exist_ok=True)
    
    year_file = os.path.join(condition_dir, f"{condition.replace('/', '_').replace(' ', '_')}_{year}.csv")
    
    # Check if already processed
    checkpoint = load_checkpoint()
    if condition in checkpoint and str(year) in checkpoint[condition]:
        if checkpoint[condition][str(year)]['status'] == 'complete':
            try:
                df = pd.read_csv(year_file)
                if 'Abstract' in df.columns and not df['Abstract'].isna().all():
                    logger.info(f"  {condition} - {year}: Already processed ({len(df):,} articles)")
                    return df.to_dict('records'), len(df), len(df)
            except:
                pass
    
    logger.info(f"Processing {condition} - Year {year}")
    save_checkpoint(condition, year, "processing")
    
    # First, check total count for this condition-year
    query = create_condition_search_query(condition=condition, year=year)
    
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=0)
        result = Entrez.read(handle)
        handle.close()
        total_count = int(result["Count"])
        
        if total_count == 0:
            logger.info(f"  No articles found for {condition} in {year}")
            save_checkpoint(condition, year, "complete")
            save_completeness_tracking(condition, year, 0, 0)
            return [], 0, 0
        
        logger.info(f"  Found {total_count:,} articles for {condition} in {year}")
        
        if total_count <= 9999:
            # Can retrieve all at once
            articles = retrieve_articles_batch(condition, year, query, total_count)
            retrieved_count = len(articles)
        else:
            # Need chunking
            articles, retrieved_count = process_with_chunking(condition, year, total_count)
        
        # Save results
        if articles:
            df = pd.DataFrame(articles)
            df['Condition'] = condition  # Add condition column
            df.to_csv(year_file, index=False)
            logger.info(f"  Saved {len(articles):,} articles for {condition} - {year}")
        
        save_checkpoint(condition, year, "complete")
        save_completeness_tracking(condition, year, total_count, retrieved_count)
        
        return articles, total_count, retrieved_count
        
    except Exception as e:
        logger.error(f"Error processing {condition} - {year}: {e}")
        save_checkpoint(condition, year, "error")
        return [], 0, 0

def process_with_chunking(condition: str, year: int, total_count: int) -> Tuple[List[Dict], int]:
    """Process a condition-year with intelligent chunking"""
    logger.info(f"  Chunking required for {condition} - {year} ({total_count:,} articles)")
    
    all_articles = []
    retrieved_count = 0
    
    # Try monthly chunks first
    for month in range(1, 13):
        query = create_condition_search_query(condition=condition, year=year, month=month)
        
        try:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=0)
            result = Entrez.read(handle)
            handle.close()
            month_count = int(result["Count"])
            
            if month_count == 0:
                continue
            
            if month_count <= 9999:
                # Month is manageable
                articles = retrieve_articles_batch(condition, year, query, month_count, f"{year}/{month:02d}")
                all_articles.extend(articles)
                retrieved_count += len(articles)
            else:
                # Need weekly or daily chunking
                logger.info(f"    Month {year}/{month:02d} has {month_count:,} articles - chunking further")
                month_articles = process_month_with_chunking(condition, year, month, month_count)
                all_articles.extend(month_articles)
                retrieved_count += len(month_articles)
                
        except Exception as e:
            logger.error(f"Error processing month {year}/{month:02d}: {e}")
            continue
    
    return all_articles, retrieved_count

def process_month_with_chunking(condition: str, year: int, month: int, total_count: int) -> List[Dict]:
    """Process a month with weekly/daily chunking"""
    month_articles = []
    days_in_month = calendar.monthrange(year, month)[1]
    
    # Try weekly chunks
    week_starts = list(range(1, days_in_month + 1, 7))
    
    for week_start in week_starts:
        week_end = min(week_start + 6, days_in_month)
        start_date = f"{year}/{month:02d}/{week_start:02d}"
        end_date = f"{year}/{month:02d}/{week_end:02d}"
        
        query = create_condition_search_query(condition=condition, date_range=(start_date, end_date))
        
        try:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=0)
            result = Entrez.read(handle)
            handle.close()
            week_count = int(result["Count"])
            
            if week_count == 0:
                continue
            
            if week_count <= 9999:
                articles = retrieve_articles_batch(condition, year, query, week_count, f"Week {start_date}")
                month_articles.extend(articles)
            else:
                # Need daily chunking
                logger.info(f"      Week {start_date} has {week_count:,} articles - using daily chunks")
                week_articles = process_week_with_chunking(condition, year, month, week_start, week_end)
                month_articles.extend(week_articles)
                
        except Exception as e:
            logger.error(f"Error processing week {start_date}: {e}")
            continue
    
    return month_articles

def process_week_with_chunking(condition: str, year: int, month: int, day_start: int, day_end: int) -> List[Dict]:
    """Process a week with daily/hourly chunking"""
    week_articles = []
    
    for day in range(day_start, day_end + 1):
        query = create_condition_search_query(condition=condition, year=year, month=month, day=day)
        
        try:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=0)
            result = Entrez.read(handle)
            handle.close()
            day_count = int(result["Count"])
            
            if day_count == 0:
                continue
            
            if day_count <= 9999:
                articles = retrieve_articles_batch(condition, year, query, day_count, f"{year}/{month:02d}/{day:02d}")
                week_articles.extend(articles)
            else:
                # Need hourly chunking
                logger.info(f"        Day {year}/{month:02d}/{day:02d} has {day_count:,} articles - using hourly chunks")
                day_articles = process_day_with_chunking(condition, year, month, day)
                week_articles.extend(day_articles)
                
        except Exception as e:
            logger.error(f"Error processing day {year}/{month:02d}/{day:02d}: {e}")
            continue
    
    return week_articles

def process_day_with_chunking(condition: str, year: int, month: int, day: int) -> List[Dict]:
    """Process a day with hourly/minute chunking"""
    day_articles = []
    
    # Try 6-hour chunks
    hour_chunks = [(0, 5), (6, 11), (12, 17), (18, 23)]
    
    for start_hour, end_hour in hour_chunks:
        query = create_condition_search_query(
            condition=condition,
            year=year,
            month=month,
            day=day,
            hour_range=(start_hour, end_hour)
        )
        
        try:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=0)
            result = Entrez.read(handle)
            handle.close()
            hour_count = int(result["Count"])
            
            if hour_count == 0:
                continue
            
            if hour_count <= 9999:
                articles = retrieve_articles_batch(
                    condition, year, query, hour_count, 
                    f"{year}/{month:02d}/{day:02d} H{start_hour}-{end_hour}"
                )
                day_articles.extend(articles)
            else:
                # Need minute-level chunking (last resort)
                logger.warning(f"          Hours {start_hour}-{end_hour} has {hour_count:,} articles - using MINUTE chunks")
                hour_articles = process_hour_with_minute_chunking(
                    condition, year, month, day, start_hour, end_hour, hour_count
                )
                day_articles.extend(hour_articles)
                
        except Exception as e:
            logger.error(f"Error processing hours {start_hour}-{end_hour}: {e}")
            continue
    
    return day_articles

def process_hour_with_minute_chunking(
    condition: str, year: int, month: int, day: int, 
    start_hour: int, end_hour: int, total_count: int
) -> List[Dict]:
    """Last resort: process with minute-level chunking"""
    hour_articles = []
    
    # Try 15-minute chunks within the hour range
    total_hours = end_hour - start_hour + 1
    total_minutes = total_hours * 60
    
    # Create 15-minute chunks
    for hour in range(start_hour, end_hour + 1):
        for minute_start in [0, 15, 30, 45]:
            minute_end = min(minute_start + 14, 59)
            
            query = create_condition_search_query(
                condition=condition,
                year=year,
                month=month,
                day=day,
                hour_range=(hour, hour),
                minute_range=(minute_start, minute_end)
            )
            
            try:
                handle = Entrez.esearch(db="pubmed", term=query, retmax=0)
                result = Entrez.read(handle)
                handle.close()
                minute_count = int(result["Count"])
                
                if minute_count == 0:
                    continue
                
                # Accept whatever we can get at minute level
                fetch_count = min(minute_count, 9999)
                if minute_count > 9999:
                    logger.error(f"            📛 LOSING {minute_count - 9999} articles from {hour:02d}:{minute_start:02d}-{hour:02d}:{minute_end:02d}")
                
                articles = retrieve_articles_batch(
                    condition, year, query, fetch_count,
                    f"{year}/{month:02d}/{day:02d} {hour:02d}:{minute_start:02d}-{minute_end:02d}"
                )
                hour_articles.extend(articles)
                
            except Exception as e:
                logger.error(f"Error processing minutes {hour:02d}:{minute_start:02d}-{minute_end:02d}: {e}")
                continue
    
    return hour_articles

def retrieve_articles_batch(condition: str, year: int, query: str, count: int, description: str = "") -> List[Dict]:
    """Retrieve a batch of articles"""
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=count)
        result = Entrez.read(handle)
        handle.close()
        
        pmids = result["IdList"]
        articles = fetch_articles_with_abstracts(pmids, f"{condition} - {description}")
        
        return articles
    except Exception as e:
        logger.error(f"Error retrieving batch for {condition}: {e}")
        return []

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

def generate_completeness_report():
    """Generate a completeness report for all conditions"""
    tracking = load_completeness_tracking()
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("RETRIEVAL COMPLETENESS REPORT")
    report_lines.append("="*80)
    
    for condition in CONDITIONS:
        if condition in tracking:
            report_lines.append(f"\n{condition}:")
            total_found_all = 0
            total_retrieved_all = 0
            
            for year in sorted(tracking[condition].keys()):
                data = tracking[condition][year]
                total_found_all += data['total_found']
                total_retrieved_all += data['total_retrieved']
                
                if data['completeness'] < 100:
                    report_lines.append(f"  {year}: {data['total_retrieved']:,}/{data['total_found']:,} ({data['completeness']:.1f}%)")
            
            overall_completeness = (total_retrieved_all / total_found_all * 100) if total_found_all > 0 else 100
            report_lines.append(f"  TOTAL: {total_retrieved_all:,}/{total_found_all:,} ({overall_completeness:.1f}%)")
    
    report_text = "\n".join(report_lines)
    
    # Save report
    report_file = os.path.join(data_dir, "completeness_report.txt")
    with open(report_file, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    return report_text

def main():
    """Main execution function"""
    print("="*80)
    print("PUBMED CONDITION-SPECIFIC ABSTRACTS RETRIEVAL")
    print("="*80)
    print(f"Conditions to search: {len(CONDITIONS)}")
    print(f"Years to cover: 1975-2024 (50 years)")
    print(f"Total combinations: {len(CONDITIONS) * 50:,}")
    print("="*80)
    
    if Entrez.email == "your.email@example.com":
        print("\n⚠️  ERROR: Please set your email address")
        return
    
    print(f"\nConfiguration:")
    print(f"  Email: {Entrez.email}")
    print(f"  API Key: {'Set' if Entrez.api_key else 'Not set'}")
    
    # Check for existing checkpoint
    checkpoint = load_checkpoint()
    if checkpoint:
        processed_count = sum(
            1 for cond in checkpoint.values() 
            for year_data in cond.values() 
            if year_data.get('status') == 'complete'
        )
        print(f"\n📊 Found checkpoint with {processed_count:,} condition-year combinations completed")
    
    proceed = input("\nProceed with retrieval? (y/n): ").strip().lower()
    if proceed not in ['y', 'yes']:
        return
    
    start_time = datetime.now()
    
    # Process each condition
    for condition in tqdm(CONDITIONS, desc="Processing conditions"):
        logger.info(f"\n{'='*60}")
        logger.info(f"CONDITION: {condition}")
        logger.info(f"{'='*60}")
        
        condition_total_found = 0
        condition_total_retrieved = 0
        
        # Process each year for this condition
        for year in range(1975, 2025):
            try:
                articles, total_found, total_retrieved = process_condition_year(condition, year)
                condition_total_found += total_found
                condition_total_retrieved += total_retrieved
                
                # Free up memory
                del articles
                gc.collect()
                
            except KeyboardInterrupt:
                print(f"\n⚠️  Interrupted at {condition} - {year}")
                print("Progress has been saved. Run again to resume.")
                generate_completeness_report()
                return
            except Exception as e:
                logger.error(f"Failed {condition} - {year}: {e}")
                continue
        
        logger.info(f"\nCondition Summary - {condition}:")
        logger.info(f"  Total found: {condition_total_found:,}")
        logger.info(f"  Total retrieved: {condition_total_retrieved:,}")
        if condition_total_found > 0:
            completeness = (condition_total_retrieved / condition_total_found * 100)
            logger.info(f"  Completeness: {completeness:.1f}%")
    
    # Generate final completeness report
    print("\n" + "="*80)
    print("GENERATING FINAL REPORT")
    print("="*80)
    generate_completeness_report()
    
    duration = datetime.now() - start_time
    print(f"\n✅ RETRIEVAL COMPLETE!")
    print(f"  Time taken: {duration}")
    print(f"  Data location: {data_dir}")
    print(f"  Completeness report: {os.path.join(data_dir, 'completeness_report.txt')}")

if __name__ == "__main__":
    main()