import os
import json
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, List
import logging
import time
import sys
from anthropic import Anthropic
from utils.logging_config import setup_logging
from utils.file_manager import file_manager
from datetime import datetime

# Set up logging for this script
logs_dir = file_manager.logs_dir / "thematizer"
logs_dir.mkdir(parents=True, exist_ok=True)  # Ensure logs directory exists
current_log = setup_logging("thematizer", log_level=logging.INFO)

# Force the root logger to use our configuration
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Add a separate file handler for additional safety
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
file_handler = logging.FileHandler(f"{logs_dir}/{timestamp}.log")
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
file_handler.setLevel(logging.INFO)
root_logger.addHandler(file_handler)

# Verify logging is working
logging.info("Thematizer script started")
logging.info(f"Log file: {current_log}")

# Load environment variables
load_dotenv()

# Initialize Anthropic client
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Validate API key
if not os.getenv("ANTHROPIC_API_KEY"):
    logging.error("ANTHROPIC_API_KEY environment variable is not set!")
    sys.exit(1)

# Use database directory from file_manager
DATABASE_DIR = file_manager.database_dir
logging.info(f"Using database directory: {DATABASE_DIR}")

def read_text_file(file_path: str) -> str:
    """Read the contents of a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # Try with different encodings if UTF-8 fails
        encodings = ['latin-1', 'iso-8859-1', 'cp1252']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        raise

# Function to get database files directly from the specified path
def get_database_files() -> List[Path]:
    """Get all database files from the specified directory."""
    return file_manager.get_database_files()

def split_text(text: str, chunk_size: int = 20000) -> List[str]:
    """Split text into chunks of approximately chunk_size characters, respecting natural breaks.
    Using 20K characters (~5-7K tokens) to stay well within rate limits while retaining context."""
    # First try to split by major section breaks
    major_breaks = ['\n\n\n', '\n\n', '\n']
    
    for break_type in major_breaks:
        paragraphs = text.split(break_type)
        if len(paragraphs) > 1:
            break
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for para in paragraphs:
        para_size = len(para)
        
        # If a single paragraph is too large, split it by sentences
        if para_size > chunk_size:
            sentences = para.replace('. ', '.\n').split('\n')
            for sentence in sentences:
                sentence_size = len(sentence)
                if current_size + sentence_size > chunk_size:
                    if current_chunk:
                        chunks.append('\n'.join(current_chunk))
                    current_chunk = [sentence]
                    current_size = sentence_size
                else:
                    current_chunk.append(sentence)
                    current_size += sentence_size
        # Otherwise handle the paragraph as a unit
        elif current_size + para_size > chunk_size:
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
            current_chunk = [para]
            current_size = para_size
        else:
            current_chunk.append(para)
            current_size += para_size
    
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks

def make_api_call_with_retry(prompt, max_retries=3, base_delay=1):
    """Make API call with retry logic."""
    
    # Format prompt for Claude
    formatted_prompt = f"\n\nHuman: {prompt}\n\nAssistant: I'll analyze this text carefully and provide a response in the requested JSON format."
    
    for attempt in range(max_retries):
        try:
            logging.info(f"Making API call attempt {attempt + 1}/{max_retries}")
            
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=8192,  # Maximum output tokens for this model
                messages=[
                    {
                        "role": "user",
                        "content": formatted_prompt
                    }
                ]
            )
            
            # Extract the response text
            content = response.content[0].text.strip()
            logging.info(f"Raw API response: {content[:100]}...")  # Log first 100 chars
            
            # Ensure we have valid JSON
            if not (content.startswith("{") and content.endswith("}")):
                content = "{" + content.split("{", 1)[1].rsplit("}", 1)[0] + "}"
            
            # Test if it's valid JSON
            result = json.loads(content)
            logging.info("Successfully parsed JSON response")
            return json.dumps(result, ensure_ascii=False)  # Return standardized JSON string
            
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON response on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                raise
        except Exception as e:
            if attempt == max_retries - 1:
                logging.error(f"API call failed after {max_retries} attempts: {e}")
                raise
            delay = base_delay * (2 ** attempt)
            logging.warning(f"API call failed, retrying in {delay} seconds... Error: {e}")
            time.sleep(delay)

def clean_and_validate_json(content: str) -> Dict:
    """Clean and validate JSON content from the API response."""
    # Remove Markdown code block formatting if present
    if content.startswith('```'):
        lines = content.split('\n')
        content_lines = [line for line in lines if not line.startswith('```')]
        content = '\n'.join(content_lines)
    
    # Attempt to parse the JSON
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON response: {e}")
        raise

def analyze_text_chunk(chunk: str, chunk_number: int, total_chunks: int) -> Dict:
    """Analyze a single chunk of text."""
    logging.info(f"Starting analysis of chunk {chunk_number}/{total_chunks} (length: {len(chunk)} chars)")
    
    prompt = f"""You are a Classical scholar specializing in ancient Greek and Arabic philosophy. 
You are analyzing an academic philosophical text that could be:
1. An ancient Greek philosophical work (like Aristotle's originals)
2. An Arabic philosophical text (like Al-Farabi's originals)
3. A commentary or interpretation of philosophical works (like Al-Farabi's commentaries)
4. Part of the established philosophical canon

Your task is to perform careful scholarly analysis of this text. Pay special attention to:
- References to authors or titles in the text
- Writing style and terminology that might indicate the text type
- Technical philosophical terms and concepts
- Section breaks or chapter divisions
- The relationship to Aristotelian thought

Please analyze the following chunk ({chunk_number} of {total_chunks}) and extract:

1. Author identification:
   - Look for explicit mentions of the author
   - Carefully consider writing style and historical context
   - Note if it's attributed to a specific philosophical school

2. Title identification:
   - Look for explicit title mentions
   - Consider standard names of philosophical works
   - Note if it's part of a larger work

3. Text type classification:
   - "original text" (e.g., Aristotle's works)
   - "commentary" (direct commentary on another work)
   - "treatise" (independent philosophical work)
   - "unknown" (if unclear)

4. Themes and concepts:
   - Major philosophical themes
   - Technical terminology
   - Key arguments and concepts
   - Philosophical methodology

5. Abstract focusing on:
   - Main philosophical arguments
   - Relationship to other philosophical works
   - Historical and intellectual context
   - Significance of the ideas presented

6. Structural elements:
   - Chapter or section divisions
   - Natural topic transitions
   - Argument structure breaks
   - Reference points in the text

You MUST respond in valid JSON format with these exact fields:
{{
    "author": "Author's name or 'Unknown' if not found in this chunk",
    "title": "Title of the work or 'Unknown' if not found in this chunk",
    "text_type": "original text|commentary|treatise|unknown",
    "themes": ["theme1", "theme2", ...],
    "abstract": "A scholarly abstract of this chunk of text.",
    "natural_breaks": ["Break 1", "Break 2", ...] or [] if none found
}}

Here's the text chunk to analyze:
{chunk}"""

    try:
        response = make_api_call_with_retry(prompt)
        result = json.loads(response)
        logging.info(f"Successfully analyzed chunk {chunk_number}/{total_chunks}")
        return result
    except Exception as e:
        logging.error(f"Failed to analyze chunk {chunk_number}/{total_chunks}: {e}")
        raise

def merge_chunk_analyses(chunks_results: List[Dict]) -> Dict:
    """Merge analyses from multiple chunks with improved logic."""
    # Count occurrences of each author and title
    author_counts = {}
    title_counts = {}
    text_type_counts = {}
    
    for r in chunks_results:
        if r["author"] != "Unknown":
            author_counts[r["author"]] = author_counts.get(r["author"], 0) + 1
        if r["title"] != "Unknown":
            title_counts[r["title"]] = title_counts.get(r["title"], 0) + 1
        if r["text_type"] != "unknown":
            text_type_counts[r["text_type"]] = text_type_counts.get(r["text_type"], 0) + 1
    
    # Select most frequent values
    author = max(author_counts.items(), key=lambda x: x[1])[0] if author_counts else "Unknown"
    title = max(title_counts.items(), key=lambda x: x[1])[0] if title_counts else "Unknown"
    text_type = max(text_type_counts.items(), key=lambda x: x[1])[0] if text_type_counts else "unknown"
    
    # Merge themes with deduplication and sorting
    all_themes = list(set(theme for r in chunks_results for theme in r["themes"]))
    all_themes.sort()
    
    # Combine abstracts intelligently
    abstracts = [r["abstract"] for r in chunks_results]
    combined_abstract = " ".join(abstracts)
    if len(combined_abstract) > 1000:  # Trim if too long
        combined_abstract = combined_abstract[:997] + "..."
    
    # Merge natural breaks in order
    all_breaks = []
    for r in chunks_results:
        for break_ in r["natural_breaks"]:
            if break_ not in all_breaks:  # Maintain order but avoid duplicates
                all_breaks.append(break_)
    
    return {
        "author": author,
        "title": title,
        "text_type": text_type,
        "themes": all_themes,
        "abstract": combined_abstract,
        "natural_breaks": all_breaks
    }

def analyze_text(text: str) -> Dict:
    """Analyze text by splitting into chunks if necessary."""
    chunks = split_text(text)
    if len(chunks) == 1:
        return analyze_text_chunk(chunks[0], 1, 1)
    
    # Process multiple chunks
    chunks_results = []
    for i, chunk in enumerate(chunks, 1):
        logging.info(f"Processing chunk {i} of {len(chunks)}")
        try:
            result = analyze_text_chunk(chunk, i, len(chunks))
            chunks_results.append(result)
        except Exception as e:
            logging.error(f"Error processing chunk {i}: {e}")
            continue
    
    if not chunks_results:
        return {
            "error": "All chunks failed to process",
            "author": "Unknown",
            "title": "Unknown",
            "text_type": "unknown",
            "themes": [],
            "abstract": "Error: Failed to analyze text",
            "natural_breaks": []
        }
    
    return merge_chunk_analyses(chunks_results)

def analyze_texts() -> Dict:
    """Main function to analyze database texts."""
    try:
        logging.info("Starting database text analysis process...")
        
        # Create thematizer directory structure if it doesn't exist
        thematizer_dir = file_manager.thematizer_dir
        
        # Create backup of any existing results
        backup_dir = thematizer_dir / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        # Initialize results structure for final compilation
        results = {
            "database_texts": {},
            "metadata": {
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_files_analyzed": 0,
                "successful_analyses": 0,
                "failed_analyses": 0
            }
        }
        
        # Process database texts individually using our custom function
        database_files = get_database_files()
        if not database_files:
            logging.warning("No database files found to analyze")
            return results
        else:
            logging.info(f"Found {len(database_files)} database files to analyze")
            results["metadata"]["total_files_analyzed"] = len(database_files)
        
        # Create organized directory structure for analyses
        analysis_base_dir = thematizer_dir / "database_analyses"
        analysis_base_dir.mkdir(exist_ok=True)
        
        # Create dated directory for this analysis run
        current_date = datetime.now().strftime("%Y%m%d")
        current_time = datetime.now().strftime("%H%M%S")
        run_dir = analysis_base_dir / f"run_{current_date}_{current_time}"
        run_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different types of outputs
        individual_dir = run_dir / "individual_analyses"
        individual_dir.mkdir(exist_ok=True)
        
        for file_path in database_files:
            try:
                logging.info(f"Starting analysis of database file: {file_path.name}")
                text = read_text_file(str(file_path))
                chunks = split_text(text)
                logging.info(f"Split {file_path.name} into {len(chunks)} chunks for analysis")
                
                chunks_results = []
                for i, chunk in enumerate(chunks, 1):
                    logging.info(f"Analyzing chunk {i}/{len(chunks)} of {file_path.name}")
                    chunk_result = analyze_text_chunk(chunk, i, len(chunks))
                    chunks_results.append(chunk_result)
                    logging.info(f"Completed analysis of chunk {i}/{len(chunks)}")
                
                merged_analysis = merge_chunk_analyses(chunks_results)
                
                # Save individual analysis with clean, descriptive filename
                clean_name = ''.join(c if c.isalnum() else '_' for c in file_path.stem)
                analysis_filename = f"{clean_name}_analysis.json"
                analysis_path = individual_dir / analysis_filename
                
                analysis_data = {
                    "metadata": {
                        "original_filename": file_path.name,
                        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "total_chunks": len(chunks),
                        "file_size": os.path.getsize(str(file_path))
                    },
                    "analysis": merged_analysis,
                    "chunk_analyses": chunks_results
                }
                
                with open(analysis_path, 'w', encoding='utf-8') as f:
                    json.dump(analysis_data, f, indent=2, ensure_ascii=False)
                logging.info(f"Saved individual analysis to: {analysis_path}")
                
                # Add to final results
                results["database_texts"][file_path.name] = {
                    "analysis": merged_analysis,
                    "chunk_analyses": chunks_results
                }
                
                results["metadata"]["successful_analyses"] += 1
                logging.info(f"Successfully completed analysis of {file_path.name}")
                
            except Exception as e:
                logging.error(f"Failed to analyze {file_path.name}: {e}", exc_info=True)
                results["database_texts"][file_path.name] = {
                    "error": str(e)
                }
                results["metadata"]["failed_analyses"] += 1
        
        # Save compiled results with clear naming
        compiled_results_file = run_dir / "compiled_results.json"
        with open(compiled_results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved compiled results to: {compiled_results_file}")
        
        # Create symlink to latest results
        latest_symlink = analysis_base_dir / "latest"
        try:
            if latest_symlink.exists():
                latest_symlink.unlink()
            if latest_symlink.is_symlink():
                latest_symlink.unlink()
            latest_symlink.symlink_to(run_dir.relative_to(analysis_base_dir), target_is_directory=True)
        except Exception as e:
            logging.warning(f"Failed to create latest symlink: {e}")
            # Continue execution as this is not critical
        
        # Generate and save summary report
        summary = generate_analysis_summary(results)
        summary_path = run_dir / "analysis_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        logging.info(f"Saved analysis summary to: {summary_path}")
        
        # Print final status
        print(f"\nThematic Analysis Complete!")
        print(f"Run Directory: {run_dir}")
        print(f"\nAnalysis Statistics:")
        print(f"- Total files processed: {results['metadata']['total_files_analyzed']}")
        print(f"- Successful analyses: {results['metadata']['successful_analyses']}")
        print(f"- Failed analyses: {results['metadata']['failed_analyses']}")
        print(f"\nOutput Locations:")
        print(f"1. Individual Analyses: {individual_dir}")
        print(f"2. Compiled Results: {compiled_results_file}")
        print(f"3. Summary Report: {summary_path}")
        print(f"\nQuick Access:")
        print(f"Latest results always available at: {latest_symlink}")
        
        return results
        
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        logging.debug("Exception details:", exc_info=True)
        raise

def generate_analysis_summary(results: Dict) -> str:
    """Generate a human-readable summary of the thematic analysis results."""
    summary = ["THEMATIC ANALYSIS SUMMARY", "=" * 25 + "\n"]
    
    # Add timestamp
    summary.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Summarize database texts
    summary.append("DATABASE TEXTS")
    summary.append("-" * 13)
    
    for filename, data in results["database_texts"].items():
        summary.append(f"\nFile: {filename}")
        if "error" in data:
            summary.append(f"Error: {data['error']}")
            continue
            
        analysis = data["analysis"]
        summary.extend([
            f"Author: {analysis['author']}",
            f"Title: {analysis['title']}",
            f"Type: {analysis['text_type']}",
            "\nThemes:",
            *[f"  • {theme}" for theme in analysis['themes']],
            "\nAbstract:",
            analysis['abstract'],
            "\nStructural Breaks:",
            *[f"  • {break_}" for break_ in analysis['natural_breaks']]
        ])
        summary.append("-" * 50)
    
    return "\n".join(summary)

if __name__ == "__main__":
    try:
        print("\nStarting thematic analysis...")
        results = analyze_texts()
        print("\nAnalysis completed successfully!")
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        logging.error("Analysis failed", exc_info=True)
        raise
