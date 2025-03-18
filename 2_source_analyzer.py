import json
from pathlib import Path
from typing import Dict, List
import logging
import os
from dotenv import load_dotenv
from anthropic import Anthropic
import time
from datetime import datetime
from utils.logging_config import setup_logging
from utils.file_manager import file_manager
import sys
from tqdm import tqdm

# Set up logging for this script
current_log = setup_logging("source_analyzer")

# Load environment variables
load_dotenv()

# Initialize Anthropic client
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Validate API key
if not os.getenv("ANTHROPIC_API_KEY"):
    logging.error("ANTHROPIC_API_KEY environment variable is not set!")
    sys.exit(1)

# Use paths from file_manager
INPUT_DIR = file_manager.input_dir
THEMATIZER_RESULTS_DIR = file_manager.thematizer_dir / "database_analyses"
logging.info(f"Using input directory: {INPUT_DIR}")
logging.info(f"Using thematizer results directory: {THEMATIZER_RESULTS_DIR}")

def load_thematizer_results() -> Dict:
    """Load the latest thematizer results."""
    # Find the latest run directory in the thematizer results directory
    if not THEMATIZER_RESULTS_DIR.exists():
        raise FileNotFoundError(f"Thematizer results directory not found: {THEMATIZER_RESULTS_DIR}")
    
    # Check if "latest" symlink exists
    latest_dir = THEMATIZER_RESULTS_DIR / "latest"
    if latest_dir.exists() and latest_dir.is_symlink():
        results_path = latest_dir / "compiled_results.json"
    else:
        # Find the most recent run directory
        run_dirs = sorted([d for d in THEMATIZER_RESULTS_DIR.glob("run_*") if d.is_dir()], 
                        key=lambda x: x.name, reverse=True)
        if not run_dirs:
            raise FileNotFoundError("No thematizer runs found")
        results_path = run_dirs[0] / "compiled_results.json"
    
    logging.info(f"Loading thematizer results from: {results_path}")
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
            if not results:
                raise FileNotFoundError("Thematizer results file is empty")
            return results
    except Exception as e:
        logging.error(f"Error loading thematizer results: {e}")
        raise FileNotFoundError(f"Thematizer results not found at {results_path}. Please run thematizer.py first.")

# Function to get input files directly
def get_input_files() -> List[Path]:
    """Get all input files from the specified directory."""
    return file_manager.get_input_files()

def read_text_file(file_path: str) -> str:
    """Read the contents of a text file with encoding fallback."""
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Failed to read {file_path} with any supported encoding")

def clean_json_response(content: str) -> str:
    """Clean and validate JSON response from the API."""
    content = content.strip()
    
    if content.startswith('```'):
        lines = content.split('\n')
        content_lines = []
        in_json = False
        for line in lines:
            if line.startswith('```json'):
                in_json = True
                continue
            elif line.startswith('```'):
                in_json = False
                continue
            if in_json:
                content_lines.append(line)
        content = '\n'.join(content_lines)
    
    if not (content.startswith('{') and content.endswith('}')):
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1:
            content = content[start:end+1]
        else:
            raise ValueError("Could not find valid JSON object in response")
    
    return content

def make_api_call_with_retry(prompt, max_retries=3, base_delay=1):
    """Make API call with retry logic."""
    formatted_prompt = f"\n\nHuman: {prompt}\n\nAssistant: I'll analyze this text pair carefully and provide a response in the requested JSON format."
    
    for attempt in range(max_retries):
        try:
            logging.info(f"Making API call attempt {attempt + 1}/{max_retries}")
            
            response = client.messages.create(
                model="claude-3-5-sonnet-latest",
                max_tokens=8192,
                messages=[{"role": "user", "content": formatted_prompt}]
            )
            
            content = response.content[0].text.strip()
            logging.info(f"Raw API response: {content[:100]}...")
            
            if not (content.startswith("{") and content.endswith("}")):
                content = "{" + content.split("{", 1)[1].rsplit("}", 1)[0] + "}"
            
            result = json.loads(content)
            logging.info("Successfully parsed JSON response")
            return json.dumps(result, ensure_ascii=False)
            
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

def find_relevant_texts(analysis_results: Dict) -> List[Dict]:
    """Find database texts that are relevant to the input texts using Claude API."""
    relevant_pairs = []
    
    for input_file, input_data in analysis_results["input_texts"].items():
        if "error" in input_data:
            logging.warning(f"Skipping input file {input_file} due to analysis error")
            continue
            
        input_analysis = input_data["analysis"]
        logging.info(f"\nAnalyzing relevance for input file: {input_file}")
        
        for db_file, db_data in analysis_results["database_texts"].items():
            if "error" in db_data:
                logging.warning(f"Skipping database file {db_file} due to analysis error")
                continue
                
            db_analysis = db_data["analysis"]
            logging.info(f"\nComparing with database file: {db_file}")
            
            context = {
                "input_file": input_file,
                "db_file": db_file,
                "input_author": input_analysis["author"],
                "input_title": input_analysis["title"],
                "input_type": input_analysis["text_type"],
                "db_author": db_analysis["author"],
                "db_title": db_analysis["title"],
                "db_type": db_analysis["text_type"]
            }
            
            prompt = f"""You are a Classical scholar specializing in classical Chinese, ancient Greek, and Arabic philosophy.
Based on the following analysis results, determine if these texts are likely to have a meaningful source relationship.

Text 1:
Author: {context['input_author']}
Title: {context['input_title']}
Type: {context['input_type']}
Themes: {', '.join(input_analysis.get('themes', []))}
Abstract: {input_analysis.get('abstract', 'No abstract available')}

Text 2:
Author: {context['db_author']}
Title: {context['db_title']}
Type: {context['db_type']}
Themes: {', '.join(db_analysis.get('themes', []))}
Abstract: {db_analysis.get('abstract', 'No abstract available')}

Respond in this exact JSON format:
{{
    "is_relevant": true/false,
    "relevance_score": 0.0-1.0,
    "common_themes": ["theme1", "theme2", ...],
    "rationale": "Brief explanation of why these texts are or aren't relevant"
}}"""

            try:
                response = make_api_call_with_retry(prompt)
                result = json.loads(response)
                
                if result.get("is_relevant", False) and result.get("relevance_score", 0) >= 0.3:
                    relevant_pairs.append({
                        "input_file": input_file,
                        "database_file": db_file,
                        "relevance_score": result["relevance_score"],
                        "common_themes": result.get("common_themes", []),
                        "rationale": result.get("rationale", "")
                    })
                    logging.info(f"Found relevant pair: {input_file} -> {db_file}")
                    logging.info(f"Rationale: {result.get('rationale', '')}")
            except Exception as e:
                logging.error(f"Error analyzing relevance for {input_file} -> {db_file}: {e}")
                continue
    
    return sorted(relevant_pairs, key=lambda x: x["relevance_score"], reverse=True)

def analyze_text_pair(input_text: str, db_text: str, context: Dict) -> Dict:
    """Perform detailed source analysis on a pair of texts."""
    prompt = f"""You are a Classical scholar specializing in classical Chinese, ancient Greek, and Arabic philosophy. 
You are analyzing two texts for potential source relationships.

Text 1 Context:
Author: {context['input_author']}
Title: {context['input_title']}
Type: {context['input_type']}

Text 2 Context:
Author: {context['db_author']}
Title: {context['db_title']}
Type: {context['db_type']}

Please analyze these texts for:

1. Verbal Parallels:
   - Direct quotations
   - Close paraphrases
   - Shared terminology
   - Similar phrasing

2. Conceptual Parallels:
   - Shared philosophical ideas
   - Similar arguments
   - Related examples
   - Common themes

3. Methodological Parallels:
   - Similar analytical approaches
   - Shared argumentative structures
   - Common organizational patterns
   - Related scholarly methods

4. Technical Vocabulary:
   - Shared philosophical terms
   - Similar technical language
   - Common specialized concepts
   - Related terminological usage

Analyze these texts and respond in this exact JSON format:
{{
    "verbal_parallels": ["parallel1", "parallel2", ...],
    "conceptual_parallels": ["parallel1", "parallel2", ...],
    "methodological_parallels": ["parallel1", "parallel2", ...],
    "technical_vocabulary": ["term1", "term2", ...],
    "analysis_summary": "A detailed summary of the relationship between these texts",
    "confidence_score": 0.0-1.0,
    "recommended_research": ["suggestion1", "suggestion2", ...]
}}

Text 1:
{input_text}

Text 2:
{db_text}"""

    try:
        response = make_api_call_with_retry(prompt)
        result = json.loads(response)
        logging.info(f"Successfully completed source analysis with confidence score: {result.get('confidence_score', 0.0)}")
        return result
    except Exception as e:
        logging.error(f"Failed to analyze text pair: {e}")
        raise

def analyze_sources() -> Dict:
    """Main function to analyze source relationships."""
    try:
        print("\nValidating file structure and loading results...")
        
        # Create necessary directories
        source_analysis_dir = file_manager.source_analysis_dir
        backup_dir = source_analysis_dir / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        # Load thematizer results for database texts
        thematizer_results = load_thematizer_results()
        logging.info("Loaded thematizer results")
        
        # Get input files and analyze them using the same format as database texts
        input_files = get_input_files()
        if not input_files:
            raise ValueError(f"No input files found in {INPUT_DIR}. Please add input files.")
            
        print(f"\nAnalyzing {len(input_files)} input files...")
        input_results = {}
        for file_path in tqdm(input_files, desc="Analyzing input texts"):
            file_name = file_path.name
            try:
                input_text = read_text_file(str(file_path))
                
                # Use Claude to analyze the input text
                prompt = f"""You are a Classical scholar specializing in ancient Greek and Arabic philosophy.
Analyze this text and provide basic metadata and themes.

Please respond in this exact JSON format:
{{
    "author": "Name of the author",
    "title": "Title of the work",
    "text_type": "Type of text (commentary/treatise/original text)",
    "themes": ["theme1", "theme2", ...],
    "abstract": "A brief abstract of the content"
}}

Text to analyze:
{input_text[:5000]}  # Using first 5000 chars for initial analysis
"""
                response = make_api_call_with_retry(prompt)
                analysis = json.loads(response)
                
                input_results[file_name] = {
                    "analysis": analysis
                }
                logging.info(f"Analyzed input file: {file_name}")
                
            except Exception as e:
                logging.error(f"Failed to analyze input file {file_name}: {e}")
                input_results[file_name] = {"error": str(e)}
        
        if not any(not "error" in data for data in input_results.values()):
            raise ValueError("All input text analyses failed")
        
        # Combine results for comparison
        combined_results = {
            "input_texts": input_results,
            "database_texts": thematizer_results.get("database_texts", {})
        }
        
        print("\nFinding relevant text pairs...")
        # Find relevant text pairs
        relevant_pairs = find_relevant_texts(combined_results)
        if not relevant_pairs:
            logging.warning("No relevant text pairs found for analysis")
            return {"error": "No relevant text pairs found"}
        
        print(f"\nFound {len(relevant_pairs)} relevant pairs for detailed analysis")
        
        # Analyze each relevant pair
        results = {
            "input_texts": input_results,
            "database_texts": thematizer_results.get("database_texts", {}),
            "comparisons": []
        }
        
        print("\nPerforming detailed source analysis...")
        for pair in tqdm(relevant_pairs, desc="Analyzing text pairs"):
            try:
                input_file = pair["input_file"]
                db_file = pair["database_file"]
                logging.info(f"Analyzing pair: {input_file} -> {db_file}")
                
                # Read the texts
                input_path = INPUT_DIR / input_file
                database_dir = file_manager.database_dir
                db_path = database_dir / db_file
                
                input_text = read_text_file(str(input_path))
                db_text = read_text_file(str(db_path))
                
                # Get analysis context
                input_analysis = input_results[input_file]["analysis"]
                db_analysis = thematizer_results["database_texts"][db_file]["analysis"]
                
                context = {
                    "input_file": input_file,
                    "db_file": db_file,
                    "input_author": input_analysis["author"],
                    "input_title": input_analysis["title"],
                    "input_type": input_analysis["text_type"],
                    "db_author": db_analysis["author"],
                    "db_title": db_analysis["title"],
                    "db_type": db_analysis["text_type"]
                }
                
                # Perform source analysis
                source_analysis = analyze_text_pair(input_text, db_text, context)
                
                # Add to results
                comparison_result = {
                    **pair,
                    "analysis": source_analysis
                }
                results["comparisons"].append(comparison_result)
                
                logging.info(f"Completed analysis with confidence score: {source_analysis.get('confidence_score', 0.0)}")
                
            except Exception as e:
                logging.error(f"Error analyzing pair: {e}", exc_info=True)
                continue
        
        print("\nSaving results...")
        # Save results
        results_dir = source_analysis_dir / "analysis_results"
        results_dir.mkdir(exist_ok=True)
        
        # Create dated directory for this analysis run
        current_date = datetime.now().strftime("%Y%m%d")
        current_time = datetime.now().strftime("%H%M%S")
        run_dir = results_dir / f"run_{current_date}_{current_time}"
        run_dir.mkdir(exist_ok=True)
        
        # Save compiled results
        results_file = run_dir / "compiled_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Generate and save summary report
        summary = generate_analysis_summary(results)
        summary_file = run_dir / "analysis_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        # Create symlink to latest results
        latest_dir = results_dir / "latest"
        try:
            if latest_dir.exists():
                latest_dir.unlink()
            if latest_dir.is_symlink():
                latest_dir.unlink()
            latest_dir.symlink_to(run_dir.relative_to(results_dir), target_is_directory=True)
        except Exception as e:
            logging.warning(f"Failed to create latest symlink: {e}")
        
        print(f"\nSource analysis complete!")
        print(f"Analyzed {len(results['comparisons'])} text pairs")
        print(f"\nResults saved to:")
        print(f"- Technical data: {results_file}")
        print(f"- Summary report: {summary_file}")
        print(f"\nLatest results always available at:")
        print(f"- {results_dir}/latest/compiled_results.json")
        print(f"- {results_dir}/latest/analysis_summary.txt")
        
        return results
        
    except Exception as e:
        logging.error(f"Source analysis failed: {e}")
        logging.debug("Exception details:", exc_info=True)
        raise

def generate_analysis_summary(results: Dict) -> str:
    """Generate a human-readable summary of the source analysis results."""
    summary = ["SOURCE ANALYSIS SUMMARY", "=" * 25 + "\n"]
    
    # Add timestamp
    summary.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Summarize comparisons
    if "comparisons" not in results:
        summary.append("No comparisons were made.")
        return "\n".join(summary)
        
    summary.append(f"Total Comparisons Analyzed: {len(results['comparisons'])}\n")
    
    for i, comparison in enumerate(results['comparisons'], 1):
        summary.extend([
            f"\nCOMPARISON {i}",
            "-" * 15,
            f"Input Text: {comparison['input_file']}",
            f"Database Text: {comparison['database_file']}",
            f"Relevance Score: {comparison['relevance_score']:.3f}",
            f"Rationale: {comparison['rationale']}",
            "\nCommon Themes:",
            *[f"  • {theme}" for theme in comparison.get('common_themes', [])]
        ])
        
        if 'analysis' in comparison:
            analysis = comparison['analysis']
            summary.extend([
                "\nAnalysis Summary:",
                analysis.get('analysis_summary', 'No summary available'),
                "\nConfidence Score:",
                f"  {analysis.get('confidence_score', 0.0):.3f}"
            ])
    
    return "\n".join(summary)

if __name__ == "__main__":
    try:
        print("\nStarting source analysis...")
        # Ensure log directory exists
        log_dir = file_manager.logs_dir / "source_analyzer"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        results = analyze_sources()
        print("\nAnalysis completed successfully!")
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        logging.error("Analysis failed", exc_info=True)
        raise 