import json
from pathlib import Path
from typing import Dict, List
import logging
import os
from dotenv import load_dotenv
from openai import OpenAI
import time
from datetime import datetime
from utils.logging_config import setup_logging
from utils.file_manager import file_manager
import sys
from tqdm import tqdm

# Set up logging for this script
current_log = setup_logging("deep_analyzer")

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Validate API key
if not os.getenv("OPENAI_API_KEY"):
    logging.error("OPENAI_API_KEY environment variable is not set!")
    sys.exit(1)

# Use paths from file_manager
SOURCE_ANALYSIS_DIR = file_manager.source_analysis_dir / "analysis_results"
DEEP_ANALYSIS_DIR = file_manager.deep_analysis_dir
logging.info(f"Using source analysis directory: {SOURCE_ANALYSIS_DIR}")
logging.info(f"Using deep analysis directory: {DEEP_ANALYSIS_DIR}")

def load_previous_analysis() -> Dict:
    """Load results from the previous source analysis."""
    try:
        # Check if "latest" symlink exists
        latest_dir = SOURCE_ANALYSIS_DIR / "latest"
        if latest_dir.exists() and latest_dir.is_symlink():
            results_path = latest_dir / "compiled_results.json"
        else:
            # Find the most recent run directory
            run_dirs = sorted([d for d in SOURCE_ANALYSIS_DIR.glob("run_*") if d.is_dir()], 
                            key=lambda x: x.name, reverse=True)
            if not run_dirs:
                raise FileNotFoundError("No source analysis runs found")
            results_path = run_dirs[0] / "compiled_results.json"
        
        logging.info(f"Loading source analysis results from: {results_path}")
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
            if not results:
                raise FileNotFoundError("Source analysis results file is empty")
            return results
    except Exception as e:
        logging.error(f"Failed to load source analysis results: {e}")
        raise FileNotFoundError(f"Source analysis results not found. Please run source_analyzer.py first.")

def make_api_call_with_retry(prompt: str, max_retries=3, base_delay=1) -> Dict:
    """Make API call with retry logic."""
    for attempt in range(max_retries):
        try:
            logging.info(f"Making API call attempt {attempt + 1}/{max_retries}")
            
            response = client.chat.completions.create(
                model="o3-mini",
                messages=[
                    {"role": "system", "content": "You are a Classical scholar specializing in ancient Greek and Arabic philosophy, focusing on detailed source analysis."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            logging.info("Successfully received and parsed JSON response")
            return result
            
        except Exception as e:
            if attempt == max_retries - 1:
                logging.error(f"API call failed after {max_retries} attempts: {e}")
                raise
            delay = base_delay * (2 ** attempt)
            logging.warning(f"API call failed, retrying in {delay} seconds... Error: {e}")
            time.sleep(delay)

def analyze_transmission_patterns(comparison: Dict) -> Dict:
    """Analyze patterns of textual transmission and influence based on concrete evidence."""
    # Extract metadata from the analysis
    input_metadata = comparison["analysis"]
    
    prompt = f"""Analyze the textual evidence for transmission and influence between these texts:

Text 1:
Author: {input_metadata.get('author', 'Unknown')}
Title: {input_metadata.get('title', 'Unknown')}
Type: {input_metadata.get('text_type', 'Unknown')}

Text 2:
Database Text: {comparison['database_file']}

Existing Analysis:
{json.dumps(comparison['analysis'], indent=2, ensure_ascii=False)}

Please analyze the concrete textual evidence for:
1. Direct textual dependencies:
   - Exact quotations or close paraphrases
   - Shared examples or illustrations
   - Similar structural organization
   - Common reference points

2. Conceptual dependencies:
   - Shared philosophical frameworks
   - Similar problem-solving approaches
   - Common argumentative patterns
   - Parallel theoretical constructs

3. Evidence of mediation:
   - References to other texts or authorities
   - Use of standard terminology or definitions
   - Common sources cited or alluded to
   - Shared technical vocabulary

4. Adaptation indicators:
   - Modifications of concepts or arguments
   - Contextual adjustments
   - Elaborations or simplifications
   - Novel applications of ideas

Focus only on evidence present in the texts themselves. Avoid speculating about historical transmission paths unless explicitly referenced in the texts.

Respond in JSON format with these fields:
- textual_dependencies: List of concrete textual parallels and dependencies
- conceptual_dependencies: List of shared philosophical frameworks and approaches
- mediation_evidence: List of references and shared sources found in the texts
- adaptation_evidence: List of documented modifications and adjustments
- evidence_strength: 0.0-1.0 score for the strength of textual evidence
- key_passages: List of specific passages that demonstrate the relationships
"""
    return make_api_call_with_retry(prompt)

def analyze_philosophical_development(comparison: Dict) -> Dict:
    """Analyze philosophical development and evolution between texts based on concrete evidence."""
    # Extract metadata from the analysis
    input_metadata = comparison["analysis"]
    
    prompt = f"""Analyze the philosophical relationship between these texts based on concrete textual evidence:

Text 1:
Author: {input_metadata.get('author', 'Unknown')}
Title: {input_metadata.get('title', 'Unknown')}
Type: {input_metadata.get('text_type', 'Unknown')}

Text 2:
Database Text: {comparison['database_file']}

Existing Analysis:
{json.dumps(comparison['analysis'], indent=2, ensure_ascii=False)}

Please analyze the concrete evidence for philosophical development in:

1. Argument Structure:
   - Logical frameworks used
   - Reasoning patterns
   - Syllogistic structures
   - Problem-solving approaches
   - Demonstration methods

2. Conceptual Framework:
   - Key philosophical terms and their usage
   - Definition patterns and techniques
   - Theoretical constructs
   - Categorical systems
   - Ontological frameworks

3. Methodological Approach:
   - Analytical techniques
   - Investigative methods
   - Demonstrative strategies
   - Examples and illustrations
   - Dialectical patterns

4. Philosophical Innovation:
   - Extensions of concepts
   - Novel applications
   - Refined arguments
   - Synthesized ideas
   - Original contributions

Focus only on philosophical elements that can be directly observed in the texts. Avoid speculation about influences unless explicitly referenced.

Respond in JSON format with these fields:
- argument_analysis: List of concrete similarities and differences in argumentation
- conceptual_analysis: List of shared and divergent philosophical concepts
- methodological_analysis: List of common and distinct analytical approaches
- innovation_evidence: List of documented philosophical developments
- evidence_strength: 0.0-1.0 score for the strength of philosophical evidence
- key_arguments: List of specific arguments that show development or influence
"""
    return make_api_call_with_retry(prompt)

def analyze_linguistic_transformation(comparison: Dict) -> Dict:
    """Analyze linguistic and terminological transformations based on concrete evidence."""
    # Extract metadata from the analysis
    input_metadata = comparison["analysis"]
    
    prompt = f"""Analyze the concrete linguistic evidence between these texts:

Text 1:
Author: {input_metadata.get('author', 'Unknown')}
Title: {input_metadata.get('title', 'Unknown')}
Type: {input_metadata.get('text_type', 'Unknown')}

Text 2:
Database Text: {comparison['database_file']}

Technical Vocabulary:
{json.dumps(comparison['analysis'].get('technical_vocabulary', []), indent=2, ensure_ascii=False)}

Please analyze the concrete linguistic evidence for:

1. Technical Terminology:
   - Shared technical terms
   - Equivalent concepts in different languages
   - Standard vs specialized usage
   - Definition patterns
   - Term relationships

2. Argumentative Language:
   - Logical connectors
   - Inferential markers
   - Demonstrative phrases
   - Dialectical terminology
   - Structural indicators

3. Conceptual Expression:
   - Abstract concept representation
   - Philosophical terminology usage
   - Definitional approaches
   - Explanatory patterns
   - Illustrative language

4. Textual Organization:
   - Section structuring
   - Argument presentation
   - Reference patterns
   - Cross-referencing
   - Internal organization

Focus only on linguistic elements that can be directly observed in the texts. Document specific examples for each observation.

Respond in JSON format with these fields:
- technical_analysis: List of shared technical terminology with specific examples
- argumentative_analysis: List of similar argumentative language patterns
- conceptual_expression: List of parallel ways of expressing philosophical concepts
- organizational_patterns: List of shared textual organization methods
- evidence_strength: 0.0-1.0 score for the strength of linguistic evidence
- key_terminology: Dictionary mapping equivalent terms between texts
- specific_examples: List of concrete examples supporting each analysis point
"""
    return make_api_call_with_retry(prompt)

def perform_deep_analysis():
    """Main function to perform deep analysis of promising source relationships."""
    try:
        print("\nStarting deep analysis...")
        
        # Create necessary directories
        DEEP_ANALYSIS_DIR.mkdir(exist_ok=True)
        results_dir = DEEP_ANALYSIS_DIR / "results"
        results_dir.mkdir(exist_ok=True)
        backup_dir = DEEP_ANALYSIS_DIR / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        # Load previous analysis results
        previous_results = load_previous_analysis()
        if not previous_results.get("comparisons"):
            raise ValueError("No comparisons found in previous analysis")
        
        # Filter for high-confidence comparisons
        promising_comparisons = [
            comp for comp in previous_results["comparisons"]
            if comp.get("analysis", {}).get("confidence_score", 0) >= 0.7
        ]
        
        if not promising_comparisons:
            print("\nNo high-confidence comparisons found for deep analysis")
            return
        
        print(f"\nFound {len(promising_comparisons)} promising comparisons for deep analysis")
        
        # Initialize results structure
        successful_analyses = {
            "metadata": {
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_pairs": len(promising_comparisons),
                "successful_analyses": 0,
                "failed_analyses": 0
            },
            "analyses": []
        }
        
        failed_analyses = []
        
        # Analyze each promising comparison
        for comparison in tqdm(promising_comparisons, desc="Performing deep analysis"):
            try:
                logging.info(f"Analyzing pair: {comparison['input_file']} -> {comparison['database_file']}")
                
                # Log the structure of the comparison for debugging
                logging.debug(f"Comparison structure: {json.dumps(comparison, indent=2, ensure_ascii=False)}")
                
                # Perform deep analyses
                transmission = analyze_transmission_patterns(comparison)
                philosophical = analyze_philosophical_development(comparison)
                linguistic = analyze_linguistic_transformation(comparison)
                
                # Combine analyses
                deep_analysis = {
                    "input_file": comparison["input_file"],
                    "database_file": comparison["database_file"],
                    "original_confidence": comparison["analysis"]["confidence_score"],
                    "transmission_analysis": transmission,
                    "philosophical_analysis": philosophical,
                    "linguistic_analysis": linguistic,
                    "aggregate_confidence": (
                        transmission.get("evidence_strength", 0) +
                        philosophical.get("evidence_strength", 0) +
                        linguistic.get("evidence_strength", 0)
                    ) / 3.0
                }
                
                successful_analyses["analyses"].append(deep_analysis)
                successful_analyses["metadata"]["successful_analyses"] += 1
                
                logging.info(f"Successfully completed deep analysis for pair")
                
            except Exception as e:
                logging.error(f"Failed to analyze pair: {e}", exc_info=True)
                failed_analyses.append({
                    "input_file": comparison.get("input_file", "Unknown"),
                    "database_file": comparison.get("database_file", "Unknown"),
                    "error": str(e),
                    "comparison_data": comparison  # Include the comparison data for debugging
                })
                successful_analyses["metadata"]["failed_analyses"] += 1
                continue
        
        if not successful_analyses["analyses"]:
            print("\nNo successful analyses completed")
            return
        
        # Save results and generate report
        print("\nSaving results...")
        
        # Create dated directory for this analysis run
        current_date = datetime.now().strftime("%Y%m%d")
        current_time = datetime.now().strftime("%H%M%S")
        run_dir = results_dir / f"run_{current_date}_{current_time}"
        run_dir.mkdir(exist_ok=True)
        
        # Save compiled results
        analysis_file = run_dir / "deep_analysis_results.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(successful_analyses, f, indent=2, ensure_ascii=False)
        
        # Save failed analyses separately
        failed_file = run_dir / "failed_analyses.json"
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump({"failed_analyses": failed_analyses}, f, indent=2, ensure_ascii=False)
        
        # Generate and save summary report
        summary = generate_analysis_report(successful_analyses)
        summary_file = run_dir / "deep_analysis_report.txt"
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
        
        print(f"\nDeep analysis complete!")
        print(f"Analyzed {len(promising_comparisons)} text pairs")
        print(f"- Successful: {successful_analyses['metadata']['successful_analyses']}")
        print(f"- Failed: {successful_analyses['metadata']['failed_analyses']}")
        print(f"\nResults saved to:")
        print(f"- Technical data: {analysis_file}")
        print(f"- Summary report: {summary_file}")
        if failed_analyses:
            print(f"- Failed analyses: {failed_file}")
        print(f"\nLatest results always available at:")
        print(f"- {results_dir}/latest/deep_analysis_results.json")
        print(f"- {results_dir}/latest/deep_analysis_report.txt")
        
        return successful_analyses
        
    except Exception as e:
        logging.error(f"Deep analysis failed: {e}")
        logging.debug("Exception details:", exc_info=True)
        raise

def generate_analysis_report(results: Dict) -> str:
    """Generate a detailed report of the deep analysis results."""
    report = ["DEEP ANALYSIS REPORT", "=" * 20 + "\n"]
    
    # Add metadata
    report.extend([
        f"Analysis Date: {results['metadata']['analysis_date']}",
        f"Total Pairs Analyzed: {results['metadata']['total_pairs']}",
        f"Successful Analyses: {results['metadata']['successful_analyses']}",
        f"Failed Analyses: {results['metadata']['failed_analyses']}\n"
    ])
    
    # Report on each analysis
    for i, analysis in enumerate(results["analyses"], 1):
        report.extend([
            f"\nANALYSIS {i}",
            "-" * 15,
            f"Input Text: {analysis['input_file']}",
            f"Database Text: {analysis['database_file']}",
            f"Original Confidence: {analysis['original_confidence']:.3f}",
            f"Aggregate Evidence Strength: {(analysis['transmission_analysis']['evidence_strength'] + analysis['philosophical_analysis']['evidence_strength'] + analysis['linguistic_analysis']['evidence_strength']) / 3.0:.3f}\n",
            
            "Textual Dependencies and Transmission:",
            "--------------------------------",
            "Direct Textual Dependencies:",
            *[f"  • {dep}" for dep in analysis["transmission_analysis"]["textual_dependencies"]],
            "\nConceptual Dependencies:",
            *[f"  • {dep}" for dep in analysis["transmission_analysis"]["conceptual_dependencies"]],
            "\nKey Supporting Passages:",
            *[f"  • {passage}" for passage in analysis["transmission_analysis"]["key_passages"]],
            
            "\nPhilosophical Analysis:",
            "---------------------",
            "Argument Structure:",
            *[f"  • {arg}" for arg in analysis["philosophical_analysis"]["argument_analysis"]],
            "\nKey Philosophical Concepts:",
            *[f"  • {concept}" for concept in analysis["philosophical_analysis"]["conceptual_analysis"]],
            "\nMethodological Approaches:",
            *[f"  • {method}" for method in analysis["philosophical_analysis"]["methodological_analysis"]],
            "\nKey Arguments:",
            *[f"  • {arg}" for arg in analysis["philosophical_analysis"]["key_arguments"]],
            
            "\nLinguistic Analysis:",
            "------------------",
            "Technical Terminology:",
            *[f"  • {term}" for term in analysis["linguistic_analysis"]["technical_analysis"]],
            "\nArgumentative Patterns:",
            *[f"  • {pattern}" for pattern in analysis["linguistic_analysis"]["argumentative_analysis"]],
            "\nConceptual Expression:",
            *[f"  • {expr}" for expr in analysis["linguistic_analysis"]["conceptual_expression"]],
            "\nSpecific Examples:",
            *[f"  • {example}" for example in analysis["linguistic_analysis"]["specific_examples"]],
            "\n"
        ])
    
    return "\n".join(report)

if __name__ == "__main__":
    try:
        # Ensure log directory exists
        log_dir = Path("logs/deep_analyzer")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        results = perform_deep_analysis()
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        logging.error("Analysis failed", exc_info=True)
        raise 