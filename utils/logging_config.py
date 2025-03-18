"""
Utility module for logging configuration shared across all scripts.
"""
from pathlib import Path
import logging
from datetime import datetime
import sys
from typing import Optional
from utils.file_manager import file_manager

def setup_logging(
    script_name: str,
    log_level: int = logging.INFO,
    retention_days: Optional[int] = 30
) -> Path:
    """
    Set up logging configuration with both file and console handlers.
    Creates a logs directory if it doesn't exist and manages log files.
    
    Args:
        script_name: Name of the script (used for log directory and file names)
        log_level: Logging level (default: logging.INFO)
        retention_days: Number of days to keep log files (default: 30, None for no cleanup)
    
    Returns:
        Path to the current log file
    """
    # Create logs directory if it doesn't exist
    logs_dir = file_manager.logs_dir
    logs_dir.mkdir(exist_ok=True)
    
    # Create script-specific log directory
    script_logs_dir = logs_dir / script_name
    script_logs_dir.mkdir(exist_ok=True)
    
    # Clean up old log files if retention period is specified
    if retention_days is not None:
        cleanup_old_logs(script_logs_dir, retention_days)
    
    # Generate timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = script_logs_dir / f"{timestamp}.log"
    
    # Create a symbolic link to the latest log
    latest_log = script_logs_dir / "latest.log"
    if latest_log.exists():
        latest_log.unlink()
    latest_log.symlink_to(log_file.name)
    
    # Set up logging configuration
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    
    # Log script start information
    logging.info(f"Starting {script_name}")
    logging.info(f"Python version: {sys.version}")
    logging.info(f"Log file: {log_file}")
    
    return log_file

def cleanup_old_logs(log_dir: Path, retention_days: int) -> None:
    """
    Remove log files older than the specified retention period.
    
    Args:
        log_dir: Directory containing log files
        retention_days: Number of days to keep log files
    """
    cutoff_time = datetime.now().timestamp() - (retention_days * 24 * 60 * 60)
    
    for log_file in log_dir.glob("*.log"):
        # Skip the latest.log symlink
        if log_file.name == "latest.log":
            continue
        
        # Remove if older than retention period
        if log_file.stat().st_mtime < cutoff_time:
            try:
                log_file.unlink()
                logging.debug(f"Removed old log file: {log_file}")
            except Exception as e:
                logging.warning(f"Failed to remove old log file {log_file}: {e}")

def get_latest_log(script_name: str) -> Optional[Path]:
    """
    Get the path to the latest log file for a script.
    
    Args:
        script_name: Name of the script
    
    Returns:
        Path to the latest log file or None if not found
    """
    latest_log = file_manager.logs_dir / script_name / "latest.log"
    return latest_log if latest_log.exists() else None 