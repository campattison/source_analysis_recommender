"""
Utility module for managing file organization and directory structure.
"""
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
import logging
import json
import shutil
import os

class FileManager:
    """Manages file organization and directory structure for the source analysis tools."""
    
    def __init__(self):
        # Base directories
        self.base_dir = Path.cwd()
        self.documents_dir = self.base_dir / "documents"
        self.logs_dir = self.base_dir / "logs"
        self.results_dir = self.documents_dir / "results"
        
        # Input/Database directories
        self.input_dir = self.documents_dir / "input"
        self.database_dir = self.documents_dir / "database"
        
        # Results subdirectories
        self.thematizer_dir = self.results_dir / "thematizer"
        self.source_analysis_dir = self.results_dir / "source_analysis"
        self.deep_analysis_dir = self.results_dir / "deep_analysis"
        
        # Create all directories
        self._create_directory_structure()
    
    def _create_directory_structure(self) -> None:
        """Create the complete directory structure."""
        directories = [
            self.documents_dir,
            self.logs_dir,
            self.results_dir,
            self.input_dir,
            self.database_dir,
            self.thematizer_dir,
            self.source_analysis_dir,
            self.deep_analysis_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logging.debug(f"Ensured directory exists: {directory}")
    
    def get_input_files(self) -> List[Path]:
        """Get list of all input files."""
        return list(self.input_dir.glob("*.txt"))
    
    def get_database_files(self) -> List[Path]:
        """Get list of all database files."""
        return list(self.database_dir.glob("*.txt"))
    
    def validate_file_structure(self) -> bool:
        """
        Validate that the required directories exist and have proper permissions.
        Returns True if structure is valid, False otherwise.
        """
        try:
            # Check all directories exist
            required_dirs = [
                self.documents_dir,
                self.input_dir,
                self.database_dir,
                self.results_dir,
                self.thematizer_dir,
                self.source_analysis_dir,
                self.deep_analysis_dir
            ]
            
            for directory in required_dirs:
                if not directory.exists():
                    logging.error(f"Required directory missing: {directory}")
                    return False
                if not os.access(directory, os.W_OK):
                    logging.error(f"No write permission for directory: {directory}")
                    return False
            
            # Check input and database directories have files
            if not self.get_input_files():
                logging.warning("No input files found in input directory")
            if not self.get_database_files():
                logging.warning("No database files found in database directory")
            
            return True
            
        except Exception as e:
            logging.error(f"Error validating file structure: {e}")
            return False
    
    def backup_results(self, script_name: str) -> Optional[Path]:
        """
        Create a backup of the latest results for a script.
        
        Args:
            script_name: Name of the script (thematizer, source_analysis, or deep_analysis)
            
        Returns:
            Path to backup directory or None if backup failed
        """
        try:
            # Create backup directory
            backup_dir = self.results_dir / "backups" / script_name
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Get source directory
            source_dir = getattr(self, f"{script_name}_dir")
            
            # Create timestamped backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"backup_{timestamp}"
            
            # Copy latest results
            shutil.copytree(source_dir, backup_path)
            
            logging.info(f"Created backup of {script_name} results at {backup_path}")
            return backup_path
            
        except Exception as e:
            logging.error(f"Failed to create backup: {e}")
            return None
    
    def load_latest_results(self, script_name: str) -> Optional[Dict]:
        """
        Load the latest results for a script.
        
        Args:
            script_name: Name of the script
            
        Returns:
            Dictionary containing the results or None if not found
        """
        try:
            results_dir = getattr(self, f"{script_name}_dir")
            latest_results = self.get_latest_file(results_dir, "analysis_results", ".json")
            
            if not latest_results:
                logging.warning(f"No latest results found for {script_name}")
                return None
            
            with open(latest_results, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            logging.error(f"Error loading latest results for {script_name}: {e}")
            return None
    
    def get_timestamped_path(self, directory: Path, prefix: str, suffix: str) -> Path:
        """
        Get a timestamped path for a new file.
        
        Args:
            directory: Directory to create the file in
            prefix: Prefix for the filename
            suffix: File extension (e.g., '.json', '.txt')
            
        Returns:
            Path object for the new file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return directory / f"{prefix}_{timestamp}{suffix}"
    
    def create_latest_symlink(self, target_file: Path, link_name: str) -> None:
        """
        Create or update a symbolic link to the latest version of a file.
        
        Args:
            target_file: Path to the file to link to
            link_name: Name for the symbolic link
        """
        link_path = target_file.parent / link_name
        if link_path.exists():
            link_path.unlink()
        link_path.symlink_to(target_file.name)
        logging.debug(f"Updated symlink {link_path} -> {target_file.name}")
    
    def save_json_with_symlink(self, data: dict, directory: Path, prefix: str) -> Path:
        """
        Save JSON data with a timestamp and update the 'latest' symlink.
        
        Args:
            data: Dictionary to save as JSON
            directory: Directory to save in
            prefix: Prefix for the filename
            
        Returns:
            Path to the saved file
        """
        # Create timestamped file
        file_path = self.get_timestamped_path(directory, prefix, ".json")
        
        # Save JSON with proper formatting
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Update latest symlink
        self.create_latest_symlink(file_path, f"latest_{prefix}.json")
        
        logging.info(f"Saved JSON file: {file_path}")
        return file_path
    
    def save_text_with_symlink(self, text: str, directory: Path, prefix: str) -> Path:
        """
        Save text data with a timestamp and update the 'latest' symlink.
        
        Args:
            text: Text to save
            directory: Directory to save in
            prefix: Prefix for the filename
            
        Returns:
            Path to the saved file
        """
        # Create timestamped file
        file_path = self.get_timestamped_path(directory, prefix, ".txt")
        
        # Save text
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        # Update latest symlink
        self.create_latest_symlink(file_path, f"latest_{prefix}.txt")
        
        logging.info(f"Saved text file: {file_path}")
        return file_path
    
    def get_latest_file(self, directory: Path, prefix: str, suffix: str) -> Optional[Path]:
        """
        Get the path to the latest version of a file.
        
        Args:
            directory: Directory to look in
            prefix: Prefix of the filename
            suffix: File extension
            
        Returns:
            Path to the latest file or None if not found
        """
        link_path = directory / f"latest_{prefix}{suffix}"
        return link_path if link_path.exists() else None
    
    def cleanup_old_files(self, directory: Path, retention_days: int) -> None:
        """
        Remove files older than the specified retention period.
        
        Args:
            directory: Directory containing files to clean
            retention_days: Number of days to keep files
        """
        cutoff_time = datetime.now().timestamp() - (retention_days * 24 * 60 * 60)
        
        for file_path in directory.glob("*"):
            # Skip symlinks and directories
            if file_path.is_symlink() or file_path.is_dir():
                continue
            
            # Remove if older than retention period
            if file_path.stat().st_mtime < cutoff_time:
                try:
                    file_path.unlink()
                    logging.debug(f"Removed old file: {file_path}")
                except Exception as e:
                    logging.warning(f"Failed to remove old file {file_path}: {e}")

# Create a global instance for use across scripts
file_manager = FileManager() 