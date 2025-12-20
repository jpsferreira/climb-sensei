"""Repository pattern for persisting analysis results.

This module provides implementations of the AnalysisRepository protocol
for saving and loading climbing analysis results in various formats.
"""

import json
import csv
from pathlib import Path
from typing import Union, List, Dict, Any

from .models import ClimbingAnalysis, ClimbingSummary
from .protocols import AnalysisRepository


class JSONRepository(AnalysisRepository):
    """Repository for saving/loading analysis results as JSON.

    This repository saves the complete analysis including summary statistics
    and frame-by-frame history in a human-readable JSON format.

    Example:
        >>> repo = JSONRepository("results/")
        >>> repo.save(analysis, "climb_001.json")
        >>> loaded = repo.load("climb_001.json")
    """

    def __init__(self, base_path: Union[str, Path] = "."):
        """Initialize JSON repository.

        Args:
            base_path: Base directory for storing JSON files
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        analysis: ClimbingAnalysis,
        filename: str = "analysis.json",
        indent: int = 2,
    ) -> Path:
        """Save analysis to JSON file.

        Args:
            analysis: ClimbingAnalysis to save
            filename: Output filename (relative to base_path)
            indent: JSON indentation level (default 2 for readability)

        Returns:
            Path to the saved file
        """
        output_path = self.base_path / filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis.to_dict(), f, indent=indent, ensure_ascii=False)

        return output_path

    def load(self, filename: str) -> ClimbingAnalysis:
        """Load analysis from JSON file.

        Args:
            filename: Input filename (relative to base_path)

        Returns:
            ClimbingAnalysis loaded from file

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file is not valid JSON
        """
        input_path = self.base_path / filename

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return ClimbingAnalysis.from_dict(data)

    def __repr__(self) -> str:
        """String representation."""
        return f"JSONRepository(base_path={self.base_path})"


class CSVRepository(AnalysisRepository):
    """Repository for exporting analysis results to CSV.

    This repository exports summary statistics and frame-by-frame metrics
    to CSV format, suitable for data analysis in spreadsheets or pandas.

    Two files are created:
    - {filename}_summary.csv: Summary statistics
    - {filename}_frames.csv: Frame-by-frame metrics

    Example:
        >>> repo = CSVRepository("results/")
        >>> repo.save(analysis, "climb_001")
        # Creates climb_001_summary.csv and climb_001_frames.csv
    """

    def __init__(self, base_path: Union[str, Path] = "."):
        """Initialize CSV repository.

        Args:
            base_path: Base directory for storing CSV files
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        analysis: ClimbingAnalysis,
        filename: str = "analysis",
    ) -> tuple[Path, Path]:
        """Save analysis to CSV files.

        Args:
            analysis: ClimbingAnalysis to save
            filename: Base filename (without extension)

        Returns:
            Tuple of (summary_path, frames_path)
        """
        # Save summary statistics
        summary_path = self._save_summary(analysis.summary, filename)

        # Save frame-by-frame history
        frames_path = self._save_frames(analysis.history, filename)

        return summary_path, frames_path

    def _save_summary(
        self,
        summary: ClimbingSummary,
        filename: str,
    ) -> Path:
        """Save summary statistics to CSV.

        Args:
            summary: ClimbingSummary to save
            filename: Base filename

        Returns:
            Path to summary CSV file
        """
        output_path = self.base_path / f"{filename}_summary.csv"
        summary_dict = summary.to_dict()

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            for key, value in summary_dict.items():
                writer.writerow([key, value])

        return output_path

    def _save_frames(
        self,
        history: Dict[str, List[Any]],
        filename: str,
    ) -> Path:
        """Save frame-by-frame metrics to CSV.

        Args:
            history: Frame history dictionary
            filename: Base filename

        Returns:
            Path to frames CSV file
        """
        output_path = self.base_path / f"{filename}_frames.csv"

        # Get all metric names (column headers)
        if not history:
            # Empty history - create empty file
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["frame"])
            return output_path

        # Determine number of frames
        num_frames = max(len(v) for v in history.values() if isinstance(v, list))

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Write header
            headers = ["frame"] + sorted(history.keys())
            writer.writerow(headers)

            # Write data
            for i in range(num_frames):
                row = [i]
                for key in sorted(history.keys()):
                    values = history[key]
                    if isinstance(values, list) and i < len(values):
                        row.append(values[i])
                    else:
                        row.append("")
                writer.writerow(row)

        return output_path

    def load(self, filename: str) -> ClimbingAnalysis:
        """Load analysis from CSV files.

        Note: This is a simplified implementation that only loads the summary.
        Full frame-by-frame history reconstruction from CSV is not implemented.

        Args:
            filename: Base filename (without _summary.csv suffix)

        Returns:
            ClimbingAnalysis with summary (history will be empty)

        Raises:
            FileNotFoundError: If summary file doesn't exist
        """
        summary_path = self.base_path / f"{filename}_summary.csv"

        summary_dict = {}
        with open(summary_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) == 2:
                    key, value = row
                    # Try to convert to appropriate type
                    try:
                        if "." in value:
                            summary_dict[key] = float(value)
                        else:
                            summary_dict[key] = int(value)
                    except ValueError:
                        summary_dict[key] = value

        summary = ClimbingSummary.from_dict(summary_dict)

        return ClimbingAnalysis(
            summary=summary,
            history={},
            video_path=None,
        )

    def __repr__(self) -> str:
        """String representation."""
        return f"CSVRepository(base_path={self.base_path})"
