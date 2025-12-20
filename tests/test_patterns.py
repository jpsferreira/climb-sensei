"""Tests for Phase 2: Facade, Builder, and Repository patterns."""

import pytest
import json
import csv
from unittest.mock import Mock

from climb_sensei import (
    ClimbingSensei,
    ClimbingAnalyzerBuilder,
    JSONRepository,
    CSVRepository,
    ClimbingAnalysis,
    ClimbingSummary,
    ClimbingAnalyzer,
    MetricsConfig,
)


class TestClimbingAnalyzerBuilder:
    """Test the Builder pattern for ClimbingAnalyzer."""

    def test_default_build(self):
        """Test building with default values."""
        builder = ClimbingAnalyzerBuilder()
        analyzer = builder.build()

        assert isinstance(analyzer, ClimbingAnalyzer)
        assert analyzer.window_size == 30
        assert analyzer.fps == 30.0

    def test_fluent_api(self):
        """Test fluent chaining of builder methods."""
        analyzer = ClimbingAnalyzerBuilder().with_window_size(60).with_fps(60.0).build()

        assert analyzer.window_size == 60
        assert analyzer.fps == 60.0

    def test_with_window_size(self):
        """Test setting window size."""
        analyzer = ClimbingAnalyzerBuilder().with_window_size(45).build()
        assert analyzer.window_size == 45

    def test_with_fps(self):
        """Test setting fps."""
        analyzer = ClimbingAnalyzerBuilder().with_fps(25.0).build()
        assert analyzer.fps == 25.0

    def test_with_config(self):
        """Test setting complete config."""
        config = MetricsConfig(lock_off_threshold_degrees=85.0)
        builder = ClimbingAnalyzerBuilder().with_config(config)
        # Just verify builder accepts config
        assert builder._config.lock_off_threshold_degrees == 85.0

    def test_with_individual_thresholds(self):
        """Test setting individual threshold values."""
        builder = (
            ClimbingAnalyzerBuilder()
            .with_velocity_threshold(0.12)
            .with_sway_threshold(0.08)
        )

        # Verify builder stores these values
        assert builder._velocity_threshold == 0.12
        assert builder._sway_threshold == 0.08

        # Build still works
        analyzer = builder.build()
        assert isinstance(analyzer, ClimbingAnalyzer)

    def test_invalid_window_size(self):
        """Test that invalid window size raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            ClimbingAnalyzerBuilder().with_window_size(0)

    def test_invalid_fps(self):
        """Test that invalid fps raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            ClimbingAnalyzerBuilder().with_fps(-10.0)

    def test_reset(self):
        """Test resetting builder to defaults."""
        builder = (
            ClimbingAnalyzerBuilder().with_window_size(100).with_fps(120.0).reset()
        )

        analyzer = builder.build()
        assert analyzer.window_size == 30  # Back to default
        assert analyzer.fps == 30.0

    def test_builder_repr(self):
        """Test string representation."""
        builder = ClimbingAnalyzerBuilder().with_window_size(45)
        repr_str = repr(builder)
        assert "ClimbingAnalyzerBuilder" in repr_str
        assert "45" in repr_str


class TestClimbingSenseiFacade:
    """Test the Facade pattern for simplified API."""

    @pytest.fixture
    def mock_video_file(self, tmp_path):
        """Create a mock video file."""
        video_path = tmp_path / "test.mp4"
        video_path.write_text("fake video")
        return str(video_path)

    def test_init(self, mock_video_file):
        """Test facade initialization."""
        sensei = ClimbingSensei(mock_video_file)
        assert sensei.video_path.name == "test.mp4"
        assert sensei.window_size == 30
        assert sensei.fps == 30.0

    def test_custom_init(self, mock_video_file):
        """Test facade with custom parameters."""
        sensei = ClimbingSensei(
            mock_video_file,
            window_size=60,
            fps=60.0,
        )
        assert sensei.window_size == 60
        assert sensei.fps == 60.0

    def test_lazy_loading(self, mock_video_file):
        """Test that pose engine and analyzer are lazy-loaded."""
        sensei = ClimbingSensei(mock_video_file)
        assert sensei._pose_engine is None
        assert sensei._analyzer is None

        # Access triggers loading
        _ = sensei.pose_engine
        assert sensei._pose_engine is not None

        _ = sensei.analyzer
        assert sensei._analyzer is not None

    def test_context_manager(self, mock_video_file):
        """Test using facade as context manager."""
        with ClimbingSensei(mock_video_file) as sensei:
            assert sensei.video_path.exists()
        # Should close resources after context

    def test_missing_video(self):
        """Test that missing video file raises error."""
        sensei = ClimbingSensei("nonexistent.mp4")
        with pytest.raises(FileNotFoundError):
            sensei.analyze()

    def test_reset(self, mock_video_file):
        """Test resetting analyzer state."""
        sensei = ClimbingSensei(mock_video_file)
        sensei._analysis = Mock()  # Simulate analysis
        sensei._analyzer = Mock()

        sensei.reset()
        assert sensei._analysis is None
        sensei._analyzer.reset.assert_called_once()

    def test_get_analysis_before_analyze(self, mock_video_file):
        """Test getting analysis before running returns None."""
        sensei = ClimbingSensei(mock_video_file)
        assert sensei.get_analysis() is None
        assert sensei.get_summary() is None

    def test_repr(self, mock_video_file):
        """Test string representation."""
        sensei = ClimbingSensei(mock_video_file)
        repr_str = repr(sensei)
        assert "ClimbingSensei" in repr_str
        assert "test.mp4" in repr_str
        assert "not analyzed" in repr_str

    def test_close_cleanup(self, mock_video_file):
        """Test that close releases resources."""
        sensei = ClimbingSensei(mock_video_file)
        mock_engine = Mock()
        sensei._pose_engine = mock_engine

        sensei.close()
        mock_engine.close.assert_called_once()
        assert sensei._pose_engine is None


class TestJSONRepository:
    """Test JSON repository for analysis persistence."""

    @pytest.fixture
    def temp_repo(self, tmp_path):
        """Create a temporary JSON repository."""
        return JSONRepository(tmp_path)

    @pytest.fixture
    def sample_analysis(self):
        """Create a sample analysis for testing."""
        summary = ClimbingSummary(
            total_frames=100,
            total_vertical_progress=2.5,
            max_height=2.8,
            avg_velocity=0.05,
            max_velocity=0.15,
            avg_sway=0.01,
            max_sway=0.03,
            avg_jerk=10.0,
            max_jerk=30.0,
            avg_body_angle=15.0,
            avg_hand_span=0.4,
            avg_foot_span=0.3,
            total_distance_traveled=3.2,
            avg_movement_economy=0.78,
            lock_off_count=25,
            lock_off_percentage=25.0,
            rest_count=10,
            rest_percentage=10.0,
            fatigue_score=0.15,
            avg_left_elbow=145.0,
            avg_right_elbow=143.0,
            avg_left_shoulder=95.0,
            avg_right_shoulder=93.0,
            avg_left_knee=168.0,
            avg_right_knee=170.0,
            avg_left_hip=120.0,
            avg_right_hip=122.0,
        )
        return ClimbingAnalysis(
            summary=summary,
            history={"velocities": [0.01, 0.02, 0.03]},
            video_path="test.mp4",
        )

    def test_init_creates_directory(self, tmp_path):
        """Test that repository creates base directory."""
        repo_path = tmp_path / "new_repo"
        JSONRepository(repo_path)
        assert repo_path.exists()

    def test_save(self, temp_repo, sample_analysis):
        """Test saving analysis to JSON."""
        output_path = temp_repo.save(sample_analysis, "test.json")
        assert output_path.exists()

        # Verify JSON content
        with open(output_path) as f:
            data = json.load(f)

        assert data["video_path"] == "test.mp4"
        assert data["summary"]["total_frames"] == 100
        assert "velocities" in data["history"]

    def test_load(self, temp_repo, sample_analysis):
        """Test loading analysis from JSON."""
        temp_repo.save(sample_analysis, "test.json")
        loaded = temp_repo.load("test.json")

        assert isinstance(loaded, ClimbingAnalysis)
        assert loaded.summary.total_frames == 100
        assert loaded.video_path == "test.mp4"

    def test_save_load_roundtrip(self, temp_repo, sample_analysis):
        """Test that save/load preserves data."""
        temp_repo.save(sample_analysis, "roundtrip.json")
        loaded = temp_repo.load("roundtrip.json")

        # Compare summaries
        assert loaded.summary.total_frames == sample_analysis.summary.total_frames
        assert loaded.summary.max_height == sample_analysis.summary.max_height
        assert loaded.summary.avg_velocity == sample_analysis.summary.avg_velocity

    def test_load_missing_file(self, temp_repo):
        """Test loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            temp_repo.load("missing.json")

    def test_repr(self, temp_repo):
        """Test string representation."""
        repr_str = repr(temp_repo)
        assert "JSONRepository" in repr_str


class TestCSVRepository:
    """Test CSV repository for analysis export."""

    @pytest.fixture
    def temp_repo(self, tmp_path):
        """Create a temporary CSV repository."""
        return CSVRepository(tmp_path)

    @pytest.fixture
    def sample_analysis(self):
        """Create a sample analysis for testing."""
        summary = ClimbingSummary(
            total_frames=50,
            total_vertical_progress=2.0,
            max_height=2.5,
            avg_velocity=0.04,
            max_velocity=0.12,
            avg_sway=0.02,
            max_sway=0.04,
            avg_jerk=8.0,
            max_jerk=25.0,
            avg_body_angle=12.0,
            avg_hand_span=0.35,
            avg_foot_span=0.28,
            total_distance_traveled=2.8,
            avg_movement_economy=0.75,
            lock_off_count=20,
            lock_off_percentage=40.0,
            rest_count=5,
            rest_percentage=10.0,
            fatigue_score=0.12,
            avg_left_elbow=140.0,
            avg_right_elbow=142.0,
            avg_left_shoulder=90.0,
            avg_right_shoulder=92.0,
            avg_left_knee=165.0,
            avg_right_knee=167.0,
            avg_left_hip=118.0,
            avg_right_hip=120.0,
        )
        return ClimbingAnalysis(
            summary=summary,
            history={
                "hip_heights": [0.5, 0.6, 0.7],
                "velocities": [0.01, 0.02, 0.03],
            },
            video_path="test.mp4",
        )

    def test_save_creates_two_files(self, temp_repo, sample_analysis):
        """Test that save creates summary and frames CSV files."""
        summary_path, frames_path = temp_repo.save(sample_analysis, "test")

        assert summary_path.exists()
        assert frames_path.exists()
        assert summary_path.name == "test_summary.csv"
        assert frames_path.name == "test_frames.csv"

    def test_summary_csv_format(self, temp_repo, sample_analysis):
        """Test summary CSV has correct format."""
        summary_path, _ = temp_repo.save(sample_analysis, "test")

        with open(summary_path) as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Check header
        assert rows[0] == ["Metric", "Value"]

        # Check some values
        metric_dict = {row[0]: row[1] for row in rows[1:]}
        assert "total_frames" in metric_dict
        assert metric_dict["total_frames"] == "50"

    def test_frames_csv_format(self, temp_repo, sample_analysis):
        """Test frames CSV has correct format."""
        _, frames_path = temp_repo.save(sample_analysis, "test")

        with open(frames_path) as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Check header includes 'frame' and metric names
        assert "frame" in rows[0]
        assert "hip_heights" in rows[0]
        assert "velocities" in rows[0]

        # Check data rows
        assert len(rows) == 4  # header + 3 data rows
        assert rows[1][0] == "0"  # First frame index

    def test_load_summary_only(self, temp_repo, sample_analysis):
        """Test loading analysis from CSV (summary only)."""
        temp_repo.save(sample_analysis, "test")
        loaded = temp_repo.load("test")

        assert isinstance(loaded, ClimbingAnalysis)
        assert loaded.summary.total_frames == 50
        assert loaded.history == {}  # History not loaded from CSV

    def test_empty_history(self, temp_repo):
        """Test saving analysis with empty history."""
        summary = ClimbingSummary(
            total_frames=10,
            total_vertical_progress=1.0,
            max_height=1.2,
            avg_velocity=0.03,
            max_velocity=0.08,
            avg_sway=0.01,
            max_sway=0.02,
            avg_jerk=5.0,
            max_jerk=15.0,
            avg_body_angle=10.0,
            avg_hand_span=0.3,
            avg_foot_span=0.25,
            total_distance_traveled=1.5,
            avg_movement_economy=0.7,
            lock_off_count=5,
            lock_off_percentage=50.0,
            rest_count=2,
            rest_percentage=20.0,
            fatigue_score=0.1,
            avg_left_elbow=135.0,
            avg_right_elbow=137.0,
            avg_left_shoulder=88.0,
            avg_right_shoulder=90.0,
            avg_left_knee=160.0,
            avg_right_knee=162.0,
            avg_left_hip=115.0,
            avg_right_hip=117.0,
        )
        analysis = ClimbingAnalysis(summary=summary, history={}, video_path=None)

        _, frames_path = temp_repo.save(analysis, "empty")
        assert frames_path.exists()

    def test_repr(self, temp_repo):
        """Test string representation."""
        repr_str = repr(temp_repo)
        assert "CSVRepository" in repr_str


class TestProtocolConformance:
    """Test that new classes conform to their protocols."""

    def test_json_repository_conforms(self, tmp_path):
        """Test JSONRepository conforms to AnalysisRepository protocol."""
        from climb_sensei.protocols import AnalysisRepository

        repo = JSONRepository(tmp_path)
        assert isinstance(repo, AnalysisRepository)

    def test_csv_repository_conforms(self, tmp_path):
        """Test CSVRepository conforms to AnalysisRepository protocol."""
        from climb_sensei.protocols import AnalysisRepository

        repo = CSVRepository(tmp_path)
        assert isinstance(repo, AnalysisRepository)


class TestIntegration:
    """Integration tests combining Phase 2 patterns."""

    def test_builder_with_facade(self, tmp_path):
        """Test using builder pattern with facade."""
        # Create a mock video file
        video_path = tmp_path / "test.mp4"
        video_path.write_text("fake")

        # Build custom analyzer
        analyzer = ClimbingAnalyzerBuilder().with_window_size(45).with_fps(25.0).build()

        # Could inject into facade if we added that feature
        assert analyzer.window_size == 45

    def test_facade_with_repository(self, tmp_path):
        """Test analyzing with facade and saving with repository."""
        # This is a conceptual test - would need real video for full test
        repo = JSONRepository(tmp_path)
        assert repo.base_path == tmp_path

        # In practice: sensei.analyze() -> repo.save(analysis)
