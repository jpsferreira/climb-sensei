#!/usr/bin/env python3
"""Launch the ClimbingSensei web app."""

import sys
from pathlib import Path

# Add parent src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import and run
import uvicorn

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🧗 ClimbingSensei Web App")
    print("=" * 60)
    print("\n📹 Two-Phase API Demo:")
    print("  1. Upload a climbing video")
    print("  2. Select analysis options")
    print("  3. Extract landmarks once (Phase 1)")
    print("  4. Generate outputs in parallel (Phase 2)")
    print("\n🚀 Starting server at http://localhost:8000")
    print("Press CTRL+C to stop\n")

    uvicorn.run(
        "app.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
