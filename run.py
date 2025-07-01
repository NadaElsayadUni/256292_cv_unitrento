#!/usr/bin/env python3
"""
Main run script for Human Motion Analysis Package
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src import HumanMotionAnalyzer
from src.config.settings import *


def run_simple_analysis():
    """Run simple analysis without occlusion tracking"""
    print("🚀 Running Simple Analysis")
    print("=" * 30)

    # Setup paths
    base_path = Path("Videos and Annotations")
    video_path = base_path / "video0" / "video.mp4"
    reference_path = base_path / "video0" / "reference.jpeg"
    annotations_path = base_path / "video0" / "annotations.txt"
    # video_path = base_path / "video3" / "video.mp4"
    # reference_path = base_path / "video3" / "reference.jpg"
    # annotations_path = base_path / "video3" / "annotations.txt"

    # Check if files exist
    if not video_path.exists():
        print(f"❌ Error: Video file not found at {video_path}")
        return
    if not reference_path.exists():
        print(f"❌ Error: Reference file not found at {reference_path}")
        return
    if not annotations_path.exists():
        print(f"❌ Error: Annotations file not found at {annotations_path}")
        return

    print(f"📹 Video: {video_path}")
    print(f"🖼️  Reference: {reference_path}")
    print(f"📝 Annotations: {annotations_path}")

    # Initialize analyzer
    print("\n🔧 Initializing analyzer...")
    analyzer = HumanMotionAnalyzer(
        video_path=str(video_path),
        reference_path=str(reference_path),
        annotations_path=str(annotations_path)
    )

    # Run analysis
    print("\n🎯 Running analysis...")
    accuracy_metrics = analyzer.run_analysis(
        show_visualization=True,
        save_results=True
    )

    # Display results
    if accuracy_metrics:
        print("\n✅ Analysis Complete!")
        print(f"📊 Total objects: {accuracy_metrics['total_objects']}")
        print(f"🎯 Tracked objects: {accuracy_metrics['matched_objects']}")
        print(f"📈 Success rate: {accuracy_metrics['matched_objects']/accuracy_metrics['total_objects']*100:.1f}%")
        print(f"📏 Average IoU: {accuracy_metrics['average_iou']:.3f}")
        print(f"📐 Average MSE: {accuracy_metrics['average_mse']:.3f}")

        print("\n📁 Generated files:")
        print("  - trajectories.txt: Raw trajectory data")
        print("  - trajectories_raw.jpg: Raw trajectory visualization")
        print("  - trajectories_kalman.jpg: Kalman-filtered visualization")
        print("  - last_frame.jpg: Last processed frame")
        print("  - detailed_metrics_analysis.txt: Detailed metrics analysis")
        print("  - final_analysis_results.txt: Final analysis summary")
        print("  - detected_trajectories/: Individual trajectory images for each object")
    else:
        print("❌ Analysis failed")


def run_occlusion_analysis():
    """Run analysis with occlusion tracking"""
    print("🚀 Running Analysis with Occlusion Tracking")
    print("=" * 45)
    
    # Setup paths
    base_path = Path("Videos and Annotations")
    video_path = base_path / "video0" / "masked" / "masked_video_0.mp4"
    reference_path = base_path / "video0" / "reference.jpeg"
    masked_reference_path = base_path / "video0" / "masked" / "masked_reference_0.jpeg"
    annotations_path = base_path / "video0" / "annotations.txt"
    # video_path = base_path / "video3" / "masked" / "masked_video_3.mp4"
    # reference_path = base_path / "video3" / "reference.jpg"
    # masked_reference_path = base_path / "video3" / "masked" / "masked_reference_3.jpeg"
    # annotations_path = base_path / "video3" / "annotations.txt"

    # Check if files exist
    if not video_path.exists():
        print(f"❌ Error: Video file not found at {video_path}")
        return
    if not reference_path.exists():
        print(f"❌ Error: Reference file not found at {reference_path}")
        return
    if not masked_reference_path.exists():
        print(f"❌ Error: Masked reference file not found at {masked_reference_path}")
        return
    if not annotations_path.exists():
        print(f"❌ Error: Annotations file not found at {annotations_path}")
        return

    print(f"📹 Video: {video_path}")
    print(f"🖼️  Reference: {reference_path}")
    print(f"🎭 Masked Reference: {masked_reference_path}")
    print(f"📝 Annotations: {annotations_path}")

    # Initialize analyzer
    print("\n🔧 Initializing analyzer...")
    analyzer = HumanMotionAnalyzer(
        video_path=str(video_path),
        reference_path=str(reference_path),
        annotations_path=str(annotations_path)
    )

    # Run analysis with occlusion tracking

    print("\n🎯 Running analysis with occlusion tracking...")
    accuracy_metrics = analyzer.run_analysis_with_occlusion_tracking(
        with_occlusion_path=str(reference_path),
        without_occlusion_path=str(masked_reference_path),
        show_visualization=True,
        save_results=True
    )
    
    # Display results
    if accuracy_metrics:
        print("\n✅ Analysis Complete!")
        print(f"📊 Total objects: {accuracy_metrics['total_objects']}")
        print(f"🎯 Tracked objects: {accuracy_metrics['matched_objects']}")
        print(f"📈 Success rate: {accuracy_metrics['matched_objects']/accuracy_metrics['total_objects']*100:.1f}%")
        print(f"📏 Average IoU: {accuracy_metrics['average_iou']:.3f}")
        print(f"📐 Average MSE: {accuracy_metrics['average_mse']:.3f}")

        print("\n📁 Generated files:")
        print("  - trajectories.txt: Raw trajectory data")
        print("  - trajectories_raw.jpg: Raw trajectory visualization")
        print("  - trajectories_kalman.jpg: Kalman-filtered visualization")
        print("  - last_frame.jpg: Last processed frame")
        print("  - aligned_occlusion_mask.png: Computed occlusion mask")
        print("  - detailed_metrics_analysis.txt: Detailed metrics analysis")
        print("  - final_analysis_results.txt: Final analysis summary")
        print("  - detected_trajectories/: Individual trajectory images for each object")
    else:
        print("❌ Analysis failed")


def run_path_analysis():
    """Run trajectory and path pattern analysis on ground truth data"""
    print("🚀 Running Trajectory and Path Pattern Analysis")
    print("=" * 50)

    # Setup paths
    base_path = Path("Videos and Annotations")
    # annotations_path = base_path / "video0" / "annotations.txt"
    # reference_image_path = base_path / "video0" / "reference.jpeg"
    annotations_path = base_path / "video3" / "annotations.txt"
    reference_image_path = base_path / "video3" / "reference.jpg"

    # Check if files exist
    if not annotations_path.exists():
        print(f"❌ Error: Annotations file not found at {annotations_path}")
        return
    if not reference_image_path.exists():
        print(f"❌ Error: Reference image not found at {reference_image_path}")
        return

    print(f"📝 Annotations: {annotations_path}")
    print(f"🖼️  Reference: {reference_image_path}")

    # Import analysis functions
    from src.data.annotations import (
        read_annotations_file,
        analyze_trajectories,
        analyze_path_patterns,
        visualize_path_patterns
    )
    from src.utils.file_utils import save_path_analysis_results

    # Read trajectories
    print("\n📖 Reading annotations...")
    trajectories, class_counts, frame_distribution = read_annotations_file(str(annotations_path))

    if not trajectories:
        print("❌ Error: Could not read trajectories from annotations file")
        return

    print(f"✅ Successfully loaded {len(trajectories)} trajectories")

    # Analyze trajectories and get statistics
    print("\n📊 Running trajectory analysis...")
    trajectory_stats = analyze_trajectories(trajectories, class_counts, frame_distribution)

    # Analyze path patterns
    print("\n🛤️  Running path pattern analysis...")
    path_patterns, detailed_patterns = analyze_path_patterns(trajectories, str(reference_image_path), trajectory_stats)


    if path_patterns:
        # Create visualizations
        print("\n🎨 Creating path patterns visualization...")
        visualize_path_patterns(path_patterns, PATH_PATTERNS_GRAPH)

        # Save analysis results to file
        print("\n💾 Saving analysis results to file...")
        save_path_analysis_results(trajectory_stats, path_patterns, detailed_patterns, PATH_ANALYSIS_FILE)


        print("\n✅ Path Analysis Complete!")
        print("\n📁 Generated files:")
        print(f"  - {PATH_PATTERNS_GRAPH}: Path patterns visualization")
        print(f"  - {PATH_ANALYSIS_FILE}: Detailed analysis results")
    else:
        print("❌ No clear path patterns detected")


def show_help():
    """Show help information"""
    print("""
🎯 Human Motion Analysis Package - Run Script

USAGE:
    python run.py [option]

OPTIONS:
    1, simple     Run simple analysis (no occlusion tracking)
    2, occlusion  Run analysis with occlusion tracking (recommended)
    3, path       Run trajectory and path pattern analysis on ground truth data
    help, -h      Show this help message

EXAMPLES:
    python run.py 1
    python run.py simple
    python run.py occlusion
    python run.py path

REQUIREMENTS:
    - Python 3.6+
    - OpenCV (opencv-python)
    - NumPy
    - Matplotlib
    - NetworkX
    - Pathlib

FILES NEEDED:
    - Videos and Annotations/video0/masked/masked_video_0.mp4
    - Videos and Annotations/video0/reference.jpeg
    - Videos and Annotations/video0/masked/masked_reference_0.jpeg
    - Videos and Annotations/video0/annotations.txt

OUTPUT:
    All results are saved to the 'output/' folder

TIMING:
    The script now includes timing information for each analysis phase
    """)


def main():
    """Main function"""
    print("🎯 Human Motion Analysis Package")
    print("=" * 40)

    # Get command line argument
    if len(sys.argv) > 1:
        choice = sys.argv[1].lower()
    else:
        # Interactive mode
        print("\nChoose an option:")
        print("1. Simple analysis (no occlusion tracking)")
        print("2. Analysis with occlusion tracking (Bonus)")
        print("3. Trajectory and path pattern analysis Based on ground truth data")
        print("4. Show help")

        choice = input("\nEnter your choice (1-4): ").strip()

    # Execute based on choice
    if choice in ['1', 'simple']:
        run_simple_analysis()
    elif choice in ['2', 'occlusion']:
        run_occlusion_analysis()
    elif choice in ['3', 'path']:
        run_path_analysis()
    elif choice in ['4', 'help', '-h', '--help']:
        show_help()
    else:
        print("❌ Invalid choice. Use 'python run.py help' for options.")
        show_help()


if __name__ == "__main__":
    main()