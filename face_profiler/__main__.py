import argparse
import json
import sys


def _print_text(results, image_path):
    count = len(results)
    print(f"\nFound {count} face{'s' if count != 1 else ''} in {image_path}\n")
    for i, r in enumerate(results, 1):
        print(f"Face {i} (confidence: {r['confidence']:.0%})")
        print(f"  Box:      {r['box']}")
        print(f"  Age:      {r['age']:.1f} years (range: {r['age_range']})")
        print(f"  Gender:   {r['gender']} ({r['gender_confidence']:.0f}% confidence)")
        race = r['race'].replace('_', ' ')
        race_conf = r['race_distribution'][r['race']]
        print(f"  Race:     {race} ({race_conf:.0f}% confidence)")
        print()


def main():
    parser = argparse.ArgumentParser(
        prog="face-profiler",
        description="Face detection and attribute analysis (age, gender, race)",
    )
    parser.add_argument("image", nargs="?", help="Path to image file")
    parser.add_argument("--text", action="store_true", help="Human-readable text output (default is JSON)")
    parser.add_argument("--annotate", metavar="FILE", help="Save annotated image to FILE")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="Force compute device")
    parser.add_argument("--gui", action="store_true", help="Launch the GUI demo")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress model loading messages")
    args = parser.parse_args()

    if args.gui:
        from face_profiler.gui import main as gui_main
        gui_main()
        return

    if not args.image:
        parser.error("image path required (or use --gui)")

    import os
    if not os.path.isfile(args.image):
        print(f"Error: file not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    # Suppress print output during model loading if --quiet
    if args.quiet:
        import io
        _real_stdout = sys.stdout
        sys.stdout = io.StringIO()

    from face_profiler import FaceProfiler
    profiler = FaceProfiler(device=args.device)
    results = profiler.analyze(args.image)

    if args.quiet:
        sys.stdout = _real_stdout

    if args.text:
        _print_text(results, args.image)
    else:
        # Convert tuples to lists for JSON serialization
        json_results = []
        for r in results:
            jr = dict(r)
            jr["box"] = list(jr["box"])
            json_results.append(jr)
        print(json.dumps(json_results, indent=2))

    if args.annotate:
        annotated = profiler.render(args.image, results)
        annotated.save(args.annotate)
        if not args.quiet:
            print(f"Annotated image saved to {args.annotate}", file=sys.stderr)


if __name__ == "__main__":
    main()
