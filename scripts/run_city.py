import argparse
from poi.pipeline import analyze_city

def main():
    ap = argparse.ArgumentParser(description="Run mobility metrics pipeline for a city")
    ap.add_argument("--config", required=True, help="Path to YAML config, e.g. configs/nyc.yaml")
    args = ap.parse_args()
    analyze_city(args.config)

if __name__ == "__main__":
    main()