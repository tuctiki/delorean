import sys
import os

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from delorean.pipeline import DailyPipeline

def main():
    pipeline = DailyPipeline()
    pipeline.run()

if __name__ == "__main__":
    main()
