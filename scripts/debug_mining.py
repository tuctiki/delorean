
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from research.mine_alphas_gp import main, POPULATION_SIZE, GENERATIONS

# Monkey patch config for speed
import research.mine_alphas_gp
research.mine_alphas_gp.POPULATION_SIZE = 5
research.mine_alphas_gp.GENERATIONS = 1

if __name__ == "__main__":
    print("Running Debug Mining...")
    main()
