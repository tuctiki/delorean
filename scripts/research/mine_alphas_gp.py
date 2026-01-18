
import random
import copy
import numpy as np
import pandas as pd
import qlib
from qlib.data import D
import sys
import os
from typing import List, Union

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from delorean.config import QLIB_PROVIDER_URI, QLIB_REGION, ETF_LIST

# --- CONFIGURATION ---
POPULATION_SIZE = 60
GENERATIONS = 6
TOURNAMENT_SIZE = 3
CROSSOVER_PROB = 0.6
MUTATION_PROB = 0.3
MAX_DEPTH = 4
HOF_SIZE = 10  # Hall of Fame

# Qlib Operators
UNARY_OPS = [
    "Abs", "Log", "Neg", "Inv", "Sign", "Rank"
]
BINARY_OPS = [
    # (Name, Arity) - Arity 2 (Field, Field)
    "Add", "Sub", "Mul", "Div", "Max", "Min"
]
TS_OPS = [
    # (Name, Arity) - Arity 2 (Field, Window)
    "Ref", "Mean", "Std", "Max", "Min", "Sum", "TsRank", "Decay", "Corr", "Cov"
]
# We map them to Qlib strings: F(A), F(A, B), F(A, d)

FIELDS = ["$close", "$open", "$high", "$low", "$volume", "$vwap"]
WINDOWS = [5, 10, 20, 40, 60]

class Node:
    def __str__(self):
        raise NotImplementedError

class Field(Node):
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return self.name

class Constant(Node):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return str(self.value)

class Op(Node):
    def __init__(self, name, args):
        self.name = name
        self.args = args
    
    def __str__(self):
        args_str = ", ".join([str(a) for a in self.args])
        if self.name == "Add": return f"({self.args[0]} + {self.args[1]})"
        if self.name == "Sub": return f"({self.args[0]} - {self.args[1]})"
        if self.name == "Mul": return f"({self.args[0]} * {self.args[1]})"
        if self.name == "Div": return f"({self.args[0]} / ({self.args[1]} + 1e-6))" # ProtectDiv
        if self.name == "Neg": return f"-1 * ({self.args[0]})"
        if self.name == "Inv": return f"1 / ({self.args[0]} + 1e-6)"
        if self.name == "Rank": return f"Rank({self.args[0]})"
        if self.name == "Sign": return f"Sign({self.args[0]})"
        if self.name == "Corr": return f"Corr({self.args[0]}, {self.args[1]}, 10)" # Fixed window for Corr/Cov
        if self.name == "Cov": return f"Cov({self.args[0]}, {self.args[1]}, 10)"
        return f"{self.name}({args_str})"

def random_tree(depth=0, method="grow"):
    # GROW: randomly choose field or op until max depth
    # FULL: force op until max depth
    
    if depth >= MAX_DEPTH or (method == "grow" and depth > 0 and random.random() < 0.3):
        return Field(random.choice(FIELDS))
    
    # Choose Operator Type
    op_type = random.choice(["unary", "binary", "ts", "corr"])
    
    if op_type == "unary":
        op = random.choice(UNARY_OPS)
        return Op(op, [random_tree(depth+1, method)])
    
    elif op_type == "binary":
        op = random.choice(BINARY_OPS)
        return Op(op, [random_tree(depth+1, method), random_tree(depth+1, method)])
        
    elif op_type == "ts":
        ts_ops_clean = [o for o in TS_OPS if o not in ["Corr", "Cov"]] # Exclude corr/cov from standard TS
        op = random.choice(ts_ops_clean)
        window = Constant(random.choice(WINDOWS))
        return Op(op, [random_tree(depth+1, method), window])

    elif op_type == "corr":
        op = random.choice(["Corr", "Cov"])
        # Corr(A, B, d) -> But Qlib Corr is Corr(A, B, d). 
        # But my Op.__str__ hardcodes window=10 for now to simplify Arity.
        # Let's actually support variable window:
        # OpStr handles args. If we pass 2 args, OpStr implementation needs to match.
        # My previous OpStr edit for Corr was: f"Corr({self.args[0]}, {self.args[1]}, 10)"
        # So we just need to provide 2 subtrees.
        return Op(op, [random_tree(depth+1, method), random_tree(depth+1, method)])

def mutate(tree):
    # Point mutation: pick a node and replace with random subtree
    if random.random() < 0.2:
        return random_tree() # Replace entire subtree
        
    if isinstance(tree, Op):
        # Mutate arguments
        new_args = [mutate(arg) if not isinstance(arg, Constant) else arg for arg in tree.args]
        
        # Mutate operator name (if compatible)
        new_name = tree.name
        if random.random() < 0.1:
            if tree.name in UNARY_OPS: new_name = random.choice(UNARY_OPS)
            elif tree.name in BINARY_OPS: new_name = random.choice(BINARY_OPS)
            elif tree.name in TS_OPS: new_name = random.choice(TS_OPS)
            
        return Op(new_name, new_args)
    
    return tree

def crossover(parent1, parent2):
    # Swap subtrees
    # Naive implementation: just return a mix if they are Ops
    if isinstance(parent1, Op) and isinstance(parent2, Op):
        if random.random() < 0.5:
             # Swap argument 0
             new_args1 = [copy.deepcopy(parent2.args[0])] + parent1.args[1:]
             new_args2 = [copy.deepcopy(parent1.args[0])] + parent2.args[1:]
             # Handle arity mismatch if any (here we grouped by arity so it's safer, but TS vs Binary is diff)
             # Simplify: Return random crossover or just mutation
             return Op(parent1.name, new_args1)
    
    return copy.deepcopy(parent1)

# --- GP ENGINE ---

def init_qlib():
    provider_uri = os.path.expanduser(QLIB_PROVIDER_URI)
    qlib.init(provider_uri=provider_uri, region=QLIB_REGION)

def evaluate_population(population, start_time, end_time):
    # 1. Generate expressions
    exprs = [str(ind) for ind in population]
    # Unique expressions to save compute
    unique_exprs = list(set(exprs))
    
    # Label: 5-day return
    label_expr = "Ref($close, -5) / $close - 1"
    
    # 2. Fetch Data
    # Only fetch if expr is valid Qlib
    valid_exprs = []
    
    # We batch fetch to utilize Qlib parallelism
    # But Qlib might fail on invalid formulas (e.g. Div by zero if handled poorly)
    # We will try robust fetch
    
    results = {}
    
    for i, expr in enumerate(unique_exprs):
        # print(f"  [{i+1}/{len(unique_exprs)}] Evaluating: {expr}")
        try:
            # Quick check: too complex?
            if len(expr) > 200: 
                results[expr] = -999
                continue
                
            # Fetch single factor to isolate errors
            df = D.features(ETF_LIST, [expr, label_expr], start_time=start_time, end_time=end_time, freq='day')
            df.columns = ["factor", "label"]
            df = df.dropna()
            
            if df.empty:
                # print(f"    -> Empty result (Invalid)")
                results[expr] = -999
                continue
                
            # Check for constant or near-constant signals
            if df["factor"].std() < 1e-5:
                results[expr] = -999
                continue
                
            # Calculate IC
            ic = df.groupby("datetime").apply(lambda x: x["factor"].corr(x["label"], method="spearman")).mean()
            if np.isnan(ic): ic = -999
            
            # Penalize Turnover?
            # For now just max IC
            results[expr] = abs(ic) # Objective: High Absolute Correlation (we can flip sign)
            
        except Exception as e:
            # print(f"Invalid expr: {expr} - {e}")
            results[expr] = -999
            
    # Assign fitness
    fitness_scores = [results[str(ind)] for ind in population]
    return fitness_scores

def main():
    print("🧬 Starting Genetic Alpha Mining...")
    init_qlib()
    
    START = "2020-01-01"
    END = "2022-12-31" # Train Period (In-Sample)
    
    # Initialize
    population = [random_tree(0, "grow") for _ in range(POPULATION_SIZE)]
    
    hall_of_fame = []
    
    for gen in range(GENERATIONS):
        print(f"\nGeneration {gen+1}/{GENERATIONS}")
        
        # Evaluate
        fitnesses = evaluate_population(population, START, END)
        
        # Stats
        max_fit = max(fitnesses)
        avg_fit = sum(fitnesses) / len(fitnesses)
        print(f"  Best IC: {max_fit:.4f} | Avg IC: {avg_fit:.4f}")
        
        # Update HOF
        for ind, fit in zip(population, fitnesses):
            if fit > 0.03: # Threshold
                # Check uniqueness
                is_unique = True
                for (hof_ind, hof_fit) in hall_of_fame:
                     if str(ind) == str(hof_ind):
                         is_unique = False
                         break
                
                if is_unique:
                    hall_of_fame.append((ind, fit))
        
        # Sort HOF
        hall_of_fame.sort(key=lambda x: x[1], reverse=True)
        hall_of_fame = hall_of_fame[:HOF_SIZE]
        
        # Selection & Resample
        new_population = []
        # Elitism
        if hall_of_fame:
            new_population.append(copy.deepcopy(hall_of_fame[0][0]))
            
        while len(new_population) < POPULATION_SIZE:
             # Tournament
             indices = random.sample(range(len(population)), TOURNAMENT_SIZE)
             tournament = [(population[i], fitnesses[i]) for i in indices]
             winner = max(tournament, key=lambda x: x[1])[0]
             
             child = copy.deepcopy(winner)
             
             # Mutate
             if random.random() < MUTATION_PROB:
                 child = mutate(child)
                 
             new_population.append(child)
        
        population = new_population

    print("\n" + "="*60)
    print("🏆 HALL OF FAME (Top Alphas)")
    print("="*60)
    
    for i, (ind, fit) in enumerate(hall_of_fame):
        print(f"{i+1}. IC: {fit:.4f} | Formula: {str(ind)}")
        
    # Save to file
    with open("artifacts/mined_alphas.txt", "w") as f:
        for ind, fit in hall_of_fame:
            f.write(f"{str(ind)}\n")
            
    print("\nSaved to artifacts/mined_alphas.txt")

if __name__ == "__main__":
    main()
