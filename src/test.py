import time
from Problem import Problem 
from s337282 import solution 

def check_solution_cost(p: Problem, path):
    total_cost = 0.0
    current_node = 0
    current_load = 0.0
    
    if not path: return float('inf')

    for next_node, gold_taken in path:
        try:
            segment_cost = p.cost([current_node, next_node], current_load)
            total_cost += segment_cost
        except AttributeError:
            #if cost function is unavailable
            dist = p.graph[current_node][next_node]['dist']
            segment_cost = dist + (p.alpha * dist * current_load) ** p.beta
            total_cost += segment_cost
        except Exception:
            pass

        current_node = next_node
        
        if current_node == 0:
            current_load = 0.0 
        else:
            current_load += gold_taken
    return total_cost

def run_test(N, density, alpha, beta):
    print(f"Test config: N={N}, density={density}, alpha={alpha}, beta={beta}")
    p = Problem(N, density=density, alpha=alpha, beta=beta)
    
    # baseline
    start = time.time()
    base_cost = p.baseline()
    base_time = time.time() - start
    
    # solution()
    start = time.time()
    my_path = solution(p)
    my_time = time.time() - start
    
    # cost and gold verification
    # pre-calculated cost if available
    if hasattr(p, 'algo_cost'):
        my_cost_val = p.algo_cost
        total_gold_collected = getattr(p, 'algo_gold', 0.0)
    else:
        # unoptimized solutions
        my_cost_val = check_solution_cost(p, my_path)
        total_gold_collected = sum(step[1] for step in my_path) if my_path else 0
    diff = base_cost - my_cost_val
    
    # total available gold in the graph
    total_gold_graph = sum(p.graph.nodes[n].get('gold', 0) for n in p.graph.nodes)

    print(f"Baseline: {base_cost:.2f} (Calc time: {base_time:.2f}s)")
    print(f"My Sol:   {my_cost_val:.2f} (Calc time: {my_time:.2f}s)")
    
    print(f"Total Gold:     {total_gold_graph:.2f}")
    print(f"Collected Gold: {total_gold_collected:.2f}")
    
    print(f"Diff:     {diff:.2f} ", end="")
    
    if diff > 0.01: 
        print("(Better)")
    elif diff < -0.01: 
        print("(Worse)")
    else: 
        print("(Equal)")

if __name__ == "__main__":
    # configs (N, density, alpha, beta)
    test_cases = [
    (50, 0.2, 1, 1),
    (50, 0.2, 2, 1),
    (50, 0.2, 1, 2),
    (50, 1, 1, 1),
    (50, 1, 2, 1),
    (50, 1, 1, 2),
    ]

    print("=" * 60)
    for N, d, a, b in test_cases:
        test_start_time = time.time()
        
        run_test(N, d, a, b)
        
        test_end_time = time.time()
        elapsed_total = test_end_time - test_start_time
        
        print(f"Total test duration: {elapsed_total:.4f}s")
        print("=" * 60)