import networkx as nx
import numpy as np
import random
import time
from Problem import Problem

def solution(p: Problem):
    # if beta <= 1 split_tour strategy
    # if beta > 1: hybrid strategy

    # takes problem specifics
    graph = p.graph
    alpha = p.alpha
    beta = p.beta
    
    # check zero nodes case
    nodes = [n for n in graph.nodes if n != 0]
    if not nodes:
        return [(0, 0)]

    # gold map    
    max_node = max(graph.nodes)
    gold_map = np.zeros(max_node + 1)
    for i in graph.nodes:
        gold_map[i] = graph.nodes[i]['gold']

    # distances btwn nodes
    path_lengths = dict(nx.all_pairs_dijkstra_path_length(graph, weight='dist'))
    # path from one node to another
    all_paths_cache = dict(nx.all_pairs_dijkstra_path(graph, weight='dist'))
    
    # creation of distance matrix
    dist_matrix = np.zeros((max_node + 1, max_node + 1))
    for u, lengths in path_lengths.items():
        for v, d in lengths.items():
            dist_matrix[u, v] = d
            
    dist_from_home = dist_matrix[0, :]
    dist_to_home = dist_matrix[:, 0]

    def get_path_segment(start, end, gold_at_end):
        # takes only decided quantity of gold at last node
        if start == end:
            return [(end, gold_at_end)]
        try:
            physical = all_paths_cache[start][end]
        except KeyError:
        # graph disconnected case
            return []
        segment = []
        for i in range(1, len(physical)):
            is_last = (i == len(physical) - 1)
            segment.append((physical[i], gold_at_end if is_last else 0))
        return segment

    # beta <= 1 case
    if beta <= 1:
        TIME_LIMIT = 5.0
        MAX_WINDOW = 70 # max nodes per trip segment

        def split_tour(tour):
            # takes a tour of all the nodes and splits it into optimal trips
            n = len(tour)
            dp = np.full(n + 1, 1e15) # inf initial cost for every subtour
            dp[0] = 0.0 # zero cost for empty tour
            parent = np.zeros(n + 1, dtype=int) # where each subtour starts
            
            for i in range(n):
                # unreachable
                if dp[i] > 1e14: 
                    continue

                curr_load = 0.0 
                curr_cost_acc = 0.0 
                prev = 0 
                # extends the trip up to MAX_WINDOW nodes
                for k in range(i, min(n, i + MAX_WINDOW)):
                    node = tour[k]
                    d_step = dist_matrix[prev, node]

                    # unreachable
                    if d_step >= 1e14:
                        break
                    
                    # cost to reach this node with current load
                    step_cost = d_step + (d_step * alpha * curr_load)**beta if alpha > 0 else d_step
                    curr_cost_acc += step_cost
                    # when beta < 1, taking all the gold is better
                    curr_load += gold_map[node]
                    
                    # cost to base from this node
                    d_home = dist_to_home[node]
                    return_cost = d_home + (d_home * alpha * curr_load)**beta if alpha > 0 else d_home
                    
                    total = curr_cost_acc + return_cost
                    
                    # check if better than previous best
                    if dp[i] + total < dp[k + 1]:
                        dp[k + 1] = dp[i] + total
                        parent[k + 1] = i # start from node i to node k
                    
                    prev = node
            # return best cost and parent array to reconstruct trips
            return dp[n], parent

        def run_ea(initial_tour, time_limit=10):
            start_time = time.time()
            
            #uses split_tour to evaluate cost
            def evaluate_tour(tour):
                cost, _ = split_tour(tour)
                return cost
            
            #population creation
            pop_size = 30
            population = []
            population.append(initial_tour.copy())
            
            if len(initial_tour) > 1:
                #sort by reverse distance
                by_dist = sorted(initial_tour, key=lambda n: dist_from_home[n])
                population.append(by_dist)
                population.append(initial_tour[::-1])
            
            #fill rest with random individuals
            while len(population) < pop_size:
                shuffled = initial_tour.copy()
                random.shuffle(shuffled)
                population.append(shuffled)

            best_tour = initial_tour
            best_cost = evaluate_tour(initial_tour)
            
            #EA parameters
            elite_size = 4
            mutation_rate = 0.7
            generations = 0
            stagnation = 0
            
            def crossover(p1, p2):
                #crossover from parent1 and parent2
                size = len(p1)
                if size < 2: return p1.copy()
                cut1, cut2 = sorted(random.sample(range(size), 2))
                child = [None] * size
                child[cut1:cut2] = p1[cut1:cut2]
                idx = 0
                for i in range(size):
                    if child[i] is None:
                        while p2[idx] in child[cut1:cut2]:
                            idx += 1
                        child[i] = p2[idx]
                        idx += 1
                return child

            def mutation(individual):
                #swap of two nodes
                if len(individual) < 2: return individual
                i, j = random.sample(range(len(individual)), 2)
                individual[i], individual[j] = individual[j], individual[i]
                return individual

            def inversion(tour, evaluate_func, max_iter=30):
                best = tour.copy()
                best_cost = evaluate_func(tour)
                improved = True
                iterations = 0
                while improved and iterations < max_iter:
                    improved = False
                    for i in range(1, len(tour) - 2):
                        for j in range(i + 2, len(tour)):
                            if j - i == 1: continue
                            new_tour = tour[:i] + tour[i:j][::-1] + tour[j:]
                            new_cost = evaluate_func(new_tour)
                            if new_cost < best_cost - 0.01:
                                best = new_tour
                                best_cost = new_cost
                                improved = True
                                break
                        if improved: break
                    iterations += 1
                return best
            
            #main loop
            while time.time() - start_time < time_limit:
                generations += 1
                scored = [(evaluate_tour(t), t) for t in population]
                scored.sort(key=lambda x: x[0])
                
                current_best = scored[0][0]
                if current_best < best_cost - 0.01:
                    best_cost = current_best
                    best_tour = scored[0][1].copy()
                    stagnation = 0
                else:
                    stagnation += 1
                
                elite = [t for _, t in scored[:elite_size]]
                #new gen
                offspring = []
                while len(offspring) < pop_size - elite_size:
                    p1 = random.choice(elite)
                    p2 = random.choice(population)
                    if len(p1) > 1:
                        child = crossover(p1, p2)
                        if random.random() < mutation_rate:
                            child = mutation(child)
                        offspring.append(child)
                    else:
                        offspring.append(p1.copy())
                population = elite + offspring
                
                if stagnation > 8:
                    for _ in range(3):
                        shuffled = best_tour.copy()
                        random.shuffle(shuffled)
                        population.append(shuffled)
                    stagnation = 0
                    
            # final 
            if len(best_tour) > 3:
                best_tour = inversion(best_tour, evaluate_tour, 15)
                best_cost = evaluate_tour(best_tour)
            return best_tour
        
        start_time = time.time()
        best_cost_b = float('inf')
        best_trips_b = None
        
        # check nodes with gold
        target_nodes = [n for n in nodes if gold_map[n] > 0]
        
        # 1. efficiency (gold / distance)
        efficiency = [(n, gold_map[n] / (dist_from_home[n] + 0.001)) for n in target_nodes]
        efficiency.sort(key=lambda x: -x[1]) # descending order
        
        # different starting points
        for start_pct in [0, 20, 40]:
            if start_pct >= len(efficiency): break
            tour_nodes = [n for n, _ in efficiency[start_pct:]] + [n for n, _ in efficiency[:start_pct]]
            cost_e, parent_e = split_tour(tour_nodes)
            
            if cost_e < best_cost_b:
                best_cost_b = cost_e
                # solution reconstruction
                trips = []
                idx = len(tour_nodes)
                while idx > 0:
                    trips.append(tour_nodes[parent_e[idx]:idx])
                    idx = parent_e[idx]
                trips.reverse()
                best_trips_b = trips

        # 2. distance
        tour_dist = sorted(target_nodes, key=lambda n: dist_from_home[n])
        cost_d, parent_d = split_tour(tour_dist)
        
        if cost_d < best_cost_b:
            best_cost_b = cost_d
            trips = []
            idx = len(tour_dist)
            while idx > 0:
                trips.append(tour_dist[parent_d[idx]:idx])
                idx = parent_d[idx]
            trips.reverse()
            best_trips_b = trips
            
        # 3. random among close nodes
        while time.time() - start_time < TIME_LIMIT:
            unvisited = set(target_nodes)
            tour = []
            if unvisited:
                top = min(15, len(efficiency))
                first = random.choice([n for n, _ in efficiency[:top]])
                tour.append(first)
                unvisited.remove(first)
                curr_node = first
                
                while unvisited:
                    candidates = sorted(unvisited, key=lambda x: dist_matrix[curr_node, x])
                    pick_from = min(7, len(candidates))
                    nxt = random.choice(candidates[:pick_from])
                    tour.append(nxt)
                    unvisited.remove(nxt)
                    curr_node = nxt
            cost_r, parent_r = split_tour(tour)

            if cost_r < best_cost_b:
                best_cost_b = cost_r
                trips = []
                idx = len(tour)
                while idx > 0:
                    trips.append(tour[parent_r[idx]:idx])
                    idx = parent_r[idx]
                trips.reverse()
                best_trips_b = trips

        # list of nodes from best trips 
        current_best_nodes = []
        if best_trips_b:
            for trip in best_trips_b:
                current_best_nodes.extend(trip)
        else:
            current_best_nodes = target_nodes.copy()

        # EA runned on the sequence of nodes
        improved_nodes = run_ea(current_best_nodes, time_limit=10)
        
        # re-split improved tour to get final trips
        _, parent_opt = split_tour(improved_nodes)
        final_trips = []
        idx = len(improved_nodes)
        while idx > 0:
            final_trips.append(improved_nodes[parent_opt[idx]:idx])
            idx = parent_opt[idx]
        final_trips.reverse()
        
        #final solution creation from EA result
        best_path = []
        curr = 0
        if final_trips:
            for trip in final_trips:
                for node in trip:
                    best_path.extend(get_path_segment(curr, node, gold_map[node]))
                    curr = node
                best_path.extend(get_path_segment(curr, 0, 0))
                curr = 0
        else:
            best_path = [(0, 0)]
        return best_path

    # beta > 1 case
    else:
        gold_nodes = [n for n in nodes if gold_map[n] > 0]
        # 1. split only
        def split_only_strategy():
            route = []
            total_cost = 0.0
            for node in gold_nodes:
                d = dist_from_home[node]
                g = gold_map[node]
                
                best_c = float("inf")
                best_k = 1
                max_k = min(int(g) + 1, 100)
                
                # number of trips to take the gold
                for k in range(1, max_k + 1):
                    w = g / k # gold taken per trip
                    c = k * (2 * d + (alpha * d * w) ** beta) # cost
                    if c < best_c:
                        best_c = c
                        best_k = k
                # final weight per trip
                w = g / best_k

                for _ in range(best_k):
                    route.append((node, w))
                    route.append((0, 0))
                total_cost += best_c
            return route, total_cost

        # 2. aggressive
        def aggressive_strategy():
            route = []
            total_cost = 0.0
            for node in gold_nodes:
                d = dist_from_home[node]
                g = gold_map[node]
                # aggresive split
                n_trips = min(int((alpha * g) ** beta) + 1, 200)
                n_trips = max(1, n_trips)
                w = g / n_trips
                for _ in range(n_trips):
                    route.append((node, w))
                    route.append((0, 0))
                    total_cost += 2 * d + (alpha * d * w) ** beta
            return route, total_cost

        # strategy selection
        selected_route = []
        r1, c1 = split_only_strategy()
        r2, c2 = aggressive_strategy()
        selected_route = r1 if c1 < c2 else r2

        # solution path creation
        final_path = []
        curr = 0
        for node, g in selected_route:
            if node == curr:
                final_path.append((node, g))
                continue
            final_path.extend(get_path_segment(curr, node, g))
            curr = node 
        best_path = final_path if final_path else [(0,0)]
        return best_path