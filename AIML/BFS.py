import heapq
Graph_nodes = {
    'S': ['A','B'],
    'A': ['C','D'],
    'B': ['E','F'],
    'E': ['H'],
    'F': ['I','G']
}
def heuristic(n):
    H_dist = {
        'S': 13,
        'A': 12,
        'B': 4,
        'C': 7,
        'D': 3,
        'E': 2,
        'F': 8,
        'H': 4,
        'I': 9,
        'G': 0
    }
    return H_dist[n]

def best_first_search(graph, start, goal):
    """
    graph: dict having tree structure
    start: starting node of the tree
    goal: node to search
    heuristic: list of heuristic cost
    """
    open = [(heuristic(start), start)]
    closed = {}
    closed[start] = None
    
    while open:
        _,peak_node = heapq.heappop(open)
        if peak_node == goal:
            break
        for neighbour in graph[peak_node]:
            if neighbour not in closed:
                heapq.heappush(open, (heuristic(neighbour), neighbour))
                closed[neighbour] = peak_node
    return closed

start_node = 'B'
goal_node = 'H'
closed = best_first_search(Graph_nodes, start_node, goal_node)
print("Closed list",closed)
node = goal_node
path = [node]

while node != start_node:
    node = closed[node]
    path.append(node)
    
path.reverse()
print("BFS path from", start_node,"to",goal_node,":",path)
