import csv
import math
from collections import deque
import sys

print("Starting BFT Agl... ")
if len(sys.argv) != 3:
    print("Usage: python [Yourcode.py] city_name_1 city_name_2")
    sys.exit(1)

start = sys.argv[1]
end = sys.argv[2]
d_city = {}
d_distination = {}
graph = {}

def ReadFiles():
    with open('cities.csv', mode='r') as f:
        data = csv.reader(f)
        for row in data:
            if len(row) < 3:
                continue
            city, lat, lon = row[0].strip(), float(row[1].strip()), float(row[2].strip())
            d_city[city] = (lat, lon)

    with open('roads.csv', mode='r') as f:
        data = csv.reader(f)
        for row in data:
            if len(row) < 2:
                continue  # Skip invalid rows
            city1, city2 = row[0].strip(), row[1].strip()

            if city1 not in d_distination:
                d_distination[city1] = []
            if city2 not in d_distination:
                d_distination[city2] = []

            d_distination[city1].append(city2)
            d_distination[city2].append(city1)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def BuildGraph():
    for city, neighbors in d_distination.items():
        graph[city] = []
        for neighbor in neighbors:
            if city in d_city and neighbor in d_city:
                lat1, lon1 = d_city[city]
                lat2, lon2 = d_city[neighbor]
                distance = round(haversine(lat1, lon1, lat2, lon2), 2)  
                graph[city].append((neighbor, distance)) 


def BFS_cost(graph, start, goal):
    visited = set()  
    queue = deque([([start], 0)])

    while queue:
        path, distance = queue.popleft()
        node = path[-1]

        if node in visited:
            continue

        visited.add(node)

        if node == goal:
            return path, distance

        for neighbor, cost in graph.get(node, []):
            if neighbor not in visited:
                new_path = path + [neighbor]
                queue.append((new_path, distance + cost))

    return None, float('inf')

def main():
    ReadFiles()
    BuildGraph()
    path, total_distance = BFS_cost(graph, start, end)
    
    if path is not None and len(path) > 0:
        print(" - ".join(path))
        print(f"Total Distance - [{total_distance:.2f}] km")
    else:
        print("Path not found!")

# Example usage
if __name__ == '__main__':
    main()


