import sys
import pandas as pd 
import csv
import math
import heapq

class AStar:
	def __init__(self, cities_file, distances_file):

		self.rad = 6371
		self.d_city = {}
		self.d_distination = {}
		self.graph = {}
		

		with open(cities_file, mode='r') as f:
		    data = csv.reader(f)
		    self.d_city = {rows[0]:(float(rows[1]), float(rows[2])) for rows in data}

		with open(distances_file, mode='r') as f:
		    data = csv.reader(f)
		    for row in data:
		    	if len(row) < 2:
		    		continue

		    	city1, city2 = row[0], row[1]
		    	
		    	if city1 not in self.d_distination:
		    		self.d_distination[city1] = []
		    	
		    	if city2 not in self.d_distination:
		    		self.d_distination[city2] = []

		    	self.d_distination[city1].append(city2)
		    	self.d_distination[city2].append(city1)
		self.BuildGraph()

	def haversine(self, lat1, lon1, lat2, lon2):
     
	    dLat = math.radians(lat2-lat1)
	    dLon = math.radians(lon2-lon1)
	 
	    # convert to radians
	    lat1 = (lat1) * math.pi / 180.0
	    lat2 = (lat2) * math.pi / 180.0


	 
	    # apply formulae
	    a = (pow(math.sin(dLat / 2), 2) +
	         pow(math.sin(dLon / 2), 2) *
	             math.cos(lat1) * math.cos(lat2));

	    
	    c = 2 * math.asin(math.sqrt(a))
	    return self.rad * c

	# { "cityname": [("neighbor1", distance), ("neighbor2", distance), ...] }
	def BuildGraph(self):

		for city , neighbors in self.d_distination.items():
			self.graph[city] = []

			for neighbor in neighbors:
				if city in self.d_city and neighbor in self.d_city:
					lat1, lon1 = self.d_city[city]
					lat2, lon2 = self.d_city[neighbor]
					distance = round(self.haversine(lat1, lon1, lat2, lon2), 2)  

					self.graph[city].append((neighbor, distance))  

	def heuristic(self, city, goal):

		if city in self.d_city and goal in self.d_city:
			lat1, lon1 = self.d_city[city]
			lat2, lon2 = self.d_city[goal]
			return self.haversine(lat1, lon1, lat2, lon2)
		return float('inf')

	def path_f_cost(self,path, goal):
	    #f(n) = g(n) + h(n)
	    g_cost = sum(cost for _, cost in path)
	    last_node = path[-1][0]
	    h_cost = self.heuristic(last_node, goal)
	    return g_cost + h_cost, g_cost, last_node


	def A_Star_Search(self, start, goal):
	    priority_queue = []
	    heapq.heappush(priority_queue, (0, 0, start, [(start, 0)]))  # (f-cost, g-cost, node, path)

	    visited = set()

	    while priority_queue:
	        f_cost, g_cost, node, path = heapq.heappop(priority_queue)

	        if node in visited:
	            continue
	        visited.add(node)

	        # If goal is reached, return the path
	        if node == goal:
	            return path, g_cost

	        # Expand adjacent cities
	        for neighbor, cost in self.graph.get(node, []):
	            if neighbor in visited:
	                continue

	            new_path = path + [(neighbor, cost)]
	            new_f, new_g, _ = self.path_f_cost(new_path, goal)

	            heapq.heappush(priority_queue, (new_f, new_g, neighbor, new_path))

	    #print("Path not found!")
	    return None, None

if __name__ == '__main__':
	print("Starting A* Search Agl... ")
	if len(sys.argv) != 3:
		print("Usage: python [Yourcode.py] city_name_1 city_name_2")
		sys.exit(1)

	Start = sys.argv[1]
	Destination = sys.argv[2]
	
	astar = AStar("cities.csv", "roads.csv")
	#print(" object created successfully!")
	
	solution, cost = astar.A_Star_Search(Start, Destination)
	
	if solution is not None and len(solution) > 0:
		first = True
		for city in solution:
			if not first:
				print(" - ", end="")
			print(city[0], end="")
			first = False
		print()
		print(f"Total Distance - [{cost:.2f}] km")
	else:
		print("Path not found!")
