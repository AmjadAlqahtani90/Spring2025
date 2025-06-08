# Re-import necessary libraries since execution state was reset
import networkx as nx
import matplotlib.pyplot as plt

# Define city connections based on given data
edges = [
    ("San_Angelo", "Midland"), 
    ("San_Angelo", "Lubbock"), 
    ("San_Angelo", "Abilene"), 
    ("San_Angelo", "San_Antonio"),
    ("San_Antonio", "New_Braunfels"), 
    ("San_Antonio", "Seguin"), 
    ("San_Antonio", "Three_Rivers"), 
    ("San_Antonio", "Uvalde"),
    ("Austin", "San_Marcos"), 
    ("Austin", "Round_Rock"), 
    ("Austin", "College_Station"), 
    ("Austin", "Houston"),
    ("Temple", "Waco"), 
    ("College_Station", "Waco"), 
    ("College_Station", "Houston"), 
    ("Houston", "Beaumont"),
    ("Houston", "Galveston"), 
    ("Houston", "Columbus"), 
    ("Houston", "Sugar_Land"), 
    ("Victoria", "Gonzalez"),
    ("Gonzalez", "Seguin"), 
    ("Corpus_Christi", "Victoria"), 
    ("Corpus_Christi", "Three_Rivers"), 
    ("Alice", "Laredo"),
    ("Alice", "Three_Rivers"), 
    ("Alice", "McAllen"), 
    ("Laredo", "McAllen"), 
    ("Del_Rio", "Uvalde"), 
    ("El_Paso", "Odessa"),
    ("McAllen", "Brownsville"), 
    ("Waco", "Palestine"), 
    ("Waco", "Fort_Worth"), 
    ("Waco", "Dallas"), 
    ("Dallas", "Fort_Worth"),
    ("Lubbock", "Midland"), 
    ("Lubbock", "Amarillo"), 
    ("Amarillo", "Dalhart"), 
    ("Fort_Worth", "Wichita_Falls"),
    ("Midland", "Odessa"), 
    ("San_Marcos", "New_Braunfels"), 
    ("San_Marcos", "Gonzalez"), 
    ("New_Braunfels", "Seguin"),
    ("Columbus", "Seguin"), 
    ("Galveston", "Jamaica_Beach"), 
    ("Texarkana", "Dallas"), 
    ("Three_Rivers", "Kenedy"),
    ("Round_Rock", "Temple")
]

G = nx.Graph()
G.add_edges_from(edges)

# Redraw the graph using a different layout to minimize overlaps
plt.figure(figsize=(14, 10))

# Use a shell layout to reduce node overlap
pos = nx.kamada_kawai_layout(G)  # Kamada-Kawai layout helps spread out nodes

nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=1500, font_size=8)

# Show the improved graph layout
plt.title("City Connectivity Graph (Minimized Overlap)")
plt.show()

