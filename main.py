import networkx as nx
import matplotlib.pyplot as plt
import xml.etree.ElementTree as etree
import heapq


class Node:
    def __init__(self, node_id, longitude, latitude):
        self.node_id = node_id
        self.longitude = float(longitude) / 1e5  # Convert from OSM scale
        self.latitude = float(latitude) / 1e5

    def __repr__(self):
        return f"Node(id={self.node_id}, longitude={self.longitude}, latitude={self.latitude})"


def parse_xml_to_nodes(xml_file):
    tree = etree.parse(xml_file)
    root = tree.getroot()
    nodes = []
    for node in root.findall('.//node'):
        node_id = node.get('id')
        longitude = node.get('longitude')
        latitude = node.get('latitude')
        node_instance = Node(node_id, latitude, longitude)
        nodes.append(node_instance)
    return nodes

class Arc:
    def __init__(self, from_node, to_node, length):
        self.from_node = from_node
        self.to_node = to_node
        self.length = float(length)  # Convert length to a float

    def __repr__(self):
        return f"Arc(from={self.from_node}, to={self.to_node}, length={self.length})"

def parse_xml_to_arcs(xml_file):
    tree = etree.parse(xml_file)
    root = tree.getroot()
    arcs = []
    for arc in root.findall('.//arc'):
        from_node = arc.get('from')
        to_node = arc.get('to')
        length = arc.get('length')
        arc_instance = Arc(from_node, to_node, length)
        arcs.append(arc_instance)
    return arcs

# Parse nodes and arcs from XML
nodes = parse_xml_to_nodes('../Tema 4.0/resources/Harta_Luxemburg.xml')
arcs = parse_xml_to_arcs('../Tema 4.0/resources/Harta_Luxemburg.xml')

# Create a graph using NetworkX
G = nx.Graph()

# Add nodes to the graph
for node in nodes:
    G.add_node(node.node_id, longitude=node.longitude, latitude=node.latitude)

# Add arcs (edges) to the graph
for arc in arcs:
    G.add_edge(arc.from_node, arc.to_node, length=arc.length)

# Scale the node positions to spread them out more
scaling_factor = 25.0  # Increased scaling factor to spread nodes further apart
scaled_node_pos = {node.node_id: (node.longitude * scaling_factor, node.latitude * scaling_factor) for node in nodes}


def dijkstra(nodes, arcs, start_node_id):
    graph = {node.node_id: {} for node in nodes}
    for arc in arcs:
        graph[arc.from_node][arc.to_node] = arc.length
        graph[arc.to_node][arc.from_node] = arc.length  # assuming undirected graph

    distances = {node.node_id: float('inf') for node in nodes}
    distances[start_node_id] = 0

    priority_queue = [(0, start_node_id)]  # (distance, node_id)

    previous_nodes = {node.node_id: None for node in nodes}

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances, previous_nodes


# Bellman-Ford using NetworkX
def networkx_bellman_ford(graph, start_node_id, end_node_id):
    try:
        # Compute shortest paths and distances using NetworkX Bellman-Ford
        length, path = nx.single_source_bellman_ford_path_length(graph, source=start_node_id), \
            nx.single_source_bellman_ford_path(graph, source=start_node_id)

        if end_node_id in path:
            shortest_path = path[end_node_id]
            shortest_distance = length[end_node_id]
            return shortest_path, shortest_distance, False
        else:
            print("No path found between the nodes.")
            return [], float('inf'), False
    except nx.NetworkXUnbounded:
        print("Negative weight cycle detected using NetworkX.")
        return [], float('inf'), True


def bellman_ford(nodes, arcs, start_node_id):
    all_node_ids = {node.node_id for node in nodes}
    for arc in arcs:
        all_node_ids.update([arc.from_node, arc.to_node])

    graph = {node_id: {} for node_id in all_node_ids}
    for arc in arcs:
        graph[arc.from_node][arc.to_node] = float(arc.length)
        graph[arc.to_node][arc.from_node] = float(arc.length)

    dist = {node_id: float('inf') for node_id in all_node_ids}
    prev = {node_id: None for node_id in all_node_ids}
    dist[start_node_id] = 0

    iteration_count = 0
    max_iterations = 50

    for _ in range(len(all_node_ids) - 1):
        if iteration_count > max_iterations:
            print("Too many iterations, check for cycles or large graph size.")
            break

        for node_id in graph:
            for neighbor, weight in graph[node_id].items():
                weight = float(weight)
                if dist[node_id] != float('inf') and dist[node_id] + weight < dist[neighbor]:
                    dist[neighbor] = dist[node_id] + weight
                    prev[neighbor] = node_id

        iteration_count += 1

    for node_id in graph:
        for neighbor, weight in graph[node_id].items():
            weight = float(weight)
            if dist[node_id] != float('inf') and dist[node_id] + weight < dist[neighbor]:
                print("Negative weight cycle detected")
                return dist, prev, True

    return dist, prev, False


def get_shortest_path(previous_nodes, start_node_id, end_node_id):
    path = []
    current_node = end_node_id
    # while current_node is not None:
    #     path.append(current_node)
    #     current_node = previous_nodes[current_node]
    while current_node is not None:
        path.append(current_node)
        current_node = previous_nodes[current_node]
        # Prevent infinite loops if the graph has disconnected components
        if current_node == start_node_id:
            path.append(current_node)
            break
    if not path or path[-1] != start_node_id:
        print("No valid path found between the selected nodes.")
        return []
    path.reverse()
    return path


# Plotting with interactive mouse click
class InteractivePlot:
    def __init__(self):
        self.start_node = None
        self.end_node = None
        self.fig, self.ax = plt.subplots(figsize=(25, 25))
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.tick_params(axis='both', colors='gray')
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        # Get the closest node to the clicked position
        closest_node = None
        min_dist = float('inf')

        for node_id, pos in scaled_node_pos.items():
            dist = (pos[0] - event.xdata) ** 2 + (pos[1] - event.ydata) ** 2
            if dist < min_dist:
                closest_node = node_id
                min_dist = dist

        if event.button == 1:  # Left click for Dijkstra
            if self.start_node is None:
                self.start_node = closest_node
                print(f"Start node selected: {self.start_node}")
            elif self.end_node is None:
                self.end_node = closest_node
                print(f"End node selected: {self.end_node}")
                # Run Dijkstra and plot the result
                distances, previous_nodes = dijkstra(nodes, arcs, self.start_node)
                shortest_path = get_shortest_path(previous_nodes, self.start_node, self.end_node)
                if shortest_path:
                    print(f"Dijkstra: Distance from {self.start_node} to {self.end_node} is {distances[self.end_node]:.2f}")
                self.plot_path(shortest_path, 'yellow')
                self.start_node = None
                self.end_node = None
        elif event.button == 3: #  Networkx Bellman Ford
            if self.start_node is None:
                self.start_node = closest_node
                print(f"Start node selected: {self.start_node}")
            elif self.end_node is None:
                self.end_node = closest_node
                print(f"End node selected: {self.end_node}")
                shortest_path_bf, distance_bf, negative_cycle = networkx_bellman_ford(G, self.start_node, self.end_node)
                if not negative_cycle:
                    print(f"Bellman-Ford: Distance from {self.start_node} to {self.end_node} is {distance_bf:.2f}")
                    self.plot_path(shortest_path_bf, 'lime')
                self.start_node = None
                self.end_node = None
        elif event.button == 2:  #  Bellman-Ford
            if self.start_node is None:
                self.start_node = closest_node
                print(f"Start node selected: {self.start_node}")
            elif self.end_node is None:
                self.end_node = closest_node
                print(f"End node selected: {self.end_node}")
                # Run Bellman-Ford and plot the result
                distances_bf, previous_nodes_bf, negative_cycle = bellman_ford(nodes, arcs, self.start_node)
                shortest_path_bf = get_shortest_path(previous_nodes_bf, self.start_node, self.end_node)
                self.plot_path(shortest_path_bf, 'red')
                self.start_node = None
                self.end_node = None

    def plot_path(self, path, color):
        # Highlight the path
        nx.draw_networkx_nodes(G, pos=scaled_node_pos, nodelist=path, node_color=color, node_size=2)
        nx.draw_networkx_edges(G, pos=scaled_node_pos, edgelist=list(zip(path[:-1], path[1:])), edge_color=color, width=1.5)
        plt.draw()


    def show(self):
        nx.draw(G, pos=scaled_node_pos, with_labels=False, node_size=2, font_size=8, edge_color='gray', node_color='blue', alpha=0.5)
        plt.tight_layout()
        plt.show()

# Initialize interactive plot
interactive_plot = InteractivePlot()
interactive_plot.show()
