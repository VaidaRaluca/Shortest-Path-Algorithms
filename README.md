# Pathfinding Algorithms on a Graph

This project demonstrates the use of different pathfinding algorithms (Dijkstra, Bellman-Ford) on a graph parsed from an XML file containing nodes and arcs. The graph is visualized using `matplotlib` and allows users to interactively select start and end nodes, and then visualize the shortest path calculated by Dijkstra or Bellman-Ford algorithms.

![Map Image](Shortest-Path-Algorithms/images/graph.png)

## Features

- **Node Parsing**: Extracts node information (ID, longitude, latitude) from an XML file.
- **Arc Parsing**: Extracts arc information (from node, to node, length) from the same XML file.
- **Graph Creation**: Constructs an undirected graph using NetworkX, where nodes and edges (arcs) are added.
- **Dijkstra's Algorithm**: Finds the shortest path between two nodes using Dijkstra’s algorithm.
- **Bellman-Ford Algorithm**: Computes the shortest path and detects negative weight cycles.
- **Interactive Visualization**: Allows the user to select start and end nodes through mouse clicks, and visualizes the computed shortest path in the graph.

## Requirements

To run this project, the following Python packages are required:

- `lxml==5.3.0`
- `osmnx~=2.0.0`
- `networkx~=3.4.2`
- `matplotlib~=3.9.0`
