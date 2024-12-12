import numpy as np
from collections import deque

"""
Naming conventions:
1. a "node" should point to a Node object
2. an "edge" should point to an Edge object
3. a "node_id" is an index of a node, can recover node object by QueueingNetwork.nodes[node_id]
4. an "edge_id" is an index of an edge, can recover edge object by QueueingNetwork.edges[edge_id]
5. a "source_id" is an index of a source node, can recover source node object by QueueingNetwork.nodes[source_id]
6. a "destination_id" is an index of a destination node, can recover destination node object by QueueingNetwork.nodes[destination_id]
7. a "traffic_class" is the index of a traffic class
"""
class Node:
    def __init__(self, id: int):
        self.id = id
        self.edges = {}
        self.queue = deque()
        self.arrival_queue = deque()
        self.arrival_processes = []

    def add_edge(self, edge):
        self.edges[edge.id] = edge

    def add_arrival_process(self, destination_id, rate, class_id):
        self.arrival_processes.append((destination_id, rate, class_id))

    def add_packets(self, packets):
        # check if packets is iterable
        if not hasattr(packets, '__iter__'):
            packets = [packets]
        self.queue.extend(packets)

    def sim_arrivals(self):
        """
        Adds packets to arrival queue based on the arrival processes
        :returns: number of packets added
        """
        arrived_packets = 0
        for destination_id, rate, traffic_class in self.arrival_processes:
            packets = [Packet(destination_id = destination_id,  traffic_class=traffic_class)
                       for _ in range(np.random.poisson(rate))]
            self.arrival_queue.extend(packets)
            arrived_packets += len(packets)
        return arrived_packets

    def process_packets(self): # must be called after calling process_packets for all edges
        """
        For each packet in its queue, remove the last edge traversed and then add it to the
        queue of the next edge in the path
        """
        while self.queue:
            packet = self.queue.popleft()
            packet.remaining_path.pop(0) # removes last traversed edge from remaining path
            if packet.remaining_path:
                self.edges[packet.remaining_path[0]].add_packets(packet)

    def __str__(self):
        return f"Node {self.id}, Queue: {len(self.queue)}, Arrival Queue: {len(self.arrival_queue)}"


class Edge:
    def __init__(self, id: int,  head_node: Node, tail_node: Node, service_rate):
        self.id = id
        self.head_node = head_node
        self.tail_node = tail_node
        self.service_rate = service_rate
        self.capacity = np.random.poisson(service_rate)
        self.queue = deque()

    def add_packets(self, packets):
        # check if packets is iterable
        if not hasattr(packets, '__iter__'):
            packets = [packets]
        self.queue.extend(packets)

    def sim_capacity(self):
        """
        Generates a new capacity for the edge
        """
        self.capacity = np.random.poisson(self.service_rate)

    def process_packets(self):
        # Process packets based on capacity
        self.tail.add_packets(
            [self.queue.popleft() for _ in range(min(self.capacity, len(self.queue)))])

    def __str__(self):
        return f"Edge {self.id}, Queue: {len(self.queue)}, Capacity: {self.capacity}"
class Packet:
    def __init__(self, destination_id = None, remaining_path = [], time_in_network=0, priority=0, traffic_class=None):
        self.destination_id = destination_id
        self.remaining_path = deque(remaining_path)
        self.time_in_network = time_in_network
        self.priority = priority
        self.traffic_class = traffic_class

class QueueingNetwork:
    def __init__(self,  edge_info: dict, traffic_info: dict):
        self.nodes = {}
        self.edges = {}
        self.edge_info = edge_info
        self.traffic_info = traffic_info
        self.initialize_network(edge_info, traffic_info)

    def initialize_network(self, edge_info, traffic_info):
        # Start by creating the nodes and edges
        for edge_id_str, info in edge_info.items():
            if info["head"] not in self.nodes:
                self.nodes[info["head"]] = Node(info["head"])
            if info["tail"] not in self.nodes:
                self.nodes[info["tail"]] = Node(info["tail"])
            edge = Edge(int(edge_id_str), info["head"], info["tail"], info["rate"])
            self.nodes[info["head"]].add_edge(edge)
            self.edges[int(edge_id_str)] = edge

        # Now initialize the ability for nodes to generate packets
        self.source_nodes = []
        for class_id_str, info in traffic_info.items():
            self.nodes[info["source"]].add_arrival_process(
                info["destination"],info["rate"] , int(class_id_str))
            self.source_nodes.append(self.nodes[info["source"]])

    def step(self, action):
        # Apply the action and process the network for one time-step

        # Process packets in the network
        for edge in self.edges:
            edge.process_packets()
        for node in self.nodes.values():
            node.process_packets()
        # Generate new packets
        for node in self.source_nodes:
            node.sim_arrivals()

    def route_packets(self):
        # Implement the routing algorithm
        pass

    def compute_shortest_path(self, source_id, destination_id):
        # Implement the shortest path algorithm
        pass