"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from argparse import Namespace
from copy import deepcopy

import networkx as nx

from mo.graph.graph import Node, Graph
from mo.middle.pattern_match import all_edges_in_nodes
from mo.utils.error import Error


def not_all_new(old_elements: list, new_elements: list):
    """
    This function check whether at least one element from new_elements are in old_elements.
    """
    return any([element in old_elements for element in new_elements])


def check_and_update_ports(node, edges_data: list, in_port: bool = True):
    key = 'in' if in_port else 'out'
    key_in_edges = [key in edge_data for edge_data in edges_data]
    if all(key_in_edges):
        ports = [edge_data[key] for edge_data in edges_data]
        if len(ports) != len(set(ports)):
            raise Error("Please, provide unique {} ports for nodes".format(key))
    elif not any(key_in_edges):
        if node.has_valid('kind') and node.kind == 'data':
            return
        for i, edge_data in enumerate(edges_data):
            edge_data[key] = i
    else:
        raise Error("Please, provide all {} ports for nodes".format(key))


def build_graph_with_attrs(nodes_with_attrs: list, edges_with_attrs: list, new_nodes_with_attrs: list = [],
                           new_edges_with_attrs: list = [], update_edge_attrs: dict = None,
                           update_nodes_attributes: dict = None, nodes_with_edges_only: bool = False,
                           add_nodes_from_edges: bool = False):
    """
    Build the Graph with specific nodes and edges. Also update of edge and node parameters is supported.
    :param nodes_with_attrs: list of tuples ('node_name', {node_attrs})
    :param edges_with_attrs: list of tuples like (start node, end node, (optional) {attrs of the edge}).
    :param new_nodes_with_attrs: analogically nodes_with_attrs
    :param new_edges_with_attrs: analogically new_edges
    :param update_edge_attrs: optional dictionary like {('from_node', 'to_node', key): {edge_attrs}}.
    :param update_nodes_attributes: optional dictionary which specifies nodes names and their attributes to be updated. The
    key is a node name to update attribute and the value is a dictionary with attribute name and its value.
    :param nodes_with_edges_only: add nodes which has at least one incoming or outcoming edge.
    :param add_nodes_from_edges: whether nodes that is not listed in all_nodes but are in all_edges is allowed.
    :return: generated graph.
    """
    if not_all_new([node[0] for node in nodes_with_attrs], [node[0] for node in new_nodes_with_attrs]):
        raise Error('Some nodes from new_nodes_with_attrs are already in nodes.'
                    ' Please, add to new_nodes_with_attrs only NEW nodes.')

    if not_all_new([(edge[0], edge[1]) for edge in edges_with_attrs],
                   [(edge[0], edge[1]) for edge in new_edges_with_attrs]):
        raise Error('Some edges from new_edges_with_attrs are already in edges.'
                    ' Please, add to new_edges_with_attrs only NEW edges.')

    # Check that all nodes from list of edges are in nodes
    all_nodes = nodes_with_attrs + new_nodes_with_attrs
    all_edges = edges_with_attrs + new_edges_with_attrs
    all_nodes_names = [node[0] for node in all_nodes]
    if not add_nodes_from_edges and not all_edges_in_nodes(nodes=all_nodes_names, edges=all_edges):
        raise Error("Some nodes from list of edges is not in nodes. Please, add all necessary nodes.")

    graph = Graph()

    # Create dict for nodes with attrs
    nodes_attrs = {}
    for node_name, attrs in all_nodes:
        nodes_attrs[node_name] = attrs
        if 'name' not in attrs:
            attrs['name'] = node_name

    if nodes_with_edges_only:
        # filter nodes to keep only ones with edges connected
        filtered_nodes = {}
        for edge in all_edges:
            node_1, node_2 = edge[0], edge[1]
            filtered_nodes[node_1] = nodes_attrs[node_1]
            filtered_nodes[node_2] = nodes_attrs[node_2]
        nodes_attrs = filtered_nodes

    # Create all nodes
    for node, attrs in nodes_attrs.items():
        graph.add_node(node, **deepcopy(attrs))

    # Connect nodes with edges (also unpack edge params)
    for edge in all_edges:
        node_1, node_2 = edge[0], edge[1]
        edge_attrs = edge[2] if len(edge) == 3 else {}
        graph.add_edge(node_1, node_2, **edge_attrs)

    # Update attributes of edges
    if update_edge_attrs:
        # it will work in 2.x networkx only
        for edge, attr in update_edge_attrs.items():
            for k, v in attr.items():
                nx.set_edge_attributes(G=graph, name=k, values={edge: v})

    # Update attributes of nodes
    if update_nodes_attributes is not None:
        for node_name, new_attrs in update_nodes_attributes:
            assert (node_name in graph.nodes())
            for attr, value in new_attrs.items():
                graph.node[node_name][attr] = value

    for node_id in graph.nodes():
        node = Node(graph, node_id)
        check_and_update_ports(node, [graph.get_edge_data(edge[0], node_id)[0] for edge in graph.in_edges(node_id)], True)
        check_and_update_ports(node, [graph.get_edge_data(node_id, edge[1])[0] for edge in graph.out_edges(node_id)], False)

    for node in graph.get_op_nodes():
        # Add in_ports attribute
        in_edges = node.in_edges()
        for i in range(len(in_edges)):
            node.add_input_port(idx=i)

        # Add out_ports attribute
        out_edges = node.out_edges()
        for i in range(len(out_edges)):
            node.add_output_port(idx=i)
    return graph


def build_graph(nodes_attrs: dict, edges: list, update_attributes: dict = None, nodes_with_edges_only: bool = False):
    """
    Build the Graph with specific nodes and edges.
    :param nodes_attrs: dictionary where key is the node name and the value is the dictionary with node attributes.
    :param edges: list of pairs with start and end node names of the edge.
    :param update_attributes: optional dictionary which specifies nodes names and their attributes to be updated. The
    key is a node name to update attribute and the value is a dictionary with attribute name and its value.
    :param nodes_with_edges_only: add nodes which has at least one incoming or outcoming edge.
    :return: generated graph.
    """
    graph = Graph()

    for node_name, attrs in nodes_attrs.items():
        if 'name' not in attrs:
            attrs['name'] = node_name

    if nodes_with_edges_only:
        # filter nodes to keep only ones with edges connected
        filtered_nodes = {}
        for item in edges:
            if len(item) == 2:  # TODO: is there any better way in python to do that?
                node1, node2 = item
            else:
                node1, node2, _ = item
            filtered_nodes[node1] = nodes_attrs[node1]
            filtered_nodes[node2] = nodes_attrs[node2]
        nodes_attrs = filtered_nodes

    # create all nodes first
    for node, attrs in nodes_attrs.items():
        assert node not in graph.nodes()
        graph.add_node(node, **deepcopy(attrs))

    # connect nodes with edges
    for item in edges:
        if len(item) == 2:  # TODO: is there any better way in python to do that?
            node_1, node_2 = item
            edge_attrs = {}
        else:
            node_1, node_2, edge_attrs = item

        common_attrs = {'in': len(graph.in_edges(node_2)),
                        'out': len(graph.out_edges(node_1)),
                        'name': nodes_attrs[node_1]['name']}
        common_attrs.update(edge_attrs)
        graph.add_edge(node_1, node_2, **common_attrs)

    if update_attributes is not None:
        for node_name, new_attrs in update_attributes.items():
            assert (node_name in graph.nodes()), 'Node with name "{}" is not in the graph'.format(node_name)
            for attr, value in new_attrs.items():
                graph.node[node_name][attr] = value

    for node in graph.get_op_nodes():
        # Add in_ports attribute
        in_edges = node.in_edges()
        for attr in in_edges.values():
            node.add_input_port(idx=attr['in'])

        # Add out_ports attribute
        out_edges = node.out_edges()
        for attr in out_edges.values():
            node.add_output_port(idx=attr['out'])

    graph.graph['cmd_params'] = Namespace(generate_experimental_IR_V10=False, keep_shape_ops=False)
    return graph


def build_graph_with_edge_attrs(nodes_attrs: dict, edges: list, update_attributes: dict = None):
    """
    Build the Graph with specific nodes and edges.
    :param nodes_attrs: dictionary where key is the node name and the value is the dictionary with node attributes.
    :param edges: list of pairs with start and end node names of the edge.
    :param update_attributes: optional dictionary which specifies nodes names and their attributes to be updated. The
    key is a node name to update attribute and the value is a dictionary with attribute name and its value.
    :return: generated graph.
    """
    graph = Graph()
    for node_1, node_2, attr in edges:
        if node_1 not in graph.nodes():
            graph.add_node(node_1, **deepcopy(nodes_attrs[node_1]))
        if node_2 not in graph.nodes():
            graph.add_node(node_2, **deepcopy(nodes_attrs[node_2]))
        graph.add_edge(node_1, node_2, **attr)
    if update_attributes is not None:
        for node_name, new_attrs in update_attributes.items():
            assert (node_name in graph.nodes())
            for attr, value in new_attrs.items():
                graph.node[node_name][attr] = value
    return graph


class FakeAttr:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, item):
        return getattr(self, item)


class FakeNode:
    def __init__(self, pl, ml):
        self.pb = pl
        self.model_pb = ml
        self.graph = FakeAttr()
        self.graph.graph = {}
        self.update_node = lambda: None

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, item):
        return getattr(self, item)
