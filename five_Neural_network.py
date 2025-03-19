"""For this assignment we will Complete Neural Network"""

from enum import Enum
from abc import ABC, abstractmethod
from math import exp, floor, sqrt
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication


class DataMismatchError(Exception):
    """ Label and example lists have different lengths"""


class NNData:
    """
    Maintain and dispense examples for use by a Neural
    Network Application
    """

    class Order(Enum):
        """ Indicate whether data will be shuffled for each new epoch """
        RANDOM = 0
        SEQUENTIAL = 1

    class Set(Enum):
        """ Indicate which set should be accessed or manipulated """
        TRAIN = 0
        TEST = 1

    @staticmethod
    def percentage_limiter(percentage: float):
        """ Ensure that percentage is bounded between 0 and 1 """
        return min(1.0, max(percentage, 0.0))

    def __init__(self, features=None, labels=None, train_factor=.9):
        if features is None:
            features = []
        if labels is None:
            labels = []
        self._labels = None
        self._features = None
        self._train_indices = []
        self._test_indices = []
        self._train_pool = deque()
        self._test_pool = deque()
        self._train_factor = NNData.percentage_limiter(train_factor)
        self.load_data(features, labels)

    def _clear_data(self):
        """ Reset features and labels, and make sure all
        indices are reset as well
        """
        self._features = None
        self._labels = None
        self.split_set()

    def load_data(self, features: list = None, labels: list = None):
        """ Load feature and label data, with some checks to ensure
        that data is valid
        """
        if features is None or labels is None:
            self._clear_data()
            return
        if len(features) != len(labels):
            self._clear_data()
            raise DataMismatchError("Label and example lists have "
                                    "different lengths")
        if len(features) > 0:
            if not (isinstance(features[0], list)
                    and isinstance(labels[0], list)):
                self._clear_data()
                raise ValueError("Label and example lists must be "
                                 "homogeneous numeric lists of lists")
        try:
            self._features = np.array(features, dtype=float)
            self._labels = np.array(labels, dtype=float)
        except ValueError:
            self._clear_data()
            raise ValueError("Label and example lists must be homogeneous "
                             "and numeric lists of lists")
        self.split_set()

    def split_set(self, new_train_factor=None):
        if new_train_factor is not None:
            self._train_factor = NNData.percentage_limiter(new_train_factor)
        if self._features is None or len(self._features) == 0:
            self._train_indices = []
            self._test_indices = []
            return
        total_set_size = len(self._features)
        train_set_size = floor(total_set_size * self._train_factor)
        self._train_indices = random.sample(range(total_set_size),
                                            train_set_size)
        self._test_indices = list(set(range(total_set_size)) -
                                  set(self._train_indices))
        random.shuffle(self._train_indices)
        random.shuffle(self._test_indices)
        self.prime_data()

    def prime_data(self, my_set=None, order=None):
        if order is None:
            order = NNData.Order.SEQUENTIAL
        if my_set is not NNData.Set.TRAIN:  # this means we need to prime test
            test_indices_temp = list(self._test_indices)
            if order == NNData.Order.RANDOM:
                random.shuffle(test_indices_temp)
            self._test_pool = deque(test_indices_temp)
        if my_set is not NNData.Set.TEST:
            train_indices_temp = list(self._train_indices)
            if order == NNData.Order.RANDOM:
                random.shuffle(train_indices_temp)
            self._train_pool = deque(train_indices_temp)

    def get_one_item(self, target_set=None):
        if (target_set == NNData.Set.TRAIN or target_set is None) and \
                len(self._train_pool) != 0:
            idx = self._train_pool.popleft()
            return self._features[idx], self._labels[idx]
        elif target_set == NNData.Set.TEST and len(self._test_pool) != 0:
            idx = self._test_pool.popleft()
            return self._features[idx], self._labels[idx]
        else:
            return None, None

    def number_of_samples(self, target_set=None):
        if target_set == NNData.Set.TEST:
            return len(self._test_indices)
        elif target_set == NNData.Set.TRAIN:
            return len(self._train_indices)
        else:
            return len(self._test_indices) + len(self._train_indices)

    def pool_is_empty(self, target_set=None):
        if target_set is None or target_set == NNData.Set.TRAIN:
            return len(self._train_pool) == 0
        else:
            return len(self._test_pool) == 0


class LayerType(Enum):
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2


class MultiLinkNode(ABC):
    """
    This is an abstract base class that will be the starting
    point for our eventual FFBPNeurode class.
    """

    class Side(Enum):
        UPSTREAM = 0
        DOWNSTREAM = 1

    def __init__(self):
        self._reporting_nodes = {MultiLinkNode.Side.UPSTREAM: 0,
                                 MultiLinkNode.Side.DOWNSTREAM: 0}
        self._reference_value = {MultiLinkNode.Side.UPSTREAM: 0,
                                 MultiLinkNode.Side.DOWNSTREAM: 0}
        self._neighbors = {MultiLinkNode.Side.UPSTREAM: [],
                           MultiLinkNode.Side.DOWNSTREAM: []}

    def __str__(self):
        list_up = []
        list_down = []
        for nnode in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            list_up.append(str(id(nnode)))
        for nnode in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            list_down.append(str(id(nnode)))
        id_node = f"The ID of node is {id(self)}."
        id_up = "The ID of upstream neighbor are "
        id_down = "The ID of downstream neighbor are "
        if list_up:
            id_up += ",".join(list_up) + "."
        else:
            id_up = "There is no upstream neighbor."
        if list_down:
            id_down += ",".join(list_down) + "."
        else:
            id_down = "There is no downstream neighbor."
        return id_node + " " + id_up + " " + id_down

    @abstractmethod
    def _process_new_neighbor(self, node, side):
        pass

    def reset_neighbors(self, nodes: list, side: Side):
        self._neighbors[side] = nodes.copy()
        for node in nodes:
            self._process_new_neighbor(node, side)
        self._reference_value[side] = (1 << (len(nodes))) - 1
        self._reporting_nodes[side] = 0


class Neurode(MultiLinkNode):

    def __init__(self, node_type, learning_rate=.05):
        self._value = 0
        self._node_type = node_type
        self._learning_rate = learning_rate
        self._weights = {}
        super().__init__()

    def _process_new_neighbor(self, node, side):
        if side == MultiLinkNode.Side.UPSTREAM:
            self._weights[node] = random.random()

    def _check_in(self, node, side):
        idx = self._neighbors[side].index(node)
        self._reporting_nodes[side] |= 1 << idx
        if self._reporting_nodes[side] == self._reference_value[side]:
            self._reporting_nodes[side] = 0
            return True
        else:
            return False

    def get_weight(self, node):
        return self._weights[node]

    @property
    def value(self):
        return self._value

    @property
    def node_type(self):
        return self._node_type

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self._learning_rate = learning_rate


class FFNeurode(Neurode):

    def __init__(self, my_type):
        super().__init__(my_type)

    @staticmethod
    def _sigmoid(value):
        return 1 / (1 + exp(-value))

    def _calculate_value(self):
        value = 0
        for up_nei in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            value += self._weights[up_nei] * up_nei.value
        self._value = self._sigmoid(value)

    def _fire_downstream(self):
        for down_nei in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            down_nei.data_ready_upstream(self)

    def data_ready_upstream(self, node):
        if self._check_in(node, MultiLinkNode.Side.UPSTREAM):
            self._calculate_value()
            self._fire_downstream()

    def set_input(self, input_value):
        self._value = input_value
        for down_nei in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            down_nei.data_ready_upstream(self)


class BPNeurode(Neurode):

    def __init__(self, my_type):
        super().__init__(my_type)
        self._delta = 0

    @staticmethod
    def _sigmoid_derivative(value):
        return value * (1.0 - value)

    def _calculate_delta(self, expected_value=None):
        if self._node_type == LayerType.OUTPUT:
            error = expected_value - self.value
            self._delta = error * self._sigmoid_derivative(self.value)
        else:
            self._delta = 0
            for down_nei in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
                self._delta += down_nei.delta * down_nei.get_weight(self)
            self._delta *= self._sigmoid_derivative(self.value)

    def data_ready_downstream(self, node):
        """Downstream neurodes call this method when they have data ready."""
        val = self._check_in(node, MultiLinkNode.Side.DOWNSTREAM)
        if val:
            self._calculate_delta()
            self._fire_upstream()
            self._update_weights()

    def set_expected(self, expected_value):
        """Used by the client to directly set the value of an output layer
        neurode."""
        self._calculate_delta(expected_value)
        self._fire_upstream()

    def adjust_weights(self, node, adjustment):
        """work for the downstream nodes"""
        self._weights[node] += adjustment

    def _update_weights(self):
        """update all the downstream nodes of the node"""
        for down_nei in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            adjustment = self.value * down_nei.delta * down_nei.learning_rate
            down_nei.adjust_weights(self, adjustment)

    def _fire_upstream(self):
        for up_nei in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            up_nei.data_ready_downstream(self)

    @property
    def delta(self):
        return self._delta


class FFBPNeurode(FFNeurode, BPNeurode):
    pass


class Node:
    def __init__(self, data=None):
        self.data = data
        self.prev = None
        self.next = None


class DoublyLinkedList:
    class EmptyListError(Exception):
        def __init__(self, message):
            self._message = message

    def __init__(self):
        self._head = None
        self._tail = None
        self._curr = None

    def move_forward(self):
        if self._curr is None:
            raise DoublyLinkedList.EmptyListError("It's an empty double "
                                                  "linked list.")
        if self._curr == self._tail:
            raise IndexError("Current node can not move beyond the end of "
                             "the list.")
        self._curr = self._curr.next
        return self._curr.data

    def move_back(self):
        if self._head is None:
            raise DoublyLinkedList.EmptyListError("It's an empty double "
                                                  "linked list.")
        if self._curr == self._head:
            raise IndexError("Current node can not move beyond the beginning "
                             "of the list.")
        self._curr = self._curr.prev
        return self._curr.data

    def reset_to_head(self):
        if self._head is None:
            raise DoublyLinkedList.EmptyListError("It's an empty double "
                                                  "linked list.")
        self._curr = self._head
        return self._curr.data

    def reset_to_tail(self):
        if self._tail is None:
            raise DoublyLinkedList.EmptyListError("It's an empty double "
                                                  "linked list.")
        self._curr = self._tail
        return self._curr.data

    def add_to_head(self, data):
        new_node = Node(data)
        if self._head is not None:
            new_node.next, self._head.prev = self._head, new_node
            self._head = new_node
        else:
            self._head = new_node
            self._tail = new_node
        self.reset_to_head()

    def add_after_cur(self, data):
        if self._curr is None:
            self.add_to_head(data)
        else:
            new_node = Node(data)
            new_node.prev, new_node.next = self._curr, self._curr.next
            if self._curr == self._tail:
                self._tail = new_node
            if self._curr.next:
                self._curr.next.prev = new_node
            self._curr.next = new_node

    def remove_from_head(self):
        if self._head is None:
            raise DoublyLinkedList.EmptyListError("It's an empty double "
                                                  "linked list.")
        ret_val = self._head.data
        self._head = self._head.next
        self._head.prev = None
        return ret_val

    def remove_after_cur(self):
        if self._head is None:
            raise DoublyLinkedList.EmptyListError("It's an empty double "
                                                  "linked list.")
        if self._curr == self._tail:
            raise IndexError("Current node is the tail.")
        ret_val = self._curr.next.data
        if self._curr.next == self._tail:
            self._tail = self._curr
        if self._curr.next.next:
            self._curr.next.next.prev = self._curr
        self._curr.next = self._curr.next.next
        return ret_val

    def get_current_data(self):
        if self._head is None:
            raise DoublyLinkedList.EmptyListError("It's an empty double "
                                                  "linked list.")
        return self._curr.data


class LayerList(DoublyLinkedList):

    def __init__(self, inputs: int, outputs: int, neurode_type: type(Neurode)):
        super().__init__()
        self._neurode_type = neurode_type
        self._input_nodes = [self._neurode_type(LayerType.INPUT) for _ in
                             range(inputs)]
        self._output_nodes = [self._neurode_type(LayerType.OUTPUT) for _ in
                              range(outputs)]
        self.connect_layer(self._input_nodes, self._output_nodes)
        self.add_to_head(self._input_nodes)
        self.add_after_cur(self._output_nodes)
        self.reset_to_head()

    def connect_layer(self, prev, after):
        for node in prev:
            node.reset_neighbors(after, self._neurode_type.Side.DOWNSTREAM)
        for node in after:
            node.reset_neighbors(prev, self._neurode_type.Side.UPSTREAM)

    def add_layer(self, num_nodes: int):
        hide_list = [self._neurode_type(LayerType.HIDDEN) for _ in
                     range(num_nodes)]
        if self.get_current_data()[0].node_type == LayerType.OUTPUT:
            raise IndexError("Current layer is output layer, and can not add "
                             "layer after it.")
        prev = self.get_current_data()
        after = self._curr.next.data
        self.add_after_cur(hide_list)
        self.connect_layer(prev, hide_list)
        self.connect_layer(hide_list, after)

    def remove_layer(self):
        if self.get_current_data()[0].node_type == LayerType.OUTPUT or \
                self._curr.next.data[0].node_type == LayerType.OUTPUT:
            raise IndexError("Current layer is output layer, and can not "
                             "remove layer after it.")
        prev = self.get_current_data()
        self.remove_after_cur()
        after = self._curr.next.data
        self.connect_layer(prev, after)

    @property
    def input_nodes(self):
        return self._input_nodes

    @property
    def output_nodes(self):
        return self._output_nodes


class FFBPNetwork:

    class EmptySetException(Exception):
        def __init__(self, message):
            self._message = message

    def __init__(self, num_inputs: int, num_outputs: int):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.my_layers = LayerList(num_inputs, num_outputs, FFBPNeurode)

    def add_hidden_layer(self, num_nodes: int, position=0):
        self.my_layers.reset_to_head()
        for _ in range(position):
            self.my_layers._curr = self.my_layers.move_forward()
        self.my_layers.add_layer(num_nodes)

    def train(self, data_set: NNData, epochs=1000, verbosity=2,
              order=NNData.Order.RANDOM):
        if data_set.pool_is_empty():
            raise FFBPNetwork.EmptySetException("The training set is empty")
        sum_error = 0
        for epoch in range(epochs):
            data_set.prime_data(order)
            sum_error = 0
            while not data_set.pool_is_empty():
                input_val, output_val, expected_val = [], [], []
                feature, label = data_set.get_one_item()
                for i in range(self.num_inputs):
                    input_val.append(feature[i])
                    self.my_layers.input_nodes[i].set_input(feature[i])
                for i in range(self.num_outputs):
                    val = self.my_layers.output_nodes[i].value
                    expected_val.append(label[i])
                    output_val.append(val)
                    error = label[i] - val
                    sum_error += error ** 2
                    self.my_layers.output_nodes[i].set_expected(label[i])
                if verbosity > 1 and epoch % 1000 == 0:
                    print(f"Sample {input_val} ", f"expected {expected_val}",
                          f"produced {output_val}")
            if verbosity > 0 and epoch % 100 == 0:
                samples = data_set.number_of_samples(NNData.Set.TRAIN)
                RMSE = sqrt(sum_error / (samples * self.num_outputs))
                print(f"Epoch {epoch} RMSE = {RMSE}")
        samples = data_set.number_of_samples(NNData.Set.TRAIN)
        RMSE = sqrt(sum_error / (samples * self.num_outputs))
        print(f"Final Epoch RMSE = {RMSE}")

    def test(self, data_set: NNData, order=NNData.Order.SEQUENTIAL):
        if data_set.pool_is_empty(data_set.Set.TEST):
            raise FFBPNetwork.EmptySetException("The testing set is empty")
        sum_error = 0
        data_set.prime_data(order)
        print("TESTING")
        while not data_set.pool_is_empty(data_set.Set.TEST):
            input_val, output_val, expected_val = [], [], []
            feature, label = data_set.get_one_item(data_set.Set.TEST)
            for i in range(self.num_inputs):
                input_val.append(feature[i])
                self.my_layers.input_nodes[i].set_input(feature[i])
            for i in range(self.num_outputs):
                val = self.my_layers.output_nodes[i].value
                output_val.append(val)
                expected_val.append(label[i])
                error = label[i] - val
                sum_error += error ** 2
                self.my_layers.output_nodes[i].set_expected(label[i])
            print(f"Sample {input_val} ", f"expected {expected_val}",
                  f"produced {output_val}")
        samples = data_set.number_of_samples(NNData.Set.TEST)
        RMSE = sqrt(sum_error / (samples * self.num_outputs))
        print(f"RMSE is {RMSE}")


def run_iris():
    network = FFBPNetwork(4, 3)
    network.add_hidden_layer(3)
    Iris_X = [[5.1, 3.5, 1.4, 0.2], [4.9, 3, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2],
              [4.6, 3.1, 1.5, 0.2],
              [5, 3.6, 1.4, 0.2], [5.4, 3.9, 1.7, 0.4], [4.6, 3.4, 1.4, 0.3],
              [5, 3.4, 1.5, 0.2],
              [4.4, 2.9, 1.4, 0.2], [4.9, 3.1, 1.5, 0.1], [5.4, 3.7, 1.5, 0.2],
              [4.8, 3.4, 1.6, 0.2],
              [4.8, 3, 1.4, 0.1], [4.3, 3, 1.1, 0.1], [5.8, 4, 1.2, 0.2],
              [5.7, 4.4, 1.5, 0.4],
              [5.4, 3.9, 1.3, 0.4], [5.1, 3.5, 1.4, 0.3], [5.7, 3.8, 1.7, 0.3],
              [5.1, 3.8, 1.5, 0.3],
              [5.4, 3.4, 1.7, 0.2], [5.1, 3.7, 1.5, 0.4], [4.6, 3.6, 1, 0.2],
              [5.1, 3.3, 1.7, 0.5],
              [4.8, 3.4, 1.9, 0.2], [5, 3, 1.6, 0.2], [5, 3.4, 1.6, 0.4],
              [5.2, 3.5, 1.5, 0.2],
              [5.2, 3.4, 1.4, 0.2], [4.7, 3.2, 1.6, 0.2], [4.8, 3.1, 1.6, 0.2],
              [5.4, 3.4, 1.5, 0.4],
              [5.2, 4.1, 1.5, 0.1], [5.5, 4.2, 1.4, 0.2], [4.9, 3.1, 1.5, 0.1],
              [5, 3.2, 1.2, 0.2],
              [5.5, 3.5, 1.3, 0.2], [4.9, 3.1, 1.5, 0.1], [4.4, 3, 1.3, 0.2],
              [5.1, 3.4, 1.5, 0.2],
              [5, 3.5, 1.3, 0.3], [4.5, 2.3, 1.3, 0.3], [4.4, 3.2, 1.3, 0.2],
              [5, 3.5, 1.6, 0.6],
              [5.1, 3.8, 1.9, 0.4], [4.8, 3, 1.4, 0.3], [5.1, 3.8, 1.6, 0.2],
              [4.6, 3.2, 1.4, 0.2],
              [5.3, 3.7, 1.5, 0.2], [5, 3.3, 1.4, 0.2], [7, 3.2, 4.7, 1.4],
              [6.4, 3.2, 4.5, 1.5],
              [6.9, 3.1, 4.9, 1.5], [5.5, 2.3, 4, 1.3], [6.5, 2.8, 4.6, 1.5],
              [5.7, 2.8, 4.5, 1.3],
              [6.3, 3.3, 4.7, 1.6], [4.9, 2.4, 3.3, 1], [6.6, 2.9, 4.6, 1.3],
              [5.2, 2.7, 3.9, 1.4], [5, 2, 3.5, 1],
              [5.9, 3, 4.2, 1.5], [6, 2.2, 4, 1], [6.1, 2.9, 4.7, 1.4],
              [5.6, 2.9, 3.6, 1.3], [6.7, 3.1, 4.4, 1.4],
              [5.6, 3, 4.5, 1.5], [5.8, 2.7, 4.1, 1], [6.2, 2.2, 4.5, 1.5],
              [5.6, 2.5, 3.9, 1.1],
              [5.9, 3.2, 4.8, 1.8], [6.1, 2.8, 4, 1.3], [6.3, 2.5, 4.9, 1.5],
              [6.1, 2.8, 4.7, 1.2],
              [6.4, 2.9, 4.3, 1.3], [6.6, 3, 4.4, 1.4], [6.8, 2.8, 4.8, 1.4],
              [6.7, 3, 5, 1.7], [6, 2.9, 4.5, 1.5],
              [5.7, 2.6, 3.5, 1], [5.5, 2.4, 3.8, 1.1], [5.5, 2.4, 3.7, 1],
              [5.8, 2.7, 3.9, 1.2],
              [6, 2.7, 5.1, 1.6], [5.4, 3, 4.5, 1.5], [6, 3.4, 4.5, 1.6],
              [6.7, 3.1, 4.7, 1.5],
              [6.3, 2.3, 4.4, 1.3], [5.6, 3, 4.1, 1.3], [5.5, 2.5, 4, 1.3],
              [5.5, 2.6, 4.4, 1.2],
              [6.1, 3, 4.6, 1.4], [5.8, 2.6, 4, 1.2], [5, 2.3, 3.3, 1],
              [5.6, 2.7, 4.2, 1.3], [5.7, 3, 4.2, 1.2],
              [5.7, 2.9, 4.2, 1.3], [6.2, 2.9, 4.3, 1.3], [5.1, 2.5, 3, 1.1],
              [5.7, 2.8, 4.1, 1.3],
              [6.3, 3.3, 6, 2.5], [5.8, 2.7, 5.1, 1.9], [7.1, 3, 5.9, 2.1],
              [6.3, 2.9, 5.6, 1.8],
              [6.5, 3, 5.8, 2.2], [7.6, 3, 6.6, 2.1], [4.9, 2.5, 4.5, 1.7],
              [7.3, 2.9, 6.3, 1.8],
              [6.7, 2.5, 5.8, 1.8], [7.2, 3.6, 6.1, 2.5], [6.5, 3.2, 5.1, 2],
              [6.4, 2.7, 5.3, 1.9],
              [6.8, 3, 5.5, 2.1], [5.7, 2.5, 5, 2], [5.8, 2.8, 5.1, 2.4],
              [6.4, 3.2, 5.3, 2.3], [6.5, 3, 5.5, 1.8],
              [7.7, 3.8, 6.7, 2.2], [7.7, 2.6, 6.9, 2.3], [6, 2.2, 5, 1.5],
              [6.9, 3.2, 5.7, 2.3],
              [5.6, 2.8, 4.9, 2], [7.7, 2.8, 6.7, 2], [6.3, 2.7, 4.9, 1.8],
              [6.7, 3.3, 5.7, 2.1],
              [7.2, 3.2, 6, 1.8], [6.2, 2.8, 4.8, 1.8], [6.1, 3, 4.9, 1.8],
              [6.4, 2.8, 5.6, 2.1],
              [7.2, 3, 5.8, 1.6], [7.4, 2.8, 6.1, 1.9], [7.9, 3.8, 6.4, 2],
              [6.4, 2.8, 5.6, 2.2],
              [6.3, 2.8, 5.1, 1.5], [6.1, 2.6, 5.6, 1.4], [7.7, 3, 6.1, 2.3],
              [6.3, 3.4, 5.6, 2.4],
              [6.4, 3.1, 5.5, 1.8], [6, 3, 4.8, 1.8], [6.9, 3.1, 5.4, 2.1],
              [6.7, 3.1, 5.6, 2.4],
              [6.9, 3.1, 5.1, 2.3], [5.8, 2.7, 5.1, 1.9], [6.8, 3.2, 5.9, 2.3],
              [6.7, 3.3, 5.7, 2.5],
              [6.7, 3, 5.2, 2.3], [6.3, 2.5, 5, 1.9], [6.5, 3, 5.2, 2],
              [6.2, 3.4, 5.4, 2.3], [5.9, 3, 5.1, 1.8]]
    Iris_Y = [[1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ]]
    print("This is the sample run of iris.")
    data = NNData(Iris_X, Iris_Y, .7)
    network.train(data, 1000, order=NNData.Order.RANDOM)
    network.test(data)


def run_sin():
    network = FFBPNetwork(1, 1)
    network.add_hidden_layer(3)
    sin_X = [[0], [0.01], [0.02], [0.03], [0.04], [0.05], [0.06], [0.07],
             [0.08], [0.09], [0.1], [0.11], [0.12],
             [0.13], [0.14], [0.15], [0.16], [0.17], [0.18], [0.19], [0.2],
             [0.21], [0.22], [0.23], [0.24], [0.25],
             [0.26], [0.27], [0.28], [0.29], [0.3], [0.31], [0.32], [0.33],
             [0.34], [0.35], [0.36], [0.37], [0.38],
             [0.39], [0.4], [0.41], [0.42], [0.43], [0.44], [0.45], [0.46],
             [0.47], [0.48], [0.49], [0.5], [0.51],
             [0.52], [0.53], [0.54], [0.55], [0.56], [0.57], [0.58], [0.59],
             [0.6], [0.61], [0.62], [0.63], [0.64],
             [0.65], [0.66], [0.67], [0.68], [0.69], [0.7], [0.71], [0.72],
             [0.73], [0.74], [0.75], [0.76], [0.77],
             [0.78], [0.79], [0.8], [0.81], [0.82], [0.83], [0.84], [0.85],
             [0.86], [0.87], [0.88], [0.89], [0.9],
             [0.91], [0.92], [0.93], [0.94], [0.95], [0.96], [0.97], [0.98],
             [0.99], [1], [1.01], [1.02], [1.03],
             [1.04], [1.05], [1.06], [1.07], [1.08], [1.09], [1.1], [1.11],
             [1.12], [1.13], [1.14], [1.15], [1.16],
             [1.17], [1.18], [1.19], [1.2], [1.21], [1.22], [1.23], [1.24],
             [1.25], [1.26], [1.27], [1.28], [1.29],
             [1.3], [1.31], [1.32], [1.33], [1.34], [1.35], [1.36], [1.37],
             [1.38], [1.39], [1.4], [1.41], [1.42],
             [1.43], [1.44], [1.45], [1.46], [1.47], [1.48], [1.49], [1.5],
             [1.51], [1.52], [1.53], [1.54], [1.55],
             [1.56], [1.57]]
    sin_Y = [[0], [0.00999983333416666], [0.0199986666933331],
             [0.0299955002024957], [0.0399893341866342],
             [0.0499791692706783], [0.0599640064794446], [0.0699428473375328],
             [0.0799146939691727],
             [0.089878549198011], [0.0998334166468282], [0.109778300837175],
             [0.119712207288919],
             [0.129634142619695], [0.139543114644236], [0.149438132473599],
             [0.159318206614246],
             [0.169182349066996], [0.179029573425824], [0.188858894976501],
             [0.198669330795061], [0.2084598998461],
             [0.218229623080869], [0.227977523535188], [0.237702626427135],
             [0.247403959254523],
             [0.257080551892155], [0.266731436688831], [0.276355648564114],
             [0.285952225104836], [0.29552020666134],
             [0.305058636443443], [0.314566560616118], [0.324043028394868],
             [0.333487092140814],
             [0.342897807455451], [0.35227423327509], [0.361615431964962],
             [0.370920469412983], [0.380188415123161],
             [0.389418342308651], [0.398609327984423], [0.40776045305957],
             [0.416870802429211], [0.425939465066],
             [0.43496553411123], [0.44394810696552], [0.452886285379068],
             [0.461779175541483], [0.470625888171158],
             [0.479425538604203], [0.488177246882907], [0.496880137843737],
             [0.505533341204847],
             [0.514135991653113], [0.522687228930659], [0.531186197920883],
             [0.539632048733969],
             [0.548023936791874], [0.556361022912784], [0.564642473395035],
             [0.572867460100481],
             [0.581035160537305], [0.58914475794227], [0.597195441362392],
             [0.60518640573604], [0.613116851973434],
             [0.62098598703656], [0.628793024018469], [0.636537182221968],
             [0.644217687237691], [0.651833771021537],
             [0.659384671971473], [0.666869635003698], [0.674287911628145],
             [0.681638760023334],
             [0.688921445110551], [0.696135238627357], [0.70327941920041],
             [0.710353272417608], [0.717356090899523],
             [0.724287174370143], [0.731145829726896], [0.737931371109963],
             [0.744643119970859],
             [0.751280405140293], [0.757842562895277], [0.764328937025505],
             [0.770738878898969],
             [0.777071747526824], [0.783326909627483], [0.78950373968995],
             [0.795601620036366], [0.801619940883777],
             [0.807558100405114], [0.813415504789374], [0.819191568300998],
             [0.82488571333845], [0.83049737049197],
             [0.836025978600521], [0.841470984807897], [0.846831844618015],
             [0.852108021949363],
             [0.857298989188603], [0.862404227243338], [0.867423225594017],
             [0.872355482344986],
             [0.877200504274682], [0.881957806884948], [0.886626914449487],
             [0.891207360061435],
             [0.895698685680048], [0.900100442176505], [0.904412189378826],
             [0.908633496115883],
             [0.912763940260521], [0.916803108771767], [0.920750597736136],
             [0.92460601240802], [0.928368967249167],
             [0.932039085967226], [0.935616001553386], [0.939099356319068],
             [0.942488801931697],
             [0.945783999449539], [0.948984619355586], [0.952090341590516],
             [0.955100855584692],
             [0.958015860289225], [0.960835064206073], [0.963558185417193],
             [0.966184951612734],
             [0.968715100118265], [0.971148377921045], [0.973484541695319],
             [0.975723357826659],
             [0.977864602435316], [0.979908061398614], [0.98185353037236],
             [0.983700814811277], [0.98544972998846],
             [0.98710010101385], [0.98865176285172], [0.990104560337178],
             [0.991458348191686], [0.992712991037588],
             [0.993868363411645], [0.994924349777581], [0.99588084453764],
             [0.996737752043143], [0.997494986604054],
             [0.998152472497548], [0.998710143975583], [0.999167945271476],
             [0.999525830605479],
             [0.999783764189357], [0.999941720229966], [0.999999682931835]]
    print("This is the sample run of sin.")
    data = NNData(sin_X, sin_Y, .1)
    network.train(data, 10001, order=NNData.Order.RANDOM)
    network.test(data)


def run_XOR():

    """ Load the complete population of XOR examples.  Note that the
    nature of this set requires 100% to be placed in training.
    """
    network = FFBPNetwork(2, 1)
    network.add_hidden_layer(5)
    XOR_X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    XOR_Y = [[0], [1], [1], [0]]
    print("This is the sample run of XOR without bias.")
    data = NNData(XOR_X, XOR_Y, 1)
    network.train(data, 10001, order=NNData.Order.RANDOM)


def run_XOR_with_bias():

    network = FFBPNetwork(3, 1)
    network.add_hidden_layer(5)
    XOR_X = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
    XOR_Y = [[0], [1], [1], [0]]
    print("This is the sample run of XOR with bias.")
    data = NNData(XOR_X, XOR_Y, 1)
    network.train(data, 10001, order=NNData.Order.RANDOM)


def plot_sin(sin_in, sin_out):
    x_actual = np.linspace(0, np.pi / 2, 158)
    y_actual = np.sin(x_actual)
    # sort produced data by x_test
    data = sorted(list(zip(sin_in, sin_out)))
    # Generate x values for test sin graph
    x_test = list(zip(*data))[0]
    y_test = list(zip(*data))[1]
    # Plot actual sin graph
    plt.plot(x_actual, y_actual, color='blue', label='Actual')

    # Plot test sin graph
    plt.plot(x_test, y_test, color='orange', label='Test')

    # Set axis labels and title
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Actual vs Test Sin Graph')

    # Show legend
    plt.legend()

    # Show plot
    plt.show()

    # Start event loop
    app = QApplication([])
    app.exec_()


def run_trans_sin():
    network = FFBPNetwork(1, 1)
    network.add_hidden_layer(3)
    sin_X = [[0.0], [0.01], [0.02], [0.03], [0.04], [0.05], [0.06], [0.07],
             [0.08], [0.09], [0.1], [0.11], [0.12], [0.13], [0.14], [0.15],
             [0.16], [0.17], [0.18], [0.19], [0.2], [0.21], [0.22], [0.23],
             [0.24], [0.25], [0.26], [0.27], [0.28], [0.29], [0.3], [0.31],
             [0.32], [0.33], [0.34], [0.35], [0.36], [0.37], [0.38], [0.39],
             [0.4], [0.41], [0.42], [0.43], [0.44], [0.45], [0.46], [0.47],
             [0.48], [0.49], [0.5], [0.51], [0.52], [0.53], [0.54], [0.55],
             [0.56], [0.57], [0.58], [0.59], [0.6], [0.61], [0.62], [0.63],
             [0.64], [0.65], [0.66], [0.67], [0.68], [0.69], [0.7], [0.71],
             [0.72], [0.73], [0.74], [0.75], [0.76], [0.77], [0.78], [0.79],
             [0.8], [0.81], [0.82], [0.83], [0.84], [0.85], [0.86], [0.87],
             [0.88], [0.89], [0.9], [0.91], [0.92], [0.93], [0.94], [0.95],
             [0.96], [0.97], [0.98], [0.99], [1.0], [1.01], [1.02], [1.03],
             [1.04], [1.05], [1.06], [1.07], [1.08], [1.09], [1.1], [1.11],
             [1.12], [1.13], [1.14], [1.15], [1.16], [1.17], [1.18], [1.19],
             [1.2], [1.21], [1.22], [1.23], [1.24], [1.25], [1.26], [1.27],
             [1.28], [1.29], [1.3], [1.31], [1.32], [1.33], [1.34], [1.35],
             [1.36], [1.37], [1.38], [1.39], [1.4], [1.41], [1.42], [1.43],
             [1.44], [1.45], [1.46], [1.47], [1.48], [1.49], [1.5], [1.51],
             [1.52], [1.53], [1.54], [1.55], [1.56], [1.57], [1.58], [1.59],
             [1.6], [1.61], [1.62], [1.63], [1.64], [1.65], [1.66], [1.67],
             [1.68], [1.69], [1.7], [1.71], [1.72], [1.73], [1.74], [1.75],
             [1.76], [1.77], [1.78], [1.79], [1.8], [1.81], [1.82], [1.83],
             [1.84], [1.85], [1.86], [1.87], [1.88], [1.89], [1.9], [1.91],
             [1.92], [1.93], [1.94], [1.95], [1.96], [1.97], [1.98], [1.99],
             [2.0], [2.01], [2.02], [2.03], [2.04], [2.05], [2.06], [2.07],
             [2.08], [2.09], [2.1], [2.11], [2.12], [2.13], [2.14], [2.15],
             [2.16], [2.17], [2.18], [2.19], [2.2], [2.21], [2.22], [2.23],
             [2.24], [2.25], [2.26], [2.27], [2.28], [2.29], [2.3], [2.31],
             [2.32], [2.33], [2.34], [2.35], [2.36], [2.37], [2.38], [2.39],
             [2.4], [2.41], [2.42], [2.43], [2.44], [2.45], [2.46], [2.47],
             [2.48], [2.49], [2.5], [2.51], [2.52], [2.53], [2.54], [2.55],
             [2.56], [2.57], [2.58], [2.59], [2.6], [2.61], [2.62], [2.63],
             [2.64], [2.65], [2.66], [2.67], [2.68], [2.69], [2.7], [2.71],
             [2.72], [2.73], [2.74], [2.75], [2.76], [2.77], [2.78], [2.79],
             [2.8], [2.81], [2.82], [2.83], [2.84], [2.85], [2.86], [2.87],
             [2.88], [2.89], [2.9], [2.91], [2.92], [2.93], [2.94], [2.95],
             [2.96], [2.97], [2.98], [2.99], [3.0], [3.01], [3.02], [3.03],
             [3.04], [3.05], [3.06], [3.07], [3.08], [3.09], [3.1], [3.11],
             [3.12], [3.13], [3.14], [3.15], [3.16], [3.17], [3.18], [3.19],
             [3.2], [3.21], [3.22], [3.23], [3.24], [3.25], [3.26], [3.27],
             [3.28], [3.29], [3.3], [3.31], [3.32], [3.33], [3.34], [3.35],
             [3.36], [3.37], [3.38], [3.39], [3.4], [3.41], [3.42], [3.43],
             [3.44], [3.45], [3.46], [3.47], [3.48], [3.49], [3.5], [3.51],
             [3.52], [3.53], [3.54], [3.55], [3.56], [3.57], [3.58], [3.59],
             [3.6], [3.61], [3.62], [3.63], [3.64], [3.65], [3.66], [3.67],
             [3.68], [3.69], [3.7], [3.71], [3.72], [3.73], [3.74], [3.75],
             [3.76], [3.77], [3.78], [3.79], [3.8], [3.81], [3.82], [3.83],
             [3.84], [3.85], [3.86], [3.87], [3.88], [3.89], [3.9], [3.91],
             [3.92], [3.93], [3.94], [3.95], [3.96], [3.97], [3.98], [3.99],
             [4.0], [4.01], [4.02], [4.03], [4.04], [4.05], [4.06], [4.07],
             [4.08], [4.09], [4.1], [4.11], [4.12], [4.13], [4.14], [4.15],
             [4.16], [4.17], [4.18], [4.19], [4.2], [4.21], [4.22], [4.23],
             [4.24], [4.25], [4.26], [4.27], [4.28], [4.29], [4.3], [4.31],
             [4.32], [4.33], [4.34], [4.35], [4.36], [4.37], [4.38], [4.39],
             [4.4], [4.41], [4.42], [4.43], [4.44], [4.45], [4.46], [4.47],
             [4.48], [4.49], [4.5], [4.51], [4.52], [4.53], [4.54], [4.55],
             [4.56], [4.57], [4.58], [4.59], [4.6], [4.61], [4.62], [4.63],
             [4.64], [4.65], [4.66], [4.67], [4.68], [4.69], [4.7], [4.71],
             [4.72], [4.73], [4.74], [4.75], [4.76], [4.77], [4.78], [4.79],
             [4.8], [4.81], [4.82], [4.83], [4.84], [4.85], [4.86], [4.87],
             [4.88], [4.89], [4.9], [4.91], [4.92], [4.93], [4.94], [4.95],
             [4.96], [4.97], [4.98], [4.99], [5.0], [5.01], [5.02], [5.03],
             [5.04], [5.05], [5.06], [5.07], [5.08], [5.09], [5.1], [5.11],
             [5.12], [5.13], [5.14], [5.15], [5.16], [5.17], [5.18], [5.19],
             [5.2], [5.21], [5.22], [5.23], [5.24], [5.25], [5.26], [5.27],
             [5.28], [5.29], [5.3], [5.31], [5.32], [5.33], [5.34], [5.35],
             [5.36], [5.37], [5.38], [5.39], [5.4], [5.41], [5.42], [5.43],
             [5.44], [5.45], [5.46], [5.47], [5.48], [5.49], [5.5], [5.51],
             [5.52], [5.53], [5.54], [5.55], [5.56], [5.57], [5.58], [5.59],
             [5.6], [5.61], [5.62], [5.63], [5.64], [5.65], [5.66], [5.67],
             [5.68], [5.69], [5.7], [5.71], [5.72], [5.73], [5.74], [5.75],
             [5.76], [5.77], [5.78], [5.79], [5.8], [5.81], [5.82], [5.83],
             [5.84], [5.85], [5.86], [5.87], [5.88], [5.89], [5.9], [5.91],
             [5.92], [5.93], [5.94], [5.95], [5.96], [5.97], [5.98], [5.99],
             [6.0], [6.01], [6.02], [6.03], [6.04], [6.05], [6.06], [6.07],
             [6.08], [6.09], [6.1], [6.11], [6.12], [6.13], [6.14], [6.15],
             [6.16], [6.17], [6.18], [6.19], [6.2], [6.21], [6.22], [6.23],
             [6.24], [6.25], [6.26], [6.27], [6.28]]
    sin_Y = [[0.0], [0.00999983333416666], [0.01999866669333308],
             [0.02999550020249566], [0.03998933418663416],
             [0.04997916927067833], [0.0599640064794446],
             [0.06994284733753277], [0.0799146939691727],
             [0.08987854919801104], [0.09983341664682815],
             [0.10977830083717482], [0.11971220728891936],
             [0.12963414261969486], [0.1395431146442365],
             [0.14943813247359922], [0.15931820661424598],
             [0.16918234906699603], [0.17902957342582418],
             [0.18885889497650055], [0.1986693307950612],
             [0.20845989984609956], [0.21822962308086932],
             [0.2279775235351884], [0.23770262642713455], [0.2474039592545229],
             [0.2570805518921551], [0.26673143668883115],
             [0.27635564856411376], [0.28595222510483553],
             [0.29552020666133955], [0.3050586364434435],
             [0.31456656061611776], [0.32404302839486837],
             [0.3334870921408144], [0.3428978074554514], [0.35227423327508994],
             [0.361615431964962], [0.3709204694129826], [0.3801884151231614],
             [0.3894183423086506], [0.39860932798442295],
             [0.40776045305957015], [0.4168708024292108], [0.4259394650659996],
             [0.43496553411123023], [0.4439481069655198],
             [0.45288628537906833], [0.4617791755414829], [0.470625888171158],
             [0.47942553860420306], [0.48817724688290753],
             [0.49688013784373675], [0.5055333412048469], [0.5141359916531132],
             [0.5226872289306592], [0.5311861979208834], [0.5396320487339693],
             [0.5480239367918736], [0.5563610229127838], [0.5646424733950354],
             [0.5728674601004813], [0.5810351605373051], [0.5891447579422695],
             [0.5971954413623921], [0.6051864057360395], [0.6131168519734338],
             [0.6209859870365597], [0.6287930240184686], [0.636537182221968],
             [0.6442176872376911], [0.6518337710215366], [0.6593846719714731],
             [0.6668696350036979], [0.674287911628145], [0.6816387600233342],
             [0.6889214451105513], [0.6961352386273567], [0.7032794192004101],
             [0.7103532724176078], [0.7173560908995228], [0.7242871743701425],
             [0.7311458297268958], [0.7379313711099628], [0.7446431199708594],
             [0.7512804051402926], [0.757842562895277], [0.7643289370255051],
             [0.7707388788989693], [0.7770717475268238], [0.7833269096274834],
             [0.7895037396899504], [0.795601620036366], [0.8016199408837771],
             [0.8075581004051142], [0.8134155047893737], [0.8191915683009983],
             [0.82488571333845], [0.8304973704919704], [0.8360259786005205],
             [0.8414709848078965], [0.8468318446180152], [0.8521080219493629],
             [0.8572989891886034], [0.8624042272433384], [0.8674232255940167],
             [0.8723554823449863], [0.8772005042746815], [0.8819578068849475],
             [0.8866269144494872], [0.8912073600614353], [0.8956986856800477],
             [0.9001004421765051], [0.904412189378826], [0.9086334961158834],
             [0.9127639402605211], [0.9168031087717669], [0.9207505977361357],
             [0.9246060124080203], [0.9283689672491665], [0.9320390859672263],
             [0.9356160015533859], [0.9390993563190677], [0.9424888019316975],
             [0.9457839994495388], [0.9489846193555862], [0.9520903415905156],
             [0.9551008555846923], [0.9580158602892249], [0.9608350642060727],
             [0.963558185417193], [0.9661849516127341], [0.9687151001182652],
             [0.9711483779210447], [0.9734845416953194], [0.9757233578266591],
             [0.9778646024353164], [0.9799080613986142], [0.9818535303723599],
             [0.9837008148112767], [0.9854497299884604], [0.9871001010138504],
             [0.9886517628517196], [0.9901045603371778], [0.9914583481916864],
             [0.9927129910375885], [0.9938683634116448], [0.9949243497775809],
             [0.99588084453764], [0.9967377520431434], [0.9974949866040544],
             [0.9981524724975481], [0.998710143975583], [0.999167945271476],
             [0.9995258306054791], [0.999783764189357], [0.9999417202299662],
             [0.9999996829318346], [0.9999576464987402], [0.9998156151342908],
             [0.9995736030415051], [0.9992316344213905], [0.998789743470524],
             [0.9982479743776325], [0.9976063813191736], [0.9968650284539189],
             [0.9960239899165368], [0.9950833498101802], [0.994043202198076],
             [0.9929036510941186], [0.9916648104524687], [0.990326804156158],
             [0.9888897660047015], [0.9873538397007164], [0.9857191788355535],
             [0.9839859468739369], [0.9821543171376184], [0.9802244727880455],
             [0.9781966068080447], [0.9760709219825242], [0.9738476308781951],
             [0.9715269558223154], [0.9691091288804563], [0.9665943918332977],
             [0.9639829961524482], [0.9612752029752999], [0.9584712830789143],
             [0.955571516852944], [0.9525761942715953], [0.9494856148646303],
             [0.9463000876874144], [0.9430199312900106], [0.9396454736853249],
             [0.9361770523163061], [0.9326150140222005], [0.9289597150038693],
             [0.9252115207881684], [0.9213708061913956], [0.9174379552818097],
             [0.9134133613412251], [0.9092974268256817], [0.905090563325201],
             [0.9007931915226272], [0.8964057411515598], [0.8919286509533795],
             [0.8873623686333755], [0.8827073508159741], [0.877964062999078],
             [0.8731329795075164], [0.8682145834456126], [0.8632093666488737],
             [0.8581178296348088], [0.8529404815528763], [0.8476778401335697],
             [0.8423304316366456], [0.8368987907984977], [0.8313834607786832],
             [0.8257849931056082], [0.8201039476213743], [0.814340892425796],
             [0.8084964038195901], [0.8025710662467473], [0.7965654722360865],
             [0.7904802223420048], [0.7843159250844198], [0.7780731968879213],
             [0.7717526620201257], [0.7653549525292536], [0.7588807081809219],
             [0.7523305763941707], [0.74570521217672], [0.7390052780594709],
             [0.7322314440302515], [0.7253843874668195], [0.7184647930691263],
             [0.7114733527908443], [0.7044107657701761], [0.6972777382599378],
             [0.6900749835569364], [0.6828032219306397], [0.675463180551151],
             [0.6680555934164909], [0.6605812012792007], [0.6530407515722648],
             [0.6454349983343707], [0.6377647021345036], [0.6300306299958922],
             [0.6222335553193046], [0.6143742578057118], [0.6064535233783147],
             [0.5984721441039564], [0.5904309181139127], [0.5823306495240819],
             [0.5741721483545723], [0.5659562304487028], [0.5576837173914166],
             [0.5493554364271266], [0.5409722203769886], [0.5325349075556212],
             [0.5240443416872761], [0.5155013718214642], [0.5069068522480534],
             [0.49826164241183857], [0.4895666068265995],
             [0.48082261498864826], [0.47203054128988264],
             [0.4631912649303451], [0.45430566983030646],
             [0.44537464454187115], [0.4363990821601263], [0.4273798802338298],
             [0.418317940675659], [0.4092141696720173], [0.4000694775924195],
             [0.39088477889845213], [0.38166099205233167],
             [0.37239903942505526], [0.3630998472041683], [0.3537643453011427],
             [0.34439346725839], [0.33498815015590466], [0.32554933451756],
             [0.3160779642170538], [0.30657498638352293], [0.2970413513068324],
             [0.2874780123425444], [0.2778859258165868], [0.2682660509296179],
             [0.2586193496611108], [0.24894678667315256], [0.2392493292139824],
             [0.2295279470212642], [0.21978361222511697],
             [0.21001729925089913], [0.20022998472177053],
             [0.19042264736102704], [0.18059626789423291],
             [0.17075182895114532], [0.16089031496745576],
             [0.15101271208634381], [0.1411200080598672], [0.1312131921501838],
             [0.12129325503062977], [0.11136118868664958],
             [0.10141798631660189], [0.09146464223243676],
             [0.08150215176026913], [0.07153151114084326],
             [0.06155371742991315], [0.05156976839853464],
             [0.04158066243329049], [0.0315873984364539],
             [0.02159097572609596], [0.01159239393615828],
             [0.00159265291648683], [-0.00840724736714862],
             [-0.01840630693305381], [-0.02840352588360379],
             [-0.03839790450523538], [-0.04838844336841414],
             [-0.05837414342758009], [-0.06835400612104778],
             [-0.0783270334708653], [-0.0882922281826076],
             [-0.09824859374510868], [-0.10819513453010839],
             [-0.1181308558918178], [-0.12805476426637968],
             [-0.1379658672712273], [-0.14786317380431852],
             [-0.15774569414324865], [-0.16761244004421832],
             [-0.17746242484086058], [-0.1872946635429032],
             [-0.19710817293466987], [-0.20690197167339977],
             [-0.21667508038737965], [-0.22642652177388317],
             [-0.236155320696897], [-0.24586050428463704],
             [-0.2555411020268312], [-0.2651961458717734],
             [-0.274824670323124], [-0.28442571253646254],
             [-0.2939983124155676], [-0.30354151270842933],
             [-0.31305435910297025], [-0.322535900322479],
             [-0.3319851882207341], [-0.34140127787682095],
             [-0.35078322768961984], [-0.36013009947196856],
             [-0.3694409585444771], [-0.3787148738289981],
             [-0.3879509179417303], [-0.39714816728596014],
             [-0.4063057021444168], [-0.4154226067712463],
             [-0.42449796948358254], [-0.4335308827527178],
             [-0.44252044329485246], [-0.4514657521614231],
             [-0.4603659148289983], [-0.46922004128872713],
             [-0.47802724613534286], [-0.4867866486556994],
             [-0.495497372916845], [-0.5041585478536115],
             [-0.5127693073557238], [-0.5213287903544065],
             [-0.5298361409084934], [-0.5382905082900177],
             [-0.5466910470692872], [-0.5550369171994238], [-0.56332728410037],
             [-0.5715613187423438], [-0.5797381977287431],
             [-0.5878571033784827], [-0.5959172238077642],
             [-0.6039177530112606], [-0.6118578909427193],
             [-0.6197368435949633], [-0.6275538230792937],
             [-0.6353080477042756], [-0.6429987420539088],
             [-0.6506251370651673], [-0.6581864701049049],
             [-0.6656819850461192], [-0.6731109323435617],
             [-0.680472569108694], [-0.6877661591839738], [-0.694990973216472],
             [-0.7021462887308054], [-0.709231390201386],
             [-0.7162455691239705], [-0.7231881240865121],
             [-0.7300583608392995], [-0.7368555923643834],
             [-0.7435791389442745], [-0.7502283282299189],
             [-0.7568024953079282], [-0.7633009827670735],
             [-0.7697231407640244], [-0.7760683270883323],
             [-0.7823359072266527], [-0.7885252544261949],
             [-0.7946357497573973], [-0.8006667821758175],
             [-0.8066177485832405], [-0.8124880538879842],
             [-0.8182771110644103], [-0.8239843412116258],
             [-0.8296091736113709], [-0.8351510457850935],
             [-0.8406094035501945], [-0.8459837010754465],
             [-0.8512734009355744], [-0.8564779741650012],
             [-0.8615969003107404], [-0.8666296674844444],
             [-0.871575772413588], [-0.8764347204918015],
             [-0.8812060258283253], [-0.8858892112966027],
             [-0.8904838085819885], [-0.8949893582285835],
             [-0.8994054096851778], [-0.9037315213503057],
             [-0.9079672606164054], [-0.9121122039130803],
             [-0.9161659367494549], [-0.920128053755624],
             [-0.9239981587231878], [-0.9277758646448755],
             [-0.9314607937532425], [-0.9350525775584494],
             [-0.9385508568851079], [-0.941955281908201],
             [-0.9452655121880633], [-0.9484812167044256],
             [-0.9516020738895161], [-0.9546277716602163],
             [-0.957558007449271], [-0.9603924882355434],
             [-0.9631309305733167], [-0.9657730606206388],
             [-0.9683186141667072], [-0.9707673366582883],
             [-0.9731189832251739], [-0.9753733187046666],
             [-0.977530117665097], [-0.9795891644283669],
             [-0.9815502530915156], [-0.9834131875473108],
             [-0.9851777815038595], [-0.9868438585032365],
             [-0.9884112519391306], [-0.989879805073504],
             [-0.9912493710522668], [-0.9925198129199632],
             [-0.9936910036334645], [-0.9947628260746755],
             [-0.9957351730622452], [-0.9966079473622856],
             [-0.9973810616980933], [-0.9980544387588796],
             [-0.9986280112074989], [-0.9991017216871848],
             [-0.999475522827284], [-0.999749377247994], [-0.9999232575641008],
             [-0.999997146387718], [-0.9999710363300245],
             [-0.9998449300020044], [-0.9996188400141854],
             [-0.9992927889753779], [-0.9988668094904143],
             [-0.9983409441568876], [-0.9977152455608933],
             [-0.9969897762717695], [-0.9961646088358407],
             [-0.9952398257691626], [-0.9942155195492713],
             [-0.9930917926059354], [-0.9918687573109126],
             [-0.9905465359667132], [-0.9891252607943698],
             [-0.9876050739202153], [-0.9859861273616704],
             [-0.9842685830120416], [-0.9824526126243325],
             [-0.9805383977940689], [-0.9785261299411385],
             [-0.9764160102906497], [-0.9742082498528091],
             [-0.9719030694018208], [-0.9695006994538088],
             [-0.967001380243766], [-0.9644053617015304],
             [-0.9617129034267934], [-0.9589242746631385],
             [-0.9560397542711181], [-0.9530596307003675],
             [-0.9499842019607608], [-0.9468137755926089],
             [-0.9435486686359066], [-0.9401892075986283],
             [-0.9367357284240789], [-0.9331885764572976],
             [-0.9295481064105251], [-0.9258146823277321],
             [-0.9219886775482161], [-0.9180704746692668],
             [-0.9140604655079071], [-0.9099590510617106],
             [-0.9057666414687044], [-0.9014836559663549],
             [-0.8971105228496424], [-0.8926476794282348],
             [-0.8880955719827542], [-0.8834546557201531],
             [-0.8787253947281898], [-0.8739082619290224],
             [-0.869003739031916], [-0.8640123164850744],
             [-0.8589344934265921], [-0.8537707776345433],
             [-0.8485216854762042], [-0.8431877418564168],
             [-0.8377694801650978], [-0.8322674422239013],
             [-0.8266821782320357], [-0.821014246711247],
             [-0.8152642144499636], [-0.8094326564466193],
             [-0.8035201558521554], [-0.7975273039117043],
             [-0.7914546999054661], [-0.7853029510887807],
             [-0.7790726726314032], [-0.7727644875559871],
             [-0.7663790266757843], [-0.759916928531561],
             [-0.7533788393277465], [-0.7467654128678123],
             [-0.7400773104888944], [-0.7333152009956565],
             [-0.726479760593413], [-0.7195716728205075],
             [-0.7125916284799616], [-0.7055403255703919],
             [-0.6984184692162135], [-0.6912267715971264],
             [-0.6839659518769006], [-0.6766367361314568],
             [-0.669239857276262], [-0.6617760549930369],
             [-0.6542460756557913], [-0.6466506722561834],
             [-0.6389906043282237], [-0.6312666378723208],
             [-0.6234795452786853], [-0.6156301052500863],
             [-0.6077191027239858], [-0.5997473287940438],
             [-0.5917155806310094], [-0.5836246614030073],
             [-0.575475380195217], [-0.5672685519289686],
             [-0.5590049972802488], [-0.5506855425976376],
             [-0.5423110198196697], [-0.5338822663916443],
             [-0.5254001251818792], [-0.5168654443974288],
             [-0.5082790774992583], [-0.4996418831169025],
             [-0.49095472496260095], [-0.48221847174493154],
             [-0.47343399708193507], [-0.46460217941375737],
             [-0.4557239019148047], [-0.44680005240543], [-0.4378315232631469],
             [-0.42881921133339584], [-0.4197640178398588],
             [-0.41066684829434086], [-0.4015286124062146],
             [-0.39235022399145386], [-0.38313260088125134],
             [-0.373876664830236], [-0.3645833414243014],
             [-0.35525355998804264], [-0.34588825349182883],
             [-0.3364883584585042], [-0.32705481486974064],
             [-0.31758856607203484], [-0.3080905586823781],
             [-0.2985617424935936], [-0.2890030703793611],
             [-0.27941549819892586], [-0.26979998470151617],
             [-0.260157491430468], [-0.25048898262707486],
             [-0.2407954251341592], [-0.23107778829939224],
             [-0.22133704387835867], [-0.21157416593738504],
             [-0.2017901307561289], [-0.191985916729955], [-0.182162504272095],
             [-0.17232087571561025], [-0.1624620152151542],
             [-0.15258690864856114], [-0.14269654351825772],
             [-0.13279190885251674], [-0.12287399510655005],
             [-0.11294379406346738], [-0.10300229873509784],
             [-0.0930505032626889], [-0.08308940281749641],
             [-0.0731199935012631], [-0.06314327224661277],
             [-0.05316023671735613], [-0.04317188520872868],
             [-0.03317921654755682], [-0.02318322999237946],
             [-0.01318492513352125], [-0.00318530179313799]]
    trans_Y = [[(1 + y) / 2] for _ in sin_Y for y in _]
    print("This is the sample run of transform sin.")
    data = NNData(sin_X, trans_Y, .1)
    network.train(data, 10001, order=NNData.Order.RANDOM)
    network.test(data)


if __name__ == "__main__":
    input("Please Enter to test iris")
    run_iris()
    input("\nPlease Enter to test sin")
    run_sin()
    input("\nPlease Enter to test XOR without bias")
    run_XOR()
    input("\nPlease Enter to test XOR with bias")
    run_XOR_with_bias()
    input("\nPlease Enter to test trans sin")
    run_trans_sin()


"""
"/Users/jiafei/Desktop/courses/W23 CS F003B 02W Intermed Software Desgn Python/
venv/bin/python" /Users/jiafei/Desktop/courses/W23 CS F003B 02W Intermed 
Software Desgn Python/projects/assignment five.py 
Please Enter to test iris
This is the sample run of iris.
Sample [6.4, 3.1, 5.5, 1.8]  expected [0.0, 0.0, 1.0] produced [0.86459403548295, 0.9049238067813041, 0.8158218656464129]
Sample [5.0, 2.0, 3.5, 1.0]  expected [0.0, 1.0, 0.0] produced [0.8624730346898658, 0.9035785567007859, 0.8160606119130807]
Sample [6.4, 2.7, 5.3, 1.9]  expected [0.0, 0.0, 1.0] produced [0.8609777652261359, 0.9040164214881334, 0.8136721366247534]
Sample [5.7, 4.4, 1.5, 0.4]  expected [1.0, 0.0, 0.0] produced [0.8589227092020547, 0.9028480826819197, 0.8141508619277057]
Sample [6.2, 2.8, 4.8, 1.8]  expected [0.0, 0.0, 1.0] produced [0.8594179338423826, 0.9019381784955021, 0.8114968645762137]
Sample [5.0, 3.6, 1.4, 0.2]  expected [1.0, 0.0, 0.0] produced [0.8569301033139564, 0.9004041399000962, 0.8116258431712494]
Sample [6.0, 2.9, 4.5, 1.5]  expected [0.0, 1.0, 0.0] produced [0.8578298358804256, 0.899775036691194, 0.8092909003436095]
Sample [6.4, 2.8, 5.6, 2.2]  expected [0.0, 0.0, 1.0] produced [0.8559413929963781, 0.8999312296040077, 0.806422226280061]
Sample [5.8, 2.8, 5.1, 2.4]  expected [0.0, 0.0, 1.0] produced [0.853968885241382, 0.8988220530537803, 0.8071182995332326]
Sample [5.3, 3.7, 1.5, 0.2]  expected [1.0, 0.0, 0.0] produced [0.8515144311328895, 0.8973453360943755, 0.807421495521694]
Sample [5.0, 3.4, 1.5, 0.2]  expected [1.0, 0.0, 0.0] produced [0.851672329703176, 0.8960358692842447, 0.8043042629611737]
Sample [6.2, 2.9, 4.3, 1.3]  expected [0.0, 1.0, 0.0] produced [0.8526425635789772, 0.8953675180295734, 0.8018613549547221]
Sample [7.7, 3.8, 6.7, 2.2]  expected [0.0, 0.0, 1.0] produced [0.8506701187658968, 0.8955629616822436, 0.7988729233608026]
Sample [6.3, 2.5, 4.9, 1.5]  expected [0.0, 1.0, 0.0] produced [0.8485510943926761, 0.8943273662630772, 0.7995907722223703]
Sample [6.3, 2.9, 5.6, 1.8]  expected [0.0, 0.0, 1.0] produced [0.8464636586381988, 0.89449643829626, 0.7965252012118036]
Sample [7.2, 3.0, 5.8, 1.6]  expected [0.0, 0.0, 1.0] produced [0.8443186247719394, 0.8933089013777754, 0.7973397214562739]
Sample [5.2, 3.4, 1.4, 0.2]  expected [1.0, 0.0, 0.0] produced [0.8414884591582151, 0.8915765989416672, 0.7975800585059217]
Sample [6.2, 2.2, 4.5, 1.5]  expected [0.0, 1.0, 0.0] produced [0.8424773737204713, 0.8907710917181624, 0.7949382832700125]
Sample [7.1, 3.0, 5.9, 2.1]  expected [0.0, 0.0, 1.0] produced [0.8403056594755927, 0.8909979726275656, 0.7918316194987494]
Sample [4.8, 3.4, 1.6, 0.2]  expected [1.0, 0.0, 0.0] produced [0.8372966315882291, 0.8891312120303381, 0.7920369033956358]
Sample [5.8, 2.7, 3.9, 1.2]  expected [0.0, 1.0, 0.0] produced [0.8383751485239744, 0.8883404725371224, 0.7893501150437532]
Sample [4.3, 3.0, 1.1, 0.1]  expected [1.0, 0.0, 0.0] produced [0.8344359510761618, 0.8871846510109529, 0.7846630583794286]
Sample [6.0, 2.7, 5.1, 1.6]  expected [0.0, 1.0, 0.0] produced [0.8365821354732584, 0.8872440043221368, 0.7827811163293595]
Sample [4.6, 3.6, 1.0, 0.2]  expected [1.0, 0.0, 0.0] produced [0.8332349514188845, 0.8866326607911259, 0.7785510276220896]
Sample [6.3, 3.3, 4.7, 1.6]  expected [0.0, 1.0, 0.0] produced [0.8347188995751283, 0.8860995816469369, 0.7759267765937337]
Sample [4.6, 3.2, 1.4, 0.2]  expected [1.0, 0.0, 0.0] produced [0.8313162575626298, 0.8854406790733332, 0.7715302391950697]
Sample [4.9, 2.4, 3.3, 1.0]  expected [0.0, 1.0, 0.0] produced [0.8325192218794496, 0.8846164196268843, 0.7684952470960525]
Sample [5.1, 3.7, 1.5, 0.4]  expected [1.0, 0.0, 0.0] produced [0.8299239523484009, 0.8847277185994433, 0.7647679740796344]
Sample [6.9, 3.1, 5.1, 2.3]  expected [0.0, 0.0, 1.0] produced [0.8309045768608306, 0.883731887233236, 0.761438270973092]
Sample [4.8, 3.0, 1.4, 0.3]  expected [1.0, 0.0, 0.0] produced [0.8274778521427193, 0.8815242645214404, 0.7618123023478133]
Sample [4.7, 3.2, 1.6, 0.2]  expected [1.0, 0.0, 0.0] produced [0.82808189068169, 0.8801493463586223, 0.7581223537268725]
Sample [6.8, 2.8, 4.8, 1.4]  expected [0.0, 1.0, 0.0] produced [0.8294494440181475, 0.879401126280516, 0.7550107893934714]
Sample [6.1, 3.0, 4.6, 1.4]  expected [0.0, 1.0, 0.0] produced [0.8269300686456876, 0.8795878103642246, 0.7511007560747494]
Sample [5.8, 2.6, 4.0, 1.2]  expected [0.0, 1.0, 0.0] produced [0.8243155514579954, 0.8797265797992816, 0.7470830876413854]
Sample [7.2, 3.2, 6.0, 1.8]  expected [0.0, 0.0, 1.0] produced [0.8218104264311048, 0.8800342591251632, 0.743167392241018]
Sample [4.8, 3.1, 1.6, 0.2]  expected [1.0, 0.0, 0.0] produced [0.8182775764177795, 0.8778094700106825, 0.7438536203740909]
Sample [5.6, 2.5, 3.9, 1.1]  expected [0.0, 1.0, 0.0] produced [0.8196124228988669, 0.8768981809451397, 0.7403793092418881]
Sample [6.0, 3.0, 4.8, 1.8]  expected [0.0, 0.0, 1.0] produced [0.8170226386353133, 0.8772307824240138, 0.7363715739053444]
Sample [6.7, 3.3, 5.7, 2.1]  expected [0.0, 0.0, 1.0] produced [0.8142947194991126, 0.8757255706164818, 0.7378851712854819]
Sample [5.0, 2.3, 3.3, 1.0]  expected [0.0, 1.0, 0.0] produced [0.8111732440469677, 0.873833700237013, 0.7390465367961888]
Sample [6.5, 3.0, 5.5, 1.8]  expected [0.0, 0.0, 1.0] produced [0.8086135573328257, 0.8743781757006334, 0.7352074442142262]
Sample [5.5, 4.2, 1.4, 0.2]  expected [1.0, 0.0, 0.0] produced [0.8053519843892271, 0.872519081397735, 0.736459722820454]
Sample [5.1, 3.3, 1.7, 0.5]  expected [1.0, 0.0, 0.0] produced [0.8059113876467919, 0.8707484406842108, 0.7321552683175222]
Sample [7.7, 2.6, 6.9, 2.3]  expected [0.0, 0.0, 1.0] produced [0.8071308156013385, 0.8695278200469966, 0.7283209339202977]
Sample [5.6, 2.7, 4.2, 1.3]  expected [0.0, 1.0, 0.0] produced [0.8040857159856559, 0.8677342461422946, 0.7298181070594607]
Sample [6.6, 3.0, 4.4, 1.4]  expected [0.0, 1.0, 0.0] produced [0.801139162563966, 0.8680663166214029, 0.725608650842053]
Sample [5.1, 2.5, 3.0, 1.1]  expected [0.0, 1.0, 0.0] produced [0.7978242999649283, 0.8680778611804572, 0.7210573169869273]
Sample [5.9, 3.0, 5.1, 1.8]  expected [0.0, 0.0, 1.0] produced [0.7949527423990471, 0.8685840042843528, 0.716887133268717]
Sample [5.8, 4.0, 1.2, 0.2]  expected [1.0, 0.0, 0.0] produced [0.791423965453813, 0.8666097182845701, 0.718397512036457]
Sample [6.8, 3.0, 5.5, 2.1]  expected [0.0, 0.0, 1.0] produced [0.7926370880236782, 0.8651628585419622, 0.7142329576242396]
Sample [5.7, 2.6, 3.5, 1.0]  expected [0.0, 1.0, 0.0] produced [0.7892615151410648, 0.8632315788162496, 0.7158815130171218]
Sample [6.7, 2.5, 5.8, 1.8]  expected [0.0, 0.0, 1.0] produced [0.7861099580004266, 0.8636573530473802, 0.7115440501495218]
Sample [5.2, 2.7, 3.9, 1.4]  expected [0.0, 1.0, 0.0] produced [0.7826489532665581, 0.8617343551934132, 0.7132636074209409]
Sample [6.4, 2.8, 5.6, 2.1]  expected [0.0, 0.0, 1.0] produced [0.7793486328361983, 0.8621517615354204, 0.7088770743341649]
Sample [6.3, 2.5, 5.0, 1.9]  expected [0.0, 0.0, 1.0] produced [0.7758576267373659, 0.8602956931597634, 0.7107163097511024]
Sample [7.9, 3.8, 6.4, 2.0]  expected [0.0, 0.0, 1.0] produced [0.7723507631594124, 0.8584648243811097, 0.7125823802270279]
Sample [6.7, 3.0, 5.0, 1.7]  expected [0.0, 1.0, 0.0] produced [0.7687304016213024, 0.8565281951778586, 0.7143662025916644]
Sample [6.5, 3.2, 5.1, 2.0]  expected [0.0, 0.0, 1.0] produced [0.7650711765390805, 0.8568593199087907, 0.7098906280706384]
Sample [5.6, 3.0, 4.5, 1.5]  expected [0.0, 1.0, 0.0] produced [0.7613020410256203, 0.8548612899641903, 0.7116873040802293]
Sample [5.7, 2.8, 4.5, 1.3]  expected [0.0, 1.0, 0.0] produced [0.7575016716217995, 0.8551770839566802, 0.7071596175046473]
Sample [6.6, 2.9, 4.6, 1.3]  expected [0.0, 1.0, 0.0] produced [0.7536835082275214, 0.8555562450713761, 0.7026318867633957]
Sample [4.8, 3.4, 1.9, 0.2]  expected [1.0, 0.0, 0.0] produced [0.7491914011150032, 0.8553568627436207, 0.6975892920688946]
Sample [5.4, 3.4, 1.7, 0.2]  expected [1.0, 0.0, 0.0] produced [0.7506431148926346, 0.8535287025874756, 0.6930430541717801]
Sample [5.1, 3.4, 1.5, 0.2]  expected [1.0, 0.0, 0.0] produced [0.7517858546927146, 0.8513837100350351, 0.6882291819186929]
Sample [5.6, 2.9, 3.6, 1.3]  expected [0.0, 1.0, 0.0] produced [0.7536116526049123, 0.8498033759604463, 0.6838315384856802]
Sample [7.2, 3.6, 6.1, 2.5]  expected [0.0, 0.0, 1.0] produced [0.7497981611598684, 0.8502852045798908, 0.679106962453738]
Sample [6.2, 3.4, 5.4, 2.3]  expected [0.0, 0.0, 1.0] produced [0.7458105049007185, 0.8481944183971158, 0.6813793500876554]
Sample [6.5, 3.0, 5.2, 2.0]  expected [0.0, 0.0, 1.0] produced [0.7417649597483147, 0.8460655610802273, 0.683621541482345]
Sample [6.3, 2.3, 4.4, 1.3]  expected [0.0, 1.0, 0.0] produced [0.7376165443264421, 0.8438307980351688, 0.6857839655916647]
Sample [5.1, 3.8, 1.9, 0.4]  expected [1.0, 0.0, 0.0] produced [0.7331618443016488, 0.8439973433647616, 0.6808156523465435]
Sample [7.3, 2.9, 6.3, 1.8]  expected [0.0, 0.0, 1.0] produced [0.7350346285231918, 0.8421173045893988, 0.676219255192859]
Sample [5.5, 2.3, 4.0, 1.3]  expected [0.0, 1.0, 0.0] produced [0.7307198984747213, 0.8397110447209177, 0.6784263121078615]
Sample [6.3, 2.7, 4.9, 1.8]  expected [0.0, 0.0, 1.0] produced [0.7265512507217216, 0.8402764169166446, 0.6736642504495366]
Sample [4.4, 2.9, 1.4, 0.2]  expected [1.0, 0.0, 0.0] produced [0.7209959160098519, 0.8367740370487003, 0.675148875666005]
Sample [5.0, 3.0, 1.6, 0.2]  expected [1.0, 0.0, 0.0] produced [0.7231760429461447, 0.8349515261606687, 0.6706679775279296]
Sample [6.1, 2.9, 4.7, 1.4]  expected [0.0, 1.0, 0.0] produced [0.7255467925498134, 0.8332743973931827, 0.6662533137972385]
Sample [5.1, 3.5, 1.4, 0.3]  expected [1.0, 0.0, 0.0] produced [0.7206781466272103, 0.8332696962903428, 0.6609713532107317]
Sample [5.1, 3.8, 1.6, 0.2]  expected [1.0, 0.0, 0.0] produced [0.7224703858965712, 0.8309553332313041, 0.6560647574591351]
Sample [7.6, 3.0, 6.6, 2.1]  expected [0.0, 0.0, 1.0] produced [0.7246014466363563, 0.8289320302606902, 0.6513352609699807]
Sample [5.7, 2.8, 4.1, 1.3]  expected [0.0, 1.0, 0.0] produced [0.7201823472218898, 0.8263187095760999, 0.6539646378249019]
Sample [5.4, 3.0, 4.5, 1.5]  expected [0.0, 1.0, 0.0] produced [0.7157946692488112, 0.8268728119812406, 0.6489406778043385]
Sample [5.4, 3.4, 1.5, 0.4]  expected [1.0, 0.0, 0.0] produced [0.7109498474905657, 0.8270597539357203, 0.6436676421207591]
Sample [5.7, 2.5, 5.0, 2.0]  expected [0.0, 0.0, 1.0] produced [0.7131768441289343, 0.8248841558637414, 0.6388075238082694]
Sample [4.9, 3.0, 1.4, 0.2]  expected [1.0, 0.0, 0.0] produced [0.7078476026474473, 0.8214647385831993, 0.6411935568134264]
Sample [6.7, 3.1, 4.7, 1.5]  expected [0.0, 1.0, 0.0] produced [0.7105544779664205, 0.81967558317772, 0.6366214576045768]
Sample [5.4, 3.7, 1.5, 0.2]  expected [1.0, 0.0, 0.0] produced [0.7056240814604617, 0.8198808131357577, 0.6312822020855497]
Sample [5.7, 2.9, 4.2, 1.3]  expected [0.0, 1.0, 0.0] produced [0.7078814240106077, 0.8175142929980433, 0.6263300597350729]
Sample [5.1, 3.5, 1.4, 0.2]  expected [1.0, 0.0, 0.0] produced [0.7027966454232961, 0.8176292681475472, 0.6209177856745589]
Sample [6.7, 3.1, 4.4, 1.4]  expected [0.0, 1.0, 0.0] produced [0.7052938140086887, 0.8154515952385254, 0.6160465358869636]
Sample [5.0, 3.3, 1.4, 0.2]  expected [1.0, 0.0, 0.0] produced [0.7000415519950458, 0.8154132430120745, 0.6105473866203691]
Sample [6.5, 2.8, 4.6, 1.5]  expected [0.0, 1.0, 0.0] produced [0.702672487845951, 0.8132976749165475, 0.6056895086784758]
Sample [5.0, 3.4, 1.6, 0.4]  expected [1.0, 0.0, 0.0] produced [0.6975732410550181, 0.8134669123268271, 0.6002902564493948]
Sample [5.0, 3.5, 1.3, 0.3]  expected [1.0, 0.0, 0.0] produced [0.699475044557997, 0.8105647950146044, 0.5950813503951883]
Sample [5.7, 3.0, 4.2, 1.2]  expected [0.0, 1.0, 0.0] produced [0.7020085782310743, 0.8082125712175241, 0.5901213130175496]
Sample [6.7, 3.1, 5.6, 2.4]  expected [0.0, 0.0, 1.0] produced [0.6974373019985606, 0.8089855910851698, 0.5849678902369753]
Sample [5.2, 3.5, 1.5, 0.2]  expected [1.0, 0.0, 0.0] produced [0.6922549939652586, 0.8055574060135339, 0.5884315934798459]
Sample [6.9, 3.2, 5.7, 2.3]  expected [0.0, 0.0, 1.0] produced [0.6948445634716353, 0.8031051034471284, 0.5834596896340324]
Sample [6.1, 2.6, 5.6, 1.4]  expected [0.0, 0.0, 1.0] produced [0.690123355263262, 0.8000235496288788, 0.5871252267300785]
Sample [6.0, 2.2, 5.0, 1.5]  expected [0.0, 0.0, 1.0] produced [0.6853567347376169, 0.7969047644387726, 0.5907450957725154]
Sample [5.5, 2.4, 3.7, 1.0]  expected [0.0, 1.0, 0.0] produced [0.6804713999769186, 0.7936354643397047, 0.5942747366367128]
Sample [7.7, 2.8, 6.7, 2.0]  expected [0.0, 0.0, 1.0] produced [0.6757539402028785, 0.7946713705868529, 0.5891734678873404]
Sample [6.0, 3.4, 4.5, 1.6]  expected [0.0, 1.0, 0.0] produced [0.6708472000145165, 0.7914479065060551, 0.5927657251004669]
Sample [5.8, 2.7, 5.1, 1.9]  expected [0.0, 0.0, 1.0] produced [0.665927312744551, 0.7922853273434732, 0.5875713619810474]
Sample [4.4, 3.2, 1.3, 0.2]  expected [1.0, 0.0, 0.0] produced [0.660026422986472, 0.7880038828312336, 0.5907730987755648]
Sample [6.1, 2.8, 4.0, 1.3]  expected [0.0, 1.0, 0.0] produced [0.6634884460964643, 0.7857326066098309, 0.5860259162283216]
Epoch 0 RMSE = 0.6544431980507073
Epoch 100 RMSE = 0.47281576487935295
Epoch 200 RMSE = 0.4727844805621441
Epoch 300 RMSE = 0.38507416643874515
Epoch 400 RMSE = 0.3418522955088099
Epoch 500 RMSE = 0.3256351040609827
Epoch 600 RMSE = 0.3164289054786331
Epoch 700 RMSE = 0.3118037926463389
Epoch 800 RMSE = 0.3089667887240442
Epoch 900 RMSE = 0.3070218679096833
Final Epoch RMSE = 0.3056002185251146
TESTING
Sample [6.1, 2.8, 4.7, 1.2]  expected [0.0, 1.0, 0.0] produced [0.012086813603246174, 0.3748180171217801, 0.1836265391257789]
Sample [4.6, 3.4, 1.4, 0.3]  expected [1.0, 0.0, 0.0] produced [0.9158705880756497, 0.2904202928392677, 0.00026063639835282893]
Sample [4.8, 3.0, 1.4, 0.1]  expected [1.0, 0.0, 0.0] produced [0.9161907788818237, 0.28917408295610075, 0.000259992288015052]
Sample [6.9, 3.1, 4.9, 1.5]  expected [0.0, 1.0, 0.0] produced [0.01632801037448337, 0.3720207848054352, 0.1421180179863235]
Sample [5.9, 3.0, 4.2, 1.5]  expected [0.0, 1.0, 0.0] produced [0.019874914517603277, 0.37336670159329916, 0.11929548100929123]
Sample [5.4, 3.9, 1.7, 0.4]  expected [1.0, 0.0, 0.0] produced [0.9179328543633463, 0.29364180193115447, 0.00025260884197036455]
Sample [5.5, 2.4, 3.8, 1.1]  expected [0.0, 1.0, 0.0] produced [0.050208706525352476, 0.36269592286201374, 0.04957078603334152]
Sample [6.3, 2.8, 5.1, 1.5]  expected [0.0, 0.0, 1.0] produced [0.002300737631400058, 0.4113170883249148, 0.5407305874844665]
Sample [4.6, 3.1, 1.5, 0.2]  expected [1.0, 0.0, 0.0] produced [0.9127129049933249, 0.2942952314090406, 0.00027419687566904025]
Sample [6.5, 3.0, 5.8, 2.2]  expected [0.0, 0.0, 1.0] produced [0.0004453966894884187, 0.43052048466351234, 0.8601175926769151]
Sample [5.8, 2.7, 5.1, 1.9]  expected [0.0, 0.0, 1.0] produced [0.000512963498922398, 0.4246303922330452, 0.8426510707903951]
Sample [6.1, 3.0, 4.9, 1.8]  expected [0.0, 0.0, 1.0] produced [0.0008299505199490237, 0.4140173033785412, 0.7688078527203904]
Sample [5.4, 3.9, 1.3, 0.4]  expected [1.0, 0.0, 0.0] produced [0.9178718612340738, 0.28576550074443474, 0.00025803934517197423]
Sample [5.6, 3.0, 4.1, 1.3]  expected [0.0, 1.0, 0.0] produced [0.010934112500325386, 0.37320689933001605, 0.2025583303797289]
Sample [4.9, 2.5, 4.5, 1.7]  expected [0.0, 0.0, 1.0] produced [0.0005951917434603896, 0.4185564867555211, 0.8227933904570054]
Sample [4.9, 3.1, 1.5, 0.1]  expected [1.0, 0.0, 0.0] produced [0.9154886079437267, 0.28593028037372537, 0.0002667863694672089]
Sample [6.8, 3.2, 5.9, 2.3]  expected [0.0, 0.0, 1.0] produced [0.0004547655093163808, 0.417337696745703, 0.8590949099946863]
Sample [7.0, 3.2, 4.7, 1.4]  expected [0.0, 1.0, 0.0] produced [0.022774057646779797, 0.3597502882904097, 0.10801251951350678]
Sample [5.0, 3.5, 1.6, 0.6]  expected [1.0, 0.0, 0.0] produced [0.9106564291704218, 0.28635627184357465, 0.0002832627286471246]
Sample [5.5, 3.5, 1.3, 0.2]  expected [1.0, 0.0, 0.0] produced [0.9182600494885038, 0.284021401023598, 0.0002573400633239053]
Sample [4.9, 3.1, 1.5, 0.1]  expected [1.0, 0.0, 0.0] produced [0.9156732624402962, 0.2833074654713003, 0.00026684543175286337]
Sample [7.7, 3.0, 6.1, 2.3]  expected [0.0, 0.0, 1.0] produced [0.0004610345419614057, 0.4139220276068714, 0.8578139304093415]
Sample [6.4, 3.2, 4.5, 1.5]  expected [0.0, 1.0, 0.0] produced [0.009620023590502971, 0.36847523395747134, 0.22508081752715195]
Sample [4.9, 3.1, 1.5, 0.1]  expected [1.0, 0.0, 0.0] produced [0.91635762929745, 0.28298787335997866, 0.0002640809868204055]
Sample [5.0, 3.2, 1.2, 0.2]  expected [1.0, 0.0, 0.0] produced [0.917403950783847, 0.28167063104348294, 0.00026065836053789135]
Sample [6.4, 3.2, 5.3, 2.3]  expected [0.0, 0.0, 1.0] produced [0.0005368221685708758, 0.41016959185381785, 0.8380978673995397]
Sample [5.5, 2.5, 4.0, 1.3]  expected [0.0, 1.0, 0.0] produced [0.006162100913458354, 0.3730348183831124, 0.3122758846226195]
Sample [5.1, 3.8, 1.5, 0.3]  expected [1.0, 0.0, 0.0] produced [0.9185796495063661, 0.2812475586530003, 0.00025513369307728827]
Sample [5.6, 2.8, 4.9, 2.0]  expected [0.0, 0.0, 1.0] produced [0.0006470803927812327, 0.40744537256247937, 0.8104233452439975]
Sample [6.3, 3.4, 5.6, 2.4]  expected [0.0, 0.0, 1.0] produced [0.0005277525019445005, 0.4068438626679929, 0.840254696751752]
Sample [5.7, 3.8, 1.7, 0.3]  expected [1.0, 0.0, 0.0] produced [0.9188493143029042, 0.27609532519906865, 0.0002554619042360751]
Sample [6.4, 2.9, 4.3, 1.3]  expected [0.0, 1.0, 0.0] produced [0.07517733713074234, 0.33490099592533357, 0.03376292561900972]
Sample [5.5, 2.6, 4.4, 1.2]  expected [0.0, 1.0, 0.0] produced [0.00721043748456348, 0.3704956209022681, 0.27921757145978193]
Sample [4.5, 2.3, 1.3, 0.3]  expected [1.0, 0.0, 0.0] produced [0.9060498868686624, 0.283026566598037, 0.00030041471148233804]
Sample [5.9, 3.2, 4.8, 1.8]  expected [0.0, 1.0, 0.0] produced [0.0027090366245999635, 0.3869642628202141, 0.5062128869510732]
Sample [7.4, 2.8, 6.1, 1.9]  expected [0.0, 0.0, 1.0] produced [0.00144912253730166, 0.40038128597404127, 0.6529705824880455]
Sample [6.3, 3.3, 6.0, 2.5]  expected [0.0, 0.0, 1.0] produced [0.0004909241167004172, 0.4124488513020981, 0.8483807101863333]
Sample [4.7, 3.2, 1.3, 0.2]  expected [1.0, 0.0, 0.0] produced [0.9179950918267854, 0.2789761666972057, 0.00025728391684989426]
Sample [6.9, 3.1, 5.4, 2.1]  expected [0.0, 0.0, 1.0] produced [0.0013375473652673848, 0.39356735405353815, 0.6739923619400667]
Sample [6.7, 3.3, 5.7, 2.5]  expected [0.0, 0.0, 1.0] produced [0.0005287463759195476, 0.40325489196972436, 0.840468511032505]
Sample [6.7, 3.0, 5.2, 2.3]  expected [0.0, 0.0, 1.0] produced [0.0006798786001330833, 0.3963825716116662, 0.80442986524477]
Sample [6.0, 2.2, 4.0, 1.0]  expected [0.0, 1.0, 0.0] produced [0.07282052782193998, 0.3310805918304313, 0.035108555187644046]
Sample [5.8, 2.7, 4.1, 1.0]  expected [0.0, 1.0, 0.0] produced [0.15623958174310695, 0.3239503211832944, 0.015231721918150271]
Sample [5.2, 4.1, 1.5, 0.1]  expected [1.0, 0.0, 0.0] produced [0.9194175166800589, 0.2777339989575796, 0.0002543311620111931]
Sample [4.4, 3.0, 1.3, 0.2]  expected [1.0, 0.0, 0.0] produced [0.914561584463596, 0.2774938198437244, 0.0002725200526311336]
RMSE is 0.2822263714070732

Please Enter to test sin
This is the sample run of sin.
Sample [0.62]  expected [0.581035160537305] produced [0.6975322566973053]
Sample [1.31]  expected [0.966184951612734] produced [0.724119157763133]
Sample [0.75]  expected [0.681638760023334] produced [0.7034177836802645]
Sample [1.0]  expected [0.841470984807897] produced [0.7134191994972203]
Sample [0.57]  expected [0.539632048733969] produced [0.695988218973831]
Sample [0.92]  expected [0.795601620036366] produced [0.7102401458955603]
Sample [0.73]  expected [0.666869635003698] produced [0.7026996299116264]
Sample [0.55]  expected [0.522687228930659] produced [0.6948709323861041]
Sample [0.3]  expected [0.29552020666134] produced [0.6832091081727749]
Sample [0.93]  expected [0.801619940883777] produced [0.7094324156611956]
Sample [0.81]  expected [0.724287174370143] produced [0.7048593607288027]
Sample [1.35]  expected [0.975723357826659] produced [0.7251424580837551]
Sample [0.21]  expected [0.2084598998461] produced [0.6790944791147521]
Sample [1.57]  expected [0.999999682931835] produced [0.7318911525528066]
Sample [1.2]  expected [0.932039085967226] produced [0.7203160998137192]
Epoch 0 RMSE = 0.21784995182013825
Epoch 100 RMSE = 0.21598779353499994
Epoch 200 RMSE = 0.21469996671244881
Epoch 300 RMSE = 0.21369565330604973
Epoch 400 RMSE = 0.2128888991920876
Epoch 500 RMSE = 0.21222216467597346
Epoch 600 RMSE = 0.2116547567029842
Epoch 700 RMSE = 0.21115595761297729
Epoch 800 RMSE = 0.2107006206314636
Epoch 900 RMSE = 0.21026595070478135
Sample [0.62]  expected [0.581035160537305] produced [0.7037556563383249]
Sample [1.31]  expected [0.966184951612734] produced [0.74266248996177]
Sample [0.75]  expected [0.681638760023334] produced [0.7134317453935008]
Sample [1.0]  expected [0.841470984807897] produced [0.7287040674798656]
Sample [0.57]  expected [0.539632048733969] produced [0.7005493808109956]
Sample [0.92]  expected [0.795601620036366] produced [0.7240297659444793]
Sample [0.73]  expected [0.666869635003698] produced [0.7120711279243975]
Sample [0.55]  expected [0.522687228930659] produced [0.698616176622878]
Sample [0.3]  expected [0.29552020666134] produced [0.6770284249516769]
Sample [0.93]  expected [0.801619940883777] produced [0.7231058528725983]
Sample [0.81]  expected [0.724287174370143] produced [0.7159809617705334]
Sample [1.35]  expected [0.975723357826659] produced [0.7435066807699597]
Sample [0.21]  expected [0.2084598998461] produced [0.6688082277573556]
Sample [1.57]  expected [0.999999682931835] produced [0.7504614418247408]
Sample [1.2]  expected [0.932039085967226] produced [0.7377106373876335]
Epoch 1000 RMSE = 0.20982864338899995
Epoch 1100 RMSE = 0.20936168443707598
Epoch 1200 RMSE = 0.2088299926871406
Epoch 1300 RMSE = 0.20818369793644814
Epoch 1400 RMSE = 0.20734712373456246
Epoch 1500 RMSE = 0.20620061603915363
Epoch 1600 RMSE = 0.204552475146594
Epoch 1700 RMSE = 0.2021048381761323
Epoch 1800 RMSE = 0.19844476230395394
Epoch 1900 RMSE = 0.19314675281160043
Sample [0.62]  expected [0.581035160537305] produced [0.68387358517968]
Sample [1.31]  expected [0.966184951612734] produced [0.7601845868880877]
Sample [0.75]  expected [0.681638760023334] produced [0.7030202716715723]
Sample [1.0]  expected [0.841470984807897] produced [0.7330649852699898]
Sample [0.57]  expected [0.539632048733969] produced [0.6766164542548657]
Sample [0.92]  expected [0.795601620036366] produced [0.7241842996112353]
Sample [0.73]  expected [0.666869635003698] produced [0.7003477272962687]
Sample [0.55]  expected [0.522687228930659] produced [0.6730901676827813]
Sample [0.3]  expected [0.29552020666134] produced [0.6280618572898873]
Sample [0.93]  expected [0.801619940883777] produced [0.7240255868925215]
Sample [0.81]  expected [0.724287174370143] produced [0.7097591031866128]
Sample [1.35]  expected [0.975723357826659] produced [0.7626488017622438]
Sample [0.21]  expected [0.2084598998461] produced [0.6102803941006852]
Sample [1.57]  expected [0.999999682931835] produced [0.7762696112448316]
Sample [1.2]  expected [0.932039085967226] produced [0.7512152332644553]
Epoch 2000 RMSE = 0.18605674654828971
Epoch 2100 RMSE = 0.17755007826839905
Epoch 2200 RMSE = 0.16837882345770558
Epoch 2300 RMSE = 0.15924638937654959
Epoch 2400 RMSE = 0.15057698444582074
Epoch 2500 RMSE = 0.14254482339456054
Epoch 2600 RMSE = 0.1351795009443709
Epoch 2700 RMSE = 0.12844563482773766
Epoch 2800 RMSE = 0.12228556356389253
Epoch 2900 RMSE = 0.11663858098104635
Sample [0.62]  expected [0.581035160537305] produced [0.6395301118563923]
Sample [1.31]  expected [0.966184951612734] produced [0.8217103572068007]
Sample [0.75]  expected [0.681638760023334] produced [0.6920683594522784]
Sample [1.0]  expected [0.841470984807897] produced [0.7664163019214791]
Sample [0.57]  expected [0.539632048733969] produced [0.6171271202926009]
Sample [0.92]  expected [0.795601620036366] produced [0.7460105656518725]
Sample [0.73]  expected [0.666869635003698] produced [0.6847393118925895]
Sample [0.55]  expected [0.522687228930659] produced [0.607424485237661]
Sample [0.3]  expected [0.29552020666134] produced [0.47251403380960294]
Sample [0.93]  expected [0.801619940883777] produced [0.7480023058940921]
Sample [0.81]  expected [0.724287174370143] produced [0.7122535558869606]
Sample [1.35]  expected [0.975723357826659] produced [0.8266061431607036]
Sample [0.21]  expected [0.2084598998461] produced [0.41999403087461995]
Sample [1.57]  expected [0.999999682931835] produced [0.8484374598556996]
Sample [1.2]  expected [0.932039085967226] produced [0.8055089614811444]
Epoch 3000 RMSE = 0.1114483803518166
Epoch 3100 RMSE = 0.1066652289099058
Epoch 3200 RMSE = 0.10224600527259874
Epoch 3300 RMSE = 0.09815349204696679
Epoch 3400 RMSE = 0.09435550332826773
Epoch 3500 RMSE = 0.09082406624924039
Epoch 3600 RMSE = 0.08753472274558434
Epoch 3700 RMSE = 0.08446595694196489
Epoch 3800 RMSE = 0.08159873255455583
Epoch 3900 RMSE = 0.07891612004502079
Sample [0.62]  expected [0.581035160537305] produced [0.6248122737343601]
Sample [1.31]  expected [0.966184951612734] produced [0.8532596968854491]
Sample [0.75]  expected [0.681638760023334] produced [0.6956898523809069]
Sample [1.0]  expected [0.841470984807897] produced [0.7900547999064474]
Sample [0.57]  expected [0.539632048733969] produced [0.5936119132040624]
Sample [0.92]  expected [0.795601620036366] produced [0.7650896601245845]
Sample [0.73]  expected [0.666869635003698] produced [0.6858315102174575]
Sample [0.55]  expected [0.522687228930659] produced [0.5802237758871132]
Sample [0.3]  expected [0.29552020666134] produced [0.3954326515218028]
Sample [0.93]  expected [0.801619940883777] produced [0.7679738203449574]
Sample [0.81]  expected [0.724287174370143] produced [0.7225185813061875]
Sample [1.35]  expected [0.975723357826659] produced [0.8585068912784224]
Sample [0.21]  expected [0.2084598998461] produced [0.3277059883183415]
Sample [1.57]  expected [0.999999682931835] produced [0.8809923078928842]
Sample [1.2]  expected [0.932039085967226] produced [0.8356177469834526]
Epoch 4000 RMSE = 0.07640299471980389
Epoch 4100 RMSE = 0.07404579022720974
Epoch 4200 RMSE = 0.07183229525562684
Epoch 4300 RMSE = 0.06975148410057201
Epoch 4400 RMSE = 0.06779337403127932
Epoch 4500 RMSE = 0.06594890410646548
Epoch 4600 RMSE = 0.06420983136873391
Epoch 4700 RMSE = 0.06256864129188332
Epoch 4800 RMSE = 0.061018470052200485
Epoch 4900 RMSE = 0.05955303671104504
Sample [0.62]  expected [0.581035160537305] produced [0.6160631313012156]
Sample [1.31]  expected [0.966184951612734] produced [0.8720562812818416]
Sample [0.75]  expected [0.681638760023334] produced [0.6979833752150854]
Sample [1.0]  expected [0.841470984807897] produced [0.8043296023090694]
Sample [0.57]  expected [0.539632048733969] produced [0.5796807077846561]
Sample [0.92]  expected [0.795601620036366] produced [0.7766825653224888]
Sample [0.73]  expected [0.666869635003698] produced [0.6865947331077368]
Sample [0.55]  expected [0.522687228930659] produced [0.5641545035528305]
Sample [0.3]  expected [0.29552020666134] produced [0.35413354662646884]
Sample [0.93]  expected [0.801619940883777] produced [0.7800827074460586]
Sample [0.81]  expected [0.724287174370143] produced [0.7288201168070023]
Sample [1.35]  expected [0.975723357826659] produced [0.8775056319059408]
Sample [0.21]  expected [0.2084598998461] produced [0.2812915835568862]
Sample [1.57]  expected [0.999999682931835] produced [0.900395064018129]
Sample [1.2]  expected [0.932039085967226] produced [0.8535925542616283]
Epoch 5000 RMSE = 0.058166583781721806
Epoch 5100 RMSE = 0.05685382494509644
Epoch 5200 RMSE = 0.055609898901614434
Epoch 5300 RMSE = 0.05443032852077862
Epoch 5400 RMSE = 0.05331098458586038
Epoch 5500 RMSE = 0.05224805354093297
Epoch 5600 RMSE = 0.051238008735924365
Epoch 5700 RMSE = 0.05027758473808054
Epoch 5800 RMSE = 0.049363754338511534
Epoch 5900 RMSE = 0.04849370793296283
Sample [0.62]  expected [0.581035160537305] produced [0.6101206134194198]
Sample [1.31]  expected [0.966184951612734] produced [0.8842423417106683]
Sample [0.75]  expected [0.681638760023334] produced [0.6992969533554121]
Sample [1.0]  expected [0.841470984807897] produced [0.8136119983407047]
Sample [0.57]  expected [0.539632048733969] produced [0.5704262030942793]
Sample [0.92]  expected [0.795601620036366] produced [0.7841935067234915]
Sample [0.73]  expected [0.666869635003698] produced [0.686894216226193]
Sample [0.55]  expected [0.522687228930659] produced [0.5535576769300897]
Sample [0.3]  expected [0.29552020666134] produced [0.3295086715123888]
Sample [0.93]  expected [0.801619940883777] produced [0.7879290544916989]
Sample [0.81]  expected [0.724287174370143] produced [0.7327999859515726]
Sample [1.35]  expected [0.975723357826659] produced [0.8898138419060398]
Sample [0.21]  expected [0.2084598998461] produced [0.25493223674808274]
Sample [1.57]  expected [0.999999682931835] produced [0.9129126934173554]
Sample [1.2]  expected [0.932039085967226] produced [0.8652650387537723]
Epoch 6000 RMSE = 0.047664834998531684
Epoch 6100 RMSE = 0.046874707424235634
Epoch 6200 RMSE = 0.04612106448424828
Epoch 6300 RMSE = 0.04540179926916064
Epoch 6400 RMSE = 0.0447149464134986
Epoch 6500 RMSE = 0.044058670977513585
Epoch 6600 RMSE = 0.043431258358418795
Epoch 6700 RMSE = 0.04283110512116335
Epoch 6800 RMSE = 0.04225671065182731
Epoch 6900 RMSE = 0.04170666954806682
Sample [0.62]  expected [0.581035160537305] produced [0.6058268474486246]
Sample [1.31]  expected [0.966184951612734] produced [0.8926793083781526]
Sample [0.75]  expected [0.681638760023334] produced [0.7000151135024758]
Sample [1.0]  expected [0.841470984807897] produced [0.8199875174952925]
Sample [0.57]  expected [0.539632048733969] produced [0.5639149276316635]
Sample [0.92]  expected [0.795601620036366] produced [0.7893049564857707]
Sample [0.73]  expected [0.666869635003698] produced [0.6869033096933744]
Sample [0.55]  expected [0.522687228930659] produced [0.5461644073094605]
Sample [0.3]  expected [0.29552020666134] produced [0.31386685440260365]
Sample [0.93]  expected [0.801619940883777] produced [0.7932752423553793]
Sample [0.81]  expected [0.724287174370143] produced [0.7353961593911253]
Sample [1.35]  expected [0.975723357826659] produced [0.8983350301892515]
Sample [0.21]  expected [0.2084598998461] produced [0.23881629800631154]
Sample [1.57]  expected [0.999999682931835] produced [0.9215655412715584]
Sample [1.2]  expected [0.932039085967226] produced [0.8733391651706967]
Epoch 7000 RMSE = 0.041179664670950834
Epoch 7100 RMSE = 0.04067446079120862
Epoch 7200 RMSE = 0.04018989877051672
Epoch 7300 RMSE = 0.039724890225130094
Epoch 7400 RMSE = 0.0392784126250258
Epoch 7500 RMSE = 0.038849504786890594
Epoch 7600 RMSE = 0.03843726272381847
Epoch 7700 RMSE = 0.038040835818595645
Epoch 7800 RMSE = 0.037659423290975
Epoch 7900 RMSE = 0.037292270932463574
Sample [0.62]  expected [0.581035160537305] produced [0.6025958748886637]
Sample [1.31]  expected [0.966184951612734] produced [0.89881622051608]
Sample [0.75]  expected [0.681638760023334] produced [0.700359409664706]
Sample [1.0]  expected [0.841470984807897] produced [0.8245388696568836]
Sample [0.57]  expected [0.539632048733969] produced [0.5591575442416467]
Sample [0.92]  expected [0.795601620036366] produced [0.7929014112394439]
Sample [0.73]  expected [0.666869635003698] produced [0.6867336524118972]
Sample [0.55]  expected [0.522687228930659] produced [0.5408123253795065]
Sample [0.3]  expected [0.29552020666134] produced [0.3035451394028879]
Sample [0.93]  expected [0.801619940883777] produced [0.7970449241690684]
Sample [0.81]  expected [0.724287174370143] produced [0.7371159114122783]
Sample [1.35]  expected [0.975723357826659] produced [0.9045379857217899]
Sample [0.21]  expected [0.2084598998461] produced [0.2285287616754965]
Sample [1.57]  expected [0.999999682931835] produced [0.9278761474080126]
Sample [1.2]  expected [0.932039085967226] produced [0.879190752059462]
Epoch 8000 RMSE = 0.036938668084901746
Epoch 8100 RMSE = 0.03659794484155076
Epoch 8200 RMSE = 0.03626946945156871
Epoch 8300 RMSE = 0.035952645910666235
Epoch 8400 RMSE = 0.035646911722435184
Epoch 8500 RMSE = 0.03535173581635706
Epoch 8600 RMSE = 0.035066616609835476
Epoch 8700 RMSE = 0.03479108020280258
Epoch 8800 RMSE = 0.034524678694511604
Epoch 8900 RMSE = 0.03426698861308186
Sample [0.62]  expected [0.581035160537305] produced [0.6000933429136025]
Sample [1.31]  expected [0.966184951612734] produced [0.9034479700860651]
Sample [0.75]  expected [0.681638760023334] produced [0.7004653456099466]
Sample [1.0]  expected [0.841470984807897] produced [0.8278809590605025]
Sample [0.57]  expected [0.539632048733969] produced [0.5555860747808097]
Sample [0.92]  expected [0.795601620036366] produced [0.7954918935772074]
Sample [0.73]  expected [0.666869635003698] produced [0.6864573160826479]
Sample [0.55]  expected [0.522687228930659] produced [0.5368340710717967]
Sample [0.3]  expected [0.29552020666134] produced [0.2965728123966655]
Sample [0.93]  expected [0.801619940883777] produced [0.799768279546522]
Sample [0.81]  expected [0.724287174370143] produced [0.7382581219084721]
Sample [1.35]  expected [0.975723357826659] produced [0.9092261686777963]
Sample [0.21]  expected [0.2084598998461] produced [0.22179880716782271]
Sample [1.57]  expected [0.999999682931835] produced [0.932667050784146]
Sample [1.2]  expected [0.932039085967226] produced [0.8835815721028871]
Epoch 9000 RMSE = 0.034017609449218236
Epoch 9100 RMSE = 0.03377616228628969
Epoch 9200 RMSE = 0.03354228851963397
Epoch 9300 RMSE = 0.03331564865857388
Epoch 9400 RMSE = 0.033095921205181
Epoch 9500 RMSE = 0.032882801604324184
Epoch 9600 RMSE = 0.0326760012599859
Epoch 9700 RMSE = 0.032475246613237405
Epoch 9800 RMSE = 0.03228027827763351
Epoch 9900 RMSE = 0.03209085022811409
Sample [0.62]  expected [0.581035160537305] produced [0.5981113235811246]
Sample [1.31]  expected [0.966184951612734] produced [0.9070455368941783]
Sample [0.75]  expected [0.681638760023334] produced [0.7004202443824095]
Sample [1.0]  expected [0.841470984807897] produced [0.8303905021495136]
Sample [0.57]  expected [0.539632048733969] produced [0.5528466320964819]
Sample [0.92]  expected [0.795601620036366] produced [0.7973918606709341]
Sample [0.73]  expected [0.666869635003698] produced [0.6861220322088003]
Sample [0.55]  expected [0.522687228930659] produced [0.5338142006445308]
Sample [0.3]  expected [0.29552020666134] produced [0.2917919635681058]
Sample [0.93]  expected [0.801619940883777] produced [0.8017731923499768]
Sample [0.81]  expected [0.724287174370143] produced [0.7390112999984534]
Sample [1.35]  expected [0.975723357826659] produced [0.9128742091729668]
Sample [0.21]  expected [0.2084598998461] produced [0.21733890848313514]
Sample [1.57]  expected [0.999999682931835] produced [0.936417695293048]
Sample [1.2]  expected [0.932039085967226] produced [0.8869673172158823]
Epoch 10000 RMSE = 0.03190672903981277
Final Epoch RMSE = 0.03190672903981277
TESTING
Sample [0.06]  expected [0.0599640064794446] produced [0.12376909012572063]
Sample [0.7]  expected [0.644217687237691] produced [0.663825539228734]
Sample [1.17]  expected [0.920750597736136] produced [0.8803174724607119]
Sample [1.05]  expected [0.867423225594017] produced [0.8476220724637707]
Sample [0.71]  expected [0.651833771021537] produced [0.6715013218131671]
Sample [0.33]  expected [0.324043028394868] produced [0.31925363934610845]
Sample [0.12]  expected [0.119712207288919] produced [0.1564361851709937]
Sample [0.8]  expected [0.717356090899523] produced [0.7330091425200512]
Sample [0.45]  expected [0.43496553411123] produced [0.43586408901305645]
Sample [0.83]  expected [0.737931371109963] produced [0.7508532858374671]
Sample [0.76]  expected [0.688921445110551] produced [0.7069199065407709]
Sample [1.12]  expected [0.900100442176505] produced [0.8678504330049547]
Sample [0.34]  expected [0.333487092140814] produced [0.3284432575235568]
Sample [0.78]  expected [0.70327941920041] produced [0.7202097354454462]
Sample [0.26]  expected [0.257080551892155] produced [0.2570830144456367]
Sample [0.59]  expected [0.556361022912784] produced [0.5708860775975919]
Sample [0.56]  expected [0.531186197920883] produced [0.5429995185399261]
Sample [0.22]  expected [0.218229623080869] produced [0.22479582381206523]
Sample [1.38]  expected [0.98185353037236] produced [0.9168515487486077]
Sample [1.52]  expected [0.998710143975583] produced [0.9321235035890176]
Sample [0.23]  expected [0.227977523535188] produced [0.23272412258766303]
Sample [1.21]  expected [0.935616001553386] produced [0.8889658208276435]
Sample [1.16]  expected [0.916803108771767] produced [0.877997682633869]
Sample [1.55]  expected [0.999783764189357] produced [0.934841509574139]
Sample [0.94]  expected [0.807558100405114] produced [0.8064413504518655]
Sample [0.91]  expected [0.78950373968995] produced [0.7928879610519001]
Sample [0.32]  expected [0.314566560616118] produced [0.3101155796306073]
Sample [0.24]  expected [0.237702626427135] produced [0.2408905672149378]
Sample [0.79]  expected [0.710353272417608] produced [0.7269436209762357]
Sample [1.23]  expected [0.942488801931697] produced [0.8931054102350349]
Sample [1.04]  expected [0.862404227243338] produced [0.8444797363154964]
Sample [1.11]  expected [0.895698685680048] produced [0.8655172021708686]
Sample [0.18]  expected [0.179029573425824] produced [0.195608699181615]
Sample [0.01]  expected [0.00999983333416666] produced [0.10108910849317014]
Sample [1.5]  expected [0.997494986604054] produced [0.9304525331631537]
Sample [0.65]  expected [0.60518640573604] produced [0.624085916688035]
Sample [1.1]  expected [0.891207360061435] produced [0.8627599869756539]
Sample [0.58]  expected [0.548023936791874] produced [0.562389013239719]
Sample [0.43]  expected [0.416870802429211] produced [0.41633731301982246]
Sample [0.17]  expected [0.169182349066996] produced [0.18858744692412738]
Sample [0.37]  expected [0.361615431964962] produced [0.3573462257869166]
Sample [0.61]  expected [0.572867460100481] produced [0.5894312984609799]
Sample [1.4]  expected [0.98544972998846] produced [0.9196089027415848]
Sample [0.96]  expected [0.819191568300998] produced [0.8149974601539423]
Sample [0.0]  expected [0.0] produced [0.09698971930768245]
Sample [1.26]  expected [0.952090341590516] produced [0.8987864696636174]
Sample [0.66]  expected [0.613116851973434] produced [0.6323742720334168]
Sample [0.97]  expected [0.82488571333845] produced [0.8190283142069733]
Sample [1.03]  expected [0.857298989188603] produced [0.8411888805313522]
Sample [1.25]  expected [0.948984619355586] produced [0.8970251112812428]
Sample [0.88]  expected [0.770738878898969] produced [0.778495056650825]
Sample [1.09]  expected [0.886626914449487] produced [0.8600429568180686]
Sample [1.46]  expected [0.993868363411645] produced [0.9265591562100886]
Sample [0.69]  expected [0.636537182221968] produced [0.656747920076101]
Sample [0.84]  expected [0.744643119970859] produced [0.7571714526029975]
Sample [0.03]  expected [0.0299955002024957] produced [0.10968137431151886]
Sample [1.39]  expected [0.983700814811277] produced [0.9184594954130784]
Sample [0.86]  expected [0.757842562895277] produced [0.7681269458159636]
Sample [0.35]  expected [0.342897807455451] produced [0.3383490311979]
Sample [1.42]  expected [0.98865176285172] produced [0.9221317467111357]
Sample [1.3]  expected [0.963558185417193] produced [0.9057520069414707]
Sample [0.63]  expected [0.58914475794227] produced [0.6075090509280597]
Sample [0.31]  expected [0.305058636443443] produced [0.3012866409096089]
Sample [1.54]  expected [0.999525830605479] produced [0.9341904582520225]
Sample [0.67]  expected [0.62098598703656] produced [0.6409883689086384]
Sample [0.09]  expected [0.089878549198011] produced [0.13940954463955205]
Sample [0.47]  expected [0.452886285379068] produced [0.45641420580212716]
Sample [1.33]  expected [0.971148377921045] produced [0.910345653552391]
Sample [1.08]  expected [0.881957806884948] produced [0.8573183596081903]
Sample [0.4]  expected [0.389418342308651] produced [0.3870843451591087]
Sample [0.27]  expected [0.266731436688831] produced [0.2661049216521207]
Sample [1.28]  expected [0.958015860289225] produced [0.9025463478043101]
Sample [0.72]  expected [0.659384671971473] produced [0.6798223567158012]
Sample [1.43]  expected [0.990104560337178] produced [0.9234055520870165]
Sample [0.74]  expected [0.674287911628145] produced [0.6942676664418782]
Sample [1.56]  expected [0.999941720229966] produced [0.9359136672060163]
Sample [0.39]  expected [0.380188415123161] produced [0.3773948707124403]
Sample [0.25]  expected [0.247403959254523] produced [0.2494150928970807]
Sample [0.29]  expected [0.285952225104836] produced [0.2835750755328354]
Sample [1.18]  expected [0.92460601240802] produced [0.8831747970526483]
Sample [0.05]  expected [0.0499791692706783] produced [0.11902549583888382]
Sample [0.44]  expected [0.425939465066] produced [0.4269508060486895]
Sample [1.51]  expected [0.998152472497548] produced [0.9316390113134452]
Sample [0.54]  expected [0.514135991653113] produced [0.5255498210839649]
Sample [1.02]  expected [0.852108021949363] produced [0.838314866824395]
Sample [1.45]  expected [0.992712991037588] produced [0.9257100160950429]
Sample [1.53]  expected [0.999167945271476] produced [0.9335061240347993]
Sample [0.04]  expected [0.0399893341866342] produced [0.11435660562568453]
Sample [0.48]  expected [0.461779175541483] produced [0.46701091735885675]
Sample [0.49]  expected [0.470625888171158] produced [0.476886109606083]
Sample [1.14]  expected [0.908633496115883] produced [0.873892731120625]
Sample [1.19]  expected [0.928368967249167] produced [0.8855584656544957]
Sample [0.02]  expected [0.0199986666933331] produced [0.10541689569706926]
Sample [0.85]  expected [0.751280405140293] produced [0.7634860381879363]
Sample [0.64]  expected [0.597195441362392] produced [0.6166468844434164]
Sample [0.2]  expected [0.198669330795061] produced [0.21030339156518438]
Sample [0.53]  expected [0.505533341204847] produced [0.5159357595191648]
Sample [1.44]  expected [0.991458348191686] produced [0.9246185929755214]
Sample [1.32]  expected [0.968715100118265] produced [0.9091345228565239]
Sample [0.14]  expected [0.139543114644236] produced [0.16906187586936003]
Sample [0.5]  expected [0.479425538604203] produced [0.4868037384977139]
Sample [1.41]  expected [0.98710010101385] produced [0.9212479823026222]
Sample [0.46]  expected [0.44394810696552] produced [0.44722531887693057]
Sample [0.38]  expected [0.370920469412983] produced [0.3679307473175808]
Sample [1.37]  expected [0.979908061398614] produced [0.9162806398021773]
Sample [0.87]  expected [0.764328937025505] produced [0.7743063973437938]
Sample [0.28]  expected [0.276355648564114] produced [0.27508686814948646]
Sample [0.07]  expected [0.0699428473375328] produced [0.12896734689814324]
Sample [1.29]  expected [0.960835064206073] produced [0.9044921568463917]
Sample [1.36]  expected [0.977864602435316] produced [0.9150042873544884]
Sample [0.82]  expected [0.731145829726896] produced [0.7467616760889332]
Sample [0.13]  expected [0.129634142619695] produced [0.16286105041611024]
Sample [0.16]  expected [0.159318206614246] produced [0.18216430637102918]
Sample [1.24]  expected [0.945783999449539] produced [0.8957471883969781]
Sample [0.98]  expected [0.83049737049197] produced [0.824041021844778]
Sample [0.41]  expected [0.398609327984423] produced [0.397669909317087]
Sample [1.15]  expected [0.912763940260521] produced [0.8766115128339765]
Sample [1.49]  expected [0.996737752043143] produced [0.9300072408469641]
Sample [1.13]  expected [0.904412189378826] produced [0.8717809945430965]
Sample [0.19]  expected [0.188858894976501] produced [0.2032981878892785]
Sample [1.06]  expected [0.872355482344986] produced [0.8520966876922794]
Sample [0.77]  expected [0.696135238627357] produced [0.7158738156349433]
Sample [0.52]  expected [0.496880137843737] produced [0.507031081756446]
Sample [0.1]  expected [0.0998334166468282] produced [0.1451892983891712]
Sample [0.42]  expected [0.40776045305957] produced [0.4077155698732878]
Sample [0.11]  expected [0.109778300837175] produced [0.15089455538288116]
Sample [1.34]  expected [0.973484541695319] produced [0.9122960752402418]
Sample [1.47]  expected [0.994924349777581] produced [0.9280624348364602]
Sample [0.68]  expected [0.628793024018469] produced [0.6502982922468894]
Sample [0.36]  expected [0.35227423327509] produced [0.3489157205559529]
Sample [0.99]  expected [0.836025978600521] produced [0.828056977030223]
Sample [0.89]  expected [0.777071747526824] produced [0.7847776809694167]
Sample [0.51]  expected [0.488177246882907] produced [0.4971978977829258]
Sample [0.9]  expected [0.783326909627483] produced [0.7895713640366936]
Sample [1.07]  expected [0.877200504274682] produced [0.8550536016611149]
Sample [0.15]  expected [0.149438132473599] produced [0.1756303420666849]
Sample [0.6]  expected [0.564642473395035] produced [0.5821695669661328]
Sample [1.48]  expected [0.99588084453764] produced [0.9290146664684672]
Sample [1.01]  expected [0.846831844618015] produced [0.8353643009229811]
Sample [1.22]  expected [0.939099356319068] produced [0.8920802220542562]
Sample [0.08]  expected [0.0799146939691727] produced [0.1342525512030492]
Sample [0.95]  expected [0.813415504789374] produced [0.8121680221434885]
Sample [1.27]  expected [0.955100855584692] produced [0.9013754115847566]
RMSE is 0.03912932583977292

Please Enter to test XOR without bias
This is the sample run of XOR without bias.
Sample [0.0, 0.0]  expected [0.0] produced [0.7170837133274212]
Sample [0.0, 1.0]  expected [1.0] produced [0.7495573497291937]
Sample [1.0, 0.0]  expected [1.0] produced [0.752049656765903]
Sample [1.0, 1.0]  expected [0.0] produced [0.7831604970473482]
Epoch 0 RMSE = 0.5594082386413934
Epoch 100 RMSE = 0.5039069519516719
Epoch 200 RMSE = 0.5028290514238914
Epoch 300 RMSE = 0.5026706456690437
Epoch 400 RMSE = 0.5025338274726315
Epoch 500 RMSE = 0.5024111370357003
Epoch 600 RMSE = 0.5023005835189718
Epoch 700 RMSE = 0.5022003530939554
Epoch 800 RMSE = 0.5021088859804764
Epoch 900 RMSE = 0.5020248444957613
Sample [0.0, 0.0]  expected [0.0] produced [0.5053746038292729]
Sample [0.0, 1.0]  expected [1.0] produced [0.5092836700396497]
Sample [1.0, 0.0]  expected [1.0] produced [0.49283140536989095]
Sample [1.0, 1.0]  expected [0.0] produced [0.5043584739925786]
Epoch 1000 RMSE = 0.5019470739910117
Epoch 1100 RMSE = 0.5018745692591108
Epoch 1200 RMSE = 0.5018064463949212
Epoch 1300 RMSE = 0.5017419191144988
Epoch 1400 RMSE = 0.5016802786197481
Epoch 1500 RMSE = 0.5016208762392691
Epoch 1600 RMSE = 0.5015631081979472
Epoch 1700 RMSE = 0.5015064019630997
Epoch 1800 RMSE = 0.5014502036866837
Epoch 1900 RMSE = 0.5013939663141843
Sample [0.0, 0.0]  expected [0.0] produced [0.5006417615315911]
Sample [0.0, 1.0]  expected [1.0] produced [0.5095921982506665]
Sample [1.0, 0.0]  expected [1.0] produced [0.4927699907017279]
Sample [1.0, 1.0]  expected [0.0] produced [0.5068840457815825]
Epoch 2000 RMSE = 0.5013371379637407
Epoch 2100 RMSE = 0.5012791501955985
Epoch 2200 RMSE = 0.5012194057932549
Epoch 2300 RMSE = 0.5011572656643316
Epoch 2400 RMSE = 0.5010920344415161
Epoch 2500 RMSE = 0.5010229443217782
Epoch 2600 RMSE = 0.5009491366253798
Epoch 2700 RMSE = 0.5008696404849956
Epoch 2800 RMSE = 0.5007833479902771
Epoch 2900 RMSE = 0.5006889850164573
Sample [0.0, 0.0]  expected [0.0] produced [0.4956100155982975]
Sample [0.0, 1.0]  expected [1.0] produced [0.5120152725733634]
Sample [1.0, 0.0]  expected [1.0] produced [0.49228300282507165]
Sample [1.0, 1.0]  expected [0.0] produced [0.510692417921391]
Epoch 3000 RMSE = 0.5005850768614952
Epoch 3100 RMSE = 0.5004699077130516
Epoch 3200 RMSE = 0.5003414728785497
Epoch 3300 RMSE = 0.5001974226620374
Epoch 3400 RMSE = 0.5000349967968903
Epoch 3500 RMSE = 0.49985094849798756
Epoch 3600 RMSE = 0.49964145755871936
Epoch 3700 RMSE = 0.4994020325934384
Epoch 3800 RMSE = 0.4991274036513228
Epoch 3900 RMSE = 0.4988114081641907
Sample [0.0, 0.0]  expected [0.0] produced [0.48612918412587325]
Sample [0.0, 1.0]  expected [1.0] produced [0.5189476926305986]
Sample [1.0, 0.0]  expected [1.0] produced [0.4924311627253932]
Sample [1.0, 1.0]  expected [0.0] produced [0.5181101436346979]
Epoch 4000 RMSE = 0.4984468757033631
Epoch 4100 RMSE = 0.4980255204318255
Epoch 4200 RMSE = 0.4975378544458163
Epoch 4300 RMSE = 0.4969731401525111
Epoch 4400 RMSE = 0.49631940476485004
Epoch 4500 RMSE = 0.49556354368317923
Epoch 4600 RMSE = 0.4946915401283994
Epoch 4700 RMSE = 0.49368882361215266
Epoch 4800 RMSE = 0.4925407775474147
Epoch 4900 RMSE = 0.4912333855743726
Sample [0.0, 0.0]  expected [0.0] produced [0.46229928210476057]
Sample [0.0, 1.0]  expected [1.0] produced [0.5390355778425429]
Sample [1.0, 0.0]  expected [1.0] produced [0.49803767576497177]
Sample [1.0, 1.0]  expected [0.0] produced [0.5303403041415033]
Epoch 5000 RMSE = 0.48975397851315844
Epoch 5100 RMSE = 0.48809201413873454
Epoch 5200 RMSE = 0.48623979817880447
Epoch 5300 RMSE = 0.4841930460587689
Epoch 5400 RMSE = 0.48195119741102543
Epoch 5500 RMSE = 0.4795174295926711
Epoch 5600 RMSE = 0.4768983652619841
Epoch 5700 RMSE = 0.47410351976461723
Epoch 5800 RMSE = 0.47114457320288444
Epoch 5900 RMSE = 0.46803457045102304
Sample [0.0, 0.0]  expected [0.0] produced [0.41387029851365514]
Sample [0.0, 1.0]  expected [1.0] produced [0.5786332818665127]
Sample [1.0, 0.0]  expected [1.0] produced [0.5174352252885988]
Sample [1.0, 1.0]  expected [0.0] produced [0.5314142218283177]
Epoch 6000 RMSE = 0.4647871480839454
Epoch 6100 RMSE = 0.46141586501767046
Epoch 6200 RMSE = 0.45793368224459097
Epoch 6300 RMSE = 0.4543526052408683
Epoch 6400 RMSE = 0.45068347689077715
Epoch 6500 RMSE = 0.44693589219480667
Epoch 6600 RMSE = 0.44311819868641195
Epoch 6700 RMSE = 0.4392375465536181
Epoch 6800 RMSE = 0.4352999573955696
Epoch 6900 RMSE = 0.43131038794170107
Sample [0.0, 0.0]  expected [0.0] produced [0.359117085015662]
Sample [0.0, 1.0]  expected [1.0] produced [0.6130285455234531]
Sample [1.0, 0.0]  expected [1.0] produced [0.5557506751286894]
Sample [1.0, 1.0]  expected [0.0] produced [0.5041613240636283]
Epoch 7000 RMSE = 0.4272727731379706
Epoch 7100 RMSE = 0.42319004068820515
Epoch 7200 RMSE = 0.41906409593149657
Epoch 7300 RMSE = 0.4148957816819621
Epoch 7400 RMSE = 0.4106848223195032
Epoch 7500 RMSE = 0.40642976494722743
Epoch 7600 RMSE = 0.4021279327176839
Epoch 7700 RMSE = 0.3977754063617145
Epoch 7800 RMSE = 0.3933670494823973
Epoch 7900 RMSE = 0.38889659137303817
Sample [0.0, 0.0]  expected [0.0] produced [0.3128017659779403]
Sample [0.0, 1.0]  expected [1.0] produced [0.6350953745745062]
Sample [1.0, 0.0]  expected [1.0] produced [0.6110782863010674]
Sample [1.0, 1.0]  expected [0.0] produced [0.45679328130640784]
Epoch 8000 RMSE = 0.384356778166142
Epoch 8100 RMSE = 0.3797395992512999
Epoch 8200 RMSE = 0.37503659130359723
Epoch 8300 RMSE = 0.37023921702767226
Epoch 8400 RMSE = 0.3653393098456176
Epoch 8500 RMSE = 0.36032956925596643
Epoch 8600 RMSE = 0.3552040846641443
Epoch 8700 RMSE = 0.34995885867052107
Epoch 8800 RMSE = 0.34459229504956707
Epoch 8900 RMSE = 0.3391056132355375
Sample [0.0, 0.0]  expected [0.0] produced [0.2711421819794922]
Sample [0.0, 1.0]  expected [1.0] produced [0.6711032545417692]
Sample [1.0, 0.0]  expected [1.0] produced [0.6680836297619009]
Sample [1.0, 1.0]  expected [0.0] produced [0.3912004333664901]
Epoch 9000 RMSE = 0.33350315137933134
Epoch 9100 RMSE = 0.3277925249565385
Epoch 9200 RMSE = 0.3219846177460496
Epoch 9300 RMSE = 0.3160933960102926
Epoch 9400 RMSE = 0.3101355531305979
Epoch 9500 RMSE = 0.3041300083250242
Epoch 9600 RMSE = 0.2980972967962753
Epoch 9700 RMSE = 0.29205889759460874
Epoch 9800 RMSE = 0.2860365484691142
Epoch 9900 RMSE = 0.2800515940249574
Sample [0.0, 0.0]  expected [0.0] produced [0.23217867567666015]
Sample [0.0, 1.0]  expected [1.0] produced [0.7271828794300994]
Sample [1.0, 0.0]  expected [1.0] produced [0.7252697566269523]
Sample [1.0, 1.0]  expected [0.0] produced [0.31106901771761564]
Epoch 10000 RMSE = 0.2741244056640323
Final Epoch RMSE = 0.2741244056640323

Please Enter to test XOR with bias
This is the sample run of XOR with bias.
Sample [0.0, 0.0, 1.0]  expected [0.0] produced [0.7848634652845952]
Sample [0.0, 1.0, 1.0]  expected [1.0] produced [0.8153506720554805]
Sample [1.0, 0.0, 1.0]  expected [1.0] produced [0.8161808441126379]
Sample [1.0, 1.0, 1.0]  expected [0.0] produced [0.842333952874806]
Epoch 0 RMSE = 0.5902164864873193
Epoch 100 RMSE = 0.5030711290851212
Epoch 200 RMSE = 0.5023818601368346
Epoch 300 RMSE = 0.502292710487406
Epoch 400 RMSE = 0.50220647666384
Epoch 500 RMSE = 0.5021225490012767
Epoch 600 RMSE = 0.5020404289902917
Epoch 700 RMSE = 0.501959590868369
Epoch 800 RMSE = 0.501879523112642
Epoch 900 RMSE = 0.5017997233637572
Sample [0.0, 0.0, 1.0]  expected [0.0] produced [0.49793976476152735]
Sample [0.0, 1.0, 1.0]  expected [1.0] produced [0.5046048433712051]
Sample [1.0, 0.0, 1.0]  expected [1.0] produced [0.4961045971609967]
Sample [1.0, 1.0, 1.0]  expected [0.0] produced [0.5095288480568946]
Epoch 1000 RMSE = 0.5017196912984632
Epoch 1100 RMSE = 0.5016389214266
Epoch 1200 RMSE = 0.501556895747344
Epoch 1300 RMSE = 0.5014730761059586
Epoch 1400 RMSE = 0.501386896083183
Epoch 1500 RMSE = 0.5012977522395704
Epoch 1600 RMSE = 0.5012049945227464
Epoch 1700 RMSE = 0.5011079156264924
Epoch 1800 RMSE = 0.5010057390667575
Epoch 1900 RMSE = 0.5008976057111917
Sample [0.0, 0.0, 1.0]  expected [0.0] produced [0.4906181080243777]
Sample [0.0, 1.0, 1.0]  expected [1.0] produced [0.5118987483510452]
Sample [1.0, 0.0, 1.0]  expected [1.0] produced [0.49018503222806054]
Sample [1.0, 1.0, 1.0]  expected [0.0] produced [0.5140743353903227]
Epoch 2000 RMSE = 0.5007825584657482
Epoch 2100 RMSE = 0.500659524784705
Epoch 2200 RMSE = 0.5005272966299337
Epoch 2300 RMSE = 0.5003845074627261
Epoch 2400 RMSE = 0.500229605809224
Epoch 2500 RMSE = 0.5000608249018134
Epoch 2600 RMSE = 0.49987614786865603
Epoch 2700 RMSE = 0.4996732679287642
Epoch 2800 RMSE = 0.4994495430600751
Epoch 2900 RMSE = 0.4992019446553737
Sample [0.0, 0.0, 1.0]  expected [0.0] produced [0.4803891333556788]
Sample [0.0, 1.0, 1.0]  expected [1.0] produced [0.5244124597451434]
Sample [1.0, 0.0, 1.0]  expected [1.0] produced [0.4817262981552547]
Sample [1.0, 1.0, 1.0]  expected [0.0] produced [0.5197573919903131]
Epoch 3000 RMSE = 0.4989269997816096
Epoch 3100 RMSE = 0.4986207268320076
Epoch 3200 RMSE = 0.4982785646310931
Epoch 3300 RMSE = 0.49789529544752265
Epoch 3400 RMSE = 0.4974649629181911
Epoch 3500 RMSE = 0.49698078662026546
Epoch 3600 RMSE = 0.4964350759754225
Epoch 3700 RMSE = 0.49581914735923277
Epoch 3800 RMSE = 0.49512324973912825
Epoch 3900 RMSE = 0.49433650588868405
Sample [0.0, 0.0, 1.0]  expected [0.0] produced [0.46112396196423144]
Sample [0.0, 1.0, 1.0]  expected [1.0] produced [0.5492398807921931]
Sample [1.0, 0.0, 1.0]  expected [1.0] produced [0.4691250339787736]
Sample [1.0, 1.0, 1.0]  expected [0.0] produced [0.5256529878094055]
Epoch 4000 RMSE = 0.4934468782216631
Epoch 4100 RMSE = 0.49244117052783004
Epoch 4200 RMSE = 0.4913050792933534
Epoch 4300 RMSE = 0.49002331066971444
Epoch 4400 RMSE = 0.48857978115163114
Epoch 4500 RMSE = 0.48695792097716595
Epoch 4600 RMSE = 0.4851410981013873
Epoch 4700 RMSE = 0.483113175780879
Epoch 4800 RMSE = 0.4808592064506572
Epoch 4900 RMSE = 0.4783662468653955
Sample [0.0, 0.0, 1.0]  expected [0.0] produced [0.4144270339210292]
Sample [0.0, 1.0, 1.0]  expected [1.0] produced [0.6037515721409057]
Sample [1.0, 0.0, 1.0]  expected [1.0] produced [0.4527527133991513]
Sample [1.0, 1.0, 1.0]  expected [0.0] produced [0.5259577420758123]
Epoch 5000 RMSE = 0.4756242535255282
Epoch 5100 RMSE = 0.4726269844703078
Epoch 5200 RMSE = 0.4693727982263253
Epoch 5300 RMSE = 0.4658652115882493
Epoch 5400 RMSE = 0.46211306631655297
Epoch 5500 RMSE = 0.4581301716014985
Epoch 5600 RMSE = 0.45393433964934243
Epoch 5700 RMSE = 0.4495458114161145
Epoch 5800 RMSE = 0.4449851626766933
Epoch 5900 RMSE = 0.4402708647709372
Sample [0.0, 0.0, 1.0]  expected [0.0] produced [0.3188269965542972]
Sample [0.0, 1.0, 1.0]  expected [1.0] produced [0.6944252869814839]
Sample [1.0, 0.0, 1.0]  expected [1.0] produced [0.45416897019716773]
Sample [1.0, 1.0, 1.0]  expected [0.0] produced [0.5151629250046775]
Epoch 6000 RMSE = 0.43541672893998273
Epoch 6100 RMSE = 0.4304294787550534
Epoch 6200 RMSE = 0.4253066782885911
Epoch 6300 RMSE = 0.4200352143620665
Epoch 6400 RMSE = 0.4145905139935352
Epoch 6500 RMSE = 0.40893669027973173
Epoch 6600 RMSE = 0.40302784925129076
Epoch 6700 RMSE = 0.3968108253186814
Epoch 6800 RMSE = 0.3902295794325891
Epoch 6900 RMSE = 0.3832313078622268
Sample [0.0, 0.0, 1.0]  expected [0.0] produced [0.2215874100538008]
Sample [0.0, 1.0, 1.0]  expected [1.0] produced [0.7542284180552736]
Sample [1.0, 0.0, 1.0]  expected [1.0] produced [0.5207244945372268]
Sample [1.0, 1.0, 1.0]  expected [0.0] produced [0.47498891173889896]
Epoch 7000 RMSE = 0.3757739104825749
Epoch 7100 RMSE = 0.3678338928825485
Epoch 7200 RMSE = 0.3594132189937791
Epoch 7300 RMSE = 0.35054340063803924
Epoch 7400 RMSE = 0.3412854675239433
Epoch 7500 RMSE = 0.33172541294239477
Epoch 7600 RMSE = 0.3219659420959367
Epoch 7700 RMSE = 0.3121163612963685
Epoch 7800 RMSE = 0.30228280977915784
Epoch 7900 RMSE = 0.29256063222480516
Sample [0.0, 0.0, 1.0]  expected [0.0] produced [0.16353449053318173]
Sample [0.0, 1.0, 1.0]  expected [1.0] produced [0.7698150473665039]
Sample [1.0, 0.0, 1.0]  expected [1.0] produced [0.6641594131073655]
Sample [1.0, 1.0, 1.0]  expected [0.0] produced [0.3576393047559422]
Epoch 8000 RMSE = 0.28302977498998894
Epoch 8100 RMSE = 0.2737531243607289
Epoch 8200 RMSE = 0.2647770692275928
Epoch 8300 RMSE = 0.2561333772215719
Epoch 8400 RMSE = 0.2478416055930891
Epoch 8500 RMSE = 0.23991153186809877
Epoch 8600 RMSE = 0.23234533820694958
Epoch 8700 RMSE = 0.22513945539045307
Epoch 8800 RMSE = 0.21828606632097614
Epoch 8900 RMSE = 0.21177430846579126
Sample [0.0, 0.0, 1.0]  expected [0.0] produced [0.12159310073162925]
Sample [0.0, 1.0, 1.0]  expected [1.0] produced [0.810282789975905]
Sample [1.0, 0.0, 1.0]  expected [1.0] produced [0.7697917498884455]
Sample [1.0, 1.0, 1.0]  expected [0.0] produced [0.25553407972690345]
Epoch 9000 RMSE = 0.20559122442735622
Epoch 9100 RMSE = 0.19972250669937328
Epoch 9200 RMSE = 0.19415307568627635
Epoch 9300 RMSE = 0.1888675230518863
Epoch 9400 RMSE = 0.1838504465679247
Epoch 9500 RMSE = 0.17908669791495482
Epoch 9600 RMSE = 0.17456156110199866
Epoch 9700 RMSE = 0.17026087606853352
Epoch 9800 RMSE = 0.16617111943899981
Epoch 9900 RMSE = 0.16227945220767037
Sample [0.0, 0.0, 1.0]  expected [0.0] produced [0.09306037468778179]
Sample [0.0, 1.0, 1.0]  expected [1.0] produced [0.8474543319777288]
Sample [1.0, 0.0, 1.0]  expected [1.0] produced [0.8263425196211337]
Sample [1.0, 1.0, 1.0]  expected [0.0] produced [0.19620191715623697]
Epoch 10000 RMSE = 0.1585737422752507
Final Epoch RMSE = 0.1585737422752507

Please Enter to test trans sin
This is the sample run of transform sin.
Sample [5.38]  expected [0.10734852445560966] produced [0.8035793954087468]
Sample [5.31]  expected [0.08665891088398214] produced [0.8006112838038436]
Sample [6.24]  expected [0.47841405739563564] produced [0.8039053935923794]
Sample [2.13]  expected [0.9238389200667849] produced [0.7495332029499673]
Sample [4.54]  expected [0.007411109248070258] produced [0.7906889143974204]
Sample [5.78]  expected [0.2588907641275342] produced [0.7978283668471949]
Sample [5.11]  expected [0.03900566122589194] produced [0.7910043675371011]
Sample [2.31]  expected [0.8695026390297355] produced [0.7471939571759761]
Sample [6.26]  expected [0.4884083850038103] produced [0.7961791125115842]
Sample [4.42]  expected [0.0212209962753645] produced [0.7809331183572763]
Sample [3.61]  expected [0.2742671239192884] produced [0.7678480663647063]
Sample [0.94]  expected [0.9037790502025571] produced [0.7033324233112225]
Sample [4.88]  expected [0.007006936319164825] produced [0.7811726091586323]
Sample [0.95]  expected [0.9067077523946869] produced [0.7017158371996673]
Sample [3.71]  expected [0.23085474585499116] produced [0.7655605943212741]
Sample [3.46]  expected [0.3434728204485149] produced [0.7598051494690382]
Sample [2.37]  expected [0.8486388691299689] produced [0.738020244611101]
Sample [3.0]  expected [0.5705600040299336] produced [0.7509902183760814]
Sample [2.98]  expected [0.5804451574837279] produced [0.7499348593978298]
Sample [2.87]  expected [0.634133025464809] produced [0.7472677446576754]
Sample [6.13]  expected [0.4237065456757194] produced [0.7824242250113627]
Sample [5.32]  expected [0.0894928766443765] produced [0.7755974279732495]
Sample [0.11]  expected [0.5548891504185874] produced [0.666414363851919]
Sample [4.52]  expected [0.009224873454242222] produced [0.7653067328036646]
Sample [0.07]  expected [0.5349714236687664] produced [0.6627141754024642]
Sample [5.41]  expected [0.11681048666210786] produced [0.7694201992929828]
Sample [0.05]  expected [0.5249895846353392] produced [0.6599496295259886]
Sample [0.71]  expected [0.8259168855107684] produced [0.681566056903298]
Sample [4.86]  expected [0.005437369602815101] produced [0.7621552578770521]
Sample [2.62]  expected [0.7491308212059193] produced [0.7283748155902177]
Sample [2.53]  expected [0.7870860741772862] produced [0.7266848060991391]
Sample [0.51]  expected [0.7440886234414538] produced [0.6733112697514639]
Sample [6.11]  expected [0.41383956214219486] produced [0.7683584629778576]
Sample [4.33]  expected [0.036112067677562254] produced [0.7525194808079158]
Sample [5.79]  expected [0.2632830014590325] produced [0.7615162399593705]
Sample [2.51]  expected [0.7952154590569563] produced [0.7200882280842077]
Sample [5.96]  expected [0.3412057169639826] produced [0.7605638308402375]
Sample [0.59]  expected [0.7781805114563919] produced [0.6697650262598822]
Sample [3.4]  expected [0.3722294489865844] produced [0.7340269890826474]
Sample [2.52]  expected [0.7911653247620409] produced [0.7175738592188097]
Sample [5.22]  expected [0.06304586903548881] produced [0.7524762613937573]
Sample [5.48]  expected [0.14021416358974625] produced [0.7510944840851709]
Sample [4.81]  expected [0.0023800871154187053] produced [0.7427379454615509]
Sample [5.49]  expected [0.1437041857600192] produced [0.7444852041626525]
Sample [0.32]  expected [0.6572832803080588] produced [0.6522560263311958]
Sample [1.63]  expected [0.9991239871888162] produced [0.6873820871855968]
Sample [0.08]  expected [0.5399573469845863] produced [0.6459639062704795]
Sample [5.61]  expected [0.18826022736065734] produced [0.7432611605252062]
Sample [0.98]  expected [0.9152486852459851] produced [0.6695604156749764]
Sample [5.77]  expected [0.2545226375186995] produced [0.7424054552518268]
Sample [1.08]  expected [0.9409789034424738] produced [0.67103079974365]
Sample [1.21]  expected [0.9678080007766929] produced [0.6751960126704261]
Sample [1.38]  expected [0.99092676518618] produced [0.680391156160278]
Sample [3.97]  expected [0.13157220381780832] produced [0.7275611190296095]
Sample [0.96]  expected [0.9095957841504991] produced [0.6682865615553056]
Sample [2.96]  expected [0.5902981339471165] produced [0.7117624224038788]
Sample [1.85]  expected [0.9806376014876499] produced [0.690262130061889]
Sample [3.42]  expected [0.36258766483843796] produced [0.7192420662996337]
Sample [3.91]  expected [0.15250451339176402] produced [0.7237860258533714]
Sample [5.72]  expected [0.23305886680417787] produced [0.7369606380928144]
Sample [1.36]  expected [0.9889323012176582] produced [0.6740861175854529]
Sample [0.04]  expected [0.5199946670933171] produced [0.639752341190965]
Epoch 0 RMSE = 0.44507816031163927
Epoch 100 RMSE = 0.26176587929323464
Epoch 200 RMSE = 0.23872595370323044
Epoch 300 RMSE = 0.22771526917719223
Epoch 400 RMSE = 0.22147648857376603
Epoch 500 RMSE = 0.21751839538013817
Epoch 600 RMSE = 0.2148049735707129
Epoch 700 RMSE = 0.21283716036732803
Epoch 800 RMSE = 0.21134772723226125
Epoch 900 RMSE = 0.21018174059960895
Sample [5.38]  expected [0.10734852445560966] produced [0.21980495483039]
Sample [5.31]  expected [0.08665891088398214] produced [0.2212630576169302]
Sample [6.24]  expected [0.47841405739563564] produced [0.16922460347671617]
Sample [2.13]  expected [0.9238389200667849] produced [0.6489149931350086]
Sample [4.54]  expected [0.007411109248070258] produced [0.29739657966453087]
Sample [5.78]  expected [0.2588907641275342] produced [0.1935606139414086]
Sample [5.11]  expected [0.03900566122589194] produced [0.23702816583177436]
Sample [2.31]  expected [0.8695026390297355] produced [0.6085640127944372]
Sample [6.26]  expected [0.4884083850038103] produced [0.1744785617820476]
Sample [4.42]  expected [0.0212209962753645] produced [0.3101627204328751]
Sample [3.61]  expected [0.2742671239192884] produced [0.3949810549031708]
Sample [0.94]  expected [0.9037790502025571] produced [0.8319461264400725]
Sample [4.88]  expected [0.007006936319164825] produced [0.2486688609503702]
Sample [0.95]  expected [0.9067077523946869] produced [0.8282662489625865]
Sample [3.71]  expected [0.23085474585499116] produced [0.36568814154487744]
Sample [3.46]  expected [0.3434728204485149] produced [0.39417242650321277]
Sample [2.37]  expected [0.8486388691299689] produced [0.5810106959896798]
Sample [3.0]  expected [0.5705600040299336] produced [0.47829817045411815]
Sample [2.98]  expected [0.5804451574837279] produced [0.48688989617782613]
Sample [2.87]  expected [0.634133025464809] produced [0.5111922629629778]
Sample [6.13]  expected [0.4237065456757194] produced [0.17942875515220072]
Sample [5.32]  expected [0.0894928766443765] produced [0.2297645319593067]
Sample [0.11]  expected [0.5548891504185874] produced [0.9147278706525266]
Sample [4.52]  expected [0.009224873454242222] produced [0.2909092571544532]
Sample [0.07]  expected [0.5349714236687664] produced [0.9171137837682379]
Sample [5.41]  expected [0.11681048666210786] produced [0.20900871428230292]
Sample [0.05]  expected [0.5249895846353392] produced [0.9183701437764463]
Sample [0.71]  expected [0.8259168855107684] produced [0.8581315857921689]
Sample [4.86]  expected [0.005437369602815101] produced [0.24480363574477607]
Sample [2.62]  expected [0.7491308212059193] produced [0.5359352161940504]
Sample [2.53]  expected [0.7870860741772862] produced [0.5628201035224653]
Sample [0.51]  expected [0.7440886234414538] produced [0.8807273946759627]
Sample [6.11]  expected [0.41383956214219486] produced [0.1768042606443022]
Sample [4.33]  expected [0.036112067677562254] produced [0.31046308611917744]
Sample [5.79]  expected [0.2632830014590325] produced [0.18787606426422285]
Sample [2.51]  expected [0.7952154590569563] produced [0.5717359351386627]
Sample [5.96]  expected [0.3412057169639826] produced [0.18749691452414127]
Sample [0.59]  expected [0.7781805114563919] produced [0.8742984904629277]
Sample [3.4]  expected [0.3722294489865844] produced [0.43600575065668873]
Sample [2.52]  expected [0.7911653247620409] produced [0.5808448736214727]
Sample [5.22]  expected [0.06304586903548881] produced [0.2399999907890877]
Sample [5.48]  expected [0.14021416358974625] produced [0.2164591720901484]
Sample [4.81]  expected [0.0023800871154187053] produced [0.263803154896958]
Sample [5.49]  expected [0.1437041857600192] produced [0.205559771152576]
Sample [0.32]  expected [0.6572832803080588] produced [0.8973032463432712]
Sample [1.63]  expected [0.9991239871888162] produced [0.7277755410718456]
Sample [0.08]  expected [0.5399573469845863] produced [0.9164326526082356]
Sample [5.61]  expected [0.18826022736065734] produced [0.20155890271353255]
Sample [0.98]  expected [0.9152486852459851] produced [0.8277848269157326]
Sample [5.77]  expected [0.2545226375186995] produced [0.1933319685185297]
Sample [1.08]  expected [0.9409789034424738] produced [0.8157369867701766]
Sample [1.21]  expected [0.9678080007766929] produced [0.7987142175922564]
Sample [1.38]  expected [0.99092676518618] produced [0.7755063902486012]
Sample [3.97]  expected [0.13157220381780832] produced [0.3583504277931892]
Sample [0.96]  expected [0.9095957841504991] produced [0.8307274533962123]
Sample [2.96]  expected [0.5902981339471165] produced [0.49743566157401703]
Sample [1.85]  expected [0.9806376014876499] produced [0.6992881667298998]
Sample [3.42]  expected [0.36258766483843796] produced [0.4389877373152843]
Sample [3.91]  expected [0.15250451339176402] produced [0.36499344429266994]
Sample [5.72]  expected [0.23305886680417787] produced [0.19573183008238174]
Sample [1.36]  expected [0.9889323012176582] produced [0.7757405099467256]
Sample [0.04]  expected [0.5199946670933171] produced [0.9195030455234728]
Epoch 1000 RMSE = 0.20924365292910954
Epoch 1100 RMSE = 0.2084715669719219
Epoch 1200 RMSE = 0.20782374283182342
Epoch 1300 RMSE = 0.20727107967433614
Epoch 1400 RMSE = 0.20679271007833527
Epoch 1500 RMSE = 0.2063733066862645
Epoch 1600 RMSE = 0.20600137514297676
Epoch 1700 RMSE = 0.20566813795558384
Epoch 1800 RMSE = 0.20536678467228348
Epoch 1900 RMSE = 0.20509195601496968
Sample [5.38]  expected [0.10734852445560966] produced [0.18680245492595499]
Sample [5.31]  expected [0.08665891088398214] produced [0.18939262071154345]
Sample [6.24]  expected [0.47841405739563564] produced [0.137591931834605]
Sample [2.13]  expected [0.9238389200667849] produced [0.6925921611696303]
Sample [4.54]  expected [0.007411109248070258] produced [0.27254478253527015]
Sample [5.78]  expected [0.2588907641275342] produced [0.1614004809853035]
Sample [5.11]  expected [0.03900566122589194] produced [0.20734089194405492]
Sample [2.31]  expected [0.8695026390297355] produced [0.647164246960523]
Sample [6.26]  expected [0.4884083850038103] produced [0.14308288608053918]
Sample [4.42]  expected [0.0212209962753645] produced [0.2892775378156646]
Sample [3.61]  expected [0.2742671239192884] produced [0.38908532594309847]
Sample [0.94]  expected [0.9037790502025571] produced [0.8838541141305641]
Sample [4.88]  expected [0.007006936319164825] produced [0.21938572997720907]
Sample [0.95]  expected [0.9067077523946869] produced [0.8806125272082226]
Sample [3.71]  expected [0.23085474585499116] produced [0.35312109068614045]
Sample [3.46]  expected [0.3434728204485149] produced [0.3870370551897086]
Sample [2.37]  expected [0.8486388691299689] produced [0.613108617866613]
Sample [3.0]  expected [0.5705600040299336] produced [0.48863388130097557]
Sample [2.98]  expected [0.5804451574837279] produced [0.4991420288362535]
Sample [2.87]  expected [0.634133025464809] produced [0.5286099900389032]
Sample [6.13]  expected [0.4237065456757194] produced [0.1473879539278417]
Sample [5.32]  expected [0.0894928766443765] produced [0.19932500275190526]
Sample [0.11]  expected [0.5548891504185874] produced [0.952002271357123]
Sample [4.52]  expected [0.009224873454242222] produced [0.26797435906423017]
Sample [0.07]  expected [0.5349714236687664] produced [0.9537969815179453]
Sample [5.41]  expected [0.11681048666210786] produced [0.17901663531715575]
Sample [0.05]  expected [0.5249895846353392] produced [0.9547500398246811]
Sample [0.71]  expected [0.8259168855107684] produced [0.9075246417762574]
Sample [4.86]  expected [0.005437369602815101] produced [0.21805244533674822]
Sample [2.62]  expected [0.7491308212059193] produced [0.5633374105493653]
Sample [2.53]  expected [0.7870860741772862] produced [0.5954402504690026]
Sample [0.51]  expected [0.7440886234414538] produced [0.9261679146582826]
Sample [6.11]  expected [0.41383956214219486] produced [0.14685349690804558]
Sample [4.33]  expected [0.036112067677562254] produced [0.29281324198676584]
Sample [5.79]  expected [0.2632830014590325] produced [0.15814638881237633]
Sample [2.51]  expected [0.7952154590569563] produced [0.60663868786149]
Sample [5.96]  expected [0.3412057169639826] produced [0.15787819249764845]
Sample [0.59]  expected [0.7781805114563919] produced [0.9210430254469805]
Sample [3.4]  expected [0.3722294489865844] produced [0.44305121676461223]
Sample [2.52]  expected [0.7911653247620409] produced [0.6164224525490716]
Sample [5.22]  expected [0.06304586903548881] produced [0.21244876565947976]
Sample [5.48]  expected [0.14021416358974625] produced [0.18803133590019158]
Sample [4.81]  expected [0.0023800871154187053] produced [0.24064983430471862]
Sample [5.49]  expected [0.1437041857600192] produced [0.17757998441119868]
Sample [0.32]  expected [0.6572832803080588] produced [0.9393375788628067]
Sample [1.63]  expected [0.9991239871888162] produced [0.7839733836016888]
Sample [0.08]  expected [0.5399573469845863] produced [0.9533739434509557]
Sample [5.61]  expected [0.18826022736065734] produced [0.1731154054476504]
Sample [0.98]  expected [0.9152486852459851] produced [0.8814003479583115]
Sample [5.77]  expected [0.2545226375186995] produced [0.1646345788022453]
Sample [1.08]  expected [0.9409789034424738] produced [0.8704270183926223]
Sample [1.21]  expected [0.9678080007766929] produced [0.8541435764033993]
Sample [1.38]  expected [0.99092676518618] produced [0.8310609224097848]
Sample [3.97]  expected [0.13157220381780832] produced [0.34651774364686455]
Sample [0.96]  expected [0.9095957841504991] produced [0.8828863434795204]
Sample [2.96]  expected [0.5902981339471165] produced [0.5124042839836922]
Sample [1.85]  expected [0.9806376014876499] produced [0.7492294362460942]
Sample [3.42]  expected [0.36258766483843796] produced [0.4390153562798514]
Sample [3.91]  expected [0.15250451339176402] produced [0.34961896695806444]
Sample [5.72]  expected [0.23305886680417787] produced [0.16252120489258626]
Sample [1.36]  expected [0.9889323012176582] produced [0.8292269865443118]
Sample [0.04]  expected [0.5199946670933171] produced [0.9553946181444208]
Epoch 2000 RMSE = 0.20483938140531924
Epoch 2100 RMSE = 0.20460561943707975
Epoch 2200 RMSE = 0.20438786889244767
Epoch 2300 RMSE = 0.20418382901184098
Epoch 2400 RMSE = 0.20399159473744377
Epoch 2500 RMSE = 0.20380957717343323
Epoch 2600 RMSE = 0.2036364424822121
Epoch 2700 RMSE = 0.20347106443108573
Epoch 2800 RMSE = 0.20331248716390155
Epoch 2900 RMSE = 0.20315989571388948
Sample [5.38]  expected [0.10734852445560966] produced [0.17175451744586925]
Sample [5.31]  expected [0.08665891088398214] produced [0.17490165392191823]
Sample [6.24]  expected [0.47841405739563564] produced [0.12571389250423884]
Sample [2.13]  expected [0.9238389200667849] produced [0.7185739845398362]
Sample [4.54]  expected [0.007411109248070258] produced [0.25722812724931443]
Sample [5.78]  expected [0.2588907641275342] produced [0.14796209177612943]
Sample [5.11]  expected [0.03900566122589194] produced [0.19281590350179179]
Sample [2.31]  expected [0.8695026390297355] produced [0.6708062700462477]
Sample [6.26]  expected [0.4884083850038103] produced [0.1305288717588675]
Sample [4.42]  expected [0.0212209962753645] produced [0.2752444759590489]
Sample [3.61]  expected [0.2742671239192884] produced [0.3841385589048329]
Sample [0.94]  expected [0.9037790502025571] produced [0.9120259720369979]
Sample [4.88]  expected [0.007006936319164825] produced [0.20532418993955653]
Sample [0.95]  expected [0.9067077523946869] produced [0.9093204180065185]
Sample [3.71]  expected [0.23085474585499116] produced [0.34648371193777444]
Sample [3.46]  expected [0.3434728204485149] produced [0.3840095742235882]
Sample [2.37]  expected [0.8486388691299689] produced [0.6345799057592574]
Sample [3.0]  expected [0.5705600040299336] produced [0.49497764393893334]
Sample [2.98]  expected [0.5804451574837279] produced [0.506188150918664]
Sample [2.87]  expected [0.634133025464809] produced [0.5384652265455426]
Sample [6.13]  expected [0.4237065456757194] produced [0.13452443322651525]
Sample [5.32]  expected [0.0894928766443765] produced [0.184096159945668]
Sample [0.11]  expected [0.5548891504185874] produced [0.9691490385508296]
Sample [4.52]  expected [0.009224873454242222] produced [0.2546190364298992]
Sample [0.07]  expected [0.5349714236687664] produced [0.9705416146800482]
Sample [5.41]  expected [0.11681048666210786] produced [0.16615856695794198]
Sample [0.05]  expected [0.5249895846353392] produced [0.9712730800446197]
Sample [0.71]  expected [0.8259168855107684] produced [0.9331706099734556]
Sample [4.86]  expected [0.005437369602815101] produced [0.20565006659046775]
Sample [2.62]  expected [0.7491308212059193] produced [0.5813878782996762]
Sample [2.53]  expected [0.7870860741772862] produced [0.6157330453929476]
Sample [0.51]  expected [0.7440886234414538] produced [0.9487431199237918]
Sample [6.11]  expected [0.41383956214219486] produced [0.13503998946609888]
Sample [4.33]  expected [0.036112067677562254] produced [0.28154805484954026]
Sample [5.79]  expected [0.2632830014590325] produced [0.14617048408402825]
Sample [2.51]  expected [0.7952154590569563] produced [0.6275183862131096]
Sample [5.96]  expected [0.3412057169639826] produced [0.14523931605705667]
Sample [0.59]  expected [0.7781805114563919] produced [0.9444096919384691]
Sample [3.4]  expected [0.3722294489865844] produced [0.44398111798231094]
Sample [2.52]  expected [0.7911653247620409] produced [0.6364727842475427]
Sample [5.22]  expected [0.06304586903548881] produced [0.19757875675005085]
Sample [5.48]  expected [0.14021416358974625] produced [0.17440013934790682]
Sample [4.81]  expected [0.0023800871154187053] produced [0.2278501070425466]
Sample [5.49]  expected [0.1437041857600192] produced [0.16552840001338204]
Sample [0.32]  expected [0.6572832803080588] produced [0.9594561398905485]
Sample [1.63]  expected [0.9991239871888162] produced [0.817246868587748]
Sample [0.08]  expected [0.5399573469845863] produced [0.9702373717222504]
Sample [5.61]  expected [0.18826022736065734] produced [0.16070908440006748]
Sample [0.98]  expected [0.9152486852459851] produced [0.9102031799084099]
Sample [5.77]  expected [0.2545226375186995] produced [0.15241913679276728]
Sample [1.08]  expected [0.9409789034424738] produced [0.9001438973094259]
Sample [1.21]  expected [0.9678080007766929] produced [0.8848546209127695]
Sample [1.38]  expected [0.99092676518618] produced [0.8624956569371927]
Sample [3.97]  expected [0.13157220381780832] produced [0.33725030966248976]
Sample [0.96]  expected [0.9095957841504991] produced [0.9111561729717708]
Sample [2.96]  expected [0.5902981339471165] produced [0.520565935302738]
Sample [1.85]  expected [0.9806376014876499] produced [0.7791604748291628]
Sample [3.42]  expected [0.36258766483843796] produced [0.4362093037048412]
Sample [3.91]  expected [0.15250451339176402] produced [0.33904951894133345]
Sample [5.72]  expected [0.23305886680417787] produced [0.14886591917282813]
Sample [1.36]  expected [0.9889323012176582] produced [0.8602357737237842]
Sample [0.04]  expected [0.5199946670933171] produced [0.9717091824922388]
Epoch 3000 RMSE = 0.20301259243546851
Epoch 3100 RMSE = 0.2028699780037899
Epoch 3200 RMSE = 0.20273153597031188
Epoch 3300 RMSE = 0.20259682011029898
Epoch 3400 RMSE = 0.2024654439805734
Epoch 3500 RMSE = 0.202337072241518
Epoch 3600 RMSE = 0.20221141339902327
Epoch 3700 RMSE = 0.20208821369878754
Epoch 3800 RMSE = 0.20196725196355128
Epoch 3900 RMSE = 0.2018483352081172
Sample [5.38]  expected [0.10734852445560966] produced [0.16393671100223323]
Sample [5.31]  expected [0.08665891088398214] produced [0.16730762314910164]
Sample [6.24]  expected [0.47841405739563564] produced [0.12101927587248126]
Sample [2.13]  expected [0.9238389200667849] produced [0.7379295090110745]
Sample [4.54]  expected [0.007411109248070258] produced [0.24649309030091737]
Sample [5.78]  expected [0.2588907641275342] produced [0.14165157603539208]
Sample [5.11]  expected [0.03900566122589194] produced [0.18446061333079183]
Sample [2.31]  expected [0.8695026390297355] produced [0.6884449189283657]
Sample [6.26]  expected [0.4884083850038103] produced [0.12503811523872868]
Sample [4.42]  expected [0.0212209962753645] produced [0.26459956086057096]
Sample [3.61]  expected [0.2742671239192884] produced [0.37952800333066483]
Sample [0.94]  expected [0.9037790502025571] produced [0.9315143622403491]
Sample [4.88]  expected [0.007006936319164825] produced [0.19735914407269167]
Sample [0.95]  expected [0.9067077523946869] produced [0.9292779574974864]
Sample [3.71]  expected [0.23085474585499116] produced [0.3420264519164587]
Sample [3.46]  expected [0.3434728204485149] produced [0.3823296577873407]
Sample [2.37]  expected [0.8486388691299689] produced [0.6516735387461877]
Sample [3.0]  expected [0.5705600040299336] produced [0.4994899244915129]
Sample [2.98]  expected [0.5804451574837279] produced [0.510968077342677]
Sample [2.87]  expected [0.634133025464809] produced [0.5451964526145496]
Sample [6.13]  expected [0.4237065456757194] produced [0.1288803856097755]
Sample [5.32]  expected [0.0894928766443765] produced [0.1753020904670647]
Sample [0.11]  expected [0.5548891504185874] produced [0.9794919655173935]
Sample [4.52]  expected [0.009224873454242222] produced [0.24538482843882997]
Sample [0.07]  expected [0.5349714236687664] produced [0.98057209358759]
Sample [5.41]  expected [0.11681048666210786] produced [0.15976243898720463]
Sample [0.05]  expected [0.5249895846353392] produced [0.9811310286582859]
Sample [0.71]  expected [0.8259168855107684] produced [0.9502418849179296]
Sample [4.86]  expected [0.005437369602815101] produced [0.19866870666104558]
Sample [2.62]  expected [0.7491308212059193] produced [0.5951144761696586]
Sample [2.53]  expected [0.7870860741772862] produced [0.6307153931889745]
Sample [0.51]  expected [0.7440886234414538] produced [0.9632647112559001]
Sample [6.11]  expected [0.41383956214219486] produced [0.12989827757661332]
Sample [4.33]  expected [0.036112067677562254] produced [0.27295210789433944]
Sample [5.79]  expected [0.2632830014590325] produced [0.14075859367770083]
Sample [2.51]  expected [0.7952154590569563] produced [0.642582698631429]
Sample [5.96]  expected [0.3412057169639826] produced [0.1391337351689126]
Sample [0.59]  expected [0.7781805114563919] produced [0.9595670354312243]
Sample [3.4]  expected [0.3722294489865844] produced [0.44225785760392483]
Sample [2.52]  expected [0.7911653247620409] produced [0.6504665905232678]
Sample [5.22]  expected [0.06304586903548881] produced [0.18833620524520012]
Sample [5.48]  expected [0.14021416358974625] produced [0.1667575646205995]
Sample [4.81]  expected [0.0023800871154187053] produced [0.219302735228264]
Sample [5.49]  expected [0.1437041857600192] produced [0.15942642106679666]
Sample [0.32]  expected [0.6572832803080588] produced [0.9719935662540784]
Sample [1.63]  expected [0.9991239871888162] produced [0.8416852893826486]
Sample [0.08]  expected [0.5399573469845863] produced [0.9803436320917722]
Sample [5.61]  expected [0.18826022736065734] produced [0.1543854611067749]
Sample [0.98]  expected [0.9152486852459851] produced [0.9299507595945076]
Sample [5.77]  expected [0.2545226375186995] produced [0.14639926982190557]
Sample [1.08]  expected [0.9409789034424738] produced [0.9207217600583022]
Sample [1.21]  expected [0.9678080007766929] produced [0.9064637864084935]
Sample [1.38]  expected [0.99092676518618] produced [0.8850115752762964]
Sample [3.97]  expected [0.13157220381780832] produced [0.32916526214515834]
Sample [0.96]  expected [0.9095957841504991] produced [0.930695377253946]
Sample [2.96]  expected [0.5902981339471165] produced [0.5258694492439537]
Sample [1.85]  expected [0.9806376014876499] produced [0.8016127192607858]
Sample [3.42]  expected [0.36258766483843796] produced [0.4325597250698138]
Sample [3.91]  expected [0.15250451339176402] produced [0.3309231568528366]
Sample [5.72]  expected [0.23305886680417787] produced [0.14264674657639573]
Sample [1.36]  expected [0.9889323012176582] produced [0.8829322600873005]
Sample [0.04]  expected [0.5199946670933171] produced [0.9814417898458497]
Epoch 4000 RMSE = 0.20173129490080796
Epoch 4100 RMSE = 0.20161598376587989
Epoch 4200 RMSE = 0.20150227304130078
Epoch 4300 RMSE = 0.2013900501216542
Epoch 4400 RMSE = 0.20127921652790973
Epoch 4500 RMSE = 0.20116968615524847
Epoch 4600 RMSE = 0.20106138375771432
Epoch 4700 RMSE = 0.20095424363465722
Epoch 4800 RMSE = 0.2008482084891028
Epoch 4900 RMSE = 0.20074322843256945
Sample [5.38]  expected [0.10734852445560966] produced [0.15960253522901652]
Sample [5.31]  expected [0.08665891088398214] produced [0.16302960170673259]
Sample [6.24]  expected [0.47841405739563564] produced [0.11943484799956805]
Sample [2.13]  expected [0.9238389200667849] produced [0.75366207913943]
Sample [4.54]  expected [0.007411109248070258] produced [0.2386339346658613]
Sample [5.78]  expected [0.2588907641275342] produced [0.13860415124859057]
Sample [5.11]  expected [0.03900566122589194] produced [0.17924554120501318]
Sample [2.31]  expected [0.8695026390297355] produced [0.7026364780732067]
Sample [6.26]  expected [0.4884083850038103] produced [0.12275192547852562]
Sample [4.42]  expected [0.0212209962753645] produced [0.2563438650459971]
Sample [3.61]  expected [0.2742671239192884] produced [0.37520074762756206]
Sample [0.94]  expected [0.9037790502025571] produced [0.9459170500078569]
Sample [4.88]  expected [0.007006936319164825] produced [0.1923626676609735]
Sample [0.95]  expected [0.9067077523946869] produced [0.9440642321087089]
Sample [3.71]  expected [0.23085474585499116] produced [0.33849133669997955]
Sample [3.46]  expected [0.3434728204485149] produced [0.38099612976664693]
Sample [2.37]  expected [0.8486388691299689] produced [0.6659263198135134]
Sample [3.0]  expected [0.5705600040299336] produced [0.5027971136430466]
Sample [2.98]  expected [0.5804451574837279] produced [0.5144028671007806]
Sample [2.87]  expected [0.634133025464809] produced [0.5501924103300349]
Sample [6.13]  expected [0.4237065456757194] produced [0.12650589352163588]
Sample [5.32]  expected [0.0894928766443765] produced [0.16992205043075964]
Sample [0.11]  expected [0.5548891504185874] produced [0.9861560167115396]
Sample [4.52]  expected [0.009224873454242222] produced [0.23860344038015627]
Sample [0.07]  expected [0.5349714236687664] produced [0.9869882639104398]
Sample [5.41]  expected [0.11681048666210786] produced [0.1563431797746117]
Sample [0.05]  expected [0.5249895846353392] produced [0.98741265496936]
Sample [0.71]  expected [0.8259168855107684] produced [0.9623874176354804]
Sample [4.86]  expected [0.005437369602815101] produced [0.19426735777271167]
Sample [2.62]  expected [0.7491308212059193] produced [0.6060742204331434]
Sample [2.53]  expected [0.7870860741772862] produced [0.6425759870843387]
Sample [0.51]  expected [0.7440886234414538] produced [0.9732590032357461]
Sample [6.11]  expected [0.41383956214219486] produced [0.12774503442930443]
Sample [4.33]  expected [0.036112067677562254] produced [0.26613469119916183]
Sample [5.79]  expected [0.2632830014590325] produced [0.13827918372069353]
Sample [2.51]  expected [0.7952154590569563] produced [0.6543885906718628]
Sample [5.96]  expected [0.3412057169639826] produced [0.13612410715395676]
Sample [0.59]  expected [0.7781805114563919] produced [0.9701193415236571]
Sample [3.4]  expected [0.3722294489865844] produced [0.4395425074079103]
Sample [2.52]  expected [0.7911653247620409] produced [0.6613006291562212]
Sample [5.22]  expected [0.06304586903548881] produced [0.18227334770958137]
Sample [5.48]  expected [0.14021416358974625] produced [0.1621730016897426]
Sample [4.81]  expected [0.0023800871154187053] produced [0.21316632985588635]
Sample [5.49]  expected [0.1437041857600192] produced [0.15606871399730168]
Sample [0.32]  expected [0.6572832803080588] produced [0.9803461158152751]
Sample [1.63]  expected [0.9991239871888162] produced [0.8609708251381587]
Sample [0.08]  expected [0.5399573469845863] produced [0.9868141463123437]
Sample [5.61]  expected [0.18826022736065734] produced [0.15092790034654582]
Sample [0.98]  expected [0.9152486852459851] produced [0.9444904341112835]
Sample [5.77]  expected [0.2545226375186995] produced [0.14327931255437565]
Sample [1.08]  expected [0.9409789034424738] produced [0.936062055844104]
Sample [1.21]  expected [0.9678080007766929] produced [0.9228507413463785]
Sample [1.38]  expected [0.99092676518618] produced [0.902431795461697]
Sample [3.97]  expected [0.13157220381780832] produced [0.32218990147391074]
Sample [0.96]  expected [0.9095957841504991] produced [0.9451468630358708]
Sample [2.96]  expected [0.5902981339471165] produced [0.5295837318361013]
Sample [1.85]  expected [0.9806376014876499] produced [0.8197937329039691]
Sample [3.42]  expected [0.36258766483843796] produced [0.42881703131617077]
Sample [3.91]  expected [0.15250451339176402] produced [0.3243864481852701]
Sample [5.72]  expected [0.23305886680417787] produced [0.1397600261160593]
Sample [1.36]  expected [0.9889323012176582] produced [0.9007061358294292]
Sample [0.04]  expected [0.5199946670933171] produced [0.987638799401237]
Epoch 5000 RMSE = 0.2006392601146239
Epoch 5100 RMSE = 0.2005362659587498
Epoch 5200 RMSE = 0.20043421348894594
Epoch 5300 RMSE = 0.20033307473395592
Epoch 5400 RMSE = 0.20023282569816586
Epoch 5500 RMSE = 0.20013344589003979
Epoch 5600 RMSE = 0.2000349179005047
Epoch 5700 RMSE = 0.19993722702500263
Epoch 5800 RMSE = 0.19984036092398277
Epoch 5900 RMSE = 0.19974430931749573
Sample [5.38]  expected [0.10734852445560966] produced [0.1569136592439089]
Sample [5.31]  expected [0.08665891088398214] produced [0.16032255939972193]
Sample [6.24]  expected [0.47841405739563564] produced [0.11907198339423024]
Sample [2.13]  expected [0.9238389200667849] produced [0.766994787480804]
Sample [4.54]  expected [0.007411109248070258] produced [0.23260695310813298]
Sample [5.78]  expected [0.2588907641275342] produced [0.13698515628337674]
Sample [5.11]  expected [0.03900566122589194] produced [0.17564744027522744]
Sample [2.31]  expected [0.8695026390297355] produced [0.7145449683567489]
Sample [6.26]  expected [0.4884083850038103] produced [0.12182278019641939]
Sample [4.42]  expected [0.0212209962753645] produced [0.2497788375093515]
Sample [3.61]  expected [0.2742671239192884] produced [0.3711953783192975]
Sample [0.94]  expected [0.9037790502025571] produced [0.9567639519193564]
Sample [4.88]  expected [0.007006936319164825] produced [0.18883260317162956]
Sample [0.95]  expected [0.9067077523946869] produced [0.9552178639261185]
Sample [3.71]  expected [0.23085474585499116] produced [0.33539817118916027]
Sample [3.46]  expected [0.3434728204485149] produced [0.37968352526413784]
Sample [2.37]  expected [0.8486388691299689] produced [0.6780736709589661]
Sample [3.0]  expected [0.5705600040299336] produced [0.5052819939066151]
Sample [2.98]  expected [0.5804451574837279] produced [0.5169875643060933]
Sample [2.87]  expected [0.634133025464809] produced [0.5541294296086605]
Sample [6.13]  expected [0.4237065456757194] produced [0.1255092697303864]
Sample [5.32]  expected [0.0894928766443765] produced [0.16633065896615928]
Sample [0.11]  expected [0.5548891504185874] produced [0.990515229666188]
Sample [4.52]  expected [0.009224873454242222] produced [0.23333997979487917]
Sample [0.07]  expected [0.5349714236687664] produced [0.99115430520184]
Sample [5.41]  expected [0.11681048666210786] produced [0.15425508370997346]
Sample [0.05]  expected [0.5249895846353392] produced [0.9914759239089844]
Sample [0.71]  expected [0.8259168855107684] produced [0.9711876487747754]
Sample [4.86]  expected [0.005437369602815101] produced [0.19111530633731158]
Sample [2.62]  expected [0.7491308212059193] produced [0.6151181721258934]
Sample [2.53]  expected [0.7870860741772862] produced [0.6523809597450944]
Sample [0.51]  expected [0.7440886234414538] produced [0.9802596867520955]
Sample [6.11]  expected [0.41383956214219486] produced [0.1268294160774737]
Sample [4.33]  expected [0.036112067677562254] produced [0.2605540636901432]
Sample [5.79]  expected [0.2632830014590325] produced [0.13702338497742322]
Sample [2.51]  expected [0.7952154590569563] produced [0.664119368196336]
Sample [5.96]  expected [0.3412057169639826] produced [0.13450416833974807]
Sample [0.59]  expected [0.7781805114563919] produced [0.9776088450167597]
Sample [3.4]  expected [0.3722294489865844] produced [0.4366032116472167]
Sample [2.52]  expected [0.7911653247620409] produced [0.6702211760222253]
Sample [5.22]  expected [0.06304586903548881] produced [0.17798097892220613]
Sample [5.48]  expected [0.14021416358974625] produced [0.1591266928664553]
Sample [4.81]  expected [0.0023800871154187053] produced [0.20844999768370012]
Sample [5.49]  expected [0.1437041857600192] produced [0.1539417722338938]
Sample [0.32]  expected [0.6572832803080588] produced [0.9860042144048694]
Sample [1.63]  expected [0.9991239871888162] produced [0.8766411356474161]
Sample [0.08]  expected [0.5399573469845863] produced [0.9910206786839215]
Sample [5.61]  expected [0.18826022736065734] produced [0.14878458439765682]
Sample [0.98]  expected [0.9152486852459851] produced [0.9554402561694277]
Sample [5.77]  expected [0.2545226375186995] produced [0.1414673797793423]
Sample [1.08]  expected [0.9409789034424738] produced [0.9477823957819577]
Sample [1.21]  expected [0.9678080007766929] produced [0.935604699182529]
Sample [1.38]  expected [0.99092676518618] produced [0.9162974472407602]
Sample [3.97]  expected [0.13157220381780832] produced [0.31621768344551837]
Sample [0.96]  expected [0.9095957841504991] produced [0.9560496517186721]
Sample [2.96]  expected [0.5902981339471165] produced [0.5323663331168778]
Sample [1.85]  expected [0.9806376014876499] produced [0.8349875204265771]
Sample [3.42]  expected [0.36258766483843796] produced [0.4252983232814018]
Sample [3.91]  expected [0.15250451339176402] produced [0.31894667328845155]
Sample [5.72]  expected [0.23305886680417787] produced [0.13828525873989905]
Sample [1.36]  expected [0.9889323012176582] produced [0.9149270634619455]
Sample [0.04]  expected [0.5199946670933171] produced [0.9916426429085005]
Epoch 6000 RMSE = 0.19964906371023858
Epoch 6100 RMSE = 0.199554617143982
Epoch 6200 RMSE = 0.19946096397474491
Epoch 6300 RMSE = 0.19936809967244512
Epoch 6400 RMSE = 0.19927602064103297
Epoch 6500 RMSE = 0.19918472405734322
Epoch 6600 RMSE = 0.1990942077270721
Epoch 6700 RMSE = 0.19900446995644622
Epoch 6800 RMSE = 0.19891550943826
Epoch 6900 RMSE = 0.19882732515107043
Sample [5.38]  expected [0.10734852445560966] produced [0.15505566652575617]
Sample [5.31]  expected [0.08665891088398214] produced [0.15841567811718257]
Sample [6.24]  expected [0.47841405739563564] produced [0.11916808427528826]
Sample [2.13]  expected [0.9238389200667849] produced [0.7785274937100486]
Sample [4.54]  expected [0.007411109248070258] produced [0.22779603693517142]
Sample [5.78]  expected [0.2588907641275342] produced [0.13601881811796013]
Sample [5.11]  expected [0.03900566122589194] produced [0.1729540445168704]
Sample [2.31]  expected [0.8695026390297355] produced [0.7247842027331585]
Sample [6.26]  expected [0.4884083850038103] produced [0.12147033995363782]
Sample [4.42]  expected [0.0212209962753645] produced [0.24441964066557617]
Sample [3.61]  expected [0.2742671239192884] produced [0.3675266746729326]
Sample [0.94]  expected [0.9037790502025571] produced [0.9650170037542748]
Sample [4.88]  expected [0.007006936319164825] produced [0.18610848296832905]
Sample [0.95]  expected [0.9067077523946869] produced [0.9637155676580148]
Sample [3.71]  expected [0.23085474585499116] produced [0.33259288564712997]
Sample [3.46]  expected [0.3434728204485149] produced [0.37834644132372547]
Sample [2.37]  expected [0.8486388691299689] produced [0.6885850329482325]
Sample [3.0]  expected [0.5705600040299336] produced [0.5071947560738654]
Sample [2.98]  expected [0.5804451574837279] produced [0.5190009418845076]
Sample [2.87]  expected [0.634133025464809] produced [0.5573527477512561]
Sample [6.13]  expected [0.4237065456757194] produced [0.12509429230792918]
Sample [5.32]  expected [0.0894928766443765] produced [0.16372948727780917]
Sample [0.11]  expected [0.5548891504185874] produced [0.9933957309439496]
Sample [4.52]  expected [0.009224873454242222] produced [0.229073803274657]
Sample [0.07]  expected [0.5349714236687664] produced [0.9938869986641892]
Sample [5.41]  expected [0.11681048666210786] produced [0.1528038725892645]
Sample [0.05]  expected [0.5249895846353392] produced [0.9941313699338835]
Sample [0.71]  expected [0.8259168855107684] produced [0.9776305597025787]
Sample [4.86]  expected [0.005437369602815101] produced [0.188639254197204]
Sample [2.62]  expected [0.7491308212059193] produced [0.622761836844072]
Sample [2.53]  expected [0.7870860741772862] produced [0.6607052815962078]
Sample [0.51]  expected [0.7440886234414538] produced [0.9852147789671467]
Sample [6.11]  expected [0.41383956214219486] produced [0.12642442247061653]
Sample [4.33]  expected [0.036112067677562254] produced [0.25585953530066924]
Sample [5.79]  expected [0.2632830014590325] produced [0.13628924498381123]
Sample [2.51]  expected [0.7952154590569563] produced [0.672375035554028]
Sample [5.96]  expected [0.3412057169639826] produced [0.1335269114852356]
Sample [0.59]  expected [0.7781805114563919] produced [0.9829823315505839]
Sample [3.4]  expected [0.3722294489865844] produced [0.4337110352911526]
Sample [2.52]  expected [0.7911653247620409] produced [0.6778071594942945]
Sample [5.22]  expected [0.06304586903548881] produced [0.1747271332988453]
Sample [5.48]  expected [0.14021416358974625] produced [0.15690597291999078]
Sample [4.81]  expected [0.0023800871154187053] produced [0.2046361159748685]
Sample [5.49]  expected [0.1437041857600192] produced [0.15240962645148634]
Sample [0.32]  expected [0.6572832803080588] produced [0.9898777136093277]
Sample [1.63]  expected [0.9991239871888162] produced [0.8895710161964154]
Sample [0.08]  expected [0.5399573469845863] produced [0.9937838583242625]
Sample [5.61]  expected [0.18826022736065734] produced [0.1472854971827991]
Sample [0.98]  expected [0.9152486852459851] produced [0.963786232782877]
Sample [5.77]  expected [0.2545226375186995] produced [0.1402755153854894]
Sample [1.08]  expected [0.9409789034424738] produced [0.9568505435501463]
Sample [1.21]  expected [0.9678080007766929] produced [0.9456604951487819]
Sample [1.38]  expected [0.99092676518618] produced [0.9274860414786639]
Sample [3.97]  expected [0.13157220381780832] produced [0.3110734972355719]
Sample [0.96]  expected [0.9095957841504991] produced [0.9643616676303759]
Sample [2.96]  expected [0.5902981339471165] produced [0.5345486989223915]
Sample [1.85]  expected [0.9806376014876499] produced [0.8478797276647148]
Sample [3.42]  expected [0.36258766483843796] produced [0.42207971035268704]
Sample [3.91]  expected [0.15250451339176402] produced [0.31430763906927506]
Sample [5.72]  expected [0.23305886680417787] produced [0.13742983961190705]
Sample [1.36]  expected [0.9889323012176582] produced [0.9264258303312138]
Sample [0.04]  expected [0.5199946670933171] produced [0.9942556717161782]
Epoch 7000 RMSE = 0.19873991627042603
Epoch 7100 RMSE = 0.1986532820910864
Epoch 7200 RMSE = 0.19856742195926433
Epoch 7300 RMSE = 0.1984823352139853
Epoch 7400 RMSE = 0.1983980211367286
Epoch 7500 RMSE = 0.19831447890856566
Epoch 7600 RMSE = 0.1982317075740712
Epoch 7700 RMSE = 0.19814970601133794
Epoch 7800 RMSE = 0.1980684729074673
Epoch 7900 RMSE = 0.19798800673896458
Sample [5.38]  expected [0.10734852445560966] produced [0.15369015256421298]
Sample [5.31]  expected [0.08665891088398214] produced [0.15698948848760994]
Sample [6.24]  expected [0.47841405739563564] produced [0.11945407373472568]
Sample [2.13]  expected [0.9238389200667849] produced [0.7885947640063891]
Sample [4.54]  expected [0.007411109248070258] produced [0.22385942426068817]
Sample [5.78]  expected [0.2588907641275342] produced [0.13540543788515352]
Sample [5.11]  expected [0.03900566122589194] produced [0.17084730465434386]
Sample [2.31]  expected [0.8695026390297355] produced [0.7336977998068922]
Sample [6.26]  expected [0.4884083850038103] produced [0.12139966861821648]
Sample [4.42]  expected [0.0212209962753645] produced [0.23996646641968009]
Sample [3.61]  expected [0.2742671239192884] produced [0.364188845707915]
Sample [0.94]  expected [0.9037790502025571] produced [0.9713510546897026]
Sample [4.88]  expected [0.007006936319164825] produced [0.18391255452555239]
Sample [0.95]  expected [0.9067077523946869] produced [0.9702453725648406]
Sample [3.71]  expected [0.23085474585499116] produced [0.33003152088864]
Sample [3.46]  expected [0.3434728204485149] produced [0.37701107419613283]
Sample [2.37]  expected [0.8486388691299689] produced [0.6977627458358373]
Sample [3.0]  expected [0.5705600040299336] produced [0.5086885454450425]
Sample [2.98]  expected [0.5804451574837279] produced [0.52059845165446]
Sample [2.87]  expected [0.634133025464809] produced [0.5600429160355795]
Sample [6.13]  expected [0.4237065456757194] produced [0.12496210860382775]
Sample [5.32]  expected [0.0894928766443765] produced [0.16175270357641466]
Sample [0.11]  expected [0.5548891504185874] produced [0.9953215146505351]
Sample [4.52]  expected [0.009224873454242222] produced [0.22553057733055598]
Sample [0.07]  expected [0.5349714236687664] produced [0.9957009605809591]
Sample [5.41]  expected [0.11681048666210786] produced [0.15172223857837858]
Sample [0.05]  expected [0.5249895846353392] produced [0.995887767961267]
Sample [0.71]  expected [0.8259168855107684] produced [0.9823932638461131]
Sample [4.86]  expected [0.005437369602815101] produced [0.18660785793571952]
Sample [2.62]  expected [0.7491308212059193] produced [0.629313646880357]
Sample [2.53]  expected [0.7870860741772862] produced [0.6678725767737977]
Sample [0.51]  expected [0.7440886234414538] produced [0.9887587188649559]
Sample [6.11]  expected [0.41383956214219486] produced [0.12626552904745025]
Sample [4.33]  expected [0.036112067677562254] produced [0.2518480128883923]
Sample [5.79]  expected [0.2632830014590325] produced [0.13582278429443367]
Sample [2.51]  expected [0.7952154590569563] produced [0.6794801829623026]
Sample [5.96]  expected [0.3412057169639826] produced [0.13289912078691873]
Sample [0.59]  expected [0.7781805114563919] produced [0.9868771847622969]
Sample [3.4]  expected [0.3722294489865844] produced [0.4309511061963381]
Sample [2.52]  expected [0.7911653247620409] produced [0.6843527506022307]
Sample [5.22]  expected [0.06304586903548881] produced [0.1721611865386217]
Sample [5.48]  expected [0.14021416358974625] produced [0.15520248457912178]
Sample [4.81]  expected [0.0023800871154187053] produced [0.2014700523525816]
Sample [5.49]  expected [0.1437041857600192] produced [0.15123107235081368]
Sample [0.32]  expected [0.6572832803080588] produced [0.9925595254177279]
Sample [1.63]  expected [0.9991239871888162] produced [0.9003377145356461]
Sample [0.08]  expected [0.5399573469845863] produced [0.9956208249590163]
Sample [5.61]  expected [0.18826022736065734] produced [0.14616749831679057]
Sample [0.98]  expected [0.9152486852459851] produced [0.9702078395560447]
Sample [5.77]  expected [0.2545226375186995] produced [0.13943507282022286]
Sample [1.08]  expected [0.9409789034424738] produced [0.9639314200724665]
Sample [1.21]  expected [0.9678080007766929] produced [0.9536587010726234]
Sample [1.38]  expected [0.99092676518618] produced [0.9365899569726364]
Sample [3.97]  expected [0.13157220381780832] produced [0.3066078872575889]
Sample [0.96]  expected [0.9095957841504991] produced [0.9707530788434263]
Sample [2.96]  expected [0.5902981339471165] produced [0.5362945320028935]
Sample [1.85]  expected [0.9806376014876499] produced [0.8589027042975707]
Sample [3.42]  expected [0.36258766483843796] produced [0.4191543603229825]
Sample [3.91]  expected [0.15250451339176402] produced [0.31029606848844343]
Sample [5.72]  expected [0.23305886680417787] produced [0.13689734194369754]
Sample [1.36]  expected [0.9889323012176582] produced [0.9357894008611151]
Sample [0.04]  expected [0.5199946670933171] produced [0.9959815047934087]
Epoch 8000 RMSE = 0.19790830575650833
Epoch 8100 RMSE = 0.19782936797360456
Epoch 8200 RMSE = 0.19775119115868073
Epoch 8300 RMSE = 0.19767377283020712
Epoch 8400 RMSE = 0.197597110254473
Epoch 8500 RMSE = 0.19752120044567786
Epoch 8600 RMSE = 0.19744604016802317
Epoch 8700 RMSE = 0.1973716259395294
Epoch 8800 RMSE = 0.1972979540373177
Epoch 8900 RMSE = 0.19722502050413404
Sample [5.38]  expected [0.10734852445560966] produced [0.15265796585593522]
Sample [5.31]  expected [0.08665891088398214] produced [0.15589336958250943]
Sample [6.24]  expected [0.47841405739563564] produced [0.11983069203087934]
Sample [2.13]  expected [0.9238389200667849] produced [0.7974261159487862]
Sample [4.54]  expected [0.007411109248070258] produced [0.22058912978052278]
Sample [5.78]  expected [0.2588907641275342] produced [0.13501612331507132]
Sample [5.11]  expected [0.03900566122589194] produced [0.1691639842406996]
Sample [2.31]  expected [0.8695026390297355] produced [0.7415121167465953]
Sample [6.26]  expected [0.4884083850038103] produced [0.12149001623355687]
Sample [4.42]  expected [0.0212209962753645] produced [0.2362229819011638]
Sample [3.61]  expected [0.2742671239192884] produced [0.3611641748033031]
Sample [0.94]  expected [0.9037790502025571] produced [0.9762558623880169]
Sample [4.88]  expected [0.007006936319164825] produced [0.18210746878532094]
Sample [0.95]  expected [0.9067077523946869] produced [0.9753076212058382]
Sample [3.71]  expected [0.23085474585499116] produced [0.32769547085990874]
Sample [3.46]  expected [0.3434728204485149] produced [0.3757068193616643]
Sample [2.37]  expected [0.8486388691299689] produced [0.7058219472757445]
Sample [3.0]  expected [0.5705600040299336] produced [0.5098635019533203]
Sample [2.98]  expected [0.5804451574837279] produced [0.5218775499698408]
Sample [2.87]  expected [0.634133025464809] produced [0.5623109028378461]
Sample [6.13]  expected [0.4237065456757194] produced [0.12499178717919124]
Sample [5.32]  expected [0.0894928766443765] produced [0.1602131458382649]
Sample [0.11]  expected [0.5548891504185874] produced [0.9966266152068084]
Sample [4.52]  expected [0.009224873454242222] produced [0.22254798652479874]
Sample [0.07]  expected [0.5349714236687664] produced [0.9969218506053185]
Sample [5.41]  expected [0.11681048666210786] produced [0.15089389095192401]
Sample [0.05]  expected [0.5249895846353392] produced [0.9970658613795866]
Sample [0.71]  expected [0.8259168855107684] produced [0.9859505608547267]
Sample [4.86]  expected [0.005437369602815101] produced [0.18491159319834496]
Sample [2.62]  expected [0.7491308212059193] produced [0.6349793401915715]
Sample [2.53]  expected [0.7870860741772862] produced [0.6740950647098121]
Sample [0.51]  expected [0.7440886234414538] produced [0.9913227813269566]
Sample [6.11]  expected [0.41383956214219486] produced [0.12625004844724066]
Sample [4.33]  expected [0.036112067677562254] produced [0.24839081259957238]
Sample [5.79]  expected [0.2632830014590325] produced [0.13552451709746852]
Sample [2.51]  expected [0.7952154590569563] produced [0.6856448971189231]
Sample [5.96]  expected [0.3412057169639826] produced [0.13249454743432562]
Sample [0.59]  expected [0.7781805114563919] produced [0.989731785168378]
Sample [3.4]  expected [0.3722294489865844] produced [0.4283519483151979]
Sample [2.52]  expected [0.7911653247620409] produced [0.6900438398315039]
Sample [5.22]  expected [0.06304586903548881] produced [0.17009592511006677]
Sample [5.48]  expected [0.14021416358974625] produced [0.15386560176736644]
Sample [4.81]  expected [0.0023800871154187053] produced [0.19880820110254327]
Sample [5.49]  expected [0.1437041857600192] produced [0.15030254748676364]
Sample [0.32]  expected [0.6572832803080588] produced [0.9944399953693269]
Sample [1.63]  expected [0.9991239871888162] produced [0.909366904323605]
Sample [0.08]  expected [0.5399573469845863] produced [0.9968591020246156]
Sample [5.61]  expected [0.18826022736065734] produced [0.1453138216546499]
Sample [0.98]  expected [0.9152486852459851] produced [0.9751949837518477]
Sample [5.77]  expected [0.2545226375186995] produced [0.13882934685312293]
Sample [1.08]  expected [0.9409789034424738] produced [0.9695090706038345]
Sample [1.21]  expected [0.9678080007766929] produced [0.9600712995885288]
Sample [1.38]  expected [0.99092676518618] produced [0.9440499725852095]
Sample [3.97]  expected [0.13157220381780832] produced [0.3027067047494646]
Sample [0.96]  expected [0.9095957841504991] produced [0.9757110237783099]
Sample [2.96]  expected [0.5902981339471165] produced [0.5377000628596496]
Sample [1.85]  expected [0.9806376014876499] produced [0.8683759449010264]
Sample [3.42]  expected [0.36258766483843796] produced [0.41649700476126456]
Sample [3.91]  expected [0.15250451339176402] produced [0.30679841643979666]
Sample [5.72]  expected [0.23305886680417787] produced [0.1365654544084172]
Sample [1.36]  expected [0.9889323012176582] produced [0.9434634747058153]
Sample [0.04]  expected [0.5199946670933171] produced [0.9971373895192756]
Epoch 9000 RMSE = 0.19715282115590171
Epoch 9100 RMSE = 0.19708135159012588
Epoch 9200 RMSE = 0.1970106071949763
Epoch 9300 RMSE = 0.19694058315890828
Epoch 9400 RMSE = 0.19687127448068537
Epoch 9500 RMSE = 0.1968026759796932
Epoch 9600 RMSE = 0.19673478230643815
Epoch 9700 RMSE = 0.19666758795314532
Epoch 9800 RMSE = 0.19660108726437514
Epoch 9900 RMSE = 0.19653527444759297
Sample [5.38]  expected [0.10734852445560966] produced [0.1518688990960006]
Sample [5.31]  expected [0.08665891088398214] produced [0.1550411011606534]
Sample [6.24]  expected [0.47841405739563564] produced [0.1202552805364669]
Sample [2.13]  expected [0.9238389200667849] produced [0.8052055359989185]
Sample [4.54]  expected [0.007411109248070258] produced [0.2178427013871976]
Sample [5.78]  expected [0.2588907641275342] produced [0.13478330264157704]
Sample [5.11]  expected [0.03900566122589194] produced [0.16780344415860368]
Sample [2.31]  expected [0.8695026390297355] produced [0.748400329778064]
Sample [6.26]  expected [0.4884083850038103] produced [0.12168238397902383]
Sample [4.42]  expected [0.0212209962753645] produced [0.23304788105105836]
Sample [3.61]  expected [0.2742671239192884] produced [0.3584276954818419]
Sample [0.94]  expected [0.9037790502025571] produced [0.9800916744277802]
Sample [4.88]  expected [0.007006936319164825] produced [0.1806085452863291]
Sample [0.95]  expected [0.9067077523946869] produced [0.9792709208863154]
Sample [3.71]  expected [0.23085474585499116] produced [0.32556836861070154]
Sample [3.46]  expected [0.3434728204485149] produced [0.37445326821967906]
Sample [2.37]  expected [0.8486388691299689] produced [0.7129334356728774]
Sample [3.0]  expected [0.5705600040299336] produced [0.5107910549583173]
Sample [2.98]  expected [0.5804451574837279] produced [0.5229071362697475]
Sample [2.87]  expected [0.634133025464809] produced [0.5642364747081349]
Sample [6.13]  expected [0.4237065456757194] produced [0.12512498436942487]
Sample [5.32]  expected [0.0894928766443765] produced [0.1589984119989898]
Sample [0.11]  expected [0.5548891504185874] produced [0.9975246593386446]
Sample [4.52]  expected [0.009224873454242222] produced [0.22001462908635547]
Sample [0.07]  expected [0.5349714236687664] produced [0.9977564226694698]
Sample [5.41]  expected [0.11681048666210786] produced [0.15025526662589853]
Sample [0.05]  expected [0.5249895846353392] produced [0.9978685321466596]
Sample [0.71]  expected [0.8259168855107684] produced [0.9886382840923843]
Sample [4.86]  expected [0.005437369602815101] produced [0.18348373188939296]
Sample [2.62]  expected [0.7491308212059193] produced [0.6399132050730673]
Sample [2.53]  expected [0.7870860741772862] produced [0.6795326824541169]
Sample [0.51]  expected [0.7440886234414538] produced [0.9932018926266982]
Sample [6.11]  expected [0.41383956214219486] produced [0.12632976029933154]
Sample [4.33]  expected [0.036112067677562254] produced [0.24539388587424762]
Sample [5.79]  expected [0.2632830014590325] produced [0.1353460868456814]
Sample [2.51]  expected [0.7952154590569563] produced [0.6910277272655445]
Sample [5.96]  expected [0.3412057169639826] produced [0.13224723837697921]
Sample [0.59]  expected [0.7781805114563919] produced [0.9918499534215575]
Sample [3.4]  expected [0.3722294489865844] produced [0.4259215756409371]
Sample [2.52]  expected [0.7911653247620409] produced [0.6950209876399921]
Sample [5.22]  expected [0.06304586903548881] produced [0.1684140520903344]
Sample [5.48]  expected [0.14021416358974625] produced [0.15280612288667703]
Sample [4.81]  expected [0.0023800871154187053] produced [0.19655334661750726]
Sample [5.49]  expected [0.1437041857600192] produced [0.14956687849453945]
Sample [0.32]  expected [0.6572832803080588] produced [0.9957773390494754]
Sample [1.63]  expected [0.9991239871888162] produced [0.9169923761122193]
Sample [0.08]  expected [0.5399573469845863] produced [0.9977068606554049]
Sample [5.61]  expected [0.18826022736065734] produced [0.1446594010479123]
Sample [0.98]  expected [0.9152486852459851] produced [0.9791074249680717]
Sample [5.77]  expected [0.2545226375186995] produced [0.1383956134640658]
Sample [1.08]  expected [0.9409789034424738] produced [0.9739442771649648]
Sample [1.21]  expected [0.9678080007766929] produced [0.9652568570327018]
Sample [1.38]  expected [0.99092676518618] produced [0.9502086962407365]
Sample [3.97]  expected [0.13157220381780832] produced [0.2992797968403968]
Sample [0.96]  expected [0.9095957841504991] produced [0.9795947702784547]
Sample [2.96]  expected [0.5902981339471165] produced [0.5388334259158212]
Sample [1.85]  expected [0.9806376014876499] produced [0.8765586208284827]
Sample [3.42]  expected [0.36258766483843796] produced [0.41407990605997314]
Sample [3.91]  expected [0.15250451339176402] produced [0.30372991335004973]
Sample [5.72]  expected [0.23305886680417787] produced [0.13637252159177732]
Sample [1.36]  expected [0.9889323012176582] produced [0.9497975767276721]
Sample [0.04]  expected [0.5199946670933171] produced [0.9979237742314229]
Epoch 10000 RMSE = 0.19647014358363318
Final Epoch RMSE = 0.19647014358363318
TESTING
Sample [5.67]  expected [0.21226230990239148] produced [0.13901498573375734]
Sample [3.75]  expected [0.21421934062882808] produced [0.3288682163894578]
Sample [5.8]  expected [0.2676989102931213] produced [0.13253504538693037]
Sample [4.78]  expected [0.0011423772195533477] produced [0.1888800035620935]
Sample [2.6]  expected [0.7577506859107321] produced [0.6439259239926203]
Sample [2.56]  expected [0.7746777182135633] produced [0.6670359859085606]
Sample [3.63]  expected [0.26538997935563646] produced [0.3592396989783648]
Sample [4.82]  expected [0.0028922402253643287] produced [0.1873789836988992]
Sample [5.62]  expected [0.19218494737495684] produced [0.13905402198779537]
Sample [4.59]  expected [0.0037400935400184188] produced [0.20464256182474838]
Sample [5.16]  expected [0.04925817201682253] produced [0.15871872372117193]
Sample [3.86]  expected [0.17090676494754753] produced [0.2928194558287243]
Sample [2.5]  expected [0.7992360720519782] produced [0.6645786612745591]
Sample [1.72]  expected [0.9944448830023507] produced [0.8934966514968391]
Sample [2.14]  expected [0.9211652158183228] produced [0.7929342672612179]
Sample [4.47]  expected [0.014616331670855853] produced [0.21775143081286627]
Sample [5.95]  expected [0.3364725925651297] produced [0.1259355030453596]
Sample [3.32]  expected [0.4112687875795697] produced [0.418660525177821]
Sample [4.1]  expected [0.09086144446779487] produced [0.26006145821315085]
Sample [0.18]  expected [0.5895147867129121] produced [0.9969802476412022]
Sample [4.28]  expected [0.04601636969179729] produced [0.23018418065232857]
Sample [2.64]  expected [0.7404113074943242] produced [0.6125138491410452]
Sample [4.25]  expected [0.05250532088570825] produced [0.2349281250775598]
Sample [0.17]  expected [0.584591174533498] produced [0.9970431735641868]
Sample [0.12]  expected [0.5598561036444597] produced [0.9974096828184641]
Sample [6.06]  expected [0.3893314780608207] produced [0.11997977777347187]
Sample [3.21]  expected [0.4658229969394761] produced [0.435341737545586]
Sample [4.87]  expected [0.006197463039892326] produced [0.17489798626680544]
Sample [0.88]  expected [0.8853694394494847] produced [0.9818234506543047]
Sample [5.81]  expected [0.27213804904259764] produced [0.1274187838726669]
Sample [1.89]  expected [0.9747428074323152] produced [0.8524963328343951]
Sample [1.65]  expected [0.9984325142269594] produced [0.9051304534088016]
Sample [0.87]  expected [0.8821644685127525] produced [0.9826894442497777]
Sample [1.02]  expected [0.9260540109746814] produced [0.9752610604483842]
Sample [3.38]  expected [0.38192233965155153] produced [0.39613630691628826]
Sample [1.26]  expected [0.9760451707952578] produced [0.9572377494967009]
Sample [3.17]  expected [0.4857982370581981] produced [0.4523885815496209]
Sample [0.5]  expected [0.7397127693021015] produced [0.9931194643492367]
Sample [3.87]  expected [0.1671590074769404] produced [0.2932577720503483]
Sample [0.15]  expected [0.5747190662367996] produced [0.9972034060784251]
Sample [1.55]  expected [0.9998918820946785] produced [0.9204899416879441]
Sample [5.9]  expected [0.31306166758488196] produced [0.12576538486187494]
Sample [0.41]  expected [0.6993046639922115] produced [0.9945140946792763]
Sample [4.93]  expected [0.011791994854675136] produced [0.17169635028589486]
Sample [1.76]  expected [0.9910771585688092] produced [0.88214545439797]
Sample [1.7]  expected [0.9958324052262344] produced [0.8955966129698472]
Sample [5.35]  expected [0.09823992207392229] produced [0.14809128996984253]
Sample [1.61]  expected [0.9996158172106953] produced [0.9123184181690461]
Sample [5.91]  expected [0.3177083292878493] produced [0.12668208132597075]
Sample [1.16]  expected [0.9584015543858835] produced [0.9663356451880996]
Sample [3.82]  expected [0.18622308846035313] produced [0.30441027942767046]
Sample [2.1]  expected [0.9316046833244369] produced [0.7987969151463357]
Sample [0.34]  expected [0.6667435460704072] produced [0.9954499412841068]
Sample [5.97]  expected [0.34595472065881094] produced [0.12555280548433742]
Sample [0.82]  expected [0.8655729148634479] produced [0.9849296802979591]
Sample [5.01]  expected [0.021980122864440954] produced [0.1699173625280873]
Sample [2.38]  expected [0.8450374917784682] produced [0.7163275565949273]
Sample [3.29]  expected [0.4260684130978407] produced [0.4324210059740962]
Sample [5.52]  expected [0.1543866142014368] produced [0.14380417607143703]
Sample [0.47]  expected [0.7264431426895341] produced [0.9937214468182778]
Sample [0.03]  expected [0.5149977501012478] produced [0.9979717956373834]
Sample [3.49]  expected [0.3292993610615895] produced [0.38050261815477554]
Sample [1.58]  expected [0.99997882324937] produced [0.9197500358598086]
Sample [6.21]  expected [0.46344000324936846] produced [0.12012909090193191]
Sample [4.38]  expected [0.02736724390596834] produced [0.22940900103580386]
Sample [4.53]  expected [0.008293406226344613] produced [0.2077124948898551]
Sample [5.07]  expected [0.031632135787960536] produced [0.16203023096110317]
Sample [5.13]  expected [0.04296976724604645] produced [0.1571924102949621]
Sample [3.23]  expected [0.4558538859086962] produced [0.4258066367612173]
Sample [3.98]  expected [0.12821043052786274] produced [0.26820079901416816]
Sample [5.5]  expected [0.14722983721480404] produced [0.13698242520792756]
Sample [2.69]  expected [0.7181995410800631] produced [0.591653452470562]
Sample [4.11]  expected [0.08800782939418711] produced [0.25115553154642345]
Sample [1.81]  expected [0.9857634779111577] produced [0.8682264187569635]
Sample [5.39]  expected [0.11046366368429839] produced [0.14339998288075162]
Sample [0.49]  expected [0.735312944085579] produced [0.9931696205857821]
Sample [3.02]  expected [0.5606466275153149] produced [0.487971864809309]
Sample [4.45]  expected [0.017113469689680583] produced [0.2108932935000411]
Sample [4.04]  expected [0.10883204638667365] produced [0.2561749849020748]
Sample [5.04]  expected [0.026593112203695535] produced [0.15773645407913817]
Sample [4.89]  expected [0.007865708493979218] produced [0.1656481459469487]
Sample [0.79]  expected [0.8551766362088039] produced [0.9847748415813868]
Sample [3.36]  expected [0.3916624598063102] produced [0.37386731188528294]
Sample [1.71]  expected [0.9951634020780791] produced [0.8842113801582231]
Sample [1.0]  expected [0.9207354924039483] produced [0.9747999923138545]
Sample [4.49]  expected [0.012313340647666704] produced [0.19667659671052704]
Sample [1.59]  expected [0.9999078075671455] produced [0.9064322589773918]
Sample [0.78]  expected [0.8516397096002051] produced [0.9851393581323278]
Sample [4.18]  expected [0.06920154984462978] produced [0.22704214397981032]
Sample [2.21]  expected [0.9012855331233737] produced [0.739643567451092]
Sample [1.88]  expected [0.9762880971357977] produced [0.8453650455494072]
Sample [1.68]  expected [0.997021601099038] produced [0.8932189471461252]
Sample [0.06]  expected [0.5299820032397223] produced [0.9977810041103602]
Sample [1.91]  expected [0.9715099656450052] produced [0.8424496713371094]
Sample [0.81]  expected [0.8621435871850712] produced [0.9846125716787123]
Sample [1.25]  expected [0.9744923096777931] produced [0.9568004563085951]
Sample [3.43]  expected [0.35778714373176873] produced [0.3728722570366434]
Sample [0.2]  expected [0.5993346653975306] produced [0.9967876508581743]
Sample [0.7]  expected [0.8221088436188455] produced [0.9882629597606094]
Sample [0.16]  expected [0.579659103307123] produced [0.9971095265930358]
Sample [4.01]  expected [0.11834950861646326] produced [0.25987441477555095]
Sample [6.04]  expected [0.3796022874329204] produced [0.11810098981384007]
Sample [3.6]  expected [0.2787397783525738] produced [0.3297400131343269]
Sample [5.99]  expected [0.35549846481031944] produced [0.11958126579486854]
Sample [5.28]  expected [0.07840612907179162] produced [0.1460906615374345]
Sample [0.44]  expected [0.7129697325329998] produced [0.9939234826260308]
Sample [1.57]  expected [0.9999998414659172] produced [0.9138469101484675]
Sample [1.84]  expected [0.9819914980762241] produced [0.8599639315514734]
Sample [1.12]  expected [0.9500502210882525] produced [0.9677597270894013]
Sample [3.06]  expected [0.5407510758801346] produced [0.47399719592295514]
Sample [1.49]  expected [0.9983688760215716] produced [0.9295900980131458]
Sample [1.27]  expected [0.9775504277923461] produced [0.9558405021595178]
Sample [5.3]  expected [0.08386627888804937] produced [0.14878772415898933]
Sample [1.33]  expected [0.9855741889605223] produced [0.9495026961407963]
Sample [4.0]  expected [0.1215987523460359] produced [0.2665586205939684]
Sample [3.18]  expected [0.4808010477473823] produced [0.4363505252382095]
Sample [5.89]  expected [0.30843369955937433] produced [0.1244502247213921]
Sample [3.93]  expected [0.14538430489930698] produced [0.27659797905346173]
Sample [6.19]  expected [0.45347474836865553] produced [0.11569041044120756]
Sample [5.59]  expected [0.18050469783588813] produced [0.13481030032390828]
Sample [4.02]  expected [0.11513842961798781] produced [0.2612872218453365]
Sample [4.27]  expected [0.04813423932484717] produced [0.22328175146641027]
Sample [4.24]  expected [0.05475809570900575] produced [0.22226228578299603]
Sample [4.17]  expected [0.07176101291749942] produced [0.22632668519709698]
Sample [2.34]  expected [0.8592323965345632] produced [0.6919708242210546]
Sample [5.54]  expected [0.1616816319342716] produced [0.13220843830699963]
Sample [0.4]  expected [0.6947091711543253] produced [0.9944682261685052]
Sample [5.45]  expected [0.12996134475555282] produced [0.13577028400653723]
Sample [2.36]  expected [0.8522053828850881] produced [0.696301401199643]
Sample [2.79]  expected [0.672196733629195] produced [0.5573810238121948]
Sample [4.34]  expected [0.034269603123378733] produced [0.22152886296258414]
Sample [2.32]  expected [0.8661157220151258] produced [0.7213865428804532]
Sample [0.53]  expected [0.7527666706024234] produced [0.9924920804250622]
Sample [4.26]  expected [0.050297295157411104] produced [0.2316997078839116]
Sample [0.38]  expected [0.6854602347064913] produced [0.9948374205016698]
Sample [5.93]  expected [0.3270558732540856] produced [0.12248006287374555]
Sample [3.27]  expected [0.4359726178668102] produced [0.4152913754240799]
Sample [4.72]  expected [1.4481834987756237e-05] produced [0.18394677266239567]
Sample [3.04]  expected [0.5507089931583009] produced [0.47694884248244723]
Sample [1.66]  expected [0.9980119949582684] produced [0.9012444896479418]
Sample [0.64]  expected [0.798597720681196] produced [0.9900457762646698]
Sample [5.44]  expected [0.12661729356609386] produced [0.14165099056770608]
Sample [3.74]  expected [0.21833635794981499] produced [0.31042357978242147]
Sample [5.21]  expected [0.06063730263590511] produced [0.15062485077974758]
Sample [4.23]  expected [0.05705539435169865] produced [0.22910392926952672]
Sample [0.39]  expected [0.6900942075615807] produced [0.9946335680489544]
Sample [2.61]  expected [0.7534534261240267] produced [0.612166004477586]
Sample [5.69]  expected [0.22049750135987561] produced [0.1303942451643583]
Sample [3.85]  expected [0.17468743146741633] produced [0.2878693475862931]
Sample [4.22]  expected [0.05939698708583735] produced [0.2292300859738759]
Sample [5.0]  expected [0.020537862668430773] produced [0.1581220078610946]
Sample [1.52]  expected [0.9993550719877915] produced [0.9193236144491308]
Sample [1.17]  expected [0.9603752988680678] produced [0.9622012429520598]
Sample [3.88]  expected [0.16344453382821916] produced [0.2704396036617128]
Sample [1.45]  expected [0.9963564955187942] produced [0.9290293224431837]
Sample [3.73]  expected [0.22248154140028809] produced [0.29260453592470526]
Sample [2.22]  expected [0.8982827361180432] produced [0.7370204012636947]
Sample [2.68]  expected [0.7226873222709356] produced [0.5856866877638217]
Sample [3.52]  expected [0.31527952072776144] produced [0.35003158710949633]
Sample [0.92]  expected [0.897800810018183] produced [0.9796387198470491]
Sample [5.43]  expected [0.12331058033612674] produced [0.13899975314958193]
Sample [0.89]  expected [0.8885358737634119] produced [0.9810346723335767]
Sample [5.56]  expected [0.16911197250348153] produced [0.13360393262703746]
Sample [1.83]  expected [0.9832971959166488] produced [0.862552980780128]
Sample [0.84]  expected [0.8723215599854297] produced [0.9834215140811291]
Sample [2.47]  expected [0.8111167776596523] produced [0.6725958244789476]
Sample [0.56]  expected [0.7655930989604417] produced [0.9919304515332116]
Sample [4.96]  expected [0.015249650273095605] produced [0.16899278297095002]
Sample [1.46]  expected [0.9969341817058224] produced [0.9337701531925474]
Sample [3.3]  expected [0.42112715292837566] produced [0.4121974806652982]
Sample [5.65]  expected [0.2041422096844953] produced [0.13334397666185846]
Sample [5.83]  expected [0.28108423836842655] produced [0.12748697566046396]
Sample [3.64]  expected [0.26098637693232857] produced [0.3358709349417135]
Sample [4.4]  expected [0.02419896305524194] produced [0.2149019515558231]
Sample [3.11]  expected [0.5157936992182269] produced [0.45685569384905145]
Sample [2.39]  expected [0.8414016109653198] produced [0.7029548966864096]
Sample [0.52]  expected [0.7484400689218684] produced [0.9927380687167265]
Sample [5.25]  expected [0.07053275328670394] produced [0.1526718972947821]
Sample [5.09]  expected [0.03522594679473745] produced [0.16085638676880287]
Sample [2.25]  expected [0.8890365984439607] produced [0.7527120352448746]
Sample [1.32]  expected [0.9843575500591326] produced [0.9522629341827066]
Sample [5.1]  expected [0.03709265883613394] produced [0.16228941277555745]
Sample [5.68]  expected [0.21636572403551568] produced [0.1334556978062704]
Sample [1.78]  expected [0.9890983034040224] produced [0.8822190456159819]
Sample [0.74]  expected [0.8371439558140725] produced [0.9875397607041426]
Sample [2.94]  expected [0.6001149923608853] produced [0.5313436202441184]
Sample [4.92]  expected [0.010736935029430739] produced [0.1771602543731476]
Sample [2.97]  expected [0.5853759144755727] produced [0.5220870641691292]
Sample [6.02]  expected [0.36992125428476597] produced [0.12477965443564223]
Sample [2.73]  expected [0.7000347387962098] produced [0.6125780251243355]
Sample [1.79]  expected [0.988035460991262] produced [0.8891176269945145]
Sample [0.62]  expected [0.7905175802686526] produced [0.9910939540410046]
Sample [3.83]  expected [0.1823459761478622] produced [0.32152937098739204]
Sample [5.76]  expected [0.25017905844154875] produced [0.1347103246141619]
Sample [4.94]  expected [0.012895875073595464] produced [0.17896996318405156]
Sample [2.45]  expected [0.8188823510672518] produced [0.7039759191908529]
Sample [2.49]  expected [0.8032267616891573] produced [0.6991579481245253]
Sample [1.97]  expected [0.9606854030956977] produced [0.8540383314613039]
Sample [2.75]  expected [0.6908304960261659] produced [0.626281914665594]
Sample [4.44]  expected [0.018434534713341666] produced [0.23808542608461916]
Sample [6.23]  expected [0.4734198816413219] produced [0.12253519016186921]
Sample [5.03]  expected [0.025007899019619584] produced [0.179394123241811]
Sample [1.42]  expected [0.9943258814258598] produced [0.9460419356156066]
Sample [0.3]  expected [0.6477601033306698] produced [0.9959852624870804]
Sample [0.83]  expected [0.8689656855549814] produced [0.9853539820941267]
Sample [0.13]  expected [0.5648170713098475] produced [0.9973812479195832]
Sample [5.12]  expected [0.04096476266536658] produced [0.17084670131066176]
Sample [4.73]  expected [7.753499899781913e-05] produced [0.19878869246111575]
Sample [5.88]  expected [0.3038248880042731] produced [0.1303888371489634]
Sample [2.99]  expected [0.5755063560431719] produced [0.5317553666432178]
Sample [6.05]  expected [0.3844611058503039] produced [0.12678166467269542]
Sample [6.09]  expected [0.4040070416350225] produced [0.12681093886018407]
Sample [0.72]  expected [0.8296923359857366] produced [0.9888387722487871]
Sample [5.14]  expected [0.0450204744691447] produced [0.17162179991669413]
Sample [1.47]  expected [0.9974621748887904] produced [0.9404582626865321]
Sample [4.03]  expected [0.11196583645583386] produced [0.28942446696324214]
Sample [2.05]  expected [0.9436811843166877] produced [0.828191671144506]
Sample [5.98]  expected [0.3507191287532032] produced [0.129292198744784]
Sample [1.73]  expected [0.9936769198503582] produced [0.9024076769979826]
Sample [1.15]  expected [0.9563819701302605] produced [0.9698130500748603]
Sample [0.58]  expected [0.7740119683959368] produced [0.9920018996291949]
Sample [3.89]  expected [0.159763715445653] produced [0.3147942675372858]
Sample [5.6]  expected [0.18436668106383958] produced [0.14237292718235414]
Sample [2.12]  expected [0.9264702407764381] produced [0.8113164887648455]
Sample [1.29]  expected [0.9804175321030364] produced [0.9586496656339161]
Sample [6.14]  expected [0.42865172824087117] produced [0.1252308621835231]
Sample [4.3]  expected [0.04191703162527255] produced [0.2516800378146978]
Sample [0.45]  expected [0.7174827670556151] produced [0.994107792232482]
Sample [5.06]  expected [0.02990539620068583] produced [0.1719089186211449]
Sample [4.5]  expected [0.011234941167451495] produced [0.21714260738150093]
Sample [0.9]  expected [0.8916634548137417] produced [0.9817550478579469]
Sample [3.03]  expected [0.5556805943433248] produced [0.5047102216453636]
Sample [1.98]  expected [0.9587189776409049] produced [0.8415302626175202]
Sample [3.58]  expected [0.28775101525820873] produced [0.36553384526779364]
Sample [0.93]  expected [0.9008099704418886] produced [0.9805917811539326]
Sample [5.15]  expected [0.047116679265647776] produced [0.1618093878886839]
Sample [1.13]  expected [0.952206094689413] produced [0.9689049952567247]
Sample [2.03]  expected [0.9482028705757799] produced [0.8260601214191226]
Sample [1.92]  expected [0.9698227368426624] produced [0.8568443628123744]
Sample [2.91]  expected [0.6147639735106321] produced [0.5552852829295392]
Sample [1.96]  expected [0.9626057603940842] produced [0.8529015239548591]
Sample [4.75]  expected [0.0003536055123110615] produced [0.19875679420143869]
Sample [2.74]  expected [0.6954423894492261] produced [0.6138253644005457]
Sample [2.63]  expected [0.7447833034132998] produced [0.6582408380161462]
Sample [5.86]  expected [0.29466657585282957] produced [0.13575231028303897]
Sample [6.17]  expected [0.4435281029682663] produced [0.12644825664026266]
Sample [0.26]  expected [0.6285402759460775] produced [0.9964168368795528]
Sample [5.27]  expected [0.0757391572618979] produced [0.16742868104707756]
Sample [2.3]  expected [0.87285260608836] produced [0.774992963032302]
Sample [3.1]  expected [0.5207903312166452] produced [0.5282461405550052]
Sample [4.95]  expected [0.014048465299089596] produced [0.1911263598921311]
Sample [2.41]  expected [0.8340277967082455] produced [0.7408947116760062]
Sample [2.86]  expected [0.6389429629082934] produced [0.6030294390677373]
Sample [3.24]  expected [0.4508757031274457] produced [0.4888214632878171]
Sample [1.48]  expected [0.99794042226882] produced [0.9426370999179606]
Sample [3.55]  expected [0.30142591635701993] produced [0.40259059066028213]
Sample [3.12]  expected [0.510795487863048] produced [0.5115035375155628]
Sample [3.26]  expected [0.4409345720540911] produced [0.47003325184283845]
Sample [3.2]  expected [0.47081292828620996] produced [0.4844518543758474]
Sample [5.94]  expected [0.3317558207707479] produced [0.13285984040461749]
Sample [2.72]  expected [0.7046070848360086] produced [0.640181667270698]
Sample [5.64]  expected [0.2001263356029781] produced [0.14776972122388649]
Sample [5.75]  expected [0.24586046125037087] produced [0.14341342280986744]
Sample [3.81]  expected [0.19013157820251836] produced [0.3452834922281038]
Sample [4.13]  expected [0.08242447710745326] produced [0.2774767102501242]
Sample [2.33]  expected [0.8626921937334098] produced [0.7517246651245475]
Sample [4.8]  expected [0.0019176955820796593] produced [0.1958631389714518]
Sample [5.26]  expected [0.07311461118272833] produced [0.15988874829509786]
Sample [5.46]  expected [0.13334239950217175] produced [0.1485150898421368]
Sample [0.68]  expected [0.8143965120092342] produced [0.9895629680429195]
Sample [2.08]  expected [0.9365664897537582] produced [0.8209059939957972]
Sample [1.87]  expected [0.977785758426472] produced [0.8741375021627416]
Sample [6.08]  expected [0.3991049346219355] produced [0.12718948338503194]
Sample [0.8]  expected [0.8586780454497613] produced [0.986531335719583]
Sample [4.06]  expected [0.10268212512130137] produced [0.2877476800895151]
Sample [3.5]  expected [0.32460838615519005] produced [0.38986845895406624]
Sample [6.27]  expected [0.4934075374332394] produced [0.11925039553677692]
Sample [5.85]  expected [0.29011799108007064] produced [0.13339530282374124]
Sample [4.37]  expected [0.029022359045899504] produced [0.23792311928022747]
Sample [5.58]  expected [0.17667466387190828] produced [0.1418074960527866]
Sample [4.21]  expected [0.061782639754099256] produced [0.2518941542184451]
Sample [3.68]  expected [0.2436153463221381] produced [0.3335665738744343]
Sample [0.25]  expected [0.6237019796272615] produced [0.9963703935149945]
Sample [1.74]  expected [0.9928595894177767] produced [0.8893303112802778]
Sample [1.82]  expected [0.9845545644402282] produced [0.8741760144648449]
Sample [4.41]  expected [0.022686114169891858] produced [0.22105959055590524]
Sample [1.64]  expected [0.9988031906595868] produced [0.9072780010091348]
Sample [1.94]  expected [0.9663075070111002] produced [0.8449343196641684]
Sample [1.01]  expected [0.9234159223090076] produced [0.9762527394519602]
Sample [4.56]  expected [0.005794374030434701] produced [0.20448880840272304]
Sample [3.15]  expected [0.4957963763164257] produced [0.4570534329022205]
Sample [4.39]  expected [0.025759391647787178] produced [0.21931823611181225]
Sample [0.77]  expected [0.8480676193136784] produced [0.986160366690436]
Sample [2.35]  expected [0.8557366763954222] produced [0.7169977082753051]
Sample [2.58]  expected [0.7662674537778106] produced [0.6482592549465279]
Sample [5.24]  expected [0.06799384175746281] produced [0.15643609329983718]
Sample [3.99]  expected [0.12488583588504054] produced [0.2783308682329771]
Sample [3.34]  expected [0.40144591353266507] produced [0.4062048085082027]
Sample [3.25]  expected [0.4459024327349458] produced [0.4301133861845328]
Sample [1.9]  expected [0.9731500438437072] produced [0.8545365241409245]
Sample [1.18]  expected [0.9623030062040101] produced [0.9649585240296968]
Sample [3.37]  expected [0.3867867391130584] produced [0.40435142362577703]
Sample [2.23]  expected [0.8952401111710024] produced [0.7660472719845137]
Sample [2.19]  expected [0.907170446212898] produced [0.7844484599152722]
Sample [4.98]  expected [0.017797319149234825] produced [0.17587246619979643]
Sample [4.14]  expected [0.07969529822490273] produced [0.2593047471616326]
Sample [3.35]  expected [0.39654901416330013] produced [0.4080665526244544]
Sample [4.19]  expected [0.06668516625777782] produced [0.24476197469803368]
Sample [3.28]  expected [0.43101706636438636] produced [0.4162144980395288]
Sample [5.05]  expected [0.028225665682046708] produced [0.1613428150334341]
Sample [1.22]  expected [0.9695496781595339] produced [0.959944363449958]
Sample [3.62]  expected [0.2698170425855009] produced [0.33205042275509294]
Sample [3.09]  expected [0.5257848841992673] produced [0.46338301835154866]
Sample [2.06]  expected [0.941353675407987] produced [0.8077430909648201]
Sample [1.19]  expected [0.9641844836245832] produced [0.9634718934007641]
Sample [1.28]  expected [0.9790079301446124] produced [0.9554871883549588]
Sample [5.71]  expected [0.22884449009016516] produced [0.13154832672854816]
Sample [6.25]  expected [0.4834103917262216] produced [0.11613233447522875]
Sample [1.56]  expected [0.999970860114983] produced [0.9228457699214224]
Sample [2.76]  expected [0.6861995197125277] produced [0.5940929872738719]
Sample [1.06]  expected [0.9361777411724932] produced [0.9741312853008995]
Sample [2.81]  expected [0.66277466725878] produced [0.5858427248495245]
Sample [0.24]  expected [0.6188513132135672] produced [0.996524011203768]
Sample [2.0]  expected [0.9546487134128409] produced [0.8429166681729869]
Sample [0.73]  expected [0.8334348175018489] produced [0.9884038165059494]
Sample [3.72]  expected [0.2266544764653564] produced [0.34451847077288655]
Sample [1.4]  expected [0.9927248649942302] produced [0.9463285205479012]
Sample [4.91]  expected [0.009730801102965536] produced [0.18101149966844846]
Sample [2.57]  expected [0.7704861101884943] produced [0.6639575814813173]
Sample [6.0]  expected [0.36029225090053707] produced [0.12745185701816264]
Sample [5.37]  expected [0.10427265004726693] produced [0.1547139913162443]
Sample [3.56]  expected [0.2968471489277916] produced [0.37753181817079456]
Sample [6.1]  expected [0.4089187478639525] produced [0.12372015961376384]
Sample [4.85]  expected [0.00472673201664342] produced [0.18769625888013797]
Sample [0.23]  expected [0.6139887617675942] produced [0.9965946256442245]
Sample [3.77]  expected [0.20607144831075863] produced [0.32420170043006247]
Sample [0.1]  expected [0.5499167083234141] produced [0.9975489015434735]
Sample [1.37]  expected [0.9899540306993071] produced [0.9477146698623493]
Sample [2.43]  expected [0.8265203757861324] produced [0.7063784794045787]
Sample [4.31]  expected [0.039935973122187995] produced [0.23908917317681713]
Sample [4.35]  expected [0.0324737112207753] produced [0.22769035923010475]
Sample [2.28]  expected [0.879440354090461] produced [0.7457287146533601]
Sample [3.76]  expected [0.21013090113562843] produced [0.3177465963904538]
Sample [1.67]  expected [0.99754167490509] produced [0.9025170662753277]
Sample [4.97]  expected [0.016499309878116986] produced [0.16932121305374978]
Sample [4.68]  expected [0.0002622385863579879] produced [0.18909484725672596]
Sample [4.61]  expected [0.0026185869626622615] produced [0.1917644301491431]
Sample [3.31]  expected [0.41619377997789087] produced [0.39788440578766615]
Sample [4.29]  expected [0.04394389804345983] produced [0.2210537817625807]
Sample [3.66]  expected [0.2522513135415775] produced [0.31173073171079596]
Sample [2.48]  expected [0.8071871289028558] produced [0.6544712573397101]
Sample [3.19]  expected [0.4758057783157929] produced [0.4321900122833129]
Sample [4.57]  expected [0.005060097463247981] produced [0.19425032903315245]
Sample [5.2]  expected [0.05827267213992343] produced [0.14803938738214756]
Sample [4.7]  expected [3.837121794958431e-05] produced [0.17890866484972165]
Sample [3.8]  expected [0.19407105452864037] produced [0.2827799329775571]
Sample [4.07]  expected [0.09966660891209123] produced [0.23804279459233216]
Sample [3.05]  expected [0.5457323211162184] produced [0.44858879912668415]
Sample [1.2]  expected [0.9660195429836131] produced [0.9589805537584841]
Sample [1.3]  expected [0.9817790927085965] produced [0.9487530835379251]
Sample [2.18]  expected [0.9100519738106871] produced [0.7545662558657114]
Sample [3.47]  expected [0.3387320498387605] produced [0.3543648780920437]
Sample [4.69]  expected [0.00012531137600302333] produced [0.1791830490344545]
Sample [5.82]  expected [0.27659997379728496] produced [0.12121243486114569]
Sample [0.86]  expected [0.8789212814476385] produced [0.9819718681019658]
Sample [2.89]  expected [0.6244733933365763] produced [0.5131622627752589]
Sample [0.29]  expected [0.6429761125524178] produced [0.9958881800985651]
Sample [0.02]  expected [0.5099993333466666] produced [0.9979892056324877]
Sample [4.55]  expected [0.006578070748381737] produced [0.1934860565345671]
Sample [6.18]  expected [0.4484988506324511] produced [0.11270560331228921]
Sample [6.03]  expected [0.37475550868646257] produced [0.11731298975046832]
Sample [3.01]  expected [0.5656065960750919] produced [0.48604682445875064]
Sample [4.36]  expected [0.030724571557446068] produced [0.2169522392455631]
Sample [5.29]  expected [0.08111525991745111] produced [0.14421567849754396]
Sample [3.65]  expected [0.2566066756721503] produced [0.3178540638944159]
Sample [3.45]  expected [0.34822924364578534] produced [0.357757433574303]
Sample [6.15]  expected [0.43360404557374166] produced [0.11366750771253016]
Sample [1.34]  expected [0.9867422708476596] produced [0.9463606270254354]
Sample [4.12]  expected [0.08519541319431456] produced [0.24043530551579936]
Sample [0.6]  expected [0.7823212366975176] produced [0.9906486334424854]
Sample [2.04]  expected [0.9459643254766897] produced [0.8015307083406857]
Sample [2.71]  expected [0.7091589703378295] produced [0.5825376740894683]
Sample [0.14]  expected [0.5697715573221183] produced [0.99725059685204]
Sample [2.16]  expected [0.9156917303893416] produced [0.7788081192802467]
Sample [5.34]  expected [0.09528367177669034] produced [0.14714146406758916]
Sample [3.44]  expected [0.3530008437922162] produced [0.38068663200348635]
Sample [2.42]  expected [0.8302906006396004] produced [0.6988801570203481]
Sample [0.67]  expected [0.8104929935182799] produced [0.98951955418528]
Sample [5.47]  expected [0.1367601197032935] produced [0.1428946443355874]
Sample [5.42]  expected [0.1200415357342195] produced [0.14514119368659334]
Sample [3.95]  expected [0.13840593795674394] produced [0.2824781146618909]
Sample [0.36]  expected [0.6761371166375449] produced [0.9951380878873003]
Sample [2.83]  expected [0.6532874931917615] produced [0.5580641176360813]
Sample [1.53]  expected [0.9995839726357381] produced [0.9269297744381949]
Sample [0.31]  expected [0.6525293182217218] produced [0.9957702934551401]
Sample [0.01]  expected [0.5049999166670833] produced [0.9980503071395652]
Sample [3.96]  expected [0.13497081958035023] produced [0.28132799198341846]
Sample [0.76]  expected [0.8444607225552756] produced [0.9866528109509995]
Sample [1.51]  expected [0.9990762362487741] produced [0.9277790141537781]
Sample [3.07]  expected [0.5357657555704216] produced [0.48258736315741096]
Sample [3.33]  expected [0.40635266822854843] produced [0.4128954512274656]
Sample [2.54]  expected [0.7829781152243513] produced [0.663647376005244]
Sample [4.83]  expected [0.003454103697032307] produced [0.18342693720802103]
Sample [3.92]  expected [0.1489268556345973] produced [0.28735721954211474]
Sample [6.07]  expected [0.39421291703130745] produced [0.11928941126379047]
Sample [6.2]  expected [0.4584552985912518] produced [0.11692065266562723]
Sample [0.91]  expected [0.8947518698449752] produced [0.9812553552162004]
Sample [2.85]  expected [0.6437390061712722] produced [0.561947918100745]
Sample [6.28]  expected [0.498407349103431] produced [0.1174488327804603]
Sample [3.22]  expected [0.46083648326456733] produced [0.4609996069330281]
Sample [0.97]  expected [0.912442856669225] produced [0.979153872708671]
Sample [1.35]  expected [0.9878616789133295] produced [0.9517268089484976]
Sample [3.79]  expected [0.1980411234943697] produced [0.3230640812608359]
Sample [6.16]  expected [0.43856300244672497] produced [0.12010756672011977]
Sample [1.43]  expected [0.9950522801685888] produced [0.9424515668490332]
Sample [3.41]  expected [0.3674019270641133] produced [0.40631412889571017]
Sample [3.53]  expected [0.310642563085501] produced [0.3734847895770158]
Sample [5.73]  expected [0.2372999374090604] produced [0.13324047079466042]
Sample [2.07]  expected [0.938982031499539] produced [0.8178045081431593]
Sample [4.09]  expected [0.09375597305600791] produced [0.269178447043797]
Sample [5.17]  expected [0.05144473857517878] produced [0.15859339653629878]
Sample [2.7]  expected [0.7136899401169149] produced [0.6098805904671948]
Sample [3.16]  expected [0.4907968465334731] produced [0.4709539924100075]
Sample [1.75]  expected [0.9919929734369685] produced [0.8932783248971455]
Sample [2.77]  expected [0.6815499236020841] produced [0.6010749971529018]
Sample [0.99]  expected [0.9180129893002602] produced [0.9784717814405645]
Sample [4.71]  expected [1.42680614101387e-06] produced [0.199792009781189]
Sample [0.35]  expected [0.6714489037277257] produced [0.9953701792338122]
Sample [0.22]  expected [0.6091148115404347] produced [0.9966728187691106]
Sample [0.27]  expected [0.6333657183444156] produced [0.9962201879230308]
Sample [3.7]  expected [0.2350819295457533] produced [0.3375330536702984]
Sample [2.55]  expected [0.7788418586957083] produced [0.6657237076069842]
Sample [1.07]  expected [0.9386002521373408] produced [0.9736212464168913]
Sample [5.7]  expected [0.2246572287011812] produced [0.13643136588803897]
Sample [4.43]  expected [0.019803755882228324] produced [0.22533031171335458]
Sample [4.64]  expected [0.0013094691509533685] produced [0.19818931236943477]
Sample [1.39]  expected [0.9918504074056383] produced [0.944199526392607]
Sample [5.08]  expected [0.03340571177135121] produced [0.16181938946672728]
Sample [1.09]  expected [0.9433134572247436] produced [0.9707859318957215]
Sample [4.77]  expected [0.0008295279215562168] produced [0.18167135562323763]
Sample [5.74]  expected [0.24156727780128562] produced [0.12818027682278754]
Sample [3.9]  expected [0.15611692040801312] produced [0.28140246086168846]
Sample [1.44]  expected [0.9957291740958432] produced [0.9348568333280994]
Sample [2.59]  expected [0.7620221708436381] produced [0.6302866965472791]
Sample [4.48]  expected [0.013440508387413064] produced [0.20766318881895696]
Sample [2.82]  expected [0.6580389821085268] produced [0.5535480183657238]
Sample [0.43]  expected [0.7084354012146055] produced [0.9941911789586475]
Sample [3.48]  expected [0.33400740588963296] produced [0.37031954032904574]
Sample [1.11]  expected [0.9478493428400239] produced [0.9692122583592889]
Sample [5.36]  expected [0.10123634804414783] produced [0.14450060141738116]
Sample [0.09]  expected [0.5449392745990055] produced [0.997589134484235]
Sample [3.39]  expected [0.3770697478576815] produced [0.38844959416250807]
Sample [2.01]  expected [0.9525452816626006] produced [0.822943912507032]
Sample [1.1]  expected [0.9456036800307177] produced [0.9702886338226077]
Sample [4.05]  expected [0.10573737278690254] produced [0.2622121694198977]
Sample [1.14]  expected [0.9543167480579418] produced [0.9665698240299527]
Sample [3.14]  expected [0.5007963264582435] produced [0.45319127384796104]
Sample [2.66]  expected [0.7315956324651726] produced [0.6142181052746151]
Sample [5.55]  expected [0.165380071361869] produced [0.13872425663178922]
Sample [4.2]  expected [0.06421211379320602] produced [0.2448236293634842]
Sample [5.57]  expected [0.17287696217210435] produced [0.13540005523779838]
Sample [0.21]  expected [0.6042299499230498] produced [0.9967028615724758]
Sample [5.84]  expected [0.2855903943333021] produced [0.12593390914206684]
Sample [5.4]  expected [0.11361775622200643] produced [0.14381290004537794]
Sample [0.48]  expected [0.7308895877707414] produced [0.9933997254140196]
Sample [1.41]  expected [0.9935500505069252] produced [0.94135676642119]
Sample [5.19]  expected [0.05595221400862288] produced [0.1543894683805818]
Sample [2.8]  expected [0.6674940750779523] produced [0.5681638130810891]
Sample [4.84]  expected [0.004065621344543691] produced [0.17945019451634364]
Sample [5.87]  expected [0.2992356937968927] produced [0.12552444145614028]
Sample [6.22]  expected [0.4684283638766936] produced [0.11647750061809044]
Sample [6.01]  expected [0.3651000076492419] produced [0.12338729162688054]
Sample [0.0]  expected [0.5] produced [0.9981035555110582]
Sample [5.18]  expected [0.053676160285882624] produced [0.1596389564028065]
Sample [4.62]  expected [0.002132413468877381] produced [0.1998827068348969]
Sample [2.09]  expected [0.9341072917228064] produced [0.8055563249174656]
Sample [3.69]  expected [0.23933560482279675] produced [0.3324046910096262]
Sample [3.54]  expected [0.30602454102913484] produced [0.35888711584805966]
Sample [3.57]  expected [0.29228869661437684] produced [0.3484172419177838]
Sample [2.46]  expected [0.8150153149979461] produced [0.6815502078157007]
Sample [5.23]  expected [0.06549813048404202] produced [0.15360272566895714]
Sample [1.04]  expected [0.9312021136216693] produced [0.9741859309705685]
Sample [3.94]  expected [0.14187721543801474] produced [0.280156068470279]
Sample [4.9]  expected [0.008773693687833761] produced [0.1696366862674325]
Sample [2.78]  expected [0.6768821726505714] produced [0.56459638265642]
Sample [3.84]  expected [0.17850062897304558] produced [0.29428717834385565]
Sample [1.5]  expected [0.9987474933020273] produced [0.9273344234549489]
Sample [1.31]  expected [0.983092475806367] produced [0.9513057856157388]
Sample [5.02]  expected [0.02347018464981626] produced [0.16137215573945243]
Sample [2.24]  expected [0.8921579625422099] produced [0.7500130397975882]
Sample [4.99]  expected [0.019143548286603307] produced [0.16520551322040464]
Sample [1.8]  expected [0.9869238154390976] produced [0.8737564860251882]
Sample [5.66]  expected [0.20818766929849636] produced [0.13189958702560156]
Sample [2.02]  expected [0.9503965957613136] produced [0.823010706011479]
Sample [2.95]  expected [0.5952113236805135] produced [0.5267713420659592]
Sample [2.88]  expected [0.6293096748305553] produced [0.556483396399931]
Sample [1.62]  expected [0.999394871735262] produced [0.9166957664761979]
Sample [5.51]  expected [0.15079076539189323] produced [0.14447523958049796]
Sample [0.66]  expected [0.8065584259867169] produced [0.9900109317008002]
Sample [4.67]  expected [0.0004491391564075964] produced [0.2000211010556497]
Sample [3.51]  expected [0.31993495026401575] produced [0.37465190774186624]
Sample [0.55]  expected [0.7613436144653296] produced [0.9922056554352178]
Sample [1.99]  expected [0.9567066806706126] produced [0.8357362693743876]
Sample [4.65]  expected [0.0009727806205602207] produced [0.19806788036267753]
Sample [4.79]  expected [0.0015051118641152272] produced [0.1821755553963088]
Sample [0.75]  expected [0.8408193800116671] produced [0.9869498419835174]
Sample [4.63]  expected [0.00169602631885718] produced [0.1923101748662643]
Sample [1.24]  expected [0.9728919997247694] produced [0.9579527614279092]
Sample [0.37]  expected [0.680807715982481] produced [0.9949534539844407]
Sample [2.93]  expected [0.6050086496254495] produced [0.5166649047897992]
Sample [1.6]  expected [0.9997868015207525] produced [0.9138738385713311]
Sample [0.19]  expected [0.5944294474882503] produced [0.9968681406036463]
Sample [4.46]  expected [0.015840692916646415] produced [0.20982358310176202]
Sample [1.54]  expected [0.9997629153027395] produced [0.9215903123788696]
Sample [0.46]  expected [0.7219740534827599] produced [0.9936544517087967]
Sample [2.27]  expected [0.8826774762646268] produced [0.7442849526197817]
Sample [0.28]  expected [0.6381778242820568] produced [0.9960517753907051]
Sample [0.33]  expected [0.6620215141974342] produced [0.9955084543812782]
Sample [1.05]  expected [0.9337116127970084] produced [0.973576506345631]
Sample [3.67]  expected [0.24792072607319426] produced [0.33002746830648366]
Sample [4.6]  expected [0.00315449818326774] produced [0.1940894328280393]
Sample [4.76]  expected [0.0005665952547928566] produced [0.1772553988305369]
Sample [2.9]  expected [0.6196246646069912] produced [0.5180889571795492]
Sample [5.92]  expected [0.3223732200059787] produced [0.1214080615949365]
Sample [1.95]  expected [0.9644798575019347] produced [0.8389594658069975]
Sample [2.26]  expected [0.8858763310100628] produced [0.7537471013475741]
Sample [1.77]  expected [0.9901122363940227] produced [0.8870030507482216]
Sample [2.67]  expected [0.7271528349151533] produced [0.6287122705885235]
Sample [2.65]  expected [0.7360152706449413] produced [0.6444500641612662]
Sample [4.58]  expected [0.004375314473866576] produced [0.21349872469333578]
Sample [1.93]  expected [0.9680885261581531] produced [0.8572662194274534]
Sample [2.44]  expected [0.8227174991671853] produced [0.7167761521854276]
Sample [5.53]  expected [0.15801702406154972] produced [0.14684615828024916]
Sample [3.08]  expected [0.5307768587149566] produced [0.5159206389186]
Sample [3.78]  expected [0.20204138809611788] produced [0.3361426246813031]
Sample [4.51]  expected [0.010205417785816573] produced [0.21759183605251686]
Sample [4.74]  expected [0.00019057999290728045] produced [0.1901246664525449]
Sample [2.4]  expected [0.8377315902755755] produced [0.7133791632983295]
Sample [2.84]  expected [0.6485206756534162] produced [0.5742305363945542]
Sample [0.85]  expected [0.8756402025701463] produced [0.9842775220500133]
Sample [1.69]  expected [0.9964518255470594] produced [0.9064738070362773]
Sample [5.33]  expected [0.09236789277501822] produced [0.15418121482754849]
Sample [2.15]  expected [0.9184493953992489] produced [0.8032756469566235]
Sample [0.57]  expected [0.7698160243669847] produced [0.992093859823573]
Sample [5.63]  expected [0.19614044863800711] produced [0.14114699366898617]
Sample [2.11]  expected [0.9290589148174044] produced [0.819331189089647]
Sample [0.69]  expected [0.8182685911109839] produced [0.9895769173704062]
Sample [0.54]  expected [0.7570679958265566] produced [0.9927341294590574]
Sample [0.65]  expected [0.8025932028680198] produced [0.9905180970166215]
Sample [2.17]  expected [0.912892496552804] produced [0.8068064058171928]
Sample [6.12]  expected [0.41876899239242293] produced [0.1265583219350217]
Sample [4.08]  expected [0.09669112570837973] produced [0.29269753731870446]
Sample [4.15]  expected [0.07700814946227674] produced [0.27079060624490764]
Sample [2.92]  expected [0.6098918061125584] produced [0.554952771102444]
Sample [0.42]  expected [0.7038802265297851] produced [0.9945176478023525]
Sample [2.29]  expected [0.8761652881970854] produced [0.7659519911827657]
Sample [3.59]  expected [0.2832345586236411] produced [0.37797019801667187]
Sample [1.03]  expected [0.9286494945943017] produced [0.9764033605545911]
Sample [1.86]  expected [0.9792356415394572] produced [0.8759004033702954]
Sample [0.63]  expected [0.7945723789711348] produced [0.9908634581200044]
Sample [2.2]  expected [0.904248201909795] produced [0.7943270125307594]
Sample [3.13]  expected [0.5057961969680791] produced [0.5048865781236775]
Sample [1.23]  expected [0.9712444009658487] produced [0.9644224402037995]
Sample [4.66]  expected [0.0006859943962505555] produced [0.20925892679494684]
Sample [0.61]  expected [0.7864337300502406] produced [0.9912427353215063]
Sample [4.32]  expected [0.0380009206384061] produced [0.24152857946187883]
Sample [4.16]  expected [0.07436329953221282] produced [0.25552063486121634]
RMSE is 0.16955802020711877

Process finished with exit code 0

"""
