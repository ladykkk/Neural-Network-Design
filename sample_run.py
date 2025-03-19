"""For this assignment we will Complete Neural Network"""

from enum import Enum
from abc import ABC, abstractmethod
from math import exp, floor
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
    """
    1.put all data in behind the code?
    2.delete output data if it's not that important in word report, or just use
    rmse for the analysis part
    3. for the graph of the sin curve, should we upload the code for it or put
    the code in the report?
    4. confirm xor part 2 change the number of layer, with or without bias
    """

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
        rmse = 0
        for epoch in range(epochs):
            data_set.prime_data(order)
            cnt = 1
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
                    rmse = np.sqrt((rmse ** 2 * (cnt - 1) + error ** 2) / cnt)
                    self.my_layers.output_nodes[i].set_expected(label[i])
                    cnt += 1
                """
                if verbosity > 1 and epoch % 1000 == 0:
                    print(f"Sample {input_val} ", f"expected {expected_val}",
                          f"produced {output_val}")
                """
            if verbosity > 0 and epoch % 100 == 0:
                print(f"Epoch {epoch} RMSE = {rmse}")
        print(f"Final Epoch RMSE = {rmse}")

    def test(self, data_set: NNData, order=NNData.Order.SEQUENTIAL):
        if data_set.pool_is_empty(data_set.Set.TEST):
            raise FFBPNetwork.EmptySetException("The testing set is empty")
        rmse = 0
        data_set.prime_data(order)
        cnt = 1
        print("TESTING")
        while not data_set.pool_is_empty(data_set.Set.TEST):
            input_val, output_val, expected_val = [], [], []
            feature, label = data_set.get_one_item(data_set.Set.TEST)
            for i in range(self.num_inputs):
                input_val.append(feature[i])
                self.my_layers.input_nodes[i].set_input(feature[i])
                sin_in.append(feature[i])
            for i in range(self.num_outputs):
                val = self.my_layers.output_nodes[i].value
                output_val.append(val)
                expected_val.append(label[i])
                error = label[i] - val
                sin_out.append(val)
                rmse = np.sqrt((rmse ** 2 * (cnt - 1) + error ** 2) / cnt)
                self.my_layers.output_nodes[i].set_expected(label[i])
                cnt += 1
            print(f"Sample {input_val} ", f"expected {expected_val}",
                  f"produced {output_val}")
        print(f"RMSE is {rmse}")


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
    sin_Y = [[0], [0.00999983333416666], [0.01999866669333308],
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
    network.train(data, 20001, order=NNData.Order.RANDOM)


def run_XOR_with_bias():

    network = FFBPNetwork(3, 1)
    network.add_hidden_layer(5)
    XOR_X = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
    XOR_Y = [[0], [1], [1], [0]]
    print("This is the sample run of XOR with bias.")
    data = NNData(XOR_X, XOR_Y, 1)
    network.train(data, 20001, order=NNData.Order.RANDOM)


def plot_sin(sin_in, sin_out):
    x_actual = np.linspace(0, np.pi * 2, 629)
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
    sin_in = []
    sin_out = []
    run_sin()
    plot_sin(sin_in, sin_out)
    input("\nPlease Enter to test XOR without bias")
    run_XOR()
    input("\nPlease Enter to test XOR with bias")
    run_XOR_with_bias()
    input("\nPlease Enter to test trans sin")
    sin_in = []
    sin_out = []
    run_trans_sin()
    sin_out = [y * 2 - 1 for y in sin_out]
    plot_sin(sin_in, sin_out)




