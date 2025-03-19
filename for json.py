from enum import Enum
from abc import ABC, abstractmethod
from math import exp, floor, sqrt
import numpy as np
from collections import deque
import random
import json


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

    @property
    def train_factor(self):
        return self._train_factor

    @train_factor.setter
    def train_factor(self, train_factor):
        self._train_factor = self.percentage_limiter(train_factor)

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


class MultiTypeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, deque):
            return {"__deque__": list(o)}
        elif isinstance(o, np.ndarray):
            return {"__NDarray__": o.tolist()}
        elif isinstance(o, NNData):
            return {"__NNData__": o.__dict__}
        else:
            super().default(o)


def multi_type_decoder(o):
    if "__deque__" in o:
        return deque(o["__deque__"])
    if "__NDarray__" in o:
        return np.array(o["__NDarray__"])
    if "__NNData__" in o:
        item = o["__NNData__"]
        ret_obj = NNData()
        # ndarray
        ret_obj._features = item["_features"]
        ret_obj._labels = item["_labels"]
        # list
        ret_obj._train_indices = item["_train_indices"]
        ret_obj._test_indices = item["_test_indices"]
        # deque
        ret_obj._train_pool = item["_train_pool"]
        ret_obj._test_pool = item["_test_pool"]
        # float
        ret_obj._train_factor = item["_train_factor"]
        return ret_obj
    else:
        return o


def run_XOR():

    network = FFBPNetwork(2, 1)
    network.add_hidden_layer(5)
    XOR_X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    XOR_Y = [[0], [1], [1], [0]]
    print("This is the sample run of XOR without bias.")
    xor_data = NNData(XOR_X, XOR_Y, 1)
    xor_data_encoded = json.dumps(xor_data, cls=MultiTypeEncoder)
    xor_data_decoded = json.loads(xor_data_encoded,
                                  object_hook=multi_type_decoder)
    print("xor_data_encoded", xor_data_encoded)
    print("xor_data_decoded.__dict__", xor_data_decoded.__dict__)
    network.train(xor_data_decoded, 10001, order=NNData.Order.RANDOM)


def run_sin():
    with open("sin_data.txt", "r") as f:
        sin_decoded = json.load(f, object_hook=multi_type_decoder)
    print("sin_decoded.__dict__", sin_decoded.__dict__)
    network = FFBPNetwork(1, 1)
    network.train(sin_decoded, 10001, order=NNData.Order.RANDOM)
    network.test(sin_decoded)


if __name__ == "__main__":
    input("Please Enter to test XOR without bias")
    run_XOR()
    input("\nPlease Enter to test sin")
    run_sin()


"""
"/Users/jiafei/Desktop/courses/W23 CS F003B 02W Intermed Software Desgn Python/
venv/bin/python" /Users/jiafei/Desktop/courses/W23 CS F003B 02W Intermed 
Software Desgn Python/projects/assignment for json.py 
Please Enter to test XOR without bias
This is the sample run of XOR without bias.
xor_data_encoded {"__NNData__": {"_labels": {"__NDarray__": [[0.0], [1.0], [1.0], [0.0]]}, "_features": {"__NDarray__": [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]}, "_train_indices": [0, 1, 3, 2], "_test_indices": [], "_train_pool": {"__deque__": [0, 1, 3, 2]}, "_test_pool": {"__deque__": []}, "_train_factor": 1.0}}
xor_data_decoded.__dict__ {'_labels': array([[0.],
       [1.],
       [1.],
       [0.]]), '_features': array([[0., 0.],
       [0., 1.],
       [1., 0.],
       [1., 1.]]), '_train_indices': [0, 1, 3, 2], '_test_indices': [], '_train_pool': deque([0, 1, 3, 2]), '_test_pool': deque([]), '_train_factor': 1.0}
Sample [0.0, 0.0]  expected [0.0] produced [0.7991858028931212]
Sample [0.0, 1.0]  expected [1.0] produced [0.8508699141713895]
Sample [1.0, 1.0]  expected [0.0] produced [0.8823994020994574]
Sample [1.0, 0.0]  expected [1.0] produced [0.8460846373370882]
Epoch 0 RMSE = 0.6048256719366142
Epoch 100 RMSE = 0.5069987517053461
Epoch 200 RMSE = 0.5009400384925484
Epoch 300 RMSE = 0.5007710884087863
Epoch 400 RMSE = 0.5006738178502214
Epoch 500 RMSE = 0.5005736004304043
Epoch 600 RMSE = 0.5004674349932736
Epoch 700 RMSE = 0.500353958189173
Epoch 800 RMSE = 0.5002317868712833
Epoch 900 RMSE = 0.5000993782084624
Sample [0.0, 0.0]  expected [0.0] produced [0.48990900057278886]
Sample [0.0, 1.0]  expected [1.0] produced [0.49965398375511805]
Sample [1.0, 1.0]  expected [0.0] produced [0.5180087247175468]
Sample [1.0, 0.0]  expected [1.0] produced [0.5089501649180085]
Epoch 1000 RMSE = 0.49995498403153826
Epoch 1100 RMSE = 0.49979660915476826
Epoch 1200 RMSE = 0.49962196464540654
Epoch 1300 RMSE = 0.49942841455872505
Epoch 1400 RMSE = 0.4992129155608418
Epoch 1500 RMSE = 0.49897194926845473
Epoch 1600 RMSE = 0.49870144769943453
Epoch 1700 RMSE = 0.4983967130882275
Epoch 1800 RMSE = 0.4980523345662524
Epoch 1900 RMSE = 0.4976621059153905
Sample [0.0, 0.0]  expected [0.0] produced [0.47641087482769684]
Sample [0.0, 1.0]  expected [1.0] produced [0.502064496556562]
Sample [1.0, 1.0]  expected [0.0] produced [0.5291620100651475]
Sample [1.0, 0.0]  expected [1.0] produced [0.5162777449551383]
Epoch 2000 RMSE = 0.4972189508067434
Epoch 2100 RMSE = 0.496714864584058
Epoch 2200 RMSE = 0.4961408845377044
Epoch 2300 RMSE = 0.4954871033155886
Epoch 2400 RMSE = 0.49474274193247625
Epoch 2500 RMSE = 0.49389629880403907
Epoch 2600 RMSE = 0.4929357882438794
Epoch 2700 RMSE = 0.49184907497571767
Epoch 2800 RMSE = 0.4906243000994692
Epoch 2900 RMSE = 0.489250379407097
Sample [0.0, 0.0]  expected [0.0] produced [0.4505552144867586]
Sample [0.0, 1.0]  expected [1.0] produced [0.5131445470355332]
Sample [1.0, 1.0]  expected [0.0] produced [0.5419329177745289]
Sample [1.0, 0.0]  expected [1.0] produced [0.5333587335790285]
Epoch 3000 RMSE = 0.487717539226607
Epoch 3100 RMSE = 0.48601784163947537
Epoch 3200 RMSE = 0.4841456440041077
Epoch 3300 RMSE = 0.4820979403552203
Epoch 3400 RMSE = 0.4798745452395472
Epoch 3500 RMSE = 0.47747810164251847
Epoch 3600 RMSE = 0.47491391903947217
Epoch 3700 RMSE = 0.47218966948448654
Epoch 3800 RMSE = 0.4693149842896305
Epoch 3900 RMSE = 0.46630099902729116
Sample [0.0, 0.0]  expected [0.0] produced [0.40784243096503425]
Sample [0.0, 1.0]  expected [1.0] produced [0.5378018846196364]
Sample [1.0, 1.0]  expected [0.0] produced [0.5366370490713659]
Sample [1.0, 0.0]  expected [1.0] produced [0.5639650592410366]
Epoch 4000 RMSE = 0.46315989095433474
Epoch 4100 RMSE = 0.4599044432699394
Epoch 4200 RMSE = 0.456547658439505
Epoch 4300 RMSE = 0.4531024312731922
Epoch 4400 RMSE = 0.4495812835057322
Epoch 4500 RMSE = 0.4459961559475878
Epoch 4600 RMSE = 0.4423582515148695
Epoch 4700 RMSE = 0.438677921704564
Epoch 4800 RMSE = 0.43496458938270804
Epoch 4900 RMSE = 0.4312267013380487
Sample [0.0, 0.0]  expected [0.0] produced [0.36121900720550004]
Sample [0.0, 1.0]  expected [1.0] produced [0.572766397305773]
Sample [1.0, 1.0]  expected [0.0] produced [0.505249859893905]
Sample [1.0, 0.0]  expected [1.0] produced [0.5967096712210026]
Epoch 5000 RMSE = 0.42747170451595334
Epoch 5100 RMSE = 0.4237060401078401
Epoch 5200 RMSE = 0.4199351498612832
Epoch 5300 RMSE = 0.41616348929351776
Epoch 5400 RMSE = 0.4123945430874997
Epoch 5500 RMSE = 0.408630838870285
Epoch 5600 RMSE = 0.4048739567555217
Epoch 5700 RMSE = 0.40112453334151116
Epoch 5800 RMSE = 0.397382260141033
Epoch 5900 RMSE = 0.39364587755503905
Sample [0.0, 0.0]  expected [0.0] produced [0.3179425892158602]
Sample [0.0, 1.0]  expected [1.0] produced [0.6131306214462667]
Sample [1.0, 1.0]  expected [0.0] produced [0.46581985750441196]
Sample [1.0, 0.0]  expected [1.0] produced [0.6253193844654632]
Epoch 6000 RMSE = 0.38991316642382284
Epoch 6100 RMSE = 0.386180939897452
Epoch 6200 RMSE = 0.3824450389188265
Epoch 6300 RMSE = 0.37870033509620116
Epoch 6400 RMSE = 0.37494074524736976
Epoch 6500 RMSE = 0.371159262486369
Epoch 6600 RMSE = 0.3673480094026735
Epoch 6700 RMSE = 0.36349831958651657
Epoch 6800 RMSE = 0.3596008543329384
Epoch 6900 RMSE = 0.35564576157833466
Sample [0.0, 0.0]  expected [0.0] produced [0.2783655480701299]
Sample [0.0, 1.0]  expected [1.0] produced [0.6526237209269088]
Sample [1.0, 1.0]  expected [0.0] produced [0.42191131254726066]
Sample [1.0, 0.0]  expected [1.0] produced [0.6559247232867578]
Epoch 7000 RMSE = 0.3516228836821121
Epoch 7100 RMSE = 0.34752201921689574
Epoch 7200 RMSE = 0.3433332411409397
Epoch 7300 RMSE = 0.3390472693589335
Epoch 7400 RMSE = 0.3346558896988235
Epoch 7500 RMSE = 0.3301524040305721
Epoch 7600 RMSE = 0.3255320883319194
Epoch 7700 RMSE = 0.32079262810945586
Epoch 7800 RMSE = 0.3159344951940699
Epoch 7900 RMSE = 0.3109612281207277
Sample [0.0, 0.0]  expected [0.0] produced [0.24196431399232549]
Sample [0.0, 1.0]  expected [1.0] produced [0.6968590747783462]
Sample [1.0, 1.0]  expected [0.0] produced [0.36280472516303586]
Sample [1.0, 0.0]  expected [1.0] produced [0.6963869983683336]
Epoch 8000 RMSE = 0.3058795813512375
Epoch 8100 RMSE = 0.3006995170569026
Epoch 8200 RMSE = 0.2954340265356145
Epoch 8300 RMSE = 0.2900987849343999
Epoch 8400 RMSE = 0.28471166022247546
Epoch 8500 RMSE = 0.27929211240847784
Epoch 8600 RMSE = 0.2738605292823477
Epoch 8700 RMSE = 0.2684375489413873
Epoch 8800 RMSE = 0.263043416806388
Epoch 8900 RMSE = 0.2576974167882221
Sample [0.0, 0.0]  expected [0.0] produced [0.2080227423918994]
Sample [0.0, 1.0]  expected [1.0] produced [0.7492735498034795]
Sample [1.0, 1.0]  expected [0.0] produced [0.2906441782816268]
Sample [1.0, 0.0]  expected [1.0] produced [0.7465302145702087]
Epoch 9000 RMSE = 0.2524174046471545
Epoch 9100 RMSE = 0.24721945863993464
Epoch 9200 RMSE = 0.24211765031205906
Epoch 9300 RMSE = 0.23712392823673642
Epoch 9400 RMSE = 0.232248100394671
Epoch 9500 RMSE = 0.2274978968455302
Epoch 9600 RMSE = 0.22287909300223746
Epoch 9700 RMSE = 0.21839567455810507
Epoch 9800 RMSE = 0.21405002723412367
Epoch 9900 RMSE = 0.20984313736734223
Sample [0.0, 0.0]  expected [0.0] produced [0.17977342819420453]
Sample [0.0, 1.0]  expected [1.0] produced [0.7959962014049122]
Sample [1.0, 1.0]  expected [0.0] produced [0.22853068177084349]
Sample [1.0, 0.0]  expected [1.0] produced [0.7921280370553756]
Epoch 10000 RMSE = 0.20577479244007257
Final Epoch RMSE = 0.20577479244007257

Please Enter to test sin
sin_decoded.__dict__ {'_labels': array([[0.        ],
       [0.00999983],
       [0.01999867],
       [0.0299955 ],
       [0.03998933],
       [0.04997917],
       [0.05996401],
       [0.06994285],
       [0.07991469],
       [0.08987855],
       [0.09983342],
       [0.1097783 ],
       [0.11971221],
       [0.12963414],
       [0.13954311],
       [0.14943813],
       [0.15931821],
       [0.16918235],
       [0.17902957],
       [0.18885889],
       [0.19866933],
       [0.2084599 ],
       [0.21822962],
       [0.22797752],
       [0.23770263],
       [0.24740396],
       [0.25708055],
       [0.26673144],
       [0.27635565],
       [0.28595223],
       [0.29552021],
       [0.30505864],
       [0.31456656],
       [0.32404303],
       [0.33348709],
       [0.34289781],
       [0.35227423],
       [0.36161543],
       [0.37092047],
       [0.38018842],
       [0.38941834],
       [0.39860933],
       [0.40776045],
       [0.4168708 ],
       [0.42593947],
       [0.43496553],
       [0.44394811],
       [0.45288629],
       [0.46177918],
       [0.47062589],
       [0.47942554],
       [0.48817725],
       [0.49688014],
       [0.50553334],
       [0.51413599],
       [0.52268723],
       [0.5311862 ],
       [0.53963205],
       [0.54802394],
       [0.55636102],
       [0.56464247],
       [0.57286746],
       [0.58103516],
       [0.58914476],
       [0.59719544],
       [0.60518641],
       [0.61311685],
       [0.62098599],
       [0.62879302],
       [0.63653718],
       [0.64421769],
       [0.65183377],
       [0.65938467],
       [0.66686964],
       [0.67428791],
       [0.68163876],
       [0.68892145],
       [0.69613524],
       [0.70327942],
       [0.71035327],
       [0.71735609],
       [0.72428717],
       [0.73114583],
       [0.73793137],
       [0.74464312],
       [0.75128041],
       [0.75784256],
       [0.76432894],
       [0.77073888],
       [0.77707175],
       [0.78332691],
       [0.78950374],
       [0.79560162],
       [0.80161994],
       [0.8075581 ],
       [0.8134155 ],
       [0.81919157],
       [0.82488571],
       [0.83049737],
       [0.83602598],
       [0.84147098],
       [0.84683184],
       [0.85210802],
       [0.85729899],
       [0.86240423],
       [0.86742323],
       [0.87235548],
       [0.8772005 ],
       [0.88195781],
       [0.88662691],
       [0.89120736],
       [0.89569869],
       [0.90010044],
       [0.90441219],
       [0.9086335 ],
       [0.91276394],
       [0.91680311],
       [0.9207506 ],
       [0.92460601],
       [0.92836897],
       [0.93203909],
       [0.935616  ],
       [0.93909936],
       [0.9424888 ],
       [0.945784  ],
       [0.94898462],
       [0.95209034],
       [0.95510086],
       [0.95801586],
       [0.96083506],
       [0.96355819],
       [0.96618495],
       [0.9687151 ],
       [0.97114838],
       [0.97348454],
       [0.97572336],
       [0.9778646 ],
       [0.97990806],
       [0.98185353],
       [0.98370081],
       [0.98544973],
       [0.9871001 ],
       [0.98865176],
       [0.99010456],
       [0.99145835],
       [0.99271299],
       [0.99386836],
       [0.99492435],
       [0.99588084],
       [0.99673775],
       [0.99749499],
       [0.99815247],
       [0.99871014],
       [0.99916795],
       [0.99952583],
       [0.99978376],
       [0.99994172],
       [0.99999968]]), '_features': array([[0.  ],
       [0.01],
       [0.02],
       [0.03],
       [0.04],
       [0.05],
       [0.06],
       [0.07],
       [0.08],
       [0.09],
       [0.1 ],
       [0.11],
       [0.12],
       [0.13],
       [0.14],
       [0.15],
       [0.16],
       [0.17],
       [0.18],
       [0.19],
       [0.2 ],
       [0.21],
       [0.22],
       [0.23],
       [0.24],
       [0.25],
       [0.26],
       [0.27],
       [0.28],
       [0.29],
       [0.3 ],
       [0.31],
       [0.32],
       [0.33],
       [0.34],
       [0.35],
       [0.36],
       [0.37],
       [0.38],
       [0.39],
       [0.4 ],
       [0.41],
       [0.42],
       [0.43],
       [0.44],
       [0.45],
       [0.46],
       [0.47],
       [0.48],
       [0.49],
       [0.5 ],
       [0.51],
       [0.52],
       [0.53],
       [0.54],
       [0.55],
       [0.56],
       [0.57],
       [0.58],
       [0.59],
       [0.6 ],
       [0.61],
       [0.62],
       [0.63],
       [0.64],
       [0.65],
       [0.66],
       [0.67],
       [0.68],
       [0.69],
       [0.7 ],
       [0.71],
       [0.72],
       [0.73],
       [0.74],
       [0.75],
       [0.76],
       [0.77],
       [0.78],
       [0.79],
       [0.8 ],
       [0.81],
       [0.82],
       [0.83],
       [0.84],
       [0.85],
       [0.86],
       [0.87],
       [0.88],
       [0.89],
       [0.9 ],
       [0.91],
       [0.92],
       [0.93],
       [0.94],
       [0.95],
       [0.96],
       [0.97],
       [0.98],
       [0.99],
       [1.  ],
       [1.01],
       [1.02],
       [1.03],
       [1.04],
       [1.05],
       [1.06],
       [1.07],
       [1.08],
       [1.09],
       [1.1 ],
       [1.11],
       [1.12],
       [1.13],
       [1.14],
       [1.15],
       [1.16],
       [1.17],
       [1.18],
       [1.19],
       [1.2 ],
       [1.21],
       [1.22],
       [1.23],
       [1.24],
       [1.25],
       [1.26],
       [1.27],
       [1.28],
       [1.29],
       [1.3 ],
       [1.31],
       [1.32],
       [1.33],
       [1.34],
       [1.35],
       [1.36],
       [1.37],
       [1.38],
       [1.39],
       [1.4 ],
       [1.41],
       [1.42],
       [1.43],
       [1.44],
       [1.45],
       [1.46],
       [1.47],
       [1.48],
       [1.49],
       [1.5 ],
       [1.51],
       [1.52],
       [1.53],
       [1.54],
       [1.55],
       [1.56],
       [1.57]]), '_train_indices': [1, 8, 15, 17, 21, 24, 34, 39, 41, 44, 47, 48, 49, 53, 54, 56, 61, 66, 69, 80, 82, 83, 87, 90, 97, 110, 120, 136, 145, 146, 148], '_test_indices': [0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 18, 19, 20, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37, 38, 40, 42, 43, 45, 46, 50, 51, 52, 55, 57, 58, 59, 60, 62, 63, 64, 65, 67, 68, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 81, 84, 85, 86, 88, 89, 91, 92, 93, 94, 95, 96, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 137, 138, 139, 140, 141, 142, 143, 144, 147, 149, 150, 151, 152, 153, 154, 155, 156, 157], '_train_pool': deque([83, 146, 120, 34, 136, 49, 66, 24, 80, 148, 48, 47, 39, 53, 82, 69, 44, 61, 1, 15, 41, 17, 97, 90, 145, 110, 8, 87, 56, 54, 21]), '_test_pool': deque([7, 155, 127, 143, 141, 121, 62, 151, 156, 93, 63, 105, 144, 113, 100, 103, 154, 84, 94, 18, 123, 134, 92, 68, 67, 35, 116, 19, 98, 75, 76, 22, 43, 132, 33, 71, 65, 95, 89, 51, 124, 91, 23, 50, 101, 16, 46, 111, 58, 102, 138, 106, 78, 4, 13, 14, 119, 99, 125, 149, 152, 153, 27, 0, 85, 131, 5, 117, 157, 57, 142, 10, 88, 72, 42, 3, 37, 147, 40, 29, 81, 36, 30, 133, 79, 55, 77, 130, 20, 150, 25, 45, 12, 11, 129, 9, 2, 114, 59, 115, 96, 38, 139, 86, 32, 118, 107, 52, 122, 126, 108, 112, 70, 6, 60, 135, 64, 73, 31, 28, 128, 109, 137, 104, 26, 140, 74]), '_train_factor': 0.2}
Sample [0.01]  expected [0.00999983333416666] produced [0.5019430668485175]
Sample [0.08]  expected [0.0799146939691727] produced [0.5155383781607008]
Sample [0.15]  expected [0.149438132473599] produced [0.5290946185116064]
Sample [0.17]  expected [0.169182349066996] produced [0.5329332913758104]
Sample [0.21]  expected [0.2084598998461] produced [0.5406112215479503]
Sample [0.24]  expected [0.237702626427135] produced [0.5463300427468749]
Sample [0.34]  expected [0.333487092140814] produced [0.5653687192420389]
Sample [0.39]  expected [0.380188415123161] produced [0.5747543777817951]
Sample [0.41]  expected [0.398609327984423] produced [0.5784334118445928]
Sample [0.44]  expected [0.425939465066] produced [0.5839712568395262]
Sample [0.47]  expected [0.452886285379068] produced [0.5894811025592575]
Sample [0.48]  expected [0.461779175541483] produced [0.5912526425523563]
Sample [0.49]  expected [0.470625888171158] produced [0.59302112048031]
Sample [0.53]  expected [0.505533341204847] produced [0.6003244799608107]
Sample [0.54]  expected [0.514135991653113] produced [0.6020867751986092]
Sample [0.56]  expected [0.531186197920883] produced [0.6056798905450147]
Sample [0.61]  expected [0.572867460100481] produced [0.614722028854783]
Sample [0.66]  expected [0.613116851973434] produced [0.6237042886782912]
Sample [0.69]  expected [0.636537182221968] produced [0.6290661583886746]
Sample [0.8]  expected [0.717356090899523] produced [0.648503728099141]
Sample [0.82]  expected [0.731145829726896] produced [0.6521027919308211]
Sample [0.83]  expected [0.737931371109963] produced [0.6539771307021802]
Sample [0.87]  expected [0.764328937025505] produced [0.6610398086258711]
Sample [0.9]  expected [0.783326909627483] produced [0.6663826952027846]
Sample [0.97]  expected [0.82488571333845] produced [0.6784846353777775]
Sample [1.1]  expected [0.891207360061435] produced [0.7002752917976486]
Sample [1.2]  expected [0.932039085967226] produced [0.7167506346647688]
Sample [1.36]  expected [0.977864602435316] produced [0.7418782146209776]
Sample [1.45]  expected [0.992712991037588] produced [0.7558528114606947]
Sample [1.46]  expected [0.993868363411645] produced [0.75813756098467]
Sample [1.48]  expected [0.99588084453764] produced [0.7618440833674721]
Epoch 0 RMSE = 0.22041648636662464
Epoch 100 RMSE = 0.21672456860615558
Epoch 200 RMSE = 0.21672695427369276
Epoch 300 RMSE = 0.21672699746733914
Epoch 400 RMSE = 0.21672699809894153
Epoch 500 RMSE = 0.21672699810815146
Epoch 600 RMSE = 0.21672699810828583
Epoch 700 RMSE = 0.2167269981082878
Epoch 800 RMSE = 0.2167269981082878
Epoch 900 RMSE = 0.2167269981082878
Sample [0.01]  expected [0.00999983333416666] produced [0.5025581870868067]
Sample [0.08]  expected [0.0799146939691727] produced [0.5204530243953066]
Sample [0.15]  expected [0.149438132473599] produced [0.538279287954869]
Sample [0.17]  expected [0.169182349066996] produced [0.5433284987672722]
Sample [0.21]  expected [0.2084598998461] produced [0.5534120255020681]
Sample [0.24]  expected [0.237702626427135] produced [0.5609182602465094]
Sample [0.34]  expected [0.333487092140814] produced [0.5857933667377648]
Sample [0.39]  expected [0.380188415123161] produced [0.5980068515232557]
Sample [0.41]  expected [0.398609327984423] produced [0.6027929466704265]
Sample [0.44]  expected [0.425939465066] produced [0.6099723016602646]
Sample [0.47]  expected [0.452886285379068] produced [0.6170944412993606]
Sample [0.48]  expected [0.461779175541483] produced [0.6193876533867572]
Sample [0.49]  expected [0.470625888171158] produced [0.6216735761804072]
Sample [0.53]  expected [0.505533341204847] produced [0.6310537263842201]
Sample [0.54]  expected [0.514135991653113] produced [0.6333113262338554]
Sample [0.56]  expected [0.531186197920883] produced [0.6379018978287102]
Sample [0.61]  expected [0.572867460100481] produced [0.6494011819929137]
Sample [0.66]  expected [0.613116851973434] produced [0.6607375160246768]
Sample [0.69]  expected [0.636537182221968] produced [0.6674421089088021]
Sample [0.8]  expected [0.717356090899523] produced [0.6915781603749126]
Sample [0.82]  expected [0.731145829726896] produced [0.6959055830950059]
Sample [0.83]  expected [0.737931371109963] produced [0.6980913725026462]
Sample [0.87]  expected [0.764328937025505] produced [0.7065994388166158]
Sample [0.9]  expected [0.783326909627483] produced [0.7129390734653931]
Sample [0.97]  expected [0.82488571333845] produced [0.7273234190498193]
Sample [1.1]  expected [0.891207360061435] produced [0.7528001562606748]
Sample [1.2]  expected [0.932039085967226] produced [0.7714539465313621]
Sample [1.36]  expected [0.977864602435316] produced [0.7991603570495699]
Sample [1.45]  expected [0.992712991037588] produced [0.8138580281774362]
Sample [1.46]  expected [0.993868363411645] produced [0.8158257729438977]
Sample [1.48]  expected [0.99588084453764] produced [0.8192977312043652]
Epoch 1000 RMSE = 0.2167269981082878
Epoch 1100 RMSE = 0.2167269981082878
Epoch 1200 RMSE = 0.2167269981082878
Epoch 1300 RMSE = 0.2167269981082878
Epoch 1400 RMSE = 0.2167269981082878
Epoch 1500 RMSE = 0.2167269981082878
Epoch 1600 RMSE = 0.2167269981082878
Epoch 1700 RMSE = 0.2167269981082878
Epoch 1800 RMSE = 0.2167269981082878
Epoch 1900 RMSE = 0.2167269981082878
Sample [0.01]  expected [0.00999983333416666] produced [0.5025581870868067]
Sample [0.08]  expected [0.0799146939691727] produced [0.5204530243953066]
Sample [0.15]  expected [0.149438132473599] produced [0.538279287954869]
Sample [0.17]  expected [0.169182349066996] produced [0.5433284987672722]
Sample [0.21]  expected [0.2084598998461] produced [0.5534120255020681]
Sample [0.24]  expected [0.237702626427135] produced [0.5609182602465094]
Sample [0.34]  expected [0.333487092140814] produced [0.5857933667377648]
Sample [0.39]  expected [0.380188415123161] produced [0.5980068515232557]
Sample [0.41]  expected [0.398609327984423] produced [0.6027929466704265]
Sample [0.44]  expected [0.425939465066] produced [0.6099723016602646]
Sample [0.47]  expected [0.452886285379068] produced [0.6170944412993606]
Sample [0.48]  expected [0.461779175541483] produced [0.6193876533867572]
Sample [0.49]  expected [0.470625888171158] produced [0.6216735761804072]
Sample [0.53]  expected [0.505533341204847] produced [0.6310537263842201]
Sample [0.54]  expected [0.514135991653113] produced [0.6333113262338554]
Sample [0.56]  expected [0.531186197920883] produced [0.6379018978287102]
Sample [0.61]  expected [0.572867460100481] produced [0.6494011819929137]
Sample [0.66]  expected [0.613116851973434] produced [0.6607375160246768]
Sample [0.69]  expected [0.636537182221968] produced [0.6674421089088021]
Sample [0.8]  expected [0.717356090899523] produced [0.6915781603749126]
Sample [0.82]  expected [0.731145829726896] produced [0.6959055830950059]
Sample [0.83]  expected [0.737931371109963] produced [0.6980913725026462]
Sample [0.87]  expected [0.764328937025505] produced [0.7065994388166158]
Sample [0.9]  expected [0.783326909627483] produced [0.7129390734653931]
Sample [0.97]  expected [0.82488571333845] produced [0.7273234190498193]
Sample [1.1]  expected [0.891207360061435] produced [0.7528001562606748]
Sample [1.2]  expected [0.932039085967226] produced [0.7714539465313621]
Sample [1.36]  expected [0.977864602435316] produced [0.7991603570495699]
Sample [1.45]  expected [0.992712991037588] produced [0.8138580281774362]
Sample [1.46]  expected [0.993868363411645] produced [0.8158257729438977]
Sample [1.48]  expected [0.99588084453764] produced [0.8192977312043652]
Epoch 2000 RMSE = 0.2167269981082878
Epoch 2100 RMSE = 0.2167269981082878
Epoch 2200 RMSE = 0.2167269981082878
Epoch 2300 RMSE = 0.2167269981082878
Epoch 2400 RMSE = 0.2167269981082878
Epoch 2500 RMSE = 0.2167269981082878
Epoch 2600 RMSE = 0.2167269981082878
Epoch 2700 RMSE = 0.2167269981082878
Epoch 2800 RMSE = 0.2167269981082878
Epoch 2900 RMSE = 0.2167269981082878
Sample [0.01]  expected [0.00999983333416666] produced [0.5025581870868067]
Sample [0.08]  expected [0.0799146939691727] produced [0.5204530243953066]
Sample [0.15]  expected [0.149438132473599] produced [0.538279287954869]
Sample [0.17]  expected [0.169182349066996] produced [0.5433284987672722]
Sample [0.21]  expected [0.2084598998461] produced [0.5534120255020681]
Sample [0.24]  expected [0.237702626427135] produced [0.5609182602465094]
Sample [0.34]  expected [0.333487092140814] produced [0.5857933667377648]
Sample [0.39]  expected [0.380188415123161] produced [0.5980068515232557]
Sample [0.41]  expected [0.398609327984423] produced [0.6027929466704265]
Sample [0.44]  expected [0.425939465066] produced [0.6099723016602646]
Sample [0.47]  expected [0.452886285379068] produced [0.6170944412993606]
Sample [0.48]  expected [0.461779175541483] produced [0.6193876533867572]
Sample [0.49]  expected [0.470625888171158] produced [0.6216735761804072]
Sample [0.53]  expected [0.505533341204847] produced [0.6310537263842201]
Sample [0.54]  expected [0.514135991653113] produced [0.6333113262338554]
Sample [0.56]  expected [0.531186197920883] produced [0.6379018978287102]
Sample [0.61]  expected [0.572867460100481] produced [0.6494011819929137]
Sample [0.66]  expected [0.613116851973434] produced [0.6607375160246768]
Sample [0.69]  expected [0.636537182221968] produced [0.6674421089088021]
Sample [0.8]  expected [0.717356090899523] produced [0.6915781603749126]
Sample [0.82]  expected [0.731145829726896] produced [0.6959055830950059]
Sample [0.83]  expected [0.737931371109963] produced [0.6980913725026462]
Sample [0.87]  expected [0.764328937025505] produced [0.7065994388166158]
Sample [0.9]  expected [0.783326909627483] produced [0.7129390734653931]
Sample [0.97]  expected [0.82488571333845] produced [0.7273234190498193]
Sample [1.1]  expected [0.891207360061435] produced [0.7528001562606748]
Sample [1.2]  expected [0.932039085967226] produced [0.7714539465313621]
Sample [1.36]  expected [0.977864602435316] produced [0.7991603570495699]
Sample [1.45]  expected [0.992712991037588] produced [0.8138580281774362]
Sample [1.46]  expected [0.993868363411645] produced [0.8158257729438977]
Sample [1.48]  expected [0.99588084453764] produced [0.8192977312043652]
Epoch 3000 RMSE = 0.2167269981082878
Epoch 3100 RMSE = 0.2167269981082878
Epoch 3200 RMSE = 0.2167269981082878
Epoch 3300 RMSE = 0.2167269981082878
Epoch 3400 RMSE = 0.2167269981082878
Epoch 3500 RMSE = 0.2167269981082878
Epoch 3600 RMSE = 0.2167269981082878
Epoch 3700 RMSE = 0.2167269981082878
Epoch 3800 RMSE = 0.2167269981082878
Epoch 3900 RMSE = 0.2167269981082878
Sample [0.01]  expected [0.00999983333416666] produced [0.5025581870868067]
Sample [0.08]  expected [0.0799146939691727] produced [0.5204530243953066]
Sample [0.15]  expected [0.149438132473599] produced [0.538279287954869]
Sample [0.17]  expected [0.169182349066996] produced [0.5433284987672722]
Sample [0.21]  expected [0.2084598998461] produced [0.5534120255020681]
Sample [0.24]  expected [0.237702626427135] produced [0.5609182602465094]
Sample [0.34]  expected [0.333487092140814] produced [0.5857933667377648]
Sample [0.39]  expected [0.380188415123161] produced [0.5980068515232557]
Sample [0.41]  expected [0.398609327984423] produced [0.6027929466704265]
Sample [0.44]  expected [0.425939465066] produced [0.6099723016602646]
Sample [0.47]  expected [0.452886285379068] produced [0.6170944412993606]
Sample [0.48]  expected [0.461779175541483] produced [0.6193876533867572]
Sample [0.49]  expected [0.470625888171158] produced [0.6216735761804072]
Sample [0.53]  expected [0.505533341204847] produced [0.6310537263842201]
Sample [0.54]  expected [0.514135991653113] produced [0.6333113262338554]
Sample [0.56]  expected [0.531186197920883] produced [0.6379018978287102]
Sample [0.61]  expected [0.572867460100481] produced [0.6494011819929137]
Sample [0.66]  expected [0.613116851973434] produced [0.6607375160246768]
Sample [0.69]  expected [0.636537182221968] produced [0.6674421089088021]
Sample [0.8]  expected [0.717356090899523] produced [0.6915781603749126]
Sample [0.82]  expected [0.731145829726896] produced [0.6959055830950059]
Sample [0.83]  expected [0.737931371109963] produced [0.6980913725026462]
Sample [0.87]  expected [0.764328937025505] produced [0.7065994388166158]
Sample [0.9]  expected [0.783326909627483] produced [0.7129390734653931]
Sample [0.97]  expected [0.82488571333845] produced [0.7273234190498193]
Sample [1.1]  expected [0.891207360061435] produced [0.7528001562606748]
Sample [1.2]  expected [0.932039085967226] produced [0.7714539465313621]
Sample [1.36]  expected [0.977864602435316] produced [0.7991603570495699]
Sample [1.45]  expected [0.992712991037588] produced [0.8138580281774362]
Sample [1.46]  expected [0.993868363411645] produced [0.8158257729438977]
Sample [1.48]  expected [0.99588084453764] produced [0.8192977312043652]
Epoch 4000 RMSE = 0.2167269981082878
Epoch 4100 RMSE = 0.2167269981082878
Epoch 4200 RMSE = 0.2167269981082878
Epoch 4300 RMSE = 0.2167269981082878
Epoch 4400 RMSE = 0.2167269981082878
Epoch 4500 RMSE = 0.2167269981082878
Epoch 4600 RMSE = 0.2167269981082878
Epoch 4700 RMSE = 0.2167269981082878
Epoch 4800 RMSE = 0.2167269981082878
Epoch 4900 RMSE = 0.2167269981082878
Sample [0.01]  expected [0.00999983333416666] produced [0.5025581870868067]
Sample [0.08]  expected [0.0799146939691727] produced [0.5204530243953066]
Sample [0.15]  expected [0.149438132473599] produced [0.538279287954869]
Sample [0.17]  expected [0.169182349066996] produced [0.5433284987672722]
Sample [0.21]  expected [0.2084598998461] produced [0.5534120255020681]
Sample [0.24]  expected [0.237702626427135] produced [0.5609182602465094]
Sample [0.34]  expected [0.333487092140814] produced [0.5857933667377648]
Sample [0.39]  expected [0.380188415123161] produced [0.5980068515232557]
Sample [0.41]  expected [0.398609327984423] produced [0.6027929466704265]
Sample [0.44]  expected [0.425939465066] produced [0.6099723016602646]
Sample [0.47]  expected [0.452886285379068] produced [0.6170944412993606]
Sample [0.48]  expected [0.461779175541483] produced [0.6193876533867572]
Sample [0.49]  expected [0.470625888171158] produced [0.6216735761804072]
Sample [0.53]  expected [0.505533341204847] produced [0.6310537263842201]
Sample [0.54]  expected [0.514135991653113] produced [0.6333113262338554]
Sample [0.56]  expected [0.531186197920883] produced [0.6379018978287102]
Sample [0.61]  expected [0.572867460100481] produced [0.6494011819929137]
Sample [0.66]  expected [0.613116851973434] produced [0.6607375160246768]
Sample [0.69]  expected [0.636537182221968] produced [0.6674421089088021]
Sample [0.8]  expected [0.717356090899523] produced [0.6915781603749126]
Sample [0.82]  expected [0.731145829726896] produced [0.6959055830950059]
Sample [0.83]  expected [0.737931371109963] produced [0.6980913725026462]
Sample [0.87]  expected [0.764328937025505] produced [0.7065994388166158]
Sample [0.9]  expected [0.783326909627483] produced [0.7129390734653931]
Sample [0.97]  expected [0.82488571333845] produced [0.7273234190498193]
Sample [1.1]  expected [0.891207360061435] produced [0.7528001562606748]
Sample [1.2]  expected [0.932039085967226] produced [0.7714539465313621]
Sample [1.36]  expected [0.977864602435316] produced [0.7991603570495699]
Sample [1.45]  expected [0.992712991037588] produced [0.8138580281774362]
Sample [1.46]  expected [0.993868363411645] produced [0.8158257729438977]
Sample [1.48]  expected [0.99588084453764] produced [0.8192977312043652]
Epoch 5000 RMSE = 0.2167269981082878
Epoch 5100 RMSE = 0.2167269981082878
Epoch 5200 RMSE = 0.2167269981082878
Epoch 5300 RMSE = 0.2167269981082878
Epoch 5400 RMSE = 0.2167269981082878
Epoch 5500 RMSE = 0.2167269981082878
Epoch 5600 RMSE = 0.2167269981082878
Epoch 5700 RMSE = 0.2167269981082878
Epoch 5800 RMSE = 0.2167269981082878
Epoch 5900 RMSE = 0.2167269981082878
Sample [0.01]  expected [0.00999983333416666] produced [0.5025581870868067]
Sample [0.08]  expected [0.0799146939691727] produced [0.5204530243953066]
Sample [0.15]  expected [0.149438132473599] produced [0.538279287954869]
Sample [0.17]  expected [0.169182349066996] produced [0.5433284987672722]
Sample [0.21]  expected [0.2084598998461] produced [0.5534120255020681]
Sample [0.24]  expected [0.237702626427135] produced [0.5609182602465094]
Sample [0.34]  expected [0.333487092140814] produced [0.5857933667377648]
Sample [0.39]  expected [0.380188415123161] produced [0.5980068515232557]
Sample [0.41]  expected [0.398609327984423] produced [0.6027929466704265]
Sample [0.44]  expected [0.425939465066] produced [0.6099723016602646]
Sample [0.47]  expected [0.452886285379068] produced [0.6170944412993606]
Sample [0.48]  expected [0.461779175541483] produced [0.6193876533867572]
Sample [0.49]  expected [0.470625888171158] produced [0.6216735761804072]
Sample [0.53]  expected [0.505533341204847] produced [0.6310537263842201]
Sample [0.54]  expected [0.514135991653113] produced [0.6333113262338554]
Sample [0.56]  expected [0.531186197920883] produced [0.6379018978287102]
Sample [0.61]  expected [0.572867460100481] produced [0.6494011819929137]
Sample [0.66]  expected [0.613116851973434] produced [0.6607375160246768]
Sample [0.69]  expected [0.636537182221968] produced [0.6674421089088021]
Sample [0.8]  expected [0.717356090899523] produced [0.6915781603749126]
Sample [0.82]  expected [0.731145829726896] produced [0.6959055830950059]
Sample [0.83]  expected [0.737931371109963] produced [0.6980913725026462]
Sample [0.87]  expected [0.764328937025505] produced [0.7065994388166158]
Sample [0.9]  expected [0.783326909627483] produced [0.7129390734653931]
Sample [0.97]  expected [0.82488571333845] produced [0.7273234190498193]
Sample [1.1]  expected [0.891207360061435] produced [0.7528001562606748]
Sample [1.2]  expected [0.932039085967226] produced [0.7714539465313621]
Sample [1.36]  expected [0.977864602435316] produced [0.7991603570495699]
Sample [1.45]  expected [0.992712991037588] produced [0.8138580281774362]
Sample [1.46]  expected [0.993868363411645] produced [0.8158257729438977]
Sample [1.48]  expected [0.99588084453764] produced [0.8192977312043652]
Epoch 6000 RMSE = 0.2167269981082878
Epoch 6100 RMSE = 0.2167269981082878
Epoch 6200 RMSE = 0.2167269981082878
Epoch 6300 RMSE = 0.2167269981082878
Epoch 6400 RMSE = 0.2167269981082878
Epoch 6500 RMSE = 0.2167269981082878
Epoch 6600 RMSE = 0.2167269981082878
Epoch 6700 RMSE = 0.2167269981082878
Epoch 6800 RMSE = 0.2167269981082878
Epoch 6900 RMSE = 0.2167269981082878
Sample [0.01]  expected [0.00999983333416666] produced [0.5025581870868067]
Sample [0.08]  expected [0.0799146939691727] produced [0.5204530243953066]
Sample [0.15]  expected [0.149438132473599] produced [0.538279287954869]
Sample [0.17]  expected [0.169182349066996] produced [0.5433284987672722]
Sample [0.21]  expected [0.2084598998461] produced [0.5534120255020681]
Sample [0.24]  expected [0.237702626427135] produced [0.5609182602465094]
Sample [0.34]  expected [0.333487092140814] produced [0.5857933667377648]
Sample [0.39]  expected [0.380188415123161] produced [0.5980068515232557]
Sample [0.41]  expected [0.398609327984423] produced [0.6027929466704265]
Sample [0.44]  expected [0.425939465066] produced [0.6099723016602646]
Sample [0.47]  expected [0.452886285379068] produced [0.6170944412993606]
Sample [0.48]  expected [0.461779175541483] produced [0.6193876533867572]
Sample [0.49]  expected [0.470625888171158] produced [0.6216735761804072]
Sample [0.53]  expected [0.505533341204847] produced [0.6310537263842201]
Sample [0.54]  expected [0.514135991653113] produced [0.6333113262338554]
Sample [0.56]  expected [0.531186197920883] produced [0.6379018978287102]
Sample [0.61]  expected [0.572867460100481] produced [0.6494011819929137]
Sample [0.66]  expected [0.613116851973434] produced [0.6607375160246768]
Sample [0.69]  expected [0.636537182221968] produced [0.6674421089088021]
Sample [0.8]  expected [0.717356090899523] produced [0.6915781603749126]
Sample [0.82]  expected [0.731145829726896] produced [0.6959055830950059]
Sample [0.83]  expected [0.737931371109963] produced [0.6980913725026462]
Sample [0.87]  expected [0.764328937025505] produced [0.7065994388166158]
Sample [0.9]  expected [0.783326909627483] produced [0.7129390734653931]
Sample [0.97]  expected [0.82488571333845] produced [0.7273234190498193]
Sample [1.1]  expected [0.891207360061435] produced [0.7528001562606748]
Sample [1.2]  expected [0.932039085967226] produced [0.7714539465313621]
Sample [1.36]  expected [0.977864602435316] produced [0.7991603570495699]
Sample [1.45]  expected [0.992712991037588] produced [0.8138580281774362]
Sample [1.46]  expected [0.993868363411645] produced [0.8158257729438977]
Sample [1.48]  expected [0.99588084453764] produced [0.8192977312043652]
Epoch 7000 RMSE = 0.2167269981082878
Epoch 7100 RMSE = 0.2167269981082878
Epoch 7200 RMSE = 0.2167269981082878
Epoch 7300 RMSE = 0.2167269981082878
Epoch 7400 RMSE = 0.2167269981082878
Epoch 7500 RMSE = 0.2167269981082878
Epoch 7600 RMSE = 0.2167269981082878
Epoch 7700 RMSE = 0.2167269981082878
Epoch 7800 RMSE = 0.2167269981082878
Epoch 7900 RMSE = 0.2167269981082878
Sample [0.01]  expected [0.00999983333416666] produced [0.5025581870868067]
Sample [0.08]  expected [0.0799146939691727] produced [0.5204530243953066]
Sample [0.15]  expected [0.149438132473599] produced [0.538279287954869]
Sample [0.17]  expected [0.169182349066996] produced [0.5433284987672722]
Sample [0.21]  expected [0.2084598998461] produced [0.5534120255020681]
Sample [0.24]  expected [0.237702626427135] produced [0.5609182602465094]
Sample [0.34]  expected [0.333487092140814] produced [0.5857933667377648]
Sample [0.39]  expected [0.380188415123161] produced [0.5980068515232557]
Sample [0.41]  expected [0.398609327984423] produced [0.6027929466704265]
Sample [0.44]  expected [0.425939465066] produced [0.6099723016602646]
Sample [0.47]  expected [0.452886285379068] produced [0.6170944412993606]
Sample [0.48]  expected [0.461779175541483] produced [0.6193876533867572]
Sample [0.49]  expected [0.470625888171158] produced [0.6216735761804072]
Sample [0.53]  expected [0.505533341204847] produced [0.6310537263842201]
Sample [0.54]  expected [0.514135991653113] produced [0.6333113262338554]
Sample [0.56]  expected [0.531186197920883] produced [0.6379018978287102]
Sample [0.61]  expected [0.572867460100481] produced [0.6494011819929137]
Sample [0.66]  expected [0.613116851973434] produced [0.6607375160246768]
Sample [0.69]  expected [0.636537182221968] produced [0.6674421089088021]
Sample [0.8]  expected [0.717356090899523] produced [0.6915781603749126]
Sample [0.82]  expected [0.731145829726896] produced [0.6959055830950059]
Sample [0.83]  expected [0.737931371109963] produced [0.6980913725026462]
Sample [0.87]  expected [0.764328937025505] produced [0.7065994388166158]
Sample [0.9]  expected [0.783326909627483] produced [0.7129390734653931]
Sample [0.97]  expected [0.82488571333845] produced [0.7273234190498193]
Sample [1.1]  expected [0.891207360061435] produced [0.7528001562606748]
Sample [1.2]  expected [0.932039085967226] produced [0.7714539465313621]
Sample [1.36]  expected [0.977864602435316] produced [0.7991603570495699]
Sample [1.45]  expected [0.992712991037588] produced [0.8138580281774362]
Sample [1.46]  expected [0.993868363411645] produced [0.8158257729438977]
Sample [1.48]  expected [0.99588084453764] produced [0.8192977312043652]
Epoch 8000 RMSE = 0.2167269981082878
Epoch 8100 RMSE = 0.2167269981082878
Epoch 8200 RMSE = 0.2167269981082878
Epoch 8300 RMSE = 0.2167269981082878
Epoch 8400 RMSE = 0.2167269981082878
Epoch 8500 RMSE = 0.2167269981082878
Epoch 8600 RMSE = 0.2167269981082878
Epoch 8700 RMSE = 0.2167269981082878
Epoch 8800 RMSE = 0.2167269981082878
Epoch 8900 RMSE = 0.2167269981082878
Sample [0.01]  expected [0.00999983333416666] produced [0.5025581870868067]
Sample [0.08]  expected [0.0799146939691727] produced [0.5204530243953066]
Sample [0.15]  expected [0.149438132473599] produced [0.538279287954869]
Sample [0.17]  expected [0.169182349066996] produced [0.5433284987672722]
Sample [0.21]  expected [0.2084598998461] produced [0.5534120255020681]
Sample [0.24]  expected [0.237702626427135] produced [0.5609182602465094]
Sample [0.34]  expected [0.333487092140814] produced [0.5857933667377648]
Sample [0.39]  expected [0.380188415123161] produced [0.5980068515232557]
Sample [0.41]  expected [0.398609327984423] produced [0.6027929466704265]
Sample [0.44]  expected [0.425939465066] produced [0.6099723016602646]
Sample [0.47]  expected [0.452886285379068] produced [0.6170944412993606]
Sample [0.48]  expected [0.461779175541483] produced [0.6193876533867572]
Sample [0.49]  expected [0.470625888171158] produced [0.6216735761804072]
Sample [0.53]  expected [0.505533341204847] produced [0.6310537263842201]
Sample [0.54]  expected [0.514135991653113] produced [0.6333113262338554]
Sample [0.56]  expected [0.531186197920883] produced [0.6379018978287102]
Sample [0.61]  expected [0.572867460100481] produced [0.6494011819929137]
Sample [0.66]  expected [0.613116851973434] produced [0.6607375160246768]
Sample [0.69]  expected [0.636537182221968] produced [0.6674421089088021]
Sample [0.8]  expected [0.717356090899523] produced [0.6915781603749126]
Sample [0.82]  expected [0.731145829726896] produced [0.6959055830950059]
Sample [0.83]  expected [0.737931371109963] produced [0.6980913725026462]
Sample [0.87]  expected [0.764328937025505] produced [0.7065994388166158]
Sample [0.9]  expected [0.783326909627483] produced [0.7129390734653931]
Sample [0.97]  expected [0.82488571333845] produced [0.7273234190498193]
Sample [1.1]  expected [0.891207360061435] produced [0.7528001562606748]
Sample [1.2]  expected [0.932039085967226] produced [0.7714539465313621]
Sample [1.36]  expected [0.977864602435316] produced [0.7991603570495699]
Sample [1.45]  expected [0.992712991037588] produced [0.8138580281774362]
Sample [1.46]  expected [0.993868363411645] produced [0.8158257729438977]
Sample [1.48]  expected [0.99588084453764] produced [0.8192977312043652]
Epoch 9000 RMSE = 0.2167269981082878
Epoch 9100 RMSE = 0.2167269981082878
Epoch 9200 RMSE = 0.2167269981082878
Epoch 9300 RMSE = 0.2167269981082878
Epoch 9400 RMSE = 0.2167269981082878
Epoch 9500 RMSE = 0.2167269981082878
Epoch 9600 RMSE = 0.2167269981082878
Epoch 9700 RMSE = 0.2167269981082878
Epoch 9800 RMSE = 0.2167269981082878
Epoch 9900 RMSE = 0.2167269981082878
Sample [0.01]  expected [0.00999983333416666] produced [0.5025581870868067]
Sample [0.08]  expected [0.0799146939691727] produced [0.5204530243953066]
Sample [0.15]  expected [0.149438132473599] produced [0.538279287954869]
Sample [0.17]  expected [0.169182349066996] produced [0.5433284987672722]
Sample [0.21]  expected [0.2084598998461] produced [0.5534120255020681]
Sample [0.24]  expected [0.237702626427135] produced [0.5609182602465094]
Sample [0.34]  expected [0.333487092140814] produced [0.5857933667377648]
Sample [0.39]  expected [0.380188415123161] produced [0.5980068515232557]
Sample [0.41]  expected [0.398609327984423] produced [0.6027929466704265]
Sample [0.44]  expected [0.425939465066] produced [0.6099723016602646]
Sample [0.47]  expected [0.452886285379068] produced [0.6170944412993606]
Sample [0.48]  expected [0.461779175541483] produced [0.6193876533867572]
Sample [0.49]  expected [0.470625888171158] produced [0.6216735761804072]
Sample [0.53]  expected [0.505533341204847] produced [0.6310537263842201]
Sample [0.54]  expected [0.514135991653113] produced [0.6333113262338554]
Sample [0.56]  expected [0.531186197920883] produced [0.6379018978287102]
Sample [0.61]  expected [0.572867460100481] produced [0.6494011819929137]
Sample [0.66]  expected [0.613116851973434] produced [0.6607375160246768]
Sample [0.69]  expected [0.636537182221968] produced [0.6674421089088021]
Sample [0.8]  expected [0.717356090899523] produced [0.6915781603749126]
Sample [0.82]  expected [0.731145829726896] produced [0.6959055830950059]
Sample [0.83]  expected [0.737931371109963] produced [0.6980913725026462]
Sample [0.87]  expected [0.764328937025505] produced [0.7065994388166158]
Sample [0.9]  expected [0.783326909627483] produced [0.7129390734653931]
Sample [0.97]  expected [0.82488571333845] produced [0.7273234190498193]
Sample [1.1]  expected [0.891207360061435] produced [0.7528001562606748]
Sample [1.2]  expected [0.932039085967226] produced [0.7714539465313621]
Sample [1.36]  expected [0.977864602435316] produced [0.7991603570495699]
Sample [1.45]  expected [0.992712991037588] produced [0.8138580281774362]
Sample [1.46]  expected [0.993868363411645] produced [0.8158257729438977]
Sample [1.48]  expected [0.99588084453764] produced [0.8192977312043652]
Epoch 10000 RMSE = 0.2167269981082878
Final Epoch RMSE = 0.2167269981082878
TESTING
Sample [0.0]  expected [0.0] produced [0.5]
Sample [0.02]  expected [0.0199986666933331] produced [0.5051162402443359]
Sample [0.03]  expected [0.0299955002024957] produced [0.5076731162853619]
Sample [0.04]  expected [0.0399893341866342] produced [0.5102284069438541]
Sample [0.05]  expected [0.0499791692706783] produced [0.5127815696283122]
Sample [0.06]  expected [0.0599640064794446] produced [0.5153320816451393]
Sample [0.07]  expected [0.0699428473375328] produced [0.5178794404759947]
Sample [0.09]  expected [0.089878549198011] produced [0.5229726648604256]
Sample [0.1]  expected [0.0998334166468282] produced [0.5255088457539987]
Sample [0.11]  expected [0.109778300837175] produced [0.5280400678199753]
Sample [0.12]  expected [0.119712207288919] produced [0.5305659323222517]
Sample [0.13]  expected [0.129634142619695] produced [0.5330860617175733]
Sample [0.14]  expected [0.139543114644236] produced [0.5356000997153805]
Sample [0.16]  expected [0.159318206614246] produced [0.5406373723500882]
Sample [0.18]  expected [0.179029573425824] produced [0.5456564812187802]
Sample [0.19]  expected [0.188858894976501] produced [0.5481391403634298]
Sample [0.2]  expected [0.198669330795061] produced [0.5506140042655625]
Sample [0.22]  expected [0.218229623080869] produced [0.5555881171536406]
Sample [0.23]  expected [0.227977523535188] produced [0.5580405681855939]
Sample [0.25]  expected [0.247403959254523] produced [0.5629784818736423]
Sample [0.26]  expected [0.257080551892155] produced [0.5654072776934874]
Sample [0.27]  expected [0.266731436688831] produced [0.5678272203692699]
Sample [0.28]  expected [0.276355648564114] produced [0.5702382392880757]
Sample [0.29]  expected [0.285952225104836] produced [0.5726402824810976]
Sample [0.3]  expected [0.29552020666134] produced [0.5750333162441541]
Sample [0.31]  expected [0.305058636443443] produced [0.5774173247372383]
Sample [0.32]  expected [0.314566560616118] produced [0.5797923095640037]
Sample [0.33]  expected [0.324043028394868] produced [0.5821582893321037]
Sample [0.35]  expected [0.342897807455451] produced [0.5869513346222803]
Sample [0.36]  expected [0.35227423327509] produced [0.5892929059774511]
Sample [0.37]  expected [0.361615431964962] produced [0.5916256923142899]
Sample [0.38]  expected [0.370920469412983] produced [0.5939497896658903]
Sample [0.4]  expected [0.389418342308651] produced [0.5986698577031215]
Sample [0.42]  expected [0.40776045305957] produced [0.6033635703088769]
Sample [0.43]  expected [0.416870802429211] produced [0.6056504423243925]
Sample [0.45]  expected [0.43496553411123] produced [0.6103028889131531]
Sample [0.46]  expected [0.44394810696552] produced [0.6125686951869271]
Sample [0.5]  expected [0.479425538604203] produced [0.6218714228310974]
Sample [0.51]  expected [0.488177246882907] produced [0.6241080972734804]
Sample [0.52]  expected [0.496880137843737] produced [0.6263385196086442]
Sample [0.55]  expected [0.522687228930659] produced [0.6331856720412479]
Sample [0.57]  expected [0.539632048733969] produced [0.6376910928100876]
Sample [0.58]  expected [0.548023936791874] produced [0.6398932487580697]
Sample [0.59]  expected [0.556361022912784] produced [0.6420908725394039]
Sample [0.6]  expected [0.564642473395035] produced [0.6442842251963129]
Sample [0.62]  expected [0.581035160537305] produced [0.6487316561674452]
Sample [0.63]  expected [0.58914475794227] produced [0.6509146523630767]
Sample [0.64]  expected [0.597195441362392] produced [0.6530944045380054]
Sample [0.65]  expected [0.60518640573604] produced [0.6552711685101962]
Sample [0.67]  expected [0.62098598703656] produced [0.6596662970026264]
Sample [0.68]  expected [0.628793024018469] produced [0.6618361291451481]
Sample [0.7]  expected [0.644217687237691] produced [0.6662029105534243]
Sample [0.71]  expected [0.651833771021537] produced [0.6683677366061564]
Sample [0.72]  expected [0.659384671971473] produced [0.67053123415293]
Sample [0.73]  expected [0.666869635003698] produced [0.6726936189318061]
Sample [0.74]  expected [0.674287911628145] produced [0.6748550991397771]
Sample [0.75]  expected [0.681638760023334] produced [0.677015874903971]
Sample [0.76]  expected [0.688921445110551] produced [0.6791761377675004]
Sample [0.77]  expected [0.696135238627357] produced [0.6813360701908647]
Sample [0.78]  expected [0.70327941920041] produced [0.6834958450698162]
Sample [0.79]  expected [0.710353272417608] produced [0.6856556252706106]
Sample [0.81]  expected [0.724287174370143] produced [0.6899318367294789]
Sample [0.84]  expected [0.744643119970859] produced [0.696285767457778]
Sample [0.85]  expected [0.751280405140293] produced [0.6984473266639365]
Sample [0.86]  expected [0.757842562895277] produced [0.7006097519086507]
Sample [0.88]  expected [0.770738878898969] produced [0.7048350729886305]
Sample [0.89]  expected [0.777071747526824] produced [0.7069999305434467]
Sample [0.91]  expected [0.78950373968995] produced [0.7112042728148699]
Sample [0.92]  expected [0.795601620036366] produced [0.7133718364564755]
Sample [0.93]  expected [0.801619940883777] produced [0.7155405099475447]
Sample [0.94]  expected [0.807558100405114] produced [0.7177102650840292]
Sample [0.95]  expected [0.813415504789374] produced [0.7198810592295338]
Sample [0.96]  expected [0.819191568300998] produced [0.7220528352729345]
Sample [0.98]  expected [0.83049737049197] produced [0.7262090817664969]
Sample [0.99]  expected [0.836025978600521] produced [0.7283827804678509]
Sample [1.0]  expected [0.841470984807897] produced [0.7305571291050322]
Sample [1.01]  expected [0.846831844618015] produced [0.7327319987211324]
Sample [1.02]  expected [0.852108021949363] produced [0.7349072462542131]
Sample [1.03]  expected [0.857298989188603] produced [0.7370827147125971]
Sample [1.04]  expected [0.862404227243338] produced [0.7392582333811856]
Sample [1.05]  expected [0.867423225594017] produced [0.74143361805847]
Sample [1.06]  expected [0.872355482344986] produced [0.7436086713238009]
Sample [1.07]  expected [0.877200504274682] produced [0.7457831828343753]
Sample [1.08]  expected [0.881957806884948] produced [0.7479569296513066]
Sample [1.09]  expected [0.886626914449487] produced [0.7501296765940383]
Sample [1.11]  expected [0.895698685680048] produced [0.7541783229767388]
Sample [1.12]  expected [0.900100442176505] produced [0.7563465329537221]
Sample [1.13]  expected [0.904412189378826] produced [0.7585127478144974]
Sample [1.14]  expected [0.908633496115883] produced [0.7606766769039126]
Sample [1.15]  expected [0.912763940260521] produced [0.7628380199424704]
Sample [1.16]  expected [0.916803108771767] produced [0.764996467550622]
Sample [1.17]  expected [0.920750597736136] produced [0.767151701792212]
Sample [1.18]  expected [0.92460601240802] produced [0.769303396735533]
Sample [1.19]  expected [0.928368967249167] produced [0.7714512190303903]
Sample [1.21]  expected [0.935616001553386] produced [0.7753831828843805]
Sample [1.22]  expected [0.939099356319068] produced [0.7775179961528149]
Sample [1.23]  expected [0.942488801931697] produced [0.7796476047385925]
Sample [1.24]  expected [0.945783999449539] produced [0.781771646106381]
Sample [1.25]  expected [0.948984619355586] produced [0.7838897539539317]
Sample [1.26]  expected [0.952090341590516] produced [0.7860015588540507]
Sample [1.27]  expected [0.955100855584692] produced [0.7881066888983571]
Sample [1.28]  expected [0.958015860289225] produced [0.7902047703409656]
Sample [1.29]  expected [0.960835064206073] produced [0.7922954282402361]
Sample [1.3]  expected [0.963558185417193] produced [0.7943782870967677]
Sample [1.31]  expected [0.966184951612734] produced [0.7964529714858278]
Sample [1.32]  expected [0.968715100118265] produced [0.7985191066824642]
Sample [1.33]  expected [0.971148377921045] produced [0.8005763192775767]
Sample [1.34]  expected [0.973484541695319] produced [0.8026242377832948]
Sample [1.35]  expected [0.975723357826659] produced [0.8046624932260542]
Sample [1.37]  expected [0.979908061398614] produced [0.8083235666718164]
Sample [1.38]  expected [0.98185353037236] produced [0.8103324126203766]
Sample [1.39]  expected [0.983700814811277] produced [0.8123302020362201]
Sample [1.4]  expected [0.98544972998846] produced [0.8143165848840732]
Sample [1.41]  expected [0.98710010101385] produced [0.8162912168493826]
Sample [1.42]  expected [0.98865176285172] produced [0.8182537598100545]
Sample [1.43]  expected [0.990104560337178] produced [0.8202038822859695]
Sample [1.44]  expected [0.991458348191686] produced [0.8221412598654411]
Sample [1.47]  expected [0.994924349777581] produced [0.8271321772745899]
Sample [1.49]  expected [0.996737752043143] produced [0.8305264842940959]
Sample [1.5]  expected [0.997494986604054] produced [0.8323876475241302]
Sample [1.51]  expected [0.998152472497548] produced [0.8342340902471277]
Sample [1.52]  expected [0.998710143975583] produced [0.836065551001605]
Sample [1.53]  expected [0.999167945271476] produced [0.8378817781708346]
Sample [1.54]  expected [0.999525830605479] produced [0.8396825301628758]
Sample [1.55]  expected [0.999783764189357] produced [0.841467575564823]
Sample [1.56]  expected [0.999941720229966] produced [0.84323669327161]
Sample [1.57]  expected [0.999999682931835] produced [0.8449896725897661]
RMSE is 0.21661494397008027

Process finished with exit code 0

"""



