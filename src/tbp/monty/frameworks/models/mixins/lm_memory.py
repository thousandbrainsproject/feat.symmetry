# Copyright 2025-2026 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import logging

import numpy as np

from tbp.monty.memento import Memento

__all__ = ["LMMemoryMixin"]

logger = logging.getLogger(__name__)


class LMMemoryMixin:
    """Storage and read access for an LM's object models, keyed by object and channel.

    Mixin shared by memories that differ only in how they build and update models
    (GraphMemory, EvidenceGraphMemory). It holds no build or update logic, so a
    memory that uses it has nothing algorithmic to override. Provides the read API
    described by ObjectMemory.

    All methods below were moved from GraphMemory unchanged; only __init__'s
    storage setup and the features_to_use declaration are new here.
    """

    # Feature tolerances per input channel. Set by the owning LM after construction.
    features_to_use: dict[str, dict]

    def __init__(self) -> None:
        """Initialize empty storage and feature-array caches."""
        self.models_in_memory = {}

        # Array representation of features for each graph -> faster matching
        self.feature_array = {}
        self.feature_order = {}  # Order in which features are stored in feature_array

    # =============== Public Interface Functions ===============

    def initialize_feature_arrays(self):
        for graph_id in self.get_memory_ids():
            if graph_id not in self.feature_array:
                self.feature_array[graph_id] = {}
                self.feature_order[graph_id] = {}
            for input_channel in self.get_input_channels_in_graph(graph_id):
                (
                    self.feature_array[graph_id][input_channel],
                    self.feature_order[graph_id][input_channel],
                ) = self._get_all_node_features(graph_id, input_channel)

    # ------------------ Getters & Setters ---------------------
    def get_graph(self, graph_id, input_channel=None):
        """Return graph from graph memory.

        Args:
            graph_id: id of graph to retrieve
            input_channel: ?

        Raises:
            ValueError: If input_channel is defined, not "first", and not in the graph
        """
        if input_channel is None:
            return self.models_in_memory[graph_id]

        if input_channel == "first":
            # Arbitrarily take first input channel. Mostly used as placeholder for now.
            # Usually this will be input from a sensor module but we do nothing to
            # guarantee this.
            first_channel = self.get_input_channels_in_graph(graph_id)[0]
            return self.models_in_memory[graph_id][first_channel]

        if input_channel in self.get_input_channels_in_graph(graph_id):
            return self.models_in_memory[graph_id][input_channel]

        raise ValueError(f"{graph_id} has no data stored for {input_channel}.")

    def get_feature_array(self, graph_id):
        return self.feature_array[graph_id]

    def get_feature_order(self, graph_id):
        return self.feature_order[graph_id]

    def get_locations_in_graph(self, graph_id, input_channel):
        return self.get_graph(graph_id, input_channel).pos

    def get_all_models_in_memory(self):
        """Return models stored in memory."""
        return self.models_in_memory.copy()

    def get_memory_ids(self):
        """Get list of all objects in memory.

        Returns:
            List of all objects in memory.
        """
        return list(self.models_in_memory.keys())

    def get_input_channels_in_graph(self, graph_id):
        return list(self.models_in_memory[graph_id].keys())

    def get_graph_node_ids(self, graph_id, input_channel):
        num_nodes = self.models_in_memory[graph_id][input_channel].x.shape[0]
        return np.linspace(0, num_nodes - 1, num_nodes, dtype=int)

    def get_num_nodes_in_graph(self, graph_id, input_channel=None):
        """Get number of nodes in graph.

        If input_channel is None, return sum over all input channels for this object.

        Returns:
            Number of nodes in graph.
        """
        if input_channel is not None:
            return self.models_in_memory[graph_id][input_channel].x.shape[0]

        return sum(
            self.get_num_nodes_in_graph(graph_id, input_channel)
            for input_channel in self.get_input_channels_in_graph(graph_id)
        )

    def get_features_at_node(self, graph_id, input_channel, node_id, feature_keys=None):
        """Get features at a specific node in the graph.

        Args:
            graph_id: Name of graph.
            input_channel: Input channel.
            node_id: Node ID of the node to get features from. Can also be an
                array of node IDs to return an array of features.
            feature_keys: Feature keys.

        Returns:
            Dict of features at this node.

        TODO: look into getting node_id > graph.x.shape[0] (by 1)
        """
        if feature_keys is None:
            feature_keys = self.features_to_use[input_channel]
        node_features = {}
        graph = self.get_graph(graph_id, input_channel)
        if graph is None:
            logger.debug(
                f"{input_channel} not stored in graph {graph_id} yet. "
                "-> Input not used for matching."
            )
        else:
            for key in feature_keys:
                key_ids = graph.feature_mapping[key]
                feature = graph.x[node_id, key_ids[0] : key_ids[1]]
                node_features[key] = feature
        return node_features

    def __len__(self):
        """Return number of graphs in memory."""
        return len(self.get_memory_ids())

    # ------------------ Logging & Saving ----------------------
    def state_dict(self) -> Memento:
        return self.models_in_memory

    def load_state_dict(self, memento: Memento) -> None:
        logger.info("loading models")
        for obj_name, model in memento.items():
            logger.info(f"loading {obj_name} with features from {model.keys()}")
            # Add loaded graph to memory
            self._add_graph_to_memory(model, obj_name)

    # ======================= Private ==========================

    # ------------------- Main Algorithm -----------------------
    def _add_graph_to_memory(self, model, graph_id):
        """Add a loaded model to memory as is.

        Memories that need to convert or repair loaded models (e.g.
        EvidenceGraphMemory) override this.

        Args:
            model: Per-channel object models to add, keyed by input channel.
            graph_id: ID of the object the models belong to.
        """
        logger.info(f"Loaded {graph_id} with channels {list(model.keys())}")

        self.models_in_memory[graph_id] = model

    def remove_graph_from_memory(self, graph_id):
        self.models_in_memory.pop(graph_id)

    # ------------------------ Helper --------------------------

    def _get_all_node_features(
        self, graph_id, input_channel
    ) -> tuple[np.ndarray, list]:
        """Create an array of all features for all nodes in a graph.

        This can be used for fast feature matching

        Args:
            graph_id: The graph descriptor e.g. 'mug'
            input_channel: ?

        Returns:
            An array, num_nodes x num_features
        """
        all_node_ids = self.get_graph_node_ids(graph_id, input_channel).astype(int)
        feature_arrays = self._get_empty_feature_arrays(
            graph_id, input_channel, len(all_node_ids)
        )
        feature_order = []
        # TODO: This should be possible without this for loop (currently 3rd slowest).
        for i, node_id in enumerate(all_node_ids):
            node_features = self.get_features_at_node(graph_id, input_channel, node_id)
            start_idx = 0
            for feature in node_features:
                if feature in [
                    "pose_vectors",
                    "pose_fully_defined",
                ]:
                    continue
                if i == 0:
                    # Store order in which features are put in array to match
                    # correctly later
                    feature_order.append(feature)
                end_idx = start_idx + len(node_features[feature])
                feature_arrays[node_id, start_idx:end_idx] = node_features[feature]
                start_idx = end_idx
        return feature_arrays, feature_order

    def _get_empty_feature_arrays(
        self, graph_id, input_channel, num_nodes
    ) -> np.ndarray:
        """Get nan array with space for all features per input channel.

        The size of the array is calculated by taking the length of all non-pose
        features stored in the graph and adding them up. This way we can turn the
        features in the form of a nested dict into an array for more efficient matrix
        operations.

        Args:
            graph_id: Graph for which to generate this array (looks at features
                stored in this graph to determine array size)
            input_channel: ?
            num_nodes: Number of nodes that will need to be stored in this array
                (determines size of array)

        Returns:
            An array filled with nans of size (sum(feature_lens), num_nodes)
        """
        node_features = self.get_features_at_node(graph_id, input_channel, node_id=0)
        feature_array_len = 0
        for feature in node_features:
            if feature in [
                "pose_vectors",
                "pose_fully_defined",
            ]:
                continue
            feature_array_len += len(node_features[feature])
        return np.zeros((num_nodes, feature_array_len)) * np.nan

    def _extract_entries_with_content(self, features, locations):
        """Only keep features & locations at steps where information was received.

        Get only the features & locations at steps where information for this input
        channel was received.

        Returns:
            Features and locations with missing features removed.
        """
        # NOTE: Could use any feature here but using pose_fully_defined since it
        # is one dimensional and a required feature in each Message.
        missing_features = np.isnan(features["pose_fully_defined"]).flatten()
        # Remove missing features (contain nan values)
        locations = locations[~missing_features]
        for feature in features:
            features[feature] = features[feature][~missing_features]
        return features, locations
