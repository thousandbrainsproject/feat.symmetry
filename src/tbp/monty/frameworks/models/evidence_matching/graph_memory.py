# Copyright 2025-2026 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import logging
from typing import Mapping

from tbp.monty.frameworks.models.evidence_matching.channels import PoseKind
from tbp.monty.frameworks.models.graph_matching import GraphMemory
from tbp.monty.frameworks.models.object_model import (
    GridObjectModel,
    GridTooSmallError,
)

logger = logging.getLogger(__name__)


class EvidenceGraphMemory(GraphMemory):
    """Custom GraphMemory that stores GridObjectModel instead of GraphObjectModel."""

    def __init__(
        self,
        max_nodes_per_graph,
        max_graph_size,
        num_model_voxels_per_dim,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.max_nodes_per_graph = max_nodes_per_graph
        self.max_graph_size = max_graph_size
        self.num_model_voxels_per_dim = num_model_voxels_per_dim

    # =============== Public Interface Functions ===============

    # ------------------- Main Algorithm -----------------------
    def update_memory(
        self,
        locations,
        features,
        graph_id,
        object_location_rel_body,
        location_rel_model,
        object_rotation,
        pose_kinds: Mapping[str, PoseKind],
    ):
        """Determine how to update memory and call corresponding function.

        Same as GraphMemory.update_memory, but each channel's pose kind is passed
        down so the grid model can pick the right pose-vector average.

        Args:
            locations: Locations of all observations in the episode.
            features: Features per input channel, aligned with locations.
            graph_id: ID of the graph to build or extend. None skips the update.
            object_location_rel_body: Location of the sensor in the body reference
                frame.
            location_rel_model: Location of the sensor in the model reference frame.
            object_rotation: Rotation of the sensed object relative to the model.
            pose_kinds: What each input channel's pose vectors represent.
        """
        if graph_id is None:
            logger.info("no match found in time, not updating memory")
            return
        # Look up every channel before touching memory so a missing pose kind
        # raises KeyError without leaving earlier channels half-updated.
        pose_kind_per_channel = {channel: pose_kinds[channel] for channel in features}
        for input_channel in features:
            pose_kind = pose_kind_per_channel[input_channel]
            (
                input_channel_features,
                input_channel_locations,
            ) = self._extract_entries_with_content(features[input_channel], locations)
            if (
                graph_id in self.get_memory_ids()
                and input_channel in self.get_input_channels_in_graph(graph_id)
            ):
                logger.info(f"{graph_id} already in memory ({self.get_memory_ids()})")
                self._extend_graph(
                    input_channel_locations,
                    input_channel_features,
                    graph_id,
                    input_channel,
                    object_location_rel_body,
                    location_rel_model,
                    object_rotation,
                    pose_kind,
                )
            else:
                logger.info(f"{graph_id} not in memory ({self.get_memory_ids()})")
                self._build_graph(
                    input_channel_locations,
                    input_channel_features,
                    graph_id,
                    input_channel,
                    pose_kind,
                )

    # ------------------ Getters & Setters ---------------------
    def get_initial_hypotheses(self):
        return self.get_memory_ids()

    def get_rotation_features_at_all_nodes(self, graph_id, input_channel):
        """Get rotation features from all N nodes. Shape=(N, 3, 3).

        Returns:
            The rotation features from all N nodes. Shape=(N, 3, 3).
        """
        all_node_r_features = self.get_features_at_node(
            graph_id,
            input_channel,
            self.get_graph_node_ids(graph_id, input_channel),
            feature_keys=["pose_vectors"],
        )
        node_directions = all_node_r_features["pose_vectors"]
        num_nodes = len(node_directions)
        return node_directions.reshape((num_nodes, 3, 3))

    # ======================= Private ==========================

    # ------------------- Main Algorithm -----------------------
    def _add_graph_to_memory(self, model, graph_id):
        """Add a pretrained graph to memory.

        Initializes GridObjectModel and calls set_graph.

        Args:
            model: New model to be added to memory.
            graph_id: ID of the graph that should be added.

        """
        self.models_in_memory[graph_id] = {}
        for input_channel in model:
            channel_model = model[input_channel]
            try:
                if not isinstance(channel_model, GridObjectModel):
                    # When loading a model trained with a different LM, need to convert
                    # it to the GridObjectModel (with use_original_graph == True)
                    loaded_graph = channel_model._graph
                    channel_model = self._initialize_model_with_graph(
                        graph_id, loaded_graph
                    )
                else:
                    # serialization seems to mess up the sparse tensors, so we need to
                    # coalesce them again.
                    if channel_model._observation_count is not None:
                        channel_model._observation_count = (
                            channel_model._observation_count.coalesce()
                        )
                    if channel_model._feature_grid is not None:
                        channel_model._feature_grid = (
                            channel_model._feature_grid.coalesce()
                        )
                    if channel_model._location_grid is not None:
                        channel_model._location_grid = (
                            channel_model._location_grid.coalesce()
                        )

                logger.info(f"Loaded {model} for {input_channel}")
                self.models_in_memory[graph_id][input_channel] = channel_model
            except GridTooSmallError:
                logger.info("Grid too small for given locations. Not adding to memory.")

    def _initialize_model_with_graph(self, graph_id, graph):
        model = GridObjectModel(
            object_id=graph_id,
            max_nodes=self.max_nodes_per_graph,
            max_size=self.max_graph_size,
            num_voxels_per_dim=self.num_model_voxels_per_dim,
        )
        # Keep benchmark results constant by still using original graph for
        # matching when loading pretrained models.
        model.use_original_graph = True
        model.set_graph(graph)
        return model

    def _build_graph(self, locations, features, graph_id, input_channel, pose_kind):
        """Build a graph from a list of features at locations and add it to memory.

        This initializes a new GridObjectModel and calls model.build_graph.

        Args:
            locations: List of x, y, z locations.
            features: List of features.
            graph_id: ID of the new graph.
            input_channel: Identifier of the input channel.
            pose_kind: What the channel's pose vectors represent.
        """
        logger.info("Adding a new graph to memory.")

        model = GridObjectModel(
            object_id=graph_id,
            max_nodes=self.max_nodes_per_graph,
            max_size=self.max_graph_size,
            num_voxels_per_dim=self.num_model_voxels_per_dim,
        )
        try:
            model.build_model(
                locations=locations, features=features, pose_kind=pose_kind
            )

            if graph_id not in self.models_in_memory:
                self.models_in_memory[graph_id] = {}
            self.models_in_memory[graph_id][input_channel] = model

            logger.info(f"Added new graph with id {graph_id} to memory.")
            logger.info(model)
        except GridTooSmallError:
            logger.info(
                "Grid too small for given locations. Not building a model "
                f"for {graph_id}"
            )

    def _extend_graph(
        self,
        locations,
        features,
        graph_id,
        input_channel,
        object_location_rel_body,
        location_rel_model,
        object_rotation,
        pose_kind,
    ):
        """Add new observations into an existing graph.

        Args:
            locations: List of x, y, z locations.
            features: Features observed at the provided locations.
            graph_id: ID of the existing graph.
            input_channel: Identifier of the input channel.
            object_location_rel_body: Location of the sensor in the body reference
                frame.
            location_rel_model: Location of the sensor in the model reference frame.
            object_rotation: Rotation of the sensed object relative to the model.
            pose_kind: What the channel's pose vectors represent.
        """
        logger.info(f"Updating existing graph for {graph_id}")

        try:
            self.models_in_memory[graph_id][input_channel].update_model(
                locations=locations,
                features=features,
                location_rel_model=location_rel_model,
                object_location_rel_body=object_location_rel_body,
                object_rotation=object_rotation,
                pose_kind=pose_kind,
            )
            logger.info(
                f"Extended graph {graph_id} with new points. New model:\n"
                f"{self.models_in_memory[graph_id]}"
            )
        except GridTooSmallError:
            logger.info("Grid too small for given locations. Not updating model.")
