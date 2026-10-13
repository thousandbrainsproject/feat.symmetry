# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from unittest import TestCase
from unittest.mock import Mock

import numpy as np

from tbp.monty.frameworks.models.evidence_matching.channels import PoseKind
from tbp.monty.frameworks.models.evidence_matching.hypotheses_updater import (
    DefaultHypothesesUpdater,
)
from tbp.monty.geometry import Rotation

NUM_NODES = 3
UMBILICAL_NUM_POSES = 8


class InformedPossiblePosesTest(TestCase):
    def setUp(self) -> None:
        self.stored_poses = Rotation.from_euler(
            "xyz", [[10, 20, 30], [40, 50, 60], [70, 80, 90]], degrees=True
        )
        graph_memory = Mock()
        # Pose vectors are the rotation's columns stored as rows
        graph_memory.get_rotation_features_at_all_nodes.return_value = (
            self.stored_poses.as_matrix().swapaxes(-1, -2)
        )
        graph_memory.get_locations_in_graph.return_value = np.zeros((NUM_NODES, 3))
        self.updater = DefaultHypothesesUpdater(
            feature_weights={},
            graph_memory=graph_memory,
            max_match_distance=0.01,
            tolerances={},
            umbilical_num_poses=UMBILICAL_NUM_POSES,
        )
        self.sensed_pose_vectors = (
            Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix().T
        )

    def _possible_rotations(
        self, pose_kind: PoseKind, pose_fully_defined: bool = True
    ) -> np.ndarray:
        _, rotations = self.updater._get_all_informed_possible_poses(
            graph_id="parent_object",
            sensed_channel_features={
                "pose_vectors": self.sensed_pose_vectors,
                "pose_fully_defined": pose_fully_defined,
            },
            input_channel="channel",
            pose_kind=pose_kind,
        )
        return rotations

    def test_object_pose_gives_one_rotation_per_node(self) -> None:
        self.assertEqual(len(self._possible_rotations(PoseKind.OBJECT)), NUM_NODES)

    def test_object_pose_ignores_pose_fully_defined(self) -> None:
        rotations = self._possible_rotations(PoseKind.OBJECT, pose_fully_defined=False)
        self.assertEqual(len(rotations), NUM_NODES)

    def test_surface_pose_gives_two_rotations_per_node(self) -> None:
        self.assertEqual(len(self._possible_rotations(PoseKind.SURFACE)), NUM_NODES * 2)

    def test_undefined_surface_pose_samples_umbilical_num_poses(self) -> None:
        rotations = self._possible_rotations(PoseKind.SURFACE, pose_fully_defined=False)
        self.assertEqual(len(rotations), UMBILICAL_NUM_POSES * NUM_NODES)

    def test_unhandled_pose_kind_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._possible_rotations(Mock())
