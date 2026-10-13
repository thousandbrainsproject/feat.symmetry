# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import unittest

import numpy as np

from tbp.monty.frameworks.models.evidence_matching.channels import PoseKind
from tbp.monty.frameworks.models.evidence_matching.graph_memory import (
    EvidenceGraphMemory,
)
from tbp.monty.geometry import Rotation
from tbp.monty.math import DEFAULT_TOLERANCE

RX_120 = Rotation.from_euler("x", 120, degrees=True)


def channel_features(*rotations):
    """Features for one channel with one observation per rotation.

    Returns:
        Feature dict with flat pose vectors, as the buffer stores them.
    """
    return {
        "pose_vectors": np.stack([r.as_matrix().flatten() for r in rotations]),
        "pose_fully_defined": np.ones((len(rotations), 1)),
    }


class EvidenceGraphMemoryUpdateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.memory = EvidenceGraphMemory(
            max_nodes_per_graph=10, max_graph_size=10, num_model_voxels_per_dim=10
        )

    def update(self, features, locations, pose_kinds):
        self.memory.update_memory(
            locations=locations,
            features={"child": features},
            graph_id="parent",
            object_location_rel_body=np.zeros(3),
            location_rel_model=np.zeros(3),
            object_rotation=Rotation.identity(),
            pose_kinds=pose_kinds,
        )

    def stored_pose(self):
        model = self.memory.models_in_memory["parent"]["child"]
        stored = np.array(model.get_values_for_feature("pose_vectors"))[0]
        return stored.reshape(3, 3)

    def test_build_averages_object_poses_with_rotation_mean(self) -> None:
        self.update(
            channel_features(Rotation.identity(), RX_120),
            np.zeros((2, 3)),
            {"child": PoseKind.OBJECT},
        )
        np.testing.assert_allclose(
            self.stored_pose(),
            Rotation.from_euler("x", 60, degrees=True).as_matrix(),
            atol=DEFAULT_TOLERANCE,
        )

    def test_extend_merges_object_poses_with_rotation_mean(self) -> None:
        kinds = {"child": PoseKind.OBJECT}
        self.update(channel_features(Rotation.identity()), np.zeros((1, 3)), kinds)
        self.update(channel_features(RX_120), np.zeros((1, 3)), kinds)
        np.testing.assert_allclose(
            self.stored_pose(),
            Rotation.from_euler("x", 60, degrees=True).as_matrix(),
            atol=DEFAULT_TOLERANCE,
        )

    def test_channel_missing_from_pose_kinds_raises(self) -> None:
        with self.assertRaises(KeyError):
            self.update(
                channel_features(Rotation.identity()), np.zeros((1, 3)), pose_kinds={}
            )
