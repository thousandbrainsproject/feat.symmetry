# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import unittest
from unittest.mock import Mock

import numpy as np
from scipy.spatial.transform import Rotation

from tbp.monty.cmp import Message
from tbp.monty.frameworks.models.evidence_matching.hypotheses import Hypotheses
from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM,
)
from tbp.monty.geometry import SurfacePoseVectors

# Tangent -> world frames at t-1 and t; non-symmetric so a transposed update fails.
F0 = Rotation.from_euler("xyz", [10, -25, 5], degrees=True).as_matrix()
F1 = Rotation.from_euler("xyz", [-15, 40, 20], degrees=True).as_matrix()
Y0 = np.array([0.01, 1.5, 0.02])
Y1 = np.array([0.02, 1.49, 0.03])
# World arc-corrected step the SM sends; deliberately not Y1 - Y0 (the chord).
L_W = np.array([0.011, -0.012, 0.009])


def make_percept(
    location: np.ndarray,
    frame: np.ndarray | None = None,
    displacement: np.ndarray | None = None,
    process_features_in_lm: bool = False,
) -> Message:
    percept = Message(
        location=location.copy(),
        morphological_features={
            "pose_vectors": SurfacePoseVectors(np.eye(3)),
            "pose_fully_defined": False,
            "on_object": 1.0,
        },
        non_morphological_features={},
        confidence=1.0,
        pass_message=True,
        sender_id="patch",
        sender_type="SM",
        process_features_in_lm=process_features_in_lm,
    )
    percept.set_sensor_to_world(frame)
    if displacement is not None:
        percept.set_displacement(displacement)
    return percept


class EvidenceGraphLMWorldFrameTest(unittest.TestCase):
    def setUp(self) -> None:
        self.lm = EvidenceGraphLM(
            max_match_distance=0.005,
            tolerances={"patch": {"hsv": [0.1, 1, 1]}},
            feature_weights={"patch": {"hsv": np.array([1, 0, 0])}},
        )
        # Input -> model poses of two hypotheses; non-symmetric on purpose.
        self.poses = np.stack(
            [
                Rotation.from_euler("xyz", [5, 15, 25], degrees=True).as_matrix(),
                Rotation.from_euler("xyz", [-30, 10, 60], degrees=True).as_matrix(),
            ]
        )
        self.locations = np.array([[0.0, 0.0, 0.0], [0.01, -0.02, 0.03]])
        self.lm._hypotheses = {
            "object": Hypotheses(
                evidence=np.array([1.0, 2.0]),
                locations=self.locations.copy(),
                poses=self.poses.copy(),
                possible=np.array([True, True]),
            )
        }
        self.lm.buffer.last_location = Y0.copy()

    def location_only_step(self, percept: Message) -> Hypotheses:
        self.lm.exploratory_step(Mock(), [percept])
        return self.lm._hypotheses["object"]

    def test_location_only_step_rotates_then_displaces(self) -> None:
        self.lm._last_frame = F0

        hypotheses = self.location_only_step(
            make_percept(Y1, frame=F1, displacement=L_W)
        )

        expected_poses = self.poses @ (F0 @ F1.T)
        np.testing.assert_allclose(hypotheses.poses, expected_poses)
        np.testing.assert_allclose(
            hypotheses.locations, self.locations + expected_poses @ L_W
        )
        np.testing.assert_allclose(self.lm.buffer.last_location, Y1)
        np.testing.assert_allclose(self.lm._last_frame, F1)

    def test_percepts_without_frame_behave_as_before(self) -> None:
        # A CameraSM-style percept: no frame, no SM displacement.
        hypotheses = self.location_only_step(make_percept(Y1))

        np.testing.assert_array_equal(hypotheses.poses, self.poses)
        np.testing.assert_allclose(
            hypotheses.locations, self.locations + self.poses @ (Y1 - Y0)
        )
        self.assertIsNone(self.lm._frame_delta)

    def test_first_frame_does_not_rotate(self) -> None:
        hypotheses = self.location_only_step(
            make_percept(Y1, frame=F1, displacement=L_W)
        )

        self.assertIsNone(self.lm._frame_delta)
        np.testing.assert_array_equal(hypotheses.poses, self.poses)
        np.testing.assert_allclose(
            hypotheses.locations, self.locations + self.poses @ L_W
        )

    def test_add_displacements_keeps_world_sm_displacement(self) -> None:
        percept = make_percept(Y1, frame=F1, displacement=L_W)

        (out,) = self.lm._add_displacements([percept])

        np.testing.assert_allclose(out.displacement["displacement"], L_W)

    def test_add_displacements_differences_locations_without_frame(self) -> None:
        # Without a frame, an SM-set displacement is replaced as before.
        percept = make_percept(Y1, displacement=L_W)

        (out,) = self.lm._add_displacements([percept])

        np.testing.assert_allclose(out.displacement["displacement"], Y1 - Y0)

    def test_reset_clears_the_frame(self) -> None:
        self.lm._last_frame = F0
        self.location_only_step(make_percept(Y1, frame=F1, displacement=L_W))

        self.lm.reset_stm()

        self.assertIsNone(self.lm._last_frame)
        self.assertIsNone(self.lm._frame_delta)

    def test_output_child_z_is_the_world_normal_after_frame_change(self) -> None:
        # A 2D child: in-plane rotation O (tangent -> model). Hypotheses were
        # initialized from world features at t-1, so their poses are O @ F0ᵀ.
        o_h = Rotation.from_euler("z", 40, degrees=True).as_matrix()
        self.lm._hypotheses["object"] = Hypotheses(
            evidence=np.array([2.0, 1.0]),
            locations=self.locations.copy(),
            poses=np.stack([o_h @ F0.T, self.poses[1]]),
            possible=np.array([True, True]),
        )
        self.lm._last_frame = F0

        self.location_only_step(make_percept(Y1, frame=F1, displacement=L_W))
        self.lm.current_mlh = self.lm._calculate_most_likely_hypothesis("object")
        rows = self.lm.get_output().morphological_features["pose_vectors"].value

        # Child axes in world: z is the surface normal at t, not at t-1.
        np.testing.assert_allclose(rows[2], F1[:, 2], atol=1e-12)
        np.testing.assert_allclose(rows, o_h @ F1.T, atol=1e-12)
        self.assertFalse(np.allclose(rows[2], F0[:, 2]))
