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
from unittest.mock import Mock, patch

import numpy as np

from tbp.monty.cmp import Message
from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM,
)
from tbp.monty.geometry import Rotation, SurfacePoseVectors
from tbp.monty.math import DEFAULT_TOLERANCE

A = np.sqrt(0.5)
# Tangent frame on a surface tilted about y: columns u, v, n.
W_T_R = np.column_stack([[A, 0.0, -A], [0.0, 1.0, 0.0], [A, 0.0, A]])


def make_percept(frame=None, sender_type="SM") -> Message:
    percept = Message(
        location=np.zeros(3),
        morphological_features={
            "pose_vectors": SurfacePoseVectors(np.eye(3)),
            "pose_fully_defined": False,
            "on_object": 1.0,
        },
        non_morphological_features={},
        confidence=1.0,
        pass_message=True,
        sender_id="patch" if sender_type == "SM" else "learning_module_0",
        sender_type=sender_type,
        process_features_in_lm=False,
    )
    percept.set_sensor_to_world(frame)
    return percept


class EvidenceGraphLMSensorToWorldTest(unittest.TestCase):
    def setUp(self) -> None:
        self.lm = EvidenceGraphLM(
            max_match_distance=0.005,
            tolerances={"patch": {"hsv": [0.1, 1, 1]}},
            feature_weights={"patch": {"hsv": np.array([1, 0, 0])}},
        )
        # Input -> model, non-symmetric so a transposed composition fails.
        self.input_to_model = Rotation.from_euler("xyz", [10, 20, 30], degrees=True)
        self.lm.current_mlh["rotation"] = self.input_to_model

    def output_rows(self) -> np.ndarray:
        return self.lm.get_output().morphological_features["pose_vectors"].value

    def test_output_unchanged_without_a_frame(self) -> None:
        self.lm._update_sensor_to_world([make_percept(frame=None)])

        np.testing.assert_allclose(
            self.output_rows(), self.input_to_model.inv().as_matrix().T
        )

    def test_output_rows_are_child_axes_in_world(self) -> None:
        self.lm._update_sensor_to_world([make_percept(frame=W_T_R)])

        tangent_rows = self.input_to_model.inv().as_matrix().T
        np.testing.assert_allclose(self.output_rows(), tangent_rows @ W_T_R.T)

    def test_2d_child_z_axis_is_the_world_normal(self) -> None:
        # A 2D child only rotates in-plane, so its z axis is the tangent normal.
        self.lm.current_mlh["rotation"] = Rotation.from_euler("z", 40, degrees=True)
        self.lm._update_sensor_to_world([make_percept(frame=W_T_R)])

        np.testing.assert_allclose(
            self.output_rows()[2], W_T_R[:, 2], atol=DEFAULT_TOLERANCE
        )

    def test_ignores_frames_from_lms(self) -> None:
        self.lm._update_sensor_to_world([make_percept(W_T_R, sender_type="LM")])

        np.testing.assert_allclose(
            self.lm._sensor_to_world.as_matrix(), np.eye(3), atol=DEFAULT_TOLERANCE
        )

    def test_location_only_step_updates_the_frame(self) -> None:
        with patch.object(self.lm, "_displace_hypotheses"):
            self.lm.exploratory_step(Mock(), [make_percept(frame=W_T_R)])

        np.testing.assert_allclose(
            self.lm._sensor_to_world.as_matrix(), W_T_R, atol=1e-12
        )

    def test_reset_returns_to_identity(self) -> None:
        self.lm._update_sensor_to_world([make_percept(frame=W_T_R)])
        self.lm.reset_stm()

        np.testing.assert_allclose(
            self.lm._sensor_to_world.as_matrix(), np.eye(3), atol=1e-12
        )
