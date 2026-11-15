# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Idea 4, step 7: votes as a virtual movement between sensors."""

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
from tbp.monty.frameworks.models.evidence_matching.model import (
    MontyForEvidenceGraphMatching,
)
from tbp.monty.frameworks.utils.spatial_arithmetics import developed_displacement
from tbp.monty.geometry import OrientationMatrix, SurfacePoseVectors

RADIUS = 0.04
# In-plane orientation of the logo on the unrolled cylinder (model frame).
LOGO = Rotation.from_euler("z", 0.6).as_matrix()


def surface_frame(theta: float) -> np.ndarray:
    # Columns u, v, n on a cylinder of axis world y, at angle theta.
    return np.column_stack(
        [
            [np.cos(theta), 0.0, -np.sin(theta)],
            [0.0, 1.0, 0.0],
            [np.sin(theta), 0.0, np.cos(theta)],
        ]
    )


def world_location(theta: float, height: float) -> np.ndarray:
    return np.array([RADIUS * np.sin(theta), height, RADIUS * np.cos(theta)])


def model_location(theta: float, height: float) -> np.ndarray:
    return LOGO @ np.array([RADIUS * theta, height, 0.0])


def world_to_model(theta: float) -> np.ndarray:
    # The correct hypothesis pose P at this point (input = world).
    return LOGO @ surface_frame(theta).T


def sm_frame(theta: float, twist: float) -> np.ndarray:
    # An SM's frame: the surface frame with an arbitrary in-plane twist.
    return surface_frame(theta) @ Rotation.from_euler("z", twist).as_matrix()


def make_vote(location, pose) -> Message:
    return Message(
        location=np.array(location, dtype=float),
        morphological_features={
            "pose_vectors": OrientationMatrix(np.array(pose, dtype=float)),
            "pose_fully_defined": True,
        },
        non_morphological_features=None,
        confidence=1.0,
        pass_message=True,
        process_features_in_lm=True,
        sender_id="learning_module_x",
        sender_type="LM",
    )


def lm_vote(world, frame, states) -> dict:
    return {
        "possible_states": {"logo": states},
        "sensed_pose_rel_body": np.vstack([world, np.eye(3)]),
        "sensed_frame": frame,
    }


def combine(receiving: dict, sending: dict) -> Message:
    monty = Mock(learning_modules=[0, 1], lm_to_lm_vote_matrix=[[1], [0]])
    combined = MontyForEvidenceGraphMatching._combine_votes(monty, [receiving, sending])
    return combined[0]["logo"][0]


class DevelopedDisplacementTest(unittest.TestCase):
    def test_equal_normals_return_the_chord(self) -> None:
        chord = np.array([0.01, -0.02, 0.005])
        z = np.array([0.0, 0.0, 1.0])
        np.testing.assert_array_equal(developed_displacement(chord, z, z), chord)

    def test_cylinder_chord_unrolls_to_arc_in_first_tangent_plane(self) -> None:
        (ti, hi), (tj, hj) = (0.1, 0.0), (0.6, 0.012)
        chord = world_location(tj, hj) - world_location(ti, hi)
        d = developed_displacement(
            chord, surface_frame(ti)[:, 2], surface_frame(tj)[:, 2]
        )
        np.testing.assert_allclose(
            d, surface_frame(ti) @ [RADIUS * (tj - ti), hj - hi, 0.0], atol=1e-15
        )


class CombineVotesTest(unittest.TestCase):
    def test_2d_vote_lands_on_receivers_true_location_and_pose(self) -> None:
        # Sensors about 2.3 cm apart on a 4 cm cylinder; the chord vote misses by
        # about 5 mm. Each SM's frame has its own in-plane twist.
        (ti, hi), (tj, hj) = (0.1, 0.0), (0.6, 0.012)
        sending = lm_vote(
            world_location(ti, hi),
            sm_frame(ti, 0.4),
            [make_vote(model_location(ti, hi), world_to_model(ti))],
        )
        receiving = lm_vote(world_location(tj, hj), sm_frame(tj, -1.1), [])

        vote = combine(receiving, sending)

        np.testing.assert_allclose(vote.location, model_location(tj, hj), atol=1e-15)
        np.testing.assert_allclose(
            vote.get_pose_vectors().value, world_to_model(tj), atol=1e-15
        )

    def test_world_frame_votes_are_unchanged(self) -> None:
        # 3D input: no frames, so the vote is L + P @ chord and the pose is kept.
        pose = Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix()
        location = np.array([0.01, 0.02, 0.03])
        sending_world = np.array([0.0, 1.5, 0.0])
        receiving_world = np.array([0.02, 1.49, 0.01])
        sending = lm_vote(sending_world, None, [make_vote(location, pose)])
        receiving = lm_vote(receiving_world, None, [])

        vote = combine(receiving, sending)

        np.testing.assert_allclose(
            vote.location, location + pose @ (receiving_world - sending_world)
        )
        np.testing.assert_allclose(vote.get_pose_vectors().value, pose)


class SendOutVoteWorldFrameTest(unittest.TestCase):
    def test_vote_carries_world_sensed_pose_and_frame(self) -> None:
        lm = EvidenceGraphLM(
            max_match_distance=0.005,
            tolerances={"patch": {"hsv": [0.1, 1, 1]}},
            feature_weights={"patch": {"hsv": np.array([1, 0, 0])}},
        )
        frame = sm_frame(0.3, 0.2)
        world = world_location(0.3, 0.01)
        world_rows = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        percept = Message(
            location=world.copy(),
            morphological_features={
                "pose_vectors": SurfacePoseVectors(world_rows @ frame.T),
                "pose_fully_defined": True,
                "on_object": 1.0,
            },
            non_morphological_features={},
            confidence=1.0,
            pass_message=True,
            sender_id="patch",
            sender_type="SM",
            process_features_in_lm=True,
        )
        percept.set_sensor_to_world(frame)
        lm.exploratory_step(Mock(), [percept])
        lm._hypotheses = {
            "logo": Hypotheses(
                evidence=np.array([5.0]),
                locations=np.zeros((1, 3)),
                poses=np.eye(3)[None],
                possible=np.array([True]),
            )
        }

        # Monty's step loop records processed steps; stub it for a bare LM.
        lm.buffer.get_last_obs_processed = Mock(return_value=True)

        vote = lm.send_out_vote()

        assert vote is not None
        # The buffer holds input-frame values (step 6); the vote must not.
        np.testing.assert_allclose(vote["sensed_pose_rel_body"][0], world)
        np.testing.assert_allclose(
            vote["sensed_pose_rel_body"][1:], world_rows @ frame.T
        )
        np.testing.assert_allclose(vote["sensed_frame"], frame)


if __name__ == "__main__":
    unittest.main()
