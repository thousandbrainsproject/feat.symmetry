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

import copy
import logging

import numpy as np

from tbp.monty.frameworks.models.graph_matching import (
    MontyForGraphMatching,
)
from tbp.monty.frameworks.utils.spatial_arithmetics import (
    developed_displacement,
    minimal_rotation,
)
from tbp.monty.geometry import OrientationMatrix

logger = logging.getLogger(__name__)


def _frame_normal(frame: np.ndarray | None) -> np.ndarray:
    """Get the z axis of a vote's input frame.

    Args:
        frame: (3, 3) input frame -> world rotation, or None for world input.

    Returns:
        The frame's z axis (the surface normal for a 2D SM), or world z.
    """
    return np.array([0.0, 0.0, 1.0]) if frame is None else frame[:, 2]


class MontyForEvidenceGraphMatching(MontyForGraphMatching):
    """Monty model for evidence-based graphs.

    Customize voting and union of possible matches.
    """

    def __init__(self, *args, **kwargs):
        """Initialize and reset LM."""
        super().__init__(*args, **kwargs)

    def _combine_votes(self, votes_per_lm):
        """Combine evidence from different LMs.

        Returns:
            The combined votes.

        Raises:
            TypeError: If vote does not have OrientationMatrix for pose vector.
        """
        combined_votes = []
        for i in range(len(self.learning_modules)):
            lm_state_votes = {}
            if votes_per_lm[i] is not None:
                receiving_lm_pose = votes_per_lm[i]["sensed_pose_rel_body"]
                for j in self.lm_to_lm_vote_matrix[i]:
                    if votes_per_lm[j] is not None:
                        sending_lm_pose = votes_per_lm[j]["sensed_pose_rel_body"]
                        sensor_disp = np.array(receiving_lm_pose[0]) - np.array(
                            sending_lm_pose[0]
                        )
                        # A vote is a virtual movement from the sending sensor to
                        # the receiving one: transport between the input frames'
                        # normals, and the chord unbent into the sender's tangent
                        # plane. With world-frame (3D) input both are identity.
                        sending_normal = _frame_normal(
                            votes_per_lm[j].get("sensed_frame")
                        )
                        receiving_normal = _frame_normal(
                            votes_per_lm[i].get("sensed_frame")
                        )
                        transport = minimal_rotation(sending_normal, receiving_normal)
                        sensor_disp = developed_displacement(
                            sensor_disp, sending_normal, receiving_normal
                        )
                        logger.debug(
                            f"LM {j} to {i} - displacement: {sensor_disp}, "
                            f"transport: {transport}"
                        )
                        for obj in votes_per_lm[j]["possible_states"]:
                            # Get the displacement between the sending and receiving
                            # sensor and take this into account when transmitting
                            # possible locations on the object.
                            # "If I am here, you should be there."
                            lm_states_for_object = votes_per_lm[j]["possible_states"][
                                obj
                            ]
                            # Take the location votes and transform them so they would
                            # apply to the receiving LM's sensor. Basically, if my
                            # sensor is here in this pose, then your sensor should be
                            # there in that pose.
                            # NOTE: rotation votes are not being used right now.
                            transformed_lm_states_for_object = []
                            for s in lm_states_for_object:
                                # need to make a copy because the same vote may be
                                # transformed in different ways depending on the
                                # receiving LMs' poses
                                new_s = copy.deepcopy(s)
                                pose = new_s.get_pose_vectors()
                                if not isinstance(pose, OrientationMatrix):
                                    raise TypeError(
                                        "Only OrientationMatrices can rotate sensor "
                                        "displacements."
                                    )
                                # Same update as displace_hypotheses with
                                # frame_delta = transport^T: the location moves by
                                # P @ d, and the pose becomes P @ transport^T.
                                new_s.location = (
                                    new_s.location + pose.value @ sensor_disp
                                )
                                new_s.morphological_features["pose_vectors"] = (
                                    OrientationMatrix(pose.value @ transport.T)
                                )
                                transformed_lm_states_for_object.append(new_s)
                            if obj in lm_state_votes:
                                lm_state_votes[obj].extend(
                                    transformed_lm_states_for_object
                                )
                            else:
                                lm_state_votes[obj] = transformed_lm_states_for_object
            logger.debug(f"VOTE from LMs {self.lm_to_lm_vote_matrix[i]} to LM {i}")
            vote = lm_state_votes
            combined_votes.append(vote)
        return combined_votes

    def switch_to_exploratory_step(self):
        """Switch to an exploratory step.

        Also set MLH evidence high enough to generate output during exploration.
        """
        super().switch_to_exploratory_step()
        # Make sure the new object ID is communicated to higher-level LMs during
        # exploration.
        for lm in self.learning_modules:
            lm.current_mlh["evidence"] = lm.object_evidence_threshold + 1
