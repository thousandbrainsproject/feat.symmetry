# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, sentinel

from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.monty_base import MontyBase


class MontyBasePrivateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sm1 = MagicMock()
        self.sm1.sensor_module_id = "sm1"
        self.sm2 = MagicMock()
        self.sm2.sensor_module_id = "sm2"
        self.lm1 = MagicMock()
        self.lm2 = MagicMock()
        self.lm3 = MagicMock()
        self.attention_system = MagicMock()
        self.monty_base = MontyBase(
            sensor_modules=[self.sm1, self.sm2],
            learning_modules=[self.lm1, self.lm2, self.lm3],
            motor_system=MagicMock(),
            sm_to_agent_dict={
                "sm1": AgentID("agent_id_0"),
                "sm2": AgentID("agent_id_0"),
            },
            sm_to_lm_matrix=[[], [], []],
            lm_to_lm_matrix=[[], [], []],
            lm_to_lm_vote_matrix=[[], [], []],
            attention_system=self.attention_system,
        )

    def test_pass_goals_collects_all_goals_from_learning_and_sensor_modules_and_filters_them_through_the_attention_system(  # noqa: E501
        self,
    ) -> None:
        self.lm1.propose_goals.return_value = []
        self.lm2.propose_goals.return_value = [sentinel.lm2_goal]
        self.lm3.propose_goals.return_value = [
            sentinel.lm3_goal_1,
            sentinel.lm3_goal_2,
        ]
        self.lm1.propose_region.return_value = sentinel.lm1_region
        self.lm2.propose_region.return_value = sentinel.lm2_region
        self.lm3.propose_region.return_value = sentinel.lm3_region
        self.sm1.propose_goals.return_value = []
        self.sm2.propose_goals.return_value = [
            sentinel.sm2_goal_1,
            sentinel.sm2_goal_2,
        ]
        self.sm1.propose_region.return_value = sentinel.sm1_region
        self.sm2.propose_region.return_value = sentinel.sm2_region
        self.attention_system.step.return_value = sentinel.attention_system_goals

        self.monty_base._pass_goals()

        goals = [
            sentinel.lm2_goal,
            sentinel.lm3_goal_1,
            sentinel.lm3_goal_2,
            sentinel.sm2_goal_1,
            sentinel.sm2_goal_2,
        ]
        regions = [
            sentinel.sm1_region,
            sentinel.sm2_region,
            sentinel.lm1_region,
            sentinel.lm2_region,
            sentinel.lm3_region,
        ]
        self.attention_system.step.assert_called_once_with(goals, regions)
        self.assertIs(self.monty_base._goals, sentinel.attention_system_goals)


class MontyBaseSameStepLMOutputsTest(unittest.TestCase):
    def make_monty_base(
        self,
        same_step_lm_outputs: bool,
        lm_to_lm_matrix: list[list[int]] | None = None,
    ) -> MontyBase:
        self.child = MagicMock()
        self.parent = MagicMock()
        self.child.get_output.return_value = MagicMock(name="current_child_output")
        sm = MagicMock()
        sm.sensor_module_id = "sm"
        monty_base = MontyBase(
            sensor_modules=[sm],
            learning_modules=[self.child, self.parent],
            motor_system=MagicMock(),
            sm_to_agent_dict={"sm": AgentID("agent_id_0")},
            sm_to_lm_matrix=[[0], [0]],
            lm_to_lm_matrix=[[], [0]] if lm_to_lm_matrix is None else lm_to_lm_matrix,
            lm_to_lm_vote_matrix=None,
            same_step_lm_outputs=same_step_lm_outputs,
        )
        self.sm_output = MagicMock(name="sm_output")
        self.previous_child_output = MagicMock(name="previous_child_output")
        monty_base.sensor_module_outputs = [self.sm_output]
        monty_base.learning_module_outputs = [
            self.previous_child_output,
            MagicMock(name="previous_parent_output"),
        ]
        return monty_base

    def test_parent_gets_previous_step_output_by_default(self) -> None:
        monty_base = self.make_monty_base(same_step_lm_outputs=False)

        monty_base._step_learning_modules(MagicMock())

        self.parent.matching_step.assert_called_once()
        _, percepts = self.parent.matching_step.call_args.args
        self.assertEqual(percepts, [self.sm_output, self.previous_child_output])

    def test_parent_gets_same_step_output_when_enabled(self) -> None:
        monty_base = self.make_monty_base(same_step_lm_outputs=True)

        monty_base._step_learning_modules(MagicMock())

        _, percepts = self.parent.matching_step.call_args.args
        self.assertEqual(percepts, [self.sm_output, self.child.get_output.return_value])

    def test_rejects_a_child_indexed_after_its_parent(self) -> None:
        with self.assertRaises(ValueError):
            self.make_monty_base(same_step_lm_outputs=True, lm_to_lm_matrix=[[1], []])

    def test_allows_any_order_when_disabled(self) -> None:
        self.make_monty_base(same_step_lm_outputs=False, lm_to_lm_matrix=[[1], []])
