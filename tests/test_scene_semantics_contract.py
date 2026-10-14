from pathlib import Path
import ast
import unittest
from typing import Any


def normalize():
    tree = ast.parse((Path(__file__).resolve().parents[1] / 'app/environment.py').read_text())
    names = {'normalize_scene_context', '_interaction_mode', '_scene_summary', '_intent_summary'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {'Any': Any}
    exec(compile(ast.Module(body=ast.parse('from __future__ import annotations').body + nodes, type_ignores=[]), 'scene-contract', 'exec'), ns)
    return ns['normalize_scene_context']


class SceneContract(unittest.TestCase):
    def test_recommendation_never_becomes_current_scene(self):
        result = normalize()({'recommendedScene': {'sceneName': 'Sleep', 'source': 'user_selected'}})
        self.assertNotIn('currentScene', result)
        self.assertNotIn('recommendedScene', result)
        self.assertEqual(result['sceneSemantics']['systemRecommendation'], 'weak_suggestion')

    def test_user_scene_requires_explicit_mode_and_source(self):
        scene = {'sceneName': 'Lunch break', 'source': 'user_selected'}
        self.assertNotIn('currentScene', normalize()({'interactionMode': 'auto', 'currentScene': scene}))
        self.assertEqual(normalize()({'interactionMode': 'scene', 'currentScene': scene})['currentScene']['sceneName'], 'Lunch break')

    def test_notification_slot_only_in_auto(self):
        self.assertIsNone(normalize()({'interactionMode': 'auto'})['potentialIntent'])
        self.assertNotIn('potentialIntent', normalize()({'interactionMode': 'goal', 'potentialIntent': None}))
        self.assertNotIn('potentialIntent', normalize()({'potentialIntent': {'source': 'notification'}}))

    def test_normalization_is_idempotent(self):
        f = normalize()
        value = {'interactionMode': 'auto', 'recommendedScene': {'sceneName': 'Sleep'}}
        self.assertEqual(f(f(value)), f(value))
