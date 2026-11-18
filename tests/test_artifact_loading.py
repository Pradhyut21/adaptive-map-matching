"""A deployment must use the complete, frozen model/decoder pair."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pathsnap import predict, refined_inputs


class ArtifactLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.artifacts = self.root / 'artifacts'
        self.artifacts.mkdir()
        self.model = self.artifacts / 'context_boost.joblib'
        self.metadata = {
            'feature_version': refined_inputs.VERSION,
            'feature_names': refined_inputs.FEATURE_NAMES,
            'candidate_k': refined_inputs.K,
            'candidate_radius_m': refined_inputs.RADIUS,
        }

    def test_missing_frozen_decoder_prevents_silent_prediction_change(self):
        with patch.object(predict.joblib, 'load', return_value=self.metadata.copy()):
            with self.assertRaises(FileNotFoundError):
                predict.load_artifact(self.model)

    def test_incomplete_frozen_decoder_is_rejected(self):
        self.model.with_suffix('.config.json').write_text('{"decoder": {}}', encoding='utf-8')
        with patch.object(predict.joblib, 'load', return_value=self.metadata.copy()):
            with self.assertRaisesRegex(ValueError, 'frozen decoder'):
                predict.load_artifact(self.model)

    def test_active_selection_loads_matching_decoder_sidecar(self):
        settings = {'decoder': {'beta': 11}, 'refinement': {'same_arc_jitter_m': 5}}
        self.model.with_suffix('.config.json').write_text(json.dumps(settings), encoding='utf-8')
        (self.artifacts / 'active_model.json').write_text('{"file":"context_boost.joblib"}', encoding='utf-8')
        with patch.object(predict, 'ROOT', self.root), patch.dict(predict.os.environ, {}, clear=True), \
                patch.object(predict.joblib, 'load', return_value=self.metadata.copy()) as read:
            artifact = predict.load_artifact()
        read.assert_called_once_with(self.model)
        self.assertEqual(artifact['decoder_config'], settings['decoder'])
        self.assertEqual(artifact['decoder_refinement'], settings['refinement'])

    def test_feature_configuration_mismatch_is_rejected(self):
        self.model.with_suffix('.config.json').write_text('{"decoder":{},"refinement":{}}', encoding='utf-8')
        mismatched = {**self.metadata, 'candidate_k': 10}
        with patch.object(predict.joblib, 'load', return_value=mismatched):
            with self.assertRaisesRegex(ValueError, 'candidate configuration'):
                predict.load_artifact(self.model)


if __name__ == '__main__':
    unittest.main()
