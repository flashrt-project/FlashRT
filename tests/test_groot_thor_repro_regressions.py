"""CPU regressions for sharded calibration identity and attention padding."""
import json
import tempfile
import unittest
from pathlib import Path

from flash_rt.core.quant.calibrator import _checkpoint_hash
from flash_rt.hardware.thor.attn_backend_groot_n17 import make_groot_n17_attention_spec


class ThorReproductionRegressions(unittest.TestCase):
    def test_shard_contents_and_statistics_identify_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            index={'weight_map':{'tensor':'model-00001-of-00001.safetensors'}}
            for name,content in [('a',b'first weights'),('b',b'other weights')]:
                p=root/name;p.mkdir()
                (p/'model.safetensors.index.json').write_text(json.dumps(index))
                (p/'model-00001-of-00001.safetensors').write_bytes(content)
                (p/'statistics.json').write_text('{}')
            self.assertNotEqual(_checkpoint_hash(root/'a'),_checkpoint_hash(root/'b'))
            before=_checkpoint_hash(root/'a')
            (root/'a'/'statistics.json').write_text('{"scale":2}')
            self.assertNotEqual(before,_checkpoint_hash(root/'a'))

    def test_non_aligned_backbone_sequence_fills_entire_logits_stride(self):
        spec=make_groot_n17_attention_spec(num_views=1,llm_seq_max=90,vl_self_attn_seq_max=90,sa=41,s_kv_text=128,s_kv_image=512)
        for name in ('llm','vl_self_attn'):
            site=spec.site(name)
            self.assertEqual(site.max_q_seq,90)
            self.assertEqual(site.max_kv_seq,96)
            # This is the count filled with -inf before attention replay.
            self.assertEqual(site.num_q_heads*site.max_q_seq*site.max_kv_seq,site.num_q_heads*90*96)


if __name__=='__main__':unittest.main()
