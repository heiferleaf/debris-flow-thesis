import unittest
import importlib
bind = importlib.import_module('experiments.runspecs.EXP-028.rebind_labels').bind


class SemanticOverlayTest(unittest.TestCase):
    def setUp(self):
        self.row={'frame_indices':list(range(16)), 'actual_start_seconds':0.,'actual_end_seconds':3.,
                  'legacy_guard_pass':True,'gray_QC_pass':True,'PTS_pass':True,
                  'retained_unique_physical_window':True}

    def label(self,state):
        known=state!='U'
        return [{'start_sec':0.,'end_sec':3.,'released_state':state,'release_id':'fixture',
                 'mask_three_class':known,'mask_existence':known,'existence_value':None if not known else int(state!='N')}]

    def test_relabel_does_not_clear_old_quality_or_guard_failure(self):
        for field in ['legacy_guard_pass','gray_QC_pass','PTS_pass']:
            composition,legal=bind({**self.row,field:False},self.label('N'))
            self.assertEqual(composition['r_existence'],0.)
            self.assertFalse(legal)

    def test_unknown_masks_and_changed_target_follow_current_release(self):
        unknown,legal=bind(self.row,self.label('U'))
        self.assertFalse(legal)
        self.assertIsNone(unknown['r_existence'])
        positive,legal=bind(self.row,self.label('P'))
        self.assertTrue(legal)
        self.assertEqual(positive['r_existence'],1.)


if __name__=='__main__':
    unittest.main()
