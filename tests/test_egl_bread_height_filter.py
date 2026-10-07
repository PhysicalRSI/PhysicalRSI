import copy
from PhysicalRSI_baselines.embodied_goodharts_law.bread_height_filter import filter_bread_height


def test_recorded_public_height_outlier_is_removed_without_changing_valid_surface():
    basket={'upper_95':[.079,-.114,.802429788374446]}
    pieces=[{'surface_quantiles':{'median':[-.166,-.036,.7987289888375101]},'detection_score':.93359375},
            {'surface_quantiles':{'median':[-.223,-.235,.9473386052117129]},'detection_score':.53125}]
    before=copy.deepcopy(pieces)
    assert filter_bread_height(pieces,basket)==pieces[:1]
    assert pieces==before


def test_empty_and_nonfinite_detections_do_not_become_candidates():
    basket={'upper_95':[0,0,.8]}
    assert filter_bread_height([],basket)==[]
    assert filter_bread_height([{'surface_quantiles':{'median':[0,0,float('nan')]}}],basket)==[]
