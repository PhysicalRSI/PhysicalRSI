from PhysicalRSI_baselines.embodied_goodharts_law.public_lift_association import associated_visible_lift

def surface(x,y,z):return {'status':'estimated_surface','surface_quantiles':{'median':[x,y,z]}}

def test_recorded_switch_from_table_bowl_to_cabinet_bowl_is_not_lift_evidence():
    assert not associated_visible_lift(surface(.369,-.132,.052),surface(.652,-.275,.249),[.38,-.13,.3])

def test_vertical_surface_displacement_near_hand_is_only_an_estimate():
    assert associated_visible_lift(surface(.65,-.28,.25),surface(.66,-.27,.35),[.65,-.28,.45])

def test_stationary_or_remote_surface_does_not_verify_lift():
    before=surface(.65,-.28,.25)
    assert not associated_visible_lift(before,before,[.65,-.28,.45])
    assert not associated_visible_lift(before,surface(.65,-.28,.35),[.2,.3,.45])
