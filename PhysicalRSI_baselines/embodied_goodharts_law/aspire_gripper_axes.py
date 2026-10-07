"""Map CGN's contact baseline onto Panda's finger closing axis.

CGN build_6d_grasp places the contact baseline in rotation column X.
Panda's prismatic finger joints move along hand-local +/-Y. ASPIRE's
sample_grasp_pose applies a local +pi/2 Z rotation to reconcile them.
The earlier translation-only conversion remains unchanged for old evidence.
Robot model-to-native calibration must still be applied separately.
"""
import numpy as np

from .aspire_grasp_frame import contact_grasp_to_model_hand


def contact_grasp_panda_hand_variants(matrix):
    """Preserve approach/position and align parallel-finger contact lines."""
    translated = contact_grasp_to_model_hand(matrix)
    quarter_turn = np.array([[0., -1., 0., 0.], [1., 0., 0., 0.],
                             [0., 0., 1., 0.], [0., 0., 0., 1.]])
    hand = translated @ quarter_turn
    return [hand, hand @ np.diag([-1., -1., 1., 1.])]
