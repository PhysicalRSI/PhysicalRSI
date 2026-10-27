"""Optional adapter to ENPIRE's image-only SAM3 function, not its oracle tools.

Install ENPIRE and run its model service separately. Importing this adapter
starts neither a service nor a robot. No upstream source is vendored here.
"""
from physicalrsi.runtime.sensors import _text


class ENPIRESegmenter:
    def __init__(self, *, url, source_revision, model_revision):
        self.url = _text(url)
        self.source_revision = _text(source_revision)
        self.model_revision = _text(model_revision)

    def identity(self):
        # Endpoint addresses are deployment configuration, not model identity.
        return dict(provider='ENPIRE.sam3_segment', source_revision=self.source_revision,
                    model_revision=self.model_revision, modality='rgb')

    def segment(self, rgb, text, *, threshold):
        from enpire.env.forge.cap.agent.tools.detection import sam3_segment
        return sam3_segment(rgb, text, threshold=threshold, url=self.url)
