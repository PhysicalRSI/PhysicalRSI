"""Optional RGB region proposals from a pinned public Grounding DINO model.

Loads only when requested. Boxes remain boxes; they are never relabeled as
segmentation masks, object poses, grasp points or physically measured contact.
"""
import re


class GroundingDINORegions:
    def __init__(self, *, model_id, revision, device='cpu'):
        if not isinstance(revision, str) or not re.fullmatch('[0-9a-f]{40}', revision):
            raise ValueError('Pin the model to its full repository commit')
        if not isinstance(model_id, str) or not model_id:
            raise ValueError('A model repository is required')
        self.model_id, self.revision, self.device = model_id, revision, device
        self._processor = self._model = None

    def identity(self):
        from importlib.metadata import version
        return dict(provider='GroundingDINO.rgb-regions', model=self.model_id,
                    revision=self.revision, device=self.device,
                    transformers=version('transformers'), torch=version('torch'),
                    numpy=version('numpy'), pillow=version('Pillow'),
                    text_threshold=.25, modality='rgb', measurement='box_center',
                    prompt_mode='joint-features-exact-label-matching')

    def segment(self, rgb, text, *, threshold):
        """Region-provider interface; return explicit boxes, not masks."""
        return self.segment_many(rgb, [text], threshold=threshold)[text]

    def segment_many(self, rgb, features, *, threshold):
        """Share one image/text forward pass across declared visual features."""
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        labels = {text.lower().strip().rstrip('.'): text for text in features}
        if len(labels) != len(features):
            raise ValueError('Visual prompts must remain distinct after normalization')
        if self._model is None:
            self._processor = AutoProcessor.from_pretrained(self.model_id, revision=self.revision)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.model_id, revision=self.revision, use_safetensors=True).to(self.device).eval()
        prompt = '. '.join(labels) + '.'
        inputs = self._processor(images=rgb, text=prompt, return_tensors='pt').to(self.device)
        with torch.inference_mode():
            output = self._model(**inputs)
        rows = self._processor.post_process_grounded_object_detection(
            output, inputs.input_ids, threshold=threshold, text_threshold=.25,
            target_sizes=[rgb.shape[:2]])[0]
        height, width = rgb.shape[:2]
        regions = {text: [] for text in features}
        for box, score, label in zip(rows['boxes'].tolist(), rows['scores'].tolist(), rows['text_labels']):
            feature = labels.get(label.lower().strip().rstrip('.'))
            if feature is None:
                continue  # No invented correspondence for an unmatched phrase.
            x1, y1, x2, y2 = box
            clipped = [max(0., x1), max(0., y1), min(float(width), x2), min(float(height), y2)]
            if clipped[0] < clipped[2] and clipped[1] < clipped[3]:
                regions[feature].append(dict(bbox_xyxy=clipped, score=float(score)))
        return regions
