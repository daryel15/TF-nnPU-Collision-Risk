# Released weights

- `pretrained_encoder1.pt`: supplied encoder state dictionary, unchanged.
- `stage2_trained_model.pt`: supplied TF encoder/pooling/head state dictionary, unchanged.
- `expected_metrics.json`: main released-checkpoint evaluation targets.

The model architecture is in `tf_nnpu/models.py`. All loads use `weights_only=True` and an explicit map_location.

The missing historical names `pretrained_encoder.pt` and `pretrained_encoder_good.pt` are not silently aliased to this encoder. New baseline and sensitivity reruns explicitly pretrain from scratch on the released dataset; `stage2` explicitly starts from the released encoder. Exact historical head initialization states and some experiment-specific checkpoints are not available, so newly trained weights are not claimed to be byte-identical.
