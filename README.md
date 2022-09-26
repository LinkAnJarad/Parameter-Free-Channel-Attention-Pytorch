# Parameter-Free-Channel-Attention-Pytorch
Unofficial implementation of Parameter-Free Channel Attention (PFCA) in Pytorch

## Usage

```python
batched_feature_maps = torch.randn(64, 128, 32, 32)

PFCA = ParameterFreeChannelAttention(channel_size=32)

PFCA(batched_feature_maps) # shape (64, 128, 32, 32)
```
