# Parameter-Free-Channel-Attention-Pytorch
Unofficial implementation of [Parameter-Free Channel Attention](https://www.researchgate.net/publication/360462671_PARAMETER-FREE_CHANNEL_ATTENTION_FOR_IMAGE_CLASSIFICATION_AND_SUPER-RESOLUTION) (PFCA) in Pytorch

![image](https://user-images.githubusercontent.com/79294502/192215784-5de47fde-4c28-4543-a9e2-fc63d51a29ce.png)

## Paper

Shi, Yuxuan & Xu, Jun & Yang, Lingxiao & An, Wangpeng & Zhen, Xiantong. (2022). PARAMETER-FREE CHANNEL ATTENTION FOR IMAGE CLASSIFICATION AND SUPER-RESOLUTION. [10.13140/RG.2.2.20039.78241](https://www.researchgate.net/publication/360462671_PARAMETER-FREE_CHANNEL_ATTENTION_FOR_IMAGE_CLASSIFICATION_AND_SUPER-RESOLUTION). 

Original License: [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)

## Usage

The module outputs the feature map **already** multiplied with the attention values.

```python
batched_feature_maps = torch.randn(64, 128, 32, 32) # (batch_size, channels, H, W)

PFCA = ParameterFreeChannelAttention(feature_map_size=32)

PFCA(batched_feature_maps) # shape (64, 128, 32, 32)
```
