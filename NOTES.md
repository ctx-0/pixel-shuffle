# Notes

Pixel shuffle means depth-to-space.

It moves values from channel depth into spatial positions:

```
H x W x (C * r^2) -> (H * r) x (W * r) x C
```

Pixel unshuffle means space-to-depth.

It moves values from spatial positions into channel depth:

```
(H * r) x (W * r) x C -> H x W x (C * r^2)
```

SmolVLM paper calls it Pixel shuffle (space-to-depth)

## Direct factor vs repeated factor

When a library receives factor `r`, it applies that factor directly.

For example, a direct `r = 4` space-to-depth operation packs each `4 x 4` spatial block into `16` channel positions:

```text
64 x 64 x 1 -> 16 x 16 x 16
```

That can differ in channel order from calling `r = 2` twice:

```text
64 x 64 x 1 -> 32 x 32 x 4 -> 16 x 16 x 16
```

Both are valid rearrangements, but they are not necessarily the same permutation.

PyTorch does not secretly call `r = 2` repeatedly. If you call `PixelUnshuffle(4)`, it applies direct factor `4`. If you want repeated `2x` behavior, you explicitly call `PixelUnshuffle(2)` twice.

## Libraries

| Library / code | Bigger spatial grid | Smaller spatial grid | Notes |
| --- | --- | --- | --- |
| PyTorch | `nn.PixelShuffle(r)` | `nn.PixelUnshuffle(r)` | Direct factor `r`. Pixel shuffle is depth-to-space. Pixel unshuffle is space-to-depth. |
| TensorFlow | `tf.nn.depth_to_space` | `tf.nn.space_to_depth` | Literal names. Direct `block_size`. |
| ONNX | `DepthToSpace` | `SpaceToDepth` | Same concepts, but `DepthToSpace` has ordering modes such as `DCR` and `CRD`. |
| SmolLM / SmolVLM linked code | named `pixel_shuffle` | semantically space-to-depth | The linked code reduces spatial tokens and increases hidden dimension, so it behaves like pixel unshuffle despite the name. |

## Animation

The visualization has four stages:

    64x64x1 -> 32x32x4 -> 16x16x16 -> 8x8x64

Each transition is animated in small phases:

    stable view -> lift into channel depth -> collapse x/y grid -> resize tiles -> stable view

During `lift`, pixels separate into channel-depth layers. During `collapse`, their x/y positions move into the smaller spatial grid. During `resize`, the visible tiles grow to fill the next grid cell size. The stable views at the start and end give each tensor shape a moment to read before the next transition begins.

These phases are visual only. The actual operation computes the final space-to-depth address directly; `lift`, `collapse`, `resize`, stable pauses, and camera zoom are animation choices that make the rearrangement easier to follow.

References:

- PyTorch PixelShuffle: https://docs.pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html
- PyTorch PixelUnshuffle: https://docs.pytorch.org/docs/stable/nn.html
- TensorFlow space_to_depth: https://www.tensorflow.org/api_docs/python/tf/compat/v1/space_to_depth
- ONNX DepthToSpace: https://onnx.ai/onnx/operators/
- SmolLM code reference: https://github.com/huggingface/smollm/blob/a041759883ec7152d18fb985ea49be641a0bceef/vision/m4/models/vllama3/modeling_vllama3.py#L1281
