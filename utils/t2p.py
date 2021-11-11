import paddle
from typing import Optional


def LongTensor(data, dtype=paddle.int64, device=None, requires_grad=True, pin_memory=False):
    return paddle.to_tensor(data, dtype=dtype, stop_gradient=not(requires_grad))


def IntTensor(data, dtype=paddle.int8, device=None, requires_grad=True, pin_memory=False):
    return paddle.to_tensor(data, dtype=dtype, stop_gradient=not(requires_grad))


def FloatTensor(data, dtype=paddle.float32, device=None, requires_grad=True, pin_memory=False):
    return paddle.to_tensor(data, dtype=dtype, stop_gradient=not(requires_grad))


class SpatialSoftArgmax2d(paddle.nn.Layer):
    r"""Creates a module that computes the Spatial Soft-Argmax 2D
    of a given input heatmap.

    Returns the index of the maximum 2d coordinates of the give map.
    The output order is x-coord and y-coord.

    Arguments:
        normalized_coordinates (Optional[bool]): wether to return the
          coordinates normalized in the range of [-1, 1]. Otherwise,
          it will return the coordinates in the range of the input shape.
          Default is True.

    Shape:
        - Input: :math:`(B, N, H, W)`
        - Output: :math:`(B, N, 2)`

    Examples::
        >>> input = paddle.rand(1, 4, 2, 3)
        >>> m = tgm.losses.SpatialSoftArgmax2d()
        >>> coords = m(input)  # 1x4x2
        >>> x_coord, y_coord = paddle.chunk(coords, dim=-1, chunks=2)
    """

    def __init__(self, normalized_coordinates: Optional[bool] = True) -> None:
        super(SpatialSoftArgmax2d, self).__init__()
        self.normalized_coordinates: Optional[bool] = normalized_coordinates
        self.eps: float = 1e-6

    def forward(self, input: paddle.Tensor) -> paddle.Tensor:
        if not paddle.is_tensor(input):
            raise TypeError("Input input type is not a paddle.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(input.shape))
        # unpack shapes and create view from input tensor
        batch_size, channels, height, width = input.shape
        x: paddle.Tensor = paddle.reshape(input, shape=[batch_size, channels, -1])

        # compute softmax with max substraction trick
        exp_x = paddle.exp(x - paddle.max(x, axis=-1, keepdim=True)[0])
        exp_x_sum = 1.0 / (exp_x.sum(axis=-1, keepdim=True) + self.eps)

        # create coordinates grid
        pos_y, pos_x = create_meshgrid(input, self.normalized_coordinates)
        pos_x = paddle.reshape(pos_x, shape=[-1])
        pos_y = paddle.reshape(pos_y, shape=[-1])

        # compute the expected coordinates
        expected_y: paddle.Tensor = paddle.sum(
            (pos_y * exp_x) * exp_x_sum, axis=-1, keepdim=True)
        expected_x: paddle.Tensor = paddle.sum(
            (pos_x * exp_x) * exp_x_sum, axis=-1, keepdim=True)
        output: paddle.Tensor = paddle.concat([expected_x, expected_y], axis=-1)
        return paddle.reshape(output, shape=[batch_size, channels, 2])  # BxNx2


def create_meshgrid(
        x: paddle.Tensor,
        normalized_coordinates: Optional[bool]) -> paddle.Tensor:
    assert len(x.shape) == 4, x.shape
    _, _, height, width = x.shape
    _dtype = x.dtype
    if normalized_coordinates:
        xs = paddle.linspace(-1.0, 1.0, width, dtype=_dtype)
        ys = paddle.linspace(-1.0, 1.0, height, dtype=_dtype)
    else:
        xs = paddle.linspace(0, width - 1, width, dtype=_dtype)
        ys = paddle.linspace(0, height - 1, height, dtype=_dtype)
    return paddle.meshgrid(ys, xs)  # pos_y, pos_x
