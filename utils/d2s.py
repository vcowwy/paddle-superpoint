"""Module used to change 2D labels to 3D labels and vise versa.
Mimic function from tensorflow.
"""
import paddle
import paddle.nn as nn


class DepthToSpace(nn.Layer):
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = paddle.transpose(input, perm=[0, 2, 3, 1])
        batch_size, d_height, d_width, d_depth = output.shape[0], output.shape[1], output.shape[2], output.shape[3]

        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)

        t_1 = paddle.reshape(output, shape=[batch_size, d_height, d_width, self.block_size_sq, s_depth])
        spl = t_1.split(self.block_size, 3)

        stack = [paddle.reshape(t_t, shape=[batch_size, d_height, s_width, s_depth]) for t_t in spl]

        output = paddle.transpose(paddle.stack(stack, 0), perm=[1, 0, 2, 3, 4])
        output = paddle.reshape(paddle.transpose(output, perm=[0, 2, 1, 3, 4]), shape=[batch_size, s_height, s_width, s_depth])
        output = paddle.transpose(output, perm=[0, 3, 1, 2])

        return output


class SpaceToDepth(nn.Layer):

    def __init__(self, block_size):
        super(SpaceToDepth, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = paddle.transpose(input, perm=[0, 2, 1, 3])
        batch_size, s_height, s_width, s_depth = output.shape()

        d_depth = s_depth * self.block_size_sq
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)

        t_1 = output.split(self.block_size, 2)
        stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]

        output = paddle.stack(stack, 1)
        output = paddle.transpose(output, perm=[0, 2, 1, 3])
        output = paddle.transpose(output, perm=[0, 3, 1, 2])

        return output
