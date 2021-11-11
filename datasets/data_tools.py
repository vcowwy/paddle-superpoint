import paddle

quan = lambda x: paddle.to_tensor(x.round(), dtype=paddle.int64)


def extrapolate_points(pnts):
    pnts_int = paddle.to_tensor(paddle.to_tensor(pnts, dtype=paddle.int64), dtype=paddle.float32)
    pnts_x, pnts_y = pnts_int[:, 0], pnts_int[:, 1]
    stack_1 = lambda x, y: paddle.stack((x, y), axis=1)
    pnts_ext = paddle.concat((pnts_int, stack_1(pnts_x, pnts_y + 1),
        stack_1(pnts_x + 1, pnts_y), pnts_int + 1), axis=0)
    pnts_res = pnts - pnts_int
    x_res, y_res = pnts_res[:, 0], pnts_res[:, 1]
    res_ext = paddle.concat(((1 - x_res) * (1 - y_res), (1 - x_res) *
        y_res, x_res * (1 - y_res), x_res * y_res), axis=0)
    return pnts_ext, res_ext


def scatter_points(warped_pnts, H, W, res_ext=1):
    warped_labels = paddle.zeros([H, W]).requires_grad_(False)
    warped_labels[quan(warped_pnts)[:, 1], quan(warped_pnts)[:, 0]] = res_ext
    warped_labels = paddle.reshape(warped_labels, shape=[-1, H, W])
    return warped_labels


def get_labels_bi(warped_pnts, H, W):
    from utils.utils import filter_points
    pnts_ext, res_ext = extrapolate_points(warped_pnts)
    pnts_ext, mask = filter_points(pnts_ext, paddle.to_tensor([W, H]),
        return_mask=True)
    res_ext = res_ext[mask]
    warped_labels_bi = scatter_points(pnts_ext, H, W, res_ext=res_ext)
    return warped_labels_bi


def warpLabels(pnts, H, W, homography, bilinear=False):
    from utils.utils import homography_scaling_torch as homography_scaling
    from utils.utils import filter_points
    from utils.utils import warp_points
    if isinstance(pnts, paddle.Tensor):
        pnts = paddle.to_tensor(pnts, dtype=paddle.int64)
    else:
        pnts = paddle.to_tensor(pnts, dtype=paddle.int64)
    warped_pnts = warp_points(paddle.stack((pnts[:, (0)], pnts[:, (1)]),
        axis=1), homography_scaling(homography, H, W))
    outs = {}
    if bilinear == True:
        warped_labels_bi = get_labels_bi(warped_pnts, H, W)
        outs['labels_bi'] = warped_labels_bi
    warped_pnts = filter_points(warped_pnts, paddle.to_tensor([W, H]))
    warped_labels = scatter_points(warped_pnts, H, W, res_ext=1)
    warped_labels_res = paddle.zeros([H, W, 2]).requires_grad_(False)
    warped_labels_res[quan(warped_pnts)[:, (1)], quan(warped_pnts)[:, (0)], :
        ] = warped_pnts - warped_pnts.round()
    outs.update({'labels': warped_labels, 'res': warped_labels_res,
        'warped_pnts': warped_pnts})
    return outs


def np_to_tensor(img, H, W):
    img = paddle.reshape(paddle.to_tensor(img, dtype=paddle.float32), shape=[-1, H, W])
    return img


if __name__ == '__main__':
    main()
