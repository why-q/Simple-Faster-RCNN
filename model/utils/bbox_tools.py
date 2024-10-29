import numpy as np
import six


def loc2bbox(src_bbox, loc):
    """Decode bounding boxes from bounding box offsets and scales.

    Given bounding box offsets and scales computed by
    :meth:`bbox2loc`, this function decodes the representation to
    coordinates in 2D image coordinates.

    Given scales and offsets :math:`t_y, t_x, t_h, t_w` and a bounding
    box whose center is :math:`(y, x) = p_y, p_x` and size :math:`p_h, p_w`,
    the decoded bounding box's center :math:`\\hat{g}_y`, :math:`\\hat{g}_x`
    and size :math:`\\hat{g}_h`, :math:`\\hat{g}_w` are calculated
    by the following formulas.

    * :math:`\\hat{g}_y = p_h t_y + p_y`
    * :math:`\\hat{g}_x = p_w t_x + p_x`
    * :math:`\\hat{g}_h = p_h \\enp(t_h)`
    * :math:`\\hat{g}_w = p_w \\enp(t_w)`

    The decoding formulas are used in works such as R-CNN [#]_.

    The output is same type as the type of the inputs.

    .. [#] Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik. \
    Rich feature hierarchies for accurate object detection and semantic \
    segmentation. CVPR 2014.

    Args:
        src_bbox (array): A coordinates of bounding boxes.
            Its shape is :math:`(R, 4)`. These coordinates are
            :math:`p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}`.
        loc (array): An array with offsets and scales.
            The shapes of :obj:`src_bbox` and :obj:`loc` should be same.
            This contains values :math:`t_y, t_x, t_h, t_w`.

    Returns:
        array:
        Decoded bounding box coordinates. Its shape is :math:`(R, 4)`. \
        The second axis contains four values \
        :math:`\\hat{g}_{ymin}, \\hat{g}_{xmin},
        \\hat{g}_{ymax}, \\hat{g}_{xmax}`.
    """
    if src_bbox.shape[0] == 0:
        return np.zeros((0, 4), dtype=loc.dtype)

    """
    bbox: [(y_min, x_min, y_max, x_max), ...]
    loc: [(dy, dx, dh, dw, ...), ...]
    """

    # src_bbox: [N, 4], tuple (y_min, x_min, y_max, x_max), N is the number of bboxes
    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)

    # All shape is: [N, ]
    src_height = src_bbox[:, 2] - src_bbox[:, 0]  # the dh in formula
    src_width = src_bbox[:, 3] - src_bbox[:, 1]  # the dw in formula
    src_ctr_y = src_bbox[:, 0] + 0.5 * src_height  # the y in formula
    src_ctr_x = src_bbox[:, 1] + 0.5 * src_width  # the x in formula

    # loc's shape is: [N, M]
    # Usually M is 4, a offset for each bbox.
    dy = loc[:, 0::4]  # [N, M/4]
    dx = loc[:, 1::4]  # [N, M/4]
    dh = loc[:, 2::4]  # [N, M/4]
    dw = loc[:, 3::4]  # [N, M/4]

    # NOTE  Calculate the ctr_y, ctr_x, h, w in formula
    # Att: Use broadcasting to calculate.
    # ctr_y = ...
    # ctr_x = ...
    # h = ...
    # w = ...

    # Trans to (y_min, x_min, y_max, x_max)
    dst_bbox = np.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox[:, 0::4] = ctr_y - 0.5 * h
    dst_bbox[:, 1::4] = ctr_x - 0.5 * w
    dst_bbox[:, 2::4] = ctr_y + 0.5 * h
    dst_bbox[:, 3::4] = ctr_x + 0.5 * w

    return dst_bbox


def bbox2loc(src_bbox, dst_bbox):
    """Encodes the source and the destination bounding boxes to "loc".

    Given bounding boxes, this function computes offsets and scales
    to match the source bounding boxes to the target bounding boxes.
    Mathematcially, given a bounding box whose center is
    :math:`(y, x) = p_y, p_x` and
    size :math:`p_h, p_w` and the target bounding box whose center is
    :math:`g_y, g_x` and size :math:`g_h, g_w`, the offsets and scales
    :math:`t_y, t_x, t_h, t_w` can be computed by the following formulas.

    * :math:`t_y = \\frac{(g_y - p_y)} {p_h}`
    * :math:`t_x = \\frac{(g_x - p_x)} {p_w}`
    * :math:`t_h = \\log(\\frac{g_h} {p_h})`
    * :math:`t_w = \\log(\\frac{g_w} {p_w})`

    The output is same type as the type of the inputs.
    The encoding formulas are used in works such as R-CNN [#]_.

    .. [#] Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik. \
    Rich feature hierarchies for accurate object detection and semantic \
    segmentation. CVPR 2014.

    Args:
        src_bbox (array): An image coordinate array whose shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
            These coordinates are
            :math:`p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}`.
        dst_bbox (array): An image coordinate array whose shape is
            :math:`(R, 4)`.
            These coordinates are
            :math:`g_{ymin}, g_{xmin}, g_{ymax}, g_{xmax}`.

    Returns:
        array:
        Bounding box offsets and scales from :obj:`src_bbox` \
        to :obj:`dst_bbox`. \
        This has shape :math:`(R, 4)`.
        The second axis contains four values :math:`t_y, t_x, t_h, t_w`.

    """

    height = src_bbox[:, 2] - src_bbox[:, 0]
    width = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_y = src_bbox[:, 0] + 0.5 * height
    ctr_x = src_bbox[:, 1] + 0.5 * width

    base_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_y = dst_bbox[:, 0] + 0.5 * base_height
    base_ctr_x = dst_bbox[:, 1] + 0.5 * base_width

    eps = np.finfo(height.dtype).eps
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = np.log(base_height / height)
    dw = np.log(base_width / width)

    loc = np.vstack((dy, dx, dh, dw)).transpose()
    return loc


def bbox_iou(bbox_a, bbox_b):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.

    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    This function accepts both :obj:`numpy.ndarray` and :obj:`cupy.ndarray` as
    inputs. Please note that both :obj:`bbox_a` and :obj:`bbox_b` need to be
    same type.
    
    The output is same type as the type of the inputs.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.

    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    """
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    """

    How the broadcast works:

    ------------------------------
    bbox_a: [N, 4]
    bbox_b: [K, 4]
    bbox_a[:, None, :2]: [N, 1, 2]
    bbox_b[:, :2]: [K, 2]

    -> tl: [N, K, 2]
    ------------------------------

    Here use `tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])` as an example.

    ------ Phase 1 ------ Initialize

    bbox_a: ([[10, 20, 30, 40],
             [15, 25, 35, 45]]) -> (2, 4)

    bbox_b: ([[12, 22, 32, 42],
             [8, 18, 28, 38],
             [16, 26, 36, 46]]) -> (3, 4)

    ------ Phase 2 ------ Add dimension

    bbox_a[:, None, :2]: ([[[10, 20],
                            [15, 25]]]) -> (2, 1, 2)

    bbox_b[:, :2]: ([[12, 22],
                     [8, 18],
                     [16, 26]]) -> (3, 2)

    ------ Phase 3 ------ Broadcast

    bbox_a: ([[[10, 20], [10, 20], [10, 20]],
              [[15, 25], [15, 25], [15, 25]]]) -> (2, 3, 2)

    bbox_b: ([[[12, 22], [8, 18], [16, 26]],
              [[12, 22], [8, 18], [16, 26]]]) -> (2, 3, 2)

    ------ Phase 4 ------ Top left result

    tl: ([[[12, 22], [10, 20], [16, 26]],
          [[15, 25], [15, 25], [16, 26]]]) -> (2, 3, 2)
    """

    # top left
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])  # [N, K, 2]
    # bottom right
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])  # [N, K, 2]

    """
    Here use `area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)` as an example:

    tl: ([[[12, 22], [10, 20], [16, 26]],
          [[15, 25], [13, 23], [19, 29]]]) -> (2, 3, 2)

    br: ([[[30, 40], [28, 38], [36, 46]],
          [[35, 45], [33, 43], [41, 51]]]) -> (2, 3, 2)

    br - tl: ([[[18, 18], [8, 18], [10, 20]],
               [[20, 20], [10, 20], [12, 24]]]) -> (2, 3, 2)

    prod(br - tl, axis=2): ([[324, 144, 200],
                             [400, 100, 288]]) -> (2, 3)

    prod(br - tl, axis=2)): ([[324, 144, 200],
                              [400, 100, 288]]) -> (2, 3)

    (tl < br).all(axis=2): ([[True, True, True],
                             [True, True, True]]) -> (2, 3)

    area_i: ([[324, 144, 200],
              [400, 100, 288]]) -> (2, 3)
    """
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)  # [N, K]
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)  # [N,]
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)  # [K,]

    # NOTE  IoU = area_i / (area_a + area_b - area_i), use broadcasting to calculate.
    # return ...


def __test():
    pass


if __name__ == "__main__":
    __test()


def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
    """Generate anchor base windows by enumerating aspect ratio and scales.

    Generate anchors that are scaled and modified to the given aspect ratios.
    Area of a scaled anchor is preserved when modifying to the given aspect
    ratio.

    :obj:`R = len(ratios) * len(anchor_scales)` anchors are generated by this
    function.
    The :obj:`i * len(anchor_scales) + j` th anchor corresponds to an anchor
    generated by :obj:`ratios[i]` and :obj:`anchor_scales[j]`.

    For example, if the scale is :math:`8` and the ratio is :math:`0.25`,
    the width and the height of the base window will be stretched by :math:`8`.
    For modifying the anchor to the given aspect ratio,
    the height is halved and the width is doubled.

    Args:
        base_size (number): The width and the height of the reference window.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.

    Returns:
        ~numpy.ndarray:
        An array of shape :math:`(R, 4)`.
        Each element is a set of coordinates of a bounding box.
        The second axis corresponds to
        :math:`(y_{min}, x_{min}, y_{max}, x_{max})` of a bounding box.

    """
    py = base_size / 2.0  # 8.0, py is the center y of the anchor
    px = base_size / 2.0  # 8.0, px is the center x of the anchor

    # anchor_base: [(len(ratios) * len(anchor_scales), 4)]
    anchor_base = np.zeros(
        (len(ratios) * len(anchor_scales), 4), dtype=np.float32
    )  # Init a zero matrix

    # Loop all ratios and scales
    # Also can use `for i, j in itertools.product(ratios, anchor_scales)`.
    for i in six.moves.range(len(ratios)):
        for j in six.moves.range(len(anchor_scales)):
            # Example: base_size = 16, anchor_scales[j] = 8, ratios[i] = 0.5
            # then h = 16 * 8 * sqrt(0.5), w = 16 * 8 * sqrt(2)
            # The area of the anchor is 16 * 16 * 8 * 8, which is 64 times of the original area

            # NOTE  b: base_size, s: anchor_scales[j], r: ratios[i]
            # h = ...
            # w = ...

            index = i * len(anchor_scales) + j

            # Example: h = 128, w = 128, px = 8, py = 8
            # Trans to (y_min, x_min, y_max, x_max)
            anchor_base[index, 0] = py - h / 2.0  # 8 - 64 = -56
            anchor_base[index, 1] = px - w / 2.0  # 8 - 64 = -56
            anchor_base[index, 2] = py + h / 2.0  # 8 + 64 = 72
            anchor_base[index, 3] = px + w / 2.0  # 8 + 64 = 72

    return anchor_base
