import numpy as np
from torch import nn
from torch.nn import functional as F

from model.utils.bbox_tools import generate_anchor_base
from model.utils.creator_tool import ProposalCreator


class RegionProposalNetwork(nn.Module):
    """Region Proposal Network introduced in Faster R-CNN.

    This is Region Proposal Network introduced in Faster R-CNN [#]_.
    This takes features extracted from images and propose
    class agnostic bounding boxes around "objects".

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        in_channels (int): The channel size of input.
        mid_channels (int): The channel size of the intermediate tensor.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.
        feat_stride (int): Stride size after extracting features from an
            image.
        initialW (callable): Initial weight value. If :obj:`None` then this
            function uses Gaussian distribution scaled by 0.1 to
            initialize weight.
            May also be a callable that takes an array and edits its values.
        proposal_creator_params (dict): Key valued paramters for
            :class:`model.utils.creator_tools.ProposalCreator`.

    .. seealso::
        :class:`~model.utils.creator_tools.ProposalCreator`

    """

    def __init__(
        self,
        in_channels=512,
        mid_channels=512,
        ratios=[0.5, 1, 2],
        anchor_scales=[8, 16, 32],
        feat_stride=16,
        proposal_creator_params=dict(),
    ):
        super(RegionProposalNetwork, self).__init__()

        # NOTE  Make sure you understand the function `generate_anchor_base`.
        self.anchor_base = generate_anchor_base(
            anchor_scales=anchor_scales, ratios=ratios
        )
        self.feat_stride = feat_stride  # 16
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)
        n_anchor = self.anchor_base.shape[0]

        # NOTE  `self.conv1` is used to extract features from the input feature map.
        # Use a 3x3 convolution with stride 1 and padding 1.
        # Att: `mid_channels` is a parameter defined in `__init__`.
        # self.conv1 = nn.Conv2d(...)

        # NOTE  `self.score` is used to predict
        # the foreground/background score of each anchor.
        # Use a 1x1 convolution to predict.
        # self.score = nn.Conv2d(...)

        # NOTE  `self.loc` is used to predict
        # the bounding box regression of each anchor.
        # Use a 1x1 convolution to predict.
        # self.loc = nn.Conv2d(...)

        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.0):
        """Forward Region Proposal Network.

        Here are notations.

        * :math:`N` is batch size.
        * :math:`C` channel size of the input.
        * :math:`H` and :math:`W` are height and witdh of the input feature.
        * :math:`A` is number of anchors assigned to each pixel.

        Args:
            x (~torch.autograd.Variable): The Features extracted from images.
                Its shape is :math:`(N, C, H, W)`.
            img_size (tuple of ints): A tuple :obj:`height, width`,
                which contains image size after scaling.
            scale (float): The amount of scaling done to the input images after
                reading them from files.

        Returns:
            (~torch.autograd.Variable, ~torch.autograd.Variable, array, array, array):

            This is a tuple of five following values.

            * **rpn_locs**: Predicted bounding box offsets and scales for \
                anchors. Its shape is :math:`(N, H W A, 4)`.
            * **rpn_scores**:  Predicted foreground scores for \
                anchors. Its shape is :math:`(N, H W A, 2)`.
            * **rois**: A bounding box array containing coordinates of \
                proposal boxes.  This is a concatenation of bounding box \
                arrays from multiple images in the batch. \
                Its shape is :math:`(R', 4)`. Given :math:`R_i` predicted \
                bounding boxes from the :math:`i` th image, \
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            * **roi_indices**: An array containing indices of images to \
                which RoIs correspond to. Its shape is :math:`(R',)`.
            * **anchor**: Coordinates of enumerated shifted anchors. \
                Its shape is :math:`(H W A, 4)`.

        """
        n, _, hh, ww = x.shape
        anchor = _enumerate_shifted_anchor(
            np.array(self.anchor_base), self.feat_stride, hh, ww
        )

        n_anchor = anchor.shape[0] // (hh * ww)
        # h's shape: [N, mid_channels, H, W],
        # H and W are the height and width of the feature map
        h = F.relu(self.conv1(x))

        # get locs
        rpn_locs = self.loc(h)  # [N, mid_channels, H, W] -> [N, 4 * A, H, W]
        rpn_locs = (
            rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        )  # [N, C, H, W] -> [N, H*W*A, 4]

        # get scores
        rpn_scores = self.score(h)  # [N, mid_channels, H, W] -> [N, 2 * A, H, W]
        rpn_scores = rpn_scores.permute(
            0, 2, 3, 1
        ).contiguous()  # [N, 2 * A, H, W] -> [N, H, W, 2 * A]

        rpn_softmax_scores = F.softmax(
            rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4
        )  # [N, H, W, C] -> [N, H, W, A, 2]
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()  # [N, H, W, A]
        rpn_fg_scores = rpn_fg_scores.view(n, -1)  # [N, H*W*A]
        rpn_scores = rpn_scores.view(n, -1, 2)  # [N, H*W*A, 2]

        rois = list()
        roi_indices = list()
        for i in range(n):
            roi = self.proposal_layer(
                rpn_locs[i].cpu().data.numpy(),  # shape: [HWA, 4]
                rpn_fg_scores[i].cpu().data.numpy(),  # shape: [HWA, 2]
                anchor,  # shape: [KA, 4], K = HW, so it is also [HWA, 4]
                img_size,
                scale=scale,
            )
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)

        rois = np.concatenate(rois, axis=0)
        roi_indices = np.concatenate(roi_indices, axis=0)
        return rpn_locs, rpn_scores, rois, roi_indices, anchor


def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    """
    Generate shifted anchors based on a base anchor and feature map dimensions.

    This function creates a set of anchors by shifting the base anchor across
    the entire feature map. It's used in region proposal networks to generate
    candidate bounding boxes.

    Args:
        anchor_base (numpy.ndarray): Base anchor shapes of shape (A, 4),
            where A is the number of anchor types.
        feat_stride (int): The stride of the feature map relative to the original image.
        height (int): Height of the feature map.
        width (int): Width of the feature map.

    Returns:
        numpy.ndarray: An array of shifted anchors with shape (K*A, 4),
        where K is the number of positions on the feature map (height * width),
        and A is the number of anchor types. Each anchor is represented by
        (y_min, x_min, y_max, x_max) coordinates.

    Note:
        The function first generates a grid of shift values, then applies these
        shifts to the base anchors to create a set of anchors for each position
        on the feature map.
    """
    # Example: feat_stride = 16, height = 3, width = 3
    shift_y = np.arange(0, height * feat_stride, feat_stride)  # [0, 16, 32]
    shift_x = np.arange(0, width * feat_stride, feat_stride)  # [0, 16, 32]

    # shift_x: [[0,16,32], [0,16,32], [0,16,32]] - (3, 3)
    # shift_y: [[0,0,0], [16,16,16], [32,32,32]] - (3, 3)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    # shift.shape: (3, 3, 2)
    # shift: ([[[0, 0, 0, 0], [0, 16, 0, 16], [0, 32, 0, 32]],
    #          [[16, 0, 16, 0], [16, 16, 16, 16], [16, 32, 16, 32]],
    #          [[32, 0, 32, 0], [32, 16, 32, 16], [32, 32, 32, 32]]]) -> (3, 3, 4)
    shift = np.stack(
        (shift_y.ravel(), shift_x.ravel(), shift_y.ravel(), shift_x.ravel()), axis=1
    )  # [K, 4], K is the number of the possible shifts

    """
    Here is an example:

    写在前面：
    - `anchor_base` 是最左上角的 anchor 的坐标，格式是 (y1, x1, y2, x2)
    - `shift` 是所有可能的偏移量，显而易见，全都是正的，格式是 (dy1, dx1, dy2, dx2)

    anchor_base: ([[-1, -1, 1, 1], [-2, -2, 2, 2]]) -> (2, 4)
    shift: ([[0, 0, 0, 0], [0, 16, 0, 16], [16, 0, 16, 0], [16, 16, 16, 16]]) -> (4, 4)

    ----- Reshape -----

    anchor_base.reshape((1, A, 4)): ([[[ -1, -1, 1, 1], [-2, -2, 2, 2]]]) -> (1, 2, 4)
    shift.reshape((1, K, 4)).transpose((1, 0, 2)): ([[[0, 0, 0, 0]],
                                                     [[0, 16, 0, 16]],
                                                     [[16, 0, 16, 0]],
                                                     [[16, 16, 16, 16]]]) -> (4, 1, 4)

    ----- Broadcast! -----

    anchor_base: ([[[ -1, -1, 1, 1], [-2, -2, 2, 2]],
                   [[ -1, -1, 1, 1], [-2, -2, 2, 2]],
                   [[ -1, -1, 1, 1], [-2, -2, 2, 2]],
                   [[ -1, -1, 1, 1], [-2, -2, 2, 2]]]) -> (4, 2, 4)
    shift: ([[[0, 0, 0, 0], [0, 0, 0, 0]],
             [[0, 16, 0, 16], [0, 16, 0, 16]],
             [[16, 0, 16, 0], [16, 0, 16, 0]],
             [[16, 16, 16, 16], [16, 16, 16, 16]]]) -> (4, 2, 4)

    ----- Add! -----

    anchor: ([[[-1, -1, 1, 1], [-2, -2, 2, 2]],
              [[-1, 15, 1, 17], [-2, 14, 2, 18]],
              [[15, -1, 17, 1], [14, -2, 18, 2]],
              [[15, 15, 17, 17], [14, 14, 18, 18]]]) -> (4, 2, 4)

    ----- Reshape! -----

    anchor: ([[-1, -1, 1, 1], [-2, -2, 2, 2], [-1, 15, 1, 17], [-2, 14, 2, 18],
              [15, -1, 17, 1], [14, -2, 18, 2], [15, 15, 17, 17], [14, 14, 18, 18]]) -> (8, 4)
    """
    A = anchor_base.shape[0]  # anchor_base: [A, 4], A is the number of anchor types
    K = shift.shape[0]  # K is the number of the possible shifts

    # NOTE  anchor_base: [A, 4], shift: [K, 4] -> [KA, 4]
    # We use broadcasting to add the shift to the anchor_base.
    # anchor = ...

    # Reshape anchor to [K*A, 4]
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)  # [K*A, 4]

    return anchor


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(
            mean
        )  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
