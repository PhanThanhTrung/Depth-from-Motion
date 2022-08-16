# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmdet3d.apis import (inference_mono_3d_detector, init_model,
                          show_result_meshlab)

def get_P(calib_file):
    """
    Get matrix P_rect_02 (camera 2 RGB)
    and transform to 3 x 4 matrix
    """
    for line in open(calib_file):
        if 'P2' in line:
            cam_P = line.strip().split(' ')
            cam_P = np.asarray([float(cam_P) for cam_P in cam_P[1:]])
            matrix = np.zeros((3, 4))
            matrix = cam_P.reshape((3, 4))
            return matrix


def main():
    parser = ArgumentParser()
    parser.add_argument('image', help='image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.15, help='bbox score threshold')
    parser.add_argument(
        '--out-dir', type=str, default='demo', help='dir to save results')
    parser.add_argument(
        '--show',
        action='store_true',
        help='show online visualization results')
    parser.add_argument(
        '--snapshot',
        action='store_true',
        help='whether to save online visualization results')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    # test a single image
    result, data = inference_mono_3d_detector(model, args.image, args.ann)
    # show the results
    show_result_meshlab(
        data,
        result,
        args.out_dir,
        args.score_thr,
        show=args.show,
        snapshot=args.snapshot,
        task='mono-det')


if __name__ == '__main__':
    image_path = 
    main()
