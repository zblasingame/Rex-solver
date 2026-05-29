from cleanfid import fid

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# Customs LUT
CUSTOMS = {
    'celeba': 'data/celeba_hq_256'
}

SPLITS = {
    'celeba': 'custom',
    'cifar10': 'train',
    'ffhq': 'trainval70k'
}

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('path', type=str, help='Path to generated images')
    parser.add_argument('dataset', type=str, default='celeba', choices=['cifar10', 'celeba', 'ffhq'])
    parser.add_argument('res', type=int, default=256, help='Resolution')
    args = parser.parse_args()


    if not fid.test_stats_exists(args.dataset, mode='clean'):
        fid.make_custom_stats(args.dataset, CUSTOMS[args.dataset], mode='clean')

    score = fid.compute_fid(args.path, dataset_name=args.dataset, dataset_split=SPLITS[args.dataset],
            dataset_res=args.res, mode='clean',
            # model_name='clip_vit_b_32'
        )


    print(f'FID is {score:.5f}')
