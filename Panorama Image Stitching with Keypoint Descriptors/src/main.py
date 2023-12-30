import argparse
from util import read_images
from stitch import stitch_all
import cv2
import os
import time
import pandas as pd
def main(images, outdir, verbose):
    results_data = {'Dataset': [], 'Method': [], 'Time': []}

    for dataset_name, dataset in list(images.items()):
        if verbose:
            print(f'{dataset_name.upper()} is being stitched...')
            print('----------------------------------------')
            print('Stitching Using SIFT Features.')
            print('----------------------------------------')

        start_time = time.time()
        sift_panorama, _ = stitch_all(images=dataset, 
                       dataset_name=dataset_name, 
                       method='SIFT',
                       outdir=outdir,
                       verbose=verbose)
        end_time = time.time()
        sift_time = end_time - start_time

        results_data['Dataset'].append(dataset_name)
        results_data['Method'].append('SIFT')
        results_data['Time'].append(sift_time)

        # Save the final panorama
        final_pano_path = os.path.join(outdir, 'SIFT', dataset_name, 'result.png')
        cv2.imwrite(final_pano_path, sift_panorama)

        if verbose:
            print('Stitching using SIFT Features Done.')
            print('----------------------------------------')
            print('Stitching Using SURF Features.')
            print('----------------------------------------')

        start_time = time.time()
        surf_panorama, _ = stitch_all(images=dataset, 
                       dataset_name=dataset_name, 
                       method='SURF',
                       outdir=outdir,
                       verbose=verbose)
        end_time = time.time()
        surf_time = end_time - start_time

        results_data['Dataset'].append(dataset_name)
        results_data['Method'].append('SURF')
        results_data['Time'].append(surf_time)

        # Save the final panorama
        final_pano_path = os.path.join(outdir, 'SURF', dataset_name, 'result.png')
        cv2.imwrite(final_pano_path, surf_panorama)

        if verbose:
            print('Stitching using SURF Features Done.')
            print('----------------------------------------')

    # Save results to Excel
    results_df = pd.DataFrame(results_data)
    excel_path = os.path.join(outdir, 'stitching_results.xlsx')
    results_df.to_excel(excel_path, index=False)

    if verbose:
        print(f'Results saved to {excel_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A simple script with command-line arguments.')
    parser.add_argument('--dataset', help='Path to the dataset')
    parser.add_argument('-o', '--output', help='Path to the output file', default='results')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose mode')
    args = parser.parse_args()
    
    images = read_images(args.dataset)
    verbose = args.verbose
    outdir = args.output

    main(images, outdir, verbose)