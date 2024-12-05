import rasterio
import numpy as np

import cv2
import numpy as np

# def combine_sar_channels(vv_path, vh_path, output_path):
    # print(vv_path)
    # print(vh_path)
    # # Load SAR channels
    # vv = cv2.imread(vv_path, cv2.IMREAD_UNCHANGED)
    # vh = cv2.imread(vh_path, cv2.IMREAD_UNCHANGED)

    # # Normalize to [0, 255]
    # vv_norm = cv2.normalize(vv, None, 0, 255, cv2.NORM_MINMAX)
    # vh_norm = cv2.normalize(vh, None, 0, 255, cv2.NORM_MINMAX)

    # # Stack as RGB-like image
    # sar_rgb = np.dstack((vv_norm, vh_norm, np.zeros_like(vv_norm))).astype(np.uint8)

    # # Save or pass to YOLO
    # cv2.imwrite(output_path, sar_rgb)

def combine_sar_channels(vv_path, vh_path, output_path):
    # Open the VV polarization file
    with rasterio.open(vv_path) as vv:
        vv_data = vv.read(1)  # Read the first band
        vv_meta = vv.meta  # Get metadata for the VV file

    # Open the VH polarization file
    with rasterio.open(vh_path) as vh:
        vh_data = vh.read(1)  # Read the first band

    # Check if the dimensions match
    if vv_data.shape != vh_data.shape:
        raise ValueError("The dimensions of VV and VH images do not match.")

    # Stack VV and VH into a single multi-channel array
    combined_data = np.stack((vv_data, vh_data), axis=0)

    # Update metadata for the combined file
    combined_meta = vv_meta.copy()
    combined_meta.update({
        "count": 2,  # Two bands (VV and VH)
        "dtype": 'float32'  # Ensure the data type remains float32
    })

    # Write the combined file
    with rasterio.open(output_path, "w", **combined_meta) as dst:
        dst.write(combined_data.astype('float32'))  # Cast to float32 explicitly if needed

    print(f"Combined image written to {output_path} with float32 datatype.")