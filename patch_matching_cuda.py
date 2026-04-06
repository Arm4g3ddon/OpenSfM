#!/usr/bin/env python3
"""Patch OpenSfM matching.py to add CUDA BruteForce matching support."""
import sys

matching_py = sys.argv[1]

with open(matching_py, 'r') as f:
    code = f.read()

# 1. Add CUDA detection after logger line
cuda_detection = '''
# CUDA support detection
_CUDA_AVAILABLE = False
try:
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        _CUDA_AVAILABLE = True
        logger.info(
            "CUDA available for matching: %d device(s) detected",
            cv2.cuda.getCudaEnabledDeviceCount(),
        )
except (cv2.error, AttributeError):
    pass

if not _CUDA_AVAILABLE:
    logger.info("CUDA not available for matching, using CPU")
'''

code = code.replace(
    'logger: logging.Logger = logging.getLogger(__name__)',
    'logger: logging.Logger = logging.getLogger(__name__)\n' + cuda_detection,
)

# 2. Add CUDA matcher functions before match_brute_force
cuda_funcs = '''
def match_brute_force_cuda(f1, f2, config):
    """CUDA-accelerated brute force matching with Lowe's ratio filtering."""
    assert f1.dtype.type == f2.dtype.type
    if f1.dtype.type == np.uint8:
        norm_type = cv2.NORM_HAMMING
    else:
        norm_type = cv2.NORM_L2
    matcher = cv2.cuda.DescriptorMatcher_createBFMatcher(norm_type)
    gpu_f1 = cv2.cuda_GpuMat()
    gpu_f2 = cv2.cuda_GpuMat()
    gpu_f1.upload(np.ascontiguousarray(f1))
    gpu_f2.upload(np.ascontiguousarray(f2))
    gpu_matches = matcher.knnMatch(gpu_f1, gpu_f2, 2)
    ratio = config["lowes_ratio"]
    good_matches = []
    for match in gpu_matches:
        if match and len(match) == 2:
            m, n = match
            if m.distance < ratio * n.distance:
                good_matches.append(m)
    return _convert_matches_to_vector(good_matches)


def match_brute_force_cuda_symmetric(fi, fj, config):
    """CUDA brute force matching in both directions, keeping consistent matches."""
    matches_ij = [(a, b) for a, b in match_brute_force_cuda(fi, fj, config)]
    matches_ji = [(b, a) for a, b in match_brute_force_cuda(fj, fi, config)]
    return list(set(matches_ij).intersection(set(matches_ji)))


'''

code = code.replace(
    'def match_brute_force(\n',
    cuda_funcs + 'def match_brute_force(\n',
)

# 3. Patch match_brute_force to try CUDA first
code = code.replace(
    '    """\n    assert f1.dtype.type == f2.dtype.type\n    if f1.dtype.type == np.uint8:\n        matcher_type = "BruteForce-Hamming"',
    '    """\n    if _CUDA_AVAILABLE and maskij is None:\n        try:\n            return match_brute_force_cuda(f1, f2, config)\n        except cv2.error as e:\n            logger.warning("CUDA matching failed, falling back to CPU: %s", e)\n    assert f1.dtype.type == f2.dtype.type\n    if f1.dtype.type == np.uint8:\n        matcher_type = "BruteForce-Hamming"',
)

# 4. Patch match_brute_force_symmetric to try CUDA first
code = code.replace(
    '    """\n    matches_ij = [(a, b) for a, b in match_brute_force(fi, fj, config, maskij)]',
    '    """\n    if _CUDA_AVAILABLE and maskij is None:\n        try:\n            return match_brute_force_cuda_symmetric(fi, fj, config)\n        except cv2.error as e:\n            logger.warning("CUDA symmetric matching failed, falling back to CPU: %s", e)\n    matches_ij = [(a, b) for a, b in match_brute_force(fi, fj, config, maskij)]',
)

# 5. Patch FLANN path to use CUDA BruteForce when GPU is available
old_flann = '''    elif matcher_type == "FLANN":
        f1 = feature_loader.instance.load_features_index(
            data,
            im1,
            masked=True,
            segmentation_in_descriptor=segmentation_in_descriptor,
        )
        if not f1:
            return dummy_ret
        feat_data_index1, index1 = f1
        if symmetric_matching:
            f2 = feature_loader.instance.load_features_index(
                data,
                im2,
                masked=True,
                segmentation_in_descriptor=segmentation_in_descriptor,
            )
            if not f2:
                return dummy_ret
            feat_data_index2, index2 = f2

            descriptors1 = feat_data_index1.descriptors
            descriptors2 = feat_data_index2.descriptors
            if descriptors1 is None or descriptors2 is None:
                return dummy_ret

            matches = match_flann_symmetric(
                descriptors1,
                index1,
                descriptors2,
                index2,
                overriden_config,
            )
        else:
            matches = match_flann(index1, d2, overriden_config)'''

new_flann = '''    elif matcher_type == "FLANN":
        # When CUDA is available, use GPU BruteForce instead of CPU FLANN
        # (GPU BF is typically faster than CPU FLANN for SfM descriptor sizes)
        if _CUDA_AVAILABLE:
            try:
                if symmetric_matching:
                    matches = match_brute_force_cuda_symmetric(d1, d2, overriden_config)
                else:
                    matches = match_brute_force_cuda(d1, d2, overriden_config)
            except cv2.error as e:
                logger.warning("CUDA matching failed for FLANN path, falling back to CPU FLANN: %s", e)
                matches = None
        else:
            matches = None

        if matches is None:
            f1 = feature_loader.instance.load_features_index(
                data,
                im1,
                masked=True,
                segmentation_in_descriptor=segmentation_in_descriptor,
            )
            if not f1:
                return dummy_ret
            feat_data_index1, index1 = f1
            if symmetric_matching:
                f2 = feature_loader.instance.load_features_index(
                    data,
                    im2,
                    masked=True,
                    segmentation_in_descriptor=segmentation_in_descriptor,
                )
                if not f2:
                    return dummy_ret
                feat_data_index2, index2 = f2

                descriptors1 = feat_data_index1.descriptors
                descriptors2 = feat_data_index2.descriptors
                if descriptors1 is None or descriptors2 is None:
                    return dummy_ret

                matches = match_flann_symmetric(
                    descriptors1,
                    index1,
                    descriptors2,
                    index2,
                    overriden_config,
                )
            else:
                matches = match_flann(index1, d2, overriden_config)'''

if old_flann in code:
    code = code.replace(old_flann, new_flann)
    print("Patched FLANN path to use CUDA when available")
else:
    print("WARNING: Could not find FLANN block to patch")

with open(matching_py, 'w') as f:
    f.write(code)

print("Patched matching.py with CUDA support successfully")
