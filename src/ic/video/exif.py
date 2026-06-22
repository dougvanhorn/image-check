# Extract exif of passed files.

import exiftool

def get_exif(files):
    metadata = None
    with exiftool.ExifToolHelper() as et:
        metadata = et.get_metadata(files)
    return metadata
