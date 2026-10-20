"""Input validation, immutable content-addressed preservation and provenance."""
import hashlib
from pathlib import Path
from PIL import Image,ExifTags
from tools.dfdc_storage import ROOT,check_budget
from tools.acquire_dfdc import sha256


def preserve_upload(data,name,kind):
    suffix=Path(name).suffix.lower()
    allowed={'.jpg','.jpeg','.png'} if kind=='Image' else {'.mp4','.mov'}
    if suffix not in allowed:raise ValueError('Unsupported input extension')
    if not data or len(data)>2*1024**3:raise ValueError('Input must contain 1 byte to 2 GiB')
    digest=hashlib.sha256(data).hexdigest();folder=ROOT/'data/dfdc/runtime/uploads';folder.mkdir(parents=True,exist_ok=True)
    path=folder/(digest+suffix)
    if not path.exists():
        check_budget(len(data)+4096)
        with path.open('xb') as stream:stream.write(data)
    elif sha256(path)!=digest:raise ValueError('Preserved input integrity mismatch')
    return path,{'sha256':digest,'original_filename':Path(name).name,'file_type':suffix,'bytes':len(data),'preserved_original':str(path),
                 'interpretation':'Hash and metadata establish input provenance only; neither establishes manipulation'}


def image_metadata(path):
    with Image.open(path) as image:
        if image.format not in {'JPEG','PNG'}:raise ValueError('Decoded image must be JPEG or PNG')
        image.verify()
    with Image.open(path) as image:
        exif={str(ExifTags.TAGS.get(k,k)):str(v)[:500] for k,v in image.getexif().items()}
        return {'decoded_format':image.format,'width':image.width,'height':image.height,'mode':image.mode,
                'EXIF':exif,'available_metadata':{str(k):str(v)[:500] for k,v in image.info.items() if k not in {'exif','icc_profile'}},
                'icc_profile_present':bool(image.info.get('icc_profile'))}


def video_container(path):
    with Path(path).open('rb') as stream:header=stream.read(32)
    atom=header[4:8]
    if len(header)<12 or atom not in {b'ftyp',b'wide',b'mdat',b'moov',b'free',b'skip',b'pnot'}:
        raise ValueError('Input is not an MP4/QuickTime MOV container')
    return {'container_family':'ISO base media / QuickTime',
            'leading_atom':atom.decode('ascii'),
            'major_brand':header[8:12].decode('ascii',errors='replace') if atom==b'ftyp' else None,
            'validation_scope':'Container signature; sampled decoding additionally verifies usable video frames'}
