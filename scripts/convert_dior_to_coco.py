#!/usr/bin/env python3
"""Convert unchanged DIOR HBB XMLs and original splits to COCO JSON."""
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'datasets' / 'DIOR'
CLASSES = (
    'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
    'chimney', 'dam', 'expressway-service-area', 'expressway-toll-station',
    'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium',
    'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill')


def main():
    output = DATA / 'coco_annotations'
    output.mkdir(exist_ok=True)
    report = {'coordinate_convention': 'MMDetection VOC converter: subtract 1 from all XYXY coordinates; COCO width=xmax-xmin, height=ymax-ymin',
              'image_sizes': 'Read from JPEG headers, original XML left unchanged',
              'classes': CLASSES, 'splits': {}}
    for split in ('train', 'val', 'test'):
        ids = (DATA / 'ImageSets' / 'Main' / f'{split}.txt').read_text().splitlines()
        directory = 'JPEGImages-test' if split == 'test' else 'JPEGImages-trainval'
        document = {'info': {'description': 'DIOR HBB original split'},
                    'licenses': [], 'images': [], 'annotations': [],
                    'categories': [{'id': index + 1, 'name': name} for index, name in enumerate(CLASSES)]}
        category_ids = {name: index + 1 for index, name in enumerate(CLASSES)}
        stats = Counter()
        for identifier in ids:
            image_path = DATA / directory / f'{identifier}.jpg'
            with Image.open(image_path) as image:
                width, height = image.size
            xml = ET.parse(DATA / 'Annotations' / 'Horizontal Bounding Boxes' / f'{identifier}.xml').getroot()
            image_id = int(identifier)
            document['images'].append({'id': image_id, 'file_name': f'{directory}/{identifier}.jpg',
                                       'width': width, 'height': height})
            if (int(xml.findtext('size/width')), int(xml.findtext('size/height'))) != (width, height):
                stats['xml_size_mismatches'] += 1
            for obj in xml.findall('object'):
                name = obj.findtext('name').lower()
                box = obj.find('bndbox')
                x1, y1, x2, y2 = [float(box.findtext(key)) - 1 for key in ('xmin', 'ymin', 'xmax', 'ymax')]
                if x2 <= x1 or y2 <= y1:
                    # CocoDataset also excludes boxes with zero/negative area.
                    stats['degenerate_boxes_excluded'] += 1
                    continue
                difficult = int(obj.findtext('difficult', '0'))
                document['annotations'].append({'id': len(document['annotations']) + 1,
                    'image_id': image_id, 'category_id': category_ids[name],
                    'bbox': [x1, y1, x2 - x1, y2 - y1], 'area': (x2 - x1) * (y2 - y1),
                    'iscrowd': difficult, 'ignore': 0, 'segmentation': []})
                stats['difficult_objects'] += difficult
        destination = output / f'instances_{split}.json'
        destination.write_text(json.dumps(document, separators=(',', ':')) + '\n')
        report['splits'][split] = {'images': len(ids), 'annotations': len(document['annotations']), **stats}
        print(f'{split}: {len(ids)} images, {len(document["annotations"])} annotations', flush=True)
    (output / 'conversion_report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
