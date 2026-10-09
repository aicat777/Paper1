#!/usr/bin/env python3
"""Verify extracted DIOR image/annotation pairs and the unchanged split files."""
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "DIOR"


def main():
    manifest = json.loads((DATA / "download_manifest.json").read_text())
    archive_report = []
    for item in manifest["archives"]:
        path = DATA / "downloads" / Path(item["path"]).name
        marker = path.with_name(path.name + ".extracted")
        if path.exists():
            assert path.stat().st_size == item["size"], f"Archive size mismatch: {path}"
            assert marker.read_text().strip() == item["lfs"]["oid"], f"Missing verified extraction marker: {path}"
        archive_report.append({"file": path.name, "bytes": item["size"],
                               "sha256": item["lfs"]["oid"], "archive_present": path.exists(), "extracted": True})

    splits = {}
    for name, expected in (("train", 5862), ("val", 5863), ("test", 11738)):
        path = DATA / "ImageSets" / "Main" / f"{name}.txt"
        raw = path.read_bytes()
        entry = next(item for item in manifest["split_files"]
                     if item["path"] == f"DIOR/ImageSets/Main/{name}.txt")
        oid = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        assert oid == entry["oid"], f"Modified split file: {path}"
        ids = raw.decode().splitlines()
        assert len(ids) == expected and len(set(ids)) == expected, f"Invalid {name} split"
        splits[name] = set(ids)
    assert not (splits["train"] & splits["val"] or splits["train"] & splits["test"]
                or splits["val"] & splits["test"]), "Overlapping splits"
    all_ids = set.union(*splits.values())
    assert len(all_ids) == 23463

    images = list(DATA.rglob("*.jpg"))
    image_ids = [path.stem for path in images]
    assert len(image_ids) == len(set(image_ids)) == 23463, "Missing or duplicate image files"
    assert set(image_ids) == all_ids, "Image IDs do not match the original splits"
    xmls = list((DATA / "Annotations" / "Horizontal Bounding Boxes").glob("*.xml"))
    assert len(xmls) == 23463 and {path.stem for path in xmls} == all_ids, "Annotation IDs mismatch"
    by_id = {path.stem: path for path in images}
    image_dirs = {}
    for name, ids in splits.items():
        image_dirs[name] = sorted({str(by_id[identifier].parent.relative_to(DATA)) for identifier in ids})

    classes = Counter()
    split_objects = Counter()
    xml_sizes = Counter()
    id_to_split = {identifier: name for name, ids in splits.items() for identifier in ids}
    for path in xmls:
        root = ET.parse(path).getroot()
        assert root.findtext("filename") == path.stem + ".jpg", f"Annotation filename mismatch: {path}"
        xml_sizes[f"{root.findtext('size/width')}x{root.findtext('size/height')}"] += 1
        for obj in root.findall("object"):
            classes[obj.findtext("name")] += 1
            split_objects[id_to_split[path.stem]] += 1
    assert len(classes) == 20, "Unexpected number of classes"
    report = {
        "status": "verified",
        "mirror_revision": manifest["revision"],
        "archives": archive_report,
        "image_count": len(images),
        "annotation_count": len(xmls),
        "class_count": len(classes),
        "object_count": sum(classes.values()),
        "splits": {name: {"images": len(ids), "objects": split_objects[name],
                          "image_directories": image_dirs[name]} for name, ids in splits.items()},
        "class_object_counts": dict(sorted(classes.items())),
        "xml_declared_image_sizes": dict(sorted(xml_sizes.items())),
        "note": "Original XML contents and split files preserved. Author homepage currently states 192518 instances; this mirror's XML count is recorded above.",
    }
    destination = DATA / "validation_report.json"
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
