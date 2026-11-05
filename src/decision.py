
from collections import Counter


def generate_inspection_summary(detections):

    """
    Generate a PCB inspection summary
    from YOLO detection results.
    """

    total_defects = len(detections)

    defect_counts = Counter(
        item["defect"] for item in detections
    )

    if total_defects > 0:
        inspection_status = "DEFECTS DETECTED"
    else:
        inspection_status = "NO DEFECTS DETECTED"

    average_confidence = (
        sum(item["confidence"] for item in detections)
        / total_defects
        if total_defects > 0
        else 0.0
    )

    return {
        "inspection_status": inspection_status,
        "total_defects": total_defects,
        "defect_counts": dict(defect_counts),
        "average_confidence": average_confidence,
        "detections": detections
    }
