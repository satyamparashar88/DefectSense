
from src.decision import generate_inspection_summary


def test_multiple_defects():

    detections = [
        {"defect": "copper", "confidence": 0.88},
        {"defect": "spur", "confidence": 0.86},
        {"defect": "copper", "confidence": 0.91}
    ]

    result = generate_inspection_summary(detections)

    assert result["inspection_status"] == "DEFECTS DETECTED"
    assert result["total_defects"] == 3
    assert result["defect_counts"]["copper"] == 2
    assert result["defect_counts"]["spur"] == 1


def test_no_defects():

    detections = []

    result = generate_inspection_summary(detections)

    assert result["inspection_status"] == "NO DEFECTS DETECTED"
    assert result["total_defects"] == 0
    assert result["average_confidence"] == 0.0


def test_average_confidence():

    detections = [
        {"defect": "open", "confidence": 0.80},
        {"defect": "spur", "confidence": 0.90}
    ]

    result = generate_inspection_summary(detections)

    assert abs(result["average_confidence"] - 0.85) < 0.0001


def test_empty_defect_counts():

    result = generate_inspection_summary([])

    assert result["defect_counts"] == {}
