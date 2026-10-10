"""Hồi quy cho helper dữ liệu của src (load_manifest / quick_stats)."""


def test_data_helpers_regression(tmp_path):
    from src.data.dataset import load_manifest, quick_stats

    csv_file = tmp_path / "temp_manifest.csv"
    csv_file.write_text(
        "relative_path,label,class_name,client_id,split,group_id\n"
        "Apple___Apple_scab/img1.JPG,0,Apple___Apple_scab,0,train,leaf_1\n"
        "Apple___Black_rot/img2.JPG,1,Apple___Black_rot,0,train,leaf_2\n",
        encoding="utf-8",
    )

    records = load_manifest(csv_file)
    assert len(records) == 2
    assert records[0]["label"] == 0
    assert records[1]["label"] == 1

    stats = quick_stats(csv_file, num_classes=2)
    assert stats["images"] == 2
    assert stats["classes_present"] == 2
    assert stats["leaf_groups"] == 2
