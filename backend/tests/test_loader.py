import asyncio
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from pydantic import BaseModel

from src.data.loader import DataLoader
from src.analytics.schema import TimeColumnDefinition, MetricDefinition, SemanticSchema


class DummySchema(SemanticSchema):
    pass


@pytest.fixture()
def tmp_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame({
        "order_date": ["2024-01-01", "2024-01-02"],
        "amount": [10, 20],
        "product": ["A", "B"],
    })
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture()
def tmp_excel(tmp_path: Path) -> Path:
    pytest.importorskip("openpyxl")
    df = pd.DataFrame({
        "order_date": ["2024-02-01", "2024-02-02"],
        "amount": [5, 15],
        "product": ["X", "Y"],
    })
    p = tmp_path / "data.xlsx"
    # engine auto-detected
    df.to_excel(p, index=False)
    return p


@pytest.fixture()
def tmp_json_array(tmp_path: Path) -> Path:
    data = [
        {"order_date": "2024-03-01", "amount": 7, "product": "C"},
        {"order_date": "2024-03-02", "amount": 9, "product": "D"},
    ]
    p = tmp_path / "data.json"
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f)
    return p


@pytest.fixture()
def tmp_json_lines(tmp_path: Path) -> Path:
    lines = [
        {"order_date": "2024-04-01", "amount": 12, "product": "E"},
        {"order_date": "2024-04-02", "amount": 18, "product": "F"},
    ]
    p = tmp_path / "data.ldjson"
    with p.open("w", encoding="utf-8") as f:
        for obj in lines:
            f.write(json.dumps(obj) + "\n")
    return p


@pytest.fixture()
def schema_stub() -> SemanticSchema:
    return SemanticSchema(
        dataset_id="ds",
        dataset_name="Dataset",
        time_column=TimeColumnDefinition(column_name="order_date"),
        metrics=[MetricDefinition(name="revenue", source_column="amount")],
    )


@pytest.fixture()
def patched_loader(monkeypatch: pytest.MonkeyPatch, schema_stub: SemanticSchema) -> DataLoader:
    # Patch normalizer.clean_dataframe to just parse dates and return df as-is otherwise
    def fake_clean(df: pd.DataFrame) -> pd.DataFrame:
        if "order_date" in df.columns:
            try:
                df["order_date"] = pd.to_datetime(df["order_date"])  # ensure datetime for _get_date_range
            except Exception:
                pass
        return df

    # Patch detector.detect_schema to async return provided stub
    async def fake_detect_schema(**kwargs):
        return schema_stub

    loader = DataLoader()
    monkeypatch.setattr(loader.normalizer, "clean_dataframe", fake_clean)
    monkeypatch.setattr(loader.detector, "detect_schema", fake_detect_schema)
    return loader


def test_load_csv_success_with_auto_detect(tmp_csv: Path, patched_loader: DataLoader, schema_stub: SemanticSchema):
    res = asyncio.run(patched_loader.load_csv(str(tmp_csv), dataset_id="csv1", dataset_name="CSV One", auto_detect=True))
    assert res["success"] is True
    assert res["dataset_id"] == "csv1"
    assert res["dataset_name"] == "CSV One"
    assert res["rows"] == 2
    assert "amount" in res["columns"]
    assert res["schema"] is not None
    # date_range present because we patched dates to datetime and schema has time_column
    assert res["date_range"] == {
        "start": patched_loader.get_dataset("csv1")["order_date"].min().isoformat(),
        "end": patched_loader.get_dataset("csv1")["order_date"].max().isoformat(),
    }


def test_load_csv_handles_empty_file(tmp_path: Path, patched_loader: DataLoader):
    empty = tmp_path / "empty.csv"
    pd.DataFrame().to_csv(empty, index=False)
    res = asyncio.run(patched_loader.load_csv(str(empty), dataset_id="empty"))
    assert res["success"] is False
    # Updated to match actual pandas error message
    assert "no columns to parse from file" in res["error"].lower() or "empty" in res["error"].lower()


def test_load_excel_success_and_sheet_name(tmp_excel: Path, patched_loader: DataLoader):
    res = asyncio.run(patched_loader.load_excel(str(tmp_excel), dataset_id="xls1", sheet_name=None))
    assert res["success"] is True
    assert res["dataset_id"] == "xls1"
    assert res["rows"] == 2
    assert res["sheet_name"] is None
    assert patched_loader.get_dataset("xls1") is not None


def test_load_json_array_and_lines(tmp_json_array: Path, tmp_json_lines: Path, patched_loader: DataLoader):
    # array
    res1 = asyncio.run(patched_loader.load_json(str(tmp_json_array), dataset_id="json_arr"))
    assert res1["success"] is True
    assert res1["rows"] == 2

    # lines
    res2 = asyncio.run(patched_loader.load_json(str(tmp_json_lines), dataset_id="json_lines"))
    assert res2["success"] is True
    assert res2["rows"] == 2


def test_load_without_auto_detect_stores_no_schema(tmp_csv: Path, patched_loader: DataLoader):
    res = asyncio.run(patched_loader.load_csv(str(tmp_csv), dataset_id="csv_no_schema", auto_detect=False))
    assert res["success"] is True
    assert res["schema"] is None
    assert patched_loader.get_schema("csv_no_schema") is None


def test_dataset_management_list_and_delete(patched_loader: DataLoader):
    # Seed
    df = pd.DataFrame({"a": [1, 2]})
    patched_loader.data_store["d1"] = df
    patched_loader.schema_store["d1"] = SemanticSchema(
        dataset_id="d1",
        dataset_name="D1",
        time_column=TimeColumnDefinition(column_name="a"),
        metrics=[MetricDefinition(name="cnt", source_column="a")],
    )

    summary = patched_loader.list_datasets()
    assert summary["count"] == 1
    assert summary["datasets"][0]["dataset_id"] == "d1"
    assert summary["datasets"][0]["has_schema"] is True

    deleted = patched_loader.delete_dataset("d1")
    assert deleted is True
    assert patched_loader.get_dataset("d1") is None
    assert patched_loader.get_schema("d1") is None


def test_json_error_reporting(tmp_path: Path, patched_loader: DataLoader):
    # Write invalid JSON (neither array nor valid lines)
    bad = tmp_path / "bad.json"
    bad.write_text("{not: json}", encoding="utf-8")
    res = asyncio.run(patched_loader.load_json(str(bad), dataset_id="bad"))
    assert res["success"] is False
    assert res["dataset_id"] == "bad"
    assert isinstance(res["error"], str)


def test_load_csv_normalizes_data_when_auto_detect_disabled(tmp_csv: Path, monkeypatch: pytest.MonkeyPatch):
    # Create a fresh loader and patch normalizer to add a marker column
    loader = DataLoader()

    def marker_clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["normalized"] = True
        return df

    monkeypatch.setattr(loader.normalizer, "clean_dataframe", marker_clean)

    res = asyncio.run(loader.load_csv(str(tmp_csv), dataset_id="csv_norm", auto_detect=False))
    assert res["success"] is True
    # Normalizer should have added the marker column
    assert "normalized" in res["columns"]
    assert "normalized" in loader.get_dataset("csv_norm").columns
    # No schema and no date_range when auto_detect is False
    assert res["schema"] is None
    assert res["date_range"] is None


def test_load_excel_handles_empty_file(tmp_path: Path, patched_loader: DataLoader):
    pytest.importorskip("openpyxl")  # Ensure openpyxl is available
    empty_xlsx = tmp_path / "empty.xlsx"
    pd.DataFrame().to_excel(empty_xlsx, index=False)
    res = asyncio.run(patched_loader.load_excel(str(empty_xlsx), dataset_id="empty_xlsx"))
    assert res["success"] is False
    # Updated to match actual pandas error message
    assert "no columns to parse from file" in res["error"].lower() or "empty" in res["error"].lower()


def test_date_range_none_when_time_column_missing(tmp_csv: Path, monkeypatch: pytest.MonkeyPatch):
    # Detector returns a schema whose time column does not exist in the data
    from src.analytics.schema import TimeColumnDefinition, MetricDefinition, SemanticSchema

    missing_time_schema = SemanticSchema(
        dataset_id="ds_missing_time",
        dataset_name="MissingTime",
        time_column=TimeColumnDefinition(column_name="does_not_exist"),
        metrics=[MetricDefinition(name="revenue", source_column="amount")],
    )

    loader = DataLoader()

    async def fake_detect_schema(**kwargs):
        return missing_time_schema

    # Keep normalizer simple to avoid altering columns related to time
    def passthrough(df: pd.DataFrame) -> pd.DataFrame:
        return df

    monkeypatch.setattr(loader.detector, "detect_schema", fake_detect_schema)
    monkeypatch.setattr(loader.normalizer, "clean_dataframe", passthrough)

    res = asyncio.run(loader.load_csv(str(tmp_csv), dataset_id="csv_missing_time", auto_detect=True))
    assert res["success"] is True
    assert res["schema"] is not None
    # Since time column is missing from the frame, date_range should be None
    assert res["date_range"] is None


def test_load_json_array_no_auto_detect_has_no_schema_and_no_date_range(tmp_json_array: Path, patched_loader: DataLoader):
    res = asyncio.run(patched_loader.load_json(str(tmp_json_array), dataset_id="json_no_auto", auto_detect=False))
    assert res["success"] is True
    assert res["schema"] is None
    # When auto_detect is False, date_range should be None
    assert res.get("date_range") is None


def test_delete_dataset_nonexistent_returns_false():
    loader = DataLoader()
    # Ensure stores are empty
    assert loader.list_datasets()["count"] == 0
    deleted = loader.delete_dataset("missing")
    assert deleted is False