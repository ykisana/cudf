# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pytest

import pylibcudf as plc


@pytest.mark.parametrize(
    "agg",
    [plc.aggregation.any, plc.aggregation.all],
)
def test_groupby_unsupported_bool_aggs_raise(agg):
    pa_table = pa.table(
        {
            "foo": pa.array(["a", "a", "b", "b", "c"]),
            "bar": pa.array([True, False, True, True, False]),
        }
    )
    plc_tbl = plc.Table.from_arrow(pa_table)

    groupby = plc.groupby.GroupBy(plc_tbl)
    request = plc.groupby.GroupByRequest(plc_tbl.columns()[0], [agg()])

    with pytest.raises(
        NotImplementedError,
        match=r"Aggregation(.*) aggregations are not supported by groupby",
    ):
        groupby.aggregate([request])


@pytest.mark.parametrize("float_count", [False, True])
def test_groupby_comoment_and_merge_comoment(float_count):
    # Two chunks for keys 1 and 2, with one-sided nulls and mixed int32/float64 children.
    # key 1 valid pairs: (1, 2), (3, 1), (4, 5) -> n=3, mx=8/3, my=8/3, m2x=14/3, m2y=26/3, cxy=11/3
    # key 2 valid pairs: (1, 6), (2, 4), (3, 2) -> n=3, mx=2, my=4, m2x=2, m2y=8, cxy=-4
    chunks = [
        (
            pa.array([1, 1, 2, 2], type=pa.int32()),
            pa.array([1, None, 1, 2], type=pa.int32()),
            pa.array([2.0, 9.0, 6.0, 4.0], type=pa.float64()),
        ),
        (
            pa.array([1, 1, 2, 2], type=pa.int32()),
            pa.array([3, 4, 3, 5], type=pa.int32()),
            pa.array([1.0, 5.0, 2.0, None], type=pa.float64()),
        ),
    ]
    partial_keys = []
    partial_states = []
    for k_arr, x_arr, y_arr in chunks:
        keys = plc.Table([plc.Column.from_arrow(k_arr)])
        pair = plc.Column.struct_from_children(
            [plc.Column.from_arrow(x_arr), plc.Column.from_arrow(y_arr)]
        )
        out_k, (out_t,) = plc.groupby.GroupBy(keys).aggregate(
            [plc.groupby.GroupByRequest(pair, [plc.aggregation.comoment()])]
        )
        state = out_t.columns()[0]
        if float_count:
            children = state.children()
            children[0] = plc.unary.cast(
                children[0], plc.DataType(plc.TypeId.FLOAT64)
            )
            state = plc.Column.struct_from_children(children)
        partial_keys.append(out_k)
        partial_states.append(state)

    merged_keys = plc.concatenate.concatenate(partial_keys)
    merged_in = plc.concatenate.concatenate(partial_states)
    out_keys, (out_table,) = plc.groupby.GroupBy(merged_keys).aggregate(
        [
            plc.groupby.GroupByRequest(
                merged_in, [plc.aggregation.merge_comoment()]
            )
        ]
    )
    rows = plc.Table(
        [out_keys.columns()[0], *out_table.columns()[0].children()]
    ).to_arrow()
    by_key = {
        row["k"]: (
            row["n"],
            row["mx"],
            row["my"],
            row["m2x"],
            row["m2y"],
            row["cxy"],
        )
        for row in rows.rename_columns(
            ["k", "n", "mx", "my", "m2x", "m2y", "cxy"]
        ).to_pylist()
    }
    assert by_key[1] == pytest.approx(
        (3, 8 / 3, 8 / 3, 14 / 3, 26 / 3, 11 / 3)
    )
    assert by_key[2] == pytest.approx((3, 2.0, 4.0, 2.0, 8.0, -4.0))
