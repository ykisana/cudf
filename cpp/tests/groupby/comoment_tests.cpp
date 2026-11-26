/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/groupby.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

using namespace cudf::test::iterators;

namespace {

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::FIRST_ERROR};
constexpr double NaN{std::numeric_limits<double>::quiet_NaN()};
constexpr double Inf{std::numeric_limits<double>::infinity()};

template <typename T>
using fixed_col   = cudf::test::fixed_width_column_wrapper<T>;
using structs_col = cudf::test::structs_column_wrapper;

auto run_groupby_agg(cudf::column_view const& keys,
                     cudf::column_view const& values,
                     std::unique_ptr<cudf::groupby_aggregation> agg)
{
  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back();
  requests[0].values = values;
  requests[0].aggregations.emplace_back(std::move(agg));

  auto gb                  = cudf::groupby::groupby(cudf::table_view{{keys}});
  auto [out_keys, results] = gb.aggregate(requests);

  auto const order       = cudf::sorted_order(out_keys->view(), {}, {});
  auto const sorted_keys = cudf::gather(out_keys->view(), *order);
  auto const sorted_vals = cudf::gather(cudf::table_view{{results[0].results[0]->view()}}, *order);
  return std::pair(std::move(sorted_keys->release()[0]), std::move(sorted_vals->release()[0]));
}

auto compute_comoment(cudf::column_view const& keys, cudf::column_view const& values)
{
  return run_groupby_agg(
    keys, values, cudf::make_comoment_aggregation<cudf::groupby_aggregation>());
}

auto merge_comoment(std::vector<cudf::column_view> const& keys_list,
                    std::vector<cudf::column_view> const& vals_list)
{
  auto const keys = cudf::concatenate(keys_list);
  auto const vals = cudf::concatenate(vals_list);
  return run_groupby_agg(
    keys->view(), vals->view(), cudf::make_merge_comoment_aggregation<cudf::groupby_aggregation>());
}

template <typename CountT>
structs_col make_state(std::initializer_list<CountT> n,
                       std::initializer_list<double> mx,
                       std::initializer_list<double> my,
                       std::initializer_list<double> m2x,
                       std::initializer_list<double> m2y,
                       std::initializer_list<double> cxy)
{
  auto c_n   = fixed_col<CountT>(n);
  auto c_mx  = fixed_col<double>(mx);
  auto c_my  = fixed_col<double>(my);
  auto c_m2x = fixed_col<double>(m2x);
  auto c_m2y = fixed_col<double>(m2y);
  auto c_cxy = fixed_col<double>(cxy);
  return structs_col{c_n, c_mx, c_my, c_m2x, c_m2y, c_cxy};
}

}  // namespace

template <typename T>
struct GroupbyComomentTypedTest : public cudf::test::BaseFixture {};

using ComomentTestTypes = cudf::test::Types<int32_t, float, double>;
TYPED_TEST_SUITE(GroupbyComomentTypedTest, ComomentTestTypes);

struct GroupbyComomentTest : public cudf::test::BaseFixture {};

TYPED_TEST(GroupbyComomentTypedTest, ComomentAndMergeEquivalence)
{
  using T = TypeParam;

  // Full dataset across 5 groups (plus 1 null key row ignored by default):
  // key 1: valid pairs (1, 2), (3, 1), (4, 5) -> n=3, mx=8/3, my=8/3, m2x=14/3, m2y=26/3, cxy=11/3
  // key 2: constant (2, 5), (2, 5)            -> n=2, mx=2, my=5, m2x=0, m2y=0, cxy=0
  // key 3: singleton (7, 9)                   -> n=1, mx=7, my=9, m2x=0, m2y=0, cxy=0
  // key 4: all pairwise-incomplete            -> n=0, all zeros
  // key 5: negative correlation (1, 6), (2, 4), (3, 2) -> n=3, mx=2, my=4, m2x=2, m2y=8, cxy=-4
  auto const keys = fixed_col<int32_t>{{1, 1, 2, 4, 5, 1, 2, 3, 4, 5, 1, 5, 99}, null_at(12)};
  auto x          = fixed_col<T>{{1, 0, 2, 0, 1, 3, 2, 7, 5, 2, 4, 3, 10}, nulls_at({1, 3})};
  auto y          = fixed_col<T>{{2, 9, 5, 8, 6, 1, 5, 9, 0, 4, 5, 2, 20}, null_at(8)};
  auto const vals = structs_col{x, y};

  auto const exp_keys = fixed_col<int32_t>{1, 2, 3, 4, 5};
  auto const exp_vals = make_state<int64_t>({3, 2, 1, 0, 3},
                                            {8.0 / 3.0, 2.0, 7.0, 0.0, 2.0},
                                            {8.0 / 3.0, 5.0, 9.0, 0.0, 4.0},
                                            {14.0 / 3.0, 0.0, 0.0, 0.0, 2.0},
                                            {26.0 / 3.0, 0.0, 0.0, 0.0, 8.0},
                                            {11.0 / 3.0, 0.0, 0.0, 0.0, -4.0});

  auto const [full_keys, full_vals] = compute_comoment(keys, vals);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(exp_keys, *full_keys, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(exp_vals, *full_vals, verbosity);

  // Split into 3 partitions and verify both 1-step and 2-step tree merges match single-shot output.
  auto const key_slices = cudf::slice(keys, {0, 4, 4, 9, 9, 13});
  auto const val_slices = cudf::slice(vals, {0, 4, 4, 9, 9, 13});
  auto const [k1, v1]   = compute_comoment(key_slices[0], val_slices[0]);
  auto const [k2, v2]   = compute_comoment(key_slices[1], val_slices[1]);
  auto const [k3, v3]   = compute_comoment(key_slices[2], val_slices[2]);

  auto const [one_k, one_v] =
    merge_comoment({k1->view(), k2->view(), k3->view()}, {v1->view(), v2->view(), v3->view()});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(exp_keys, *one_k, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(exp_vals, *one_v, verbosity);

  auto const [k12, v12]     = merge_comoment({k1->view(), k2->view()}, {v1->view(), v2->view()});
  auto const [two_k, two_v] = merge_comoment({k12->view(), k3->view()}, {v12->view(), v3->view()});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(exp_keys, *two_k, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(exp_vals, *two_v, verbosity);
}

TEST_F(GroupbyComomentTest, EmptyInput)
{
  auto const keys = fixed_col<int32_t>{};
  auto x          = fixed_col<int32_t>{};
  auto y          = fixed_col<double>{};
  auto const pair = structs_col{x, y};

  auto const [cm_keys, cm_vals] = compute_comoment(keys, pair);
  auto const exp_i64_state      = make_state<int64_t>({}, {}, {}, {}, {}, {});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(keys, *cm_keys, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(exp_i64_state, *cm_vals, verbosity);

  auto const [m1_keys, m1_vals] = merge_comoment({keys}, {exp_i64_state});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(keys, *m1_keys, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(exp_i64_state, *m1_vals, verbosity);

  auto const exp_f64_state      = make_state<double>({}, {}, {}, {}, {}, {});
  auto const [m2_keys, m2_vals] = merge_comoment({keys}, {exp_f64_state});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(keys, *m2_keys, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(exp_f64_state, *m2_vals, verbosity);
}

TEST_F(GroupbyComomentTest, MixedChildTypesDictionaryAndNullableStruct)
{
  // Mixed child types (int16_t + dictionary<double>) with a nullable parent struct.
  // Row 1 is excluded by parent struct null, leaving pairs (1, 10), (3, 30) for key 1.
  auto const keys = fixed_col<int32_t>{1, 1, 1};
  auto raw_x      = fixed_col<int16_t>{1, 99, 3};
  auto raw_y      = fixed_col<double>{10.0, 999.0, 30.0};
  auto dict_y     = cudf::dictionary::encode(raw_y);
  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(std::make_unique<cudf::column>(raw_x));
  children.push_back(std::move(dict_y));
  auto const pair = structs_col(std::move(children), {true, false, true});

  auto const [out_keys, out_vals] = compute_comoment(keys, pair);
  auto const exp_keys             = fixed_col<int32_t>{1};
  auto const exp_vals             = make_state<int64_t>({2}, {2.0}, {20.0}, {2.0}, {200.0}, {20.0});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(exp_keys, *out_keys, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(exp_vals, *out_vals, verbosity);
}

TEST_F(GroupbyComomentTest, NonFiniteOrderIndependentAndEmptySentinels)
{
  // Merging (+inf, finite) in either order keeps mean_x=+inf, m2_x=NaN, c_xy=NaN while
  // keeping finite y statistics intact; an empty (count=0) sentinel row is ignored.
  for (auto const& mx_order : {std::vector<double>{Inf, 5.0}, std::vector<double>{5.0, Inf}}) {
    auto const keys = fixed_col<int32_t>{1, 1, 1};
    auto counts     = fixed_col<int64_t>{2, 2, 0};
    auto mx         = fixed_col<double>{mx_order[0], mx_order[1], NaN};
    auto my         = fixed_col<double>{1.0, 3.0, Inf};
    auto m2x        = fixed_col<double>{NaN, 4.0, Inf};
    auto m2y        = fixed_col<double>{2.0, 2.0, NaN};
    auto cxy        = fixed_col<double>{NaN, 1.0, Inf};
    auto const vals = structs_col{counts, mx, my, m2x, m2y, cxy};

    auto const [out_k, out_v] = merge_comoment({keys}, {vals});
    auto const exp_k          = fixed_col<int32_t>{1};
    // y: two groups of n=2 with means 1 and 3 and m2=2 each -> combined mean=2, m2 = 2+2+4*(2^2)/4
    // = 8.
    auto const exp_v = make_state<int64_t>({4}, {Inf}, {2.0}, {NaN}, {8.0}, {NaN});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(exp_k, *out_k, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(exp_v, *out_v, verbosity);
  }
}

template <typename CountT>
void test_large_and_mixed_groups()
{
  // Exercise small (5 <= 32), warp (64 <= 1024), and multi-block (2500 > 1024) reduction paths
  // for both COMOMENT (when CountT is int64_t) and MERGE_COMOMENT.
  constexpr int32_t small_n = 5;
  constexpr int32_t warp_n  = 64;
  constexpr int32_t block_n = 2500;
  std::vector<int32_t> keys_h;
  std::vector<CountT> counts_h;
  std::vector<double> xs_h;
  std::vector<double> ys_h;
  std::vector<double> zeros_h;

  auto push_group = [&](int32_t key, int32_t n) {
    for (int32_t i = 0; i < n; ++i) {
      keys_h.push_back(key);
      counts_h.push_back(CountT{1});
      xs_h.push_back(static_cast<double>(i));
      ys_h.push_back(2.0 * static_cast<double>(i) + 1.0);
      zeros_h.push_back(0.0);
    }
  };
  push_group(0, small_n);
  push_group(1, warp_n);
  push_group(2, block_n);

  auto const keys = fixed_col<int32_t>(keys_h.begin(), keys_h.end());
  auto x_col      = fixed_col<double>(xs_h.begin(), xs_h.end());
  auto y_col      = fixed_col<double>(ys_h.begin(), ys_h.end());

  auto expected_mx  = [](int32_t n) { return 0.5 * static_cast<double>(n - 1); };
  auto expected_my  = [](int32_t n) { return static_cast<double>(n); };
  auto expected_m2x = [](int32_t n) {
    auto const d = static_cast<double>(n);
    return d * (d * d - 1.0) / 12.0;
  };
  auto const exp_keys = fixed_col<int32_t>{0, 1, 2};
  auto const exp_vals = make_state<CountT>(
    {static_cast<CountT>(small_n), static_cast<CountT>(warp_n), static_cast<CountT>(block_n)},
    {expected_mx(small_n), expected_mx(warp_n), expected_mx(block_n)},
    {expected_my(small_n), expected_my(warp_n), expected_my(block_n)},
    {expected_m2x(small_n), expected_m2x(warp_n), expected_m2x(block_n)},
    {4.0 * expected_m2x(small_n), 4.0 * expected_m2x(warp_n), 4.0 * expected_m2x(block_n)},
    {2.0 * expected_m2x(small_n), 2.0 * expected_m2x(warp_n), 2.0 * expected_m2x(block_n)});
  constexpr cudf::size_type ulp_tolerance{64};

  if constexpr (std::is_same_v<CountT, int64_t>) {
    auto pair                 = structs_col{x_col, y_col};
    auto const [out_k, out_v] = compute_comoment(keys, pair);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(exp_keys, *out_k, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(exp_vals, *out_v, verbosity, ulp_tolerance);
  }

  auto c_n   = fixed_col<CountT>(counts_h.begin(), counts_h.end());
  auto c_mx  = fixed_col<double>(xs_h.begin(), xs_h.end());
  auto c_my  = fixed_col<double>(ys_h.begin(), ys_h.end());
  auto c_m2x = fixed_col<double>(zeros_h.begin(), zeros_h.end());
  auto c_m2y = fixed_col<double>(zeros_h.begin(), zeros_h.end());
  auto c_cxy = fixed_col<double>(zeros_h.begin(), zeros_h.end());
  auto state = structs_col{c_n, c_mx, c_my, c_m2x, c_m2y, c_cxy};

  auto const [out_k, out_v] = merge_comoment({keys}, {state});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(exp_keys, *out_k, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(exp_vals, *out_v, verbosity, ulp_tolerance);
}

TEST_F(GroupbyComomentTest, LargeAndMixedGroupsInt64Count)
{
  test_large_and_mixed_groups<int64_t>();
}

TEST_F(GroupbyComomentTest, LargeAndMixedGroupsDoubleCount)
{
  test_large_and_mixed_groups<double>();
}

TEST_F(GroupbyComomentTest, ConstantStaysExactAndHugeConstantDoesNotOverflow)
{
  // Welford updates keep non-binary-exact constants (0.1) and huge constants (1e305) exact
  // across uneven partitions without rounding drift or sum overflow.
  constexpr int32_t n = 1500;
  std::vector<int32_t> keys_h(n);
  std::vector<double> xs_h(n);
  std::vector<double> ys_h(n);
  for (int32_t i = 0; i < n; ++i) {
    keys_h[i] = i % 2;
    xs_h[i]   = (i % 2 == 0) ? 0.1 : 1e305;
    ys_h[i]   = static_cast<double>(i % 7);
  }
  auto const keys = fixed_col<int32_t>(keys_h.begin(), keys_h.end());
  auto x          = fixed_col<double>(xs_h.begin(), xs_h.end());
  auto y          = fixed_col<double>(ys_h.begin(), ys_h.end());
  auto const pair = structs_col{x, y};

  auto const key_slices = cudf::slice(keys, {0, 13, 13, 601, 601, n});
  auto const val_slices = cudf::slice(pair, {0, 13, 13, 601, 601, n});
  auto const [k1, v1]   = compute_comoment(key_slices[0], val_slices[0]);
  auto const [k2, v2]   = compute_comoment(key_slices[1], val_slices[1]);
  auto const [k3, v3]   = compute_comoment(key_slices[2], val_slices[2]);
  auto const [mk, mv] =
    merge_comoment({k1->view(), k2->view(), k3->view()}, {v1->view(), v2->view(), v3->view()});

  // Exact check on count, mean_x, m2_x (= 0.0), and c_xy (= 0.0).
  auto const exp_keys = fixed_col<int32_t>{0, 1};
  auto const exp_n    = fixed_col<int64_t>{n / 2, n / 2};
  auto const exp_mx   = fixed_col<double>{0.1, 1e305};
  auto const exp_zero = fixed_col<double>{0.0, 0.0};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(exp_keys, *mk, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(exp_n, mv->child(0), verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(exp_mx, mv->child(1), verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(exp_zero, mv->child(3), verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(exp_zero, mv->child(5), verbosity);
}

TEST_F(GroupbyComomentTest, InvalidInput)
{
  auto const keys = fixed_col<int32_t>{1, 2};

  // Non-struct input rejected (including on 0-row inputs via target_type_impl).
  auto const flat       = fixed_col<double>{1.0, 2.0};
  auto const empty_keys = fixed_col<int32_t>{};
  auto const empty_flat = fixed_col<double>{};
  EXPECT_THROW(compute_comoment(keys, flat), cudf::logic_error);
  EXPECT_THROW(compute_comoment(empty_keys, empty_flat), cudf::logic_error);
  EXPECT_THROW(merge_comoment({keys}, {flat}), cudf::logic_error);
  EXPECT_THROW(merge_comoment({empty_keys}, {empty_flat}), cudf::logic_error);

  // Wrong child count for COMOMENT (!= 2) and MERGE_COMOMENT (!= 6).
  auto a            = fixed_col<double>{1.0, 2.0};
  auto b            = fixed_col<double>{3.0, 4.0};
  auto c            = fixed_col<double>{5.0, 6.0};
  auto const triple = structs_col{a, b, c};
  EXPECT_THROW(compute_comoment(keys, triple), cudf::logic_error);
  EXPECT_THROW(merge_comoment({keys}, {triple}), cudf::logic_error);

  // Non-numeric child for COMOMENT.
  auto str_col        = cudf::test::strings_column_wrapper{"a", "b"};
  auto num_col        = fixed_col<double>{1.0, 2.0};
  auto const bad_pair = structs_col{str_col, num_col};
  EXPECT_THROW(compute_comoment(keys, bad_pair), cudf::logic_error);

  // Narrow int32_t count child rejected by MERGE_COMOMENT.
  auto bad_state =
    make_state<int32_t>({1, 1}, {1.0, 2.0}, {1.0, 2.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0});
  EXPECT_THROW(merge_comoment({keys}, {bad_state}), cudf::logic_error);
}
