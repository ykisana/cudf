/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "groupby/hash/group_reductions.hpp"
#include "groupby/hash/grouped_reductions.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cub/block/block_reduce.cuh>
#include <cuda/iterator>
#include <cuda/std/cmath>
#include <cuda/std/limits>
#include <cuda/std/tuple>
#include <cuda/stream>

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace cudf {
namespace groupby {
namespace detail {
namespace {

template <typename T>
concept comoment_count_type = cudf::is_integral_not_bool<T>() || cudf::is_floating_point<T>();

template <comoment_count_type CountT>
struct comoment_state {
  CountT count;
  double mean_x;
  double mean_y;
  double m2_x;
  double m2_y;
  double c_xy;
};

struct comoment_value {
  double x;
  double y;
  bool valid;

  template <comoment_count_type CountT>
  __device__ operator comoment_state<CountT>() const
  {
    if (!valid) { return {CountT{0}, 0.0, 0.0, 0.0, 0.0, 0.0}; }
    auto constexpr nan = cuda::std::numeric_limits<double>::quiet_NaN();
    bool const xf      = cuda::std::isfinite(x);
    bool const yf      = cuda::std::isfinite(y);
    return {CountT{1}, x, y, xf ? 0.0 : nan, yf ? 0.0 : nan, (xf && yf) ? 0.0 : nan};
  }
};

/**
 * @brief Parallel Chan-Pebay merge operator for bivariate second-order co-moments.
 *
 * See Pebay (2008), "Formulas for Robust, One-Pass Parallel Computation of Covariances
 * and Arbitrary-Order Statistical Moments", Sandia Report SAND2008-6212.
 */
template <comoment_count_type CountT>
struct merge_comoment_op {
  static constexpr auto block_reduce_algorithm = cub::BLOCK_REDUCE_RAKING;

  __device__ comoment_state<CountT> operator()(comoment_state<CountT> const& a,
                                               comoment_state<CountT> const& b) const
  {
    if (b.count == CountT{0}) { return a; }
    if (a.count == CountT{0}) { return b; }

    auto const count   = a.count + b.count;
    auto const n_a     = static_cast<double>(a.count);
    auto const n_b     = static_cast<double>(b.count);
    auto const inv_n   = __drcp_rn(static_cast<double>(count));
    auto constexpr nan = cuda::std::numeric_limits<double>::quiet_NaN();

    bool const x_finite = cuda::std::isfinite(a.mean_x) && cuda::std::isfinite(b.mean_x);
    bool const y_finite = cuda::std::isfinite(a.mean_y) && cuda::std::isfinite(b.mean_y);

    double mean_x{};
    double m2_x{};
    double dx{0.0};
    if (!x_finite) {
      mean_x = a.mean_x + b.mean_x;
      m2_x   = nan;
    } else {
      dx              = b.mean_x - a.mean_x;
      auto const dx_n = dx * inv_n;
      mean_x          = a.mean_x + dx_n * n_b;
      m2_x            = a.m2_x + b.m2_x + dx * dx_n * n_a * n_b;
    }

    double mean_y{};
    double m2_y{};
    double dy_n{0.0};
    if (!y_finite) {
      mean_y = a.mean_y + b.mean_y;
      m2_y   = nan;
    } else {
      auto const dy = b.mean_y - a.mean_y;
      dy_n          = dy * inv_n;
      mean_y        = a.mean_y + dy_n * n_b;
      m2_y          = a.m2_y + b.m2_y + dy * dy_n * n_a * n_b;
    }

    double const c_xy = (x_finite && y_finite) ? (a.c_xy + b.c_xy + dx * dy_n * n_a * n_b) : nan;
    return {count, mean_x, mean_y, m2_x, m2_y, c_xy};
  }

  // Thread-local Welford fast path when folding an individual observation (b.count == 1).
  __device__ comoment_state<CountT> operator()(comoment_state<CountT> const& a,
                                               comoment_value const& b) const
  {
    if (!b.valid) { return a; }
    if (a.count == CountT{0}) { return static_cast<comoment_state<CountT>>(b); }
    auto const dx = b.x - a.mean_x;
    auto const dy = b.y - a.mean_y;
    if (!cuda::std::isfinite(dx) || !cuda::std::isfinite(dy)) {
      return (*this)(a, static_cast<comoment_state<CountT>>(b));
    }
    auto const count = a.count + CountT{1};
    auto const inv_n = __drcp_rn(static_cast<double>(count));
    auto const dx_n  = dx * inv_n;
    auto const dy_n  = dy * inv_n;
    auto const n_a   = static_cast<double>(a.count);
    return {count,
            a.mean_x + dx_n,
            a.mean_y + dy_n,
            a.m2_x + dx * dx_n * n_a,
            a.m2_y + dy * dy_n * n_a,
            a.c_xy + dx * dy_n * n_a};
  }
};

struct double_accessor {
  template <typename T>
  __device__ double operator()(column_device_view const& col, size_type i) const
  {
    if constexpr (cudf::is_numeric<T>()) { return static_cast<double>(col.element<T>(i)); }
    return 0.0;
  }
};

struct grouped_comoment_pair {
  size_type const* rows;
  bitmask_type const* struct_null_mask;
  size_type struct_offset;
  column_device_view d_x;
  column_device_view d_y;
  bool x_has_nulls;
  bool y_has_nulls;
  bool x_is_dict;
  bool y_is_dict;

  __device__ static double read_double(column_device_view const& view, size_type i, bool is_dict)
  {
    if (is_dict) {
      i               = static_cast<size_type>(view.element<dictionary32>(i));
      auto const keys = view.child(dictionary_column_view::keys_column_index);
      return type_dispatcher(keys.type(), double_accessor{}, keys, i);
    }
    return type_dispatcher(view.type(), double_accessor{}, view, i);
  }

  __device__ comoment_value operator()(size_type position) const
  {
    auto const row = rows[position];
    if ((struct_null_mask != nullptr && !bit_is_set(struct_null_mask, struct_offset + row)) ||
        (x_has_nulls && d_x.is_null_nocheck(row)) || (y_has_nulls && d_y.is_null_nocheck(row))) {
      return {0.0, 0.0, false};
    }
    return {read_double(d_x, row, x_is_dict), read_double(d_y, row, y_is_dict), true};
  }
};

template <comoment_count_type CountT>
struct grouped_comoment_state {
  size_type const* rows;
  bitmask_type const* struct_null_mask;
  size_type struct_offset;
  bitmask_type const* count_null_mask;
  size_type count_offset;
  CountT const* counts;
  double const* means_x;
  double const* means_y;
  double const* m2s_x;
  double const* m2s_y;
  double const* cxys;

  __device__ comoment_state<CountT> operator()(size_type position) const
  {
    auto const row = rows[position];
    if ((struct_null_mask != nullptr && !bit_is_set(struct_null_mask, struct_offset + row)) ||
        (count_null_mask != nullptr && !bit_is_set(count_null_mask, count_offset + row))) {
      return {CountT{0}, 0.0, 0.0, 0.0, 0.0, 0.0};
    }
    auto const count = counts[row];
    if (count == CountT{0}) { return {CountT{0}, 0.0, 0.0, 0.0, 0.0, 0.0}; }
    return {count, means_x[row], means_y[row], m2s_x[row], m2s_y[row], cxys[row]};
  }
};

template <comoment_count_type CountT>
struct split_comoment_state {
  __device__ cuda::std::tuple<CountT, double, double, double, double, double> operator()(
    comoment_state<CountT> const& state) const
  {
    return {state.count, state.mean_x, state.mean_y, state.m2_x, state.m2_y, state.c_xy};
  }
};

template <comoment_count_type CountT, typename ValueIterator>
std::unique_ptr<column> reduce_comoment_states(ValueIterator values,
                                               hash::grouped_rows const& grouped,
                                               cuda::stream_ref stream,
                                               cudf::memory_resources mr)
{
  auto const num_groups = static_cast<size_type>(grouped.offsets.size() - 1);
  auto const output_mr  = mr.get_output_mr();

  auto out_count = make_numeric_column(
    data_type{type_to_id<CountT>()}, num_groups, mask_state::UNALLOCATED, stream, output_mr);
  auto out_mean_x = make_numeric_column(
    data_type{type_id::FLOAT64}, num_groups, mask_state::UNALLOCATED, stream, output_mr);
  auto out_mean_y = make_numeric_column(
    data_type{type_id::FLOAT64}, num_groups, mask_state::UNALLOCATED, stream, output_mr);
  auto out_m2_x = make_numeric_column(
    data_type{type_id::FLOAT64}, num_groups, mask_state::UNALLOCATED, stream, output_mr);
  auto out_m2_y = make_numeric_column(
    data_type{type_id::FLOAT64}, num_groups, mask_state::UNALLOCATED, stream, output_mr);
  auto out_c_xy = make_numeric_column(
    data_type{type_id::FLOAT64}, num_groups, mask_state::UNALLOCATED, stream, output_mr);

  if (num_groups > 0) {
    auto const output = cuda::transform_output_iterator{
      cuda::make_zip_iterator(out_count->mutable_view().begin<CountT>(),
                              out_mean_x->mutable_view().begin<double>(),
                              out_mean_y->mutable_view().begin<double>(),
                              out_m2_x->mutable_view().begin<double>(),
                              out_m2_y->mutable_view().begin<double>(),
                              out_c_xy->mutable_view().begin<double>()),
      split_comoment_state<CountT>{}};
    hash::reduce_groups(grouped,
                        values,
                        output,
                        merge_comoment_op<CountT>{},
                        comoment_state<CountT>{CountT{0}, 0.0, 0.0, 0.0, 0.0, 0.0},
                        stream,
                        mr);
  }

  std::vector<std::unique_ptr<column>> children;
  children.reserve(6);
  children.emplace_back(std::move(out_count));
  children.emplace_back(std::move(out_mean_x));
  children.emplace_back(std::move(out_mean_y));
  children.emplace_back(std::move(out_m2_x));
  children.emplace_back(std::move(out_m2_y));
  children.emplace_back(std::move(out_c_xy));
  return make_structs_column(
    num_groups,
    std::move(children),
    0,
    cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, output_mr),
    stream,
    output_mr);
}

template <comoment_count_type CountT>
std::unique_ptr<column> merge_comoment_impl(column_view const& values,
                                            hash::grouped_rows const& grouped,
                                            cuda::stream_ref stream,
                                            cudf::memory_resources mr)
{
  auto const partials = structs_column_view{values};
  auto const counts   = partials.get_sliced_child(0, stream);
  auto const means_x  = partials.get_sliced_child(1, stream);
  auto const means_y  = partials.get_sliced_child(2, stream);
  auto const m2s_x    = partials.get_sliced_child(3, stream);
  auto const m2s_y    = partials.get_sliced_child(4, stream);
  auto const cxys     = partials.get_sliced_child(5, stream);

  auto const state_values = cudf::detail::make_counting_transform_iterator(
    0,
    grouped_comoment_state<CountT>{grouped.rows.data(),
                                   values.has_nulls() ? values.null_mask() : nullptr,
                                   values.offset(),
                                   counts.has_nulls() ? counts.null_mask() : nullptr,
                                   counts.offset(),
                                   counts.begin<CountT>(),
                                   means_x.begin<double>(),
                                   means_y.begin<double>(),
                                   m2s_x.begin<double>(),
                                   m2s_y.begin<double>(),
                                   cxys.begin<double>()});
  return reduce_comoment_states<CountT>(state_values, grouped, stream, mr);
}

}  // namespace

std::unique_ptr<column> group_comoment(column_view const& values,
                                       hash::grouped_rows const& grouped,
                                       cuda::stream_ref stream,
                                       cudf::memory_resources mr)
{
  CUDF_EXPECTS(values.type().id() == type_id::STRUCT,
               "Input to `group_comoment` must be a structs column.");
  CUDF_EXPECTS(values.num_children() == 2,
               "Input to `group_comoment` must be a structs column having 2 children columns.");

  auto const structs_values = structs_column_view{values};
  auto const child_x        = structs_values.get_sliced_child(0, stream);
  auto const child_y        = structs_values.get_sliced_child(1, stream);

  auto const base_type = [](column_view const& c) {
    return cudf::is_dictionary(c.type()) ? dictionary_column_view{c}.keys().type() : c.type();
  };
  CUDF_EXPECTS(cudf::is_numeric(base_type(child_x)) && cudf::is_numeric(base_type(child_y)),
               "Input to `group_comoment` must have numeric child columns.");

  auto const d_x         = column_device_view::create(child_x, stream, mr.get_temporary_mr());
  auto const d_y         = column_device_view::create(child_y, stream, mr.get_temporary_mr());
  auto const pair_values = cudf::detail::make_counting_transform_iterator(
    0,
    grouped_comoment_pair{grouped.rows.data(),
                          values.has_nulls() ? values.null_mask() : nullptr,
                          values.offset(),
                          *d_x,
                          *d_y,
                          child_x.has_nulls(),
                          child_y.has_nulls(),
                          cudf::is_dictionary(child_x.type()),
                          cudf::is_dictionary(child_y.type())});
  return reduce_comoment_states<int64_t>(pair_values, grouped, stream, mr);
}

std::unique_ptr<column> group_merge_comoment(column_view const& values,
                                             hash::grouped_rows const& grouped,
                                             cuda::stream_ref stream,
                                             cudf::memory_resources mr)
{
  CUDF_EXPECTS(values.type().id() == type_id::STRUCT,
               "Input to `group_merge_comoment` must be a structs column.");
  CUDF_EXPECTS(
    values.num_children() == 6,
    "Input to `group_merge_comoment` must be a structs column having 6 children columns.");

  auto const count_type_id = values.child(0).type().id();
  CUDF_EXPECTS((count_type_id == type_id::INT64 || count_type_id == type_id::FLOAT64) &&
                 values.child(1).type().id() == type_id::FLOAT64 &&
                 values.child(2).type().id() == type_id::FLOAT64 &&
                 values.child(3).type().id() == type_id::FLOAT64 &&
                 values.child(4).type().id() == type_id::FLOAT64 &&
                 values.child(5).type().id() == type_id::FLOAT64,
               "Input to `group_merge_comoment` has invalid children types.");

  return count_type_id == type_id::INT64 ? merge_comoment_impl<int64_t>(values, grouped, stream, mr)
                                         : merge_comoment_impl<double>(values, grouped, stream, mr);
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
