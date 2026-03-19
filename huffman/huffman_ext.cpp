#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <thread>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

std::vector<uint8_t> decode_symbols(
    const std::string& bitstream,
    int64_t num_valid_bits,
    const std::vector<int>& left,
    const std::vector<int>& right,
    const std::vector<int>& symbol,
    int64_t expected_num_symbols) {
  if (num_valid_bits < 0) {
    throw std::invalid_argument("num_valid_bits must be >= 0");
  }
  if (expected_num_symbols < 0) {
    throw std::invalid_argument("expected_num_symbols must be >= 0");
  }
  if (left.size() != right.size() || left.size() != symbol.size()) {
    throw std::invalid_argument("decode trie arrays must share the same length");
  }
  if (left.empty()) {
    throw std::invalid_argument("decode trie must be non-empty");
  }

  const auto* bitstream_ptr = reinterpret_cast<const uint8_t*>(bitstream.data());
  const int64_t bitstream_num_bits = static_cast<int64_t>(bitstream.size()) * 8;
  if (num_valid_bits > bitstream_num_bits) {
    throw std::invalid_argument("num_valid_bits exceeds bitstream length");
  }

  std::vector<uint8_t> out(static_cast<size_t>(expected_num_symbols));

  int node = 0;
  int64_t out_idx = 0;
  int64_t bits_read = 0;

  for (size_t byte_idx = 0; byte_idx < bitstream.size() && bits_read < num_valid_bits; ++byte_idx) {
    const uint8_t byte = bitstream_ptr[byte_idx];
    for (int bit_idx = 7; bit_idx >= 0 && bits_read < num_valid_bits; --bit_idx) {
      const bool bit = ((byte >> bit_idx) & 1U) != 0U;
      node = bit ? right[node] : left[node];
      if (node < 0) {
        throw std::runtime_error("Invalid Huffman bitstream for this codebook");
      }

      const int sym = symbol[node];
      if (sym >= 0) {
        if (out_idx >= expected_num_symbols) {
          throw std::runtime_error("Decoded more symbols than expected");
        }
        out[static_cast<size_t>(out_idx++)] = static_cast<uint8_t>(sym);
        node = 0;
      }
      ++bits_read;
    }
  }

  if (out_idx != expected_num_symbols) {
    throw std::runtime_error(
        "Huffman decode failed: expected " + std::to_string(expected_num_symbols) +
        " symbols, got " + std::to_string(out_idx));
  }
  return out;
}

}  // namespace

py::tuple huffman_encode_symbols_cpp(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> symbols,
    const std::vector<std::string>& codes) {
  if (codes.size() != 256) {
    throw std::invalid_argument("codes must have length 256");
  }

  py::buffer_info symbols_buf = symbols.request();
  const auto* symbols_ptr = static_cast<const uint8_t*>(symbols_buf.ptr);
  const ssize_t num_symbols = symbols_buf.size;

  std::vector<uint8_t> out;
  out.reserve(static_cast<size_t>(num_symbols));

  uint8_t bit_buffer = 0;
  int bit_count = 0;
  int64_t num_valid_bits = 0;

  {
    py::gil_scoped_release release;
    for (ssize_t i = 0; i < num_symbols; ++i) {
      const std::string& code = codes[symbols_ptr[i]];
      if (code.empty()) {
        throw std::invalid_argument("missing Huffman code for one or more symbols");
      }

      for (const char bit : code) {
        bit_buffer = static_cast<uint8_t>((bit_buffer << 1) | (bit == '1' ? 1 : 0));
        ++bit_count;
        ++num_valid_bits;

        if (bit_count == 8) {
          out.push_back(bit_buffer);
          bit_buffer = 0;
          bit_count = 0;
        }
      }
    }
  }

  if (bit_count > 0) {
    bit_buffer = static_cast<uint8_t>(bit_buffer << (8 - bit_count));
    out.push_back(bit_buffer);
  }

  std::string payload(reinterpret_cast<const char*>(out.data()), out.size());
  return py::make_tuple(py::bytes(payload), num_valid_bits);
}

py::array_t<uint8_t> huffman_decode_symbols_cpp(
    py::bytes bitstream,
    int64_t num_valid_bits,
    const std::vector<int>& left,
    const std::vector<int>& right,
    const std::vector<int>& symbol,
    int64_t expected_num_symbols) {
  std::string bitstream_str = bitstream;
  std::vector<uint8_t> decoded;

  {
    py::gil_scoped_release release;
    decoded = decode_symbols(bitstream_str, num_valid_bits, left, right, symbol, expected_num_symbols);
  }

  py::array_t<uint8_t> out(expected_num_symbols);
  py::buffer_info out_buf = out.request();
  auto* out_ptr = static_cast<uint8_t*>(out_buf.ptr);
  std::copy(decoded.begin(), decoded.end(), out_ptr);
  return out;
}

py::list decompress_layer_parallel_cpp(
    const std::vector<py::bytes>& non_exp_bytes_list,
    const std::vector<py::bytes>& bitstream_list,
    const std::vector<int64_t>& num_valid_bits_list,
    const std::vector<int64_t>& expected_num_symbols_list,
    const std::vector<std::vector<int>>& left_nodes_list,
    const std::vector<std::vector<int>>& right_nodes_list,
    const std::vector<std::vector<int>>& symbol_nodes_list,
    int64_t num_threads) {
  const size_t n = non_exp_bytes_list.size();
  if (n == 0) {
    return py::list();
  }
  if (bitstream_list.size() != n || num_valid_bits_list.size() != n ||
      expected_num_symbols_list.size() != n || left_nodes_list.size() != n ||
      right_nodes_list.size() != n || symbol_nodes_list.size() != n) {
    throw std::invalid_argument("all input lists must have the same length");
  }
  if (num_threads <= 0) {
    throw std::invalid_argument("num_threads must be > 0");
  }

  std::vector<std::string> non_exp_strings(n);
  std::vector<std::string> bitstream_strings(n);
  for (size_t i = 0; i < n; ++i) {
    non_exp_strings[i] = py::cast<std::string>(non_exp_bytes_list[i]);
    bitstream_strings[i] = py::cast<std::string>(bitstream_list[i]);
  }

  std::vector<std::vector<uint16_t>> merged_bits(n);
  std::atomic<size_t> next{0};
  std::exception_ptr first_error = nullptr;
  std::mutex err_mu;

  const int64_t worker_count = std::min<int64_t>(num_threads, static_cast<int64_t>(n));
  {
    py::gil_scoped_release release;
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(worker_count));
    for (int64_t t = 0; t < worker_count; ++t) {
      workers.emplace_back([&]() {
        while (true) {
          const size_t idx = next.fetch_add(1);
          if (idx >= n) {
            break;
          }
          try {
            const std::vector<uint8_t> exp = decode_symbols(
                bitstream_strings[idx],
                num_valid_bits_list[idx],
                left_nodes_list[idx],
                right_nodes_list[idx],
                symbol_nodes_list[idx],
                expected_num_symbols_list[idx]);

            const std::string& non_exp = non_exp_strings[idx];
            if (non_exp.size() != exp.size()) {
              throw std::runtime_error("Size mismatch during decompression");
            }

            auto& dst = merged_bits[idx];
            dst.resize(exp.size());
            const auto* non_exp_ptr = reinterpret_cast<const uint8_t*>(non_exp.data());
            for (size_t j = 0; j < exp.size(); ++j) {
              const uint16_t sign = static_cast<uint16_t>(non_exp_ptr[j] >> 7);
              const uint16_t mant = static_cast<uint16_t>(non_exp_ptr[j] & 0x7FU);
              const uint16_t exp16 = static_cast<uint16_t>(exp[j]);
              dst[j] = static_cast<uint16_t>((sign << 15) | (exp16 << 7) | mant);
            }
          } catch (...) {
            std::lock_guard<std::mutex> lock(err_mu);
            if (!first_error) {
              first_error = std::current_exception();
            }
            break;
          }
        }
      });
    }
    for (auto& w : workers) {
      w.join();
    }
  }

  if (first_error) {
    std::rethrow_exception(first_error);
  }

  py::list out;
  for (size_t i = 0; i < n; ++i) {
    const auto& bits = merged_bits[i];
    auto arr = py::array_t<uint16_t>(bits.size());
    py::buffer_info buf = arr.request();
    auto* ptr = static_cast<uint16_t*>(buf.ptr);
    std::copy(bits.begin(), bits.end(), ptr);
    out.append(arr);
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("huffman_encode_symbols_cpp", &huffman_encode_symbols_cpp,
        "Encode uint8 symbols with Huffman code strings (CPU)");
  m.def("huffman_decode_symbols_cpp", &huffman_decode_symbols_cpp,
        "Decode Huffman bitstream to uint8 symbols (CPU)");
  m.def("decompress_layer_parallel_cpp", &decompress_layer_parallel_cpp,
        "Parallel decompress for one layer, output uint16 BF16 bit patterns (CPU)");
}
