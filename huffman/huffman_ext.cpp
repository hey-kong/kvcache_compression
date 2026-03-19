#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

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

  std::string bitstream_str = bitstream;
  const auto* bitstream_ptr = reinterpret_cast<const uint8_t*>(bitstream_str.data());
  const int64_t bitstream_num_bits = static_cast<int64_t>(bitstream_str.size()) * 8;
  if (num_valid_bits > bitstream_num_bits) {
    throw std::invalid_argument("num_valid_bits exceeds bitstream length");
  }

  py::array_t<uint8_t> out(expected_num_symbols);
  py::buffer_info out_buf = out.request();
  auto* out_ptr = static_cast<uint8_t*>(out_buf.ptr);

  int node = 0;
  int64_t out_idx = 0;
  int64_t bits_read = 0;

  {
    py::gil_scoped_release release;
    for (size_t byte_idx = 0; byte_idx < bitstream_str.size() && bits_read < num_valid_bits; ++byte_idx) {
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
          out_ptr[out_idx++] = static_cast<uint8_t>(sym);
          node = 0;
        }
        ++bits_read;
      }
    }
  }

  if (out_idx != expected_num_symbols) {
    throw std::runtime_error(
        "Huffman decode failed: expected " + std::to_string(expected_num_symbols) +
        " symbols, got " + std::to_string(out_idx));
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("huffman_encode_symbols_cpp", &huffman_encode_symbols_cpp,
        "Encode uint8 symbols with Huffman code strings (CPU)");
  m.def("huffman_decode_symbols_cpp", &huffman_decode_symbols_cpp,
        "Decode Huffman bitstream to uint8 symbols (CPU)");
}
