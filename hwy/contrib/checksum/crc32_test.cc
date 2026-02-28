// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <vector>

#include "hwy/aligned_allocator.h"
#include "hwy/base.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/checksum/crc32_test.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/contrib/checksum/crc32-inl.h"
#include "hwy/tests/test_util-inl.h"
// clang-format on

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {

// Scalar reference implementation (table-based).
static uint32_t ReferenceCrc32(const uint8_t* data, size_t size,
                               uint32_t crc = 0) {
  crc ^= 0xFFFFFFFF;
  for (size_t i = 0; i < size; ++i) {
    crc = detail::kCrc32Table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
  }
  return crc ^ 0xFFFFFFFF;
}

void TestEmpty() {
  const uint32_t result = Crc32(nullptr, 0, 0);
  HWY_ASSERT_EQ(static_cast<uint32_t>(0), result);
}

void TestSingleByte() {
  const uint8_t data0[] = {0x00};
  HWY_ASSERT_EQ(ReferenceCrc32(data0, 1), Crc32(data0, 1));

  const uint8_t data1[] = {0xFF};
  HWY_ASSERT_EQ(ReferenceCrc32(data1, 1), Crc32(data1, 1));

  const uint8_t data2[] = {0x01};
  HWY_ASSERT_EQ(ReferenceCrc32(data2, 1), Crc32(data2, 1));
}

void TestKnownValues() {
  // CRC-32 of "123456789" = 0xCBF43926
  const uint8_t check[] = "123456789";
  const uint32_t expected = 0xCBF43926;
  HWY_ASSERT_EQ(expected, Crc32(check, 9));
}

void TestAllZeros() {
  const size_t n = 1024;
  std::vector<uint8_t> data(n, 0);
  const uint32_t expected = ReferenceCrc32(data.data(), n);
  HWY_ASSERT_EQ(expected, Crc32(data.data(), n));
}

void TestAllFF() {
  const size_t n = 1024;
  std::vector<uint8_t> data(n, 0xFF);
  const uint32_t expected = ReferenceCrc32(data.data(), n);
  HWY_ASSERT_EQ(expected, Crc32(data.data(), n));
}

void TestVaryingLengths() {
  // Test lengths from 1 to 300.
  for (size_t len = 1; len <= 300; ++len) {
    std::vector<uint8_t> data(len);
    for (size_t i = 0; i < len; ++i) {
      data[i] = static_cast<uint8_t>(i & 0xFF);
    }
    const uint32_t expected = ReferenceCrc32(data.data(), len);
    const uint32_t actual = Crc32(data.data(), len);
    if (expected != actual) {
      fprintf(stderr,
              "CRC32 mismatch at len=%zu: expected 0x%08X got 0x%08X\n", len,
              expected, actual);
    }
    HWY_ASSERT_EQ(expected, actual);
  }
}

void TestLargeInput() {
  const size_t n = 65536;
  std::vector<uint8_t> data(n);
  for (size_t i = 0; i < n; ++i) {
    data[i] = static_cast<uint8_t>((i * 37 + 13) & 0xFF);
  }
  const uint32_t expected = ReferenceCrc32(data.data(), n);
  HWY_ASSERT_EQ(expected, Crc32(data.data(), n));
}

void TestStreaming() {
  // Verify streaming: computing in parts gives the same result.
  const size_t n = 5000;
  std::vector<uint8_t> data(n);
  for (size_t i = 0; i < n; ++i) {
    data[i] = static_cast<uint8_t>((i * 7) & 0xFF);
  }

  const uint32_t one_shot = Crc32(data.data(), n);

  uint32_t streaming = 0;
  const size_t chunk_sizes[] = {100, 200, 500, 1000, 3200};
  size_t offset = 0;
  for (size_t cs : chunk_sizes) {
    streaming = Crc32(data.data() + offset, cs, streaming);
    offset += cs;
  }

  HWY_ASSERT_EQ(one_shot, streaming);
}

void TestPowerOfTwo() {
  const size_t sizes[] = {16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
  for (size_t n : sizes) {
    std::vector<uint8_t> data(n);
    for (size_t i = 0; i < n; ++i) {
      data[i] = static_cast<uint8_t>(i & 0xFF);
    }
    const uint32_t expected = ReferenceCrc32(data.data(), n);
    const uint32_t actual = Crc32(data.data(), n);
    if (expected != actual) {
      fprintf(stderr,
              "CRC32 mismatch at len=%zu: expected 0x%08X got 0x%08X\n", n,
              expected, actual);
    }
    HWY_ASSERT_EQ(expected, actual);
  }
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
namespace {
HWY_BEFORE_TEST(Crc32Test);
HWY_EXPORT_AND_TEST_P(Crc32Test, TestEmpty);
HWY_EXPORT_AND_TEST_P(Crc32Test, TestSingleByte);
HWY_EXPORT_AND_TEST_P(Crc32Test, TestKnownValues);
HWY_EXPORT_AND_TEST_P(Crc32Test, TestAllZeros);
HWY_EXPORT_AND_TEST_P(Crc32Test, TestAllFF);
HWY_EXPORT_AND_TEST_P(Crc32Test, TestVaryingLengths);
HWY_EXPORT_AND_TEST_P(Crc32Test, TestLargeInput);
HWY_EXPORT_AND_TEST_P(Crc32Test, TestStreaming);
HWY_EXPORT_AND_TEST_P(Crc32Test, TestPowerOfTwo);
HWY_AFTER_TEST();
}  // namespace
}  // namespace hwy
HWY_TEST_MAIN();
#endif  // HWY_ONCE
