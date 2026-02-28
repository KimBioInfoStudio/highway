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
#define HWY_TARGET_INCLUDE "hwy/contrib/checksum/adler32_test.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/contrib/checksum/adler32-inl.h"
#include "hwy/tests/test_util-inl.h"
// clang-format on

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {

// Scalar reference implementation for verification.
static uint32_t ReferenceAdler32(const uint8_t* data, size_t size,
                                 uint32_t adler = 1) {
  uint32_t s1 = adler & 0xFFFF;
  uint32_t s2 = (adler >> 16) & 0xFFFF;
  for (size_t i = 0; i < size; ++i) {
    s1 = (s1 + data[i]) % 65521;
    s2 = (s2 + s1) % 65521;
  }
  return (s2 << 16) | s1;
}

void TestEmpty() {
  const uint32_t result = Adler32(nullptr, 0, 1);
  HWY_ASSERT_EQ(static_cast<uint32_t>(1), result);
}

void TestSingleByte() {
  const uint8_t data[] = {0x00};
  HWY_ASSERT_EQ(ReferenceAdler32(data, 1), Adler32(data, 1));

  const uint8_t data2[] = {0xFF};
  HWY_ASSERT_EQ(ReferenceAdler32(data2, 1), Adler32(data2, 1));

  const uint8_t data3[] = {0x01};
  HWY_ASSERT_EQ(ReferenceAdler32(data3, 1), Adler32(data3, 1));
}

void TestKnownValues() {
  // "Wikipedia" example: adler32("Wikipedia") = 0x11E60398
  const uint8_t wiki[] = "Wikipedia";
  const uint32_t expected = 0x11E60398;
  HWY_ASSERT_EQ(expected, Adler32(wiki, 9));
}

void TestAllZeros() {
  // All zeros: s1 = 1, s2 = n (mod 65521)
  const size_t n = 1024;
  std::vector<uint8_t> data(n, 0);
  const uint32_t expected = ReferenceAdler32(data.data(), n);
  HWY_ASSERT_EQ(expected, Adler32(data.data(), n));
}

void TestAllOnes() {
  const size_t n = 1024;
  std::vector<uint8_t> data(n, 1);
  const uint32_t expected = ReferenceAdler32(data.data(), n);
  HWY_ASSERT_EQ(expected, Adler32(data.data(), n));
}

void TestAllFF() {
  const size_t n = 1024;
  std::vector<uint8_t> data(n, 0xFF);
  const uint32_t expected = ReferenceAdler32(data.data(), n);
  HWY_ASSERT_EQ(expected, Adler32(data.data(), n));
}

void TestVaryingLengths() {
  // Test lengths from 1 to 300 with sequential byte values.
  for (size_t len = 1; len <= 300; ++len) {
    std::vector<uint8_t> data(len);
    for (size_t i = 0; i < len; ++i) {
      data[i] = static_cast<uint8_t>(i & 0xFF);
    }
    const uint32_t expected = ReferenceAdler32(data.data(), len);
    const uint32_t actual = Adler32(data.data(), len);
    if (expected != actual) {
      fprintf(stderr, "Adler32 mismatch at len=%zu: expected 0x%08X got 0x%08X\n",
              len, expected, actual);
    }
    HWY_ASSERT_EQ(expected, actual);
  }
}

void TestLargeInput() {
  // Large input that crosses the NMAX boundary (5552 bytes).
  const size_t n = 10000;
  std::vector<uint8_t> data(n);
  for (size_t i = 0; i < n; ++i) {
    data[i] = static_cast<uint8_t>((i * 37 + 13) & 0xFF);
  }
  const uint32_t expected = ReferenceAdler32(data.data(), n);
  HWY_ASSERT_EQ(expected, Adler32(data.data(), n));
}

void TestStreaming() {
  // Verify streaming: computing in parts gives the same result.
  const size_t n = 5000;
  std::vector<uint8_t> data(n);
  for (size_t i = 0; i < n; ++i) {
    data[i] = static_cast<uint8_t>((i * 7) & 0xFF);
  }

  const uint32_t one_shot = Adler32(data.data(), n);

  // Compute in chunks.
  uint32_t streaming = 1;  // initial adler
  const size_t chunk_sizes[] = {100, 200, 500, 1000, 3200};
  size_t offset = 0;
  for (size_t cs : chunk_sizes) {
    streaming = Adler32(data.data() + offset, cs, streaming);
    offset += cs;
  }

  HWY_ASSERT_EQ(one_shot, streaming);
}

void TestExactlyNmax() {
  // Test with exactly NMAX bytes.
  const size_t n = 5552;
  std::vector<uint8_t> data(n, 0xAB);
  const uint32_t expected = ReferenceAdler32(data.data(), n);
  HWY_ASSERT_EQ(expected, Adler32(data.data(), n));
}

void TestPowerOfTwo() {
  // Test power-of-two lengths (common SIMD vector sizes).
  const size_t sizes[] = {16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
  for (size_t n : sizes) {
    std::vector<uint8_t> data(n);
    for (size_t i = 0; i < n; ++i) {
      data[i] = static_cast<uint8_t>(i & 0xFF);
    }
    const uint32_t expected = ReferenceAdler32(data.data(), n);
    const uint32_t actual = Adler32(data.data(), n);
    if (expected != actual) {
      fprintf(stderr,
              "Adler32 mismatch at len=%zu: expected 0x%08X got 0x%08X\n", n,
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
HWY_BEFORE_TEST(Adler32Test);
HWY_EXPORT_AND_TEST_P(Adler32Test, TestEmpty);
HWY_EXPORT_AND_TEST_P(Adler32Test, TestSingleByte);
HWY_EXPORT_AND_TEST_P(Adler32Test, TestKnownValues);
HWY_EXPORT_AND_TEST_P(Adler32Test, TestAllZeros);
HWY_EXPORT_AND_TEST_P(Adler32Test, TestAllOnes);
HWY_EXPORT_AND_TEST_P(Adler32Test, TestAllFF);
HWY_EXPORT_AND_TEST_P(Adler32Test, TestVaryingLengths);
HWY_EXPORT_AND_TEST_P(Adler32Test, TestLargeInput);
HWY_EXPORT_AND_TEST_P(Adler32Test, TestStreaming);
HWY_EXPORT_AND_TEST_P(Adler32Test, TestExactlyNmax);
HWY_EXPORT_AND_TEST_P(Adler32Test, TestPowerOfTwo);
HWY_AFTER_TEST();
}  // namespace
}  // namespace hwy
HWY_TEST_MAIN();
#endif  // HWY_ONCE
