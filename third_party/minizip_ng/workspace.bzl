# Copyright 2023 The TensorStore Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load(
    "//third_party:repo.bzl",
    "third_party_http_archive",
)
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
    maybe(
        third_party_http_archive,
        name = "minizip_ng",
        sha256 = "27cc2f62cd02d79b71b346fc6ace02728385f8ba9c6b5f124062b0790a04629a",
        strip_prefix = "minizip-ng-3.0.8",
        urls = [
            "https://github.com/zlib-ng/minizip-ng/archive/refs/tags/3.0.8.tar.gz",
        ],
        build_file = Label("//third_party:minizip_ng/minizip_ng.BUILD.bazel"),
        cmake_name = "MINIZIP",
        bazel_to_cmake = {},
        cmake_target_mapping = {
            "@minizip_ng//:minizip_ng": "MINIZIP::minizip-ng",
        },
    )
