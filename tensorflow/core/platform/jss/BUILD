# Description:
# Cloud file system implementation.

package(
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

cc_library(
    name = "jss_file_system",
    srcs = [
        "jss_file_system.cc",
    ],
    hdrs = [
        "jss_file_system.h",
    ],
    linkstatic = 1,  # Needed since alwayslink is broken in bazel b/27630669
    visibility = ["//visibility:public"],
    deps = [
        ":jss_auth_provider",
        "//tensorflow/core/platform/cloud:http_request",
        "//tensorflow/core/platform/cloud:retrying_file_system",
        "//tensorflow/core/platform/cloud:time_util",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib_internal",
        "@jsoncpp_git//:jsoncpp",
    ],
    alwayslink = 1,
)

cc_library(
    name = "jss_auth_provider",
    srcs = [
        "jss_auth_provider.cc",
    ],
    hdrs = [
        "jss_auth_provider.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/platform/cloud:http_request",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@jsoncpp_git//:jsoncpp",
    ],
)
