/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <malloc.h>
#include <openssl/hmac.h>
#include <openssl/evp.h>
#include <math.h>
#include "tensorflow/core/platform/jss/jss_auth_provider.h"

namespace tensorflow {

  namespace {
    constexpr char kJssUriBase[] = "https://storage.jd.local/";
    constexpr char kJssUploadUriBase[] =
        "https://storage.jd.local/";
    constexpr char kStorageHost[] = "storage.jd.local";
  }

/// sign string for jss auth
  Status sign(const string *secret_key, const string *input, string *output) {
    // check
    if (!secret_key) { return errors::Internal("secret_key for sign is required."); }
    if (!input) { return errors::Internal("input for sign is required."); }
    if (!output) { return errors::Internal("output for sign is required."); }

    // hmac
    const EVP_MD *engine = EVP_sha1();
    unsigned char buffer[EVP_MAX_MD_SIZE];
    unsigned int buffer_len;

    HMAC_CTX ctx;
    HMAC_CTX_init(&ctx);
    HMAC_Init_ex(&ctx, secret_key->c_str(), (int) (secret_key->length()), engine, NULL);
    HMAC_Update(&ctx, (unsigned char *) input->c_str(), input->length()); // input is OK; &input is WRONG !!!
    HMAC_Final(&ctx, (unsigned char *) &buffer, &buffer_len);
    HMAC_CTX_cleanup(&ctx);

    // base64
    size_t size = (size_t) buffer_len * 2;
    size = size > 64 ? size : 64;
    unsigned char *out = (unsigned char *) malloc(size);
    int out_len = EVP_EncodeBlock(out, (unsigned char *) &buffer, buffer_len);

    // set
    output->assign((char *) out, (size_t) out_len);

    // clean
    free(out);
    return Status::OK();
  }

  JssAuthProvider::JssAuthProvider() {
    char *ak = getenv("JSS_ACCESS_KEY");
    char *sk = getenv("JSS_SECRET_KEY");
    char *ep = getenv("JSS_ENDPOINT");
    if (ak != NULL && strlen(ak) > 0) { access_key_ = string(ak); }
    if (sk != NULL && strlen(sk) > 0) { secret_key_ = string(sk); }
    if (ep != NULL && strlen(ep) > 0) { endpoint_ = string(ep); }
    if (&endpoint_ == NULL || endpoint_.size() == 0) {
      endpoint_ = string(kStorageHost);
    }
  }

  JssAuthProvider::JssAuthProvider(const string &access_key, const string &secret_key_,
                                   const string &endpoint)
      : access_key_(access_key),
        secret_key_(secret_key_),
        endpoint_(endpoint) {
    if (&endpoint_ == NULL || endpoint_.size() == 0) {
      endpoint_.assign(kStorageHost);
    }
  }

  Status JssAuthProvider::GetToken(HttpRequest *request, string *token,
                                   const string *method, const string *date, const string *content_type,
                                   const string *customHead, const string *resource) {
    if (!request) {
      return errors::Internal("http request is required.");
    }
    if (!token) {
      return errors::Internal("token for out is required.");
    }
    if (!method) { return errors::Internal("method is required."); }
    if (!content_type) { return errors::Internal("content-type is required."); }
    if (!date) { return errors::Internal("date is required."); }
    if (!resource) { return errors::Internal("resource is required."); }

    string string_to_sign;
    string_to_sign.append(method->c_str()) // method
        .append("\n").append("") // md5
        .append("\n").append(content_type->c_str()) // Content-Type
        .append("\n").append(date->c_str()) // date string
        ;
    if (customHead && customHead->size() > 0) {
      string_to_sign.append("\n").append(customHead->c_str()); // customHead
    }
    string_to_sign.append("\n").append(resource->c_str()); // resource

    string signed_str;
    TF_RETURN_IF_ERROR(sign(&secret_key_, &string_to_sign, &signed_str));
    token->assign(strings::StrCat("jingdong ", access_key_, ":", signed_str));

    return Status::OK();
  }

  const string JssAuthProvider::GetEndPoint() {
    return endpoint_;
  }

}  // namespace tensorflow
