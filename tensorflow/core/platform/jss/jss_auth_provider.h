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

#ifndef TENSORFLOW_CORE_PLATFORM_JSS_AUTH_PROVIDER_H_
#define TENSORFLOW_CORE_PLATFORM_JSS_AUTH_PROVIDER_H_

#include "tensorflow/core/platform/cloud/http_request.h"

namespace tensorflow {

/// Implementation based on JSS Default Credentials.
  class JssAuthProvider {
  public:
    JssAuthProvider();

    JssAuthProvider(const string &access_key, const string &access_secret_key, const string &endpoint);

    Status GetToken(HttpRequest *request, string *token,
                    const string *method, const string *date, const string *content_type,
                    const string *customHead, const string *resource);

    static Status GetToken(JssAuthProvider *provider, HttpRequest *request, string *token,
                           const string *method, const string *date, const string *content_type,
                           const string *customHead, const string *resource) {
      if (!provider) {
        return errors::Internal("Auth provider is required.");
      }
      return provider->GetToken(request, token, method, date, content_type, customHead, resource);
    }

    const string GetEndPoint();

  private:
    string access_key_;
    string secret_key_;
    string endpoint_;
  };

  Status sign(const string *secret_key, const string *input, string *output);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_JSS_AUTH_PROVIDER_H_
