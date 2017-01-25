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

#include "tensorflow/core/platform/jss/jss_file_system.h"
#include <stdio.h>
#include <unistd.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#include "include/json/json.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/cloud/time_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {

  namespace {

    constexpr char kJssUriBase[] = "https://storage.jd.local/";
    constexpr char kJssUploadUriBase[] =
        "https://storage.jd.local/";
    constexpr char kStorageHost[] = "storage.jd.local";
    constexpr size_t kReadAppendableFileBufferSize = 1024 * 1024;  // In bytes.
    constexpr int kGetChildrenDefaultPageSize = 1000;
// Initial delay before retrying a JSS upload.
// Subsequent delays can be larger due to exponential back-off.
    constexpr uint64 kUploadRetryDelayMicros = 1000000L;
// The HTTP response code "308 Resume Incomplete".
    constexpr uint64 HTTP_CODE_RESUME_INCOMPLETE = 308;

// The file statistics returned by Stat() for directories.
    const FileStatistics DIRECTORY_STAT(0, 0, true);

    Status GetTmpFilename(string *filename) {
      if (!filename) {
        return errors::Internal("'filename' cannot be nullptr.");
      }
      char buffer[] = "/tmp/jss_filesystem_XXXXXX";
      int fd = mkstemp(buffer);
      if (fd < 0) {
        return errors::Internal("Failed to create a temporary file.");
      }
      close(fd);
      *filename = buffer;
      return Status::OK();
    }

/// \brief Splits a JSS path to a bucket and an object.
///
/// For example, "jss://bucket-name/path/to/file.txt" gets split into
/// "bucket-name" and "path/to/file.txt".
/// If fname only contains the bucket and empty_object_ok = true, the returned
/// object is empty.
    Status ParseJssPath(StringPiece fname, bool empty_object_ok, string *bucket,
                        string *object) {
      if (!bucket || !object) {
        return errors::Internal("bucket and object cannot be null.");
      }
      StringPiece scheme, bucketp, objectp;
      io::ParseURI(fname, &scheme, &bucketp, &objectp);
      if (scheme != "jss") {
        return errors::InvalidArgument("JSS path doesn't start with 'jss://': ",
                                       fname);
      }
      *bucket = bucketp.ToString();
      if (bucket->empty() || *bucket == ".") {
        return errors::InvalidArgument("JSS path doesn't contain a bucket name: ",
                                       fname);
      }
      objectp.Consume("/");
      *object = objectp.ToString();
      if (!empty_object_ok && object->empty()) {
        return errors::InvalidArgument("JSS path doesn't contain an object name: ",
                                       fname);
      }
      return Status::OK();
    }

/// Appends a trailing slash if the name doesn't already have one.
    string MaybeAppendSlash(const string &name) {
      if (name.empty()) {
        return "/";
      }
      if (name.back() != '/') {
        return strings::StrCat(name, "/");
      }
      return name;
    }

// io::JoinPath() doesn't work in cases when we want an empty subpath
// to result in an appended slash in order for directory markers
// to be processed correctly: "jss://a/b" + "" should give "jss://a/b/".
    string JoinJssPath(const string &path, const string &subpath) {
      return strings::StrCat(MaybeAppendSlash(path), subpath);
    }

/// \brief Returns the given paths appending all their subfolders.
///
/// For every path X in the list, every subfolder in X is added to the
/// resulting list.
/// For example:
///  - for 'a/b/c/d' it will append 'a', 'a/b' and 'a/b/c'
///  - for 'a/b/c/' it will append 'a', 'a/b' and 'a/b/c'
    std::set<string> AddAllSubpaths(const std::vector<string> &paths) {
      std::set<string> result;
      result.insert(paths.begin(), paths.end());
      for (const string &path : paths) {
        StringPiece subpath = io::Dirname(path);
        while (!subpath.empty()) {
          result.emplace(subpath.ToString());
          subpath = io::Dirname(subpath);
        }
      }
      return result;
    }

    Status ParseJson(StringPiece json, Json::Value *result) {
      Json::Reader reader;
      if (!reader.parse(json.ToString(), *result)) {
        return errors::Internal("Couldn't parse JSON response from JSS.");
      }
      return Status::OK();
    }

/// Reads a JSON value with the given name from a parent JSON value.
    Status GetValue(const Json::Value &parent, const string &name,
                    Json::Value *result) {
      *result = parent.get(name, Json::Value::null);
      if (*result == Json::Value::null) {
        return errors::Internal("The field '", name,
                                "' was expected in the JSON response.");
      }
      return Status::OK();
    }

/// Reads a string JSON value with the given name from a parent JSON value.
    Status GetStringValue(const Json::Value &parent, const string &name,
                          string *result) {
      Json::Value result_value;
      TF_RETURN_IF_ERROR(GetValue(parent, name, &result_value));
      if (!result_value.isString()) {
        return errors::Internal(
            "The field '", name,
            "' in the JSON response was expected to be a string.");
      }
      *result = result_value.asString();
      return Status::OK();
    }

/// Reads a long JSON value with the given name from a parent JSON value.
    Status GetInt64Value(const Json::Value &parent, const string &name,
                         int64 *result) {
      Json::Value result_value;
      TF_RETURN_IF_ERROR(GetValue(parent, name, &result_value));
      if (result_value.isNumeric()) {
        *result = result_value.asInt64();
        return Status::OK();
      }
      if (result_value.isString() &&
          strings::safe_strto64(result_value.asString().c_str(), result)) {
        return Status::OK();
      }
      return errors::Internal(
          "The field '", name,
          "' in the JSON response was expected to be a number.");
    }

/// Reads a boolean JSON value with the given name from a parent JSON value.
    Status GetBoolValue(const Json::Value &parent, const string &name,
                        bool *result) {
      Json::Value result_value;
      TF_RETURN_IF_ERROR(GetValue(parent, name, &result_value));
      if (!result_value.isBool()) {
        return errors::Internal(
            "The field '", name,
            "' in the JSON response was expected to be a boolean.");
      }
      *result = result_value.asBool();
      return Status::OK();
    }

/// some tool functions
/// sign string for jss auth
    Status get_time_str(string *output) {
      if (!output) {
        return errors::Internal("output for time string is required.");
      }
      const int BUF_SIZE = 64;
      char buf[BUF_SIZE];
      time_t t = time(NULL);
      strftime((char *) &buf, BUF_SIZE, "%a, %d %b %G %T GMT", gmtime(&t));
      output->assign((char *) buf);

      return Status::OK();
    }

    constexpr int64 kNanosecondsPerSecond = 1000 * 1000 * 1000;

    Status parse_time_str(const string &time, int64 *mtime_nsec) {
      struct tm parsed;
      strptime(time.c_str(), "%a, %d %b %Y %T GMT", &parsed);
      *mtime_nsec = timegm(&parsed) * kNanosecondsPerSecond;

      return Status::OK();
    }


/// Reads a long JSON value with the given name from a parent JSON value.
    Status parse_int64_string(const string &str, int64 *result) {
      if (strings::safe_strto64(str.c_str(), result)) {
        return Status::OK();
      }
      return errors::Internal(
          "The string '", str,
          "' in was expected to be a number.");
    }


/// A JSS-based implementation of a random access file with a read-ahead buffer.
    class JssRandomAccessFile : public RandomAccessFile {
    public:
      JssRandomAccessFile(const string &bucket, const string &object,
                          JssAuthProvider *auth_provider,
                          HttpRequest::Factory *http_request_factory,
                          size_t read_ahead_bytes)
          : bucket_(bucket),
            object_(object),
            auth_provider_(auth_provider),
            http_request_factory_(http_request_factory),
            read_ahead_bytes_(read_ahead_bytes) {}

      /// The implementation of reads with a read-ahead buffer. Thread-safe.
      Status Read(uint64 offset, size_t n, StringPiece *result,
                  char *scratch) const override {

        mutex_lock lock(mu_);
        const bool range_start_included = offset >= buffer_start_offset_;
        const bool range_end_included =
            offset + n <= buffer_start_offset_ + buffer_.size();
        if (range_start_included && range_end_included) {
          // The requested range can be filled from the buffer.
          const size_t offset_in_buffer =
              std::min<uint64>(offset - buffer_start_offset_, buffer_.size());
          const auto copy_size = std::min(n, buffer_.size() - offset_in_buffer);
          std::copy(buffer_.begin() + offset_in_buffer,
                    buffer_.begin() + offset_in_buffer + copy_size, scratch);
          *result = StringPiece(scratch, copy_size);
        } else {
          // Update the buffer content based on the new requested range.
          const size_t desired_buffer_size = n + read_ahead_bytes_;
          if (n > buffer_.capacity() ||
              desired_buffer_size > 2 * buffer_.capacity()) {
            // Re-allocate only if buffer capacity increased significantly.
            if (offset == 0 && buffer_.capacity() == 0) {
              FileStatistics stat;
              TF_RETURN_IF_ERROR(StatForObject(&stat));
              if (desired_buffer_size > stat.length)
                buffer_.reserve(size_t(stat.length));
            } else {
              buffer_.reserve(desired_buffer_size);
            }
          }

          buffer_start_offset_ = offset;
          TF_RETURN_IF_ERROR(LoadBufferFromJSS());

          // Set the results.
          std::memcpy(scratch, buffer_.data(), std::min(buffer_.size(), n));
          *result = StringPiece(scratch, std::min(buffer_.size(), n));
        }

        if (result->size() < n) {
          // This is not an error per se. The RandomAccessFile interface expects
          // that Read returns OutOfRange if fewer bytes were read than requested.
          return errors::OutOfRange("EOF reached, ", result->size(),
                                    " bytes were read out of ", n,
                                    " bytes requested.");
        }
        return Status::OK();
      }

    private:
      Status StatForObject(FileStatistics *stat) const {
        if (!stat) {
          return errors::Internal("'stat' cannot be nullptr.");
        }

        string auth_token;
        string method = "GET";
        string date_str;
        string content_type;
        string customHead;
        string resource;

        std::unique_ptr<HttpRequest> request(http_request_factory_->Create());
        TF_RETURN_IF_ERROR(request->Init());
        TF_RETURN_IF_ERROR(
            request->SetUri(strings::StrCat("http://", auth_provider_->GetEndPoint(), "/", bucket_,
                                            "/", object_)));

        TF_RETURN_IF_ERROR(get_time_str(&date_str));
        TF_RETURN_IF_ERROR(request->AddHeader("Date", date_str));
        resource = strings::StrCat("/", bucket_, "/", object_);
        TF_RETURN_IF_ERROR(JssAuthProvider::GetToken(auth_provider_, request.get(), &auth_token,
                                                     &method, &date_str, &content_type, &customHead,
                                                     &resource));
        TF_RETURN_IF_ERROR(request->AddHeader("Authorization", auth_token));

        TF_RETURN_WITH_CONTEXT_IF_ERROR(
            request->Send(), " when reading metadata of jss://", bucket_, "/", object_);


        // Parse file size.
        string content_length = request->GetResponseHeader("Content-Length");
        TF_RETURN_IF_ERROR(parse_int64_string(content_length, &(stat->length)));

        // Parse file modification time.
        string updated = request->GetResponseHeader("Last-Modified");
        TF_RETURN_IF_ERROR(parse_time_str(updated, &(stat->mtime_nsec)));

        stat->is_directory = false;

        return Status::OK();
      }

      /// A helper function to actually read the data from JSS. This function loads
      /// buffer_ from JSS based on its current capacity.
      Status LoadBufferFromJSS() const EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        string auth_token;
        string method = "GET";
        string date_str;
        string content_type;
        string customHead;
        string resource;

        std::unique_ptr<HttpRequest> request(http_request_factory_->Create());
        TF_RETURN_IF_ERROR(request->Init());
        TF_RETURN_IF_ERROR(
            request->SetUri(strings::StrCat("http://", auth_provider_->GetEndPoint(), "/", bucket_,
                                            "/", object_)));

        TF_RETURN_IF_ERROR(get_time_str(&date_str));
        TF_RETURN_IF_ERROR(request->AddHeader("Date", date_str));
        resource = strings::StrCat("/", bucket_, "/", object_);
        TF_RETURN_IF_ERROR(JssAuthProvider::GetToken(auth_provider_, request.get(), &auth_token,
                                                     &method, &date_str, &content_type, &customHead,
                                                     &resource));
        TF_RETURN_IF_ERROR(request->AddHeader("Authorization", auth_token));

        TF_RETURN_IF_ERROR(request->SetRange(
            buffer_start_offset_, buffer_start_offset_ + buffer_.capacity() - 1));
        TF_RETURN_IF_ERROR(request->SetResultBuffer(&buffer_));
        TF_RETURN_WITH_CONTEXT_IF_ERROR(request->Send(), " when reading jss://",
                                        bucket_, "/", object_);
        return Status::OK();
      }

      string bucket_;
      string object_;
      JssAuthProvider *auth_provider_;
      HttpRequest::Factory *http_request_factory_;
      const size_t read_ahead_bytes_;

      // The buffer-related members need to be mutable, because they are modified
      // by the const Read() method.
      mutable mutex mu_;
      mutable std::vector<char> buffer_ GUARDED_BY(mu_);
      // The original file offset of the first byte in the buffer.
      mutable size_t buffer_start_offset_ GUARDED_BY(mu_) = 0;
    };

/// \brief JSS-based implementation of a writeable file.
///
/// Since JSS objects are immutable, this implementation writes to a local
/// tmp file and copies it to JSS on flush/close.
    class JssWritableFile : public WritableFile {
    public:
      JssWritableFile(const string &bucket, const string &object,
                      JssAuthProvider *auth_provider,
                      HttpRequest::Factory *http_request_factory,
                      int32 max_upload_attempts)
          : bucket_(bucket),
            object_(object),
            auth_provider_(auth_provider),
            http_request_factory_(http_request_factory),
            max_upload_attempts_(max_upload_attempts) {
        if (GetTmpFilename(&tmp_content_filename_).ok()) {
          outfile_.open(tmp_content_filename_,
                        std::ofstream::binary | std::ofstream::app);
        }
      }

      /// \brief Constructs the writable file in append mode.
      ///
      /// tmp_content_filename should contain a path of an existing temporary file
      /// with the content to be appended. The class takes onwnership of the
      /// specified tmp file and deletes it on close.
      JssWritableFile(const string &bucket, const string &object,
                      JssAuthProvider *auth_provider,
                      const string &tmp_content_filename,
                      HttpRequest::Factory *http_request_factory,
                      int32 max_upload_attempts)
          : bucket_(bucket),
            object_(object),
            auth_provider_(auth_provider),
            http_request_factory_(http_request_factory),
            max_upload_attempts_(max_upload_attempts) {
        tmp_content_filename_ = tmp_content_filename;
        outfile_.open(tmp_content_filename_,
                      std::ofstream::binary | std::ofstream::app);
      }

      ~JssWritableFile() { Close(); }

      Status Append(const StringPiece &data) override {
        TF_RETURN_IF_ERROR(CheckWritable());
        outfile_ << data;
        if (!outfile_.good()) {
          return errors::Internal(
              "Could not append to the internal temporary file.");
        }
        return Status::OK();
      }

      Status Close() override {
        if (outfile_.is_open()) {
          TF_RETURN_IF_ERROR(Sync());
          outfile_.close();
          std::remove(tmp_content_filename_.c_str());
        }
        return Status::OK();
      }

      Status Flush() override { return Sync(); }

      /// Copies the current version of the file to JSS.
      ///
      /// This Sync() uploads the object to JSS.
      /// In case of a failure, it resumes failed uploads as recommended by the JSS
      /// resumable API documentation. When the whole upload needs to be
      /// restarted, Sync() returns UNAVAILABLE and relies on RetryingFileSystem.
      Status Sync() override {
        TF_RETURN_IF_ERROR(CheckWritable());
        outfile_.flush();
        if (!outfile_.good()) {
          return errors::Internal(
              "Could not write to the internal temporary file.");
        }
        uint64 already_uploaded = 0;
        for (int attempt = 0; attempt < max_upload_attempts_; attempt++) { // TODO make upload resumable
          const Status upload_status = UploadFile();
          if (upload_status.ok()) {
            return Status::OK();
          }
          switch (upload_status.code()) {
            case errors::Code::NOT_FOUND:
              // JSS docs recommend retrying the whole upload. We're relying on the
              // RetryingFileSystem to retry the Sync() call.
              return errors::Unavailable("Could not upload jss://", bucket_, "/",
                                         object_);
            case errors::Code::UNAVAILABLE:
              // The upload can be resumed, but JSS docs recommend an exponential
              // back-off.
              Env::Default()->SleepForMicroseconds(kUploadRetryDelayMicros
                                                       << attempt);
              break;
            default:
              // Something unexpected happen, fail.
              return upload_status;
          }
        }
        return errors::Aborted("Upload jss://", bucket_, "/", object_, " failed.");
      }

    private:
      Status CheckWritable() const {
        if (!outfile_.is_open()) {
          return errors::FailedPrecondition(
              "The internal temporary file is not writable.");
        }
        return Status::OK();
      }

      Status GetCurrentFileSize(uint64 *size) {
        if (size == nullptr) {
          return errors::Internal("'size' cannot be nullptr");
        }
        const auto tellp = outfile_.tellp();
        if (tellp == -1) {
          return errors::Internal(
              "Could not get the size of the internal temporary file.");
        }
        *size = tellp;
        return Status::OK();
      }

      Status UploadFile() { // simple put file
        string auth_token;
        string method = "PUT";
        string date_str;
        string content_type = "";
        string customHead;
        string resource;

        std::unique_ptr<HttpRequest> request(http_request_factory_->Create());
        TF_RETURN_IF_ERROR(request->Init());
        TF_RETURN_IF_ERROR(
            request->SetUri(strings::StrCat("http://", auth_provider_->GetEndPoint(), "/", bucket_,
                                            "/", object_)));

        TF_RETURN_IF_ERROR(get_time_str(&date_str));
        TF_RETURN_IF_ERROR(request->AddHeader("Date", date_str));
        resource = strings::StrCat("/", bucket_, "/", object_);
        TF_RETURN_IF_ERROR(JssAuthProvider::GetToken(auth_provider_, request.get(), &auth_token,
                                                     &method, &date_str, &content_type, &customHead,
                                                     &resource));
        TF_RETURN_IF_ERROR(request->AddHeader("Authorization", auth_token));

        uint64 file_size;
        TF_RETURN_IF_ERROR(GetCurrentFileSize(&file_size));

        size_t start_offset = 0;
        if (file_size > 0) {
          TF_RETURN_IF_ERROR(request->AddHeader(
              "Content-Range", strings::StrCat("bytes ", start_offset, "-",
                                               file_size - 1, "/", file_size)));
        }
        TF_RETURN_IF_ERROR(
            request->SetPutFromFile(tmp_content_filename_, start_offset));
        TF_RETURN_WITH_CONTEXT_IF_ERROR(request->Send(), " when uploading ",
                                        GetJssPath());
        return Status::OK();
      }

      string GetJssPath() const {
        return strings::StrCat("jss://", bucket_, "/", object_);
      }

      string bucket_;
      string object_;
      JssAuthProvider *auth_provider_;
      string tmp_content_filename_;
      std::ofstream outfile_;
      HttpRequest::Factory *http_request_factory_;
      int32 max_upload_attempts_;
    };

    class JssReadOnlyMemoryRegion : public ReadOnlyMemoryRegion {
    public:
      JssReadOnlyMemoryRegion(std::unique_ptr<char[]> data, uint64 length)
          : data_(std::move(data)), length_(length) {}

      const void *data() override { return reinterpret_cast<void *>(data_.get()); }

      uint64 length() override { return length_; }

    private:
      std::unique_ptr<char[]> data_;
      uint64 length_;
    };
  }  // namespace

  JssFileSystem::JssFileSystem()
      : auth_provider_(new JssAuthProvider()),
        http_request_factory_(new HttpRequest::Factory()) {}

  JssFileSystem::JssFileSystem(
      std::unique_ptr<JssAuthProvider> auth_provider,
      std::unique_ptr<HttpRequest::Factory> http_request_factory,
      size_t read_ahead_bytes, int32 max_upload_attempts)
      : auth_provider_(std::move(auth_provider)),
        http_request_factory_(std::move(http_request_factory)),
        read_ahead_bytes_(read_ahead_bytes),
        max_upload_attempts_(max_upload_attempts) {}

  Status JssFileSystem::NewRandomAccessFile(
      const string &fname, std::unique_ptr<RandomAccessFile> *result) {
    string bucket, object;
    TF_RETURN_IF_ERROR(ParseJssPath(fname, false, &bucket, &object));
    result->reset(new JssRandomAccessFile(bucket, object, auth_provider_.get(),
                                          http_request_factory_.get(),
                                          read_ahead_bytes_));
    return Status::OK();
  }

  Status JssFileSystem::NewWritableFile(const string &fname,
                                        std::unique_ptr<WritableFile> *result) {
    string bucket, object;
    TF_RETURN_IF_ERROR(ParseJssPath(fname, false, &bucket, &object));
    result->reset(new JssWritableFile(bucket, object, auth_provider_.get(),
                                      http_request_factory_.get(),
                                      max_upload_attempts_));
    return Status::OK();
  }

// Reads the file from JSS in chunks and stores it in a tmp file,
// which is then passed to JssWritableFile.
  Status JssFileSystem::NewAppendableFile(const string &fname,
                                          std::unique_ptr<WritableFile> *result) {
    std::unique_ptr<RandomAccessFile> reader;
    TF_RETURN_IF_ERROR(NewRandomAccessFile(fname, &reader));
    std::unique_ptr<char[]> buffer(new char[kReadAppendableFileBufferSize]);
    Status status;
    uint64 offset = 0;
    StringPiece read_chunk;

    // Read the file from JSS in chunks and save it to a tmp file.
    string old_content_filename;
    TF_RETURN_IF_ERROR(GetTmpFilename(&old_content_filename));
    std::ofstream old_content(old_content_filename, std::ofstream::binary);
    while (true) {
      status = reader->Read(offset, kReadAppendableFileBufferSize, &read_chunk,
                            buffer.get());
      if (status.ok()) {
        old_content << read_chunk;
        offset += kReadAppendableFileBufferSize;
      } else if (status.code() == error::OUT_OF_RANGE) {
        // Expected, this means we reached EOF.
        old_content << read_chunk;
        break;
      } else {
        return status;
      }
    }
    old_content.close();

    // Create a writable file and pass the old content to it.
    string bucket, object;
    TF_RETURN_IF_ERROR(ParseJssPath(fname, false, &bucket, &object));
    result->reset(new JssWritableFile(
        bucket, object, auth_provider_.get(), old_content_filename,
        http_request_factory_.get(), max_upload_attempts_));
    return Status::OK();
  }

  Status JssFileSystem::NewReadOnlyMemoryRegionFromFile(
      const string &fname, std::unique_ptr<ReadOnlyMemoryRegion> *result) {
    uint64 size;
    TF_RETURN_IF_ERROR(GetFileSize(fname, &size));
    std::unique_ptr<char[]> data(new char[size]);

    std::unique_ptr<RandomAccessFile> file;
    TF_RETURN_IF_ERROR(NewRandomAccessFile(fname, &file));

    StringPiece piece;
    TF_RETURN_IF_ERROR(file->Read(0, size, &piece, data.get()));

    result->reset(new JssReadOnlyMemoryRegion(std::move(data), size));
    return Status::OK();
  }

  Status JssFileSystem::FileExists(const string &fname) {
    string bucket, object;
    TF_RETURN_IF_ERROR(ParseJssPath(fname, true, &bucket, &object));
    if (object.empty()) {
      bool result;
      TF_RETURN_IF_ERROR(BucketExists(bucket, &result));
      if (result) {
        return Status::OK();
      }
    }
    bool result;
    TF_RETURN_IF_ERROR(ObjectExists(bucket, object, &result));
    if (result) {
      return Status::OK();
    }
    TF_RETURN_IF_ERROR(FolderExists(fname, &result));
    if (result) {
      return Status::OK();
    }
    return errors::NotFound("The specified path ", fname, " was not found.");
  }

  Status JssFileSystem::ObjectExists(const string &bucket, const string &object,
                                     bool *result) {
    if (!result) {
      return errors::Internal("'result' cannot be nullptr.");
    }
    FileStatistics not_used_stat;
    const Status status = StatForObject(bucket, object, &not_used_stat);
    switch (status.code()) {
      case errors::Code::OK:
        *result = true;
        return Status::OK();
      case errors::Code::NOT_FOUND:
        *result = false;
        return Status::OK();
      default:
        return status;
    }
  }

  Status JssFileSystem::StatForObject(const string &bucket, const string &object,
                                      FileStatistics *stat) {
    if (!stat) {
      return errors::Internal("'stat' cannot be nullptr.");
    }
    if (object.empty()) {
      return errors::InvalidArgument("'object' must be a non-empty string.");
    }

    string auth_token;
    string method = "GET";
    string date_str;
    string content_type;
    string customHead;
    string resource;

    std::unique_ptr<HttpRequest> request(http_request_factory_->Create());
    TF_RETURN_IF_ERROR(request->Init());
    TF_RETURN_IF_ERROR(
        request->SetUri(strings::StrCat("http://", auth_provider_->GetEndPoint(), "/", bucket,
                                        "/", object)));

    TF_RETURN_IF_ERROR(get_time_str(&date_str));
    TF_RETURN_IF_ERROR(request->AddHeader("Date", date_str));
    resource = strings::StrCat("/", bucket, "/", object);
    TF_RETURN_IF_ERROR(JssAuthProvider::GetToken(auth_provider_.get(), request.get(), &auth_token,
                                                 &method, &date_str, &content_type, &customHead, &resource));
    TF_RETURN_IF_ERROR(request->AddHeader("Authorization", auth_token));

    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        request->Send(), " when reading metadata of jss://", bucket, "/", object);


    // Parse file size.
    string content_length = request->GetResponseHeader("Content-Length");
    TF_RETURN_IF_ERROR(parse_int64_string(content_length, &(stat->length)));

    // Parse file modification time.
    string updated = request->GetResponseHeader("Last-Modified");
    TF_RETURN_IF_ERROR(parse_time_str(updated, &(stat->mtime_nsec)));

    stat->is_directory = false;

    return Status::OK();
  }

  Status JssFileSystem::BucketExists(const string &bucket, bool *result) {
    if (!result) {
      return errors::Internal("'result' cannot be nullptr.");
    }

    string auth_token;
    string method = "GET";
    string date_str;
    string content_type;
    string customHead;
    string resource = "/";

    std::unique_ptr<HttpRequest> request(http_request_factory_->Create());
    TF_RETURN_IF_ERROR(request->Init());
    TF_RETURN_IF_ERROR(request->SetUri(strings::StrCat("http://", auth_provider_->GetEndPoint(), "/")));
    TF_RETURN_IF_ERROR(get_time_str(&date_str));
    TF_RETURN_IF_ERROR(request->AddHeader("Date", date_str));
    TF_RETURN_IF_ERROR(JssAuthProvider::GetToken(auth_provider_.get(), request.get(), &auth_token,
                                                 &method, &date_str, &content_type, &customHead, &resource));
    TF_RETURN_IF_ERROR(request->AddHeader("Authorization", auth_token));

    std::vector<char> output_buffer;
    TF_RETURN_IF_ERROR(request->SetResultBuffer(&output_buffer));

    TF_RETURN_WITH_CONTEXT_IF_ERROR(request->Send(), " when reading metadata of jss://", bucket);

    StringPiece response_piece =
        StringPiece(output_buffer.data(), output_buffer.size());
    Json::Value root;
    TF_RETURN_IF_ERROR(ParseJson(response_piece, &root));

    Json::Value buckets = root.get("Buckets", Json::Value::null);
    if (buckets != Json::Value::null) {
      for (int i = 0; i < buckets.size(); i++) {
        Json::Value bucket_json = buckets.get(i, Json::Value::null);
        if (bucket_json == Json::Value::null) continue;
        string bucket_name;
        TF_RETURN_IF_ERROR(GetStringValue(bucket_json, "Name", &bucket_name));
        if (bucket.compare(bucket_name) == 0) {
          *result = true;
          return Status::OK();
        }
      }
    }
    *result = false;
    return Status::OK();
  }

  Status JssFileSystem::FolderExists(const string &dirname, bool *result) {
    if (!result) {
      return errors::Internal("'result' cannot be nullptr.");
    }
    std::vector<string> children;
    TF_RETURN_IF_ERROR(
        GetChildrenBounded(dirname, 1, &children, true /* recursively */,
                           true /* include_self_directory_marker */));
    *result = !children.empty();
    return Status::OK();
  }

  Status JssFileSystem::GetChildren(const string &dirname,
                                    std::vector<string> *result) {
    return GetChildrenBounded(dirname, UINT64_MAX, result,
                              false /* recursively */,
                              false /* include_self_directory_marker */);
  }

  Status JssFileSystem::GetMatchingPaths(const string &pattern,
                                         std::vector<string> *results) {
    results->clear();
    // Find the fixed prefix by looking for the first wildcard.
    const string &fixed_prefix =
        pattern.substr(0, pattern.find_first_of("*?[\\"));
    const string &dir = io::Dirname(fixed_prefix).ToString();
    if (dir.empty()) {
      return errors::InvalidArgument("A JSS pattern doesn't have a bucket name: ",
                                     pattern);
    }
    std::vector<string> all_files;
    TF_RETURN_IF_ERROR(
        GetChildrenBounded(dir, UINT64_MAX, &all_files, true /* recursively */,
                           false /* include_self_directory_marker */));

    const auto &files_and_folders = AddAllSubpaths(all_files);

    // Match all obtained paths to the input pattern.
    for (const auto &path : files_and_folders) {
      const string &full_path = io::JoinPath(dir, path);
      if (Env::Default()->MatchPath(full_path, pattern)) {
        results->push_back(full_path);
      }
    }
    return Status::OK();
  }

  Status JssFileSystem::GetChildrenBounded(const string &dirname,
                                           uint64 max_results,
                                           std::vector<string> *result,
                                           bool recursive,
                                           bool include_self_directory_marker) {
    if (max_results >= 2147483647L) max_results = 2147483647L;
    if (!result) {
      return errors::InvalidArgument("'result' cannot be null");
    }
    string bucket, object_prefix;
    TF_RETURN_IF_ERROR(
        ParseJssPath(MaybeAppendSlash(dirname), true, &bucket, &object_prefix));
    // while (object_prefix[object_prefix.size() - 1] == '/') object_prefix.resize(object_prefix.size() - 1);

    string auth_token;
    string method = "GET";
    string date_str;
    string content_type;
    string customHead;
    string resource = strings::StrCat("/", bucket);

    std::unique_ptr<HttpRequest> request(http_request_factory_->Create());
    TF_RETURN_IF_ERROR(request->Init());

    string url_full;
    if (!include_self_directory_marker && object_prefix.back() != '/') {
      object_prefix.append("/");
    }
    url_full = strings::StrCat("http://", auth_provider_->GetEndPoint(), "/", bucket,
                               "?maxKeys=", max_results);
    if (object_prefix.size() > 0 && object_prefix.compare("/") != 0) {
      url_full.append("&prefix=").append(request->EscapeString(object_prefix));
    }
    if (!recursive) {
      url_full.append("&delimiter=%2F");
    }
    TF_RETURN_IF_ERROR(request->SetUri(url_full));
    TF_RETURN_IF_ERROR(get_time_str(&date_str));
    TF_RETURN_IF_ERROR(request->AddHeader("Date", date_str));
    TF_RETURN_IF_ERROR(JssAuthProvider::GetToken(auth_provider_.get(), request.get(), &auth_token,
                                                 &method, &date_str, &content_type, &customHead, &resource));
    TF_RETURN_IF_ERROR(request->AddHeader("Authorization", auth_token));

    std::vector<char> output_buffer;
    TF_RETURN_IF_ERROR(request->SetResultBuffer(&output_buffer));

    TF_RETURN_WITH_CONTEXT_IF_ERROR(request->Send(), " when reading ", dirname);

    StringPiece response_piece =
        StringPiece(output_buffer.data(), output_buffer.size());
    Json::Value root;
    TF_RETURN_IF_ERROR(ParseJson(response_piece, &root));
    Json::Value common_prefixes = root.get("CommonPrefixes", Json::Value::null);
    if (common_prefixes != Json::Value::null) {
      for (int i = 0; i < common_prefixes.size(); i++) {
        Json::Value dir_key = common_prefixes.get(i, Json::Value::null);
        if (dir_key != Json::Value::null && dir_key.isString()) {
          result->emplace_back(strings::StrCat("jss://", bucket, "/", dir_key.asString()));
        }
      }
    }
    Json::Value contents = root.get("Contents", Json::Value::null);
    if (contents != Json::Value::null) {
      for (int i = 0; i < contents.size(); i++) {
        Json::Value object_info = contents.get(i, Json::Value::null);
        if (object_info == Json::Value::null) continue;
        string object_key;
        TF_RETURN_IF_ERROR(GetStringValue(object_info, "Key", &object_key));
        if (include_self_directory_marker || object_prefix.compare(object_key) != 0) {
          result->emplace_back(strings::StrCat("jss://", bucket, "/", object_key));
        }
      }
    }
    return Status::OK();
  }

  Status JssFileSystem::Stat(const string &fname, FileStatistics *stat) {
    if (!stat) {
      return errors::Internal("'stat' cannot be nullptr.");
    }
    string bucket, object;
    TF_RETURN_IF_ERROR(ParseJssPath(fname, true, &bucket, &object));
    if (object.empty()) {
      bool is_bucket;
      TF_RETURN_IF_ERROR(BucketExists(bucket, &is_bucket));
      if (is_bucket) {
        *stat = DIRECTORY_STAT;
        return Status::OK();
      }
      return errors::NotFound("The specified bucket ", fname, " was not found.");
    }

    const Status status = StatForObject(bucket, object, stat);
    if (status.ok()) {
      return Status::OK();
    }
    if (status.code() != errors::Code::NOT_FOUND) {
      return status;
    }
    bool is_folder;
    TF_RETURN_IF_ERROR(FolderExists(fname, &is_folder));
    if (is_folder) {
      *stat = DIRECTORY_STAT;
      return Status::OK();
    }
    return errors::NotFound("The specified path ", fname, " was not found.");
  }

  Status JssFileSystem::DeleteFile(const string &fname) {
    string bucket, object;
    TF_RETURN_IF_ERROR(ParseJssPath(fname, false, &bucket, &object));

    string auth_token;
    string method = "DELETE";
    string date_str;
    string content_type;
    string customHead;
    string resource;

    std::unique_ptr<HttpRequest> request(http_request_factory_->Create());
    TF_RETURN_IF_ERROR(request->Init());
    TF_RETURN_IF_ERROR(
        request->SetUri(strings::StrCat("http://", auth_provider_->GetEndPoint(), "/", bucket,
                                        "/", object)));
    TF_RETURN_IF_ERROR(get_time_str(&date_str));
    TF_RETURN_IF_ERROR(request->AddHeader("Date", date_str));

    resource = strings::StrCat("/", bucket, "/", object);
    TF_RETURN_IF_ERROR(JssAuthProvider::GetToken(auth_provider_.get(), request.get(), &auth_token,
                                                 &method, &date_str, &content_type, &customHead, &resource));
    TF_RETURN_IF_ERROR(request->AddHeader("Authorization", auth_token));

    TF_RETURN_IF_ERROR(request->SetDeleteRequest());
    TF_RETURN_WITH_CONTEXT_IF_ERROR(request->Send(), " when deleting", fname);
    return Status::OK();
  }

  Status JssFileSystem::CreateDir(const string &dirname) {
    string bucket, object;
    TF_RETURN_IF_ERROR(ParseJssPath(dirname, true, &bucket, &object));
    if (object.empty()) {
      bool is_bucket;
      TF_RETURN_IF_ERROR(BucketExists(bucket, &is_bucket));
      return is_bucket ? Status::OK()
                       : errors::NotFound("The specified bucket ", dirname,
                                          " was not found.");
    }
    // Create a zero-length directory marker object.
    std::unique_ptr<WritableFile> file;
    TF_RETURN_IF_ERROR(NewWritableFile(MaybeAppendSlash(dirname), &file));
    TF_RETURN_IF_ERROR(file->Close());
    return Status::OK();
  }

// Checks that the directory is empty (i.e no objects with this prefix exist).
// If it is, does nothing, because directories are not entities in JSS.
  Status JssFileSystem::DeleteDir(const string &dirname) {
    std::vector<string> children;
    // A directory is considered empty either if there are no matching objects
    // with the corresponding name prefix or if there is exactly one matching
    // object and it is the directory marker. Therefore we need to retrieve
    // at most two children for the prefix to detect if a directory is empty.
    TF_RETURN_IF_ERROR(
        GetChildrenBounded(dirname, 2, &children, true /* recursively */,
                           true /* include_self_directory_marker */));

    if (children.size() > 1 || (children.size() == 1 && !children[0].empty())) {
      return errors::FailedPrecondition("Cannot delete a non-empty directory.");
    }
    if (children.size() == 1 && children[0].empty()) {
      // This is the directory marker object. Delete it.
      return DeleteFile(MaybeAppendSlash(dirname));
    }
    return Status::OK();
  }

  Status JssFileSystem::GetFileSize(const string &fname, uint64 *file_size) {
    if (!file_size) {
      return errors::Internal("'file_size' cannot be nullptr.");
    }

    // Only validate the name.
    string bucket, object;
    TF_RETURN_IF_ERROR(ParseJssPath(fname, false, &bucket, &object));

    FileStatistics stat;
    TF_RETURN_IF_ERROR(Stat(fname, &stat));
    *file_size = stat.length;
    return Status::OK();
  }

  Status JssFileSystem::RenameFile(const string &src, const string &target) {
    if (!IsDirectory(src).ok()) {
      return RenameObject(src, target);
    }
    // Rename all individual objects in the directory one by one.
    std::vector<string> children;
    TF_RETURN_IF_ERROR(
        GetChildrenBounded(src, UINT64_MAX, &children, true /* recursively */,
                           true /* include_self_directory_marker */));
    for (const string &subpath : children) {
      TF_RETURN_IF_ERROR(
          RenameObject(JoinJssPath(src, subpath), JoinJssPath(target, subpath)));
    }
    return Status::OK();
  }

// Uses a JSS API command to copy the object and then deletes the old one.
  Status JssFileSystem::RenameObject(const string &src, const string &target) {
    string src_bucket, src_object, target_bucket, target_object;
    TF_RETURN_IF_ERROR(ParseJssPath(src, false, &src_bucket, &src_object));
    TF_RETURN_IF_ERROR(ParseJssPath(target, false, &target_bucket, &target_object));


    string auth_token;
    string method = "PUT";
    string date_str;
    string content_type;
    string customHead;
    string resource;

    std::unique_ptr<HttpRequest> request(http_request_factory_->Create());
    TF_RETURN_IF_ERROR(request->Init());
    TF_RETURN_IF_ERROR(
        request->SetUri(strings::StrCat("http://", auth_provider_->GetEndPoint(), "/", target_bucket,
                                        "/", target_object)));

    TF_RETURN_IF_ERROR(get_time_str(&date_str));
    TF_RETURN_IF_ERROR(request->AddHeader("Date", date_str));
    string copy_header = strings::StrCat("/", src_bucket, "/", request->EscapeString(src_object));
    TF_RETURN_IF_ERROR(request->AddHeader("x-jss-copy-source", date_str));
    customHead = strings::StrCat("x-jss-copy-source:", copy_header);
    resource = strings::StrCat("/", target_bucket, "/", target_object);
    TF_RETURN_IF_ERROR(JssAuthProvider::GetToken(auth_provider_.get(), request.get(), &auth_token,
                                                 &method, &date_str, &content_type, &customHead, &resource));
    TF_RETURN_IF_ERROR(request->AddHeader("Authorization", auth_token));

    TF_RETURN_IF_ERROR(request->SetPutEmptyBody());
    std::vector<char> output_buffer;
    TF_RETURN_IF_ERROR(request->SetResultBuffer(&output_buffer));

    TF_RETURN_WITH_CONTEXT_IF_ERROR(request->Send(), " when renaming ", src, " to ", target);

    StringPiece response_piece = StringPiece(output_buffer.data(), output_buffer.size());
    Json::Value root;
    TF_RETURN_IF_ERROR(ParseJson(response_piece, &root));

    string etag; // 几乎没用，就算验证接口正常罢了
    TF_RETURN_IF_ERROR(GetStringValue(root, "Etag", &etag));

    TF_RETURN_IF_ERROR(DeleteFile(src));
    return Status::OK();
  }

  Status JssFileSystem::IsDirectory(const string &fname) {
    string bucket, object;
    TF_RETURN_IF_ERROR(ParseJssPath(fname, true, &bucket, &object));
    if (object.empty()) {
      bool is_bucket;
      TF_RETURN_IF_ERROR(BucketExists(bucket, &is_bucket));
      if (is_bucket) {
        return Status::OK();
      }
      return errors::NotFound("The specified bucket jss://", bucket,
                              " was not found.");
    }
    bool is_folder;
    TF_RETURN_IF_ERROR(FolderExists(fname, &is_folder));
    if (is_folder) {
      return Status::OK();
    }
    bool is_object;
    TF_RETURN_IF_ERROR(ObjectExists(bucket, object, &is_object));
    if (is_object) {
      return errors::FailedPrecondition("The specified path ", fname,
                                        " is not a directory.");
    }
    return errors::NotFound("The specified path ", fname, " was not found.");
  }

  Status JssFileSystem::DeleteRecursively(const string &dirname,
                                          int64 *undeleted_files,
                                          int64 *undeleted_dirs) {
    if (!undeleted_files || !undeleted_dirs) {
      return errors::Internal(
          "'undeleted_files' and 'undeleted_dirs' cannot be nullptr.");
    }
    *undeleted_files = 0;
    *undeleted_dirs = 0;
    if (!IsDirectory(dirname).ok()) {
      *undeleted_dirs = 1;
      return Status(
          error::NOT_FOUND,
          strings::StrCat(dirname, " doesn't exist or not a directory."));
    }
    std::vector<string> all_objects;
    // Get all children in the directory recursively.
    TF_RETURN_IF_ERROR(GetChildrenBounded(
        dirname, UINT64_MAX, &all_objects, true /* recursively */,
        true /* include_self_directory_marker */));
    for (const string &object : all_objects) {
      const string &full_path = JoinJssPath(dirname, object);
      // Delete all objects including directory markers for subfolders.
      if (!DeleteFile(full_path).ok()) {
        if (IsDirectory(full_path).ok()) {
          // The object is a directory marker.
          (*undeleted_dirs)++;
        } else {
          (*undeleted_files)++;
        }
      }
    }
    return Status::OK();
  }

  REGISTER_FILE_SYSTEM("jss", RetryingJssFileSystem);

}  // namespace tensorflow
