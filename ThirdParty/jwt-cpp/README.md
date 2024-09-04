<img src="https://raw.githubusercontent.com/Thalhammer/jwt-cpp/master/.github/logo.svg" alt="logo" width="100%">

[![License Badge](https://img.shields.io/github/license/Thalhammer/jwt-cpp)](https://github.com/Thalhammer/jwt-cpp/blob/master/LICENSE)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/5f7055e294744901991fd0a1620b231d)](https://app.codacy.com/gh/Thalhammer/jwt-cpp/dashboard)
[![Linux Badge][Linux]][Cross-Platform]
[![MacOS Badge][MacOS]][Cross-Platform]
[![Windows Badge][Windows]][Cross-Platform]
[![Coverage Status](https://coveralls.io/repos/github/Thalhammer/jwt-cpp/badge.svg?branch=master)](https://coveralls.io/github/Thalhammer/jwt-cpp?branch=master)

[![Documentation Badge](https://img.shields.io/badge/Documentation-master-blue)](https://thalhammer.github.io/jwt-cpp/)

[![Stars Badge](https://img.shields.io/github/stars/Thalhammer/jwt-cpp?style=flat)](https://github.com/Thalhammer/jwt-cpp/stargazers)
[![GitHub release (latest SemVer including pre-releases)](https://img.shields.io/github/v/release/Thalhammer/jwt-cpp?include_prereleases)](https://github.com/Thalhammer/jwt-cpp/releases)
[![ConanCenter package](https://repology.org/badge/version-for-repo/conancenter/jwt-cpp.svg)](https://repology.org/project/jwt-cpp/versions)
[![Vcpkg package](https://repology.org/badge/version-for-repo/vcpkg/jwt-cpp.svg)](https://repology.org/project/jwt-cpp/versions)

[Linux]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Thalhammer/jwt-cpp/badges/cross-platform/ubuntu-latest/shields.json
[MacOS]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Thalhammer/jwt-cpp/badges/cross-platform/macos-latest/shields.json
[Windows]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Thalhammer/jwt-cpp/badges/cross-platform/windows-latest/shields.json
[Cross-Platform]: https://github.com/Thalhammer/jwt-cpp/actions?query=workflow%3A%22Cross-Platform+CI%22

## Overview

A header only library for creating and validating [JSON Web Tokens](https://tools.ietf.org/html/rfc7519) in C++11. For a great introduction, [read this](https://jwt.io/introduction/).

The objective is to deliver a versatile and universally applicable collection of algorithms, classes, and data structures, fostering adaptability and seamless integration with other libraries that you may already be employing.

## Signature algorithms

jwt-cpp comprehensively supports all algorithms specified in the standard. Its modular design facilitates the seamless [inclusion of additional algorithms](docs/signing.md#custom-signature-algorithms) without encountering any complications. Should you wish to contribute new algorithms, feel free to initiate a pull request or [open an issue](https://github.com/Thalhammer/jwt-cpp/issues/new).

For completeness, here is a list of all supported algorithms:

| HMSC  | RSA   | ECDSA  | PSS   | EdDSA   |
| ----- | ----- | ------ | ----- | ------- |
| HS256 | RS256 | ES256  | PS256 | Ed25519 |
| HS384 | RS384 | ES384  | PS384 | Ed448   |
| HS512 | RS512 | ES512  | PS512 |         |
|       |       | ES256K |       |         |

## Getting Started

Installation instructions can be found [here](docs/install.md).

A simple example is decoding a token and printing all of its [claims](https://tools.ietf.org/html/rfc7519#section-4) let's ([try it out](https://jwt.io/#debugger-io?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXUyJ9.eyJpc3MiOiJhdXRoMCIsInNhbXBsZSI6InRlc3QifQ.lQm3N2bVlqt2-1L-FsOjtR6uE-L4E9zJutMWKIe1v1M)):

```cpp
#include <jwt-cpp/jwt.h>
#include <iostream>

int main() {
    std::string token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXUyJ9.eyJpc3MiOiJhdXRoMCIsInNhbXBsZSI6InRlc3QifQ.lQm3N2bVlqt2-1L-FsOjtR6uE-L4E9zJutMWKIe1v1M";
    auto decoded = jwt::decode(token);

    for(auto& e : decoded.get_payload_json())
        std::cout << e.first << " = " << e.second << std::endl;
}
```

You can build and run [this example](example/print-claims.cpp) locally after cloning the repository.
Running some commands, we can see the contents of the [JWT payload](https://datatracker.ietf.org/doc/html/rfc7519#section-3)

```sh
cmake .
cmake --build . --target print-claims
./print-claims
# iss = "auth0"
# sample = "test"
```

You'll very quickly notice JWT are not encrypted but rather cryptographically signed to
provide [non-repudiation](https://csrc.nist.gov/glossary/term/non_repudiation).

In order to verify a token you first build a verifier and use it to verify a decoded token.

```cpp
auto verifier = jwt::verify()
    .with_issuer("auth0")
    .with_claim("sample", jwt::claim(std::string("test")));
    .allow_algorithm(jwt::algorithm::hs256{"secret"})

verifier.verify(decoded_token);
```

The verifier is stateless so you can reuse it for different tokens.

Creating the token above (and signing it) is equally as easy.

```cpp
auto token = jwt::create()
    .set_type("JWS")
    .set_issuer("auth0")
    .set_payload_claim("sample", jwt::claim(std::string("test")))
    .sign(jwt::algorithm::hs256{"secret"});
```

If you are looking to issue or verify more unique tokens, checkout out the [examples](https://github.com/Thalhammer/jwt-cpp/tree/master/example) working with RSA public and private keys, elliptic curve tokens, and much more!

### Configuration Options

Building on the goal of providing flexibility.

#### SSL Compatibility

jwt-cpp supports [OpenSSL](https://github.com/openssl/openssl), [LibreSSL](https://github.com/libressl-portable/portable), and [wolfSSL](https://github.com/wolfSSL/wolfssl). For a listed of tested versions, check [this page](docs/ssl.md) for more details.

#### JSON Implementation

There is no strict reliance on a specific JSON library in this context. Instead, the jwt-cpp utilizes a generic `jwt::basic_claim` that is templated based on type trait. This trait provides the semantic [JSON types](https://json-schema.org/understanding-json-schema/reference/type.html) for values, objects, arrays, strings, numbers, integers, and booleans, along with methods to seamlessly translate between them.

This design offers flexibility in choosing the JSON library that best suits your needs. To leverage one of the provided JSON traits, refer to [docs/traits.md](docs/traits.md#selecting-a-json-library) for detailed guidance.

##### Providing your own JSON Traits

```cpp
jwt::basic_claim<my_favorite_json_library_traits> claim(json::object({{"json", true},{"example", 0}}));
```

To learn how to writes a trait's implementation, checkout the [these instructions](docs/traits.md#providing-your-own-json-traits)

#### Base64 Options

With regard to the base64 specifications for JWTs, this library includes `base.h` encompassing all necessary variants. While the library itself offers a proficient base64 implementation, it's worth noting that base64 implementations are widely available, exhibiting diverse performance levels. If you prefer to use your own base64 implementation, you have the option to define `JWT_DISABLE_BASE64` to exclude the jwt-cpp implementation.

## Contributing

If you have suggestions for improvement or if you've identified a bug, please don't hesitate to [open an issue](https://github.com/Thalhammer/jwt-cpp/issues/new) or contribute by creating a pull request. When reporting a bug, provide comprehensive details about your environment, including compiler version and other relevant information, to facilitate issue reproduction. Additionally, if you're introducing a new feature, ensure that you include corresponding test cases to validate its functionality.

### Dependencies

In order to use jwt-cpp you need the following tools.

* libcrypto (openssl or compatible)
* libssl-dev (for the header files)
* a compiler supporting at least c++11
* basic stl support

In order to build the test cases you also need

* gtest
* pthread

## Troubleshooting

See the [FAQs](docs/faqs.md) for tips.

## Conference Coverage

[![CppCon](https://img.youtube.com/vi/Oq4NW5idmiI/0.jpg)](https://www.youtube.com/watch?v=Oq4NW5idmiI)
