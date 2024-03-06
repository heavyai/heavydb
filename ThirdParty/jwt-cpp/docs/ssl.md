# Cryptography Libraries

The underlying cryptography libraries describe [here](../README.md#ssl-compatibility) can be selected when configuring CMake by explicitly setting `JWT_SSL_LIBRARY` to one of three values. The default is to use OpenSSL.

- OpenSSL
- LibreSSL
- wolfSSL

Here's an example:

```sh
cmake . -DJWT_SSL_LIBRARY:STRING=wolfSSL 
```

## Supported Versions

These are the version which are currently being tested:

| OpenSSL           | LibreSSL       | wolfSSL        |
| ----------------- | -------------- | -------------- |
| ![1.0.2u][o1.0.2] | ![3.3.6][l3.3] | ![5.1.1][w5.1] |
| ![1.1.0i][o1.1.0] | ![3.4.3][l3.4] | ![5.2.0][w5.2] |
| ![1.1.1q][o1.1.1] | ![3.5.3][l3.5] | ![5.3.0][w5.3] |
| ![3.0.5][o3.0]    |                |                |

> [!NOTE]
> A complete list of versions tested in the past can be found [here](https://github.com/Thalhammer/jwt-cpp/tree/badges).

[o1.0.2]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Thalhammer/jwt-cpp/badges/openssl/1.0.2u/shields.json
[o1.1.0]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Thalhammer/jwt-cpp/badges/openssl/1.1.0i/shields.json
[o1.1.1]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Thalhammer/jwt-cpp/badges/openssl/1.1.1q/shields.json
[o3.0]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Thalhammer/jwt-cpp/badges/openssl/3.0.5/shields.json
[l3.3]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Thalhammer/jwt-cpp/badges/libressl/3.3.6/shields.json
[l3.4]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Thalhammer/jwt-cpp/badges/libressl/3.4.3/shields.json
[l3.5]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Thalhammer/jwt-cpp/badges/libressl/3.5.3/shields.json
[w5.1]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Thalhammer/jwt-cpp/badges/wolfssl/5.1.1/shields.json
[w5.2]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Thalhammer/jwt-cpp/badges/wolfssl/5.2.0/shields.json
[w5.3]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Thalhammer/jwt-cpp/badges/wolfssl/5.3.0/shields.json

## Notes

JWT-CPP relies on the OpenSSL API, as a result both LibreSSL and wolfSSL need to include their respective compatibility layers.
Most system already have OpenSSL so it's important to make sure when compiling your application it only includes one. Otherwise you may have missing symbols when linking.
