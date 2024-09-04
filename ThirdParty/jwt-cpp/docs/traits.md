# JSON Traits

Traits define the compatibility mapping for JWT-CPP required functionality to the JSON implementation of choice.

## Selecting a JSON library

For your convenience there are serval traits implementation which provide some popular JSON libraries. They are:

[![picojson][picojson]](https://github.com/kazuho/picojson)
[![nlohmann][nlohmann]](https://github.com/nlohmann/json)
[![jsoncons][jsoncons]](https://github.com/danielaparker/jsoncons)
[![boostjson][boostjson]](https://github.com/boostorg/json)
[![jsoncpp][jsoncpp]](https://github.com/open-source-parsers/jsoncpp)

[picojson]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Thalhammer/jwt-cpp/badges/traits/kazuho-picojson/shields.json
[nlohmann]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Thalhammer/jwt-cpp/badges/traits/nlohmann-json/shields.json
[jsoncons]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Thalhammer/jwt-cpp/badges/traits/danielaparker-jsoncons/shields.json
[boostjson]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Thalhammer/jwt-cpp/badges/traits/boost-json/shields.json
[jsoncpp]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Thalhammer/jwt-cpp/badges/traits/open-source-parsers-jsoncpp/shields.json

In order to maintain compatibility, [picojson](https://github.com/kazuho/picojson) is still used to provide a specialized `jwt::claim` along with all helpers. Defining `JWT_DISABLE_PICOJSON` will remove this optional dependency. It's possible to directly include the traits defaults for the other JSON libraries. See the [traits examples](https://github.com/Thalhammer/jwt-cpp/tree/master/example/traits) for details.

```cpp
//include "jwt-cpp/traits/author-library/traits.h"
#include "jwt-cpp/traits/nlohmann-json/traits.h"
// There is also a "defaults.h" if you's like to skip providing the
// template specializations for the JSON traits

int main() {
    // All the provided traits are in jwt::traits namespace
    using traits = jwt::traits::nlohmann_json;

    const auto time = jwt::date::clock::now();
    const auto token = jwt::create<traits>()
                           .set_type("JWT")
                           .set_issuer("auth.mydomain.io")
                           .set_audience("mydomain.io")
                           .set_issued_at(time)
                           .set_not_before(time)
                           .set_expires_at(time + std::chrono::minutes{2} + std::chrono::seconds{15})
                           .sign(jwt::algorithm::none{});
    const auto decoded = jwt::decode<traits>(token);

    jwt::verify<traits>()
        .allow_algorithm(jwt::algorithm::none{})
        .with_issuer("auth.mydomain.io")
        .with_audience("mydomain.io")
        .verify(decoded);
```

## Providing your own JSON Traits

There are several key items that need to be provided to a `jwt::basic_claim` in order for it to be interoperable with you JSON library of choice.

* type specifications
* conversion from generic "value type" to a specific type
* serialization and parsing

If ever you are not sure, the traits are heavily checked against static asserts to make sure you provide everything that's required.

> [!important]
> Not all JSON libraries are a like, you may need to extend certain types such that it can be used. See this [provided implementation](https://github.com/Thalhammer/jwt-cpp/blob/e6b92cca0b7088027269c481fa244e5c39df88ff/include/jwt-cpp/traits/danielaparker-jsoncons/traits.h#L18).

```cpp
struct my_favorite_json_library_traits {
    // Type Specifications
    using value_type = json; // The generic "value type" implementation, most libraries have one
    using object_type = json::object_t; // The "map type" string to value
    using array_type = json::array_t; // The "list type" array of values
    using string_type = std::string; // The "list of chars", must be a narrow char
    using number_type = double; // The "precision type"
    using integer_type = int64_t; // The "integral type"
    using boolean_type = bool; // The "boolean type"

    // Translation between the implementation notion of type, to the jwt::json::type equivalent
    static jwt::json::type get_type(const value_type &val) {
        using jwt::json::type;

        if (val.type() == json::value_t::object)
            return type::object;
        if (val.type() == json::value_t::array)
            return type::array;
        if (val.type() == json::value_t::string)
            return type::string;
        if (val.type() == json::value_t::number_float)
            return type::number;
        if (val.type() == json::value_t::number_integer)
            return type::integer;
        if (val.type() == json::value_t::boolean)
            return type::boolean;

        throw std::logic_error("invalid type");
    }

    // Conversion from generic value to specific type
    static object_type as_object(const value_type &val);
    static array_type as_array(const value_type &val);
    static string_type as_string(const value_type &val);
    static number_type as_number(const value_type &val);
    static integer_type as_integer(const value_type &val);
    static boolean_type as_boolean(const value_type &val);

    // serialization and parsing
    static bool parse(value_type &val, string_type str);
    static string_type serialize(const value_type &val); // with no extra whitespace, padding or indentation
};
```
