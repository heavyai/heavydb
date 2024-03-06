/// @file print-claims.cpp
#include <iostream>
#include <jwt-cpp/jwt.h>

int main() {
	std::string token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXUyJ9.eyJpc3MiOiJhdXRoMCIsInNhbXBsZSI6InRlc3QifQ.lQm3N2bVlqt2-"
						"1L-FsOjtR6uE-L4E9zJutMWKIe1v1M";
	auto decoded = jwt::decode(token);

	for (auto& e : decoded.get_payload_json())
		std::cout << e.first << " = " << e.second << std::endl;
}
