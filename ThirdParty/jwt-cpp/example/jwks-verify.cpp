/** 
 * \file jwks-verify.cpp
 * 
 * Novel example using a JWT's "key ID" to match with a JWK Set
 * and using the corresponding x5c from the JWK to verify the token
 */
#include <iostream>
#include <jwt-cpp/jwt.h>
#include <openssl/rand.h>

int main() {
	std::string raw_jwks =
		R"({"keys": [{
		"kid":"internal-gateway-jwt.api.sc.net",
		"alg": "RS256",
    "kty": "RSA",
    "use": "sig",
    "x5c": [
      "MIIE2jCCAsICAQEwDQYJKoZIhvcNAQELBQAwMzELMAkGA1UEBhMCVVMxEDAOBgNVBAoMB0pXVC1DUFAxEjAQBgNVBAMMCWxvY2FsaG9zdDAeFw0yMzEyMjIxMzIzNTdaFw0zMzEyMTkxMzIzNTdaMDMxCzAJBgNVBAYTAlVTMRAwDgYDVQQKDAdKV1QtQ1BQMRIwEAYDVQQDDAlsb2NhbGhvc3QwggIiMA0GCSqGSIb3DQEBAQUAA4ICDwAwggIKAoICAQDl0gyL9KCpoXsJlvBaBUsJTAgvFhbgsBjpxT6m2xv0fgtGDBwgaiPvuMnClTU/kYkKb8c1GTkMedKp9YcM57HWTk9yqGTy6QBnMMxbAJYNwWQ4Dbr4qKSC6C3KzYws/Bqyv8OC9NAOyqJbtdp4iObRjyaet+PLTXywuu02xtyRg3B+1aAONgUVDyS5u57NSD4rEZ+rw30Ne1doSClWmMDqEd72y8cjx3eAqn0HcAxSQ6MNMmNk7/M8FQD3DTM1Ef0G5oHyJIw7WmY+gxuD8386r/CkswINzadMwObPlTSdAN8BRzedtrqgb+D/K4pi2zhCiuIVujFX6M/hsGvj7g2M9E9MR8iEuHWCY9frQKIR+JTH3D1snoJp60qKoa51qBznsEr9RP2utGniPCq3+JY+ZX0JK8vl5tiSZpy6N0yRbRmY3XLdA5fKRzhcsB3eUrmTtr9ywjZX7Ll6QMvUyicubGTojhqJFQbvuvvops9PoCMXFE3x6cJ2QhPoi8+BvUdYisrtjDFe+YgrgQvPMa/CpOpDJJDEs2SVRcauCZOUdqLCwZylNuW0CgIjWP8l99P7l1zGeT8VJPhmABYyPM+RtNYDamAlUOCqRqgz/gPjEeMeulQTvH1lAqATAAX1oftlq6o4VoqROs2M3eAXqPhvsLBeTmCob+5ca887MkcP6wIDAQABMA0GCSqGSIb3DQEBCwUAA4ICAQBW2kREK4hlzxCDqykxrwfbQpiPwrbFmn+3RDJla+pI4L3wrvYT1nU96guFIU3zKnbMzqwPMRUCUjadr2jKxAmMWxCd/ThHQB+ne5xTvx7/6RVQfGjyMCG/SZtSH8/aO7ILNRtPT+SL5ZZwezaqv6gD89tSXB/w/0pYXy70wDuU17KCrTsKSISWGJ1cKi5l2R/m/ZaGjcV8U8NcFepF2bX3u/i0zhaqOqjiwrSEt7fWGDLabPs6n7GtfibZROEDZ/h0JrDINC+6mSfTOYAMJvGjeHA3H/NvzqR+CJgpXGCqElqVuBF0HdxPmwRRBoZC/BLIEcz0VHmB4rcpfaV47TZT+J+04fHYp4Y1S0u112CDrDe+61cDrnbDHC7aGX0G93pYSBKAB1e3LLc9rXQgf2F0pRtFB3rgZA9MtJ+TL7DUvY4VXJNq3v7UolIdldYRdk21YqAS2Hp0fivvFoEk2P/WbwDEErxR0FkZ/JQoI9FMJ9AvDxa4MsFFtlQVInfD2HUu+nhnuEAA8R6L+F2XqhfLY/H7H31iFBK6UCuqptED71VwWHqfBsAPRhLXAqGco7Ln2dzioyj0QdwJqQQIqigltSYtXxfIMLW0BekQ5yln7QTxnZlobkPHUW9s3NK+OMLuKCzVREzjic/aioQP3cRBMXkG2deMwrk3aX8yJuz4gA=="
    ],
    "n": "5dIMi_SgqaF7CZbwWgVLCUwILxYW4LAY6cU-ptsb9H4LRgwcIGoj77jJwpU1P5GJCm_HNRk5DHnSqfWHDOex1k5Pcqhk8ukAZzDMWwCWDcFkOA26-Kikgugtys2MLPwasr_DgvTQDsqiW7XaeIjm0Y8mnrfjy018sLrtNsbckYNwftWgDjYFFQ8kubuezUg-KxGfq8N9DXtXaEgpVpjA6hHe9svHI8d3gKp9B3AMUkOjDTJjZO_zPBUA9w0zNRH9BuaB8iSMO1pmPoMbg_N_Oq_wpLMCDc2nTMDmz5U0nQDfAUc3nba6oG_g_yuKYts4QoriFboxV-jP4bBr4-4NjPRPTEfIhLh1gmPX60CiEfiUx9w9bJ6CaetKiqGudagc57BK_UT9rrRp4jwqt_iWPmV9CSvL5ebYkmacujdMkW0ZmN1y3QOXykc4XLAd3lK5k7a_csI2V-y5ekDL1MonLmxk6I4aiRUG77r76KbPT6AjFxRN8enCdkIT6IvPgb1HWIrK7YwxXvmIK4ELzzGvwqTqQySQxLNklUXGrgmTlHaiwsGcpTbltAoCI1j_JffT-5dcxnk_FST4ZgAWMjzPkbTWA2pgJVDgqkaoM_4D4xHjHrpUE7x9ZQKgEwAF9aH7ZauqOFaKkTrNjN3gF6j4b7CwXk5gqG_uXGvPOzJHD-s",
    "e": "AQAB"
	},
{
		"kid":"internal-123456",
		"use":"sig",
		"x5c":["MIIG1TCCBL2gAwIBAgIIFvMVGp6t\/cMwDQYJKoZIhvcNAQELBQAwZjELMAkGA1UEBhMCR0IxIDAeBgNVBAoMF1N0YW5kYXJkIENoYXJ0ZXJlZCBCYW5rMTUwMwYDVQQDDCxTdGFuZGFyZCBDaGFydGVyZWQgQmFuayBTaWduaW5nIENBIEcxIC0gU0hBMjAeFw0xODEwMTAxMTI2MzVaFw0yMjEwMTAxMTI2MzVaMIG9MQswCQYDVQQGEwJTRzESMBAGA1UECAwJU2luZ2Fwb3JlMRIwEAYDVQQHDAlTaW5nYXBvcmUxIDAeBgNVBAoMF1N0YW5kYXJkIENoYXJ0ZXJlZCBCYW5rMRwwGgYDVQQLDBNGb3VuZGF0aW9uIFNlcnZpY2VzMSgwJgYDVQQDDB9pbnRlcm5hbC1nYXRld2F5LWp3dC5hcGkuc2MubmV0MRwwGgYJKoZIhvcNAQkBFg1BUElQU1NAc2MuY29tMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEArVWBoIi3IJ4nOWXu7\/SDxczqMou1B+c4c2FdQrOXrK31HxAaz4WEtma9BLXFdFHJ5mCCPIvdUcVxxnCynqhMOkZ\/a7acQbUD9cDzI8isMB9JL7VooDw0CctxHxffjqQQVIEhC2Q7zsM1pQayR7cl+pbBlvHIoRxq2n1B0fFvfoiosjf4kDiCpgHdM+v5Hw9aVYmUbroHxmQWqhB0iRTJQPPLZqqQVC50A1Q\/96gkwoODyotc46Uy9wYEpdGrtDG\/thWay3fmMsjpWR0U25xFIrxTrfCGBblYpD7juukWWml2E9rtE2rHgUxbymxXjEw7xrMwcGrhOGyqwoBqJy1JVwIDAQABo4ICLTCCAikwZAYIKwYBBQUHAQEEWDBWMFQGCCsGAQUFBzABhkhodHRwOi8vY29yZW9jc3AuZ2xvYmFsLnN0YW5kYXJkY2hhcnRlcmVkLmNvbS9lamJjYS9wdWJsaWN3ZWIvc3RhdHVzL29jc3AwHQYDVR0OBBYEFIinW4BNDeVEFcuLf8YjZjtySoW9MAwGA1UdEwEB\/wQCMAAwHwYDVR0jBBgwFoAUfNZMoZi33nKrcmVU3TFVQnuEi\/4wggFCBgNVHR8EggE5MIIBNTCCATGggcKggb+GgbxodHRwOi8vY29yZWNybC5nbG9iYWwuc3RhbmRhcmRjaGFydGVyZWQuY29tL2VqYmNhL3B1YmxpY3dlYi93ZWJkaXN0L2NlcnRkaXN0P2NtZD1jcmwmaXNzdWVyPUNOPVN0YW5kYXJkJTIwQ2hhcnRlcmVkJTIwQmFuayUyMFNpZ25pbmclMjBDQSUyMEcxJTIwLSUyMFNIQTIsTz1TdGFuZGFyZCUyMENoYXJ0ZXJlZCUyMEJhbmssQz1HQqJqpGgwZjE1MDMGA1UEAwwsU3RhbmRhcmQgQ2hhcnRlcmVkIEJhbmsgU2lnbmluZyBDQSBHMSAtIFNIQTIxIDAeBgNVBAoMF1N0YW5kYXJkIENoYXJ0ZXJlZCBCYW5rMQswCQYDVQQGEwJHQjAOBgNVHQ8BAf8EBAMCBsAwHQYDVR0lBBYwFAYIKwYBBQUHAwIGCCsGAQUFBwMEMA0GCSqGSIb3DQEBCwUAA4ICAQBtsoRlDHuOTDChcWdfdVUtRgP0U0ijDSeJi8vULN1rgYnqqJc4PdJno50aiu9MGlxY02O7HW7ZVD6QEG\/pqHmZ0sbWpb\/fumMgZSjP65IcGuS53zgcNtLYnyXyEv+v5T\/CK3bk4Li6tUW3ScJPUwVWwP1E0\/u6aBSb5k\/h4lTwS1o88ybS5pJOg6XutXByp991QQrrs7tp7fKNynjNZbFuG3J1e09X+zTfJOpjaDUofQTkt8IyMRI6Cs4wI1eZA+dAIL8B0n8ze1mRl1FOJqgdZrAQjoqZkCTnc0Il5VY\/dUXxGVg6D9e5pfck3FWT107K9\/5EZoxytpqYXFCjMXi5hx4YjK17OUgm82mZhvqkNdzF8Yq2vFuB3LPfyelESq99xFLykvinrVm1NtZKeDTT1Jq\/VvZt6stO\/tovq1RfJJcznpYcwOzxlnhGR6E+hxuBx7aDJzGf0JaoRxQILH1B2XV9WDI3HPYQsP7XtriX+QUJ\/aly28QkV48RmaGYCsly43YZu1MKudSsw+dhnbZzRsg\/aes3dzGW2x137bQPtux7k2LCSpsTXgedhOys28YoGlsoe8kUv0myAU4Stt+I3mrwO3BKUn+tJggvlDiiiyT1tg2HiklyU\/2FxQkZRMeB0eRrXTpg3l9x2mpF+dDFxOMKszxwD2kgoEZgo6o58A=="],
		"n":"nr9UsxnPVd21iuiGcIJ_Qli2XVlAZe5VbELA1hO2-L4k5gI4fjHZ3ysUcautLpbOYogOQgsnlpsLrCmvNDvBDVzVp2nMbpguJlt12vHSP1fRJJpipGQ8qU-VaXsC4OjOQf3H9ojAU5Vfnl5gZ7kVCd8g4M29l-IRyNpxE-Ccxc2Y7molsCHT6GHLMMBVsd11JIOXMICJf4hz2YYkQ1t7C8SaB2RFRPuGO5Mn6mfAnwdmRera4TBz6_pIPPCgCbN8KOdJItWkr9F7Tjv_0nhh-ZVlQvbQ9PXHyKTj00g3IYUlbZIWHm0Ley__fzNZk2dyAAVjNA2QSzTZJc33MQx1pQ",
		"e":"AQAB",
		"x5t":"-qC0akuyiHTV5aFsKVWM9da7lzq6DLrj09I",
		"alg":"RS256",
		"kty":"RSA"
	}
]})";

	std::string pem_priv_key = R"(-----BEGIN PRIVATE KEY-----
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
+grQSrtFfSnickR6i3XrDlspd/khcQ==
-----END PRIVATE KEY-----)";

	// https://stackoverflow.com/a/30138974
	unsigned char nonce[24];
	RAND_bytes(nonce, sizeof(nonce));
	std::string jti =
		jwt::base::encode<jwt::alphabet::base64url>(std::string{reinterpret_cast<const char*>(nonce), sizeof(nonce)});

	std::string token = jwt::create()
							.set_issuer("auth0")
							.set_type("JWT")
							.set_id(jti)
							.set_key_id("internal-gateway-jwt.api.sc.net")
							.set_subject("jwt-cpp.example.localhost")
							.set_issued_at(std::chrono::system_clock::now())
							.set_expires_at(std::chrono::system_clock::now() + std::chrono::seconds{36000})
							.set_payload_claim("sample", jwt::claim(std::string{"test"}))
							.sign(jwt::algorithm::rs256("", pem_priv_key, "", ""));

	auto decoded_jwt = jwt::decode(token);
	auto jwks = jwt::parse_jwks(raw_jwks);
	auto jwk = jwks.get_jwk(decoded_jwt.get_key_id());

	auto issuer = decoded_jwt.get_issuer();
	auto x5c = jwk.get_x5c_key_value();

	if (!x5c.empty() && !issuer.empty()) {
		std::cout << "Verifying with 'x5c' key" << std::endl;
		auto verifier =
			jwt::verify()
				.allow_algorithm(jwt::algorithm::rs256(jwt::helper::convert_base64_der_to_pem(x5c), "", "", ""))
				.with_issuer(issuer)
				.with_id(jti)
				.leeway(60UL); // value in seconds, add some to compensate timeout

		verifier.verify(decoded_jwt);
	}
	// else if the optional 'x5c' was not present
	{
		std::cout << "Verifying with RSA components" << std::endl;
		const auto modulus = jwk.get_jwk_claim("n").as_string();
		const auto exponent = jwk.get_jwk_claim("e").as_string();
		auto verifier = jwt::verify()
							.allow_algorithm(jwt::algorithm::rs256(
								jwt::helper::create_public_key_from_rsa_components(modulus, exponent)))
							.with_issuer(issuer)
							.with_id(jti)
							.leeway(60UL); // value in seconds, add some to compensate timeout

		verifier.verify(decoded_jwt);
	}
}
