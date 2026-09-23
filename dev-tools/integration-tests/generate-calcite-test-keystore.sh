#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Generate dev TLS material for HeavyDB + Calcite using a single self-signed CA.
#
# Requires: openssl 3.x at OPENSSL_BIN and keytool (JDK) for the Java trust store.
#
# Two TLS links (both must be configured):
#   Calcite (Java) -> HeavyDB: Calcite uses ssl-trust-store (PKCS#12) to trust HeavyDB.
#   HeavyDB (C++)  -> Calcite: HeavyDB uses ssl-trust-ca (PEM CA) to trust Calcite.
#
# Output layout (OUT_DIR):
#   ca/ca.key, ca/ca.crt
#   server/calcite-*          Calcite Thrift server (Java --keystore)
#   heavydb/heavydb-*         HeavyDB Thrift server
#   truststore/truststore.p12 Java trust store (--ssl-trust-store for Calcite client)
#
# HeavyDB C++ server (--ssl-cert / --ssl-private-key) uses PEM files:
#   heavydb/heavydb-server.crt
#   heavydb/heavydb-server.key
#
# PKCS#12 bundles (password defaults to banana):
#   server/calcite-keystore.p12   Calcite --keystore
#   heavydb/heavydb-server.p12    optional bundle of server identity
#   truststore/truststore.p12     Calcite client trust of HeavyDB / Calcite chain
#   heavyai-tls.conf              heavydb --config file (paths + Java store passwords)

set -euo pipefail

OPENSSL_BIN="${OPENSSL_BIN:-/usr/local/mapd-deps/bin/openssl}"
KEYTOOL_BIN="${KEYTOOL_BIN:-keytool}"
OUT_DIR="${OUT_DIR:-./calcite-tls}"
CONFIG_FILENAME="${CONFIG_FILENAME:-heavyai-tls.conf}"
KEYSTORE_PASSWORD="${KEYSTORE_PASSWORD:-banana}"
TRUSTSTORE_PASSWORD="${TRUSTSTORE_PASSWORD:-banana}"
DAYS_CA="${DAYS_CA:-3650}"
DAYS_SERVER="${DAYS_SERVER:-3650}"
KEY_BITS="${KEY_BITS:-4096}"

CA_SUBJ="${CA_SUBJ:-/CN=HeavyDB Dev CA/O=HeavyAI/C=US}"
CALCITE_SUBJ="${CALCITE_SUBJ:-/CN=calcite/O=HeavyAI/C=US}"
CALCITE_SAN="${CALCITE_SAN:-DNS:localhost,DNS:calcite,IP:127.0.0.1}"
HEAVYDB_SUBJ="${HEAVYDB_SUBJ:-/CN=heavydb/O=HeavyAI/C=US}"
HEAVYDB_SAN="${HEAVYDB_SAN:-DNS:localhost,DNS:heavydb,IP:127.0.0.1}"

init_output_paths() {
  mkdir -p "${OUT_DIR}"
  OUT_DIR="$(cd "${OUT_DIR}" && pwd)"
  CA_DIR="${OUT_DIR}/ca"
  CALCITE_DIR="${OUT_DIR}/server"
  HEAVYDB_DIR="${OUT_DIR}/heavydb"
  TRUSTSTORE_DIR="${OUT_DIR}/truststore"
  CA_KEY="${CA_DIR}/ca.key"
  CA_CERT="${CA_DIR}/ca.crt"
  CONFIG_FILE="${OUT_DIR}/${CONFIG_FILENAME}"
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_openssl() {
  command -v "${OPENSSL_BIN}" >/dev/null 2>&1 || die "openssl not found at ${OPENSSL_BIN}"
  echo "Using $("${OPENSSL_BIN}" version)"
}

write_server_ext_file() {
  local ext_file="$1"
  local san_list="$2"
  cat > "${ext_file}" <<EOF
basicConstraints = critical, CA:FALSE
keyUsage = critical, digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = ${san_list}
EOF
}

generate_self_signed_ca() {
  mkdir -p "${CA_DIR}"

  if [[ ! -f "${CA_KEY}" ]]; then
    echo "Generating CA private key..."
    "${OPENSSL_BIN}" genrsa -out "${CA_KEY}" "${KEY_BITS}"
    chmod 600 "${CA_KEY}"
  fi

  if [[ ! -f "${CA_CERT}" ]]; then
    echo "Generating self-signed CA certificate..."
    "${OPENSSL_BIN}" req -new -x509 \
      -days "${DAYS_CA}" \
      -key "${CA_KEY}" \
      -out "${CA_CERT}" \
      -subj "${CA_SUBJ}" \
      -sha256
  fi
}

sign_server_certificate() {
  local name="$1"
  local subj="$2"
  local san_list="$3"
  local out_dir="$4"
  local key_file="${out_dir}/${name}.key"
  local csr_file="${out_dir}/${name}.csr"
  local cert_file="${out_dir}/${name}.crt"
  local ext_file="${out_dir}/${name}-ext.cnf"

  mkdir -p "${out_dir}"

  if [[ ! -f "${key_file}" ]]; then
    echo "Generating ${name} private key..."
    "${OPENSSL_BIN}" genrsa -out "${key_file}" "${KEY_BITS}"
    chmod 600 "${key_file}"
  fi

  if [[ ! -f "${csr_file}" ]]; then
    echo "Generating ${name} CSR..."
    "${OPENSSL_BIN}" req -new \
      -key "${key_file}" \
      -out "${csr_file}" \
      -subj "${subj}" \
      -sha256
  fi

  write_server_ext_file "${ext_file}" "${san_list}"

  echo "Signing ${name} certificate with CA..."
  "${OPENSSL_BIN}" x509 -req \
    -in "${csr_file}" \
    -CA "${CA_CERT}" \
    -CAkey "${CA_KEY}" \
    -CAcreateserial \
    -out "${cert_file}" \
    -days "${DAYS_SERVER}" \
    -sha256 \
    -extfile "${ext_file}"

  "${OPENSSL_BIN}" verify -CAfile "${CA_CERT}" "${cert_file}"
}

create_pkcs12_keystore() {
  local key_file="$1"
  local cert_file="$2"
  local p12_file="$3"
  local entry_name="$4"
  local password="$5"

  echo "Creating PKCS#12 keystore ${p12_file} (password: ${password})..."
  rm -f "${p12_file}"
  "${OPENSSL_BIN}" pkcs12 -export \
    -inkey "${key_file}" \
    -in "${cert_file}" \
    -certfile "${CA_CERT}" \
    -out "${p12_file}" \
    -name "${entry_name}" \
    -passout "pass:${password}"
  chmod 600 "${p12_file}"
}

generate_calcite_server_material() {
  local name="calcite-server"
  local key_file="${CALCITE_DIR}/${name}.key"
  local cert_file="${CALCITE_DIR}/${name}.crt"
  local p12_file="${CALCITE_DIR}/calcite-keystore.p12"

  sign_server_certificate "${name}" "${CALCITE_SUBJ}" "${CALCITE_SAN}" "${CALCITE_DIR}"
  create_pkcs12_keystore \
    "${key_file}" \
    "${cert_file}" \
    "${p12_file}" \
    "calcite-server" \
    "${KEYSTORE_PASSWORD}"
}

generate_heavydb_server_material() {
  local name="heavydb-server"
  local key_file="${HEAVYDB_DIR}/${name}.key"
  local cert_file="${HEAVYDB_DIR}/${name}.crt"
  local p12_file="${HEAVYDB_DIR}/${name}.p12"

  sign_server_certificate "${name}" "${HEAVYDB_SUBJ}" "${HEAVYDB_SAN}" "${HEAVYDB_DIR}"
  create_pkcs12_keystore \
    "${key_file}" \
    "${cert_file}" \
    "${p12_file}" \
    "heavydb-server" \
    "${KEYSTORE_PASSWORD}"

  echo ""
  echo "HeavyDB loads PEM paths at startup (C++ Thrift SSL):"
  echo "  --ssl-cert ${cert_file}"
  echo "  --ssl-private-key ${key_file}"
  echo "PKCS#12 bundle (same identity): ${p12_file}"
}

verify_java_truststore() {
  local truststore_file="$1"
  command -v "${KEYTOOL_BIN}" >/dev/null 2>&1 || return 0
  "${KEYTOOL_BIN}" -list -keystore "${truststore_file}" \
    -storetype PKCS12 \
    -storepass "${TRUSTSTORE_PASSWORD}" >/dev/null
}

generate_java_truststore() {
  local truststore_file="${TRUSTSTORE_DIR}/truststore.p12"

  mkdir -p "${TRUSTSTORE_DIR}"

  command -v "${KEYTOOL_BIN}" >/dev/null 2>&1 \
    || die "keytool not found (${KEYTOOL_BIN}). Required for a Java-readable trust store."

  echo "Creating Java trust store with CA certificate (password: ${TRUSTSTORE_PASSWORD})..."
  rm -f "${truststore_file}"
  # OpenSSL's cert-only PKCS#12 export is not loaded correctly by Java (0 entries).
  # Use keytool so Calcite can trust HeavyDB's server certificate.
  "${KEYTOOL_BIN}" -importcert -noprompt -trustcacerts \
    -alias heavydb-dev-ca \
    -file "${CA_CERT}" \
    -keystore "${truststore_file}" \
    -storetype PKCS12 \
    -storepass "${TRUSTSTORE_PASSWORD}"
  chmod 600 "${truststore_file}"
  verify_java_truststore "${truststore_file}"

  echo "Java trust store: ${truststore_file}"
  echo "  Calcite client -> HeavyDB: --ssl-trust-store ${truststore_file}"
  echo "  Password: --ssl-trust-password ${TRUSTSTORE_PASSWORD}"
  echo "  (or ssl-trust-password=${TRUSTSTORE_PASSWORD} in config file)"
  echo "HeavyDB -> Calcite uses PEM CA: ${CA_CERT} (ssl-trust-ca in config)"
}

generate_heavydb_config() {
  local heavydb_cert="${HEAVYDB_DIR}/heavydb-server.crt"
  local heavydb_key="${HEAVYDB_DIR}/heavydb-server.key"
  local truststore="${TRUSTSTORE_DIR}/truststore.p12"
  local calcite_keystore="${CALCITE_DIR}/calcite-keystore.p12"

  cat > "${CONFIG_FILE}" <<EOF
# Generated by $(basename "$0") on $(date -u +"%Y-%m-%d %H:%M:%S UTC")
# Dev / test TLS only — do not use in production.
#
# Start HeavyDB:
#   heavydb --config "${CONFIG_FILE}" --data /path/to/storage
#
# Connect with heavysql (TLS required when ssl-cert is enabled):
#   heavysql --ca-cert "${CA_CERT}" -s localhost --port 6274 -u admin -p password heavyai
#
# heavydb uses ssl-cert / ssl-private-key for its Thrift listener (PEM).
# ssl-trust-store: Calcite (Java) trusts HeavyDB when connecting to db port.
# ssl-trust-ca: HeavyDB (C++) trusts Calcite when connecting to calcite port.
# Calcite passwords are read from this file (ssl-trust-password,
# ssl-keystore-password) when heavydb passes --config to the Calcite process.

ssl-cert = "${heavydb_cert}"
ssl-private-key = "${heavydb_key}"
ssl-trust-store = "${truststore}"
ssl-trust-password = ${TRUSTSTORE_PASSWORD}
ssl-trust-ca = "${CA_CERT}"
ssl-keystore = "${calcite_keystore}"
ssl-keystore-password = ${KEYSTORE_PASSWORD}

[web]
port = 6273
frontend = "frontend"
EOF
  chmod 644 "${CONFIG_FILE}"
  echo "HeavyDB config file: ${CONFIG_FILE}"
}

print_summary() {
  cat <<EOF

=== TLS material generated under ${OUT_DIR} ===

CA (distribute to clients that must trust this dev environment):
  ${CA_CERT}

Calcite server identity (Java --keystore):
  ${CALCITE_DIR}/calcite-keystore.p12
  password: ${KEYSTORE_PASSWORD}

HeavyDB server identity:
  PEM (used by heavydb --ssl-cert / --ssl-private-key):
    ${HEAVYDB_DIR}/heavydb-server.crt
    ${HEAVYDB_DIR}/heavydb-server.key
  PKCS#12:
    ${HEAVYDB_DIR}/heavydb-server.p12
  password: ${KEYSTORE_PASSWORD}

Java trust store (Calcite client -> HeavyDB):
  ${TRUSTSTORE_DIR}/truststore.p12
  password: ${TRUSTSTORE_PASSWORD}

PEM CA (HeavyDB client -> Calcite):
  ${CA_CERT}
  (--ssl-trust-ca / ssl-trust-ca in config)

heavydb config file (recommended):
  ${CONFIG_FILE}
  heavydb --config ${CONFIG_FILE} --data /path/to/storage

heavysql (TLS):
  heavysql --ca-cert ${CA_CERT} -s localhost --port 6274 -u admin -p password heavyai

Example heavydb flags (equivalent to the config file):
  --ssl-cert ${HEAVYDB_DIR}/heavydb-server.crt \\
  --ssl-private-key ${HEAVYDB_DIR}/heavydb-server.key \\
  --ssl-trust-store ${TRUSTSTORE_DIR}/truststore.p12 \\
  --ssl-trust-password ${TRUSTSTORE_PASSWORD} \\
  --ssl-trust-ca ${CA_CERT} \\
  --ssl-keystore ${CALCITE_DIR}/calcite-keystore.p12 \\
  --ssl-keystore-password ${KEYSTORE_PASSWORD}

Verify Calcite keystore:
  ${OPENSSL_BIN} pkcs12 -in ${CALCITE_DIR}/calcite-keystore.p12 -nokeys \\
    -passin pass:${KEYSTORE_PASSWORD} | ${OPENSSL_BIN} x509 -noout -subject -issuer -dates

Verify HeavyDB PKCS#12:
  ${OPENSSL_BIN} pkcs12 -in ${HEAVYDB_DIR}/heavydb-server.p12 -nokeys \\
    -passin pass:${KEYSTORE_PASSWORD} | ${OPENSSL_BIN} x509 -noout -subject -issuer -dates

Verify Java trust store (should list heavydb-dev-ca):
  ${KEYTOOL_BIN} -list -keystore ${TRUSTSTORE_DIR}/truststore.p12 \\
    -storetype PKCS12 -storepass ${TRUSTSTORE_PASSWORD}

EOF
}

main() {
  require_openssl
  init_output_paths
  generate_self_signed_ca
  generate_calcite_server_material
  generate_heavydb_server_material
  generate_java_truststore
  generate_heavydb_config
  print_summary
}

main "$@"
