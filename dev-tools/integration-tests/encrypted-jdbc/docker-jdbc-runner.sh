#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Container entrypoint for verify-jdbc: build a merged JVM+dev-CA truststore,
# then run JDBC tests specified by JDBC_TEST_METHOD via Maven.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f /usr/local/mapd-deps/mapd-deps.sh ]]; then
  set +u
  # shellcheck disable=SC1091
  source /usr/local/mapd-deps/mapd-deps.sh
  set -u
fi

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

HEAVYDB_HTTP_PORT="${HEAVYDB_HTTP_PORT:-6278}"
JAVA_DIR="${HEAVYDB_SOURCE}/java"
JDBC_TRUSTSTORE="${CA_INFO_DIR}/jdbc-truststore.jks"
JDBC_TRUSTSTORE_PASSWORD=""
JDBC_TRUSTSTORE_TYPE=""
: "${OMNISCI_JAR_RELEASE_VERSION:=10.0.0-SNAPSHOT}"
KEYTOOL_BIN="${KEYTOOL_BIN:-keytool}"
TEST_METHOD="${JDBC_TEST_METHOD:?JDBC_TEST_METHOD must be set (e.g. via docker-compose environment)}"
JDBC_TEST_HTTPS_URL="${JDBC_TEST_HTTPS_URL:-jdbc:heavyai:${HEAVYDB_HOST}:${HEAVYDB_HTTP_PORT}}"
# Host ~/.m2 is bind-mounted at /m2; Maven's cache lives in repository/, not the .m2 root.
MAVEN_LOCAL="${MAVEN_LOCAL:-/m2/repository}"

log() {
  echo "[encrypted-jdbc] $*"
}

die() {
  echo "[encrypted-jdbc] ERROR: $*" >&2
  exit 1
}

resolve_java_cacerts() {
  if [[ -z "${JAVA_HOME:-}" ]]; then
    local java_bin
    java_bin="$(command -v java)" || die "java not found"
    JAVA_HOME="$(dirname "$(dirname "$(readlink -f "${java_bin}")")")"
  fi
  JVM_CACERTS="${JAVA_HOME}/lib/security/cacerts"
  [[ -f "${JVM_CACERTS}" ]] || die "JVM cacerts not found at ${JVM_CACERTS}"
}

detect_truststore_type() {
  local keystore="$1"
  local password="$2"
  if "${KEYTOOL_BIN}" -list -keystore "${keystore}" -storepass "${password}" -storetype PKCS12 \
    >/dev/null 2>&1; then
    echo "PKCS12"
  else
    echo "JKS"
  fi
}

prepare_jdbc_truststore() {
  [[ -f "${CA_CERT}" ]] || die "CA certificate not found at ${CA_CERT}"
  command -v "${KEYTOOL_BIN}" >/dev/null 2>&1 || die "keytool not found (${KEYTOOL_BIN})"

  resolve_java_cacerts
  local cacerts_password="${JVM_CACERTS_PASSWORD:-changeit}"

  log "Creating merged JDBC truststore ${JDBC_TRUSTSTORE} (JVM cacerts + dev CA)"
  rm -f "${JDBC_TRUSTSTORE}"
  cp "${JVM_CACERTS}" "${JDBC_TRUSTSTORE}"

  JDBC_TRUSTSTORE_TYPE="$(detect_truststore_type "${JDBC_TRUSTSTORE}" "${cacerts_password}")"
  JDBC_TRUSTSTORE_PASSWORD="${cacerts_password}"

  "${KEYTOOL_BIN}" -importcert -noprompt -trustcacerts \
    -alias heavydb-dev-ca \
    -file "${CA_CERT}" \
    -keystore "${JDBC_TRUSTSTORE}" \
    -storetype "${JDBC_TRUSTSTORE_TYPE}" \
    -storepass "${JDBC_TRUSTSTORE_PASSWORD}"
}

ensure_maven_repo() {
  [[ -d "${MAVEN_LOCAL}" ]] || die "Maven local repository not found at ${MAVEN_LOCAL} (expected host ~/.m2/repository or compose volume encrypted_jdbc_m2_repository)"
  if ! touch "${MAVEN_LOCAL}/.write-test" 2>/dev/null; then
    die "Maven local repository ${MAVEN_LOCAL} is not writable (check ownership under ~/.m2/repository; root-owned paths from prior runs block downloads)"
  fi
  rm -f "${MAVEN_LOCAL}/.write-test"
}

run_jdbc_test() {
  [[ -d "${JAVA_DIR}" ]] || die "Java tree not found at ${JAVA_DIR}"

  prepare_jdbc_truststore
  ensure_maven_repo

  log "Running Maven JDBC test HeavyAIConnectionTest#${TEST_METHOD}"
  log "  java.dir=${JAVA_DIR}"
  log "  maven.repo.local=${MAVEN_LOCAL}"
  log "  server_trust_store=${JDBC_TRUSTSTORE} (${JDBC_TRUSTSTORE_TYPE})"
  log "  jdbc_test_https_url=${JDBC_TEST_HTTPS_URL}"

  cd "${JAVA_DIR}"
  # Reuse host dependency cache (/m2/repository); compile and test output go under JAVA_DIR.
  # -am tells mv to rebuild dependencies on the same java tree - thrift & heavydbcommon
  # and -pl says only run the test target on heavyaijdbc
  # Merged truststore keeps public CAs for Maven HTTPS and adds the dev CA for HeavyDB TLS.
  mvn -q test -pl heavyaijdbc -am \
    "-Dmaven.repo.local=${MAVEN_LOCAL}" \
    -DskipTests=false \
    -Dsurefire.failIfNoSpecifiedTests=false \
    "-Domnisci.release.version=${OMNISCI_JAR_RELEASE_VERSION}" \
    "-Dtest=HeavyAIConnectionTest#${TEST_METHOD}" \
    -Dencrypted_server=true \
    "-Dserver_trust_store=${JDBC_TRUSTSTORE}" \
    "-Dserver_trust_store_pwd=${JDBC_TRUSTSTORE_PASSWORD}" \
    "-Djdbc_test_https_url=${JDBC_TEST_HTTPS_URL}" \
    "-Djavax.net.ssl.trustStore=${JDBC_TRUSTSTORE}" \
    "-Djavax.net.ssl.trustStorePassword=${JDBC_TRUSTSTORE_PASSWORD}" \
    "-Djavax.net.ssl.trustStoreType=${JDBC_TRUSTSTORE_TYPE}"

  local report="${JAVA_DIR}/heavyaijdbc/target/surefire-reports/TEST-ai.heavy.jdbc.HeavyAIConnectionTest.xml"
  if [[ -f "${report}" ]]; then
    log "JDBC test results:"
    grep '<testcase ' "${report}" | sed 's/.*<testcase name="\([^"]*\)".*/\1/' | while read -r test; do
      log "  ran: ${test}"
    done
  else
    log "WARNING: surefire report not found at ${report}"
  fi

  log "JDBC TLS connection test passed"
}

run_jdbc_test
