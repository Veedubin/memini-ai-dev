#!/usr/bin/env bash
# generate-local-certs.sh — Generate self-signed PostgreSQL TLS certificates
#
# Creates a CA, server certificate, and client certificate in the
# memini-ai-dev/certs/ directory for local development.
#
# Usage:
#   ./scripts/generate-local-certs.sh
#
# Certificate files produced:
#   certs/ca.crt          — CA certificate (trusted by client)
#   certs/ca.key          — CA private key
#   certs/server.crt      — PostgreSQL server certificate
#   certs/server.key      — PostgreSQL server private key
#   certs/client.crt      — Client certificate (for verify-full)
#   certs/client.key      — Client private key
#
# After generating:
#   1. Mount certs/ in docker-compose.yml: ./memini-ai-dev/certs:/certs
#   2. Set DB_SSLMODE=require (or verify-ca / verify-full)
#   3. Set DB_SSLROOTCERT=/certs/ca.crt
#   4. Configure PostgreSQL to use server.crt/server.key (see docker-compose.yml)
#
set -euo pipefail

CERTS_DIR="$(cd "$(dirname "$0")/.." && pwd)/certs"
DAYS="${CERT_VALIDITY_DAYS:-3650}"  # 10 years default

# PostgreSQL server SANs — adjust for your environment
SERVER_HOSTNAMES="DNS:localhost,DNS:postgres,IP:127.0.0.1,IP:::1"

echo "Generating PostgreSQL TLS certificates in: ${CERTS_DIR}"
echo "Certificate validity: ${DAYS} days"
echo "Server SANs: ${SERVER_HOSTNAMES}"
echo ""

# Create certs directory
mkdir -p "${CERTS_DIR}"

# --------------------------------------------------------------------------
# 1. Certificate Authority (CA)
# --------------------------------------------------------------------------
if [ -f "${CERTS_DIR}/ca.crt" ] && [ -f "${CERTS_DIR}/ca.key" ]; then
    echo "CA certificate already exists — skipping CA generation"
else
    echo "Generating CA certificate..."
    openssl req -new -x509 \
        -days "${DAYS}" \
        -nodes \
        -newkey rsa:2048 \
        -keyout "${CERTS_DIR}/ca.key" \
        -out "${CERTS_DIR}/ca.crt" \
        -subj "/CN=memini-ai-local-CA/O=memini-ai-dev" \
        -addext "basicConstraints=critical,CA:TRUE" \
        -addext "keyUsage=critical,digitalSignature,keyCertSign,cRLSign"
    chmod 600 "${CERTS_DIR}/ca.key"
    echo "  CA certificate: ${CERTS_DIR}/ca.crt"
    echo "  CA private key:  ${CERTS_DIR}/ca.key"
fi

# --------------------------------------------------------------------------
# 2. Server Certificate
# --------------------------------------------------------------------------
if [ -f "${CERTS_DIR}/server.crt" ] && [ -f "${CERTS_DIR}/server.key" ]; then
    echo "Server certificate already exists — skipping server generation"
else
    echo "Generating server certificate..."
    # Create CSR with SANs
    openssl req -new -nodes \
        -newkey rsa:2048 \
        -keyout "${CERTS_DIR}/server.key" \
        -out "${CERTS_DIR}/server.csr" \
        -subj "/CN=postgres/O=memini-ai-dev" \
        -addext "subjectAltName=${SERVER_HOSTNAMES}"

    # Sign with CA
    openssl x509 -req \
        -days "${DAYS}" \
        -in "${CERTS_DIR}/server.csr" \
        -CA "${CERTS_DIR}/ca.crt" \
        -CAkey "${CERTS_DIR}/ca.key" \
        -CAcreateserial \
        -out "${CERTS_DIR}/server.crt" \
        -ext <(printf "subjectAltName=%s\nbasicConstraints=CA:FALSE\nkeyUsage=digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth" "${SERVER_HOSTNAMES}")

    # PostgreSQL requires the server key to be owned by the postgres user
    # and have 0600 permissions
    chmod 600 "${CERTS_DIR}/server.key"
    rm -f "${CERTS_DIR}/server.csr"
    echo "  Server certificate: ${CERTS_DIR}/server.crt"
    echo "  Server private key:  ${CERTS_DIR}/server.key"
fi

# --------------------------------------------------------------------------
# 3. Client Certificate (for verify-full)
# --------------------------------------------------------------------------
if [ -f "${CERTS_DIR}/client.crt" ] && [ -f "${CERTS_DIR}/client.key" ]; then
    echo "Client certificate already exists — skipping client generation"
else
    echo "Generating client certificate..."
    openssl req -new -nodes \
        -newkey rsa:2048 \
        -keyout "${CERTS_DIR}/client.key" \
        -out "${CERTS_DIR}/client.csr" \
        -subj "/CN=memini-ai-client/O=memini-ai-dev"

    openssl x509 -req \
        -days "${DAYS}" \
        -in "${CERTS_DIR}/client.csr" \
        -CA "${CERTS_DIR}/ca.crt" \
        -CAkey "${CERTS_DIR}/ca.key" \
        -CAcreateserial \
        -out "${CERTS_DIR}/client.crt" \
        -ext <(printf "basicConstraints=CA:FALSE\nkeyUsage=digitalSignature\nextendedKeyUsage=clientAuth")

    chmod 600 "${CERTS_DIR}/client.key"
    rm -f "${CERTS_DIR}/client.csr"
    echo "  Client certificate: ${CERTS_DIR}/client.crt"
    echo "  Client private key:  ${CERTS_DIR}/client.key"
fi

# Clean up serial files
rm -f "${CERTS_DIR}/ca.srl"

echo ""
echo "========================================="
echo "  Certificate generation complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Docker Compose: mount ./memini-ai-dev/certs:/certs"
echo "  2. Set environment variables:"
echo "     DB_SSLMODE=require"
echo "     DB_SSLROOTCERT=/certs/ca.crt"
echo "  3. Restart PostgreSQL and memini-ai"
echo ""
echo "For production, replace these self-signed certificates"
echo "with certificates from a trusted CA."