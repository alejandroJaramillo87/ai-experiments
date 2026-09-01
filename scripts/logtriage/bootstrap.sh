#!/bin/bash
#
# One-time bootstrap for the log triage stack.
#
# Two steps of MongoDB's containerised search deployment are imperative and
# cannot be expressed in a compose file: initiating the replica set, and
# creating the user mongot authenticates as. Both are idempotent here, so a
# second run is a no-op rather than an error -- which matters because the
# natural thing to do after a failed `make logtriage-up` is to run this again.
#
# Run after the first `make logtriage-up`. mongot will report NOT_SERVING until
# it completes, because until the user exists mongot cannot open its change
# stream against the deployment.
#
# Reads the same password file that is mounted into mongot, so the credential
# is written once and the two cannot drift apart.

set -euo pipefail

RED='\033[0;31m' GREEN='\033[0;32m' YELLOW='\033[1;33m'
BLUE='\033[0;34m' CYAN='\033[0;36m' NC='\033[0m'

if [[ -n "${NO_COLOR:-}" ]] || [[ "${1:-}" == "--no-color" ]]; then
    RED='' GREEN='' YELLOW='' BLUE='' CYAN='' NC=''
fi

print_ok()    { printf "${GREEN}[OK]${NC} %s\n" "$*"; }
print_warn()  { printf "${YELLOW}[WARN]${NC} %s\n" "$*"; }
print_error() { printf "${RED}[ERROR]${NC} %s\n" "$*"; }
print_info()  { printf "${CYAN}[INFO]${NC} %s\n" "$*"; }
print_run()   { printf "${BLUE}[RUN]${NC} %s\n" "$*"; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SECRETS_DIR="${REPO_ROOT}/.secrets/logtriage"
PASSWORD_FILE="${SECRETS_DIR}/passwordFile"
CONTAINER="${LOGTRIAGE_MONGO_CONTAINER:-logtriage-mongo}"
MONGOT_USER="mongotUser"

echo "=== Log triage stack bootstrap ==="
echo

if ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
    print_error "Container not found: $CONTAINER"
    echo "  Solution: start the stack first"
    echo "  Command: make logtriage-up"
    exit 1
fi

if [[ ! -f "$PASSWORD_FILE" ]]; then
    print_error "Password file not found: $PASSWORD_FILE"
    echo "  This is the credential mongot authenticates with; it is generated, not committed."
    echo "  Solution: generate the stack's secrets"
    echo "  Command: make logtriage-secrets"
    exit 1
fi

ROOT_USER="${LOGTRIAGE_MONGO_ROOT_USER:-logtriage}"
if [[ -z "${LOGTRIAGE_MONGO_ROOT_PASSWORD:-}" ]]; then
    print_error "LOGTRIAGE_MONGO_ROOT_PASSWORD is not set"
    echo "  Solution: export it from the stack's secrets file before running"
    echo "  Command: set -a; . ./.secrets/logtriage/stack.env; set +a"
    exit 1
fi

MONGOT_PASSWORD="$(cat "$PASSWORD_FILE")"

# mongosh runs inside the container: the server binds the compose network and
# publishes only to loopback, and this way the script needs no host mongosh.
mongo_eval() {
    docker exec -i "$CONTAINER" mongosh --quiet \
        --username "$ROOT_USER" --password "$LOGTRIAGE_MONGO_ROOT_PASSWORD" \
        --authenticationDatabase admin \
        "$@"
}

echo "Replica set:"
# A set with no configuration answers NotYetInitialized (code 94). Any other
# failure is a real one and should not be swallowed by an unconditional
# initiate, so the two cases are separated rather than run-and-ignore.
if mongo_eval --eval 'rs.status().ok' >/dev/null 2>&1; then
    print_ok "rs0 already initiated"
else
    print_run "Initiating rs0 as a single-member set"
    # The host must match --replSetMember in the compose command and the
    # syncSource address in mongot.conf; all three name the compose service.
    mongo_eval --eval \
        'rs.initiate({_id: "rs0", members: [{_id: 0, host: "logtriage-mongo:27017"}]})' \
        >/dev/null
    print_ok "rs0 initiated"
fi

echo
echo "Search coordinator user:"
# searchCoordinator carries readAnyDatabase plus write access to the internal
# __mdb_internal_search database. Nothing else should hold it, and nothing
# should write to that database by hand.
EXISTS="$(mongo_eval --eval \
    "db.getSiblingDB('admin').getUser('${MONGOT_USER}') ? 'yes' : 'no'" | tr -d '[:space:]')"

if [[ "$EXISTS" == "yes" ]]; then
    print_run "Updating ${MONGOT_USER} to match the current password file"
    mongo_eval --eval \
        "db.getSiblingDB('admin').updateUser('${MONGOT_USER}', {pwd: '${MONGOT_PASSWORD}', roles: ['searchCoordinator']})" \
        >/dev/null
    print_ok "${MONGOT_USER} updated"
else
    print_run "Creating ${MONGOT_USER} with the searchCoordinator role"
    mongo_eval --eval \
        "db.getSiblingDB('admin').createUser({user: '${MONGOT_USER}', pwd: '${MONGOT_PASSWORD}', roles: ['searchCoordinator']})" \
        >/dev/null
    print_ok "${MONGOT_USER} created"
fi

echo
print_info "mongot may take a minute to report SERVING while it opens its change stream"
echo "  Check: curl -s http://127.0.0.1:8008/health"
echo
print_ok "Bootstrap complete"
