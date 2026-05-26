# Load environment variables from a .env file into the current shell.
#
# This script MUST be sourced for the variables to persist in the caller:
#
#     . scripts/load_env.sh                  # default: loads ./.env
#     . scripts/load_env.sh .env.prod        # custom path
#
# Lines starting with '#' are treated as comments. Surrounding quotes around
# values are stripped. Blank lines are ignored.

_ENV_FILE="${1:-.env}"

if [[ ! -f "$_ENV_FILE" ]]; then
    echo "No env file at '$_ENV_FILE' - skipping load." >&2
    return 0 2>/dev/null || exit 0
fi

_count=0
while IFS= read -r line || [[ -n "$line" ]]; do
    # Trim leading/trailing whitespace.
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    # Skip blanks and comments.
    [[ -z "$line" || "${line:0:1}" == "#" ]] && continue
    # Split on the first '='.
    name="${line%%=*}"
    value="${line#*=}"
    # Strip surrounding quotes.
    if [[ "${value:0:1}" == '"' && "${value: -1}" == '"' ]]; then
        value="${value:1:-1}"
    elif [[ "${value:0:1}" == "'" && "${value: -1}" == "'" ]]; then
        value="${value:1:-1}"
    fi
    export "$name=$value"
    _count=$((_count + 1))
done < "$_ENV_FILE"

echo "Loaded $_count variables from $_ENV_FILE"
unset _ENV_FILE _count
